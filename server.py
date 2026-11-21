"""
Backend FastAPI - Gerador de Sinais IQ Option / PolariumBroker.
Estratégia: MA9 x MA21 (direção) + MACD cruzamento (gatilho) + RSI + Agulhada do Didi (confirmação).
"""
from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
import asyncio
import logging
import os
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import bcrypt
import pandas as pd
import jwt
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field
from pymongo import ASCENDING, DESCENDING, IndexModel
from starlette.middleware.cors import CORSMiddleware
from broker import (
    SIM_PAYOUT, TIMEFRAME_SECONDS, BrokerSession, check_trade, connect_broker, disconnect_broker,
    get_candles, get_candles_many, place_trade, refresh_balance,
)
from strategy import analyze_candles, sma

# -------- Infra --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sinais")
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALG = "HS256"
ACCESS_MIN = 60 * 24  # 24h
app = FastAPI(title="Purple Signal Analyst")
api = APIRouter(prefix="/api")

# -------- Models --------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class BrokerConnectIn(BaseModel):
    broker: str = Field(pattern="^(iqoption|polarium)$")
    email: str = ""
    password: str = ""
    account_type: str = Field(default="demo", pattern="^(demo|real)$")
    simulate: bool = False


class ScanStartIn(BaseModel):
    timeframe: str = Field(default="M1", pattern="^(M1|M5|M15)$")
    min_confidence: int = Field(default=70, ge=50, le=99)
    auto_trade: bool = False
    amount: float = Field(default=2.0, ge=1, le=10000)


# -------- Auth helpers --------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def make_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_MIN),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sessão expirada")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0, "password_hash": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")
    return user


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=ACCESS_MIN * 60,
        path="/",
    )


# -------- In-memory per-user state --------
_BROKER_SESSIONS: dict[str, BrokerSession] = {}
_SCAN_TASKS: dict[str, asyncio.Task] = {}
_SCAN_STATE: dict[str, dict] = {}


def _state(user_id: str) -> dict:
    return _SCAN_STATE.setdefault(
        user_id,
        {
            "running": False,
            "timeframe": "M1",
            "min_confidence": 70,
            "auto_trade": False,
            "amount": 2.0,
            "last_scan_ts": None,
            "best": None,
            "ranking": [],
            "candles": {},
            "message": "Aguardando início do scanner",
        },
    )


def _signal_doc(user_id: str, best: dict, tf: str, session: BrokerSession) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "asset": best["asset"],
        "direction": best["direction"],
        "confidence": best["confidence"],
        "price": best["price"],
        "timeframe": tf,
        "broker": session.broker,
        "simulated": session.simulated,
        "account_type": session.account_type,
        "indicators": best["indicators"],
        "ts": now.isoformat(),
        "expires_at": (now + timedelta(seconds=TIMEFRAME_SECONDS[tf])).isoformat(),
        "result": "PENDENTE",
        "trade": None,
    }


async def _auto_trade(state: dict, session: BrokerSession, doc: dict) -> None:
    amount = float(state["amount"])
    try:
        order_id = await place_trade(session, doc["asset"], doc["direction"], amount, doc["timeframe"])
        doc["trade"] = {"order_id": order_id, "amount": amount, "status": "ABERTA", "profit": None}
        state["message"] = f"Entrada automática: {doc['asset']} {doc['direction']} {session.currency} {amount:.2f}"
    except ConnectionError:
        raise
    except Exception as e:  # noqa: BLE001
        doc["trade"] = {"order_id": None, "amount": amount, "status": "ERRO", "error": str(e)[:200], "profit": None}
        state["message"] = f"Falha na entrada automática: {str(e)[:120]}"


async def _scan_loop(user_id: str) -> None:
    state = _state(user_id)
    session = _BROKER_SESSIONS.get(user_id)
    if not session:
        state["running"] = False
        state["message"] = "Nenhuma corretora conectada"
        return
    logger.info("Scanner user=%s broker=%s simulado=%s", user_id, session.broker, session.simulated)
    while state["running"]:
        try:
            tf = state["timeframe"]
            all_candles = await get_candles_many(session, session.assets, tf, count=120)
            state["candles"] = all_candles
            results = []
            for asset, candles in all_candles.items():
                analysis = analyze_candles(candles)
                if analysis.get("ready") and analysis["direction"] != "NEUTRO":
                    results.append({"asset": asset, **analysis})
            results.sort(key=lambda r: r["confidence"], reverse=True)
            state["ranking"] = results[:10]
            state["last_scan_ts"] = datetime.now(timezone.utc).isoformat()
            best = results[0] if results else None
            state["best"] = best
            if best and best["confidence"] >= state["min_confidence"]:
                same = await db.signals.find_one({
                    "user_id": user_id, "asset": best["asset"],
                    "direction": best["direction"], "result": "PENDENTE",
                })
                if not same:
                    doc = _signal_doc(user_id, best, tf, session)
                    state["message"] = f"Novo sinal: {best['asset']} {best['direction']} ({best['confidence']}%)"
                    if state["auto_trade"]:
                        await _auto_trade(state, session, doc)
                    await db.signals.insert_one(doc)
            elif not results:
                state["message"] = "Analisando mercado..."
            await _settle_pending(user_id, session)
            await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.warning("Conexão perdida user=%s: %s", user_id, e)
            state["running"] = False
            session.connected = False
            state["message"] = "Conexão com a corretora perdida. Reconecte a conta."
            break
        except Exception as e:  # noqa: BLE001
            logger.exception("Erro no scan loop: %s", e)
            await asyncio.sleep(2.0)
    logger.info("Scanner parado user=%s", user_id)


async def _stop_scan(user_id: str) -> None:
    state = _state(user_id)
    state["running"] = False
    task = _SCAN_TASKS.pop(user_id, None)
    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


def _price_at_expiry(sig: dict, candles: list[dict]) -> float:
    tf = TIMEFRAME_SECONDS[sig["timeframe"]]
    exp_ts = int(datetime.fromisoformat(sig["expires_at"]).timestamp())
    at = next((c for c in candles if c["ts"] == exp_ts // tf * tf), None)
    if at:
        return at["open"]
    return candles[-1]["close"] if candles else sig["price"]


async def _settle_pending(user_id: str, session: BrokerSession) -> None:
    """WIN/LOSS: ordem real pelo resultado da corretora; senão pelo preço na expiração."""
    now = datetime.now(timezone.utc)
    pending = await db.signals.find(
        {"user_id": user_id, "result": "PENDENTE", "expires_at": {"$lte": now.isoformat()}}, {"_id": 0}
    ).to_list(50)
    balance_changed = False
    for sig in pending:
        trade = sig.get("trade") or {}
        if trade.get("order_id") and not sig["simulated"]:
            res = await check_trade(session, trade["order_id"])
            if not res.get("done"):
                continue
            result = {"win": "WIN", "loose": "LOSS"}.get(res["win"], "EMPATE")
            update = {"result": result, "trade.status": "FECHADA", "trade.profit": res["profit"]}
            balance_changed = True
        else:
            candles = _state(user_id)["candles"].get(sig["asset"]) or await get_candles(
                session, sig["asset"], sig["timeframe"], count=60)
            close = _price_at_expiry(sig, candles)
            won = close > sig["price"] if sig["direction"] == "CALL" else close < sig["price"]
            update = {"result": "WIN" if won else "LOSS", "close_price": close}
            if trade.get("order_id"):
                profit = round(trade["amount"] * SIM_PAYOUT, 2) if won else -trade["amount"]
                if won:
                    session.balance = round(session.balance + trade["amount"] + profit, 2)
                update.update({"trade.status": "FECHADA", "trade.profit": profit})
        await db.signals.update_one({"id": sig["id"]}, {"$set": update})
    if balance_changed:
        await refresh_balance(session)


# -------- Routes: Auth --------
@api.get("/")
async def root():
    return {"service": "Purple Signal Analyst", "status": "ok"}


@api.post("/auth/register")
async def register(payload: RegisterIn, response: Response):
    email = payload.email.lower()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="E-mail já cadastrado")
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "name": payload.name,
        "password_hash": hash_password(payload.password),
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user)
    token = make_token(user["id"], user["email"])
    set_auth_cookie(response, token)
    return {"token": token, "user": {"id": user["id"], "email": email, "name": user["name"], "role": "user"}}


@api.post("/auth/login")
async def login(payload: LoginIn, response: Response):
    email = payload.email.lower()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    token = make_token(user["id"], user["email"])
    set_auth_cookie(response, token)
    return {
        "token": token,
        "user": {"id": user["id"], "email": email, "name": user["name"], "role": user.get("role", "user")},
    }


@api.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"ok": True}


@api.get("/auth/me")
async def me(user=Depends(get_current_user)):
    return {"user": user}


# -------- Routes: Broker --------
@api.post("/broker/connect")
async def broker_connect(payload: BrokerConnectIn, user=Depends(get_current_user)):
    # desconecta sessão antiga
    old = _BROKER_SESSIONS.pop(user["id"], None)
    if old:
        await _stop_scan(user["id"])
        disconnect_broker(old)
    try:
        session = await connect_broker(payload.broker, payload.email, payload.password, payload.account_type, payload.simulate)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _BROKER_SESSIONS[user["id"]] = session
    return _session_out(session)


def _session_out(session: BrokerSession) -> dict:
    return {
        "broker": session.broker,
        "connected": session.connected,
        "simulated": session.simulated,
        "account_type": session.account_type,
        "balance": session.balance,
        "currency": session.currency,
        "message": session.message,
        "assets_count": len(session.assets),
    }


@api.post("/broker/disconnect")
async def broker_disconnect(user=Depends(get_current_user)):
    await _stop_scan(user["id"])
    state = _state(user["id"])
    state["best"] = None
    state["ranking"] = []
    state["message"] = "Desconectado"
    session = _BROKER_SESSIONS.pop(user["id"], None)
    if session:
        disconnect_broker(session)
    return {"ok": True}


@api.get("/broker/status")
async def broker_status(user=Depends(get_current_user)):
    session = _BROKER_SESSIONS.get(user["id"])
    if not session:
        return {"connected": False}
    return _session_out(session)


# -------- Routes: Scan --------
@api.post("/scan/start")
async def scan_start(payload: ScanStartIn, user=Depends(get_current_user)):
    session = _BROKER_SESSIONS.get(user["id"])
    if not session or not session.connected:
        raise HTTPException(status_code=400, detail="Conecte uma corretora antes de iniciar a análise")
    if payload.auto_trade and payload.amount > session.balance:
        raise HTTPException(status_code=400, detail="Valor da entrada maior que o saldo")
    state = _state(user["id"])
    if state["running"]:
        return {"ok": True, "message": "Scanner já em execução"}
    state["running"] = True
    state["timeframe"] = payload.timeframe
    state["min_confidence"] = payload.min_confidence
    state["auto_trade"] = payload.auto_trade
    state["amount"] = payload.amount
    state["message"] = "Iniciando varredura do mercado..."
    old = _SCAN_TASKS.get(user["id"])
    if old and not old.done():
        old.cancel()
    _SCAN_TASKS[user["id"]] = asyncio.create_task(_scan_loop(user["id"]))
    return {"ok": True, "message": "Análise iniciada"}


@api.post("/scan/stop")
async def scan_stop(user=Depends(get_current_user)):
    await _stop_scan(user["id"])
    _state(user["id"])["message"] = "Scanner parado"
    return {"ok": True}


@api.get("/scan/state")
async def scan_state(user=Depends(get_current_user)):
    state = _state(user["id"])
    return {
        "running": state["running"],
        "timeframe": state["timeframe"],
        "min_confidence": state["min_confidence"],
        "auto_trade": state["auto_trade"],
        "amount": state["amount"],
        "last_scan_ts": state["last_scan_ts"],
        "best": state["best"],
        "ranking": state["ranking"],
        "message": state["message"],
    }


@api.get("/signals/history")
async def signals_history(user=Depends(get_current_user), limit: int = 50):
    cursor = db.signals.find({"user_id": user["id"]}, {"_id": 0}).sort("ts", DESCENDING).limit(limit)
    items = await cursor.to_list(length=limit)
    wins = await db.signals.count_documents({"user_id": user["id"], "result": "WIN"})
    losses = await db.signals.count_documents({"user_id": user["id"], "result": "LOSS"})
    total = wins + losses
    agg = await db.signals.aggregate([
        {"$match": {"user_id": user["id"], "trade.profit": {"$ne": None}}},
        {"$group": {"_id": None, "profit": {"$sum": "$trade.profit"}, "n": {"$sum": 1}}},
    ]).to_list(1)
    return {"items": items, "stats": {
        "wins": wins, "losses": losses, "winrate": round(wins * 100 / total, 1) if total else 0,
        "trades": agg[0]["n"] if agg else 0, "profit": round(agg[0]["profit"], 2) if agg else 0,
    }}


@api.get("/chart")
async def chart(asset: str, user=Depends(get_current_user)):
    state = _state(user["id"])
    candles = state["candles"].get(asset)
    if not candles:
        raise HTTPException(status_code=404, detail="Sem dados para este ativo ainda")
    close = pd.Series([c["close"] for c in candles])
    ma9, ma21 = sma(close, 9), sma(close, 21)
    out = candles[-80:]
    off = len(candles) - len(out)
    return {
        "asset": asset,
        "timeframe": state["timeframe"],
        "candles": out,
        "ma9": [{"ts": c["ts"], "value": float(ma9.iloc[off + i])} for i, c in enumerate(out) if not pd.isna(ma9.iloc[off + i])],
        "ma21": [{"ts": c["ts"], "value": float(ma21.iloc[off + i])} for i, c in enumerate(out) if not pd.isna(ma21.iloc[off + i])],
    }


# -------- App setup --------
app.include_router(api)


@app.get("/", include_in_schema=False)
async def home():
    index_file = ROOT_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="index.html não encontrado no diretório do servidor")
    return FileResponse(index_file, media_type="text/html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEXES: dict[str, list[IndexModel]] = {
    "users": [IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
              IndexModel([("id", ASCENDING)], unique=True, name="id_unique")],
    "signals": [IndexModel([("user_id", ASCENDING), ("ts", DESCENDING)], name="user_ts")],
}


async def seed_admin() -> None:
    email = os.environ.get("ADMIN_EMAIL", "admin@sinais.com").lower()
    password = os.environ.get("ADMIN_PASSWORD", "admin123")
    existing = await db.users.find_one({"email": email})
    if existing:
        return
    await db.users.insert_one({
        "id": str(uuid.uuid4()),
        "email": email,
        "name": "Administrador",
        "password_hash": hash_password(password),
        "role": "admin",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Admin seed criado: %s", email)


@app.on_event("startup")
async def _startup() -> None:
    for coll, models in INDEXES.items():
        for m in models:
            with suppress(Exception):
                await db[coll].create_indexes([m])
    await seed_admin()


@app.on_event("shutdown")
async def _shutdown() -> None:
    for task in list(_SCAN_TASKS.values()):
        task.cancel()
    _SCAN_TASKS.clear()
    for s in list(_BROKER_SESSIONS.values()):
        disconnect_broker(s)
    client.close()
