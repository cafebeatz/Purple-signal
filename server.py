"""
Backend FastAPI - Purple Signal.

Integração do painel Purple Signal:
- Autenticação e conexão da corretora.
- Scanner com candles fechados.
- Robô automático com pausa, parada e reset de sessão.
- Limites diários calculados sobre lucro/prejuízo realizado.
- Placar separado por sessão e arquivo pesquisável de sessões anteriores.
- Atualização explícita do saldo pela corretora.

IMPORTANTE:
A inicialização preserva os documentos existentes da coleção "signals".
Nenhum histórico é apagado por esta versão.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import asyncio
import logging
import os
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone, time
from typing import Optional

import bcrypt
import jwt
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field
from pymongo import ASCENDING, DESCENDING, IndexModel
from starlette.middleware.cors import CORSMiddleware

from broker import (
    SIM_PAYOUT,
    TIMEFRAME_SECONDS,
    BrokerSession,
    check_trade,
    connect_broker,
    disconnect_broker,
    get_candles,
    get_candles_many,
    place_trade,
    refresh_balance,
)
from strategy import analyze_candles, sma


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("purple_signal")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALG = "HS256"
ACCESS_MIN = 60 * 24

app = FastAPI(title="Purple Signal Analyst")
api = APIRouter(prefix="/api")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def new_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

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
    auto_trade: bool = True
    amount: float = Field(default=2.0, ge=1, le=10000)
    target_gain: float = Field(default=0.0, ge=0, le=100000000)
    max_loss: float = Field(default=0.0, ge=0, le=100000000)


class RobotConfigIn(BaseModel):
    auto_trade: bool = True
    amount: float = Field(default=2.0, ge=1, le=10000)
    target_gain: float = Field(default=0.0, ge=0, le=100000000)
    max_loss: float = Field(default=0.0, ge=0, le=100000000)


class SessionStartIn(BaseModel):
    reason: str = Field(default="Nova sessão iniciada", max_length=120)


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

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
        "exp": utc_now() + timedelta(minutes=ACCESS_MIN),
        "iat": utc_now(),
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

    user = await db.users.find_one(
        {"id": payload["sub"]},
        {"_id": 0, "password_hash": 0},
    )
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


# ---------------------------------------------------------------------------
# Estado por usuário
# ---------------------------------------------------------------------------

_BROKER_SESSIONS: dict[str, BrokerSession] = {}
_SCAN_TASKS: dict[str, asyncio.Task] = {}
_SCAN_STATE: dict[str, dict] = {}

# Liquidação resiliente: tempos em segundos.
SETTLEMENT_CHECK_TIMEOUT = 5
SETTLEMENT_RETRY_SECONDS = 5
SETTLEMENT_FALLBACK_AFTER = 60
SETTLEMENT_FINAL_WAIT_MIN = 300
_SETTLEMENT_LOCKS: dict[str, asyncio.Lock] = {}


def _settlement_lock(user_id: str) -> asyncio.Lock:
    lock = _SETTLEMENT_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _SETTLEMENT_LOCKS[user_id] = lock
    return lock


def _state(user_id: str) -> dict:
    state = _SCAN_STATE.get(user_id)
    if state is None:
        state = {
            "running": False,
            "paused": False,
            "timeframe": "M1",
            "min_confidence": 70,
            "auto_trade": False,
            "amount": 2.0,
            "target_gain": 0.0,
            "max_loss": 0.0,
            "risk_halted": False,
            "risk_reason": "",
            "daily_profit": 0.0,
            "last_scan_ts": None,
            "best": None,
            "ranking": [],
            "candles": {},
            "last_signal_candle": {},
            "message": "Aguardando início do scanner",
            "session_id": new_session_id(),
            "session_started_at": iso_now(),
            "session_reason": "Sessão inicial",
        }
        _SCAN_STATE[user_id] = state
    return state


async def _daily_realized_profit(user_id: str) -> float:
    """Soma somente resultados monetários já realizados no dia UTC."""
    start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    docs = await db.signals.find(
        {
            "user_id": user_id,
            "settled_at": {
                "$gte": start.isoformat(),
                "$lt": end.isoformat(),
            },
            "trade.profit": {"$ne": None},
        },
        {"_id": 0, "trade.profit": 1},
    ).to_list(5000)

    total = 0.0
    for doc in docs:
        try:
            total += float((doc.get("trade") or {}).get("profit") or 0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


async def _refresh_risk_state(user_id: str) -> dict:
    state = _state(user_id)
    pnl = await _daily_realized_profit(user_id)
    state["daily_profit"] = pnl

    target = float(state.get("target_gain") or 0)
    max_loss = float(state.get("max_loss") or 0)

    reason = ""
    if target > 0 and pnl >= target:
        reason = f"Meta diária atingida ({pnl:.2f} / {target:.2f})"
    elif max_loss > 0 and pnl <= -max_loss:
        reason = f"Limite de perda atingido ({pnl:.2f} / -{max_loss:.2f})"

    state["risk_halted"] = bool(reason)
    state["risk_reason"] = reason
    return state


def _start_new_session(state: dict, reason: str) -> None:
    state["session_id"] = new_session_id()
    state["session_started_at"] = iso_now()
    state["session_reason"] = reason
    state["last_signal_candle"] = {}
    state["best"] = None
    state["ranking"] = []
    state["message"] = reason


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


# ---------------------------------------------------------------------------
# Sinais, ordens e liquidação
# ---------------------------------------------------------------------------

def _signal_doc(
    user_id: str,
    best: dict,
    tf: str,
    session: BrokerSession,
    state: dict,
) -> dict:
    now = utc_now()
    return {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": state["session_id"],
        "session_started_at": state["session_started_at"],
        "asset": best["asset"],
        "direction": best["direction"],
        "confidence": best["confidence"],
        "price": best["price"],
        "timeframe": tf,
        "candle_ts": best.get("candle_ts"),
        "broker": session.broker,
        "simulated": session.simulated,
        "account_type": session.account_type,
        "indicators": best.get("indicators", {}),
        "ts": now.isoformat(),
        "expires_at": (now + timedelta(seconds=TIMEFRAME_SECONDS[tf])).isoformat(),
        "result": "PENDENTE",
        "trade": None,
        "profit": None,
        "settled_at": None,
    }


async def _auto_trade(
    user_id: str,
    state: dict,
    session: BrokerSession,
    doc: dict,
) -> bool:
    """Executa ordem somente após validar conexão, pausa, risco e saldo."""
    if not state.get("auto_trade"):
        return False
    if state.get("paused"):
        state["message"] = "Robô pausado: nenhuma nova ordem será enviada."
        return False
    if not session.connected:
        state["message"] = "Ordem bloqueada: corretora desconectada."
        return False

    await _refresh_risk_state(user_id)
    if state.get("risk_halted"):
        state["message"] = "Ordem bloqueada pela proteção diária: " + state["risk_reason"]
        return False

    amount = float(state["amount"])
    if amount < 1 or amount > 10000:
        state["message"] = "Ordem bloqueada: valor fora do limite permitido."
        return False
    if amount > float(session.balance or 0):
        state["message"] = "Ordem bloqueada: saldo insuficiente."
        return False

    try:
        order_id = await place_trade(
            session,
            doc["asset"],
            doc["direction"],
            amount,
            doc["timeframe"],
        )
        doc["trade"] = {
            "order_id": order_id,
            "amount": amount,
            "status": "ABERTA",
            "profit": None,
        }
        state["message"] = (
            f"Entrada automática enviada: {doc['asset']} "
            f"{doc['direction']} {session.currency} {amount:.2f}"
        )
        return True
    except ConnectionError:
        raise
    except Exception as exc:  # noqa: BLE001
        doc["trade"] = {
            "order_id": None,
            "amount": amount,
            "status": "ERRO",
            "error": str(exc)[:200],
            "profit": None,
        }
        state["message"] = f"Falha na entrada automática: {str(exc)[:120]}"
        return False


def _price_at_expiry(sig: dict, candles: list[dict], now: Optional[datetime] = None) -> Optional[float]:
    """Retorna preço de candle FECHADO na expiração; nunca inventa preço."""
    if not candles:
        return None

    tf = TIMEFRAME_SECONDS[sig["timeframe"]]
    exp_ts = int(parse_iso(sig["expires_at"]).timestamp())
    target_ts = exp_ts // tf * tf
    now_ts = int((now or utc_now()).timestamp())

    normalized = []
    for candle in candles:
        try:
            ts = int(candle.get("ts", 0))
            if ts <= 0 or ts + tf > now_ts:
                continue  # candle ainda aberto
            normalized.append((ts, candle))
        except (TypeError, ValueError):
            continue

    normalized.sort(key=lambda item: item[0])
    if not normalized:
        return None

    # O candle que começa na expiração representa a primeira cotação do período.
    exact = next((c for ts, c in normalized if ts == target_ts), None)
    if exact is not None:
        try:
            return float(exact["open"])
        except (KeyError, TypeError, ValueError):
            return None

    # Fallback conservador: último candle fechado, desde que não esteja muito antigo.
    last_ts, last = normalized[-1]
    if last_ts > target_ts or target_ts - last_ts > tf * 2:
        return None
    try:
        return float(last["close"])
    except (KeyError, TypeError, ValueError):
        return None


def _classify_candle_result(sig: dict, close_price: Optional[float]) -> str:
    if close_price is None:
        return "INDETERMINADO"
    try:
        entry = float(sig["price"])
    except (KeyError, TypeError, ValueError):
        return "INDETERMINADO"
    direction = str(sig.get("direction", "")).upper()
    tolerance = max(abs(entry) * 1e-10, 1e-12)
    if abs(close_price - entry) <= tolerance:
        return "EMPATE"
    if direction == "CALL":
        return "WIN" if close_price > entry else "LOSS"
    if direction == "PUT":
        return "WIN" if close_price < entry else "LOSS"
    return "INDETERMINADO"


async def _get_settlement_candles(
    user_id: str, sig: dict, session: BrokerSession, now: datetime
) -> list[dict]:
    """Prioriza cache e consulta a corretora se o cache não contém candle útil."""
    cached = _state(user_id).get("candles", {}).get(sig.get("asset"), []) or []
    if _price_at_expiry(sig, cached, now) is not None:
        return cached
    try:
        candles = await asyncio.wait_for(
            get_candles(session, sig["asset"], sig["timeframe"], count=80),
            timeout=4,
        )
        return candles or cached
    except (asyncio.TimeoutError, TimeoutError, ConnectionError) as exc:
        logger.warning("Fallback de candles temporariamente indisponível: %s", exc)
    except Exception:
        logger.exception("Falha ao buscar candles para liquidação")
    return cached


async def _settle_pending(user_id: str, session: BrokerSession) -> None:
    """Liquida pendências sem bloquear o scanner e sem confundir estimativa com resultado real."""
    lock = _settlement_lock(user_id)
    if lock.locked():
        return

    async with lock:
        now = utc_now()
        pending = await db.signals.find(
            {
                "user_id": user_id,
                "result": "PENDENTE",
                "expires_at": {"$lte": now.isoformat()},
            },
            {"_id": 0},
        ).to_list(100)

        broker_balance_changed = False

        for sig in pending:
            signal_id = sig.get("id")
            if not signal_id:
                continue

            # Evita consultas repetidas a cada ciclo e persiste as tentativas no MongoDB.
            next_check = sig.get("settlement_next_check_at")
            if next_check:
                try:
                    if parse_iso(next_check) > now:
                        continue
                except (TypeError, ValueError):
                    pass

            trade = sig.get("trade") or {}
            has_real_order = bool(trade.get("order_id") and not sig.get("simulated"))
            is_simulated_order = bool(trade.get("order_id") and sig.get("simulated"))
            update: dict = {}
            source = None
            result = None
            profit = None
            close_price = None

            expires_at = parse_iso(sig["expires_at"])
            elapsed = max(0.0, (now - expires_at).total_seconds())
            tf_seconds = TIMEFRAME_SECONDS.get(sig.get("timeframe"), 60)
            final_wait = max(SETTLEMENT_FINAL_WAIT_MIN, tf_seconds + 180)

            if has_real_order:
                # Primeiro tenta a fonte oficial. A chamada nunca pode prender o loop.
                try:
                    result_data = await asyncio.wait_for(
                        check_trade(session, trade["order_id"]),
                        timeout=SETTLEMENT_CHECK_TIMEOUT,
                    )
                except (asyncio.TimeoutError, TimeoutError, ConnectionError) as exc:
                    result_data = None
                    logger.warning(
                        "Timeout/ conexão consultando ordem %s: %s",
                        trade.get("order_id"), exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    result_data = None
                    logger.warning(
                        "Falha consultando ordem %s: %s", trade.get("order_id"), exc,
                    )

                if isinstance(result_data, dict) and result_data.get("done"):
                    raw_win = result_data.get("win")
                    raw = str(raw_win).strip().lower()
                    if raw_win is True or raw in {"win", "won", "true", "1"}:
                        result = "WIN"
                    elif raw_win is False or raw in {"loose", "loss", "lost", "false", "0"}:
                        result = "LOSS"
                    else:
                        try:
                            reported_profit = float(result_data.get("profit"))
                            result = "WIN" if reported_profit > 0 else (
                                "LOSS" if reported_profit < 0 else "EMPATE"
                            )
                        except (TypeError, ValueError):
                            result = "INDETERMINADO"

                    try:
                        raw_profit = result_data.get("profit")
                        profit = float(raw_profit) if raw_profit is not None else None
                    except (TypeError, ValueError):
                        profit = None
                    source = "BROKER"
                    update = {
                        "result": result,
                        "settlement_source": source,
                        "trade.status": "FECHADA",
                        "settled_at": now.isoformat(),
                        "settlement_next_check_at": None,
                    }
                    # Só registra PnL se a corretora realmente devolveu o valor.
                    if profit is not None:
                        update["trade.profit"] = profit
                        update["profit"] = profit
                    broker_balance_changed = True

                elif elapsed >= SETTLEMENT_FALLBACK_AFTER:
                    candles = await _get_settlement_candles(user_id, sig, session, now)
                    close_price = _price_at_expiry(sig, candles, now)
                    if close_price is not None:
                        result = _classify_candle_result(sig, close_price)
                        source = "CANDLE_FALLBACK"
                        update = {
                            "result": result,
                            "settlement_source": source,
                            "close_price": close_price,
                            "trade.status": (
                                "FECHADA_ESTIMADA" if result != "INDETERMINADO"
                                else "INDETERMINADA"
                            ),
                            "settled_at": now.isoformat(),
                            "settlement_next_check_at": None,
                            "settlement_note": "Resultado estimado por candle; não confirmado pela corretora.",
                        }
                        # Deliberadamente NÃO grava profit nem altera saldo de conta real.
                    elif elapsed >= final_wait:
                        result = "INDETERMINADO"
                        source = "TIMEOUT_SEM_DADOS"
                        update = {
                            "result": result,
                            "settlement_source": source,
                            "trade.status": "INDETERMINADA",
                            "settled_at": now.isoformat(),
                            "settlement_next_check_at": None,
                            "settlement_note": "Corretora sem confirmação e sem candle fechado válido.",
                        }
                    else:
                        attempts = int(sig.get("settlement_attempts") or 0) + 1
                        update = {
                            "settlement_attempts": attempts,
                            "settlement_last_error": "Ordem ainda não confirmada; aguardando candle fechado.",
                            "settlement_next_check_at": (now + timedelta(seconds=SETTLEMENT_RETRY_SECONDS)).isoformat(),
                        }
                else:
                    attempts = int(sig.get("settlement_attempts") or 0) + 1
                    update = {
                        "settlement_attempts": attempts,
                        "settlement_last_error": "Aguardando confirmação da corretora.",
                        "settlement_next_check_at": (now + timedelta(seconds=SETTLEMENT_RETRY_SECONDS)).isoformat(),
                    }

            else:
                # Sinal analítico ou operação simulada: candles são a fonte de resultado.
                candles = await _get_settlement_candles(user_id, sig, session, now)
                close_price = _price_at_expiry(sig, candles, now)
                if close_price is not None:
                    result = _classify_candle_result(sig, close_price)
                    source = "CANDLE_SIMULATED" if is_simulated_order else "CANDLE_ANALYTIC"
                    update = {
                        "result": result,
                        "settlement_source": source,
                        "close_price": close_price,
                        "settled_at": now.isoformat(),
                        "settlement_next_check_at": None,
                    }
                    if is_simulated_order and result in {"WIN", "LOSS", "EMPATE"}:
                        amount = float(trade.get("amount") or 0)
                        sim_profit = (
                            round(amount * float(SIM_PAYOUT), 2) if result == "WIN"
                            else (-amount if result == "LOSS" else 0.0)
                        )
                        if result == "WIN":
                            session.balance = round(
                                float(session.balance or 0) + amount + sim_profit, 2
                            )
                        update.update({
                            "trade.status": "FECHADA_SIMULADA",
                            "trade.profit": sim_profit,
                            "profit": sim_profit,
                        })
                elif elapsed >= final_wait:
                    result = "INDETERMINADO"
                    update = {
                        "result": result,
                        "settlement_source": "TIMEOUT_SEM_CANDLE",
                        "trade.status": "INDETERMINADA" if trade else None,
                        "settled_at": now.isoformat(),
                        "settlement_next_check_at": None,
                        "settlement_note": "Não foi possível obter candle fechado válido.",
                    }
                    if not trade:
                        update.pop("trade.status", None)
                else:
                    attempts = int(sig.get("settlement_attempts") or 0) + 1
                    update = {
                        "settlement_attempts": attempts,
                        "settlement_last_error": "Aguardando candle fechado válido.",
                        "settlement_next_check_at": (now + timedelta(seconds=SETTLEMENT_RETRY_SECONDS)).isoformat(),
                    }

            if not update:
                continue

            # Compare-and-set: somente um worker pode fechar uma pendência.
            result_update = await db.signals.update_one(
                {"id": signal_id, "user_id": user_id, "result": "PENDENTE"},
                {"$set": update},
            )
            if result_update.modified_count == 1 and result in {"WIN", "LOSS", "EMPATE", "INDETERMINADO"}:
                logger.info(
                    "Sinal liquidado user=%s signal=%s result=%s source=%s",
                    user_id, signal_id, result, source,
                )

        # Só atualiza saldo real quando a corretora confirmou ao menos uma ordem.
        if broker_balance_changed:
            try:
                await asyncio.wait_for(refresh_balance(session), timeout=8)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao atualizar saldo após liquidação confirmada: %s", exc)

        await _refresh_risk_state(user_id)


# ---------------------------------------------------------------------------
# Loop do scanner
# ---------------------------------------------------------------------------

async def _scan_loop(user_id: str) -> None:
    state = _state(user_id)
    session = _BROKER_SESSIONS.get(user_id)

    if not session:
        state["running"] = False
        state["message"] = "Nenhuma corretora conectada"
        return

    logger.info(
        "Scanner user=%s broker=%s simulated=%s",
        user_id,
        session.broker,
        session.simulated,
    )

    while state["running"]:
        try:
            if not session.connected:
                state["running"] = False
                state["message"] = "Corretora desconectada. Reconecte a conta."
                break

            # Mesmo pausado, acompanha a liquidação de sinais já abertos.
            if state.get("paused"):
                await _settle_pending(user_id, session)
                await asyncio.sleep(1.0)
                continue

            tf = state["timeframe"]
            all_candles = await get_candles_many(
                session,
                session.assets,
                tf,
                count=120,
            )
            tf_seconds = TIMEFRAME_SECONDS[tf]
            now_ts = int(utc_now().timestamp())

            results = []
            closed_candles = {}

            for asset, candles in all_candles.items():
                closed = [
                    c for c in candles
                    if int(c.get("ts", 0)) + tf_seconds <= now_ts
                ]
                if not closed:
                    continue

                closed_candles[asset] = closed
                analysis = analyze_candles(closed)

                if analysis.get("ready") and analysis.get("direction") != "NEUTRO":
                    results.append(
                        {
                            "asset": asset,
                            "candle_ts": int(closed[-1]["ts"]),
                            **analysis,
                        }
                    )

            state["candles"] = closed_candles
            results.sort(
                key=lambda item: float(item.get("confidence") or 0),
                reverse=True,
            )
            state["ranking"] = results[:10]
            state["last_scan_ts"] = iso_now()
            state["best"] = results[0] if results else None

            best = state["best"]
            if best and float(best.get("confidence") or 0) >= state["min_confidence"]:
                last_candle = state["last_signal_candle"].get(best["asset"])
                same = await db.signals.find_one(
                    {
                        "user_id": user_id,
                        "session_id": state["session_id"],
                        "asset": best["asset"],
                        "candle_ts": best["candle_ts"],
                    },
                    {"_id": 1},
                )

                if last_candle != best["candle_ts"] and not same:
                    doc = _signal_doc(user_id, best, tf, session, state)
                    state["last_signal_candle"][best["asset"]] = best["candle_ts"]

                    if state["auto_trade"]:
                        await _auto_trade(user_id, state, session, doc)

                    await db.signals.insert_one(doc)
                    state["message"] = (
                        f"Vela {tf} fechada — sinal: {best['asset']} "
                        f"{best['direction']} ({best['confidence']}%)"
                    )

            elif not results:
                state["message"] = "Analisando mercado..."

            await _settle_pending(user_id, session)
            await asyncio.sleep(3.0)

        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError) as exc:
            logger.warning("Conexão perdida user=%s: %s", user_id, exc)
            state["running"] = False
            session.connected = False
            state["message"] = "Conexão com a corretora perdida. Reconecte a conta."
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro no scan loop: %s", exc)
            state["message"] = f"Erro temporário no scanner: {str(exc)[:140]}"
            await asyncio.sleep(2.0)

    logger.info("Scanner parado user=%s", user_id)


async def _start_scan_task(user_id: str) -> None:
    state = _state(user_id)
    old = _SCAN_TASKS.get(user_id)
    if old and not old.done():
        old.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await old
    _SCAN_TASKS[user_id] = asyncio.create_task(_scan_loop(user_id))


async def _stop_scan(user_id: str) -> None:
    state = _state(user_id)
    state["running"] = False
    task = _SCAN_TASKS.pop(user_id, None)
    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


async def _start_scan_with_config(
    user_id: str,
    *,
    timeframe: str,
    min_confidence: int,
    auto_trade: bool,
    amount: float,
    target_gain: float,
    max_loss: float,
    new_session: bool,
) -> dict:
    session = _BROKER_SESSIONS.get(user_id)
    if not session or not session.connected:
        raise HTTPException(
            status_code=400,
            detail="Conecte uma corretora antes de iniciar o scanner",
        )

    if auto_trade and amount > float(session.balance or 0):
        raise HTTPException(
            status_code=400,
            detail="Valor da entrada maior que o saldo confirmado",
        )

    state = _state(user_id)
    if state["running"]:
        return {
            "ok": True,
            "running": True,
            "message": "Scanner já está em execução",
            "session_id": state["session_id"],
        }

    if new_session:
        _start_new_session(state, "Nova sessão iniciada")

    state.update(
        {
            "running": True,
            "paused": False,
            "timeframe": timeframe,
            "min_confidence": min_confidence,
            "auto_trade": auto_trade,
            "amount": amount,
            "target_gain": target_gain,
            "max_loss": max_loss,
            "message": "Iniciando varredura do mercado...",
        }
    )

    await _refresh_risk_state(user_id)
    await _start_scan_task(user_id)

    return {
        "ok": True,
        "running": True,
        "message": "Análise iniciada",
        "session_id": state["session_id"],
    }


# ---------------------------------------------------------------------------
# Rotas de autenticação
# ---------------------------------------------------------------------------

@api.get("/")
async def api_root():
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
        "created_at": iso_now(),
    }
    await db.users.insert_one(user)

    token = make_token(user["id"], user["email"])
    set_auth_cookie(response, token)
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": email,
            "name": user["name"],
            "role": "user",
        },
    }


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
        "user": {
            "id": user["id"],
            "email": email,
            "name": user["name"],
            "role": user.get("role", "user"),
        },
    }


@api.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"ok": True}


@api.get("/auth/me")
async def me(user=Depends(get_current_user)):
    return {"user": user}


# ---------------------------------------------------------------------------
# Rotas da corretora
# ---------------------------------------------------------------------------

@api.post("/broker/connect")
async def broker_connect(
    payload: BrokerConnectIn,
    user=Depends(get_current_user),
):
    user_id = user["id"]
    old = _BROKER_SESSIONS.pop(user_id, None)

    if old:
        await _stop_scan(user_id)
        disconnect_broker(old)

    try:
        session = await connect_broker(
            payload.broker,
            payload.email,
            payload.password,
            payload.account_type,
            payload.simulate,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    _BROKER_SESSIONS[user_id] = session
    return _session_out(session)


@api.post("/broker/disconnect")
async def broker_disconnect(user=Depends(get_current_user)):
    user_id = user["id"]
    await _stop_scan(user_id)

    state = _state(user_id)
    state["best"] = None
    state["ranking"] = []
    state["paused"] = False
    state["auto_trade"] = False
    state["message"] = "Desconectado"

    session = _BROKER_SESSIONS.pop(user_id, None)
    if session:
        disconnect_broker(session)

    return {"ok": True, "connected": False}


@api.get("/broker/status")
async def broker_status(
    refresh: bool = Query(default=False),
    user=Depends(get_current_user),
):
    session = _BROKER_SESSIONS.get(user["id"])
    if not session or not session.connected:
        return {
            "connected": False,
            "session": {"connected": False},
            "message": "Nenhuma corretora conectada",
        }

    if refresh:
        try:
            await refresh_balance(session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao atualizar saldo da corretora: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"Não foi possível atualizar o saldo diretamente na corretora: {str(exc)[:160]}",
            )

    output = _session_out(session)
    return {
        **output,
        "session": output,
        "balance_refreshed": bool(refresh),
        "balance_updated_at": iso_now() if refresh else None,
    }


@api.post("/broker/refresh-balance")
async def broker_refresh_balance(user=Depends(get_current_user)):
    session = _BROKER_SESSIONS.get(user["id"])
    if not session or not session.connected:
        raise HTTPException(status_code=400, detail="Corretora desconectada")

    try:
        await refresh_balance(session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao atualizar saldo diretamente na corretora")
        raise HTTPException(
            status_code=502,
            detail=f"Falha ao consultar saldo na corretora: {str(exc)[:160]}",
        )

    output = _session_out(session)
    return {
        "ok": True,
        **output,
        "session": output,
        "balance_refreshed": True,
        "balance_updated_at": iso_now(),
    }


# ---------------------------------------------------------------------------
# Rotas do scanner
# ---------------------------------------------------------------------------

@api.post("/scan/start")
async def scan_start(
    payload: ScanStartIn,
    user=Depends(get_current_user),
):
    return await _start_scan_with_config(
        user["id"],
        timeframe=payload.timeframe,
        min_confidence=payload.min_confidence,
        auto_trade=payload.auto_trade,
        amount=payload.amount,
        target_gain=payload.target_gain,
        max_loss=payload.max_loss,
        new_session=True,
    )


@api.post("/scan/stop")
async def scan_stop(user=Depends(get_current_user)):
    user_id = user["id"]
    await _stop_scan(user_id)

    state = _state(user_id)
    state["paused"] = False
    state["auto_trade"] = False
    state["message"] = "Scanner parado"

    return {"ok": True, "running": False, "message": state["message"]}


@api.get("/scan/state")
async def scan_state(user=Depends(get_current_user)):
    user_id = user["id"]
    state = await _refresh_risk_state(user_id)

    tf = state["timeframe"]
    tf_seconds = TIMEFRAME_SECONDS[tf]
    now_ts = int(utc_now().timestamp())
    candle_close_ts = (now_ts // tf_seconds + 1) * tf_seconds

    session_query = {
        "user_id": user_id,
        "session_id": state["session_id"],
    }
    wins = await db.signals.count_documents({**session_query, "result": "WIN"})
    losses = await db.signals.count_documents({**session_query, "result": "LOSS"})
    pending = await db.signals.count_documents({**session_query, "result": "PENDENTE"})
    indeterminate = await db.signals.count_documents({**session_query, "result": "INDETERMINADO"})
    total = wins + losses

    return {
        "running": state["running"],
        "paused": state["paused"],
        "timeframe": tf,
        "min_confidence": state["min_confidence"],
        "auto_trade": state["auto_trade"],
        "amount": state["amount"],
        "target_gain": state["target_gain"],
        "max_loss": state["max_loss"],
        "last_scan_ts": state["last_scan_ts"],
        "candle_close_ts": candle_close_ts,
        "server_time": now_ts,
        "best": state["best"],
        "ranking": state["ranking"],
        "message": state["message"],
        "daily_profit": state["daily_profit"],
        "risk_halted": state["risk_halted"],
        "risk_reason": state["risk_reason"],
        "session_id": state["session_id"],
        "session_started_at": state["session_started_at"],
        "session_reason": state["session_reason"],
        "session_stats": {
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "indeterminate": indeterminate,
            "total": total,
            "winrate": round(wins * 100 / total, 1) if total else 0,
        },
    }


# ---------------------------------------------------------------------------
# Rotas de controle do robô
# ---------------------------------------------------------------------------

@api.post("/robot/start")
async def robot_start(
    payload: Optional[RobotConfigIn] = None,
    user=Depends(get_current_user),
):
    user_id = user["id"]
    state = _state(user_id)
    config = payload or RobotConfigIn(
        auto_trade=True,
        amount=state["amount"],
        target_gain=state["target_gain"],
        max_loss=state["max_loss"],
    )

    if state["running"]:
        if state.get("risk_halted"):
            raise HTTPException(
                status_code=423,
                detail="Robô bloqueado pela proteção diária: " + state["risk_reason"],
            )
        state["auto_trade"] = True
        state["paused"] = False
        return {"ok": True, "message": "Robô automático ativado", "running": True}

    return await _start_scan_with_config(
        user_id,
        timeframe=state["timeframe"],
        min_confidence=state["min_confidence"],
        auto_trade=True,
        amount=config.amount,
        target_gain=config.target_gain,
        max_loss=config.max_loss,
        new_session=True,
    )


@api.post("/robot/stop")
async def robot_stop(user=Depends(get_current_user)):
    state = _state(user["id"])
    state["auto_trade"] = False
    state["paused"] = False
    state["message"] = "Robô automático desativado; scanner analítico preservado."
    return {
        "ok": True,
        "auto_trade": False,
        "running": state["running"],
        "message": state["message"],
    }


@api.post("/robot/pause")
async def robot_pause(user=Depends(get_current_user)):
    state = _state(user["id"])
    if not state["running"]:
        raise HTTPException(status_code=400, detail="Inicie o scanner antes de pausar.")

    state["paused"] = not state["paused"]
    state["message"] = (
        "Robô pausado. Ordens novas bloqueadas; sinais pendentes continuam sendo acompanhados."
        if state["paused"]
        else "Robô retomado."
    )
    return {
        "ok": True,
        "paused": state["paused"],
        "message": state["message"],
    }


@api.post("/robot/reset")
async def robot_reset(user=Depends(get_current_user)):
    user_id = user["id"]
    state = _state(user_id)

    # O reset cria uma sessão nova. A sessão anterior permanece no arquivo.
    _start_new_session(state, "Placar reiniciado: nova sessão")
    await _refresh_risk_state(user_id)

    return {
        "ok": True,
        "session_id": state["session_id"],
        "session_started_at": state["session_started_at"],
        "message": "Nova sessão criada. O placar desta sessão começa do zero.",
    }


@api.post("/sessions/start")
async def session_start(
    payload: SessionStartIn,
    user=Depends(get_current_user),
):
    state = _state(user["id"])
    _start_new_session(state, payload.reason)
    await _refresh_risk_state(user["id"])
    return {
        "ok": True,
        "session_id": state["session_id"],
        "session_started_at": state["session_started_at"],
        "message": payload.reason,
    }


# ---------------------------------------------------------------------------
# Histórico atual e arquivo de sessões
# ---------------------------------------------------------------------------

def _history_stats(items: list[dict]) -> dict:
    wins = sum(1 for item in items if item.get("result") == "WIN")
    losses = sum(1 for item in items if item.get("result") == "LOSS")
    total = wins + losses

    profit = 0.0
    for item in items:
        try:
            profit += float(item.get("profit") or (item.get("trade") or {}).get("profit") or 0)
        except (TypeError, ValueError):
            continue

    return {
        "wins": wins,
        "losses": losses,
        "pending": sum(1 for item in items if item.get("result") == "PENDENTE"),
        "ties": sum(1 for item in items if item.get("result") == "EMPATE"),
        "indeterminate": sum(1 for item in items if item.get("result") == "INDETERMINADO"),
        "total": total,
        "winrate": round(wins * 100 / total, 1) if total else 0,
        "profit": round(profit, 2),
    }


@api.get("/signals/history")
async def signals_history(
    user=Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=500),
):
    state = _state(user["id"])
    query = {
        "user_id": user["id"],
        "session_id": state["session_id"],
    }
    cursor = (
        db.signals.find(query, {"_id": 0})
        .sort("ts", DESCENDING)
        .limit(limit)
    )
    items = await cursor.to_list(length=limit)

    stats_cursor = db.signals.find(query, {"_id": 0}).sort("ts", DESCENDING)
    all_session_items = await stats_cursor.to_list(length=5000)

    return {
        "items": items,
        "stats": _history_stats(all_session_items),
        "session_id": state["session_id"],
        "session_started_at": state["session_started_at"],
    }


@api.get("/signals/sessions")
async def signals_sessions(
    user=Depends(get_current_user),
    limit: int = Query(default=100, ge=1, le=500),
):
    pipeline = [
        {"$match": {"user_id": user["id"], "session_id": {"$exists": True}}},
        {
            "$group": {
                "_id": "$session_id",
                "session_started_at": {"$first": "$session_started_at"},
                "first_signal_at": {"$min": "$ts"},
                "last_signal_at": {"$max": "$ts"},
                "signals": {"$sum": 1},
                "wins": {
                    "$sum": {"$cond": [{"$eq": ["$result", "WIN"]}, 1, 0]}
                },
                "losses": {
                    "$sum": {"$cond": [{"$eq": ["$result", "LOSS"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"last_signal_at": -1}},
        {"$limit": limit},
    ]
    sessions = await db.signals.aggregate(pipeline).to_list(length=limit)

    for item in sessions:
        total = item.get("wins", 0) + item.get("losses", 0)
        item["session_id"] = item.pop("_id")
        item["winrate"] = round(item["wins"] * 100 / total, 1) if total else 0

    return {"sessions": sessions}


@api.get("/signals/archive")
async def signals_archive(
    user=Depends(get_current_user),
    result: Optional[str] = Query(default=None),
    asset: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    time_from: Optional[str] = Query(default=None),
    time_to: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    query: dict = {
        "user_id": user["id"],
        "session_id": {"$exists": True},
    }

    if session_id:
        query["session_id"] = session_id

    if result:
        normalized = result.upper()
        if normalized not in {"WIN", "LOSS", "PENDENTE", "EMPATE", "INDETERMINADO"}:
            raise HTTPException(status_code=400, detail="Filtro de resultado inválido")
        query["result"] = normalized

    if asset:
        query["asset"] = {"$regex": asset.strip(), "$options": "i"}

    # Filtros de data e horário são convertidos em intervalo ISO UTC.
    if date_from or date_to or time_from or time_to:
        # Se o usuário filtrar apenas por horário, aplica o intervalo ao dia atual (UTC).
        today_utc = utc_now().date().isoformat()
        start_date = date_from or (today_utc if (time_from or time_to) else "1970-01-01")
        end_date = date_to or today_utc
        start_clock = time_from or "00:00"
        end_clock = time_to or "23:59"

        try:
            start_dt = datetime.fromisoformat(f"{start_date}T{start_clock}:00").replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(f"{end_date}T{end_clock}:59").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Data ou horário inválido")

        query["ts"] = {
            "$gte": start_dt.isoformat(),
            "$lte": end_dt.isoformat(),
        }

    cursor = (
        db.signals.find(query, {"_id": 0})
        .sort("ts", DESCENDING)
        .limit(limit)
    )
    items = await cursor.to_list(length=limit)
    return {"items": items, "count": len(items), "filters": {
        "result": result,
        "asset": asset,
        "date_from": date_from,
        "date_to": date_to,
        "time_from": time_from,
        "time_to": time_to,
        "session_id": session_id,
    }}


@api.post("/signals/clear-current-session")
async def clear_current_session_history(user=Depends(get_current_user)):
    """Zera somente o placar corrente, preservando sessões arquivadas."""
    state = _state(user["id"])
    old_session_id = state["session_id"]

    # Em vez de apagar documentos, encerra a sessão e inicia outra.
    _start_new_session(state, "Placar limpo: nova sessão")
    await _refresh_risk_state(user["id"])

    return {
        "ok": True,
        "previous_session_id": old_session_id,
        "session_id": state["session_id"],
        "message": "Placar corrente zerado; sessão anterior preservada no arquivo.",
    }


# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

@api.get("/chart")
async def chart(asset: str, user=Depends(get_current_user)):
    state = _state(user["id"])
    candles = state["candles"].get(asset)

    if not candles:
        raise HTTPException(
            status_code=404,
            detail="Sem dados para este ativo ainda",
        )

    close = pd.Series([c["close"] for c in candles])
    ma9 = sma(close, 9)
    ma21 = sma(close, 21)

    out = candles[-80:]
    offset = len(candles) - len(out)

    return {
        "asset": asset,
        "timeframe": state["timeframe"],
        "candles": out,
        "ma9": [
            {"ts": c["ts"], "value": float(ma9.iloc[offset + i])}
            for i, c in enumerate(out)
            if not pd.isna(ma9.iloc[offset + i])
        ],
        "ma21": [
            {"ts": c["ts"], "value": float(ma21.iloc[offset + i])}
            for i, c in enumerate(out)
            if not pd.isna(ma21.iloc[offset + i])
        ],
    }


# ---------------------------------------------------------------------------
# Servir o HTML
# ---------------------------------------------------------------------------

app.include_router(api)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    candidates = (
        ROOT_DIR / "template" / "index.html",
        ROOT_DIR / "templates" / "index.html",
        ROOT_DIR / "index.html",
        ROOT_DIR / "Purple_Signal_robo_interface.html",
    )
    for html_file in candidates:
        if html_file.is_file():
            return FileResponse(html_file)

    raise HTTPException(
        status_code=404,
        detail="Interface HTML não encontrada na raiz do projeto.",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Índices e migração única do histórico antigo
# ---------------------------------------------------------------------------

INDEXES: dict[str, list[IndexModel]] = {
    "users": [
        IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
        IndexModel([("id", ASCENDING)], unique=True, name="id_unique"),
    ],
    "signals": [
        IndexModel([("user_id", ASCENDING), ("ts", DESCENDING)], name="user_ts"),
        IndexModel(
            [("user_id", ASCENDING), ("session_id", ASCENDING), ("ts", DESCENDING)],
            name="user_session_ts",
        ),
        IndexModel(
            [("user_id", ASCENDING), ("result", ASCENDING), ("asset", ASCENDING), ("ts", DESCENDING)],
            name="archive_filters",
        ),
    ],
}


async def clear_legacy_signals_once() -> None:
    """Compatibilidade com startup antigo: preserva integralmente o histórico."""
    marker_id = "purple_signal_history_preserved_v2"
    await db.app_meta.update_one(
        {"_id": marker_id},
        {"$setOnInsert": {"completed": True, "completed_at": iso_now(), "deleted_count": 0}},
        upsert=True,
    )
    logger.info("Migração de histórico: nenhum sinal foi apagado.")


async def seed_admin() -> None:
    email = os.environ.get("ADMIN_EMAIL", "admin@sinais.com").lower()
    password = os.environ.get("ADMIN_PASSWORD", "admin123")

    existing = await db.users.find_one({"email": email})
    if existing:
        return

    await db.users.insert_one(
        {
            "id": str(uuid.uuid4()),
            "email": email,
            "name": "Administrador",
            "password_hash": hash_password(password),
            "role": "admin",
            "created_at": iso_now(),
        }
    )
    logger.info("Admin seed criado: %s", email)


@app.on_event("startup")
async def startup() -> None:
    for collection, models in INDEXES.items():
        for model in models:
            with suppress(Exception):
                await db[collection].create_indexes([model])

    # Migração não destrutiva: preserva todo o histórico existente.
    await clear_legacy_signals_once()
    await seed_admin()


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in list(_SCAN_TASKS.values()):
        task.cancel()
    _SCAN_TASKS.clear()

    for session in list(_BROKER_SESSIONS.values()):
        with suppress(Exception):
            disconnect_broker(session)

    client.close()
