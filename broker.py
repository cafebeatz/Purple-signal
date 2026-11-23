"""
Camada de corretoras: IQ Option e Polarium Broker.
Cada sessão real roda num processo separado (broker_worker) para não travar o servidor.
Modo simulado: mercado sintético persistente por ativo/timeframe.
"""
from __future__ import annotations
import asyncio
import logging
import multiprocessing as mp
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from broker_worker import worker_main
logger = logging.getLogger(__name__)
_CTX = mp.get_context("spawn")
DEFAULT_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "EURGBP", "GBPJPY", "EURCHF", "AUDJPY", "AUDCAD", "CADJPY",
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDCAD-OTC", "EURJPY-OTC",
]
MAX_REAL_ASSETS = 60
TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900}
SIM_PAYOUT = 0.85
@dataclass
class BrokerSession:
    broker: str
    email: str
    account_type: str = "demo"
    connected: bool = False
    simulated: bool = True
    balance: float = 10000.0
    currency: str = "USD"
    message: str = ""
    assets: list[str] = field(default_factory=list)
    market_assets: dict[str, list[str]] = field(default_factory=lambda: {"open": [], "otc": []})
    proc: Optional[object] = field(default=None, repr=False)
    conn: Optional[object] = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
async def _recv(conn, timeout: float):
    loop = asyncio.get_running_loop()
    ready = await loop.run_in_executor(None, conn.poll, timeout)
    if not ready:
        raise TimeoutError("Corretora não respondeu a tempo")
    return conn.recv()
async def _call(session: BrokerSession, cmd: dict, timeout: float = 45.0) -> dict:
    async with session.lock:
        if session.conn is None or session.proc is None or not session.proc.is_alive():
            raise ConnectionError("Conexão com a corretora perdida")
        session.conn.send(cmd)
        try:
            return await _recv(session.conn, timeout)
        except TimeoutError:
            _kill(session)
            session.message = "A corretora parou de responder. Reconecte."
            raise
def _kill(session: BrokerSession) -> None:
    if session.proc is not None:
        try:
            session.proc.terminate()
            session.proc.join(2)
            if session.proc.is_alive():
                session.proc.kill()
        except Exception:  # noqa: BLE001
            pass
    session.proc = None
    session.conn = None
def _broker_name(broker: str) -> str:
    return "Polarium Broker" if broker == "polarium" else "IQ Option"
def market_type_for_asset(asset: str) -> str:
    """Classificação operacional: OTC pelo nome; demais ativos são mercado regular."""
    return "otc" if "OTC" in str(asset).upper() else "open"


def split_assets_by_market(assets: list[str]) -> dict[str, list[str]]:
    grouped = {"open": [], "otc": []}
    for asset in sorted(set(str(a).strip() for a in assets if str(a).strip())):
        grouped[market_type_for_asset(asset)].append(asset)
    return grouped


async def connect_broker(broker: str, email: str, password: str, account_type: str, simulate: bool = False) -> BrokerSession:
    session = BrokerSession(broker=broker, email=email, account_type=account_type)
    if simulate:
        session.assets = list(DEFAULT_ASSETS)
        session.market_assets = split_assets_by_market(session.assets)
        session.connected = True
        session.message = "Modo simulado escolhido pelo usuário (mercado aberto + OTC sintéticos)"
        return session
    parent, child = _CTX.Pipe()
    proc = _CTX.Process(target=worker_main, args=(child, broker, email, password, account_type), daemon=True)
    proc.start()
    session.proc, session.conn = proc, parent
    try:
        res = await _recv(parent, 90.0)
    except TimeoutError:
        _kill(session)
        res = {"ok": False, "error": "Timeout na conexão com a corretora"}
    if res.get("ok"):
        session.connected = True
        session.simulated = False
        session.balance = res["balance"]
        session.currency = res["currency"]
        # Usa somente ativos confirmados pela API da corretora; sem fallback misto.
        supplied_groups = res.get("assets_by_market") or {}
        if supplied_groups:
            grouped = split_assets_by_market(
                list(supplied_groups.get("open") or []) +
                list(supplied_groups.get("otc") or [])
            )
        else:
            grouped = split_assets_by_market(list(res.get("assets") or []))

        # Limite independente por mercado para preservar espaço para os dois.
        session.market_assets = {
            "open": grouped["open"][:MAX_REAL_ASSETS],
            "otc": grouped["otc"][:MAX_REAL_ASSETS],
        }
        session.assets = session.market_assets["open"] + session.market_assets["otc"]
        logger.info(
            "Ativos confirmados de %s: total=%d, mercado_aberto=%d, otc=%d",
            _broker_name(broker), len(session.assets),
            len(session.market_assets["open"]), len(session.market_assets["otc"])
        )
        if not session.assets:
            session.message = (
                f"Conectado à {_broker_name(broker)}, mas a API não confirmou ativos. "
                "Scanner bloqueado para evitar usar uma lista fictícia."
            )
        else:
            session.message = f"Conectado à {_broker_name(broker)} ({account_type.upper()})"
        return session
    _kill(session)
    msg = str(res.get("error", ""))
    if "invalid_credentials" in msg or "wrong credentials" in msg:
        raise PermissionError("Login ou senha da corretora incorretos")
    if "2fa" in msg.lower() or "two_factor" in msg or "verify" in msg.lower():
        raise PermissionError("Sua conta usa verificação em 2 etapas — desative-a na corretora para conectar")
    logger.warning("Falha conectando %s: %s", broker, msg[:300])
    raise ConnectionError(f"Não foi possível conectar à {_broker_name(broker)}: {msg[:160] or 'erro desconhecido'}")
def disconnect_broker(session: BrokerSession) -> None:
    if session.conn is not None:
        try:
            session.conn.send({"op": "close"})
        except Exception:  # noqa: BLE001
            pass
    _kill(session)
    session.connected = False
# ------------------ Mercado simulado ------------------
_SIM_STATE: dict[str, dict] = {}
def _new_candle(rng: random.Random, ts: int, price: float, tf: int, phase: float) -> dict:
    drift = np.sin(ts / (tf * 40) + phase) * 0.0006
    vol = 0.0012 + rng.random() * 0.0008
    close_p = price + rng.gauss(drift, vol)
    return {
        "ts": ts, "open": price,
        "high": max(price, close_p) + abs(rng.gauss(0, vol / 2)),
        "low": min(price, close_p) - abs(rng.gauss(0, vol / 2)),
        "close": close_p, "volume": rng.randint(100, 5000),
    }
def _simulate_candles(asset: str, timeframe: str, count: int) -> list[dict]:
    tf = TIMEFRAME_SECONDS.get(timeframe, 60)
    key = f"{asset}:{timeframe}"
    now_bucket = int(time.time()) // tf * tf
    st = _SIM_STATE.get(key)
    if st is None:
        rng = random.Random(f"{asset}{timeframe}")
        price = 0.8 + rng.random() * 1.2
        phase = rng.random() * 6.28
        candles = []
        for i in range(count, 0, -1):
            c = _new_candle(rng, now_bucket - i * tf, price, tf, phase)
            candles.append(c)
            price = c["close"]
        candles.append(_new_candle(rng, now_bucket, price, tf, phase))
        st = _SIM_STATE[key] = {"rng": rng, "phase": phase, "candles": candles}
    rng, candles = st["rng"], st["candles"]
    while candles[-1]["ts"] < now_bucket:
        candles.append(_new_candle(rng, candles[-1]["ts"] + tf, candles[-1]["close"], tf, st["phase"]))
    cur = candles[-1]
    cur["close"] += rng.gauss(0, 0.0003)
    cur["high"] = max(cur["high"], cur["close"])
    cur["low"] = min(cur["low"], cur["close"])
    del candles[:-max(count, 150)]
    return [dict(c) for c in candles[-count:]]
# ------------------ API usada pelo servidor ------------------
async def get_candles_many(session: BrokerSession, assets: list[str], timeframe: str, count: int = 120) -> dict:
    if session.simulated:
        return {a: _simulate_candles(a, timeframe, count) for a in assets}
    res = await _call(session, {"op": "candles_many", "assets": assets,
                                "tf": TIMEFRAME_SECONDS[timeframe], "count": count}, timeout=90.0)
    return res.get("data", {}) if res.get("ok") else {}
async def get_candles(session: BrokerSession, asset: str, timeframe: str, count: int = 120) -> list[dict]:
    return (await get_candles_many(session, [asset], timeframe, count)).get(asset, [])
async def place_trade(session: BrokerSession, asset: str, direction: str, amount: float, timeframe: str) -> str:
    if asset not in session.assets:
        raise ValueError(f"Ativo {asset} não foi confirmado na lista da corretora.")
    if session.simulated:
        if amount > session.balance:
            raise ValueError("Saldo simulado insuficiente")
        session.balance = round(session.balance - amount, 2)
        return f"SIM-{uuid.uuid4().hex[:10]}"
    minutes = TIMEFRAME_SECONDS[timeframe] // 60
    res = await _call(session, {"op": "buy", "asset": asset, "action": direction.lower(),
                                "amount": amount, "minutes": minutes}, timeout=20.0)
    if not res.get("ok"):
        raise ValueError(res.get("error") or "Ordem recusada")
    return str(res["order_id"])
async def check_trade(session: BrokerSession, order_id: str) -> dict:
    res = await _call(session, {"op": "check", "order_id": int(order_id)}, timeout=10.0)
    return res if res.get("ok") else {"done": False}
async def refresh_balance(session: BrokerSession) -> None:
    if session.simulated:
        return
    res = await _call(session, {"op": "balance"}, timeout=10.0)
    if res.get("ok"):
        session.balance = res["balance"]
