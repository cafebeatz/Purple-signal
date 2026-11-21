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
MAX_REAL_ASSETS = 25
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
    assets: list[str] = field(default_factory=lambda: list(DEFAULT_ASSETS))
    proc: Optional[object] = field(default=None, repr=False)
    conn: Optional[object] = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


async def _recv(conn, timeout: float):
    """Recebe uma resposta normal do worker (sem mensagens de progresso)."""
    loop = asyncio.get_running_loop()
    ready = await loop.run_in_executor(None, conn.poll, timeout)
    if not ready:
        raise TimeoutError("Corretora não respondeu a tempo")
    return conn.recv()


async def _recv_connect(conn, proc, timeout: float = 45.0) -> dict:
    """Aguarda resposta final e consome mensagens de progresso do worker."""
    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + timeout
    last_stage = "inicialização do processo"

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timeout após {timeout:.0f}s. Última etapa: {last_stage}"
            )

        ready = await loop.run_in_executor(
            None, conn.poll, min(0.5, remaining)
        )

        if ready:
            try:
                message = conn.recv()
            except EOFError as exc:
                raise ConnectionError(
                    f"O processo de conexão fechou o canal. Última etapa: {last_stage}"
                ) from exc

            if isinstance(message, dict) and message.get("_stage"):
                last_stage = str(message["_stage"])
                logger.info("Broker worker (%s): %s", proc.pid, last_stage)
                continue

            if not isinstance(message, dict):
                raise ConnectionError("Resposta inválida recebida do processo da corretora")
            return message

        if not proc.is_alive():
            proc.join(timeout=0)
            # Ainda pode existir uma última mensagem no canal.
            if conn.poll(0):
                continue
            raise ConnectionError(
                "Processo de conexão encerrou antes da resposta final. "
                f"Exit code: {proc.exitcode}; última etapa: {last_stage}"
            )


def _call_sync_poll(conn, timeout: float) -> bool:
    """Função síncrona pequena para usar no executor do event loop."""
    return conn.poll(timeout)


async def _call(session: BrokerSession, cmd: dict, timeout: float = 30.0) -> dict:
    async with session.lock:
        if session.conn is None or session.proc is None or not session.proc.is_alive():
            raise ConnectionError("Conexão com a corretora perdida")
        try:
            session.conn.send(cmd)
            return await _recv(session.conn, timeout)
        except TimeoutError:
            _kill(session)
            session.message = "A corretora parou de responder. Reconecte."
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            _kill(session)
            session.message = "O processo de conexão foi encerrado. Reconecte."
            raise ConnectionError(session.message) from exc


def _kill(session: BrokerSession) -> None:
    """Encerra o worker e fecha o endpoint IPC pertencente à sessão."""
    proc = session.proc
    conn = session.conn

    if proc is not None:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(1)
            else:
                proc.join(timeout=0)
        except Exception:  # noqa: BLE001
            logger.exception("Erro ao encerrar broker worker")

    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    session.proc = None
    session.conn = None


def _broker_name(broker: str) -> str:
    return "Polarium Broker" if broker == "polarium" else "IQ Option"


async def connect_broker(
    broker: str,
    email: str,
    password: str,
    account_type: str,
    simulate: bool = False,
) -> BrokerSession:
    session = BrokerSession(broker=broker, email=email, account_type=account_type)
    if simulate:
        session.connected = True
        session.message = "Modo simulado escolhido pelo usuário"
        return session

    parent, child = _CTX.Pipe(duplex=True)
    proc = _CTX.Process(
        target=worker_main,
        args=(child, broker, email, password, account_type),
        daemon=True,
    )

    try:
        proc.start()
    except Exception:
        parent.close()
        child.close()
        raise

    # O processo pai não deve manter aberto o endpoint destinado ao filho.
    child.close()
    session.proc, session.conn = proc, parent

    try:
        res = await _recv_connect(parent, proc, 45.0)
    except (TimeoutError, ConnectionError, EOFError, OSError) as exc:
        logger.error("Falha no handshake de %s: %s", broker, exc)
        _kill(session)
        res = {"ok": False, "error": str(exc)}

    if res.get("ok"):
        session.connected = True
        session.simulated = False
        session.balance = res["balance"]
        session.currency = res["currency"]
        if res.get("assets"):
            otc_last = sorted(res["assets"], key=lambda a: ("OTC" in a, a))
            session.assets = otc_last[:MAX_REAL_ASSETS]
        session.message = f"Conectado à {_broker_name(broker)} ({account_type.upper()})"
        return session

    _kill(session)
    msg = str(res.get("error", ""))
    if "invalid_credentials" in msg or "wrong credentials" in msg:
        raise PermissionError("Login ou senha da corretora incorretos")
    if "2fa" in msg.lower() or "two_factor" in msg or "verify" in msg.lower():
        raise PermissionError("Sua conta usa verificação em 2 etapas — desative-a na corretora para conectar")
    logger.warning("Falha conectando %s: %s", broker, msg[:300])
    raise ConnectionError(
        f"Não foi possível conectar à {_broker_name(broker)}: "
        f"{msg[:240] or 'erro desconhecido'}"
    )


def disconnect_broker(session: BrokerSession) -> None:
    if session.conn is not None:
        try:
            session.conn.send({"op": "close"})
            if session.proc is not None:
                session.proc.join(timeout=0.5)
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
async def get_candles_many(
    session: BrokerSession, assets: list[str], timeframe: str, count: int = 120
) -> dict:
    if session.simulated:
        return {a: _simulate_candles(a, timeframe, count) for a in assets}
    res = await _call(
        session,
        {"op": "candles_many", "assets": assets,
         "tf": TIMEFRAME_SECONDS[timeframe], "count": count},
        timeout=90.0,
    )
    return res.get("data", {}) if res.get("ok") else {}


async def get_candles(
    session: BrokerSession, asset: str, timeframe: str, count: int = 120
) -> list[dict]:
    return (await get_candles_many(session, [asset], timeframe, count)).get(asset, [])


async def place_trade(
    session: BrokerSession, asset: str, direction: str, amount: float, timeframe: str
) -> str:
    if session.simulated:
        if amount > session.balance:
            raise ValueError("Saldo simulado insuficiente")
        session.balance = round(session.balance - amount, 2)
        return f"SIM-{uuid.uuid4().hex[:10]}"
    minutes = TIMEFRAME_SECONDS[timeframe] // 60
    res = await _call(
        session,
        {"op": "buy", "asset": asset, "action": direction.lower(),
         "amount": amount, "minutes": minutes},
        timeout=20.0,
    )
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
