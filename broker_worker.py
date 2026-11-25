"""Processo isolado por sessão de corretora.

O worker envia mensagens de etapa durante o handshake para que o processo pai
consiga distinguir inicialização, autenticação e consultas posteriores.
"""
from __future__ import annotations

import time
import traceback

HOSTS = {
    "iqoption": {
        "ws": "wss://iqoption.com/echo/websocket",
        "https": "https://iqoption.com/api",
        "auth": "auth.iqoption.com",
        "event": "event.iqoption.com",
    },
    "polarium": {
        "ws": "wss://ws.trade.polariumbroker.com/echo/websocket",
        "https": "https://trade.polariumbroker.com/api",
        "auth": "auth.trade.polariumbroker.com",
        "event": "event.trade.polariumbroker.com",
    },
}


def _send_stage(conn, stage: str) -> None:
    """Envia progresso sem incluir usuário, senha ou tokens."""
    try:
        conn.send({"_stage": stage})
    except (BrokenPipeError, EOFError, OSError):
        pass


def _send_result(conn, payload: dict) -> bool:
    try:
        conn.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _patch_hosts(broker: str) -> None:
    if broker == "iqoption":
        return
    if broker not in HOSTS:
        raise ValueError(f"Corretora não suportada: {broker}")

    import iqoptionapi.api as api_mod
    import iqoptionapi.stable_api as stable_mod

    h = HOSTS[broker]
    base = api_mod.IQOptionAPI

    class PatchedAPI(base):
        def __init__(self, host, username, password, proxies=None):
            super().__init__(host, username, password, proxies)
            self.wss_url = h["ws"]
            self.https_url = h["https"]

        def send_http_request_v2(self, url, method, data=None, params=None, headers=None):
            url = url.replace("auth.iqoption.com", h["auth"]).replace(
                "event.iqoption.com", h["event"]
            )
            return super().send_http_request_v2(
                url, method, data=data, params=params, headers=headers
            )

    stable_mod.IQOptionAPI = PatchedAPI


def _open_assets(client) -> list[str]:
    from iqoptionapi.constants import ACTIVES

    data = client.get_all_init_v2() or {}
    names: list[str] = []
    for kind in ("turbo", "binary"):
        for active in (data.get(kind, {}) or {}).get("actives", {}).values():
            try:
                name = str(active["name"]).split(".")[1]
            except Exception:
                continue
            if active.get("enabled") and not active.get("is_suspended") and name in ACTIVES:
                names.append(name)
    return sorted(set(names))


def _candles(client, asset: str, tf: int, count: int):
    raw = client.get_candles(asset, tf, count, time.time())
    if not raw:
        return None
    return [
        {
            "ts": int(c["from"]),
            "open": float(c["open"]),
            "high": float(c["max"]),
            "low": float(c["min"]),
            "close": float(c["close"]),
            "volume": float(c.get("volume") or 0),
        }
        for c in raw
    ]


def worker_main(conn, broker: str, email: str, password: str, account_type: str) -> None:
    client = None
    try:
        _send_stage(conn, "iniciando worker")

        _send_stage(conn, "configurando corretora")
        _patch_hosts(broker)

        _send_stage(conn, "importando biblioteca IQ Option")
        from iqoptionapi.stable_api import IQ_Option

        _send_stage(conn, "criando cliente")
        client = IQ_Option(email, password)

        _send_stage(conn, "autenticando na corretora")
        ok, reason = client.connect()
        if not ok:
            _send_result(conn, {"ok": False, "error": str(reason)})
            return

        _send_stage(conn, "autenticação aceita")

        _send_stage(conn, "selecionando conta")
        client.change_balance("PRACTICE" if account_type == "demo" else "REAL")

        _send_stage(conn, "consultando saldo")
        info = {
            "balance": float(client.get_balance()),
            "currency": str(client.get_currency() or "USD"),
        }

        _send_stage(conn, "consultando ativos")
        try:
            info["assets"] = _open_assets(client)
        except Exception as exc:  # noqa: BLE001
            # A conexão pode estar autenticada mesmo se a lista de ativos falhar.
            info["assets"] = []
            _send_stage(conn, f"consulta de ativos falhou: {type(exc).__name__}")

        _send_stage(conn, "enviando confirmação de conexão")
        if not _send_result(conn, {"ok": True, **info}):
            return

    except Exception as exc:  # noqa: BLE001
        _send_result(
            conn,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1200:],
            },
        )
        return

    # Loop de comandos da sessão: mantido após o handshake.
    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            break
        except (OSError, BrokenPipeError):
            break

        op = cmd.get("op")
        try:
            if op == "close":
                break

            if op == "candles_many":
                out = {}
                for asset in cmd["assets"]:
                    try:
                        candles = _candles(client, asset, cmd["tf"], cmd["count"])
                        if candles:
                            out[asset] = candles
                    except Exception:  # noqa: BLE001
                        continue
                _send_result(conn, {"ok": True, "data": out})

            elif op == "balance":
                _send_result(conn, {"ok": True, "balance": float(client.get_balance())})

            elif op == "buy":
                minutes = cmd["minutes"]
                ok, order_id = client.buy(
                    float(cmd["amount"]),
                    cmd["asset"],
                    cmd["action"],
                    int(minutes),
                )
                if ok:
                    _send_result(conn, {"ok": True, "order_id": order_id})
                else:
                    _send_result(
                        conn,
                        {"ok": False, "error": str(order_id or "Ordem recusada pela corretora")},
                    )

            elif op == "check":
                closed = client.api.socket_option_closed.get(cmd["order_id"])
                if not closed:
                    _send_result(conn, {"ok": True, "done": False})
                else:
                    msg = closed["msg"]
                    win = msg.get("win")
                    profit = 0.0 if win == "equal" else (
                        -float(msg.get("sum", 0))
                        if win == "loose"
                        else float(msg.get("win_amount", 0)) - float(msg.get("sum", 0))
                    )
                    _send_result(
                        conn,
                        {"ok": True, "done": True, "win": win, "profit": round(profit, 2)},
                    )

            else:
                _send_result(conn, {"ok": False, "error": f"op desconhecida {op}"})

        except Exception as exc:  # noqa: BLE001
            _send_result(
                conn,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-1200:],
                },
            )

    if client is not None:
        try:
            client.close_connect()
        except Exception:  # noqa: BLE001
            pass
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass
