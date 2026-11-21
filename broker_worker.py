
"""Processo isolado por sessão de corretora.
A lib iqoptionapi usa busy-wait e estado global.
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


def _send(conn, data: dict) -> bool:
    try:
        conn.send(data)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _progress(conn, broker: str, stage: str) -> None:
    _send(conn, {
        "type": "progress",
        "broker": broker,
        "stage": stage,
    })


def _patch_hosts(broker: str) -> None:
    if broker not in HOSTS:
        raise ValueError(f"Corretora não suportada: {broker}")

    if broker == "iqoption":
        return

    import iqoptionapi.api as api_mod
    import iqoptionapi.stable_api as stable_mod

    h = HOSTS[broker]
    base = api_mod.IQOptionAPI

    class PatchedAPI(base):
        def __init__(self, host, username, password, proxies=None):
            super().__init__(host, username, password, proxies)
            self.wss_url = h["ws"]
            self.https_url = h["https"]

        def send_http_request_v2(
            self, url, method, data=None, params=None, headers=None
        ):
            url = (
                url.replace("auth.iqoption.com", h["auth"])
                .replace("event.iqoption.com", h["event"])
            )
            return super().send_http_request_v2(
                url,
                method,
                data=data,
                params=params,
                headers=headers,
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

            if (
                active.get("enabled")
                and not active.get("is_suspended")
                and name in ACTIVES
            ):
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


def worker_main(
    conn,
    broker: str,
    email: str,
    password: str,
    account_type: str,
) -> None:
    client = None

    try:
        _progress(conn, broker, "Preparando integração")

        _patch_hosts(broker)

        _progress(conn, broker, "Carregando biblioteca da corretora")

        from iqoptionapi.stable_api import IQ_Option

        _progress(conn, broker, "Inicializando cliente")

        client = IQ_Option(email, password)

        _progress(conn, broker, "Autenticando e estabelecendo conexão")

        ok, reason = client.connect()

        if not ok:
            _send(conn, {
                "ok": False,
                "error": str(reason or "A corretora recusou a conexão"),
            })
            return

        _progress(conn, broker, "Selecionando conta")

        client.change_balance(
            "PRACTICE" if account_type == "demo" else "REAL"
        )

        _progress(conn, broker, "Consultando saldo")

        info = {
            "balance": float(client.get_balance()),
            "currency": str(client.get_currency() or "USD"),
        }

        _progress(conn, broker, "Consultando ativos disponíveis")

        try:
            info["assets"] = _open_assets(client)
        except Exception:
            info["assets"] = []

        _send(conn, {"ok": True, **info})

    except Exception as e:
        _send(conn, {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()[-1500:],
        })
        return

    # ---------------------------------------------------------
    # Operações posteriores à conexão: mantidas como estavam
    # ---------------------------------------------------------

    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            break

        op = cmd.get("op")

        try:
            if op == "close":
                break

            if op == "candles_many":
                out = {}

                for asset in cmd["assets"]:
                    try:
                        c = _candles(
                            client,
                            asset,
                            cmd["tf"],
                            cmd["count"],
                        )
                        if c:
                            out[asset] = c
                    except Exception:
                        continue

                conn.send({"ok": True, "data": out})

            elif op == "balance":
                conn.send({
                    "ok": True,
                    "balance": float(client.get_balance()),
                })

            elif op == "buy":
                minutes = cmd["minutes"]

                ok, order_id = client.buy(
                    float(cmd["amount"]),
                    cmd["asset"],
                    cmd["action"],
                    int(minutes),
                )

                if ok:
                    conn.send({
                        "ok": True,
                        "order_id": order_id,
                    })
                else:
                    conn.send({
                        "ok": False,
                        "error": str(
                            order_id or "Ordem recusada pela corretora"
                        ),
                    })

            elif op == "check":
                closed = client.api.socket_option_closed.get(
                    cmd["order_id"]
                )

                if not closed:
                    conn.send({
                        "ok": True,
                        "done": False,
                    })
                else:
                    m = closed["msg"]
                    win = m.get("win")

                    profit = (
                        0.0
                        if win == "equal"
                        else (
                            -float(m.get("sum", 0))
                            if win == "loose"
                            else (
                                float(m.get("win_amount", 0))
                                - float(m.get("sum", 0))
                            )
                        )
                    )

                    conn.send({
                        "ok": True,
                        "done": True,
                        "win": win,
                        "profit": round(profit, 2),
                    })

            else:
                conn.send({
                    "ok": False,
                    "error": f"op desconhecida {op}",
                })

        except Exception as e:
            _send(conn, {
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc()[-500:],
            })

    if client is not None:
        try:
            client.close_connect()
        except Exception:
            pass

    try:
        conn.close()
    except Exception:
        pass
