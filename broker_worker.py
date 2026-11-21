
"""
Processo isolado por sessão de corretora.

Suporta:
- IQ Option com IQAir (padrão)
- IQ Option com iqoptionapi (alternativa)
- Polarium utilizando a camada compatível da iqoptionapi

A escolha da biblioteca da IQ Option é feita pela variável:
IQOPTION_LIBRARY=iqair
ou
IQOPTION_LIBRARY=iqoptionapi
"""

from __future__ import annotations

import os
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


def _selected_iq_library() -> str:
    """
    IQAir é o padrão.
    Para voltar à biblioteca antiga, configure:
    IQOPTION_LIBRARY=iqoptionapi
    """
    selected = os.getenv("IQOPTION_LIBRARY", "iqair").strip().lower()

    aliases = {
        "new": "iqair",
        "legacy": "iqoptionapi",
        "old": "iqoptionapi",
    }

    selected = aliases.get(selected, selected)

    if selected not in ("iqair", "iqoptionapi"):
        raise ValueError(
            "IQOPTION_LIBRARY inválida. "
            "Use 'iqair' ou 'iqoptionapi'."
        )

    return selected


def _patch_hosts(broker: str) -> None:
    """
    Redireciona a biblioteca antiga para os endereços da Polarium.
    Não altera os endereços da IQ Option.
    """
    if broker == "iqoption":
        return

    if broker != "polarium":
        raise ValueError(f"Corretora desconhecida: {broker}")

    import iqoptionapi.api as api_mod
    import iqoptionapi.stable_api as stable_mod

    hosts = HOSTS[broker]
    base_api = api_mod.IQOptionAPI

    class PatchedAPI(base_api):

        def __init__(
            self,
            host,
            username,
            password,
            proxies=None
        ):
            super().__init__(
                host,
                username,
                password,
                proxies
            )

            self.wss_url = hosts["ws"]
            self.https_url = hosts["https"]

        def send_http_request_v2(
            self,
            url,
            method,
            data=None,
            params=None,
            headers=None
        ):
            url = (
                url
                .replace(
                    "auth.iqoption.com",
                    hosts["auth"]
                )
                .replace(
                    "event.iqoption.com",
                    hosts["event"]
                )
            )

            return super().send_http_request_v2(
                url,
                method,
                data=data,
                params=params,
                headers=headers
            )

    stable_mod.IQOptionAPI = PatchedAPI


def _create_client(
    broker: str,
    email: str,
    password: str
):
    """
    Cria o cliente correspondente à corretora.

    IQ Option:
        IQAir ou iqoptionapi, selecionável por variável de ambiente.

    Polarium:
        Utiliza a biblioteca antiga com hosts adaptados.
    """

    if broker == "iqoption":

        library = _selected_iq_library()

        if library == "iqair":
            from iqair.client import IQOptionClient

            client = IQOptionClient(
                email,
                password
            )

            return client, "iqair"

        from iqoptionapi.stable_api import IQ_Option

        client = IQ_Option(
            email,
            password
        )

        return client, "iqoptionapi"

    if broker == "polarium":

        _patch_hosts("polarium")

        from iqoptionapi.stable_api import IQ_Option

        client = IQ_Option(
            email,
            password
        )

        return client, "iqoptionapi"

    raise ValueError(
        f"Corretora não suportada: {broker}"
    )


def _open_assets(client, library: str) -> list[str]:
    """
    Retorna os ativos disponíveis.

    Na IQAir, utiliza get_asset_metadata().
    Na biblioteca antiga, utiliza get_all_init_v2().
    """

    names = []

    if library == "iqair":

        import iqair.constants as constants

        metadata = client.get_asset_metadata() or {}

        for category in ("turbo", "binary"):

            assets = metadata.get(category, {}) or {}

            for ticker, info in assets.items():

                try:
                    active_id = int(info["id"])
                except (KeyError, TypeError, ValueError):
                    continue

                # A função get_candles da IQAir utiliza
                # o mapa ACTIVES internamente.
                constants.ACTIVES[ticker] = active_id

                if info.get("is_open"):
                    names.append(str(ticker))

        return sorted(set(names))

    from iqoptionapi.constants import ACTIVES

    data = client.get_all_init_v2() or {}

    for category in ("turbo", "binary"):

        actives = (
            data.get(category, {}) or {}
        ).get("actives", {}) or {}

        for active in actives.values():

            try:
                name = (
                    active.get("ticker")
                    or str(active["name"]).split(".")[-1]
                )
            except (KeyError, TypeError, IndexError):
                continue

            if (
                active.get("enabled")
                and not active.get("is_suspended")
                and name in ACTIVES
            ):
                names.append(name)

    return sorted(set(names))


def _candles(
    client,
    asset: str,
    timeframe: int,
    count: int
):
    """
    Normaliza candles das duas bibliotecas
    para o formato utilizado pelo servidor.
    """

    raw = client.get_candles(
        asset,
        timeframe,
        count,
        time.time()
    )

    if not raw:
        return None

    result = []

    for candle in raw:

        try:
            result.append({
                "ts": int(candle["from"]),
                "open": float(candle["open"]),
                "high": float(
                    candle.get("max", candle.get("high"))
                ),
                "low": float(
                    candle.get("min", candle.get("low"))
                ),
                "close": float(candle["close"]),
                "volume": float(
                    candle.get("volume") or 0
                ),
            })

        except (KeyError, TypeError, ValueError):
            continue

    return result or None


def _find_iqair_position(client, order_id):
    """
    Procura uma operação da IQAir pelo ID.

    Primeiro consulta posições abertas.
    Depois consulta o histórico de operações encerradas.
    """

    order_id_text = str(order_id)

    def matches(position):

        if str(position.get("external_id")) == order_id_text:
            return True

        raw_event = position.get("raw_event") or {}

        for value in raw_event.values():

            if not isinstance(value, dict):
                continue

            order_ids = value.get("order_ids", [])

            if any(
                str(item) == order_id_text
                for item in order_ids
            ):
                return True

        return False

    # Primeiro: posições abertas.
    try:
        ok, data = client.get_positions(
            "turbo-option",
            max_wait_sec=3
        )

        if ok and data:

            for position in data.get("positions", []):

                if matches(position):
                    return position, True

    except Exception:
        pass

    # Segundo: histórico de posições encerradas.
    try:
        ok, data = client.get_position_history_v2(
            "turbo-option",
            limit=100,
            offset=0,
            max_wait_sec=3
        )

        if ok and data:

            for position in data.get("positions", []):

                if matches(position):
                    return position, False

    except Exception:
        pass

    return None, False


def _check_iqair(client, order_id):
    """
    Consulta o resultado da operação usando
    posições e histórico da IQAir.
    """

    position, is_open = _find_iqair_position(
        client,
        order_id
    )

    if position is None:
        return {
            "ok": True,
            "done": False
        }

    if is_open or position.get("status") == "open":
        return {
            "ok": True,
            "done": False
        }

    pnl_data = client.get_pnl(position) or {}

    raw_profit = pnl_data.get("pnl_net")

    if raw_profit is None:
        raw_profit = pnl_data.get("pnl")

    if raw_profit is None:
        raw_profit = pnl_data.get("profit")

    if raw_profit is None:
        return {
            "ok": False,
            "error": "Resultado encontrado sem valor de lucro."
        }

    profit = round(float(raw_profit), 2)

    if profit > 0:
        win = "win"
    elif profit < 0:
        win = "loose"
    else:
        win = "equal"

    return {
        "ok": True,
        "done": True,
        "win": win,
        "profit": profit
    }


def _close_client(client, library: str) -> None:
    """
    Fecha a conexão de acordo com a biblioteca utilizada.
    """

    if client is None:
        return

    try:

        if library == "iqair":
            client.api.close()
        else:
            client.close_connect()

    except Exception:
        pass


def worker_main(
    conn,
    broker: str,
    email: str,
    password: str,
    account_type: str
) -> None:

    client = None
    library = None

    try:

        client, library = _create_client(
            broker,
            email,
            password
        )

        connected, reason = client.connect()

        if not connected:

            conn.send({
                "ok": False,
                "error": str(reason)
            })

            return

        # Mantém a escolha de conta solicitada pelo servidor.
        selected_balance = (
            "PRACTICE"
            if account_type == "demo"
            else "REAL"
        )

        client.change_balance(selected_balance)

        # A IQAir recomenda aguardar a troca de saldo.
        if library == "iqair":
            time.sleep(1)

        info = {
            "balance": float(client.get_balance()),
            "currency": str(
                client.get_currency() or "USD"
            )
        }

        try:
            info["assets"] = _open_assets(
                client,
                library
            )
        except Exception:
            info["assets"] = []

        conn.send({
            "ok": True,
            **info
        })

    except Exception as error:

        try:
            conn.send({
                "ok": False,
                "error": str(error),
                "trace": traceback.format_exc()[-1000:]
            })
        except Exception:
            pass

        _close_client(client, library)
        return

    # Loop de comunicação com broker.py.
    while True:

        try:
            command = conn.recv()

        except EOFError:
            break

        operation = command.get("op")

        try:

            if operation == "close":
                break

            elif operation == "candles_many":

                output = {}

                for asset in command["assets"]:

                    try:

                        candles = _candles(
                            client,
                            asset,
                            int(command["tf"]),
                            int(command["count"])
                        )

                        if candles:
                            output[asset] = candles

                    except Exception:
                        continue

                conn.send({
                    "ok": True,
                    "data": output
                })

            elif operation == "balance":

                conn.send({
                    "ok": True,
                    "balance": float(
                        client.get_balance()
                    )
                })

            elif operation == "buy":

                minutes = int(command["minutes"])

                ok, order_id = client.buy(
                    float(command["amount"]),
                    command["asset"],
                    command["action"],
                    minutes
                )

                if ok:

                    conn.send({
                        "ok": True,
                        "order_id": order_id
                    })

                else:

                    conn.send({
                        "ok": False,
                        "error": str(
                            order_id
                            or "Ordem recusada pela corretora"
                        )
                    })

            elif operation == "check":

                order_id = command["order_id"]

                if library == "iqair":

                    result = _check_iqair(
                        client,
                        order_id
                    )

                    conn.send(result)

                else:

                    closed = (
                        client.api.socket_option_closed.get(
                            order_id
                        )
                    )

                    if not closed:

                        conn.send({
                            "ok": True,
                            "done": False
                        })

                    else:

                        message = closed["msg"]
                        win = message.get("win")

                        profit = (
                            0.0
                            if win == "equal"
                            else (
                                -float(message.get("sum", 0))
                                if win == "loose"
                                else (
                                    float(
                                        message.get(
                                            "win_amount",
                                            0
                                        )
                                    )
                                    - float(
                                        message.get("sum", 0)
                                    )
                                )
                            )
                        )

                        conn.send({
                            "ok": True,
                            "done": True,
                            "win": win,
                            "profit": round(profit, 2)
                        })

            else:

                conn.send({
                    "ok": False,
                    "error": f"op desconhecida: {operation}"
                })

        except Exception as error:

            try:
                conn.send({
                    "ok": False,
                    "error": str(error),
                    "trace": traceback.format_exc()[-1000:]
                })
            except Exception:
                break

    _close_client(client, library)

    try:
        conn.close()
    except Exception:
        pass
