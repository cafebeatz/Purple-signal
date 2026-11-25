"""Processo isolado por sessão de corretora (a lib iqoptionapi usa busy-wait e estado global)."""
from __future__ import annotations

import logging
import time
import traceback

# Os registros aparecem nos logs do serviço (por exemplo, no Render).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [broker_worker] %(message)s",
    force=True,
)
logger = logging.getLogger("broker_worker")


def _log_stage(message: str, started: float | None = None) -> None:
    if started is None:
        logger.info("%s", message)
    else:
        logger.info("%s (%.2fs)", message, time.monotonic() - started)


HOSTS = {
    "iqoption": {"ws": "wss://iqoption.com/echo/websocket", "https": "https://iqoption.com/api",
                 "auth": "auth.iqoption.com", "event": "event.iqoption.com"},
    "polarium": {"ws": "wss://ws.trade.polariumbroker.com/echo/websocket", "https": "https://trade.polariumbroker.com/api",
                 "auth": "auth.trade.polariumbroker.com", "event": "event.trade.polariumbroker.com"},
}


def _patch_hosts(broker: str) -> None:
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

        def send_http_request_v2(self, url, method, data=None, params=None, headers=None):
            url = url.replace("auth.iqoption.com", h["auth"]).replace("event.iqoption.com", h["event"])
            return super().send_http_request_v2(url, method, data=data, params=params, headers=headers)

    stable_mod.IQOptionAPI = PatchedAPI


def _open_assets(client) -> list[str]:
    """Retorna ativos digital/binários/turbo habilitados, incluindo mercado aberto e otc."""
    from iqoptionapi.constants import ACTIVES

    data = client.get_all_init_v2() or {}
    active_names = set(ACTIVES.keys())
    names: set[str] = set()
    rejected = 0

    for kind in ("turbo", "binary"):
        actives = (data.get(kind, {}) or {}).get("actives", {}) or {}

        for active in actives.values():
            if not isinstance(active, dict):
                rejected += 1
                continue

            if not active.get("enabled") or active.get("is_suspended"):
                continue

            raw_name = active.get("name")
            if not raw_name:
                rejected += 1
                continue

            # Divide apenas no primeiro ponto e aceita nomes sem prefixo.
            raw_name = str(raw_name).strip()
            candidate = raw_name.split(".", 1)[-1].strip()

            if candidate in active_names:
                names.add(candidate)
            elif raw_name in active_names:
                names.add(raw_name)
            else:
                rejected += 1

    result = sorted(names)
    open_count = sum("otc" not in name.upper() for name in result)
    otc_count = len(result) - open_count
    logger.info(
        "Ativos retornados pela API: total=%d, mercado_aberto=%d, otc=%d, "
        "nomes_nao_reconhecidos=%d",
        len(result), open_count, otc_count, rejected
    )
    return result


def _candles(client, asset: str, tf: int, count: int):
    raw = client.get_candles(asset, tf, count, time.time())
    if not raw:
        return None
    return [{"ts": int(c["from"]), "open": float(c["open"]), "high": float(c["max"]),
             "low": float(c["min"]), "close": float(c["close"]), "volume": float(c.get("volume") or 0)}
            for c in raw]


def worker_main(conn, broker: str, email: str, password: str, account_type: str) -> None:
    stage = "inicialização"
    try:
        _log_stage(f"Worker iniciado; corretora={broker}; conta={account_type}")

        stage = "preparação dos hosts"
        started = time.monotonic()
        _patch_hosts(broker)
        _log_stage("Preparação dos hosts concluída", started)

        stage = "importação da biblioteca IQ Option"
        started = time.monotonic()
        from iqoptionapi.stable_api import IQ_Option
        _log_stage("Biblioteca IQ Option importada", started)

        stage = "criação do cliente"
        started = time.monotonic()
        client = IQ_Option(email, password)
        _log_stage("Cliente IQ Option criado", started)

        stage = "login / client.connect()"
        started = time.monotonic()
        logger.info("Iniciando client.connect()")
        ok, reason = client.connect()
        _log_stage(f"client.connect() terminou; ok={ok}", started)
        if not ok:
            logger.error("Login recusado/falhou: %s", reason)
            conn.send({"ok": False, "error": str(reason)})
            return

        stage = "seleção da conta"
        started = time.monotonic()
        balance_mode = "PRACTICE" if account_type == "demo" else "REAL"
        logger.info("Iniciando change_balance(%s)", balance_mode)
        client.change_balance(balance_mode)
        _log_stage("Troca de conta concluída", started)

        stage = "consulta do saldo"
        started = time.monotonic()
        balance = float(client.get_balance())
        _log_stage("Consulta do saldo concluída", started)

        stage = "consulta da moeda"
        started = time.monotonic()
        currency = str(client.get_currency() or "USD")
        _log_stage("Consulta da moeda concluída", started)

        info = {"balance": balance, "currency": currency}

        stage = "carregamento dos ativos"
        started = time.monotonic()
        try:
            logger.info("Iniciando carregamento dos ativos")
            info["assets"] = _open_assets(client)
            _log_stage(
                "Carregamento dos ativos concluído; "
                f"total={len(info['assets'])}, "
                f"mercado_aberto={sum('OTC' not in a.upper() for a in info['assets'])}, "
                f"otc={sum('OTC' in a.upper() for a in info['assets'])}",
                started
            )
        except Exception:
            info["assets"] = []
            logger.exception("Falha ao carregar ativos; seguindo sem lista de ativos")

        stage = "envio da confirmação ao servidor"
        started = time.monotonic()
        conn.send({"ok": True, **info})
        _log_stage("Confirmação de conexão enviada ao servidor", started)
        logger.info("Inicialização da sessão concluída")

    except Exception as e:  # noqa: BLE001
        logger.exception("Erro na etapa '%s': %s", stage, e)
        try:
            conn.send({"ok": False, "error": f"Falha na etapa '{stage}': {e}",
                       "trace": traceback.format_exc()[-1000:]})
        except Exception:
            logger.exception("Não foi possível enviar o erro ao processo principal")
        return

    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            logger.info("Canal com o processo principal encerrado")
            break
        op = cmd.get("op")
        try:
            if op == "close":
                logger.info("Comando de encerramento recebido")
                break
            if op == "candles_many":
                out = {}
                for asset in cmd["assets"]:
                    try:
                        c = _candles(client, asset, cmd["tf"], cmd["count"])
                        if c:
                            out[asset] = c
                    except Exception:
                        logger.exception("Falha ao buscar candles do ativo %s", asset)
                        continue
                conn.send({"ok": True, "data": out})
            elif op == "balance":
                conn.send({"ok": True, "balance": float(client.get_balance())})
            elif op == "buy":
                minutes = cmd["minutes"]
                ok, order_id = client.buy(float(cmd["amount"]), cmd["asset"], cmd["action"], int(minutes))
                if ok:
                    conn.send({"ok": True, "order_id": order_id})
                else:
                    conn.send({"ok": False, "error": str(order_id or "Ordem recusada pela corretora")})
            elif op == "check":
                closed = client.api.socket_option_closed.get(cmd["order_id"])
                if not closed:
                    conn.send({"ok": True, "done": False})
                else:
                    m = closed["msg"]
                    win = m.get("win")
                    profit = 0.0 if win == "equal" else (
                        -float(m.get("sum", 0)) if win == "loose" else float(m.get("win_amount", 0)) - float(m.get("sum", 0)))
                    conn.send({"ok": True, "done": True, "win": win, "profit": round(profit, 2)})
            else:
                conn.send({"ok": False, "error": f"op desconhecida {op}"})
        except Exception as e:  # noqa: BLE001
            logger.exception("Erro ao processar comando '%s': %s", op, e)
            conn.send({"ok": False, "error": f"{e}", "trace": traceback.format_exc()[-500:]})

    try:
        client.close_connect()
        logger.info("Conexão com a corretora encerrada")
    except Exception:
        logger.exception("Erro ao encerrar conexão com a corretora")
