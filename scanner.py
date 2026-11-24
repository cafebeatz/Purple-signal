import csv
import logging
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Iterator, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scanner")


# Fonte oficial da lista de ativos negociados na B3.
UNIVERSE_URL = "https://arquivos.b3.com.br/api/download/requestname?fileName=InstrumentsConsolidated&language=pt-BR"
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 15
OUTPUT_FILE = "ranking.csv"


@dataclass
class AssetScore:
    symbol: str
    price: float
    change_percent: float
    volume: int
    score: float


def fetch_universe(session: requests.Session) -> List[str]:
    """Retorna a lista completa de tickers, sem duplicados e em ordem estável."""
    response = session.get(UNIVERSE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    symbols = set()
    for line in response.text.splitlines()[1:]:
        parts = line.split(";")
        if not parts:
            continue
        ticker = parts[0].strip().upper()
        if ticker and ticker.isalnum():
            symbols.add(f"{ticker}.SA")
    ordered = sorted(symbols)
    log.info("Universo carregado: %d ativos", len(ordered))
    return ordered


def chunked(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def get_with_retry(session: requests.Session, url: str, params: dict) -> Optional[dict]:
    """Faz a requisição com retentativas e backoff exponencial."""
    delay = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                raise requests.HTTPError("limite de requisições atingido")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Tentativa %d/%d falhou: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def score_quote(quote: dict) -> Optional[AssetScore]:
    """Converte uma cotação bruta em pontuação. Ignora ativos sem preço válido."""
    price = quote.get("regularMarketPrice")
    change = quote.get("regularMarketChangePercent")
    volume = quote.get("regularMarketVolume") or 0
    if price is None or change is None or price <= 0:
        return None
    # Pontuação simples: variação ponderada pelo logaritmo do volume.
    import math
    score = float(change) * math.log10(volume + 10)
    return AssetScore(
        symbol=quote.get("symbol", ""),
        price=float(price),
        change_percent=float(change),
        volume=int(volume),
        score=round(score, 4),
    )


def scan(symbols: Iterable[str], session: requests.Session,
         progress: Optional[Callable[[int, int], None]] = None) -> List[AssetScore]:
    symbols = list(symbols)
    results: List[AssetScore] = []
    failed_batches = 0
    total = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
    for index, batch in enumerate(chunked(symbols, BATCH_SIZE), start=1):
        payload = get_with_retry(session, QUOTE_URL, {"symbols": ",".join(batch)})
        if payload is None:
            failed_batches += 1
            log.error("Lote %d/%d descartado após retentativas", index, total)
            continue
        quotes = payload.get("quoteResponse", {}).get("result", [])
        for quote in quotes:
            scored = score_quote(quote)
            if scored:
                results.append(scored)
        if progress:
            progress(index, total)
    log.info("Lotes com falha: %d de %d", failed_batches, total)
    return results


def save_ranking(results: List[AssetScore], path: str = OUTPUT_FILE) -> None:
    ordered = sorted(results, key=lambda r: r.score, reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(ordered[0]).keys()) if ordered else ["symbol"])
        writer.writeheader()
        for item in ordered:
            writer.writerow(asdict(item))
    log.info("Ranking salvo em %s com %d ativos", path, len(ordered))


def main() -> None:
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 scanner/2.0"})
        universe = fetch_universe(session)
        results = scan(universe, session,
                       progress=lambda done, total: log.info("Progresso: %d/%d lotes", done, total))
    save_ranking(results)


if __name__ == "__main__":
    main()
