"""
Polygon.io WebSocket → Supabase live_ticks worker.

Connects to Polygon's Stocks WebSocket, subscribes to trade events,
and upserts price ticks to Supabase for Realtime broadcast to the frontend.

Key design decisions:
- Batches writes every 1 second to avoid overwhelming Supabase
- Deduplicates: only keeps latest tick per ticker per batch
- Auto-reconnects on disconnect with exponential backoff
- Dynamically updates subscriptions from scan_candidates if WATCHED_TICKERS=DYNAMIC
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from collections import defaultdict

import websockets
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ws_worker")

# ── Config ──

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

BATCH_INTERVAL_SECONDS = 1.0
MAX_BATCH_SIZE = 200
TICKER_REFRESH_INTERVAL = 300
MAX_RECONNECT_DELAY = 60

# ── Supabase client ──

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── Tick buffer ──
tick_buffer: dict[str, dict] = {}
buffer_lock = asyncio.Lock()


def get_watched_tickers() -> list[str]:
    """Get tickers to watch — either static list or dynamic from scan_candidates."""
    raw = os.environ.get("WATCHED_TICKERS", "DYNAMIC")
    if raw != "DYNAMIC":
        return [t.strip().upper() for t in raw.split(",") if t.strip()]

    try:
        candidates = (
            sb.table("scan_candidates")
            .select("ticker")
            .eq("is_active", True)
            .execute()
        )
        signals = (
            sb.table("signal_edges")
            .select("ticker")
            .eq("status", "open")
            .execute()
        )
        tickers = set()
        for row in (candidates.data or []):
            tickers.add(row["ticker"])
        for row in (signals.data or []):
            tickers.add(row["ticker"])

        tickers.update(["SPY", "QQQ", "IWM", "DIA"])

        result = sorted(tickers)
        logger.info(f"Dynamic ticker list: {len(result)} tickers")
        return result
    except Exception as e:
        logger.error(f"Failed to fetch dynamic tickers: {e}")
        return ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL"]


async def flush_buffer():
    """Flush accumulated ticks to Supabase."""
    async with buffer_lock:
        if not tick_buffer:
            return
        rows = list(tick_buffer.values())
        tick_buffer.clear()

    if not rows:
        return

    for i in range(0, len(rows), MAX_BATCH_SIZE):
        batch = rows[i : i + MAX_BATCH_SIZE]
        try:
            sb.table("live_ticks").insert(batch).execute()
            logger.debug(f"Flushed {len(batch)} ticks to Supabase")
        except Exception as e:
            logger.error(f"Failed to flush ticks: {e}")


async def buffer_flusher():
    """Background task that flushes the tick buffer periodically."""
    while True:
        await asyncio.sleep(BATCH_INTERVAL_SECONDS)
        await flush_buffer()


async def ticker_refresher(ws, current_tickers: set):
    """Periodically refresh the watched tickers list and update subscriptions."""
    while True:
        await asyncio.sleep(TICKER_REFRESH_INTERVAL)
        try:
            new_tickers = set(get_watched_tickers())
            to_add = new_tickers - current_tickers
            to_remove = current_tickers - new_tickers

            if to_add:
                sub_msg = json.dumps({
                    "action": "subscribe",
                    "params": ",".join(f"T.{t}" for t in to_add)
                })
                await ws.send(sub_msg)
                logger.info(f"Subscribed to {len(to_add)} new tickers: {sorted(to_add)[:10]}...")
                current_tickers.update(to_add)

            if to_remove:
                unsub_msg = json.dumps({
                    "action": "unsubscribe",
                    "params": ",".join(f"T.{t}" for t in to_remove)
                })
                await ws.send(unsub_msg)
                logger.info(f"Unsubscribed from {len(to_remove)} tickers")
                current_tickers -= to_remove

        except Exception as e:
            logger.error(f"Ticker refresh error: {e}")


async def connect_and_stream():
    """Main WebSocket connection loop with auto-reconnect."""
    reconnect_delay = 1

    while True:
        try:
            async with websockets.connect(POLYGON_WS_URL) as ws:
                logger.info("Connected to Polygon WebSocket")
                reconnect_delay = 1

                msg = await ws.recv()
                data = json.loads(msg)
                logger.info(f"Connection message: {data}")

                auth_msg = json.dumps({"action": "auth", "params": POLYGON_API_KEY})
                await ws.send(auth_msg)
                auth_resp = await ws.recv()
                auth_data = json.loads(auth_resp)
                logger.info(f"Auth response: {auth_data}")

                if not any(m.get("status") == "auth_success" for m in auth_data if isinstance(m, dict)):
                    logger.error("Authentication failed!")
                    await asyncio.sleep(10)
                    continue

                tickers = get_watched_tickers()
                current_tickers = set(tickers)
                sub_params = ",".join(f"T.{t}" for t in tickers)
                sub_msg = json.dumps({"action": "subscribe", "params": sub_params})
                await ws.send(sub_msg)
                logger.info(f"Subscribed to {len(tickers)} tickers")

                flusher_task = asyncio.create_task(buffer_flusher())
                refresher_task = asyncio.create_task(ticker_refresher(ws, current_tickers))

                try:
                    async for raw_msg in ws:
                        events = json.loads(raw_msg)
                        if not isinstance(events, list):
                            events = [events]

                        for event in events:
                            ev_type = event.get("ev")
                            if ev_type != "T":
                                continue

                            ticker = event.get("sym", "")
                            price = event.get("p", 0)
                            size = event.get("s", 0)
                            timestamp = event.get("t", 0)
                            conditions = event.get("c", [])

                            if not ticker or not price:
                                continue

                            async with buffer_lock:
                                tick_buffer[ticker] = {
                                    "ticker": ticker,
                                    "price": float(price),
                                    "size": int(size),
                                    "timestamp": int(timestamp),
                                    "conditions": [str(c) for c in conditions] if conditions else [],
                                }

                finally:
                    flusher_task.cancel()
                    refresher_task.cancel()
                    await flush_buffer()

        except websockets.ConnectionClosed as e:
            logger.warning(f"WebSocket closed: {e}. Reconnecting in {reconnect_delay}s...")
        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in {reconnect_delay}s...")

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)


if __name__ == "__main__":
    logger.info("Starting Polygon WebSocket worker...")
    asyncio.run(connect_and_stream())
