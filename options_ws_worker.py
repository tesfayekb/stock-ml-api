"""
Tradier WebSocket → Real-Time 0DTE Options Monitor.

Connects to Tradier's WebSocket streaming API, subscribes to quote/trade events
for active SPX Iron Butterfly legs, evaluates exit conditions in real-time,
and triggers the broker-tradier edge function when thresholds are breached.

Key design decisions:
- Streams OPTION leg quotes directly (bid/ask) for MTM calculation
- Evaluates exit rules on every quote tick (sub-second)
- Triggers exit via HTTP call to broker-tradier edge function
- Logs all ticks + decisions to Supabase for audit trail
- Uses SPY as real-time proxy for SPX (index data is 15-min delayed)
- Auto-reconnects with exponential backoff
- Only runs during market hours (9:30 AM – 4:00 PM ET)
"""

import os
import json
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import websockets
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("options_ws")

# ── Config ──

TRADIER_DATA_TOKEN = os.environ["TRADIER_DATA_TOKEN"]
TRADIER_BASE_URL = "https://api.tradier.com"
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
FUNCTIONS_URL = os.environ.get("SUPABASE_FUNCTIONS_URL", f"{SUPABASE_URL}/functions/v1")

# Exit thresholds (mirror edge function constants)
PROFIT_TARGET_PCT = 50        # Close at 50% of max credit
STOP_LOSS_MULTIPLIER = 2      # Close at 2x credit
TIME_DECAY_EXIT_PCT = 20      # After 2:30 PM, close at 20% remaining
WING_BREACH_BUFFER = 5        # SPX within $5 of wing
HARD_CLOSE_HOUR = 15
HARD_CLOSE_MIN = 45

# Latency tracking
TICK_LOG_INTERVAL = 60         # Log tick stats every 60 seconds
MAX_RECONNECT_DELAY = 60

# ── Supabase client ──
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class PositionState:
    """Tracks the active iron butterfly position for real-time monitoring."""
    
    def __init__(self):
        self.trade_id: Optional[str] = None
        self.entry_credit: float = 0
        self.wing_width: float = 20
        self.atm_strike: float = 0
        self.call_symbol: str = ""      # short call leg
        self.put_symbol: str = ""       # short put leg
        self.long_call_symbol: str = "" # long call (wing)
        self.long_put_symbol: str = ""  # long put (wing)
        self.contracts: int = 2
        self.is_active: bool = False
        
        # Live quotes
        self.leg_bids: dict[str, float] = {}
        self.leg_asks: dict[str, float] = {}
        self.spy_price: float = 0
        self.last_mtm: float = 0
        self.last_evaluated_at: float = 0
        
        # Stats
        self.ticks_received: int = 0
        self.exit_triggered: bool = False
    
    def load_from_trade(self, trade: dict):
        """Load position from active options_trade record."""
        self.trade_id = trade["id"]
        self.entry_credit = abs(trade.get("entry_premium", 0))
        structure = trade.get("structure", {}) or {}
        self.atm_strike = structure.get("atm_strike", 0)
        self.wing_width = structure.get("wing_width", 20)
        self.contracts = structure.get("contracts", 2)
        
        legs = structure.get("legs", [])
        for leg in legs:
            symbol = leg.get("symbol", "")
            side = leg.get("side", "")
            option_type = leg.get("option_type", "")
            
            if side == "sell_to_open" and option_type == "call":
                self.call_symbol = symbol
            elif side == "sell_to_open" and option_type == "put":
                self.put_symbol = symbol
            elif side == "buy_to_open" and option_type == "call":
                self.long_call_symbol = symbol
            elif side == "buy_to_open" and option_type == "put":
                self.long_put_symbol = symbol
        
        self.is_active = True
        logger.info(f"Loaded position: {self.trade_id}, credit={self.entry_credit}, "
                     f"strike={self.atm_strike}, wings={self.wing_width}")
    
    def get_all_symbols(self) -> list[str]:
        """Get all symbols to stream (4 option legs + SPY proxy)."""
        symbols = []
        for s in [self.call_symbol, self.put_symbol, 
                  self.long_call_symbol, self.long_put_symbol]:
            if s:
                symbols.append(s)
        symbols.append("SPY")  # Real-time proxy for SPX
        return symbols
    
    def update_quote(self, symbol: str, bid: float, ask: float):
        """Update live quote for a symbol."""
        self.leg_bids[symbol] = bid
        self.leg_asks[symbol] = ask
        self.ticks_received += 1
    
    def calculate_mtm(self) -> float:
        """
        Calculate mark-to-market debit to close the position.
        
        To close: buy back shorts, sell longs.
        Debit = (ask of short call + ask of short put) - (bid of long call + bid of long put)
        """
        short_call_ask = self.leg_asks.get(self.call_symbol, 0)
        short_put_ask = self.leg_asks.get(self.put_symbol, 0)
        long_call_bid = self.leg_bids.get(self.long_call_symbol, 0)
        long_put_bid = self.leg_bids.get(self.long_put_symbol, 0)
        
        if not all([short_call_ask, short_put_ask]):
            return self.last_mtm  # Not enough data yet
        
        mtm = (short_call_ask + short_put_ask) - (long_call_bid + long_put_bid)
        self.last_mtm = mtm
        return mtm
    
    def evaluate_exit(self) -> Optional[dict]:
        """
        Evaluate all exit conditions. Returns exit trigger info or None.
        
        Priority order (same as edge function):
        1. Profit target (MTM ≤ 50% of credit)
        2. Stop loss (MTM ≥ 2× credit)  
        3. Time decay (after 2:30 PM, MTM ≤ 20%)
        4. Wing breach (SPY proxy within $5/10 of wing)
        5. Hard close (after 3:45 PM)
        """
        if not self.is_active or self.exit_triggered:
            return None
        
        now_et = datetime.now(timezone(timedelta(hours=-4)))
        mtm = self.calculate_mtm()
        
        # 1. Profit target
        profit_threshold = self.entry_credit * (PROFIT_TARGET_PCT / 100)
        if mtm <= profit_threshold and mtm > 0:
            return {"reason": "profit_target", "mtm": mtm, "threshold": profit_threshold}
        
        # 2. Stop loss
        stop_threshold = self.entry_credit * STOP_LOSS_MULTIPLIER
        if mtm >= stop_threshold:
            return {"reason": "stop_loss", "mtm": mtm, "threshold": stop_threshold, "urgency": "critical"}
        
        # 3. Time decay exit
        if now_et.hour >= 14 and now_et.minute >= 30:
            time_decay_threshold = self.entry_credit * (TIME_DECAY_EXIT_PCT / 100)
            if mtm <= time_decay_threshold:
                return {"reason": "time_decay", "mtm": mtm, "threshold": time_decay_threshold}
        
        # 4. Wing breach (SPY proxy: SPY ≈ SPX/10)
        if self.spy_price > 0:
            spx_estimate = self.spy_price * 10
            upper_wing = self.atm_strike + self.wing_width
            lower_wing = self.atm_strike - self.wing_width
            
            if spx_estimate >= (upper_wing - WING_BREACH_BUFFER):
                return {"reason": "wing_breach_upper", "spx_est": spx_estimate, 
                        "wing": upper_wing, "urgency": "high"}
            if spx_estimate <= (lower_wing + WING_BREACH_BUFFER):
                return {"reason": "wing_breach_lower", "spx_est": spx_estimate,
                        "wing": lower_wing, "urgency": "high"}
        
        # 5. Hard close
        if now_et.hour >= HARD_CLOSE_HOUR and now_et.minute >= HARD_CLOSE_MIN:
            return {"reason": "hard_close", "mtm": mtm}
        
        return None


async def create_streaming_session() -> str:
    """Create a Tradier streaming session and return the session ID."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{TRADIER_BASE_URL}/v1/markets/events/session",
            headers={
                "Authorization": f"Bearer {TRADIER_DATA_TOKEN}",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data["stream"]["sessionid"]
        logger.info(f"Created Tradier streaming session: {session_id[:8]}...")
        return session_id


async def trigger_exit(position: PositionState, exit_info: dict):
    """Call the broker-tradier edge function to execute the exit."""
    position.exit_triggered = True
    logger.warning(f"🚨 EXIT TRIGGERED: {exit_info['reason']} — MTM={exit_info.get('mtm', 'N/A')}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{FUNCTIONS_URL}/broker-tradier",
                headers={
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "action": "monitor",  # Triggers the monitoring/exit logic
                    "force_exit": True,
                    "exit_reason": exit_info["reason"],
                    "ws_mtm": exit_info.get("mtm"),
                    "ws_triggered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"✅ Exit executed: {result}")
                
                # Log to Supabase
                sb.table("audit_log").insert({
                    "user_id": "00000000-0000-0000-0000-000000000000",
                    "table_name": "options_trades",
                    "operation": "ws_exit_trigger",
                    "record_id": position.trade_id or "",
                    "new_data": {
                        "exit_info": exit_info,
                        "response_status": resp.status_code,
                        "latency_ms": int((time.time() - position.last_evaluated_at) * 1000),
                    },
                }).execute()
            else:
                logger.error(f"❌ Exit call failed: {resp.status_code} — {resp.text}")
                position.exit_triggered = False  # Allow retry on next tick
                
    except Exception as e:
        logger.error(f"❌ Exit trigger error: {e}")
        position.exit_triggered = False


async def load_active_position() -> Optional[PositionState]:
    """Load the active 0DTE iron butterfly position from options_trades."""
    try:
        result = sb.table("options_trades").select("*").eq(
            "status", "open"
        ).eq("strategy", "iron_butterfly_0dte").eq(
            "is_system", True
        ).order("created_at", desc=True).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            pos = PositionState()
            pos.load_from_trade(result.data[0])
            return pos
        
        logger.info("No active 0DTE position found")
        return None
    except Exception as e:
        logger.error(f"Failed to load position: {e}")
        return None


def is_market_hours() -> bool:
    """Check if within US market hours (9:30 AM – 4:00 PM ET)."""
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    if now_et.weekday() >= 5:  # Weekend
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


async def log_tick_stats(position: PositionState):
    """Periodically log streaming statistics."""
    while True:
        await asyncio.sleep(TICK_LOG_INTERVAL)
        if position.is_active:
            mtm = position.calculate_mtm()
            logger.info(
                f"📊 Ticks: {position.ticks_received} | "
                f"MTM: ${mtm:.2f} | Credit: ${position.entry_credit:.2f} | "
                f"SPY: ${position.spy_price:.2f} | "
                f"SPX est: ${position.spy_price * 10:.0f}"
            )
            
            # Write snapshot to Supabase for admin dashboard
            try:
                sb.table("live_ticks").insert({
                    "ticker": "0DTE_MTM",
                    "price": round(mtm, 2),
                    "size": position.ticks_received,
                    "timestamp": int(time.time() * 1000),
                    "conditions": [f"credit:{position.entry_credit}", 
                                   f"spy:{position.spy_price}"],
                }).execute()
            except Exception as e:
                logger.error(f"Failed to log tick stats: {e}")


async def connect_and_stream():
    """Main WebSocket connection loop with auto-reconnect."""
    reconnect_delay = 1
    
    while True:
        # Wait for market hours
        if not is_market_hours():
            logger.info("Outside market hours. Sleeping 60s...")
            await asyncio.sleep(60)
            continue
        
        # Load active position
        position = await load_active_position()
        if not position:
            logger.info("No active position. Checking again in 30s...")
            await asyncio.sleep(30)
            continue
        
        symbols = position.get_all_symbols()
        if len(symbols) < 2:
            logger.warning("Not enough symbols to stream. Retrying in 30s...")
            await asyncio.sleep(30)
            continue
        
        try:
            # Create streaming session
            session_id = await create_streaming_session()
            
            # Connect to WebSocket
            async with websockets.connect(TRADIER_WS_URL) as ws:
                logger.info(f"Connected to Tradier WebSocket")
                reconnect_delay = 1
                
                # Send subscription payload
                payload = {
                    "symbols": symbols,
                    "sessionid": session_id,
                    "filter": ["quote", "trade"],  # Quote for bid/ask, trade for last price
                    "linebreak": True,
                }
                await ws.send(json.dumps(payload))
                logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")
                
                # Start background stats logger
                stats_task = asyncio.create_task(log_tick_stats(position))
                
                try:
                    async for raw_msg in ws:
                        if not raw_msg.strip():
                            continue
                        
                        try:
                            event = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue
                        
                        ev_type = event.get("type", "")
                        symbol = event.get("symbol", "")
                        
                        if ev_type == "quote":
                            bid = float(event.get("bid", 0))
                            ask = float(event.get("ask", 0))
                            
                            if symbol == "SPY":
                                position.spy_price = (bid + ask) / 2
                            else:
                                position.update_quote(symbol, bid, ask)
                            
                            # Evaluate exit on every quote tick
                            position.last_evaluated_at = time.time()
                            exit_info = position.evaluate_exit()
                            if exit_info:
                                await trigger_exit(position, exit_info)
                                # After exit, reload position
                                await asyncio.sleep(5)
                                new_pos = await load_active_position()
                                if not new_pos:
                                    logger.info("Position closed. Waiting for next trade...")
                                    break
                                position = new_pos
                        
                        elif ev_type == "trade" and symbol == "SPY":
                            position.spy_price = float(event.get("price", 0))
                        
                        # Check if position was closed externally (by cron monitor)
                        if position.ticks_received % 500 == 0 and position.ticks_received > 0:
                            fresh = await load_active_position()
                            if not fresh:
                                logger.info("Position closed externally. Breaking stream.")
                                break
                
                finally:
                    stats_task.cancel()
        
        except websockets.ConnectionClosed as e:
            logger.warning(f"WebSocket closed: {e}. Reconnecting in {reconnect_delay}s...")
        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in {reconnect_delay}s...")
        
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)


if __name__ == "__main__":
    logger.info("Starting Tradier Options WebSocket worker (0DTE Iron Butterfly monitor)...")
    asyncio.run(connect_and_stream())
```

## Testing Locally

```bash
export TRADIER_DATA_TOKEN="your_production_token"
export SUPABASE_URL="https://crfixmrwtzzontcsrdug.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"
export SUPABASE_FUNCTIONS_URL="https://crfixmrwtzzontcsrdug.supabase.co/functions/v1"

python options_ws_worker.py
```

## Deployment Checklist

- [ ] Railway service created with Python runtime
- [ ] All 4 env vars configured  
- [ ] `requirements.txt` includes `websockets httpx supabase`
- [ ] Start command: `python options_ws_worker.py`
- [ ] Health check: watch Railway logs for "Connected to Tradier WebSocket"
- [ ] Verify ticks appear in Supabase `live_ticks` table with ticker `0DTE_MTM`
