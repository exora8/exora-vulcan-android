import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import asyncio
import math

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"
REFRESH_INTERVAL_SECONDS = 0.5 # Interval refresh data untuk live trading

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
market_state = {} 
is_ai_thinking = False # Global flag to prevent multiple decision making processes concurrently
is_autopilot_in_cooldown = {} 
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
    print(bright + color + text + Style.RESET_ALL, end=end)

def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        safe_title = title.replace('"', "'"); safe_content = content.replace('"', "'")
        command = f'termux-notification --title "{safe_title}" --content "{safe_content}"'
        os.system(command)
    except Exception as e: print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("     Strategic AI Analyst (Full Vulcan's Logic)   ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYan, Style.BRIGHT)
    print_colored("Setiap trade sekarang direkam dengan data forensik yang lengkap.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk masuk ke Live Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, bt_months)", Fore.GREEN) # Updated help text
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 0.20,
        "fee_pct": 0.1,
        "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30,
        "trailing_tp_gap_pct": 0.05,
        "caution_level": 0.5,             
        "backtest_duration_months": 3,   # NEW: Default backtest duration in months
        "watched_pairs": {}
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else:
        current_settings = default_settings; save_settings()

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
        except json.JSONDecodeError:
            print_colored(f"Warning: Failed to load trades from '{TRADES_FILE}'. File might be corrupted. Starting with empty trades.", Fore.YELLOW)
            autopilot_trades = [] # Reset if corrupted
    # Ensure new fields for trailing TP checkpoint are initialized if loading old trades
    for trade in autopilot_trades:
        if 'current_tp_checkpoint_level' not in trade:
            trade['current_tp_checkpoint_level'] = 0.0

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades):
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        trade_type = trade.get('type', 'LONG'); type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0); is_profit = pl_percent > current_settings.get('fee_pct', 0.1)
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
            run_up = trade.get('run_up_percent', pl_percent)
            print_colored(f"  Profit Tertinggi (Run-up): {run_up:.2f}%", Fore.YELLOW)
            if 'entry_snapshot' in trade and not is_profit:
                snapshot = trade['entry_snapshot']
                print_colored(f"  Pelajaran (Snapshot):", Fore.MAGENTA)
                print_colored(f"    Bias: {snapshot.get('bias', 'N/A')}", Fore.MAGENTA)
                
                if 'ema50' in snapshot and 'ema100' in snapshot:
                    ema50_val = snapshot.get('ema50', 0)
                    ema100_val = snapshot.get('ema100', 0)
                    spread_val = ema50_val - ema100_val
                    print_colored(f"    EMA50: {ema50_val:.4f} | EMA100: {ema100_val:.4f} | Spread: {spread_val:.6f}", Fore.MAGENTA)
                
                print_colored(f"    Prev Candle: Close {snapshot.get('prev_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_prev'):.4f}", Fore.MAGENTA)
                print_colored(f"    Current Candle: Close {snapshot.get('current_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_current'):.4f}", Fore.MAGENTA)
                
                if 'pre_entry_candle_solidity' in snapshot and 'pre_entry_candle_direction' in snapshot:
                    solidity_str = [f"{s:.2f}" for s in snapshot['pre_entry_candle_solidity']]
                    print_colored(f"    3 Prev Candles Solidity: {solidity_str}", Fore.MAGENTA)
                    print_colored(f"    3 Prev Candles Direction: {snapshot['pre_entry_candle_direction']}", Fore.MAGENTA)
        print()

# --- FUNGSI API (BYBIT) ---
# NEW: Function to fetch historical candles in chunks for backtesting
def fetch_historical_bybit_candles(instId, timeframe, start_ms, end_ms, limit=1000):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60')
    bybit_symbol = instId.replace('-', '')
    
    all_candles = []
    current_end_ms = end_ms # Start fetching from the end_ms backwards
    
    # Bybit API fetches newest to oldest, so we iterate backwards in time
    # until we reach the start_ms
    while current_end_ms > start_ms:
        try:
            url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}&endTime={current_end_ms}"
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()

            if data.get("retCode") == 0 and 'list' in data.get('result', {}):
                candle_list_raw = data['result']['list']
                
                if not candle_list_raw:
                    break # No more data to fetch

                # Convert raw data to our internal candle format
                # Bybit returns newest first, so we reverse to get oldest first for backtesting
                current_chunk = [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list_raw][::-1]
                
                # Filter out candles before start_ms
                current_chunk = [c for c in current_chunk if c['time'] >= start_ms]
                
                all_candles.extend(current_chunk)
                
                # Update current_end_ms to the timestamp of the oldest candle in this chunk - 1ms
                # for the next iteration to avoid overlap
                current_end_ms = current_chunk[0]['time'] - 1
                
                # Add a small delay to respect API rate limits
                time.sleep(0.1) 

            else:
                print_colored(f"Error fetching historical data for {instId}: {data.get('retMsg', 'Unknown error')}", Fore.RED)
                break
        except requests.exceptions.RequestException as e:
            print_colored(f"Network error fetching historical data for {instId}: {e}", Fore.RED)
            break
        except Exception as e:
            print_colored(f"An unexpected error occurred while fetching historical data for {instId}: {e}", Fore.RED)
            break

    # Sort all_candles by timestamp to ensure proper chronological order
    all_candles.sort(key=lambda x: x['time'])
    
    # Ensure enough candles for full analysis (EMA100 + 3)
    if len(all_candles) < 100 + 3:
        print_colored(f"Warning: Not enough historical data for {instId} to perform full analysis ({len(all_candles)} candles fetched). Minimum {100+3} required.", Fore.YELLOW)
        return []
    
    return all_candles

# Original fetch_bybit_candle_data for LIVE data
def fetch_bybit_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            # We need enough data for EMA100 and at least 4 candles for trigger + 3 pre-entry snapshot
            if len(candle_list) < 100 + 3: 
                return None 
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception: return None

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# --- OTAK LOCAL AI ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair): self.settings = settings; self.past_trades = past_trades_for_pair
    
    def calculate_ema(self, data, period):
        if len(data) < period: return []
        closes = [d['close'] for d in data]
        ema_values = []
        initial_sma = sum(closes[:period]) / period
        ema_values.append(initial_sma)
        multiplier = 2 / (period + 1)
        for i in range(period, len(closes)):
            ema = (closes[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        return ema_values
    
    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        if full_range == 0: return 1.0
        return body / full_range

    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100 + 3: # Need 100 for EMA100 and 3 more for pre-entry snapshot + current/prev candle.
            return None

        ema9_series = self.calculate_ema(candle_data, 9)
        ema50_series = self.calculate_ema(candle_data, 50)
        ema100_series = self.calculate_ema(candle_data, 100)

        if len(ema9_series) < 2 or len(ema50_series) < 1 or len(ema100_series) < 1:
            return None

        analysis = {
            "ema9_current": ema9_series[-1],
            "ema9_prev": ema9_series[-2], 
            "ema50": ema50_series[-1],
            "ema100": ema100_series[-1],
            "current_candle_close": candle_data[-1]['close'],
            "prev_candle_close": candle_data[-2]['close']
        }
        
        bias = "RANGING";
        if analysis["ema50"] > analysis["ema100"]: bias = "BULLISH"
        elif analysis["ema50"] < analysis["ema100"]: bias = "BEARISH"
        analysis["bias"] = bias; 

        pre_entry_candles = candle_data[-4:-1] 
        pre_entry_solidity = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        pre_entry_direction = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        
        analysis["pre_entry_candle_solidity"] = pre_entry_solidity
        analysis["pre_entry_candle_direction"] = pre_entry_direction

        return analysis
    
    def check_for_repeated_mistake(self, current_analysis, trade_type, instrument_id):
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        
        if not losing_trades: return False

        current_bias = current_analysis['bias']
        current_prev_close = current_analysis['prev_candle_close']
        current_curr_close = current_analysis['current_candle_close']
        current_ema9_prev = current_analysis['ema9_prev']
        current_ema9_current = current_analysis['ema9_current']
        current_pre_solidity = current_analysis.get('pre_entry_candle_solidity', [])
        current_pre_direction = current_analysis.get('pre_entry_candle_direction', [])
        current_ema50 = current_analysis['ema50'] 
        current_ema100 = current_analysis['ema100'] 

        caution_level = self.settings.get("caution_level", 0.5) 

        BASE_SOL_TOL = 0.05      
        MAX_ADD_SOL_TOL = 0.20   
        
        BASE_EMA_TOL = 0.00005   
        MAX_ADD_EMA_TOL = 0.0005 

        actual_solidity_tolerance = BASE_SOL_TOL + (caution_level * MAX_ADD_SOL_TOL)
        actual_ema_spread_tolerance = BASE_EMA_TOL + (caution_level * MAX_ADD_EMA_TOL)

        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            required_keys = ['bias', 'prev_candle_close', 'current_candle_close', 
                             'ema9_prev', 'ema9_current', 'pre_entry_candle_solidity', 
                             'pre_entry_candle_direction', 'ema50', 'ema100'] 
            if not past_snapshot or not all(key in past_snapshot for key in required_keys):
                continue

            if current_bias != past_snapshot['bias']:
                continue

            if current_bias != "RANGING":
                current_spread = abs(current_ema50 - current_ema100)
                past_ema50 = past_snapshot.get('ema50', 0)
                past_ema100 = past_snapshot.get('ema100', 0)
                past_spread = abs(past_ema50 - past_ema100)
                
                if abs(current_spread - past_spread) > actual_ema_spread_tolerance:
                    continue
            else: 
                pass 

            past_prev_close = past_snapshot['prev_candle_close']
            past_curr_close = past_snapshot['current_candle_close']
            past_ema9_prev = past_snapshot['ema9_prev']
            past_ema9_current = past_snapshot['ema9_current']

            ema9_cross_match = False
            if trade_type == 'LONG' and past_snapshot['bias'] == 'BULLISH':
                current_long_cross = (current_prev_close <= current_ema9_prev) and (current_curr_close > current_ema9_current)
                past_long_cross = (past_prev_close <= past_ema9_prev) and (past_curr_close > past_ema9_current)
                if current_long_cross and past_long_cross:
                    ema9_cross_match = True
            elif trade_type == 'SHORT' and past_snapshot['bias'] == 'BEARISH':
                current_short_cross = (current_prev_close >= current_ema9_prev) and (current_curr_close < current_ema9_current)
                past_short_cross = (past_prev_close >= past_ema9_prev) and (past_curr_close < past_ema9_current)
                if current_short_cross and past_short_cross:
                    ema9_cross_match = True
            
            if not ema9_cross_match:
                continue
            
            past_pre_direction = past_snapshot['pre_entry_candle_direction']
            if len(current_pre_direction) < 3 or len(past_pre_direction) < 3:
                continue

            last_direction_matches = (current_pre_direction[-1] == past_pre_direction[-1])
            matching_directions_count = sum(1 for x, y in zip(current_pre_direction, past_pre_direction) if x == y)

            if not (last_direction_matches and matching_directions_count >= 2):
                continue

            past_pre_solidity = past_snapshot['pre_entry_candle_solidity']
            if len(current_pre_solidity) < 3 or len(past_pre_solidity) < 3:
                continue

            matching_solidity_count = 0
            for i in range(len(current_pre_solidity)):
                if abs(current_pre_solidity[i] - past_pre_solidity[i]) <= actual_solidity_tolerance:
                    matching_solidity_count += 1
            
            if matching_solidity_count < 2:
                continue 

            return True 

        return False

    def get_decision(self, candle_data, open_position, instrument_id):
        analysis = self.get_market_analysis(candle_data)
        
        if not analysis: 
            return {"action": "HOLD", "reason": "Data tidak cukup atau analisis tidak valid."}
        
        if open_position:
            return {"action": "HOLD", "reason": "Memantau posisi terbuka..."}
        
        potential_trade_type = None
        if analysis['bias'] == 'BULLISH':
            if analysis['prev_candle_close'] <= analysis['ema9_prev'] and \
               analysis['current_candle_close'] > analysis['ema9_current']:
                potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH':
            if analysis['prev_candle_close'] >= analysis['ema9_prev'] and \
               analysis['current_candle_close'] < analysis['ema9_current']:
                potential_trade_type = 'SHORT'
        
        if potential_trade_type:
            if self.check_for_repeated_mistake(analysis, potential_trade_type, instrument_id):
                return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan {potential_trade_type} berdasarkan riwayat loss."}
            
            if potential_trade_type == 'LONG':
                return {"action": "BUY", "reason": "BULLISH TREND: Candle retrace dan close di atas EMA9.", "snapshot": analysis}
            else: 
                return {"action": "SELL", "reason": "BEARISH TREND: Candle retrace dan close di bawah EMA9.", "snapshot": analysis}
        
        return {"action": "HOLD", "reason": f"Menunggu setup Exora Vulcan Sniper. Bias: {analysis['bias']}."}

# --- LOGIKA TRADING UTAMA ---
# Modified to accept current_trades_for_pair for backtesting context
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, current_candle_data_for_pair=None):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    # Use provided candle data for snapshot, otherwise fall back to market_state (live)
    exit_snapshot = None
    if current_candle_data_for_pair:
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(current_candle_data_for_pair)
    elif trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    
    # save_trades() will be called outside the async function to avoid race conditions
    # or inside the backtest loop for progress saving.
    # For live, save_trades will be called by the worker loop.
    
    # Only send notification for live trades
    if not current_candle_data_for_pair: # Simple heuristic: if no candle data provided, assume it's live
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f} | Trigger: {close_trigger_reason}"
        send_termux_notification(notif_title, notif_content)

# MODIFIED: run_autopilot_analysis to be used by both live and backtest.
# It now takes candle_data_for_analysis and relevant_trades as arguments explicitly.
async def run_autopilot_analysis(instrument_id, candle_data_for_analysis, relevant_trades_for_pair):
    global is_ai_thinking
    # This function is now called directly by backtest or live, 
    # so we manage the thinking state here for each call.
    
    # No cooldown for backtesting, only for live if API fails
    
    # Check if there are enough candles for full analysis based on get_market_analysis requirements
    if not candle_data_for_analysis or len(candle_data_for_analysis) < 100 + 3:
        # print_colored(f"Warning: Not enough candle data for {instrument_id} to perform full analysis ({len(candle_data_for_analysis) if candle_data_for_analysis else 0} candles).", Fore.YELLOW)
        return # Skip analysis if data insufficient

    # Acquire thinking lock (important for live, less so for serial backtest)
    # For backtest, this flag might need to be context-specific if run in parallel,
    # but currently backtest runs serially per pair then live runs.
    if is_ai_thinking and not relevant_trades_for_pair: # If it's live and another pair is thinking
        return

    # Temporarily mark as thinking for this operation
    _is_thinking_for_this_call = False
    if not relevant_trades_for_pair: # Only manage global lock for live mode
        is_ai_thinking = True
        _is_thinking_for_this_call = True

    try:
        local_brain = LocalAI(current_settings, relevant_trades_for_pair)
        decision = local_brain.get_decision(candle_data_for_analysis, None, instrument_id) # No open_position check here, handled higher
        
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data_for_analysis[-1]['close'] # Entry price is the close of the trigger candle
        
        # Find if there is an open position for this pair
        open_position_for_this_pair = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)

        if action in ["BUY", "SELL"] and not open_position_for_this_pair:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {
                "id": int(time.time()),
                "instrumentId": instrument_id,
                "type": trade_type,
                "entryTimestamp": datetime.now().isoformat(),
                "entryPrice": current_price,
                "entryReason": reason,
                "status": 'OPEN',
                "entry_snapshot": decision.get("snapshot"), 
                "run_up_percent": 0.0,
                "max_drawdown_percent": 0.0,
                "trailing_stop_price": None, 
                "current_tp_checkpoint_level": 0.0 
            }
            autopilot_trades.append(new_trade)
            # save_trades() will be called by the calling loop (backtest_worker or data_refresh_worker)
            
            # Send notification for live trades only
            if not relevant_trades_for_pair: # Heuristic: If relevant_trades_for_pair is from global (live), send notif
                notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
                notif_content = f"{instrument_id}: Entry @ {current_price:.4f} | {reason}"
                send_termux_notification(notif_title, notif_content)
        
        return decision # Return decision for external use (e.g., backtest can check it)

    except Exception as e:
        print_colored(f"Error in autopilot analysis for {instrument_id}: {e}", Fore.RED)
        if not relevant_trades_for_pair: # Only apply cooldown for live API failures
            is_autopilot_in_cooldown[instrument_id] = True
            await asyncio.sleep(60)
            is_autopilot_in_cooldown[instrument_id] = False
    finally:
        if _is_thinking_for_this_call: # Release lock only if we acquired it
            is_ai_thinking = False

# check_realtime_position_management for new Trailing TP logic (Checkpoint system)
# Modified to accept current_trades_for_pair and current_candle_data_for_pair for backtesting context
async def check_realtime_position_management(instrument_id, latest_price, current_trades_for_pair=None, current_candle_data_for_pair=None):
    # Use global autopilot_trades if not provided (live mode)
    trades_to_check = current_trades_for_pair if current_trades_for_pair is not None else autopilot_trades

    open_position = next((t for t in trades_to_check if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl
        
    sl_pct = current_settings.get('stop_loss_pct')
    if sl_pct is not None and current_pnl <= -abs(sl_pct):
        # We pass current_candle_data_for_pair to analyze_and_close_trade for backtest context
        await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.", current_candle_data_for_pair)
        # Mark position as closed in the global list if this was live, or in current_trades_for_pair for backtest
        open_position['status'] = 'CLOSED' # Ensure status is updated locally before next save
        return

    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    
    current_tp_checkpoint_level = open_position.get("current_tp_checkpoint_level", 0.0)

    if current_tp_checkpoint_level == 0.0 and current_pnl >= activation_pct:
        open_position['current_tp_checkpoint_level'] = activation_pct
        open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + activation_pct / 100) if open_position['type'] == 'LONG' else \
                                                open_position['entryPrice'] * (1 - activation_pct / 100)

    if current_tp_checkpoint_level > 0.0:
        steps_passed = math.floor((current_pnl - current_tp_checkpoint_level) / gap_pct)
        
        if steps_passed > 0:
            open_position['current_tp_checkpoint_level'] += steps_passed * gap_pct
            open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + open_position['current_tp_checkpoint_level'] / 100) if open_position['type'] == 'LONG' else \
                                                    open_position['entryPrice'] * (1 - open_position['current_tp_checkpoint_level'] / 100)
        
        if open_position['type'] == 'LONG' and latest_price <= open_position.get('trailing_stop_price', 0):
            await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", current_candle_data_for_pair)
            open_position['status'] = 'CLOSED' # Ensure status is updated locally before next save
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
             await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", current_candle_data_for_pair)
             open_position['status'] = 'CLOSED' # Ensure status is updated locally before next save

    # save_trades() will be called by the calling loop (backtest_worker or data_refresh_worker)

# NEW: Backtest function
async def run_backtest_for_pair(pair_id, timeframe, duration_months):
    print_colored(f"\n🚀 Memulai Backtest untuk {pair_id} ({timeframe}) selama {duration_months} bulan...", Fore.YELLOW, Style.BRIGHT)
    
    end_time_ms = int(datetime.now().timestamp() * 1000)
    start_time_ms = int((datetime.now() - timedelta(days=30 * duration_months)).timestamp() * 1000)

    historical_candles = fetch_historical_bybit_candles(pair_id, timeframe, start_time_ms, end_time_ms)
    
    if not historical_candles:
        print_colored(f"❌ Backtest {pair_id}: Gagal mengambil data historis atau data tidak cukup.", Fore.RED)
        return

    # Filter out existing trades for this pair within the backtest period to avoid re-simulating existing ones
    # This is a heuristic; a more robust solution might clear trades for the pair or compare timestamps
    existing_trades_for_pair = [t for t in autopilot_trades if t['instrumentId'] == pair_id]
    
    # Backtest will simulate candle by candle
    temp_candle_data_history = [] # To hold candles needed for EMA calculations (minimum 100+3 candles)
    backtest_trades_for_pair = [] # Trades generated during this backtest run for this pair

    progress_step = len(historical_candles) // 10 if len(historical_candles) >= 10 else 1
    
    # Loop through each historical candle
    for i, candle in enumerate(historical_candles):
        # Build up candle history for LocalAI.get_market_analysis()
        temp_candle_data_history.append(candle)
        
        # Only start analysis once enough candles are accumulated
        if len(temp_candle_data_history) < 100 + 3: # Minimum for EMA100 + 3 for snapshot/trigger
            continue

        # Simulate analysis and decision making for the current candle
        # We pass a slice of history to mimic what LocalAI would see at that point in time
        await run_autopilot_analysis(pair_id, temp_candle_data_history, backtest_trades_for_pair)

        # Simulate position management for current open trades
        # Need to ensure only trades for *this pair* are managed by this backtest loop
        open_position = next((t for t in backtest_trades_for_pair if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        if open_position: # Only manage if there is an open position
            await check_realtime_position_management(pair_id, candle['close'], backtest_trades_for_pair, temp_candle_data_history)

        # Update global autopilot_trades with changes from backtest_trades_for_pair
        # For simplicity, we just extend and then filter. A better way might be to track indices.
        
        # Periodically save trades to avoid data loss on crash and provide feedback
        if (i + 1) % progress_step == 0 or (i + 1) == len(historical_candles):
            print_colored(f"  Backtest {pair_id}: Memproses {i+1}/{len(historical_candles)} candles. Posisi buka: {len([t for t in backtest_trades_for_pair if t['status'] == 'OPEN'])}", Fore.BLUE, end='\r')
            # Append newly closed trades to global autopilot_trades, and update open ones
            # A more robust system would update global trades more carefully.
            # For simplicity, we'll just re-load/save at the end of the backtest.
            # But the 'autopilot_trades' passed to LocalAI is the global one.
            # Let's clarify: backtest_trades_for_pair is for the LocalAI.past_trades,
            # and actual trades are added to the global autopilot_trades directly.
            pass # The save_trades() will happen after appending all results

    # After loop, append all new trades from this backtest to the global list
    # Need to remove any OPEN trades from backtest_trades_for_pair if they were not closed by end of backtest period
    for trade in backtest_trades_for_pair:
        if trade['status'] == 'OPEN':
            trade['status'] = 'CLOSED' # Assume all open trades are closed at end of backtest
            trade['exitPrice'] = historical_candles[-1]['close'] # Close at last candle price
            trade['exitTimestamp'] = datetime.fromtimestamp(historical_candles[-1]['time'] / 1000).isoformat()
            trade['pl_percent'] = calculate_pnl(trade['entryPrice'], trade['exitPrice'], trade.get('type'))

    # Add backtest trades to the global list. Filter out potential duplicates if run multiple times.
    # A cleaner approach is to use trade IDs to ensure uniqueness or clear prior backtest trades for the pair.
    # For now, append and rely on `load_trades` initializing new fields.
    
    # Filter out trades for this pair that might have been simulated BEFORE this backtest run
    global autopilot_trades
    autopilot_trades = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
    autopilot_trades.extend(backtest_trades_for_pair)
    save_trades() # Save after all trades from this backtest are added

    print_colored(f"\n✅ Backtest untuk {pair_id} selesai! Total trade: {len(backtest_trades_for_pair)}.", Fore.GREEN, Style.BRIGHT)
    time.sleep(1) # Give time to read message


# NEW: Function to check and run backtests for all watched pairs
async def run_initial_backtest_if_needed():
    bt_duration_months = current_settings.get("backtest_duration_months", 3)
    
    # Get all pairs that are watched
    watched_pairs_to_backtest = list(current_settings.get("watched_pairs", {}).items()) # (pair_id, timeframe) tuples

    if not watched_pairs_to_backtest:
        print_colored("Watchlist kosong, tidak ada pair untuk dibacktest.", Fore.YELLOW)
        return

    print_colored("\n--- Memeriksa Pair untuk Backtest Awal ---", Fore.CYAN, Style.BRIGHT)
    
    # Filter trades for relevant pairs and within backtest duration
    cutoff_timestamp_ms = int((datetime.now() - timedelta(days=30 * bt_duration_months)).timestamp() * 1000)
    
    for pair_id, timeframe in watched_pairs_to_backtest:
        # Check if pair has trades within the backtest duration
        recent_trades_for_pair = [
            t for t in autopilot_trades 
            if t['instrumentId'] == pair_id and 
               (t['entryTimestamp'] and datetime.fromisoformat(t['entryTimestamp'].replace('Z', '')).timestamp() * 1000 >= cutoff_timestamp_ms)
        ]
        
        if not recent_trades_for_pair:
            print_colored(f"  {pair_id} ({timeframe}): Tidak ada riwayat trade yang cukup terbaru. Menjalankan backtest...", Fore.YELLOW)
            await run_backtest_for_pair(pair_id, timeframe, bt_duration_months)
        else:
            print_colored(f"  {pair_id} ({timeframe}): Riwayat trade terbaru ditemukan. Melanjutkan...", Fore.GREEN)
    
    print_colored("--- Selesai Pengecekan Backtest ---", Fore.CYAN, Style.BRIGHT)
    time.sleep(1) # Give time to read the summary

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    # Get relevant trades for LocalAI from global autopilot_trades
                    relevant_trades_for_pair = [t for t in autopilot_trades if t['instrumentId'] == pair_id]
                    # Get candle data from global market_state
                    candle_data = market_state.get(pair_id, {}).get("candle_data")
                    
                    if candle_data:
                        # Find if there is an open position for this pair
                        open_position_for_this_pair = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                        
                        # Run analysis if no open position, or if there is an open position (for management)
                        if not open_position_for_this_pair:
                            # Only run decision making if no open position
                            asyncio.run(run_autopilot_analysis(pair_id, candle_data, relevant_trades_for_pair))
                            # save_trades() is now called periodically by data_refresh_worker or when trades are closed
                        else:
                            # Position management (SL/TP) is handled in data_refresh_worker,
                            # so no direct action needed here if open_position_for_this_pair exists.
                            # But we need to ensure the status update from analyze_and_close_trade gets saved.
                            pass # Actual position management happens in data_refresh_worker loop
                    # Small delay between pairs to avoid overwhelming
                    time.sleep(0.1) 
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

def data_refresh_worker():
    global market_state
    last_save_time = time.time()
    SAVE_INTERVAL_SECONDS = 30 # Save trades every 30 seconds to disk
    
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, timeframe in watched_pairs.items():
                data = fetch_bybit_candle_data(pair_id, timeframe)
                if data: 
                    if len(data) >= 100 + 3: 
                        analysis = LocalAI(current_settings, []).get_market_analysis(data)
                    else:
                        analysis = None 
                    market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                    
                    if is_autopilot_running:
                        latest_price = data[-1]['close']
                        # Position management runs for open trades
                        # No need for is_ai_thinking check here as it's for external updates
                        asyncio.run(check_realtime_position_management(pair_id, latest_price, autopilot_trades, data)) # Pass global trades and current data

                time.sleep(0.5) 
        
        # Periodically save all trades for safety
        if time.time() - last_save_time >= SAVE_INTERVAL_SECONDS:
            save_trades()
            last_save_time = time.time()

        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 
        'bt_months': ('backtest_duration_months', ' bulan') # NEW: Backtest duration setting
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full_key, unit) in setting_map.items():
            display_key = key.capitalize().ljust(10)
            print_colored(f"{display_key} ({key:<10}) : {current_settings[full_key]}{unit}", Fore.WHITE)
        print(); return
    if len(parts) == 3 and parts[0] == '!set':
        key_short = parts[1].lower()
        if key_short not in setting_map: print_colored(f"Error: Kunci '{key_short}' tidak dikenal.", Fore.RED); return
        try:
            value = float(parts[2])
            if key_short == 'caution' and not (0.0 <= value <= 1.0):
                print_colored("Error: Nilai 'caution' harus antara 0.0 dan 1.0.", Fore.RED); return
            if key_short == 'bt_months' and value < 1: # Backtest duration must be at least 1 month
                print_colored("Error: Durasi backtest minimal 1 bulan.", Fore.RED); return
            if value < 0: print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
        except ValueError: print_colored(f"Error: Nilai '{parts[2]}' harus berupa angka.", Fore.RED); return
        key_full, unit = setting_map[key_short]
        current_settings[key_full] = value; save_settings()
        print_colored(f"Pengaturan '{key_full}' berhasil diubah menjadi {value}{unit}.", Fore.GREEN, Style.BRIGHT); return
    print_colored("Format salah. Gunakan '!settings' atau '!set <key> <value>'.", Fore.RED)

def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="") # Clear console
            print_colored("--- VULCAN'S EDITION LIVE DASHBOARD ---", Fore.CYAN, Style.BRIGHT)
            print_colored(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Refresh: {REFRESH_INTERVAL_SECONDS}s | AI Cycle: {current_settings.get('analysis_interval_sec')}s", Fore.WHITE)
            print_colored("="*60, Fore.CYAN)
            watched_pairs = current_settings.get("watched_pairs", {})
            if not watched_pairs:
                print_colored("\nWatchlist kosong. Tekan Ctrl+C untuk kembali dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            for pair_id, timeframe in watched_pairs.items():
                print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                pair_state = market_state.get(pair_id)
                
                if open_pos:
                    price = pair_state['candle_data'][-1]['close'] if pair_state and pair_state.get('candle_data') and pair_state['candle_data'] else open_pos['entryPrice']
                    pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
                    pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                    type_color = Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED
                    print_colored("  Status    : ", end=''); print_colored("POSITION OPEN", Fore.YELLOW, Style.BRIGHT)
                    print_colored("  Tipe      : ", end=''); print_colored(f"{open_pos.get('type')}", type_color, Style.BRIGHT)
                    print_colored("  Entry     : ", end=''); print_colored(f"{open_pos['entryPrice']:.4f}", Fore.WHITE)
                    print_colored("  Current   : ", end=''); print_colored(f"{price:.4f}", Fore.WHITE)
                    print_colored("  PnL       : ", end=''); print_colored(f"{pnl:.2f}%", pnl_color, Style.BRIGHT)
                    
                    if open_pos.get("current_tp_checkpoint_level", 0.0) > 0:
                        cp_level = open_pos["current_tp_checkpoint_level"]
                        ts_price = open_pos.get("trailing_stop_price")
                        print_colored("  TP Checkpoint: ", end=''); print_colored(f"Aktif @ {cp_level:.2f}% PnL ({ts_price:.4f})", Fore.MAGENTA)
                    else:
                        print_colored("  TP Checkpoint: ", end=''); print_colored(f"Menunggu {current_settings.get('trailing_tp_activation_pct'):.2f}% PnL", Fore.WHITE)
                else:
                    print_colored("  Status    : ", end=''); print_colored("Searching for setup...", Fore.BLUE)
                    if pair_state and pair_state.get("analysis"):
                        analysis = pair_state["analysis"]
                        bias_color = Fore.GREEN if analysis['bias'] == 'BULLISH' else Fore.RED if analysis['bias'] == 'BEARISH' else Fore.YELLOW
                        print_colored("  Trend     : ", end=''); print_colored(analysis['bias'], bias_color)
                        
                        print_colored(f"  Caution Lv: {current_settings.get('caution_level'):.2f}", Fore.YELLOW)
                        caution_level_current = current_settings.get("caution_level", 0.5)
                        BASE_SOL_TOL_DASH = 0.05
                        MAX_ADD_SOL_TOL_DASH = 0.20
                        BASE_EMA_TOL_DASH = 0.00005
                        MAX_ADD_EMA_TOL_DASH = 0.0005
                        actual_sol_tol_display = BASE_SOL_TOL_DASH + (caution_level_current * MAX_ADD_SOL_TOL_DASH)
                        actual_ema_tol_display = BASE_EMA_TOL_DASH + (caution_level_current * MAX_ADD_EMA_TOL_DASH)
                        print_colored(f"  Solid. Tol: {actual_sol_tol_display:.3f} | EMA Tol: {actual_ema_tol_display:.6f}", Fore.CYAN)
                        
                        if analysis.get('ema9_current') is not None and analysis.get('ema9_prev') is not None:
                            print_colored(f"  EMA9 Data : Prev Close {analysis['prev_candle_close']:.4f} vs EMA9 {analysis['ema9_prev']:.4f}", Fore.CYAN)
                            print_colored(f"              Current Close {analysis['current_candle_close']:.4f} vs EMA9 {analysis['ema9_current']:.4f}", Fore.CYAN)
                        
                        if 'pre_entry_candle_solidity' in analysis and 'pre_entry_candle_direction' in analysis:
                            solidity_str = [f"{s:.2f}" for s in analysis['pre_entry_candle_solidity']]
                            print_colored(f"  Pre-Entry Solidity: {solidity_str}", Fore.BLUE)
                            print_colored(f"  Pre-Entry Direction: {analysis['pre_entry_candle_direction']}", Fore.BLUE)
                    else:
                        print_colored("  Trend     : Menunggu data atau data tidak cukup...", Fore.WHITE)
            print_colored("\n"+"="*60, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard dan kembali ke command prompt.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def main():
    global is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    
    # Run initial backtest for un-backtested pairs before starting live operations
    # This must be run synchronously before autopilot_thread starts to prevent race conditions
    asyncio.run(run_initial_backtest_if_needed()) 

    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()
    
    while True:
        try:
            prompt_text = f"[Command] > "
            user_input = input(prompt_text)
            command_parts = user_input.split()
            if not command_parts: continue
            cmd = command_parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan. Dashboard aktif.", Fore.YELLOW)
                elif not current_settings.get("watched_pairs"): print_colored("Error: Watchlist kosong. Gunakan '!watch <PAIR>' dulu.", Fore.RED)
                else: 
                    is_autopilot_running = True
                    print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN, Style.BRIGHT)
                    time.sleep(1); run_dashboard_mode()
                    is_autopilot_running = False
                    print_colored("\n🛑 Live Dashboard ditutup. Autopilot dinonaktifkan.", Fore.RED, Style.BRIGHT)
                    print_colored("Ketik '!start' untuk masuk lagi atau '!exit' untuk keluar.", Fore.YELLOW)
            elif cmd == '!stop': print_colored("Gunakan '!start' untuk masuk ke dashboard, lalu Ctrl+C untuk berhenti.", Fore.YELLOW)
            elif cmd == '!watchlist':
                watched = current_settings.get("watched_pairs", {})
                if not watched: print_colored("Watchlist kosong.", Fore.YELLOW)
                else:
                    print_colored("\n--- Watchlist ---", Fore.CYAN, Style.BRIGHT)
                    for pair, tf in watched.items(): print_colored(f"- {pair} (Timeframe: {tf})", Fore.WHITE)
            elif cmd == '!watch':
                if len(command_parts) >= 2:
                    pair_id = command_parts[1].upper()
                    tf = command_parts[2] if len(command_parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"Pair {pair_id} dengan TF {tf} ditambahkan ke watchlist.", Fore.GREEN)
                else: print_colored("Format salah. Gunakan: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(command_parts) == 2:
                    pair_id = command_parts[1].upper()
                    if pair_id in current_settings['watched_pairs']:
                        del current_settings['watched_pairs'][pair_id]
                        if pair_id in market_state: del market_state[pair_id]
                        save_settings()
                        print_colored(f"Pair {pair_id} dihapus dari watchlist.", Fore.YELLOW)
                    else: print_colored(f"Error: Pair {pair_id} tidak ada di watchlist.", Fore.RED)
                else: print_colored("Format salah. Gunakan: !unwatch <PAIR>", Fore.RED)
            elif cmd == '!history':
                display_history()
            elif cmd in ['!settings', '!set']:
                handle_settings_command(command_parts)
            else:
                print_colored(f"Perintah '{user_input}' tidak dikenal. Ketik '!help'.", Fore.RED)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga di main loop: {e}", Fore.RED)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
