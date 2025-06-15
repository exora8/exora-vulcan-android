import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import asyncio
import math
import sys # <-- TAMBAHKAN INI UNTUK PROGRESS BAR

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"
REFRESH_INTERVAL_SECONDS = 0.5
BYBIT_MAX_KLINE_LIMIT = 1000 # Max candles per API request for Bybit

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
market_state = {} 
is_ai_thinking = False
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
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, bt_dur)", Fore.GREEN) 
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    # Updated default settings for Exora Vulcan Trailing TP and SL
    default_settings = {
        "stop_loss_pct": 0.20, # Default -0.20% as per user
        "fee_pct": 0.1,
        "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30, # 0.30% activation as per user
        "trailing_tp_gap_pct": 0.05,       # 0.05% gap as per user
        "caution_level": 0.5,              # Overarching sensitivity for learning from losses (0.0 to 1.0)
        "backtest_duration_months": 2,     # NEW: Default backtest duration in months
        "watched_pairs": {} # Structure: {"BTC-USDT": {"tf": "1H", "backtested": false}}
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            # Ensure new fields are added and old formats are updated
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
            # Update old watched_pairs format if necessary (from {pair:tf} to {pair:{tf, backtested}})
            if isinstance(current_settings.get("watched_pairs"), dict):
                for pair, tf_or_dict in list(current_settings["watched_pairs"].items()):
                    if isinstance(tf_or_dict, str): # Old format: "PAIR": "TF"
                        current_settings["watched_pairs"][pair] = {"tf": tf_or_dict, "backtested": False}
                    elif "backtested" not in tf_or_dict: # New format but missing flag
                         current_settings["watched_pairs"][pair]["backtested"] = False
    else:
        current_settings = default_settings; save_settings()

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
    # Ensure new fields for trailing TP checkpoint are initialized if loading old trades
    for trade in autopilot_trades:
        if 'current_tp_checkpoint_level' not in trade:
            trade['current_tp_checkpoint_level'] = 0.0
        # Add source if missing
        if 'source' not in trade:
            trade['source'] = 'LIVE' # Assume old trades were live
        # Initialize run_up_percent and max_drawdown_percent if missing
        if 'run_up_percent' not in trade:
            trade['run_up_percent'] = 0.0
        if 'max_drawdown_percent' not in trade:
            trade['max_drawdown_percent'] = 0.0

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades):
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        trade_type = trade.get('type', 'LONG'); type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        trade_source = trade.get('source', 'LIVE') # Display source
        source_color = Fore.BLUE if trade_source == 'BACKTEST' else Fore.MAGENTA

        print_colored(f"--- Trade ID: {trade['id']} ({trade_source}) ---", source_color, Style.BRIGHT) # Show source
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
            print_colored(f"  Max Drawdown: {trade.get('max_drawdown_percent', 0.0):.2f}%", Fore.YELLOW) # Ensure drawdown is always displayed
            if 'entry_snapshot' in trade and not is_profit:
                snapshot = trade['entry_snapshot']
                print_colored(f"  Pelajaran (Snapshot):", Fore.MAGENTA)
                print_colored(f"    Bias: {snapshot.get('bias', 'N/A')}", Fore.MAGENTA)
                
                # Display EMA spread information if available
                if 'ema50' in snapshot and 'ema100' in snapshot:
                    ema50_val = snapshot.get('ema50', 0)
                    ema100_val = snapshot.get('ema100', 0)
                    spread_val = ema50_val - ema100_val
                    print_colored(f"    EMA50: {ema50_val:.4f} | EMA100: {ema100_val:.4f} | Spread: {spread_val:.6f}", Fore.MAGENTA)
                
                print_colored(f"    Prev Candle: Close {snapshot.get('prev_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_prev'):.4f}", Fore.MAGENTA)
                print_colored(f"    Current Candle: Close {snapshot.get('current_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_current'):.4f}", Fore.MAGENTA)
                
                # Display 3 previous candles snapshot
                if 'pre_entry_candle_solidity' in snapshot and 'pre_entry_candle_direction' in snapshot:
                    solidity_str = [f"{s:.2f}" for s in snapshot['pre_entry_candle_solidity']]
                    print_colored(f"    3 Prev Candles Solidity: {solidity_str}", Fore.MAGENTA)
                    print_colored(f"    3 Prev Candles Direction: {snapshot['pre_entry_candle_direction']}", Fore.MAGENTA)
        print()

# --- FUNGSI API (BYBIT) ---
def _fetch_kline_single_request(instId, interval, limit, end_time_ms):
    """Helper to fetch a single batch of kline data."""
    bybit_symbol = instId.replace('-', '')
    url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={interval}&limit={limit}&endTime={end_time_ms}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            # Bybit returns newest first, we want oldest first
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in data['result']['list']][::-1]
        else:
            print_colored(f"Error fetching kline for {instId}: {data.get('retMsg', 'Unknown error')}", Fore.RED)
            return None
    except requests.exceptions.RequestException as e:
        print_colored(f"Network error fetching kline for {instId}: {e}", Fore.RED)
        return None
    except Exception as e:
        print_colored(f"Unexpected error fetching kline for {instId}: {e}", Fore.RED)
        return None

def _fetch_historical_candle_data(instId, timeframe, duration_months):
    """Fetches historical candle data for backtesting."""
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60')
    
    # Calculate estimated candles needed based on interval in minutes
    interval_in_minutes = 1 # Default to 1 minute
    if bybit_interval == 'D':
        interval_in_minutes = 24 * 60
    elif bybit_interval == 'W':
        interval_in_minutes = 7 * 24 * 60
    elif bybit_interval.isdigit():
        interval_in_minutes = int(bybit_interval) # For 1, 3, 5, 15, 30, 60, 120, 240 minutes

    # Max candles needed: duration_months * (days per month) * (hours per day) * (minutes per hour) / (interval in minutes)
    estimated_candles_needed = math.ceil(duration_months * 30.4375 * 24 * 60 / interval_in_minutes) # Average days per month

    print_colored(f"Fetching approx. {estimated_candles_needed} candles for {instId} ({timeframe}) over {duration_months} months...", Fore.YELLOW)

    all_candles = []
    current_end_time_ms = int(datetime.now().timestamp() * 1000) # Start from now, go backwards

    # Calculate target historical start time
    target_start_timestamp_ms = int((datetime.now() - timedelta(days=duration_months * 30.4375)).timestamp() * 1000)

    while True:
        # Fetch up to BYBIT_MAX_KLINE_LIMIT candles
        batch = _fetch_kline_single_request(instId, bybit_interval, BYBIT_MAX_KLINE_LIMIT, current_end_time_ms)
        
        if batch is None or not batch:
            print_colored(f"Failed to fetch historical data or no more data for {instId}. Fetched {len(all_candles)} candles so far.", Fore.RED)
            break
        
        # Prepend the batch to maintain chronological order (oldest to newest)
        all_candles = batch + all_candles 
        
        # Set new end time to the timestamp of the oldest candle in the current batch
        # Minus 1 millisecond to avoid fetching the same candle in the next request
        current_end_time_ms = batch[0]['time'] - 1 
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.1) 
        print_colored(f"  Fetched {len(batch)} candles. Total: {len(all_candles)}. Oldest in current batch: {datetime.fromtimestamp(batch[0]['time']/1000).strftime('%Y-%m-%d %H:%M')}", Fore.YELLOW)
        
        # Stop criteria:
        # 1. We have enough candles
        # 2. We've gone back far enough in time
        # 3. No more candles returned by API (empty batch)
        if len(all_candles) >= estimated_candles_needed or current_end_time_ms <= target_start_timestamp_ms:
            # Trim to ensure we have exactly `estimated_candles_needed` or stop at target_start_timestamp_ms
            # Filter candles that are too old or too many
            all_candles = [c for c in all_candles if c['time'] >= target_start_timestamp_ms]
            # If we still have more than estimated, take the latest `estimated_candles_needed`
            if len(all_candles) > estimated_candles_needed:
                all_candles = all_candles[-estimated_candles_needed:]
            break # Exit the loop

    if len(all_candles) < (100 + 3): # Minimum for analysis
         print_colored(f"Warning: Not enough historical candle data ({len(all_candles)} candles) for {instId} to perform comprehensive backtest. Required ~103. Skipping.", Fore.YELLOW)
         return None

    return all_candles

def fetch_bybit_candle_data(instId, timeframe):
    """Fetches real-time candle data for live trading."""
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            # We need enough data for EMA100 and at least 4 candles for trigger + 3 pre-entry snapshot
            if len(candle_list) < 100 + 3: 
                return None # Not enough data
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception: return None

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# --- OTAK LOCAL AI (Updated for Exora Vulcan Sniper Entry with 3-candle snapshot and learning) ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair): self.settings = settings; self.past_trades = past_trades_for_pair
    
    # calculate_ema to return series of EMAs
    def calculate_ema(self, data, period):
        if len(data) < period: return []
        closes = [d['close'] for d in data]
        ema_values = []
        
        # Calculate initial SMA for the first 'period' closes
        initial_sma = sum(closes[:period]) / period
        ema_values.append(initial_sma)
        
        multiplier = 2 / (period + 1)
        
        # Calculate subsequent EMAs
        for i in range(period, len(closes)):
            ema = (closes[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
            
        return ema_values
    
    # analyze_candle_solidity function
    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        if full_range == 0: return 1.0 # Avoid division by zero, full solidity if no range
        return body / full_range

    def get_market_analysis(self, candle_data):
        # Need enough data for EMA 100, and at least 4 candles for EMA9 trigger logic + 3 pre-entry snapshot
        # candle_data[-1] (current), candle_data[-2] (prev), candle_data[-3], candle_data[-4] (oldest for snapshot)
        if len(candle_data) < 100 + 3: # 100 for EMA100, plus 3 additional candles for full snapshot (candle_data[-4] is needed)
            return None # Not enough data for full analysis and snapshot

        ema9_series = self.calculate_ema(candle_data, 9)
        ema50_series = self.calculate_ema(candle_data, 50)
        ema100_series = self.calculate_ema(candle_data, 100)

        # Ensure EMA series are long enough for the required indices
        if len(ema9_series) < 2 or len(ema50_series) < 1 or len(ema100_series) < 1:
            return None # Not enough EMA data calculated, perhaps due to insufficient initial candle data for periods

        analysis = {
            "ema9_current": ema9_series[-1],
            "ema9_prev": ema9_series[-2], # Get previous EMA9
            "ema50": ema50_series[-1],
            "ema100": ema100_series[-1],
            "current_candle_close": candle_data[-1]['close'],
            "prev_candle_close": candle_data[-2]['close']
        }
        
        bias = "RANGING";
        if analysis["ema50"] > analysis["ema100"]: bias = "BULLISH"
        elif analysis["ema50"] < analysis["ema100"]: bias = "BEARISH"
        analysis["bias"] = bias; 

        # Add 3 pre-entry candle snapshot for forensic analysis and learning
        pre_entry_candles = candle_data[-4:-1] # Candles before the current (trigger) candle
        pre_entry_solidity = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        pre_entry_direction = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        
        analysis["pre_entry_candle_solidity"] = pre_entry_solidity
        analysis["pre_entry_candle_direction"] = pre_entry_direction

        return analysis
    
    # MODIFIED: check_for_repeated_mistake for learning from losing trades with "similar" criteria
    def check_for_repeated_mistake(self, current_analysis, trade_type, instrument_id):
        # Only learn from losing trades (where PnL < Fee percentage)
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        
        if not losing_trades:
            return False # No losing trades to learn from

        # Extract current analysis features for comparison
        current_bias = current_analysis['bias']
        current_prev_close = current_analysis['prev_candle_close']
        current_curr_close = current_analysis['current_candle_close']
        current_ema9_prev = current_analysis['ema9_prev']
        current_ema9_current = current_analysis['ema9_current']
        current_pre_solidity = current_analysis.get('pre_entry_candle_solidity', [])
        current_pre_direction = current_analysis.get('pre_entry_candle_direction', [])
        current_ema50 = current_analysis['ema50'] 
        current_ema100 = current_analysis['ema100'] 

        # Get caution_level from settings
        caution_level = self.settings.get("caution_level", 0.5) 

        # Define base and max additional tolerances
        BASE_SOL_TOL = 0.05      # Minimum solidity tolerance (most strict)
        MAX_ADD_SOL_TOL = 0.20   # Max additional tolerance for solidity when caution_level is 1.0 (e.g., 0.05 + 0.20 = 0.25)
        
        BASE_EMA_TOL = 0.00005   # Minimum EMA spread tolerance (most strict)
        MAX_ADD_EMA_TOL = 0.0005 # Max additional tolerance for EMA spread (e.g., 0.00005 + 0.0005 = 0.00055)

        # Calculate actual tolerances based on caution_level
        actual_solidity_tolerance = BASE_SOL_TOL + (caution_level * MAX_ADD_SOL_TOL)
        actual_ema_spread_tolerance = BASE_EMA_TOL + (caution_level * MAX_ADD_EMA_TOL)

        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            
            # Ensure the past snapshot has all necessary data points
            required_keys = ['bias', 'prev_candle_close', 'current_candle_close', 
                             'ema9_prev', 'ema9_current', 'pre_entry_candle_solidity', 
                             'pre_entry_candle_direction', 'ema50', 'ema100'] 
            if not past_snapshot or not all(key in past_snapshot for key in required_keys):
                continue # Skip malformed snapshots

            # 1. Bias match: Trend must be the same (STILL EXACT)
            if current_bias != past_snapshot['bias']:
                continue

            # 1.1. EMA Spread match (Tolerance for trend strength)
            # Only compare spread if bias is not RANGING (meaning there's a clear trend)
            if current_bias != "RANGING":
                current_spread = abs(current_ema50 - current_ema100)
                past_ema50 = past_snapshot.get('ema50', 0)
                past_ema100 = past_snapshot.get('ema100', 0)
                past_spread = abs(past_ema50 - past_ema100)
                
                # If the difference in spread is too large, they are not "similar" trend contexts
                if abs(current_spread - past_spread) > actual_ema_spread_tolerance:
                    continue
            else: # If bias is RANGING, EMA spread comparison is not applicable in this context.
                  # We simply pass this check, assuming RANGING has no specific spread pattern to compare.
                pass 


            # 2. EMA9 Cross state match: Entry trigger conditions must be similar (STILL EXACT)
            # This is the core setup for the sniper entry, so it must be consistent.
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
            
            # 3. Relaxed 3-Candle Direction match: Last direction must match AND at least 2 out of 3 overall directions must match
            past_pre_direction = past_snapshot['pre_entry_candle_direction']
            
            # Ensure enough data points for direction comparison (should be 3)
            if len(current_pre_direction) < 3 or len(past_pre_direction) < 3:
                continue # Cannot compare if data is missing

            last_direction_matches = (current_pre_direction[-1] == past_pre_direction[-1]) # Paling penting, arah candle terakhir
            
            # Hitung berapa arah yang cocok dari 3 candle
            matching_directions_count = sum(1 for x, y in zip(current_pre_direction, past_pre_direction) if x == y)

            # Jika arah candle terakhir cocok DAN setidaknya 2 dari 3 arah keseluruhan cocok
            if not (last_direction_matches and matching_directions_count >= 2):
                continue

            # 4. Relaxed 3-Candle Solidity match: At least 2 out of 3 solidities must be within actual_solidity_tolerance
            past_pre_solidity = past_snapshot['pre_entry_candle_solidity']
            
            if len(current_pre_solidity) < 3 or len(past_pre_solidity) < 3:
                continue # Cannot compare if data is missing

            matching_solidity_count = 0
            for i in range(len(current_pre_solidity)):
                if abs(current_pre_solidity[i] - past_pre_solidity[i]) <= actual_solidity_tolerance:
                    matching_solidity_count += 1
            
            # Check if at least 2 out of 3 solidities are within tolerance
            if matching_solidity_count < 2:
                continue 

            # If all relaxed criteria match, this is a pattern that led to a loss.
            return True 

        return False # No repeated mistake pattern found

    # get_decision for Exora Vulcan Sniper Entry Logic
    def get_decision(self, candle_data, open_position, instrument_id):
        analysis = self.get_market_analysis(candle_data)
        
        # Ensure we have enough data and valid analysis values
        if not analysis: 
            return {"action": "HOLD", "reason": "Data tidak cukup atau analisis tidak valid."}
        
        if open_position:
            return {"action": "HOLD", "reason": "Memantau posisi terbuka..."}
        
        # Activate learning: Check if current setup is a repeated mistake
        # This check should be performed BEFORE considering opening a new position
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
            # If a potential entry signal exists, check if it's a repeated mistake
            if self.check_for_repeated_mistake(analysis, potential_trade_type, instrument_id):
                return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan {potential_trade_type} berdasarkan riwayat loss."}
            
            # If not a repeated mistake, then proceed with the entry
            if potential_trade_type == 'LONG':
                return {"action": "BUY", "reason": "BULLISH TREND: Candle retrace dan close di atas EMA9.", "snapshot": analysis}
            else: # potential_trade_type == 'SHORT'
                return {"action": "SELL", "reason": "BEARISH TREND: Candle retrace dan close di bawah EMA9.", "snapshot": analysis}
        
        return {"action": "HOLD", "reason": f"Menunggu setup Exora Vulcan Sniper. Bias: {analysis['bias']}."}

# --- LOGIKA TRADING UTAMA ---
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, is_backtest=False):
    """
    Menutup trade dan merekam hasilnya.
    Parameter is_backtest digunakan untuk membedakan notifikasi atau logging tambahan.
    """
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    # Ensure candle_data exists for the pair when closing
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        # For consistency, use a new LocalAI instance to get the analysis for the exit snapshot
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    # Keep entry_snapshot only if trade is a loss, for learning purposes
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    
    # Only save trades to file if it's not a temporary backtest (or if it's the final stage of backtest)
    # For simplicity, we save every closed trade. The `source` field will differentiate.
    save_trades()
    
    if not is_backtest: # Only send Termux notification for real-time trades
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f} | Trigger: {close_trigger_reason}"
        send_termux_notification(notif_title, notif_content)
    else:
        # Optional: Print a simpler message for backtest trades
        # print_colored(f"  [BT] {trade['instrumentId']} {trade.get('type')} Closed. PnL: {pnl:.2f}%", Fore.BLUE)
        pass # Keep quiet for backtest, only show final summary

async def run_autopilot_analysis(instrument_id, is_backtest=False):
    global is_ai_thinking
    
    # In backtest mode, is_ai_thinking is managed internally per candle.
    # In live mode, it's global to prevent concurrent analysis.
    if not is_backtest and (is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id)): return

    pair_state = market_state.get(instrument_id)
    
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3:
        # print_colored(f"Warning: Not enough candle data for {instrument_id} to perform full analysis ({len(pair_state['candle_data']) if pair_state and pair_state.get('candle_data') else 0} candles).", Fore.YELLOW)
        return # Skip analysis if data insufficient

    is_ai_thinking = True # Set for both live and backtest
    try:
        candle_data = market_state[instrument_id]["candle_data"] # Use updated candle_data from market_state
        
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        local_brain = LocalAI(current_settings, relevant_trades) # Pass current_settings here

        # Perform market analysis for the current candle_data
        analysis = local_brain.get_market_analysis(candle_data)
        if not analysis:
            # print_colored(f"Warning: Insufficient data for analysis for {instrument_id} at current point.", Fore.YELLOW)
            return {"action": "HOLD", "reason": "Insufficient analysis data."} # Return if analysis not possible
        market_state[instrument_id]["analysis"] = analysis # Store analysis in market_state

        decision = local_brain.get_decision(candle_data, open_position, instrument_id)
        
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close'] # Entry price is the close of the trigger candle
        
        # Handle position management (SL/TP) *before* potentially opening a new trade in backtest
        # For live, this is handled by check_realtime_position_management separately in data_refresh_worker
        if is_backtest and open_position:
            # Pass is_backtest=True to analyze_and_close_trade
            closed = await _check_realtime_position_management_internal(open_position, current_price, is_backtest=True)
            # After potential closure, re-check open_position status for new entry decision
            open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)


        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {
                "id": int(time.time()),
                "instrumentId": instrument_id,
                "type": trade_type,
                "entryTimestamp": datetime.fromtimestamp(candle_data[-1]['time'] / 1000).isoformat(), # Use candle timestamp for backtest
                "entryPrice": current_price,
                "entryReason": reason,
                "status": 'OPEN',
                "entry_snapshot": decision.get("snapshot"), # Snapshot contains EMA values + 3 candle info
                "run_up_percent": 0.0,
                "max_drawdown_percent": 0.0,
                "trailing_stop_price": None, 
                "current_tp_checkpoint_level": 0.0,
                "source": 'BACKTEST' if is_backtest else 'LIVE' # Mark trade source
            }
            autopilot_trades.append(new_trade)
            save_trades()
            if not is_backtest:
                notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
                notif_content = f"{instrument_id}: Entry @ {current_price:.4f} | {reason}"
                send_termux_notification(notif_title, notif_content)
            # else:
            #     print_colored(f"  [BT] {instrument_id} {trade_type} Opened @ {current_price:.4f}", Fore.BLUE)
    except Exception as e:
        print_colored(f"Error in autopilot analysis for {instrument_id} (backtest: {is_backtest}): {e}", Fore.RED)
        if not is_backtest: # Only apply cooldown for live errors
            is_autopilot_in_cooldown[instrument_id] = True
            await asyncio.sleep(60) 
            is_autopilot_in_cooldown[instrument_id] = False
    finally: 
        if not is_backtest: # Only manage global thinking flag for live mode
            is_ai_thinking = False

# --- BACKTESTING LOGIC ---
async def _check_realtime_position_management_internal(open_position, latest_price, is_backtest=False):
    """Internal helper for position management that can be called by backtest or live."""
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl
        
    sl_pct = current_settings.get('stop_loss_pct')
    if sl_pct is not None and current_pnl <= -abs(sl_pct):
        await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.", is_backtest=is_backtest)
        return True # Position closed

    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    
    current_tp_checkpoint_level = open_position.get("current_tp_checkpoint_level", 0.0)

    if current_tp_checkpoint_level == 0.0 and current_pnl >= activation_pct:
        open_position['current_tp_checkpoint_level'] = activation_pct
        open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + activation_pct / 100) if open_position['type'] == 'LONG' else \
                                                open_position['entryPrice'] * (1 - activation_pct / 100)

    if current_tp_checkpoint_level > 0.0:
        # Calculate the actual PnL represented by the current checkpoint
        checkpoint_price = open_position['entryPrice'] * (1 + current_tp_checkpoint_level / 100) if open_position['type'] == 'LONG' else \
                           open_position['entryPrice'] * (1 - current_tp_checkpoint_level / 100)
        
        # Check if the market price has moved sufficiently beyond the current checkpoint to set a new one
        if open_position['type'] == 'LONG':
            # Price moved up enough to create a new checkpoint
            # Use current_pnl for dynamic step calculation instead of fixed checkpoint price calculation
            # This logic should be `if current_pnl >= (current_tp_checkpoint_level + gap_pct):`
            # The steps_passed logic from previous version was more robust for multiple steps.
            steps_passed = math.floor((current_pnl - current_tp_checkpoint_level) / gap_pct)
            if steps_passed > 0:
                open_position['current_tp_checkpoint_level'] += steps_passed * gap_pct
                open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + open_position['current_tp_checkpoint_level'] / 100)
        else: # SHORT
            # Price moved down enough to create a new checkpoint
            # This logic should be `if current_pnl >= (current_tp_checkpoint_level + gap_pct):`
            steps_passed = math.floor((current_pnl - current_tp_checkpoint_level) / gap_pct)
            if steps_passed > 0:
                open_position['current_tp_checkpoint_level'] += steps_passed * gap_pct
                open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 - open_position['current_tp_checkpoint_level'] / 100)

        # Check for exit condition (price falls back to or crosses trailing_stop_price/checkpoint)
        if open_position['type'] == 'LONG' and latest_price <= open_position.get('trailing_stop_price', 0):
            await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest=is_backtest)
            return True # Position closed
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
            await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest=is_backtest)
            return True # Position closed
    
    return False # Position not closed


async def perform_backtest(instrument_id, timeframe, duration_months):
    """
    Performs backtesting for a given instrument and timeframe over a specified duration.
    All trades during backtest will be marked with 'source': 'BACKTEST'.
    """
    print_colored(f"\n[BACKTEST] Memulai backtest untuk {instrument_id} ({timeframe}) selama {duration_months} bulan...", Fore.CYAN, Style.BRIGHT)
    
    backtest_start_time = time.time()
    backtest_total_trades_count = 0
    backtest_win_trades = 0
    backtest_loss_trades = 0
    backtest_total_profit_sum = 0.0

    historical_candles = _fetch_historical_candle_data(instrument_id, timeframe, duration_months)

    if not historical_candles:
        print_colored(f"[BACKTEST] Gagal mendapatkan data historis untuk {instrument_id}. Backtest dilewati.", Fore.RED)
        return False
    
    # Calculate total candles to process for progress bar
    total_candles_to_process = len(historical_candles)
    
    open_position_during_backtest = None # Track a single open position for this backtest run
    
    # Iterate through historical candles, simulating live data feed
    for i, current_candle in enumerate(historical_candles):
        # Update current market data slice for this backtest step
        # The slice includes all candles from the beginning up to the current_candle
        market_state[instrument_id] = {"candle_data": historical_candles[:i+1], "analysis": None} # Analysis is done below

        # Before deciding on new trade, check any open position's SL/TP
        if open_position_during_backtest:
            closed = await _check_realtime_position_management_internal(
                open_position_during_backtest, 
                current_candle['close'], 
                is_backtest=True # Mark as backtest
            )
            if closed:
                # Update counters based on the just-closed trade
                if open_position_during_backtest['pl_percent'] > current_settings.get('fee_pct', 0.1):
                    backtest_win_trades += 1
                else:
                    backtest_loss_trades += 1
                backtest_total_trades_count += 1
                backtest_total_profit_sum += open_position_during_backtest['pl_percent']
                open_position_during_backtest = None # Reset open position if closed

        # Run autopilot analysis for potential new trade entry
        await run_autopilot_analysis(instrument_id, is_backtest=True)
        
        # Update open_position_during_backtest from autopilot_trades after potential new trade
        # Find the latest open backtest trade for this instrument.
        latest_open_trade = next((t for t in reversed(autopilot_trades) if t['instrumentId'] == instrument_id and t['status'] == 'OPEN' and t.get('source') == 'BACKTEST'), None)
        open_position_during_backtest = latest_open_trade

        # Update and display progress bar
        if total_candles_to_process > 0:
            progress_percent = (i + 1) / total_candles_to_process * 100
            sys.stdout.write(f"\r[BACKTEST] Progress: {progress_percent:.2f}% | Trades: {backtest_total_trades_count} (W:{backtest_win_trades}/L:{backtest_loss_trades}) | PnL: {backtest_total_profit_sum:.2f}%")
            sys.stdout.flush()

    # After iterating all candles, if there's an open position, close it
    if open_position_during_backtest:
        sys.stdout.write('\n') # Move to next line before final message
        print_colored(f"[BACKTEST] Menutup sisa posisi terbuka untuk {instrument_id} pada akhir backtest.", Fore.YELLOW)
        await analyze_and_close_trade(open_position_during_backtest, historical_candles[-1]['close'], "End of backtest", is_backtest=True)
        # Update final counters
        if open_position_during_backtest['pl_percent'] > current_settings.get('fee_pct', 0.1):
            backtest_win_trades += 1
        else:
            backtest_loss_trades += 1
        backtest_total_trades_count += 1
        backtest_total_profit_sum += open_position_during_backtest['pl_percent']
        sys.stdout.write(f"\r{' ' * 100}\r") # Clear the progress line before final summary
        sys.stdout.flush()


    backtest_end_time = time.time()
    duration = backtest_end_time - backtest_start_time
    
    print_colored(f"\n[BACKTEST] Backtest untuk {instrument_id} selesai!", Fore.GREEN, Style.BRIGHT)
    print_colored(f"  Durasi: {duration:.2f} detik", Fore.GREEN)
    print_colored(f"  Total Trade: {backtest_total_trades_count}", Fore.GREEN)
    print_colored(f"  Win Trades: {backtest_win_trades}", Fore.GREEN)
    print_colored(f"  Loss Trades: {backtest_loss_trades}", Fore.RED)
    print_colored(f"  Total PnL: {backtest_total_profit_sum:.2f}%", Fore.GREEN if backtest_total_profit_sum > 0 else Fore.RED)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)

    return True


# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    # Run live analysis only for pairs that have completed backtest
                    if current_settings['watched_pairs'][pair_id].get("backtested", False):
                        asyncio.run(run_autopilot_analysis(pair_id))
                        time.sleep(0.1) # Small sleep between pairs to avoid overwhelming
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

# Renamed to distinguish from internal helper, called by data_refresh_worker for live trades
async def check_realtime_position_management_live(instrument_id, latest_price):
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN' and t.get('source') == 'LIVE'), None)
    if not open_position: return
    
    # Call the internal helper, marking it as a live trade
    await _check_realtime_position_management_internal(open_position, latest_price, is_backtest=False)
    save_trades() # Save state after every check

def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, pair_data in watched_pairs.items(): # Iterate over pair_data dict
                if not pair_data.get("backtested", False): # Skip pairs not yet backtested
                    continue

                data = fetch_bybit_candle_data(pair_id, pair_data['tf']) # Pass timeframe from pair_data
                if data: 
                    # Only calculate analysis if we have enough data after fetching (min 100+3 candles)
                    if len(data) >= 100 + 3: 
                        analysis = LocalAI(current_settings, []).get_market_analysis(data)
                    else:
                        analysis = None # Not enough data for full analysis

                    market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                    
                    if is_autopilot_running:
                        latest_price = data[-1]['close']
                        asyncio.run(check_realtime_position_management_live(pair_id, latest_price))
                time.sleep(0.5) # Short delay between fetching data for different pairs
        stop_event.wait(REFRESH_INTERVAL_SECONDS) # Main refresh interval

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 
        'bt_dur': ('backtest_duration_months', ' bulan') # Added backtest duration
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
            # Validate caution_level range
            if key_short == 'caution' and not (0.0 <= value <= 1.0):
                print_colored("Error: Nilai 'caution' harus antara 0.0 dan 1.0.", Fore.RED); return
            # Validate backtest_duration_months
            if key_short == 'bt_dur' and value <= 0:
                print_colored("Error: Durasi backtest harus lebih dari 0 bulan.", Fore.RED); return
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
            for pair_id, pair_data in watched_pairs.items(): # Iterate over pair_data dict
                # Only display if backtested for now, to ensure data is rich
                # if not pair_data.get("backtested", False):
                #     print_colored(f"\n⦿ {pair_id} ({pair_data['tf']}) [BELUM DIBACKTEST]", Fore.YELLOW, Style.BRIGHT)
                #     continue

                print_colored(f"\n⦿ {pair_id} ({pair_data['tf']}) {'[BT-OK]' if pair_data.get('backtested') else '[BT-PENDING]'}", Fore.WHITE, Style.BRIGHT) # Show backtest status
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN' and t.get('source') == 'LIVE'), None) # Only show live open positions
                pair_state = market_state.get(pair_id)
                
                if open_pos:
                    price = pair_state['candle_data'][-1]['close'] if pair_state and pair_state.get('candle_data') and pair_state['candle_data'] else open_pos['entryPrice']
                    pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
                    pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                    type_color = Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED
                    print_colored("  Status    : ", end=''); print_colored("POSITION OPEN", Fore.YELLOW, Style.BRIGHT)
                    print_colored("  Tipe      : ", end=''); print_colored(f"{open_pos.get('type')}", type_color, Style.BRIGHT)
                    print_colored("  Entry     : ", end=''); print_colored(f"{open_pos['entryPrice']:.4f}", Fore.WHITE)
                    print_colored("  Current   : ", end=''); print_colored(f"{price:.4f}", Fore.WHITE) # Display current price
                    print_colored("  PnL       : ", end=''); print_colored(f"{pnl:.2f}%", pnl_color, Style.BRIGHT)
                    
                    # Display new Trailing TP checkpoint status
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
                        # Display Caution Level
                        print_colored(f"  Caution Lv: {current_settings.get('caution_level'):.2f}", Fore.YELLOW)
                        # Display current actual tolerances for debug/info
                        caution_level_current = current_settings.get("caution_level", 0.5)
                        BASE_SOL_TOL_DASH = 0.05
                        MAX_ADD_SOL_TOL_DASH = 0.20
                        BASE_EMA_TOL_DASH = 0.00005
                        MAX_ADD_EMA_TOL_DASH = 0.0005
                        actual_sol_tol_display = BASE_SOL_TOL_DASH + (caution_level_current * MAX_ADD_SOL_TOL_DASH)
                        actual_ema_tol_display = BASE_EMA_TOL_DASH + (caution_level_current * MAX_ADD_EMA_TOL_DASH)
                        print_colored(f"  Solid. Tol: {actual_sol_tol_display:.3f} | EMA Tol: {actual_ema_tol_display:.6f}", Fore.CYAN)
                        
                        # Optionally display EMA9 status for debug/insight
                        if analysis.get('ema9_current') is not None and analysis.get('ema9_prev') is not None:
                            print_colored(f"  EMA9 Data : Prev Close {analysis['prev_candle_close']:.4f} vs EMA9 {analysis['ema9_prev']:.4f}", Fore.CYAN)
                            print_colored(f"              Current Close {analysis['current_candle_close']:.4f} vs EMA9 {analysis['ema9_current']:.4f}", Fore.CYAN)
                        # Display 3 previous candles data in dashboard for current analysis
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
    
    # Initial backtest check before starting threads
    # Use asyncio.get_event_loop for backtest execution if it's async
    # Ensure there is a running event loop, or create one for this part.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Iterate through a copy of watched_pairs to avoid modifying dict during iteration
    watched_pairs_copy = list(current_settings.get("watched_pairs", {}).keys()) 
    for pair_id in watched_pairs_copy:
        pair_data = current_settings["watched_pairs"][pair_id]
        if not pair_data.get("backtested", False):
            print_colored(f"\n[INIT] Pair {pair_id} belum dibacktest. Melakukan backtest...", Fore.YELLOW, Style.BRIGHT)
            try:
                # Call async backtest function
                backtest_success = loop.run_until_complete(
                    perform_backtest(pair_id, pair_data['tf'], current_settings.get("backtest_duration_months", 2))
                )
                if backtest_success:
                    current_settings["watched_pairs"][pair_id]["backtested"] = True
                    save_settings()
                    print_colored(f"[INIT] Backtest untuk {pair_id} selesai dan status diupdate.", Fore.GREEN)
                else:
                    print_colored(f"[INIT] Backtest untuk {pair_id} gagal atau tidak cukup data.", Fore.RED)
            except Exception as e:
                print_colored(f"[INIT] Error saat backtest {pair_id}: {e}", Fore.RED)
                # Keep backtested: false if error, so it can be re-attempted
            time.sleep(1) # Small pause after each backtest

    # Start the worker threads after initial backtests
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
                # Check if ALL watched pairs are backtested before entering dashboard
                all_backtested = all(pd.get("backtested", False) for pd in current_settings.get("watched_pairs", {}).values())
                if not all_backtested:
                     print_colored("Error: Beberapa pair di watchlist belum selesai dibacktest. Silakan coba '!start' lagi.", Fore.RED)
                     print_colored("Pastikan ada data historis yang cukup dan tidak ada error pada backtest.", Fore.RED)
                     # Re-run initial backtest check if needed
                     # Create a new event loop if the old one is closed (e.g., after an earlier run_until_complete call)
                     try:
                         loop = asyncio.get_event_loop()
                     except RuntimeError:
                         loop = asyncio.new_event_loop()
                         asyncio.set_event_loop(loop)

                     for pair_id_retry in list(current_settings.get("watched_pairs", {}).keys()): # Use a different loop var
                         pair_data_retry = current_settings["watched_pairs"][pair_id_retry]
                         if not pair_data_retry.get("backtested", False):
                            print_colored(f"[INIT] Mencoba lagi backtest untuk {pair_id_retry}...", Fore.YELLOW)
                            try:
                                backtest_success = loop.run_until_complete(
                                    perform_backtest(pair_id_retry, pair_data_retry['tf'], current_settings.get("backtest_duration_months", 2))
                                )
                                if backtest_success:
                                    current_settings["watched_pairs"][pair_id_retry]["backtested"] = True
                                    save_settings()
                                    print_colored(f"[INIT] Backtest untuk {pair_id_retry} selesai dan status diupdate.", Fore.GREEN)
                                else:
                                    print_colored(f"[INIT] Backtest untuk {pair_id_retry} gagal atau tidak cukup data.", Fore.RED)
                            except Exception as e:
                                print_colored(f"[INIT] Error saat backtest {pair_id_retry}: {e}", Fore.RED)
                            time.sleep(1)

                     all_backtested = all(pd.get("backtested", False) for pd in current_settings.get("watched_pairs", {}).values())
                     if not all_backtested: # Still not all backtested after retry
                         print_colored("Tidak semua pair berhasil dibacktest. Tidak bisa memulai live dashboard.", Fore.RED)
                         continue # Stay in command mode
                
                # If all backtested, proceed to live dashboard
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
                    for pair, pair_data in watched.items(): # Iterate over pair_data dict
                        backtest_status = "OK" if pair_data.get("backtested") else "PENDING"
                        print_colored(f"- {pair} (TF: {pair_data['tf']}) [BT: {backtest_status}]", Fore.WHITE)
            elif cmd == '!watch':
                if len(command_parts) >= 2:
                    pair_id = command_parts[1].upper()
                    tf = command_parts[2] if len(command_parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = {"tf": tf, "backtested": False} # Set backtested to False
                    save_settings()
                    print_colored(f"Pair {pair_id} dengan TF {tf} ditambahkan ke watchlist. Akan dibacktest saat '!start'.", Fore.GREEN)
                else: print_colored("Format salah. Gunakan: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(command_parts) == 2:
                    pair_id = command_parts[1].upper()
                    if pair_id in current_settings['watched_pairs']:
                        del current_settings['watched_pairs'][pair_id]
                        # When unwatching, also clean up from market_state if present
                        if pair_id in market_state: del market_state[pair_id]
                        # Remove relevant trades for this pair from autopilot_trades (optional, but good for cleanliness)
                        global autopilot_trades
                        autopilot_trades = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
                        save_trades() # Save trades after removal
                        save_settings()
                        print_colored(f"Pair {pair_id} dihapus dari watchlist dan riwayat tradesnya dibersihkan.", Fore.YELLOW)
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
