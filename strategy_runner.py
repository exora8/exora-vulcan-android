import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import asyncio
import math
import sys # Untuk progress bar

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"
REFRESH_INTERVAL_SECONDS = 0.5

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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, bt_dur)", Fore.GREEN) # Added bt_dur
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
        "backtest_duration_months": 2, # NEW: Default backtest duration
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
            print_colored(f"Error: Format JSON di '{TRADES_FILE}' tidak valid. Membuat file baru.", Fore.RED)
            autopilot_trades = []
            save_trades() # Save empty list to fix the file
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
# NOTE: This function is primarily for live data fetching for analysis in data_refresh_worker.
# For backtesting, we need to fetch a larger range, handled by _get_candles_for_range.
def fetch_bybit_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: 
                return None
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception: return None

# NEW: Function to fetch historical candles iteratively for backtesting
def _get_candles_for_range(instId, interval, start_ms, end_ms):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(interval, '60'); bybit_symbol = instId.replace('-', '')
    all_candles = []
    current_end_time = end_ms
    max_limit = 1000 # Max candles per request for Bybit kline

    # Convert interval to milliseconds for calculation
    interval_ms_map = {'1m': 60 * 1000, '3m': 3 * 60 * 1000, '5m': 5 * 60 * 1000, '15m': 15 * 60 * 1000, '30m': 30 * 60 * 1000,
                       '1H': 60 * 60 * 1000, '2H': 2 * 60 * 60 * 1000, '4H': 4 * 60 * 60 * 1000, '1D': 24 * 60 * 60 * 1000}
    interval_duration_ms = interval_ms_map.get(interval, 60 * 60 * 1000) # Default 1H

    while current_end_time > start_ms:
        try:
            # Calculate start time for this specific request, ensuring it doesn't go below overall start_ms
            request_start_time = max(start_ms, current_end_time - (max_limit * interval_duration_ms))

            url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit={max_limit}&end={current_end_time}&start={request_start_time}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0 and 'list' in data.get('result', {}):
                candles_batch = data['result']['list']
                if not candles_batch:
                    break # No more data for this range

                # Format and add to all_candles (list of lists, reverse to chronological later)
                # Bybit returns latest first, we want oldest first for backtesting
                formatted_candles = [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candles_batch]
                all_candles.extend(formatted_candles)
                
                # Update current_end_time to the timestamp of the oldest candle fetched in this batch
                current_end_time = int(candles_batch[-1][0]) 
                
                # If we requested 1000 and got less, it means we hit the start of available data for this symbol
                if len(candles_batch) < max_limit:
                    break
                
                time.sleep(0.1) # Respect rate limits
            else:
                print_colored(f"Error fetching historical data for {instId}: {data.get('retMsg', 'Unknown error')}", Fore.RED)
                break
        except requests.exceptions.RequestException as e:
            print_colored(f"Network error fetching historical data for {instId}: {e}", Fore.RED)
            break
        except Exception as e:
            print_colored(f"An unexpected error occurred while fetching historical data for {instId}: {e}", Fore.RED)
            break
            
    # Sort candles by timestamp to ensure chronological order (oldest first)
    all_candles.sort(key=lambda x: x['time'])
    
    # Filter to ensure they are strictly within the requested start_ms and end_ms
    final_filtered_candles = [c for c in all_candles if start_ms <= c['time'] <= end_ms]

    return final_filtered_candles

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# NEW: Progress Bar utility
def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, bar_length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        sys.stdout.write('\n')

# OTAK LOCAL AI (Updated for Exora Vulcan Sniper Entry with 3-candle snapshot and learning)
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
        if len(candle_data) < 100 + 3: 
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
        
        if not losing_trades:
            return False

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

            # 1. Bias match: Trend must be the same (STILL EXACT)
            if current_bias != past_snapshot['bias']:
                continue

            # 1.1. EMA Spread match (Tolerance for trend strength)
            if current_bias != "RANGING":
                current_spread = abs(current_ema50 - current_ema100)
                past_ema50 = past_snapshot.get('ema50', 0)
                past_ema100 = past_snapshot.get('ema100', 0)
                past_spread = abs(past_ema50 - past_ema100)
                
                if abs(current_spread - past_spread) > actual_ema_spread_tolerance:
                    continue
            else: 
                pass 

            # 2. EMA9 Cross state match: Entry trigger conditions must be similar (STILL EXACT)
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
            
            if len(current_pre_direction) < 3 or len(past_pre_direction) < 3:
                continue

            last_direction_matches = (current_pre_direction[-1] == past_pre_direction[-1])
            
            matching_directions_count = sum(1 for x, y in zip(current_pre_direction, past_pre_direction) if x == y)

            if not (last_direction_matches and matching_directions_count >= 2):
                continue

            # 4. Relaxed 3-Candle Solidity match: At least 2 out of 3 solidities must be within actual_solidity_tolerance
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
            else: # potential_trade_type == 'SHORT'
                return {"action": "SELL", "reason": "BEARISH TREND: Candle retrace dan close di bawah EMA9.", "snapshot": analysis}
        
        return {"action": "HOLD", "reason": f"Menunggu setup Exora Vulcan Sniper. Bias: {analysis['bias']}."}

# NEW: Helper to process an open trade within a backtest candle
def _process_open_trade_in_backtest(trade, current_candle, settings, autopilot_trades_ref):
    # This simulates SL/TP hits WITHIN a single candle
    entry_price = trade['entryPrice']
    trade_type = trade.get('type', 'LONG')
    
    sl_pct = settings.get('stop_loss_pct')
    activation_pct = settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = settings.get("trailing_tp_gap_pct", 0.05)
    fee_pct = settings.get('fee_pct', 0.1)

    # Calculate current PnL based on the candle's close for display/trailing purposes
    current_pnl = calculate_pnl(entry_price, current_candle['close'], trade_type)

    # Update run_up_percent and max_drawdown_percent
    if current_pnl > trade.get('run_up_percent', 0.0):
        trade['run_up_percent'] = current_pnl
    if current_pnl < trade.get('max_drawdown_percent', 0.0):
        trade['max_drawdown_percent'] = current_pnl

    # 1. Check Stop Loss (prioritize SL hit)
    sl_price = 0
    if trade_type == 'LONG':
        sl_price = entry_price * (1 - abs(sl_pct) / 100)
        # Check if low of current candle hit SL
        if current_candle['low'] <= sl_price:
            analyze_and_close_trade_sync(trade, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai (Backtest).", autopilot_trades_ref, settings)
            return True # Trade closed
    elif trade_type == 'SHORT':
        sl_price = entry_price * (1 + abs(sl_pct) / 100)
        # Check if high of current candle hit SL
        if current_candle['high'] >= sl_price:
            analyze_and_close_trade_sync(trade, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai (Backtest).", autopilot_trades_ref, settings)
            return True # Trade closed

    # 2. Check Trailing Take Profit (if SL not hit)
    current_tp_checkpoint_level = trade.get("current_tp_checkpoint_level", 0.0)

    # Activation Phase
    if current_tp_checkpoint_level == 0.0 and current_pnl >= activation_pct:
        trade['current_tp_checkpoint_level'] = activation_pct
        trade['trailing_stop_price'] = entry_price * (1 + activation_pct / 100) if trade_type == 'LONG' else \
                                       entry_price * (1 - activation_pct / 100)

    # Update Checkpoint Phase
    if current_tp_checkpoint_level > 0.0:
        steps_passed = math.floor((current_pnl - current_tp_checkpoint_level) / gap_pct)
        
        if steps_passed > 0:
            trade['current_tp_checkpoint_level'] += steps_passed * gap_pct
            trade['trailing_stop_price'] = entry_price * (1 + trade['current_tp_checkpoint_level'] / 100) if trade_type == 'LONG' else \
                                           entry_price * (1 - trade['current_tp_checkpoint_level'] / 100)
        
        # Exit Condition Phase
        if trade_type == 'LONG' and current_candle['close'] <= trade.get('trailing_stop_price', 0):
             # Check if price hit or went below trailing_stop_price
            analyze_and_close_trade_sync(trade, trade['trailing_stop_price'], f"Trailing TP (checkpoint {trade['current_tp_checkpoint_level']:.2f}%) tercapai (Backtest).", autopilot_trades_ref, settings)
            return True # Trade closed
        elif trade_type == 'SHORT' and current_candle['close'] >= trade.get('trailing_stop_price', float('inf')):
             # Check if price hit or went above trailing_stop_price
            analyze_and_close_trade_sync(trade, trade['trailing_stop_price'], f"Trailing TP (checkpoint {trade['current_tp_checkpoint_level']:.2f}%) tercapai (Backtest).", autopilot_trades_ref, settings)
            return True # Trade closed
    
    return False # Trade not closed

# NEW: Synchronous version of analyze_and_close_trade for backtesting
def analyze_and_close_trade_sync(trade, exit_price, close_trigger_reason, autopilot_trades_ref, settings):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    # In backtest, we don't have a live market_state snapshot from external source.
    # We can pass dummy data or simply omit the exit_snapshot for backtest closures if not critical for future learning.
    # For now, let's omit the exit_snapshot, as its primary use is for manual review of live closed trades.
    # Or, we could pass the candle_data_buffer to LocalAI to generate a valid snapshot for backtest.
    # For simplicity, we won't generate exit_snapshot for backtest closed trades, to keep it minimal.
    
    trade.update({
        'status': 'CLOSED', 
        'exitPrice': exit_price, 
        'exitTimestamp': datetime.fromtimestamp(trade['time'] / 1000).isoformat() if 'time' in trade else datetime.now().isoformat(), # Use candle time if available
        'pl_percent': pnl
        # entry_snapshot is kept if trade is loss, otherwise deleted. Handled at end of backtest.
    })
    
    # Mark trade in the reference list as closed.
    # In this backtesting setup, we modify the existing trade object reference in autopilot_trades_ref directly.
    # No need to remove and re-add.

# NEW: Main backtesting function for a single pair
def perform_backtest_for_pair(instrument_id, timeframe, duration_months):
    global autopilot_trades # Access global list to add new trades

    end_datetime = datetime.now()
    start_datetime = end_datetime - timedelta(days=duration_months * 30) # Approximate months to days
    
    start_ms = int(start_datetime.timestamp() * 1000)
    end_ms = int(end_datetime.timestamp() * 1000)

    print_colored(f"  Fetching historical data for {instrument_id} ({timeframe}) from {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}...", Fore.CYAN)
    historical_candles = _get_candles_for_range(instrument_id, timeframe, start_ms, end_ms)
    
    if not historical_candles:
        print_colored(f"  Tidak ada data historis yang cukup untuk {instrument_id}.", Fore.YELLOW)
        return

    print_colored(f"  Mulai backtest untuk {instrument_id} ({len(historical_candles)} candles)...", Fore.CYAN)

    backtest_local_brain = LocalAI(current_settings, []) # Start with empty past trades for backtest brain
    backtest_open_position = None
    candle_data_buffer = [] # Buffer to simulate market_state's candle_data for analysis
    
    # Store initial count of trades to only save new ones (or modified ones)
    initial_trades_count = len([t for t in autopilot_trades if t['instrumentId'] == instrument_id])
    
    # Temporarily store new/modified trades from this backtest to append later
    backtest_trades_results = []
    
    min_candles_for_analysis = 100 + 3 # Requirement for LocalAI.get_market_analysis

    for i, candle in enumerate(historical_candles):
        _print_progress_bar(i + 1, len(historical_candles), prefix=f'    Progress {instrument_id}:', suffix=f'({i+1}/{len(historical_candles)} candles)', bar_length=30)
        
        candle_data_buffer.append(candle)
        if len(candle_data_buffer) > 300: # Keep buffer size reasonable, e.g., max 300 candles for EMA calc
            candle_data_buffer.pop(0) # Remove oldest candle

        if len(candle_data_buffer) < min_candles_for_analysis:
            continue # Not enough data for initial analysis

        # Use a shallow copy of autopilot_trades to simulate the "current" past_trades for the local_brain
        # This allows the backtest_local_brain to learn from trades closed within this very backtest.
        trades_for_brain = [t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'CLOSED'] + backtest_trades_results

        # Update the brain's past_trades with the latest closed trades
        backtest_local_brain.past_trades = trades_for_brain


        if backtest_open_position:
            # Pass autopilot_trades directly so _process_open_trade_in_backtest can modify the global list
            is_closed = _process_open_trade_in_backtest(backtest_open_position, candle, current_settings, autopilot_trades)
            if is_closed:
                # Add a unique identifier for this particular trade closure snapshot
                backtest_open_position['time'] = candle['time'] # Store candle time of closure for history display
                backtest_trades_results.append(backtest_open_position) # Add to results
                backtest_open_position = None # Reset open position
        
        if not backtest_open_position: # Only look for new entry if no position is open
            analysis = backtest_local_brain.get_market_analysis(candle_data_buffer)
            if analysis:
                decision = backtest_local_brain.get_decision(candle_data_buffer, None, instrument_id)
                action = decision.get('action', 'HOLD').upper()
                reason = decision.get('reason', 'No reason provided.')
                
                if action in ["BUY", "SELL"]:
                    trade_type = "LONG" if action == "BUY" else "SHORT"
                    new_trade = {
                        "id": int(candle['time']), # Use candle time as ID for backtest uniqueness
                        "instrumentId": instrument_id,
                        "type": trade_type,
                        "entryTimestamp": datetime.fromtimestamp(candle['time'] / 1000).isoformat(),
                        "entryPrice": candle['close'],
                        "entryReason": reason,
                        "status": 'OPEN',
                        "entry_snapshot": decision.get("snapshot"),
                        "run_up_percent": 0.0,
                        "max_drawdown_percent": 0.0,
                        "trailing_stop_price": None,
                        "current_tp_checkpoint_level": 0.0
                    }
                    backtest_open_position = new_trade # Set this as the open position for the backtest


    # After loop, if there's any open position, close it with current_candle['close']
    if backtest_open_position:
        analyze_and_close_trade_sync(backtest_open_position, historical_candles[-1]['close'], "Closed at end of backtest (Backtest).", autopilot_trades, current_settings)
        backtest_open_position['time'] = historical_candles[-1]['time']
        backtest_trades_results.append(backtest_open_position)

    # Add all new/modified trades from this backtest to the global autopilot_trades list
    # Filter out initial trades to avoid duplicates from existing trades.json
    existing_pair_trades = {t['id'] for t in autopilot_trades if t['instrumentId'] == instrument_id}
    newly_added_count = 0
    for trade_result in backtest_trades_results:
        # Before adding, ensure entry_snapshot is deleted if profitable, as per live logic
        if trade_result.get('pl_percent', 0) > current_settings.get('fee_pct', 0.1) and 'entry_snapshot' in trade_result:
            del trade_result['entry_snapshot']
        
        # Only add if it's not already in the global list (e.g., if trade ID based on timestamp might clash)
        if trade_result['id'] not in existing_pair_trades:
            autopilot_trades.append(trade_result)
            newly_added_count += 1
            
    # Sort the global trades list by timestamp after adding
    autopilot_trades.sort(key=lambda t: t.get('entryTimestamp', '0').replace('Z', ''))
    
    save_trades()
    print_colored(f"  Backtest untuk {instrument_id} selesai. {newly_added_count} trade baru ditambahkan. Total trades untuk {instrument_id}: {len([t for t in autopilot_trades if t['instrumentId'] == instrument_id])}", Fore.GREEN)


# --- LOGIKA TRADING UTAMA ---
# Existing async analyze_and_close_trade for live trading
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f} | Trigger: {close_trigger_reason}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    pair_state = market_state.get(instrument_id)
    
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3:
        return

    is_ai_thinking = True
    try:
        candle_data = pair_state["candle_data"]
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        local_brain = LocalAI(current_settings, relevant_trades)
        
        decision = local_brain.get_decision(candle_data, open_position, instrument_id)
        
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close']
        
        if action in ["BUY", "SELL"] and not open_position:
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
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
            notif_content = f"{instrument_id}: Entry @ {current_price:.4f} | {reason}"
            send_termux_notification(notif_title, notif_content)
    except Exception as e:
        print_colored(f"Error in autopilot analysis for {instrument_id}: {e}", Fore.RED)
        is_autopilot_in_cooldown[instrument_id] = True
        await asyncio.sleep(60)
        is_autopilot_in_cooldown[instrument_id] = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    asyncio.run(run_autopilot_analysis(pair_id))
                    time.sleep(0.1)
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

async def check_realtime_position_management(instrument_id, latest_price):
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl
        
    sl_pct = current_settings.get('stop_loss_pct')
    if sl_pct is not None and current_pnl <= -abs(sl_pct):
        global is_ai_thinking
        if not is_ai_thinking:
            is_ai_thinking = True
            await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.")
            is_ai_thinking = False
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
            if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, open_position['trailing_stop_price'], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.")
                is_ai_thinking = False
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
             if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, open_position['trailing_stop_price'], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.")
                is_ai_thinking = False
    
    save_trades()


def data_refresh_worker():
    global market_state
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
                        asyncio.run(check_realtime_position_management(pair_id, latest_price))
                time.sleep(0.5)
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''),
        'bt_dur': ('backtest_duration_months', ' bulan') # NEW: for backtest duration
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
            if value < 0: print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
        except ValueError: print_colored(f"Error: Nilai '{parts[2]}' harus berupa angka.", Fore.RED); return
        key_full, unit = setting_map[key_short]
        current_settings[key_full] = value; save_settings()
        print_colored(f"Pengaturan '{key_full}' berhasil diubah menjadi {value}{unit}.", Fore.GREEN, Style.BRIGHT); return
    print_colored("Format salah. Gunakan '!settings' atau '!set <key> <value>'.", Fore.RED)

def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="")
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
                # Identify pairs to backtest: those in watchlist without any trade history
                pairs_to_backtest = []
                existing_pair_trades_ids = {t['instrumentId'] for t in autopilot_trades}
                
                print_colored("\nMemeriksa watchlist untuk backtest awal...", Fore.CYAN)
                for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                    # Check if this pair has NO trade history at all in the trades.json
                    if pair_id not in existing_pair_trades_ids:
                        pairs_to_backtest.append((pair_id, timeframe))
                
                if pairs_to_backtest:
                    print_colored(f"Mulai backtest awal untuk {len(pairs_to_backtest)} pair baru...", Fore.CYAN, Style.BRIGHT)
                    for pair_id, timeframe in pairs_to_backtest:
                        try:
                            # Pass relevant settings to backtest function
                            perform_backtest_for_pair(pair_id, timeframe, current_settings.get("backtest_duration_months", 2))
                        except Exception as e:
                            print_colored(f"  Gagal backtest untuk {pair_id}: {e}", Fore.RED)
                    print_colored("\nBacktest awal selesai.", Fore.GREEN, Style.BRIGHT)
                    time.sleep(1) # Give user time to read

                # Now proceed with live dashboard
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
                        # Remove existing trades for this pair from global autopilot_trades list
                        global autopilot_trades
                        autopilot_trades = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
                        save_trades()
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
