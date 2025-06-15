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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, bt_dur)", Fore.GREEN) # Updated help text
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
        "backtest_duration_months": 6,     # NEW: Default backtest duration in months
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
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
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
# MODIFIED: fetch_bybit_candle_data to include historical fetching via multiple calls
async def fetch_historical_kline_data(instId, timeframe, start_timestamp_ms):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    
    all_candles = []
    end_time = int(time.time() * 1000) # Start from now (latest data)
    
    # Bybit limit is 1000 candles per request
    # Max candles needed for EMA100 and snapshot is ~103, but for backtesting we need much more.
    # We fetch data in chunks backward in time until start_timestamp_ms is met.
    
    while end_time > start_timestamp_ms:
        try:
            url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=1000&endTime={end_time}"
            response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
            
            if data.get("retCode") == 0 and 'list' in data.get('result', {}):
                candle_list = data['result']['list']
                if not candle_list:
                    break # No more data available
                
                parsed_candles = [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list]
                all_candles.extend(parsed_candles)
                
                # Set end_time to the timestamp of the oldest candle fetched, for the next request
                end_time = parsed_candles[-1]['time'] - 1 # Subtract 1ms to avoid overlap
                
                # Add a small delay to respect API rate limits
                time.sleep(0.05) 
            else:
                print_colored(f"Error fetching historical data for {instId}: {data.get('retMsg', 'Unknown error')}", Fore.RED)
                break
        except requests.exceptions.RequestException as e:
            print_colored(f"Network error fetching historical data for {instId}: {e}", Fore.RED)
            break
        except Exception as e:
            print_colored(f"Unexpected error fetching historical data for {instId}: {e}", Fore.RED)
            break
            
    # Sort candles by time in ascending order (oldest to newest)
    all_candles.sort(key=lambda x: x['time'])
    
    # Filter out candles older than start_timestamp_ms if any overlap occurred
    final_candles = [c for c in all_candles if c['time'] >= start_timestamp_ms]

    # Ensure enough data for EMA100 + 3 candles for analysis during backtest
    if len(final_candles) < 100 + 3: 
        print_colored(f"Warning: Not enough historical data for {instId} ({len(final_candles)} candles) to perform a proper backtest up to EMA100+3. Backtest might be limited.", Fore.YELLOW)
        return None # Return None if data is severely lacking for analysis
        
    return final_candles

# Existing function, but renamed/adapted for clarity
# This function is used by the data_refresh_worker for live market data
def fetch_latest_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300" # Limit 300 is fine for live updates
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: # Still need enough for basic analysis on live data
                return None 
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

# --- LOGIKA TRADING UTAMA (Live & Backtest) ---
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, is_backtest=False):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    # Ensure candle_data exists for the pair when closing (only relevant for live/latest data)
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    # Keep entry_snapshot only if trade is a loss, for learning purposes
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    
    save_trades() # Save trades immediately after closing a position

    if not is_backtest: # Only send notifications for live trades
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f} | Trigger: {close_trigger_reason}"
        send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    # This is primarily for live analysis, backtest will run separately
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    
    pair_state = market_state.get(instrument_id)
    # Check if there are enough candles for full analysis based on get_market_analysis requirements
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3:
        return # Skip analysis if data insufficient

    is_ai_thinking = True
    try:
        candle_data = pair_state["candle_data"]
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        local_brain = LocalAI(current_settings, relevant_trades)
        
        decision = local_brain.get_decision(candle_data, open_position, instrument_id)
        
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close'] # Entry price is the close of the trigger candle
        
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

# NEW: Backtesting function
async def run_backtest_for_pair(instrument_id, timeframe):
    global autopilot_trades # Access global trade list to add backtested trades
    
    # Check if pair already has trades in trades.json
    existing_trades_for_pair = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
    if existing_trades_for_pair:
        print_colored(f"Skipping backtest for {instrument_id}: Already has {len(existing_trades_for_pair)} existing trades in trades.json.", Fore.YELLOW)
        return

    print_colored(f"\nStarting backtest for {instrument_id} ({timeframe})...", Fore.CYAN, Style.BRIGHT)
    
    bt_duration_months = current_settings.get("backtest_duration_months", 6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=bt_duration_months * 30) # Approx days in months
    start_timestamp_ms = int(start_date.timestamp() * 1000)

    historical_candles = await fetch_historical_kline_data(instrument_id, timeframe, start_timestamp_ms)
    
    if not historical_candles:
        print_colored(f"Failed to get sufficient historical data for {instrument_id}. Skipping backtest.", Fore.RED)
        return

    print_colored(f"Fetched {len(historical_candles)} historical candles for {instrument_id}. Running simulation...", Fore.GREEN)
    
    # Temporary state for this backtest run
    current_open_position = None
    backtest_trade_count = 0
    
    # Need to keep a rolling window of candles for EMA calculation within the loop
    # Minimum for EMA100 + 3 for snapshot is 103 candles.
    rolling_candles = [] 
    MIN_CANDLES_FOR_ANALYSIS = 100 + 3 # Needs to match LocalAI.get_market_analysis

    # Pass the current settings to LocalAI for consistent behavior during backtest
    local_brain = LocalAI(current_settings, [t for t in autopilot_trades if t['instrumentId'] == instrument_id]) # Only pass relevant trades for "learning"

    for i, candle in enumerate(historical_candles):
        rolling_candles.append(candle)
        if len(rolling_candles) > 300: # Keep a reasonable window size (e.g., 300 for Bybit's default limit)
            rolling_candles.pop(0) 

        # Ensure enough candles for analysis
        if len(rolling_candles) < MIN_CANDLES_FOR_ANALYSIS:
            continue # Not enough data yet to perform analysis

        # Simulate market state for this candle
        temp_market_state_for_backtest = {
            instrument_id: {"candle_data": rolling_candles, "analysis": local_brain.get_market_analysis(rolling_candles)}
        }
        
        # Simulate check_realtime_position_management for current open position
        if current_open_position:
            # Need to create a mutable copy to update
            current_open_position_copy = current_open_position.copy()
            # Pass is_backtest=True to analyze_and_close_trade to suppress notifications
            await check_realtime_position_management_backtest(current_open_position_copy, candle['close'], is_backtest=True)
            # Update the global autopilot_trades with the closed trade
            if current_open_position_copy['status'] == 'CLOSED':
                # Replace the original open trade with the closed one (if it was added globally as open)
                for j, t in enumerate(autopilot_trades):
                    if t['id'] == current_open_position_copy['id']:
                        autopilot_trades[j] = current_open_position_copy
                        break
                current_open_position = None # No open position anymore
        
        # Simulate decision making for new entry
        decision = local_brain.get_decision(rolling_candles, current_open_position, instrument_id)
        
        action = decision.get('action', 'HOLD').upper()
        reason = decision.get('reason', 'No reason provided.')
        
        if action in ["BUY", "SELL"] and not current_open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade_id = int(candle['time'] / 1000) # Use candle time as ID for backtest
            new_trade = {
                "id": new_trade_id, # Using candle time for unique ID in backtest
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
            autopilot_trades.append(new_trade)
            current_open_position = new_trade # Set the current_open_position for backtest
            backtest_trade_count += 1
            # print_colored(f"DEBUG: Backtest - Opened {trade_type} at {candle['close']} on {datetime.fromtimestamp(candle['time']/1000).strftime('%Y-%m-%d %H:%M')}", Fore.BLUE)

        # Print progress every 1000 candles or so
        if (i + 1) % 1000 == 0:
            print_colored(f"  ... Processing candle {i+1}/{len(historical_candles)} for {instrument_id}", Fore.BLUE)

    if current_open_position: # Close any remaining open positions at the end of backtest
        # Pass is_backtest=True to analyze_and_close_trade to suppress notifications
        await analyze_and_close_trade(current_open_position, historical_candles[-1]['close'], "Backtest End", is_backtest=True)
        # Update the trade status in the global list if it was the last one
        for j, t in enumerate(autopilot_trades):
            if t['id'] == current_open_position['id']:
                autopilot_trades[j] = current_open_position
                break

    save_trades() # Save all trades after backtest is complete
    print_colored(f"Backtest for {instrument_id} completed. Total {backtest_trade_count} trades simulated.", Fore.GREEN, Style.BRIGHT)

# Modified check_realtime_position_management to be callable by backtest (without global market_state dependency)
async def check_realtime_position_management_backtest(open_position, latest_price, is_backtest=False):
    # This function operates on the passed open_position object directly,
    # allowing the backtest to simulate management. It will modify open_position in place.
    
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl
        
    sl_pct = current_settings.get('stop_loss_pct')
    if sl_pct is not None and current_pnl <= -abs(sl_pct):
        await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.", is_backtest=is_backtest)
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
            await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest=is_backtest)
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
             await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest=is_backtest)

    # Save trades is handled by analyze_and_close_trade, or implicitly by the backtest loop adding trades.
    # No need for save_trades() here, it causes too many writes during backtest.


# --- THREAD WORKERS (Live Trading) ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running: # Only run if autopilot is active (after backtest)
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    asyncio.run(run_autopilot_analysis(pair_id))
                    time.sleep(0.1) 
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        if is_autopilot_running: # Only fetch live data if autopilot is active
            watched_pairs = current_settings.get("watched_pairs", {})
            if watched_pairs:
                for pair_id, timeframe in watched_pairs.items():
                    data = fetch_latest_candle_data(pair_id, timeframe) # Use fetch_latest_candle_data for live
                    if data: 
                        if len(data) >= 100 + 3: 
                            analysis = LocalAI(current_settings, []).get_market_analysis(data)
                        else:
                            analysis = None 
                        market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                        
                        # Live position management uses the global market_state and is_ai_thinking
                        # But is_ai_thinking is only set by run_autopilot_analysis or analyze_and_close_trade
                        # which are not called by this worker directly.
                        # It's better to ensure this call doesn't conflict.
                        # For simplicity, we assume this doesn't conflict heavily or is covered by is_ai_thinking global lock.
                        asyncio.run(check_realtime_position_management(pair_id, data[-1]['close']))
                    time.sleep(0.5) 
            stop_event.wait(REFRESH_INTERVAL_SECONDS) 
        else: time.sleep(1)


def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 
        'bt_dur': ('backtest_duration_months', ' bulan') # NEW: Backtest duration setting
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
            if key_short == 'caution':
                if not (0.0 <= value <= 1.0):
                    print_colored("Error: Nilai 'caution' harus antara 0.0 dan 1.0.", Fore.RED); return
            elif key_short == 'bt_dur': # Specific validation for backtest duration
                if not (1 <= value <= 24): # Example: between 1 and 24 months
                    print_colored("Error: Durasi backtest harus antara 1 dan 24 bulan.", Fore.RED); return
            elif value < 0: # General check for other positive settings
                 print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
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
                # --- NEW: Backtesting Before Starting Live Autopilot ---
                print_colored("\n--- Memulai Backtesting (Jika Perlu) ---", Fore.CYAN, Style.BRIGHT)
                # Ensure live threads are paused/inactive during backtest
                is_autopilot_running = False 
                
                watched_pairs = current_settings.get("watched_pairs", {})
                if not watched_pairs:
                    print_colored("Error: Watchlist kosong. Gunakan '!watch <PAIR>' dulu sebelum !start.", Fore.RED)
                    continue

                # Run backtest for each pair in watchlist that doesn't have existing trades
                for pair_id, timeframe in watched_pairs.items():
                    # Check if this pair already has trades recorded (case-sensitive)
                    has_existing_trades = any(t['instrumentId'] == pair_id for t in autopilot_trades)
                    
                    if not has_existing_trades:
                        print_colored(f"  Backtesting {pair_id} ({timeframe})...", Fore.YELLOW)
                        # Run backtest asynchronously
                        asyncio.run(run_backtest_for_pair(pair_id, timeframe))
                    else:
                        print_colored(f"  {pair_id} sudah memiliki riwayat trade. Melewatkan backtest.", Fore.GREEN)
                
                print_colored("\n--- Backtesting Selesai ---", Fore.CYAN, Style.BRIGHT)
                # END NEW BACKTESTING BLOCK
                
                is_autopilot_running = True # Activate live autopilot after backtest
                print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN, Style.BRIGHT)
                time.sleep(1); run_dashboard_mode()
                is_autopilot_running = False # Deactivate when exiting dashboard
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
                        # Optionally, remove trades for this pair from global list / trades.json
                        # autopilot_trades = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
                        # save_trades() # Uncomment if you want to clear its history too
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
