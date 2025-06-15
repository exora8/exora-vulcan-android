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
REFRESH_INTERVAL_SECONDS = 0.5 # Interval refresh data live

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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, bt_months)", Fore.GREEN) # Added bt_months
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
        "backtest_duration_months": 2, # NEW: Default 2 months for backtest
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
            print_colored(f"Error: Format JSON di '{TRADES_FILE}' tidak valid. Mengosongkan riwayat trade.", Fore.RED)
            autopilot_trades = [] # Reset if corrupt
    else:
        autopilot_trades = []

    # Ensure new fields for trailing TP checkpoint are initialized if loading old trades
    for trade in autopilot_trades:
        if 'current_tp_checkpoint_level' not in trade:
            trade['current_tp_checkpoint_level'] = 0.0
        # For backtest, ensure entry_snapshot is not removed for profit trades until after full analysis by LocalAI
        # This part of cleanup is handled by analyze_and_close_trade
        
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
def fetch_bybit_candle_data(instId, timeframe, limit=300):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            # We need enough data for EMA100 and at least 4 candles for trigger + 3 pre-entry snapshot
            if len(candle_list) < 100 + 3: 
                return None # Not enough data
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception as e: 
        print_colored(f"Error fetching live data for {instId}: {e}", Fore.RED)
        return None

# NEW: Function to fetch historical candles for a specific range
def fetch_candles_in_range(instId, timeframe, start_time_ms, end_time_ms):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60')
    bybit_symbol = instId.replace('-', '')
    all_candles = []
    current_end_time = end_time_ms
    max_limit = 1000 # Max candles per request

    print_colored(f"  Fetching historical data for {instId} ({timeframe})...", Fore.YELLOW)

    while True:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit={max_limit}&endTime={current_end_time}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0 and 'list' in data.get('result', {}):
                candles_batch = data['result']['list']
                if not candles_batch:
                    break # No more candles

                # Convert to desired format and reverse to get oldest first
                formatted_batch = [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candles_batch][::-1]
                
                # Add only candles within the desired range [start_time_ms, end_time_ms]
                new_candles = [c for c in formatted_batch if c['time'] >= start_time_ms]
                all_candles.extend(new_candles)
                
                # Update current_end_time to the timestamp of the oldest candle in this batch
                current_end_time = candles_batch[-1][0] # Oldest candle in the batch

                if int(current_end_time) < start_time_ms:
                    break # Reached or passed the start_time

                # Small delay to avoid hitting rate limits
                time.sleep(0.05) 
            else:
                print_colored(f"  API Error fetching historical data for {instId}: {data.get('retMsg', 'Unknown error')}", Fore.RED)
                break
        except Exception as e:
            print_colored(f"  Network error fetching historical data for {instId}: {e}", Fore.RED)
            break
    
    # Sort all_candles by timestamp to ensure chronological order
    all_candles.sort(key=lambda x: x['time'])

    # Filter to ensure only candles within exact range are kept (from all_candles.extend)
    all_candles = [c for c in all_candles if c['time'] >= start_time_ms and c['time'] <= end_time_ms]

    print_colored(f"  Fetched {len(all_candles)} candles for {instId}.", Fore.GREEN)
    return all_candles

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
            if current_bias != "RANGING":
                current_spread = abs(current_ema50 - current_ema100)
                past_ema50 = past_snapshot.get('ema50', 0)
                past_ema100 = past_snapshot.get('ema100', 0)
                past_spread = abs(past_ema50 - past_ema100)
                
                if abs(current_spread - past_spread) > actual_ema_spread_tolerance:
                    continue
            else: # If bias is RANGING, EMA spread comparison is not applicable in this context.
                  # We simply pass this check, assuming RANGING has no specific spread pattern to compare.
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
                continue # Cannot compare if data is missing

            last_direction_matches = (current_pre_direction[-1] == past_pre_direction[-1]) 
            
            matching_directions_count = sum(1 for x, y in zip(current_pre_direction, past_pre_direction) if x == y)

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
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, is_backtest=False):
    # This function is now used by both live trading and backtesting
    # We pass market_state directly for backtest context
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        # Pass appropriate market_state context to LocalAI
        # For backtest, market_state['candle_data'] will be the historical data *up to current candle*
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    
    # Save trades only once after all backtests or as they happen in live
    # If is_backtest, we will save all trades at the end of backtest run_pair_backtest
    if not is_backtest:
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
                "entry_snapshot": decision.get("snapshot"), # Snapshot contains EMA values + 3 candle info
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

async def check_realtime_position_management(instrument_id, latest_price, is_backtest=False):
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
        if not is_ai_thinking: # Avoid race condition if another thread is closing
            is_ai_thinking = True
            await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.", is_backtest)
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
                await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest)
                is_ai_thinking = False
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
             if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest)
                is_ai_thinking = False
    
    if not is_backtest:
        save_trades() # Save only in live context

def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, timeframe in watched_pairs.items():
                data = fetch_bybit_candle_data(pair_id, timeframe) # This fetches LIVE data
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

# NEW: Backtesting Logic
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print_colored(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total: 
        print()

def run_pair_backtest(pair_id, timeframe, duration_months):
    global autopilot_trades # Need to modify global list
    print_colored(f"\n🚀 Memulai Backtest untuk {pair_id} ({timeframe})...", Fore.CYAN, Style.BRIGHT)

    # 1. Calculate time range
    end_time_dt = datetime.now()
    start_time_dt = end_time_dt - timedelta(days=duration_months * 30) # Approximate months
    end_time_ms = int(end_time_dt.timestamp() * 1000)
    start_time_ms = int(start_time_dt.timestamp() * 1000)

    # 2. Fetch all historical data for the period
    historical_candles = fetch_candles_in_range(pair_id, timeframe, start_time_ms, end_time_ms)

    if not historical_candles or len(historical_candles) < 100 + 3: # Need minimum candles for EMA
        print_colored(f"  Gagal backtest {pair_id}: Tidak cukup data historis atau gagal fetching.", Fore.RED)
        return False

    # Filter out existing trades for this pair to avoid re-adding
    initial_trades_for_pair = [t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'CLOSED']
    backtested_trades_for_pair = []
    open_backtest_trades = [] # Keep track of trades currently open during backtest

    total_candles = len(historical_candles)
    print_colored(f"  Menganalisis {total_candles} candle...", Fore.YELLOW)

    # Simulate candles one by one
    for i in range(len(historical_candles)):
        current_historical_data = historical_candles[:i+1] # All candles up to current point
        current_candle = historical_candles[i]
        
        # Only analyze if we have enough candles for EMA100 + trigger
        if len(current_historical_data) < 100 + 3:
            print_progress_bar(i + 1, total_candles, prefix=f'  {pair_id} Analisis', suffix='Lengkap', length=50, fill='=')
            continue

        # Update market_state for LocalAI's analysis
        # This simulates the bot seeing the market up to this historical candle
        market_state[pair_id] = {"candle_data": current_historical_data, 
                                 "analysis": LocalAI(current_settings, initial_trades_for_pair + backtested_trades_for_pair).get_market_analysis(current_historical_data)}

        # Check existing open backtest trades for SL/TP
        trades_to_remove = []
        for open_bt_trade in open_backtest_trades:
            # For backtesting, check high and low of the current candle for SL/TP hit
            current_price_for_check = current_candle['close'] # Use close for latest PnL calculation
            hit_price = current_price_for_check # Default, if SL/TP not hit within candle
            
            # Check SL for both LONG and SHORT
            sl_price = open_bt_trade['entryPrice'] * (1 - current_settings['stop_loss_pct'] / 100) if open_bt_trade['type'] == 'LONG' else \
                       open_bt_trade['entryPrice'] * (1 + current_settings['stop_loss_pct'] / 100)
            
            sl_hit = False
            if open_bt_trade['type'] == 'LONG' and current_candle['low'] <= sl_price:
                sl_hit = True; hit_price = sl_price
            elif open_bt_trade['type'] == 'SHORT' and current_candle['high'] >= sl_price:
                sl_hit = True; hit_price = sl_price
            
            if sl_hit:
                # Need to use asyncio.run because analyze_and_close_trade is async
                asyncio.run(analyze_and_close_trade(open_bt_trade, hit_price, "Backtest SL", is_backtest=True))
                trades_to_remove.append(open_bt_trade)
                continue # Skip TP check if SL hit

            # Check Trailing TP for both LONG and SHORT (if not already hit SL)
            # We need to simulate the trailing TP moving. The check_realtime_position_management
            # already handles the logic for current_tp_checkpoint_level and trailing_stop_price.
            # We must pass the is_backtest=True flag.
            # Use high/low for TP hit in backtest
            tp_hit = False
            if open_bt_trade.get('trailing_stop_price') is not None:
                if open_bt_trade['type'] == 'LONG' and current_candle['low'] <= open_bt_trade['trailing_stop_price']:
                    tp_hit = True; hit_price = open_bt_trade['trailing_stop_price']
                elif open_bt_trade['type'] == 'SHORT' and current_candle['high'] >= open_bt_trade['trailing_stop_price']:
                    tp_hit = True; hit_price = open_bt_trade['trailing_stop_price']
            
            # This is critical: if TP condition is met or updated *within* the candle, update the trade object.
            # We call it even if TP not immediately hit to update trailing_stop_price
            asyncio.run(check_realtime_position_management(open_bt_trade['instrumentId'], current_candle['high'] if open_bt_trade['type'] == 'LONG' else current_candle['low'], is_backtest=True))
            
            # If the trade was closed by the above call to check_realtime_position_management (which calls analyze_and_close_trade), then it will be removed
            if open_bt_trade['status'] == 'CLOSED':
                 trades_to_remove.append(open_bt_trade)
                 continue


        # Remove closed trades
        for trade_removed in trades_to_remove:
            open_backtest_trades.remove(trade_removed)
            backtested_trades_for_pair.append(trade_removed) # Add to list of completed backtested trades


        # Get decision for new entry (only if no open position for this pair during backtest)
        if not any(t for t in open_backtest_trades if t['instrumentId'] == pair_id):
            local_brain_for_decision = LocalAI(current_settings, initial_trades_for_pair + backtested_trades_for_pair)
            decision = local_brain_for_decision.get_decision(current_historical_data, None, pair_id) # No open position
            
            if decision.get('action') in ["BUY", "SELL"]:
                trade_type = "LONG" if decision['action'] == "BUY" else "SHORT"
                new_trade = {
                    "id": int(current_candle['time'] / 1000), # Use candle time for backtest ID
                    "instrumentId": pair_id,
                    "type": trade_type,
                    "entryTimestamp": datetime.fromtimestamp(current_candle['time'] / 1000).isoformat(),
                    "entryPrice": current_candle['close'],
                    "entryReason": decision.get("reason"),
                    "status": 'OPEN',
                    "entry_snapshot": decision.get("snapshot"),
                    "run_up_percent": 0.0,
                    "max_drawdown_percent": 0.0,
                    "trailing_stop_price": None,
                    "current_tp_checkpoint_level": 0.0
                }
                open_backtest_trades.append(new_trade)


        print_progress_bar(i + 1, total_candles, prefix=f'  {pair_id} Analisis', suffix='Lengkap', length=50, fill='=')

    # Close any remaining open trades at the end of backtest period
    for open_bt_trade in open_backtest_trades:
        asyncio.run(analyze_and_close_trade(open_bt_trade, historical_candles[-1]['close'], "Backtest End", is_backtest=True))
        backtested_trades_for_pair.append(open_bt_trade)

    # Add all newly backtested trades to the global list
    autopilot_trades.extend(backtested_trades_for_pair)
    autopilot_trades.sort(key=lambda x: x['entryTimestamp']) # Ensure sorted by time
    save_trades() # Save all trades after backtest
    
    print_colored(f"✅ Backtest untuk {pair_id} selesai. {len(backtested_trades_for_pair)} trade disimulasikan dan ditambahkan.", Fore.GREEN, Style.BRIGHT)
    return True


def check_and_run_backtests():
    watched_pairs = current_settings.get("watched_pairs", {})
    backtest_months = current_settings.get("backtest_duration_months", 2)
    
    pairs_to_backtest = []
    
    for pair_id, timeframe in watched_pairs.items():
        # Find the oldest trade for this pair
        oldest_trade_date = None
        for trade in autopilot_trades:
            if trade['instrumentId'] == pair_id:
                trade_date = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', ''))
                if oldest_trade_date is None or trade_date < oldest_trade_date:
                    oldest_trade_date = trade_date
        
        # Calculate the required backtest start date
        required_start_date = datetime.now() - timedelta(days=backtest_months * 30)

        # If no trades for this pair, or the oldest trade is newer than required, need backtest
        if oldest_trade_date is None or oldest_trade_date > required_start_date:
            pairs_to_backtest.append((pair_id, timeframe))
    
    if pairs_to_backtest:
        print_colored("\nMemerlukan Backtest untuk pembelajaran AI:", Fore.CYAN, Style.BRIGHT)
        for pair_id, timeframe in pairs_to_backtest:
            print_colored(f"- {pair_id} ({timeframe}) selama {backtest_months} bulan terakhir.", Fore.YELLOW)
        
        print_colored("Memulai proses Backtest. Mohon tunggu...", Fore.BLUE)
        for pair_id, timeframe in pairs_to_backtest:
            run_pair_backtest(pair_id, timeframe, backtest_months)
        
        print_colored("\nBacktest Selesai untuk semua pair yang diperlukan.", Fore.GREEN, Style.BRIGHT)
        # Reload trades after all backtests
        load_trades() 
    else:
        print_colored("\nTidak ada Backtest yang diperlukan. Riwayat trade sudah cukup.", Fore.GREEN)


def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 
        'bt_months': ('backtest_duration_months', ' bulan') # NEW: for backtest duration
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
            if key_short == 'bt_months':
                if not value.is_integer() or value <= 0:
                    print_colored("Error: Nilai 'bt_months' harus berupa bilangan bulat positif.", Fore.RED); return
                value = int(value) # Convert to int if it's a valid integer float
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
                        # Display Caution Level and derived tolerances
                        caution_level_display = current_settings.get("caution_level", 0.5)
                        BASE_SOL_TOL_DASH = 0.05
                        MAX_ADD_SOL_TOL_DASH = 0.20
                        BASE_EMA_TOL_DASH = 0.00005
                        MAX_ADD_EMA_TOL_DASH = 0.0005
                        actual_sol_tol_display = BASE_SOL_TOL_DASH + (caution_level_display * MAX_ADD_SOL_TOL_DASH)
                        actual_ema_tol_display = BASE_EMA_TOL_DASH + (caution_level_display * MAX_ADD_EMA_TOL_DASH)
                        print_colored(f"  Caution Lv: {caution_level_display:.2f} | Solid. Tol: {actual_sol_tol_display:.3f} | EMA Tol: {actual_ema_tol_display:.6f}", Fore.YELLOW)
                        
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
    
    # NEW: Run backtests for new/unbacktested pairs BEFORE starting live threads
    check_and_run_backtests()

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
                    # Ensure backtests are run again if new pairs were added since last !start
                    check_and_run_backtests() # Re-check before going live

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
                    # Clear any existing trade data in market_state to force a re-fetch/backtest later
                    if pair_id in market_state: del market_state[pair_id]
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"Pair {pair_id} dengan TF {tf} ditambahkan ke watchlist. Backtest akan berjalan saat '!start' berikutnya.", Fore.GREEN)
                else: print_colored("Format salah. Gunakan: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(command_parts) == 2:
                    pair_id = command_parts[1].upper()
                    if pair_id in current_settings['watched_pairs']:
                        del current_settings['watched_pairs'][pair_id]
                        if pair_id in market_state: del market_state[pair_id]
                        # Optionally, remove trades for this pair from autopilot_trades/trades.json
                        # autopilot_trades[:] = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
                        # save_trades()
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
