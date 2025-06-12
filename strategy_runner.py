import json
import os
import time
import threading
import requests
from datetime import datetime
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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap)", Fore.GREEN)
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
                print_colored(f"    Prev Candle: Close {snapshot.get('prev_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_prev'):.4f}", Fore.MAGENTA)
                print_colored(f"    Current Candle: Close {snapshot.get('current_candle_close'):.4f} vs EMA9 {snapshot.get('ema9_current'):.4f}", Fore.MAGENTA)
                
                # New: Display 3 previous candles snapshot
                if 'pre_entry_candle_solidity' in snapshot and 'pre_entry_candle_direction' in snapshot:
                    solidity_str = [f"{s:.2f}" for s in snapshot['pre_entry_candle_solidity']]
                    print_colored(f"    3 Prev Candles Solidity: {solidity_str}", Fore.MAGENTA)
                    print_colored(f"    3 Prev Candles Direction: {snapshot['pre_entry_candle_direction']}", Fore.MAGENTA)
        print()

# --- FUNGSI API (BYBIT) ---
def fetch_bybit_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            # We need enough data for EMA100 and at least 4 candles for trigger + 3 pre-entry snapshot
            # 100 for EMA100, 2 for current/previous candle for EMA9 cross, 3 for pre-entry snapshot
            # So, at least 100 + 3 = 103 for the latest relevant candle to be included in the analysis,
            # and previous ones to be available for the snapshot.
            # The indexing will be candle_data[-1] (current), candle_data[-2] (prev), candle_data[-3], candle_data[-4].
            # So minimum list length is 4. Combined with EMA100, minimum is 100.
            if len(candle_list) < 100: # get_market_analysis will handle specific minimums for its calculations
                return None # Not enough data
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception: return None

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# --- OTAK LOCAL AI (Updated for Exora Vulcan Sniper Entry with 3-candle snapshot) ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair): self.settings = settings; self.past_trades = past_trades_for_pair
    
    # MODIFIED: calculate_ema to return series of EMAs
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
    
    # NEW: analyze_candle_solidity function (re-added)
    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        if full_range == 0: return 1.0 # Avoid division by zero, full solidity if no range
        return body / full_range

    def get_market_analysis(self, candle_data):
        # Need enough data for EMA 100, and at least 4 candles for EMA9 trigger logic (prev and current)
        # AND 3 PRE-ENTRY CANDLES for snapshot (candle_data[-4], candle_data[-3], candle_data[-2])
        if len(candle_data) < 100 or len(candle_data) < 4: # Minimum 100 for EMA100, and minimum 4 for 3-candle snapshot + current
            return None # Not enough data for full analysis

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

        # Add 3 pre-entry candle snapshot for forensic analysis
        pre_entry_candles = candle_data[-4:-1] # Candles before the current (trigger) candle
        pre_entry_solidity = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        pre_entry_direction = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        
        analysis["pre_entry_candle_solidity"] = pre_entry_solidity
        analysis["pre_entry_candle_direction"] = pre_entry_direction

        return analysis
    
    # This function's criteria were based on old analysis metrics (RSI, avg_solidity).
    # Keeping it disabled for now as the user didn't specify new learning criteria for the EMA strategy.
    def check_for_repeated_mistake(self, current_analysis, trade_type, instrument_id):
        return False 
        
    # MODIFIED: get_decision for Exora Vulcan Sniper Entry Logic (no changes to logic itself)
    def get_decision(self, candle_data, open_position, instrument_id):
        analysis = self.get_market_analysis(candle_data)
        
        # Ensure we have enough data and valid analysis values
        if not analysis: 
            return {"action": "HOLD", "reason": "Data tidak cukup atau EMA tidak valid untuk analisis."}
        
        if open_position:
            return {"action": "HOLD", "reason": "Memantau posisi terbuka..."}
        
        # Keeping repeated mistake check if it's meant to be a general safety.
        # For now, it will effectively do nothing as it returns False.
        if self.check_for_repeated_mistake(analysis, "LONG", instrument_id):
            return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan Long."}
        if self.check_for_repeated_mistake(analysis, "SHORT", instrument_id):
             return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan Short."}

        # Exora Vulcan Sniper Entry Logic (unchanged from previous version)
        current_candle_close = analysis['current_candle_close']
        prev_candle_close = analysis['prev_candle_close']
        ema9_current = analysis['ema9_current']
        ema9_prev = analysis['ema9_prev']
        
        # LONG Entry Condition:
        # 1. Bullish Trend (EMA50 > EMA100)
        # 2. Previous candle closed AT OR BELOW EMA9 (retrace/touch)
        # 3. Current candle closed ABOVE EMA9 (cross from below)
        if analysis['bias'] == 'BULLISH':
            if prev_candle_close <= ema9_prev and \
               current_candle_close > ema9_current:
                return {"action": "BUY", "reason": "BULLISH TREND: Candle retrace dan close di atas EMA9.", "snapshot": analysis}
        
        # SHORT Entry Condition:
        # 1. Bearish Trend (EMA50 < EMA100)
        # 2. Previous candle closed AT OR ABOVE EMA9 (retrace/touch)
        # 3. Current candle closed BELOW EMA9 (cross from above)
        elif analysis['bias'] == 'BEARISH':
            if prev_candle_close >= ema9_prev and \
               current_candle_close < ema9_current:
                return {"action": "SELL", "reason": "BEARISH TREND: Candle retrace dan close di bawah EMA9.", "snapshot": analysis}
        
        return {"action": "HOLD", "reason": f"Menunggu setup Exora Vulcan Sniper. Bias: {analysis['bias']}."}

# --- LOGIKA TRADING UTAMA ---
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    # Ensure candle_data exists for the pair when closing
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        # For consistency, use a new LocalAI instance to get the analysis for the exit snapshot
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    # Simpan snapshot entry hanya jika trade rugi untuk dipelajari
    # entry_snapshot will contain the new analysis structure (ema9_current, ema9_prev, etc.)
    # and also 'pre_entry_candle_solidity' and 'pre_entry_candle_direction'
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] # Remove if profitable, only keep for learning losing trades
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f} | Trigger: {close_trigger_reason}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    pair_state = market_state.get(instrument_id)
    
    # Check if there are enough candles for full analysis based on get_market_analysis requirements
    # Minimum 100 for EMA100 and at least 4 candles for the snapshot + current/prev candle for trigger
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100:
        # print_colored(f"Warning: Not enough candle data for {instrument_id} to perform full analysis ({len(pair_state['candle_data']) if pair_state and pair_state.get('candle_data') else 0} candles).", Fore.YELLOW)
        return # Skip analysis if data insufficient

    is_ai_thinking = True
    try:
        candle_data = pair_state["candle_data"]
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        local_brain = LocalAI(current_settings, relevant_trades)
        
        # Pass full candle_data to get_decision
        decision = local_brain.get_decision(candle_data, open_position, instrument_id)
        
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close'] # Entry price is the close of the trigger candle
        
        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            # Initialize current_tp_checkpoint_level for the new trailing TP logic
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
                "trailing_stop_price": None, # This will now be derived from current_tp_checkpoint_level
                "current_tp_checkpoint_level": 0.0 # New field for checkpoint logic
            }
            autopilot_trades.append(new_trade)
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
            notif_content = f"{instrument_id}: Entry @ {current_price:.4f} | {reason}"
            send_termux_notification(notif_title, notif_content)
    except Exception as e:
        print_colored(f"Error in autopilot analysis for {instrument_id}: {e}", Fore.RED)
        is_autopilot_in_cooldown[instrument_id] = True
        await asyncio.sleep(60) # Cooldown to prevent spamming errors
        is_autopilot_in_cooldown[instrument_id] = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    # Run asynchronously but within a synchronous loop
                    asyncio.run(run_autopilot_analysis(pair_id))
                    time.sleep(0.1) # Small sleep between pairs to avoid overwhelming
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

# MODIFIED: check_realtime_position_management for new Trailing TP logic (Checkpoint system)
async def check_realtime_position_management(instrument_id, latest_price):
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    
    # Update run_up_percent and max_drawdown_percent (existing logic, keep)
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl
        
    # SL Logic (unchanged, fixed percentage)
    sl_pct = current_settings.get('stop_loss_pct')
    # SL is a negative value or zero, so PnL <= -sl_pct (e.g. -1.0 <= -0.20 is false, but -0.21 <= -0.20 is true)
    # The default setting is 0.20. So, if PnL is -0.20% or less, it should trigger.
    if sl_pct is not None and current_pnl <= -abs(sl_pct): # Use abs() to handle settings where SL might be positive
        global is_ai_thinking
        if not is_ai_thinking:
            is_ai_thinking = True
            await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.")
            is_ai_thinking = False
        return

    # NEW TRAILING TP LOGIC (Checkpoint system)
    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    
    # Retrieve current checkpoint level from trade object, default to 0.0
    current_tp_checkpoint_level = open_position.get("current_tp_checkpoint_level", 0.0)

    # 1. Activation Phase: If PnL reaches activation_pct and checkpoint is not yet active
    if current_tp_checkpoint_level == 0.0 and current_pnl >= activation_pct:
        open_position['current_tp_checkpoint_level'] = activation_pct
        # Calculate the price at which the PnL matches this checkpoint level
        open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + activation_pct / 100) if open_position['type'] == 'LONG' else \
                                                open_position['entryPrice'] * (1 - activation_pct / 100)
        # print_colored(f"DEBUG: Trailing TP activated for {instrument_id} at {activation_pct:.2f}%. Checkpoint price: {open_position['trailing_stop_price']:.4f}", Fore.BLUE)

    # 2. Update Checkpoint Phase: If trailing TP is active and PnL surpasses the next target checkpoint
    if current_tp_checkpoint_level > 0.0: # Only if trailing TP has been activated
        
        # Calculate the next full checkpoint level based on current_tp_checkpoint_level and gap_pct
        # We find how many steps of 'gap_pct' the PnL has passed BEYOND the current checkpoint
        # E.g., if current_tp_checkpoint_level=0.30, gap=0.05, and PnL=0.37
        # steps_passed = floor((0.37 - 0.30) / 0.05) = floor(0.07 / 0.05) = floor(1.4) = 1
        steps_passed = math.floor((current_pnl - current_tp_checkpoint_level) / gap_pct)
        
        if steps_passed > 0:
            open_position['current_tp_checkpoint_level'] += steps_passed * gap_pct
            # Recalculate trailing_stop_price to match the new, higher checkpoint PnL level
            open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + open_position['current_tp_checkpoint_level'] / 100) if open_position['type'] == 'LONG' else \
                                                    open_position['entryPrice'] * (1 - open_position['current_tp_checkpoint_level'] / 100)
            # print_colored(f"DEBUG: {instrument_id} Passed new checkpoint. Current checkpoint: {open_position['current_tp_checkpoint_level']:.2f}%. New TP Price: {open_position['trailing_stop_price']:.4f}", Fore.MAGENTA)
        
        # 3. Exit Condition Phase: If current price drops back to or crosses the trailing_stop_price (checkpoint price)
        # This condition checks if the PnL has fallen back to the value of the last activated checkpoint.
        if open_position['type'] == 'LONG' and latest_price <= open_position.get('trailing_stop_price', 0):
            if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.")
                is_ai_thinking = False
        elif open_position['type'] == 'SHORT' and latest_price >= open_position.get('trailing_stop_price', float('inf')):
             if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, latest_price, f"Trailing TP (checkpoint {open_position['current_tp_checkpoint_level']:.2f}%) tercapai.")
                is_ai_thinking = False
    
    # Save the trade state after potential updates to run_up_percent, max_drawdown_percent, and trailing TP info
    save_trades()


def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, timeframe in watched_pairs.items():
                data = fetch_bybit_candle_data(pair_id, timeframe)
                if data: 
                    # Only calculate analysis if we have enough data after fetching
                    # get_market_analysis itself also checks data length, but this is a pre-check
                    if len(data) >= 100: # Minimum for EMA100 and 3 pre-entry candles + current
                        analysis = LocalAI(current_settings, []).get_market_analysis(data)
                    else:
                        analysis = None # Not enough data for full analysis
                        # print_colored(f"Skipping analysis for {pair_id} due to insufficient candle data ({len(data)} candles).", Fore.YELLOW)

                    market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                    
                    if is_autopilot_running:
                        latest_price = data[-1]['close']
                        asyncio.run(check_realtime_position_management(pair_id, latest_price))
                time.sleep(0.5) # Short delay between fetching data for different pairs
        stop_event.wait(REFRESH_INTERVAL_SECONDS) # Main refresh interval

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'),
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%')
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
                        print_colored("  Trailing TP: ", end=''); print_colored(f"Aktif @ {cp_level:.2f}% PnL ({ts_price:.4f})", Fore.MAGENTA)
                    else:
                        print_colored("  Trailing TP: ", end=''); print_colored(f"Menunggu {current_settings.get('trailing_tp_activation_pct'):.2f}% PnL", Fore.WHITE)
                else:
                    print_colored("  Status    : ", end=''); print_colored("Searching for setup...", Fore.BLUE)
                    if pair_state and pair_state.get("analysis"):
                        analysis = pair_state["analysis"]
                        bias_color = Fore.GREEN if analysis['bias'] == 'BULLISH' else Fore.RED if analysis['bias'] == 'BEARISH' else Fore.YELLOW
                        print_colored("  Trend     : ", end=''); print_colored(analysis['bias'], bias_color)
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
