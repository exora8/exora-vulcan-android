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
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data/v2/" 
REFRESH_INTERVAL_SECONDS = 0.5 
BACKTEST_FETCH_CHUNK_LIMIT = 1000 

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
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, winrate, cc_key)", Fore.GREEN) 
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
        "target_winrate_pct": 85.0, 
        "cryptocompare_api_key": "YOUR_CRYPTOCOMPARE_API_KEY", 
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
            print_colored(f"  Max Drawdown (MDD): {trade.get('max_drawdown_percent', 0.0):.2f}%", Fore.YELLOW) 
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

# --- FUNGSI API ---
def fetch_recent_candles(instId, timeframe, limit=300):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: 
                return None 
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: 
            print_colored(f"API Error fetching live data for {instId}: retCode {data.get('retCode')}, retMsg {data.get('retMsg', 'Unknown')}", Fore.RED)
            return None
    except requests.exceptions.RequestException as e:
        print_colored(f"Network/Request Error fetching live data for {instId}: {e}", Fore.RED)
        return None
    except Exception as e: 
        print_colored(f"Unknown Error fetching live data for {instId}: {e}", Fore.RED)
        return None

def fetch_historical_candles_backward_from_ts(instId, timeframe, to_ts_seconds, limit_per_request):
    """
    Fetches historical candles backward from a given timestamp (to_ts_seconds) using CryptoCompare.
    Returns the fetched candles (oldest first) and the timestamp of the earliest candle in the batch.
    """
    timeframe_map = {'1m': 'histominute', '1H': 'histohour', '1D': 'histoday'} 
    
    cc_endpoint = timeframe_map.get(timeframe)
    if not cc_endpoint:
        print_colored(f"Error: Timeframe '{timeframe}' tidak didukung untuk CryptoCompare backtest.", Fore.RED)
        return [], 0 

    try:
        fsym, tsym = instId.split('-')
    except ValueError:
        print_colored(f"Error: Pair ID '{instId}' tidak dalam format FSMBAL-TSYMBOL (e.g., APE-USDT) untuk CryptoCompare.", Fore.RED)
        return [], 0

    api_key = current_settings.get("cryptocompare_api_key")
    if not api_key or api_key == "YOUR_CRYPTOCOMPARE_API_KEY":
        print_colored("Error: CryptoCompare API Key belum diatur di settings.json atau menggunakan placeholder.", Fore.RED, Style.BRIGHT)
        print_colored("Dapatkan API Key di cryptocompare.com dan set dengan '!set cc_key <your_key>'.", Fore.YELLOW)
        return [], 0

    url = f"{CRYPTOCOMPARE_API_URL}{cc_endpoint}?fsym={fsym}&tsym={tsym}&limit={limit_per_request}&toTs={to_ts_seconds}&api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=20) 
        response.raise_for_status() 
        data = response.json()

        if data.get("Response") == "Success" and 'Data' in data.get('Data', {}):
            candles_batch_raw = data['Data']['Data']
            
            if not candles_batch_raw:
                return [], 0 

            formatted_batch = [{"time": c['time'] * 1000, "open": c['open'], "high": c['high'], "low": c['low'], "close": c['close'], "volume": c['volumefrom']} for c in candles_batch_raw]
            
            earliest_ts_in_batch = candles_batch_raw[0]['time'] 
            return formatted_batch, earliest_ts_in_batch
        else:
            print_colored(f"  API Error fetching historical data from CC for {instId}: Response: {data.get('Response')}, Message: {data.get('Message', 'Unknown error')}", Fore.RED)
            return [], 0
    except requests.exceptions.RequestException as e:
        print_colored(f"  Network/Request Error fetching historical data from CC for {instId}: {e}. Retrying in 5s...", Fore.RED)
        time.sleep(5) 
        return [], 0 
    except Exception as e: 
        print_colored(f"  Unknown Error fetching historical data from CC for {instId}: {e}. Breaking fetching.", Fore.RED)
        return [], 0

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

def calculate_winrate(trades_list, fee_pct):
    if not trades_list:
        return 0.0
    
    total_trades = len(trades_list)
    profitable_trades = 0
    
    for trade in trades_list:
        if trade.get('status') == 'CLOSED' and trade.get('pl_percent', 0) > fee_pct:
            profitable_trades += 1
            
    if total_trades == 0:
        return 0.0
        
    return (profitable_trades / total_trades) * 100

# --- OTAK LOCAL AI (Updated for Exora Vulcan Sniper Entry with 3-candle snapshot and learning) ---
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
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, is_backtest=False):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    exit_snapshot = None
    if trade['instrumentId'] in market_state and market_state[trade['instrumentId']].get('candle_data'):
        exit_snapshot = LocalAI(current_settings, []).get_market_analysis(market_state[trade['instrumentId']]['candle_data'])
    
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.now().isoformat(), 'pl_percent': pnl, 'exit_snapshot': exit_snapshot})
    
    if is_profit and 'entry_snapshot' in trade:
        del trade['entry_snapshot'] 
    
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
                    time.sleep(current_settings.get("analysis_interval_sec", 10)) 
            stop_event.wait(current_settings.get("analysis_interval_sec", 10)) 
        else: time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data, is_backtest=False):
    if not trade_obj: return
    
    pnl_at_close = calculate_pnl(trade_obj['entryPrice'], current_candle_data['close'], trade_obj.get('type'))
    
    if trade_obj['type'] == 'LONG':
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG')
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'LONG')
        
        if pnl_at_high > trade_obj.get('run_up_percent', 0.0):
            trade_obj['run_up_percent'] = pnl_at_high
        if pnl_at_low < trade_obj.get('max_drawdown_percent', 0.0): 
            trade_obj['max_drawdown_percent'] = pnl_at_low

    elif trade_obj['type'] == 'SHORT':
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT') 
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'SHORT') 
        
        if pnl_at_low > trade_obj.get('run_up_percent', 0.0):
            trade_obj['run_up_percent'] = pnl_at_low
        if pnl_at_high < trade_obj.get('max_drawdown_percent', 0.0): 
            trade_obj['max_drawdown_percent'] = pnl_at_high

    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else \
               trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    
    sl_hit = False
    if trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price:
        sl_hit = True; exit_price = sl_price
    elif trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price:
        sl_hit = True; exit_price = sl_price
    
    if sl_hit:
        global is_ai_thinking
        if not is_ai_thinking: 
            is_ai_thinking = True
            await analyze_and_close_trade(trade_obj, exit_price, f"Stop Loss @ {-abs(sl_pct):.2f}% tercapai.", is_backtest)
            is_ai_thinking = False
        return 

    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    
    current_tp_checkpoint_level = trade_obj.get("current_tp_checkpoint_level", 0.0)

    if current_tp_checkpoint_level == 0.0: 
        if trade_obj['type'] == 'LONG' and calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG') >= activation_pct:
            trade_obj['current_tp_checkpoint_level'] = activation_pct
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + activation_pct / 100)
        elif trade_obj['type'] == 'SHORT' and calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT') >= activation_pct:
            trade_obj['current_tp_checkpoint_level'] = activation_pct
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 - activation_pct / 100)

    if trade_obj['current_tp_checkpoint_level'] > 0.0: 
        if trade_obj['type'] == 'LONG':
            potential_new_checkpoint_pnl = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG')
            steps_passed = math.floor((potential_new_checkpoint_pnl - trade_obj['current_tp_checkpoint_level']) / gap_pct)
        else: 
            potential_new_checkpoint_pnl = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT')
            steps_passed = math.floor((potential_new_checkpoint_pnl - trade_obj['current_tp_checkpoint_level']) / gap_pct)

        if steps_passed > 0:
            trade_obj['current_tp_checkpoint_level'] += steps_passed * gap_pct
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + trade_obj['current_tp_checkpoint_level'] / 100) if trade_obj['type'] == 'LONG' else \
                                                    trade_obj['entryPrice'] * (1 - trade_obj['current_tp_checkpoint_level'] / 100)
        
        tp_hit = False
        if trade_obj.get('trailing_stop_price') is not None:
            if trade_obj['type'] == 'LONG' and current_candle_data['low'] <= trade_obj['trailing_stop_price']:
                tp_hit = True; exit_price = trade_obj['trailing_stop_price']
            elif trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= trade_obj['trailing_stop_price']:
                tp_hit = True; exit_price = trade_obj['trailing_stop_price']
        
        if tp_hit:
            if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(trade_obj, exit_price, f"Trailing TP (checkpoint {trade_obj['current_tp_checkpoint_level']:.2f}%) tercapai.", is_backtest)
                is_ai_thinking = False
            return 
    
    if not is_backtest:
        save_trades() 

def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, timeframe in watched_pairs.items():
                data = fetch_recent_candles(pair_id, timeframe) 
                if data: 
                    if len(data) < 100 + 3: 
                        print_colored(f"Warning: Not enough live candle data for {pair_id} to analyze ({len(data)} candles). Skipping analysis.", Fore.YELLOW)
                        market_state[pair_id] = {"candle_data": data, "analysis": None} 
                        continue

                    analysis = LocalAI(current_settings, []).get_market_analysis(data)
                    market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                    
                    if is_autopilot_running:
                        open_pos_for_management = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                        if open_pos_for_management:
                            asyncio.run(check_realtime_position_management(open_pos_for_management, data[-1])) 
                time.sleep(0.5) 
        stop_event.wait(REFRESH_INTERVAL_SECONDS) 

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print_colored(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def run_pair_backtest(pair_id, timeframe): 
    global autopilot_trades 
    print_colored(f"\n🚀 Memulai Backtest untuk {pair_id} ({timeframe})...", Fore.CYAN, Style.BRIGHT)

    # 1. Clear existing trades for this pair from global autopilot_trades and save
    autopilot_trades[:] = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
    save_trades() 
    
    cumulative_historical_candles = [] 
    
    current_to_ts_for_fetch = int(datetime.now().timestamp()) 
    target_winrate = current_settings.get("target_winrate_pct", 85.0)
    
    highest_achieved_winrate = 0.0
    trades_at_highest_winrate = 0
    
    fetch_iteration = 0
    while True: 
        fetch_iteration += 1
        print_colored(f"  [DEBUG] Fetching historical chunk #{fetch_iteration} for {pair_id}...", Fore.BLUE)
        
        fetched_chunk, earliest_ts_in_chunk = fetch_historical_candles_backward_from_ts(
            pair_id, timeframe, current_to_ts_for_fetch, BACKTEST_FETCH_CHUNK_LIMIT
        )
        
        if not fetched_chunk:
            print_colored(f"  [DEBUG] No more historical data to fetch for {pair_id}. Ending backtest.", Fore.YELLOW)
            break 
            
        cumulative_historical_candles = fetched_chunk + cumulative_historical_candles 
        
        cumulative_historical_candles_dict = {c['time']: c for c in cumulative_historical_candles}
        cumulative_historical_candles = sorted(cumulative_historical_candles_dict.values(), key=lambda x: x['time'])

        current_to_ts_for_fetch = earliest_ts_in_chunk - 1 

        temp_open_backtest_trades = []
        temp_backtested_trades_list_in_run = [] 

        total_candles_in_cumulative = len(cumulative_historical_candles)
        print_colored(f"  [DEBUG] Re-analyzing {total_candles_in_cumulative} cumulative candles for {pair_id}...", Fore.BLUE)
        
        for i in range(total_candles_in_cumulative): 
            current_historical_data_slice = cumulative_historical_candles[:i+1] 
            current_candle = cumulative_historical_candles[i] 
            
            if len(current_historical_data_slice) < 100 + 3: 
                print_progress_bar(i + 1, total_candles_in_cumulative, prefix=f'  {pair_id} Analisis', suffix='Lengkap', length=50, fill='=')
                continue

            learning_trades_for_local_ai = [t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'CLOSED'] + temp_backtested_trades_list_in_run

            market_state[pair_id] = {"candle_data": current_historical_data_slice, 
                                     "analysis": LocalAI(current_settings, learning_trades_for_local_ai).get_market_analysis(current_historical_data_slice)}

            trades_to_close_in_current_candle = []
            for open_bt_trade in list(temp_open_backtest_trades): 
                asyncio.run(check_realtime_position_management(open_bt_trade, current_candle, is_backtest=True))
                
                if open_bt_trade['status'] == 'CLOSED':
                     trades_to_close_in_current_candle.append(open_bt_trade)
            
            for trade_to_remove in trades_to_close_in_current_candle:
                if trade_to_remove in temp_open_backtest_trades: 
                    temp_open_backtest_trades.remove(trade_to_remove)
                    temp_backtested_trades_list_in_run.append(trade_to_remove)


            if not any(t for t in temp_open_backtest_trades if t['instrumentId'] == pair_id): 
                local_brain_for_decision = LocalAI(current_settings, learning_trades_for_local_ai) 
                decision = local_brain_for_decision.get_decision(current_historical_data_slice, None, pair_id) 
                
                if decision.get('action') in ["BUY", "SELL"]:
                    trade_type = "LONG" if decision['action'] == "BUY" else "SHORT"
                    new_trade = {
                        "id": int(current_candle['time'] / 1000), 
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
                    temp_open_backtest_trades.append(new_trade)


            print_progress_bar(i + 1, total_candles_in_cumulative, prefix=f'  {pair_id} Analisis', suffix='Lengkap', length=50, fill='=')

        for open_bt_trade in list(temp_open_backtest_trades): 
            asyncio.run(analyze_and_close_trade(open_bt_trade, cumulative_historical_candles[-1]['close'], "Backtest End (Force Close)", is_backtest=True))
            temp_backtested_trades_list_in_run.append(open_bt_trade)

        # Update global autopilot_trades with trades from this re-analysis run
        current_autopilot_trades_without_this_pair = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
        current_autopilot_trades_without_this_pair.extend(temp_backtested_trades_list_in_run)
        current_autopilot_trades_without_this_pair.sort(key=lambda x: x['entryTimestamp'])
        autopilot_trades[:] = current_autopilot_trades_without_this_pair 
        
        save_trades() # Save trades.json after each chunk analysis is fully processed
        load_trades() # Reload trades from the updated trades.json to refresh global autopilot_trades
        
        trades_for_winrate_calc = [t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'CLOSED']
        current_cumulative_winrate = calculate_winrate(trades_for_winrate_calc, current_settings.get('fee_pct', 0.1))
        
        # NEW: Calculate and display new metrics
        total_pnl = sum(t.get('pl_percent', 0.0) for t in trades_for_winrate_calc)
        num_trades = len(trades_for_winrate_calc)
        average_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
        
        print_colored(f"\n  [DEBUG] Backtest Stats for {pair_id} ({num_trades} trades): Winrate: {current_cumulative_winrate:.2f}% | Avg P/L: {average_pnl:.2f}% | Total P/L: {total_pnl:.2f}% (Target Winrate: {target_winrate:.2f}%)", Fore.CYAN)
        
        if current_cumulative_winrate > highest_achieved_winrate:
            highest_achieved_winrate = current_cumulative_winrate
            trades_at_highest_winrate = len(trades_for_winrate_calc)

        MIN_TRADES_FOR_WINRATE_STABILITY = 50 
        if current_cumulative_winrate >= target_winrate and len(trades_for_winrate_calc) >= MIN_TRADES_FOR_WINRATE_STABILITY:
            print_colored(f"✅ Target Winrate ({target_winrate:.2f}%) tercapai untuk {pair_id} setelah {total_candles_in_cumulative} candle dan {len(trades_for_winrate_calc)} trade.", Fore.GREEN, Style.BRIGHT)
            break
        elif total_candles_in_cumulative >= BACKTEST_FETCH_CHUNK_LIMIT * 100: # Fail-safe: stop after X chunks
             print_colored(f"⚠️ Backtest untuk {pair_id} dihentikan: Terlalu banyak candle diproses tanpa mencapai target winrate. Winrate tertinggi: {highest_achieved_winrate:.2f}% ({trades_at_highest_winrate} trade).", Fore.YELLOW, Style.BRIGHT)
             break

        time.sleep(0.5) 

    autopilot_trades.sort(key=lambda x: x['entryTimestamp']) 
    save_trades() 
    
    print_colored(f"✅ Backtest untuk {pair_id} selesai. Total trade di history untuk pair ini: {len([t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'CLOSED'])}.", Fore.GREEN, Style.BRIGHT)
    return True


def check_and_run_backtests():
    watched_pairs = current_settings.get("watched_pairs", {})
    
    pairs_to_backtest = []
    
    for pair_id, timeframe in watched_pairs.items():
        has_existing_trades = any(t for t in autopilot_trades if t['instrumentId'] == pair_id)
        
        if not has_existing_trades: 
            pairs_to_backtest.append((pair_id, timeframe))
    
    if pairs_to_backtest:
        print_colored(f"\nMemerlukan Backtest untuk pembelajaran AI (hingga Winrate {current_settings.get('target_winrate_pct'):.2f}% tercapai):", Fore.CYAN, Style.BRIGHT)
        for pair_id, timeframe in pairs_to_backtest:
            print_colored(f"- {pair_id} ({timeframe})", Fore.YELLOW)
        
        print_colored("Memulai proses Backtest. Mohon tunggu...", Fore.BLUE)
        for pair_id, timeframe in pairs_to_backtest:
            run_pair_backtest(pair_id, timeframe) 
        
        print_colored("\nBacktest Selesai untuk semua pair yang diperlukan.", Fore.GREEN, Style.BRIGHT)
        load_trades() 
    else:
        print_colored(f"\nTidak ada Backtest yang diperlukan. Riwayat trade sudah cukup untuk pembelajaran.", Fore.GREEN)


def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'),
        'delay': ('analysis_interval_sec', ' detik'), 
        'tp_act': ('trailing_tp_activation_pct', '%'),
        'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 
        'winrate': ('target_winrate_pct', '%'), 
        'cc_key': ('cryptocompare_api_key', '') 
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full_key, unit) in setting_map.items():
            display_key = key.capitalize().ljust(10)
            display_value = current_settings[full_key]
            if full_key == 'cryptocompare_api_key' and display_value != "YOUR_CRYPTOCOMPARE_API_KEY":
                display_value = display_value[:4] + '...' + display_value[-4:] 
            print_colored(f"{display_key} ({key:<10}) : {display_value}{unit}", Fore.WHITE)
        print_colored(f"Backtest Chunk  : {BACKTEST_FETCH_CHUNK_LIMIT} candles (Static)", Fore.WHITE) 
        print(); return
    if len(parts) == 3 and parts[0] == '!set':
        key_short = parts[1].lower()
        if key_short not in setting_map: print_colored(f"Error: Kunci '{key_short}' tidak dikenal.", Fore.RED); return
        try:
            value = parts[2] 
            if key_short in ['sl', 'fee', 'delay', 'tp_act', 'tp_gap', 'caution', 'winrate']: 
                value = float(value)
                if key_short == 'caution' and value < 0:
                    print_colored("Error: Nilai 'caution' harus positif.", Fore.RED); return
                if key_short == 'winrate' and not (0.0 <= value <= 100.0): 
                    print_colored("Error: Nilai 'winrate' harus antara 0.0 dan 100.0.", Fore.RED); return
                if value < 0: print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
        except ValueError: print_colored(f"Error: Nilai '{parts[2]}' harus berupa angka untuk kunci ini.", Fore.RED); return
        
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
                        caution_level_display = current_settings.get("caution_level", 0.5)
                        BASE_SOL_TOL_DASH = 0.05
                        MAX_ADD_SOL_TOL_DASH = 0.20
                        BASE_EMA_TOL_DASH = 0.00005
                        MAX_ADD_EMA_TOL_DASH = 0.0005
                        actual_sol_tol_display = BASE_SOL_TOL_DASH + (caution_level_display * MAX_ADD_SOL_TOL_DASH)
                        actual_ema_tol_display = BASE_EMA_TOL_DASH + (caution_level_display * MAX_ADD_EMA_TOL_DASH)
                        print_colored(f"  Caution Lv: {caution_level_display:.2f} | Solid. Tol: {actual_sol_tol_display:.3f} | EMA Tol: {actual_ema_tol_display:.6f}", Fore.YELLOW)
                        
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
    
    check_and_run_backtests()

    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()
    while True:
        try:
            prompt_text = f"[Command] > "
            user_input = input(prompt_text)
            command_parts = user_input.split()
            if not command_parts: 
                continue 

            cmd = command_parts[0].lower() 

            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan. Dashboard aktif.", Fore.YELLOW)
                elif not current_settings.get("watched_pairs"): print_colored("Error: Watchlist kosong. Gunakan '!watch <PAIR>' dulu.", Fore.RED)
                else: 
                    check_and_run_backtests() 

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
