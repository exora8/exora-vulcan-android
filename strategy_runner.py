import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math

# --- Prasyarat Web ---
try:
    from flask import Flask, jsonify, request, render_template_string
except ImportError:
    print("Peringatan: Pustaka 'Flask' tidak ditemukan. Fitur web tidak akan tersedia.")
    print("Silakan install dengan 'pip install Flask'")
    Flask = None

# --- Dummy Colorama for environments where it's not installed ---
class DummyColor:
    def __init__(self):
        self.BLACK = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.MAGENTA = self.CYAN = self.WHITE = self.RESET = ''
class DummyStyle:
     def __init__(self):
        self.DIM = self.NORMAL = self.BRIGHT = self.RESET_ALL = ''

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Peringatan: Pustaka 'colorama' tidak ditemukan. Output tidak akan berwarna.")
    Fore = DummyColor()
    Style = DummyStyle()

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"
REFRESH_INTERVAL_SECONDS = 0.5
MAX_TRADES_IN_HISTORY = 800

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
market_state = {}
is_ai_thinking = False
is_autopilot_in_cooldown = {}
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ

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
    print_colored("FITUR BARU: Kontrol & Monitoring Real-time via Web.", Fore.GREEN)
    if Flask:
        print_colored("Web Dashboard aktif di http://127.0.0.1:5001", Fore.GREEN)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk Live Dashboard AI.", Fore.YELLOW)
    print_colored("Gunakan '!start_manual' untuk Live Dashboard Manual.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot AI", Fore.GREEN)
    print_colored("!start_manual         - Masuk ke Dashboard Trading Manual", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade (terbatas 80 terakhir)", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, winrate, fr_max)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# ... [BAGIAN KODE LAINNYA YANG TIDAK BERUBAH: load_settings, save_settings, load_trades, save_trades, display_history, fetch_funding_rate, fetch_recent_candles, semua fungsi calculate_*, class LocalAI, manage_trade_closure, dll... TETAP SAMA] ...
# (Untuk keringkasan, saya tidak menampilkan ulang semua fungsi yang tidak berubah, tetapi di skrip akhir Anda, semua fungsi itu harus ada)
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 0.20, "fee_pct": 0.1, "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30, "trailing_tp_gap_pct": 0.05,
        "caution_level": 0.5, "target_winrate_pct": 85.0,
        "max_allowed_funding_rate_pct": 0.075, "watched_pairs": {}
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                for key, value in default_settings.items():
                    if key not in loaded_settings: loaded_settings[key] = value
                current_settings = loaded_settings
        except (json.JSONDecodeError, IOError):
            current_settings = default_settings
    else:
        current_settings = default_settings
    save_settings()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)
    except IOError as e: print_colored(f"Error saving settings: {e}", Fore.RED)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
        except (json.JSONDecodeError, IOError): autopilot_trades = []
    else: autopilot_trades = []
    for trade in autopilot_trades:
        if 'current_tp_checkpoint_level' not in trade: trade['current_tp_checkpoint_level'] = 0.0

def save_trades():
    global autopilot_trades
    autopilot_trades.sort(key=lambda x: x['entryTimestamp'])
    if len(autopilot_trades) > MAX_TRADES_IN_HISTORY:
        num_to_trim = len(autopilot_trades) - MAX_TRADES_IN_HISTORY
        autopilot_trades = autopilot_trades[num_to_trim:]
        print_colored(f"Riwayat trade dibatasi. {num_to_trim} trade tertua telah dihapus.", Fore.YELLOW)
    try:
        with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)
    except IOError as e: print_colored(f"Error saving trades: {e}", Fore.RED)

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in sorted(autopilot_trades, key=lambda x: x['entryTimestamp'], reverse=True):
        entry_time_str = trade['entryTimestamp'].replace('Z', '')
        exit_time_str = trade.get('exitTimestamp', '').replace('Z', '')
        entry_time = datetime.fromisoformat(entry_time_str).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        trade_type = trade.get('type', 'LONG'); type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        
        if trade.get('entryReason'):
            reason_lines = trade['entryReason'].split('\n')
            print_colored("  Alasan Entry:", Fore.YELLOW)
            for line in reason_lines:
                print_colored(f"    {line}", Fore.WHITE)

        if trade['status'] == 'CLOSED' and exit_time_str:
            exit_time = datetime.fromisoformat(exit_time_str).strftime('%Y-%m-%d %H:%M')
            pl_percent_gross = trade.get('pl_percent', 0.0)
            pl_percent_net = pl_percent_gross - fee_pct
            is_profit = pl_percent_net > 0
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L (Net): {pl_percent_net:.2f}%", pl_color, Style.BRIGHT)

# --- FUNGSI API ---
def fetch_funding_rate(instId):
    bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/tickers?category=linear&symbol={bybit_symbol}"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}) and data['result']['list']:
            return float(data['result']['list'][0].get('fundingRate', '0')) * 100
        return None
    except (requests.exceptions.RequestException, ValueError, KeyError): return None

def fetch_recent_candles(instId, timeframe, limit=300):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=linear&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: return None
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        return None
    except requests.exceptions.RequestException: return None
    except Exception: return None

# --- FUNGSI KALKULASI PNL ---
def calculate_pnl(entry_price, current_price, trade_type):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

def calculate_winrate(trades_list, fee_pct):
    closed_trades = [t for t in trades_list if t.get('status') == 'CLOSED']
    if not closed_trades: return 0.0
    profitable_trades = sum(1 for t in closed_trades if (t.get('pl_percent', 0) - fee_pct) > 0)
    return (profitable_trades / len(closed_trades)) * 100

def calculate_todays_pnl(all_trades):
    today_utc = datetime.utcnow().date(); total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() == today_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

def calculate_this_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date(); start_of_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_week_utc = start_of_week_utc + timedelta(days=6); total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if start_of_week_utc <= datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() <= end_of_week_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

def calculate_last_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date(); start_of_current_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_last_week_utc = start_of_current_week_utc - timedelta(days=1); start_of_last_week_utc = end_of_last_week_utc - timedelta(days=6)
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if start_of_last_week_utc <= datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() <= end_of_last_week_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

# --- OTAK LOCAL AI ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair):
        self.settings = settings
        self.past_trades = past_trades_for_pair

    def calculate_ema(self, data, period):
        if len(data) < period: return []
        closes = [d['close'] for d in data]
        ema_values = [sum(closes[:period]) / period]
        multiplier = 2 / (period + 1)
        for i in range(period, len(closes)):
            ema = (closes[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        return ema_values

    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        return body / full_range if full_range > 0 else 1.0

    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100 + 3: return None
        ema9 = self.calculate_ema(candle_data, 9)
        ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100)
        if len(ema9) < 2 or not ema50 or not ema100: return None
        
        analysis = {
            "ema9_current": ema9[-1], "ema9_prev": ema9[-2], "ema50": ema50[-1], "ema100": ema100[-1],
            "current_candle_close": candle_data[-1]['close'], "prev_candle_close": candle_data[-2]['close'],
            "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING",
        }
        pre_entry_candles = candle_data[-4:-1]
        analysis["pre_entry_candle_solidity"] = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        analysis["pre_entry_candle_direction"] = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        return analysis

    def check_for_repeated_mistake(self, current_analysis):
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - self.settings.get('fee_pct', 0.1)) < 0]
        if not losing_trades: return (False, None)

        SIMILARITY_THRESHOLD = 3 
        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or past_snapshot.get('bias') != current_analysis['bias']:
                continue

            similarity_score = 1
            current_pos_vs_ema50 = 'above' if current_analysis['current_candle_close'] > current_analysis['ema50'] else 'below'
            past_pos_vs_ema50 = 'above' if past_snapshot['current_candle_close'] > past_snapshot['ema50'] else 'below'
            if current_pos_vs_ema50 == past_pos_vs_ema50: similarity_score += 1
            if current_analysis['pre_entry_candle_direction'] == past_snapshot.get('pre_entry_candle_direction', []): similarity_score += 1
            
            avg_solidity_current = sum(current_analysis['pre_entry_candle_solidity']) / 3
            past_solidity_list = past_snapshot.get('pre_entry_candle_solidity', [0,0,0])
            avg_solidity_past = sum(past_solidity_list) / 3 if past_solidity_list else 0
            if abs(avg_solidity_current - avg_solidity_past) < 0.2: similarity_score += 1

            if similarity_score >= SIMILARITY_THRESHOLD:
                reason = (f"PERINGATAN: Setup ini {similarity_score*25}% mirip dengan loss sebelumnya (ID: {loss['id']}).\n"
                          f"  - Kedua setup memiliki bias {current_analysis['bias']} dan harga di {current_pos_vs_ema50} EMA50.")
                return (True, reason)
        return (False, None)

    def get_decision(self, candle_data, open_position, funding_rate=0.0):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data teknikal tidak cukup untuk analisis."}
        if open_position: return {"action": "HOLD", "reason": "Sudah ada posisi terbuka, sedang memantau."}

        max_funding_rate = self.settings.get("max_allowed_funding_rate_pct", 0.075)
        potential_trade_type = None

        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']:
            potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']:
            potential_trade_type = 'SHORT'

        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate:
                return {"action": "HOLD", "reason": f"Sinyal LONG diabaikan. Funding rate terlalu tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate:
                return {"action": "HOLD", "reason": f"Sinyal SHORT diabaikan. Funding rate terlalu negatif: {funding_rate:.4f}%"}

            is_repeated_mistake, warning_reason = self.check_for_repeated_mistake(analysis)
            if is_repeated_mistake: return {"action": "HOLD", "reason": warning_reason}

            ai_reason = (f"Alasan AI (tinyllama): Entry {potential_trade_type} diambil berdasarkan konfirmasi kelanjutan tren. "
                         f"Pasar menunjukkan bias {analysis['bias']} yang kuat, dan sinyal ini muncul setelah harga melakukan pullback sehat ke EMA9, "
                         "menunjukkan potensi kekuatan untuk melanjutkan pergerakan.")
            avg_solidity = sum(analysis['pre_entry_candle_solidity']) / 3
            bot_details = (f"Detail Bot: \n"
                           f"  - Bias Pasar: {analysis['bias']} (EMA50: {analysis['ema50']:.2f} vs EMA100: {analysis['ema100']:.2f})\n"
                           f"  - Trigger: Candle close ({analysis['current_candle_close']:.2f}) melintasi EMA9 ({analysis['ema9_current']:.2f})\n"
                           f"  - Kondisi Pra-Entry: Arah 3 candle = {analysis['pre_entry_candle_direction']}, Rata-rata Soliditas = {avg_solidity:.2f}")
            full_reason = f"{ai_reason}\n{bot_details}"
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": full_reason, "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias saat ini: {analysis['bias']}."}

# --- LOGIKA TRADING UTAMA ---
async def manage_trade_closure(trade, exit_price, reason, exit_timestamp_ms=None):
    pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    exit_dt = datetime.fromtimestamp(exit_timestamp_ms / 1000) if exit_timestamp_ms else datetime.utcnow()
    
    trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })
    
    is_profit = (pnl_gross - current_settings.get('fee_pct', 0.1)) > 0
    if is_profit and 'entry_snapshot' in trade:
        try: del trade['entry_snapshot']
        except KeyError: pass

    save_trades()
    pnl_net = pnl_gross - current_settings.get('fee_pct', 0.1)
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | Trigger: {reason}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3: return

    is_ai_thinking = True
    try:
        open_pos = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        ai = LocalAI(current_settings, relevant_trades)
        funding_rate = pair_state.get("funding_rate", 0.0)
        decision = ai.get_decision(pair_state["candle_data"], open_pos, funding_rate)

        if decision.get('action') in ["BUY", "SELL"] and not open_pos:
            snapshot = decision.get("snapshot", {})
            snapshot["funding_rate"] = funding_rate
            new_trade = {
                "id": int(time.time()), "instrumentId": instrument_id,
                "type": "LONG" if decision['action'] == "BUY" else "SHORT",
                "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": pair_state["candle_data"][-1]['close'],
                "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot,
                "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
            }
            autopilot_trades.append(new_trade); save_trades()
            ai_reason_short = decision.get("reason").split('\n')[0]
            notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {instrument_id}"
            notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | {ai_reason_short}"
            send_termux_notification(notif_title, notif_content)
    except Exception as e:
        print_colored(f"Error dalam autopilot analysis: {e}", Fore.RED)
    finally: is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            for pair_id in list(current_settings.get("watched_pairs", {})):
                asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data):
    if not trade_obj: return
    current_pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], trade_obj['type'])
    current_pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], trade_obj['type'])
    
    if trade_obj['type'] == 'LONG':
        if current_pnl_at_high > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = current_pnl_at_high
        if current_pnl_at_low < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = current_pnl_at_low
    else: # SHORT
        if current_pnl_at_low > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = current_pnl_at_low
        if current_pnl_at_high < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = current_pnl_at_high

    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or \
       (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
        await manage_trade_closure(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}%")
        return

    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30); gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    ts_price = trade_obj.get('trailing_stop_price')
    if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or \
                                 (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
        await manage_trade_closure(trade_obj, ts_price, f"Trailing TP")
        return
    
    pnl_now = trade_obj.get('run_up_percent', 0.0)
    if pnl_now >= activation_pct:
        current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
        if current_cp == 0.0: current_cp = activation_pct
        
        steps_passed = math.floor((pnl_now - current_cp) / gap_pct)
        if steps_passed >= 0:
            new_cp = current_cp + (steps_passed * gap_pct)
            if new_cp > trade_obj.get('current_tp_checkpoint_level', 0.0):
                trade_obj['current_tp_checkpoint_level'] = new_cp
                new_ts_level = new_cp - gap_pct
                trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + new_ts_level / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 - new_ts_level / 100)

    save_trades()

def data_refresh_worker():
    while not stop_event.is_set():
        for pair_id, timeframe in list(current_settings.get("watched_pairs", {}).items()):
            if pair_id not in market_state: market_state[pair_id] = {}
            candle_data = fetch_recent_candles(pair_id, timeframe)
            funding_rate = fetch_funding_rate(pair_id)
            market_state[pair_id]['funding_rate'] = funding_rate if funding_rate is not None else market_state[pair_id].get('funding_rate', 0.0)
            if candle_data:
                market_state[pair_id]["candle_data"] = candle_data
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos: asyncio.run(check_realtime_position_management(open_pos, candle_data[-1]))
            time.sleep(0.5)
        time.sleep(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'), 'fee': ('fee_pct', '%'), 'delay': ('analysis_interval_sec', 's'),
        'tp_act': ('trailing_tp_activation_pct', '%'), 'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 'winrate': ('target_winrate_pct', '%'),
        'fr_max': ('max_allowed_funding_rate_pct', '%')
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full, unit) in setting_map.items():
            val = current_settings.get(full, 'N/A')
            print_colored(f"{key.capitalize():<10} ({key:<7}) : {val}{unit}", Fore.WHITE)
        return
    if len(parts) == 3 and parts[0] == '!set':
        key, val_str = parts[1].lower(), parts[2]
        if key not in setting_map: print_colored(f"Kunci '{key}' tidak dikenal.", Fore.RED); return
        try:
            full, unit = setting_map[key]
            current_settings[full] = float(val_str)
            save_settings(); print_colored(f"Pengaturan '{full}' diubah menjadi {current_settings[full]}{unit}.", Fore.GREEN)
        except ValueError: print_colored(f"Nilai '{val_str}' tidak valid untuk '{key}'.", Fore.RED)

def run_ai_dashboard():
    try:
        while True:
            print("\033[H\033[J", end="") # Clear screen
            print_colored("--- VULCAN'S AI LIVE DASHBOARD (AUTOPILOT) ---", Fore.CYAN, Style.BRIGHT)
            
            todays_pnl = calculate_todays_pnl(autopilot_trades)
            this_weeks_pnl = calculate_this_weeks_pnl(autopilot_trades)
            pnl_color_today = Fore.GREEN if todays_pnl > 0 else Fore.RED if todays_pnl < 0 else Fore.WHITE
            pnl_color_week = Fore.GREEN if this_weeks_pnl > 0 else Fore.RED if this_weeks_pnl < 0 else Fore.WHITE

            print_colored(f"Today's P/L: ", end=""); print_colored(f"{todays_pnl:.2f}%", pnl_color_today, Style.BRIGHT, end="")
            print_colored(f" | This Week: ", end=""); print_colored(f"{this_weeks_pnl:.2f}%", pnl_color_week, Style.BRIGHT)

            print_colored("="*80, Fore.CYAN)
            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Tekan Ctrl+C dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            
            for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                pair_state = market_state.get(pair_id, {})
                if open_pos:
                    price = pair_state.get('candle_data', [{}])[-1].get('close', open_pos['entryPrice'])
                    pnl_net = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
                    pnl_color = Fore.GREEN if pnl_net > 0 else Fore.RED
                    print_colored(f"  Status: OPEN {open_pos.get('type')} | Entry: {open_pos['entryPrice']:.4f} | PnL(Net): ", end="")
                    print_colored(f"{pnl_net:.2f}%", pnl_color, Style.BRIGHT)
                    if open_pos.get("current_tp_checkpoint_level", 0.0) > 0:
                        cp_level = open_pos["current_tp_checkpoint_level"]
                        ts_price = open_pos.get("trailing_stop_price", 0)
                        print_colored(f"  TP Checkpoint: Aktif @ {cp_level:.2f}% ({ts_price:.4f})", Fore.MAGENTA)
                else:
                    funding_rate = pair_state.get('funding_rate', 0.0)
                    funding_color = Fore.RED if funding_rate > 0.01 else Fore.GREEN if funding_rate < -0.01 else Fore.WHITE
                    last_reason = "Mencari setup..."
                    relevant_trades = sorted([t for t in autopilot_trades if t['instrumentId'] == pair_id], key=lambda x: x['entryTimestamp'], reverse=True)
                    if relevant_trades:
                        last_reason_full = relevant_trades[0].get('entryReason', '')
                        if "PERINGATAN" in last_reason_full or "Alasan AI" in last_reason_full:
                             last_reason = last_reason_full.split('\n')[0]
                    print_colored(f"  Status: Waiting | Funding: ", end=""); print_colored(f"{funding_rate:.4f}%", funding_color)
                    print_colored(f"  AI Log: {last_reason}", Fore.YELLOW)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def run_manual_dashboard():
    while True:
        try:
            print("\033[H\033[J", end="") # Clear screen
            print_colored("--- MANUAL TRADING DASHBOARD ---", Fore.CYAN, Style.BRIGHT)
            print_colored("Mode ini akan menyimpan trade Anda agar AI bisa belajar.", Fore.YELLOW)
            print_colored("="*80, Fore.CYAN)

            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Keluar (Ctrl+C) dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
                time.sleep(5)
                continue

            for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                pair_state = market_state.get(pair_id, {})
                current_price = pair_state.get('candle_data', [{}])[-1].get('close', 0.0)
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                
                price_str = f"{current_price:.4f}" if current_price > 0 else "Menunggu data..."
                print_colored(f"\n⦿ {pair_id} ({timeframe}) | Harga Saat Ini: {price_str}", Fore.WHITE, Style.BRIGHT)

                if open_pos:
                    pnl_net = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
                    pnl_color = Fore.GREEN if pnl_net > 0 else Fore.RED
                    print_colored(f"  Status: OPEN {open_pos.get('type')} | Entry: {open_pos['entryPrice']:.4f} | PnL(Net): ", end="")
                    print_colored(f"{pnl_net:.2f}%", pnl_color, Style.BRIGHT)
                else:
                    print_colored("  Status: Tidak ada posisi terbuka.", Fore.WHITE)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("AKSI YANG TERSEDIA:", Style.BRIGHT)
            print_colored("  long <PAIR>    - Buka posisi LONG (contoh: long BTC-USDT)", Fore.GREEN)
            print_colored("  short <PAIR>   - Buka posisi SHORT (contoh: short ETH-USDT)", Fore.RED)
            print_colored("  close <PAIR>   - Tutup posisi yang ada (contoh: close BTC-USDT)", Fore.YELLOW)
            print_colored("  exit           - Kembali ke menu utama", Fore.WHITE)
            
            user_input = input("\n[Manual Trade] > ").strip().lower()
            parts = user_input.split()
            if not parts: continue

            cmd = parts[0]
            if cmd == 'exit':
                break
            
            if len(parts) < 2:
                print_colored("Perintah tidak lengkap. Contoh: 'long BTC-USDT'", Fore.RED); time.sleep(2)
                continue
            
            action_pair = parts[1].upper()
            if action_pair not in current_settings.get("watched_pairs", {}):
                print_colored(f"Pair '{action_pair}' tidak ada di watchlist.", Fore.RED); time.sleep(2)
                continue

            open_pos = next((t for t in autopilot_trades if t['instrumentId'] == action_pair and t['status'] == 'OPEN'), None)
            
            if cmd in ['long', 'short']:
                if open_pos:
                    print_colored(f"Sudah ada posisi terbuka untuk {action_pair}.", Fore.RED); time.sleep(2)
                    continue
                
                pair_state = market_state.get(action_pair, {})
                current_price = pair_state.get('candle_data', [{}])[-1].get('close')
                if not current_price:
                     print_colored(f"Harga untuk {action_pair} belum tersedia.", Fore.RED); time.sleep(2)
                     continue

                new_trade = {
                    "id": int(time.time()), "instrumentId": action_pair,
                    "type": "LONG" if cmd == "long" else "SHORT",
                    "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": current_price,
                    "entryReason": "Entry Manual oleh Pengguna", "status": 'OPEN', 
                    "entry_snapshot": None,
                    "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
                }
                autopilot_trades.append(new_trade); save_trades()
                print_colored(f"Posisi {new_trade['type']} untuk {action_pair} dibuka @ {current_price:.4f}", Fore.GREEN); time.sleep(2)

            elif cmd == 'close':
                if not open_pos:
                    print_colored(f"Tidak ada posisi terbuka untuk ditutup pada {action_pair}.", Fore.RED); time.sleep(2)
                    continue

                pair_state = market_state.get(action_pair, {})
                current_price = pair_state.get('candle_data', [{}])[-1].get('close')
                if not current_price:
                     print_colored(f"Harga untuk {action_pair} belum tersedia untuk menutup posisi.", Fore.RED); time.sleep(2)
                     continue
                
                asyncio.run(manage_trade_closure(open_pos, current_price, "Penutupan Manual oleh Pengguna (Web)"))
                print_colored(f"Posisi untuk {action_pair} ditutup @ {current_price:.4f}", Fore.GREEN); time.sleep(2)

        except (KeyboardInterrupt, EOFError):
            break

# --- WEB FLASK INTEGRATION START ---

if Flask:
    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vulcan AI - Real-time Dashboard</title>
        <style>
            :root {
                --bg-color: #121212;
                --card-color: #1e1e1e;
                --text-color: #e0e0e0;
                --text-secondary-color: #b0b0b0;
                --border-color: #333;
                --green-color: #4CAF50;
                --red-color: #F44336;
                --yellow-color: #FFC107;
                --blue-color: #2196F3;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: auto;
            }
            h1, h2 {
                color: var(--blue-color);
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }
            .pnl-summary {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .pnl-card {
                background-color: var(--card-color);
                padding: 15px 20px;
                border-radius: 8px;
                border-left: 5px solid var(--blue-color);
                flex-grow: 1;
            }
            .pnl-card h3 {
                margin: 0 0 5px 0;
                color: var(--text-secondary-color);
                font-size: 1em;
            }
            .pnl-value { font-size: 1.5em; font-weight: bold; }
            .pairs-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px;
            }
            .pair-card {
                background-color: var(--card-color);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            .pair-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .pair-header h2 {
                margin: 0;
                border: none;
                font-size: 1.5em;
            }
            .pair-price { font-size: 1.2em; }
            .status-waiting { color: var(--text-secondary-color); }
            .status-open { font-weight: bold; }
            .pnl-positive { color: var(--green-color); }
            .pnl-negative { color: var(--red-color); }
            .reason-box {
                background-color: rgba(255, 255, 255, 0.05);
                padding: 10px;
                border-radius: 5px;
                font-size: 0.9em;
                color: var(--text-secondary-color);
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: auto;
            }
            .btn {
                flex-grow: 1;
                padding: 10px;
                border: none;
                border-radius: 5px;
                color: white;
                font-size: 1em;
                cursor: pointer;
                transition: opacity 0.2s;
            }
            .btn:hover { opacity: 0.8; }
            .btn-long { background-color: var(--green-color); }
            .btn-short { background-color: var(--red-color); }
            .btn-close { background-color: var(--yellow-color); color: #121212; }
            .btn:disabled { background-color: #555; cursor: not-allowed; }
            .footer { text-align: center; margin-top: 30px; color: var(--text-secondary-color); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vulcan AI - Dashboard</h1>
            <div class="pnl-summary">
                <div class="pnl-card">
                    <h3>Today's PnL</h3>
                    <p id="pnl-today" class="pnl-value">0.00%</p>
                </div>
                <div class="pnl-card">
                    <h3>This Week's PnL</h3>
                    <p id="pnl-week" class="pnl-value">0.00%</p>
                </div>
                <div class="pnl-card">
                    <h3>AI Autopilot Status</h3>
                    <p id="ai-status" class="pnl-value">Inactive</p>
                </div>
            </div>
            <h2>Market Watchlist</h2>
            <div id="pairs-grid" class="pairs-grid">
                <!-- Pair cards will be dynamically inserted here -->
            </div>
        </div>
        <div class="footer">
            <p>Data auto-refreshes every second. Page loaded at: <span id="load-time"></span></p>
        </div>

        <script>
            function setPnlColor(element, value) {
                element.classList.remove('pnl-positive', 'pnl-negative');
                if (value > 0) {
                    element.classList.add('pnl-positive');
                } else if (value < 0) {
                    element.classList.add('pnl-negative');
                }
            }
            
            async function executeTrade(action, pair) {
                const endpoint = action === 'close' ? '/api/trade/close' : '/api/trade/open';
                const body = { pair: pair };
                if (action !== 'close') {
                    body.type = action.toUpperCase();
                }

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const result = await response.json();
                    console.log(result.message);
                    updateData(); // Refresh data immediately after action
                } catch (error) {
                    console.error('Error executing trade:', error);
                }
            }

            function updateData() {
                fetch('/api/data')
                    .then(response => response.json())
                    .then(data => {
                        // Update PnL Summary
                        const pnlTodayEl = document.getElementById('pnl-today');
                        pnlTodayEl.textContent = data.pnl.today.toFixed(2) + '%';
                        setPnlColor(pnlTodayEl, data.pnl.today);
                        
                        const pnlWeekEl = document.getElementById('pnl-week');
                        pnlWeekEl.textContent = data.pnl.week.toFixed(2) + '%';
                        setPnlColor(pnlWeekEl, data.pnl.week);

                        const aiStatusEl = document.getElementById('ai-status');
                        aiStatusEl.textContent = data.ai_status ? 'Active' : 'Inactive';
                        aiStatusEl.className = 'pnl-value ' + (data.ai_status ? 'pnl-positive' : 'pnl-negative');

                        // Update Pair Cards
                        const grid = document.getElementById('pairs-grid');
                        grid.innerHTML = ''; // Clear old cards

                        for (const pairId in data.pairs) {
                            const pair = data.pairs[pairId];
                            
                            let statusHtml;
                            let buttonsHtml;
                            
                            if (pair.open_position) {
                                const pos = pair.open_position;
                                const pnlColorClass = pos.pnl_net > 0 ? 'pnl-positive' : (pos.pnl_net < 0 ? 'pnl-negative' : '');
                                statusHtml = `
                                    <div class="status-open">
                                        Status: OPEN ${pos.type} @ ${pos.entryPrice.toFixed(4)}
                                    </div>
                                    <div>
                                        PnL (Net): <span class="${pnlColorClass}">${pos.pnl_net.toFixed(2)}%</span>
                                    </div>
                                    <div class="reason-box">Trigger: ${pos.entryReason.split('\\n')[0]}</div>
                                `;
                                buttonsHtml = `
                                    <button class="btn btn-long" disabled>Long</button>
                                    <button class="btn btn-short" disabled>Short</button>
                                    <button class="btn btn-close" onclick="executeTrade('close', '${pairId}')">Close</button>
                                `;
                            } else {
                                statusHtml = `
                                    <div class="status-waiting">Status: Waiting for signal...</div>
                                    <div>Funding: ${pair.funding_rate.toFixed(4)}%</div>
                                    <div class="reason-box">AI Log: ${pair.last_reason.split('\\n')[0]}</div>
                                `;
                                buttonsHtml = `
                                    <button class="btn btn-long" onclick="executeTrade('long', '${pairId}')">Long</button>
                                    <button class="btn btn-short" onclick="executeTrade('short', '${pairId}')">Short</button>
                                    <button class="btn btn-close" disabled>Close</button>
                                `;
                            }

                            const cardHtml = \`
                                <div class="pair-card">
                                    <div class="pair-header">
                                        <h2>${pairId}</h2>
                                        <span class="pair-price">${pair.current_price > 0 ? pair.current_price.toFixed(4) : 'Loading...'}</span>
                                    </div>
                                    ${statusHtml}
                                    <div class="button-group">
                                        ${buttonsHtml}
                                    </div>
                                </div>
                            \`;
                            grid.innerHTML += cardHtml;
                        }
                    })
                    .catch(error => console.error('Error fetching data:', error));
            }

            document.addEventListener('DOMContentLoaded', () => {
                document.getElementById('load-time').textContent = new Date().toLocaleTimeString();
                updateData();
                setInterval(updateData, 1000); // Poll every second
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/data')
    def get_data():
        data = {
            'pnl': {
                'today': calculate_todays_pnl(autopilot_trades),
                'week': calculate_this_weeks_pnl(autopilot_trades)
            },
            'ai_status': is_autopilot_running,
            'pairs': {}
        }
        for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
            pair_state = market_state.get(pair_id, {})
            current_price = pair_state.get('candle_data', [{}])[-1].get('close', 0.0)
            open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
            
            pair_data = {
                'timeframe': timeframe,
                'current_price': current_price,
                'funding_rate': pair_state.get('funding_rate', 0.0),
                'open_position': None,
                'last_reason': "Mencari setup..."
            }

            if open_pos:
                pnl_net = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
                pair_data['open_position'] = {
                    'type': open_pos.get('type'),
                    'entryPrice': open_pos['entryPrice'],
                    'entryReason': open_pos.get('entryReason', 'N/A'),
                    'pnl_net': pnl_net
                }
            else:
                 relevant_trades = sorted([t for t in autopilot_trades if t['instrumentId'] == pair_id], key=lambda x: x['entryTimestamp'], reverse=True)
                 if relevant_trades:
                     last_reason_full = relevant_trades[0].get('entryReason', '')
                     if "PERINGATAN" in last_reason_full or "Alasan AI" in last_reason_full:
                          pair_data['last_reason'] = last_reason_full

            data['pairs'][pair_id] = pair_data
        return jsonify(data)

    @app.route('/api/trade/open', methods=['POST'])
    def open_trade():
        req_data = request.get_json()
        pair_id = req_data.get('pair')
        trade_type = req_data.get('type')

        if not all([pair_id, trade_type]):
            return jsonify({'status': 'error', 'message': 'Pair and type are required'}), 400

        open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        if open_pos:
            return jsonify({'status': 'error', 'message': f'Position already open for {pair_id}'}), 409
        
        pair_state = market_state.get(pair_id, {})
        current_price = pair_state.get('candle_data', [{}])[-1].get('close')
        if not current_price:
            return jsonify({'status': 'error', 'message': f'Price data not available for {pair_id}'}), 404

        new_trade = {
            "id": int(time.time()), "instrumentId": pair_id,
            "type": trade_type,
            "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": current_price,
            "entryReason": "Entry Manual oleh Pengguna (Web)", "status": 'OPEN', 
            "entry_snapshot": None,
            "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
        }
        autopilot_trades.append(new_trade)
        save_trades()
        return jsonify({'status': 'success', 'message': f'{trade_type} position opened for {pair_id} at {current_price}'})

    @app.route('/api/trade/close', methods=['POST'])
    def close_trade():
        req_data = request.get_json()
        pair_id = req_data.get('pair')
        if not pair_id:
            return jsonify({'status': 'error', 'message': 'Pair is required'}), 400

        open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        if not open_pos:
            return jsonify({'status': 'error', 'message': f'No open position for {pair_id}'}), 404

        pair_state = market_state.get(pair_id, {})
        current_price = pair_state.get('candle_data', [{}])[-1].get('close')
        if not current_price:
            return jsonify({'status': 'error', 'message': f'Price data not available for {pair_id}'}), 404

        asyncio.run(manage_trade_closure(open_pos, current_price, "Penutupan Manual oleh Pengguna (Web)"))
        return jsonify({'status': 'success', 'message': f'Position closed for {pair_id} at {current_price}'})

def run_flask():
    if Flask:
        # Menjalankan Flask di host 0.0.0.0 agar bisa diakses dari perangkat lain di jaringan yang sama
        app.run(host='0.0.0.0', port=5001, debug=False)

# --- WEB FLASK INTEGRATION END ---


def main():
    global is_autopilot_running
    load_settings(); load_trades()
    
    # --- Start Flask in a separate thread ---
    if Flask:
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
    
    display_welcome_message()

    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    autopilot_thread.start()
    data_thread.start()

    while True:
        try:
            user_input = input("[Command] > ").strip()
            if not user_input: continue
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if not current_settings.get("watched_pairs"):
                    print_colored("Watchlist kosong. Gunakan '!watch <PAIR>'.", Fore.RED); continue
                is_autopilot_running = True
                print_colored("✅ Autopilot AI diaktifkan. Memasuki Live Dashboard...", Fore.GREEN)
                run_ai_dashboard()
                is_autopilot_running = False
                print_colored("\n🛑 Dashboard AI ditutup.", Fore.RED)
            elif cmd == '!start_manual':
                if not current_settings.get("watched_pairs"):
                    print_colored("Watchlist kosong. Gunakan '!watch <PAIR>'.", Fore.RED); continue
                print_colored(" Memasuki Dashboard Trading Manual...", Fore.GREEN)
                run_manual_dashboard()
                print_colored("\n🛑 Dashboard Manual ditutup.", Fore.RED)
            elif cmd == '!watch':
                if len(parts) >= 2:
                    pair_id = parts[1].upper(); tf = parts[2] if len(parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"{pair_id} ({tf}) ditambahkan ke watchlist.", Fore.GREEN)
                else: print_colored("Format: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(parts) == 2:
                    pair_id = parts[1].upper()
                    if current_settings['watched_pairs'].pop(pair_id, None):
                        save_settings(); print_colored(f"{pair_id} dihapus.", Fore.YELLOW)
                    else: print_colored(f"{pair_id} tidak ditemukan.", Fore.RED)
                else: print_colored("Format: !unwatch <PAIR>", Fore.RED)
            elif cmd == '!watchlist':
                watched = current_settings.get("watched_pairs", {})
                if not watched: print_colored("Watchlist kosong.", Fore.YELLOW)
                else:
                    print_colored("\n--- Watchlist ---", Fore.CYAN, Style.BRIGHT)
                    for pair, tf in watched.items(): print_colored(f"- {pair} ({tf})", Fore.WHITE)
            elif cmd == '!history': display_history()
            elif cmd in ['!settings', '!set']: handle_settings_command(parts)
            else: print_colored(f"Perintah '{cmd}' tidak dikenal.", Fore.RED)
        except (KeyboardInterrupt, EOFError): break
        except Exception as e: print_colored(f"\nError di main loop: {e}", Fore.RED)
    
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
