import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math
import logging

# --- Prasyarat Web ---
try:
    from flask import Flask, jsonify, request, render_template_string
except ImportError:
    print("Peringatan: Pustaka 'Flask' tidak ditemukan. Fitur web tidak akan tersedia.")
    print("Silakan install dengan 'pip install Flask'")
    Flask = None

# --- Nonaktifkan logging default Flask untuk menjaga kebersihan CLI ---
if Flask:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

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
        print_colored("Web Dashboard aktif di http://127.0.0.1:5001 (dan di IP lokal Anda)", Fore.GREEN)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk Live Dashboard AI di CLI.", Fore.YELLOW)
    print_colored("Gunakan '!start_manual' untuk Live Dashboard Manual di CLI.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot AI (CLI)", Fore.GREEN)
    print_colored("!start_manual         - Masuk ke Dashboard Trading Manual (CLI)", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade (terbatas 80 terakhir)", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, winrate, fr_max)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

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
        # No print here to avoid cluttering CLI from web actions
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

def calculate_pnl(entry_price, current_price, trade_type):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

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
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() >= start_of_week_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

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
            if not past_snapshot or past_snapshot.get('bias') != current_analysis['bias']: continue
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
            time.sleep(0.2) # More frequent updates for web
        time.sleep(REFRESH_INTERVAL_SECONDS)

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
                --bg-color: #121212; --card-color: #1e1e1e; --text-color: #e0e0e0;
                --text-secondary-color: #b0b0b0; --border-color: #333; --green-color: #4CAF50;
                --red-color: #F44336; --yellow-color: #FFC107; --blue-color: #2196F3;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 20px;
                -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
            }
            .container { max-width: 1300px; margin: auto; }
            .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; margin-bottom: 20px;}
            .header h1 { color: var(--blue-color); margin: 0; }
            .ai-toggle { display: flex; align-items: center; gap: 10px; background-color: var(--card-color); padding: 10px 15px; border-radius: 25px;}
            .switch { position: relative; display: inline-block; width: 50px; height: 28px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 28px;}
            .slider:before { position: absolute; content: ""; height: 20px; width: 20px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%;}
            input:checked + .slider { background-color: var(--green-color); }
            input:checked + .slider:before { transform: translateX(22px); }
            
            .pnl-summary { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
            .pnl-card { background-color: var(--card-color); padding: 15px 20px; border-radius: 8px; border-left: 5px solid var(--blue-color); flex-grow: 1; }
            .pnl-card h3 { margin: 0 0 5px 0; color: var(--text-secondary-color); font-size: 1em; text-transform: uppercase; letter-spacing: 0.5px; }
            .pnl-value { font-size: 1.5em; font-weight: bold; }
            .pairs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }
            .pair-card { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; display: flex; flex-direction: column; gap: 12px; transition: box-shadow 0.3s; }
            .pair-card:hover { box-shadow: 0 0 15px rgba(33, 150, 243, 0.2); }
            .pair-header { display: flex; justify-content: space-between; align-items: baseline; }
            .pair-name { font-size: 1.6em; font-weight: bold; color: var(--text-color);}
            .pair-timeframe { font-size: 0.9em; color: var(--text-secondary-color); }
            .price-info { display: flex; justify-content: space-between; align-items: center; }
            .pair-price { font-size: 1.2em; color: var(--blue-color);}
            .funding-rate { font-size: 0.9em; }
            .status-box { padding: 12px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05); }
            .status-line { display: flex; justify-content: space-between; }
            .pnl-positive { color: var(--green-color) !important; }
            .pnl-negative { color: var(--red-color) !important; }
            .reason-box { font-size: 0.85em; color: var(--text-secondary-color); margin-top: 8px; white-space: pre-wrap; word-wrap: break-word; }
            .button-group { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: auto; }
            .btn { padding: 12px; border: none; border-radius: 5px; color: white; font-size: 1em; font-weight: bold; cursor: pointer; transition: opacity 0.2s; }
            .btn:hover:not(:disabled) { opacity: 0.8; }
            .btn-long { background-color: var(--green-color); }
            .btn-short { background-color: var(--red-color); }
            .btn-close { background-color: var(--yellow-color); color: #121212; }
            .btn:disabled { background-color: #444; color: #888; cursor: not-allowed; }
            #toast { visibility: hidden; min-width: 250px; background-color: #333; color: #fff; text-align: center; border-radius: 2px; padding: 16px; position: fixed; z-index: 1; left: 50%; transform: translateX(-50%); bottom: 30px; }
            #toast.show { visibility: visible; -webkit-animation: fadein 0.5s, fadeout 0.5s 2.5s; animation: fadein 0.5s, fadeout 0.5s 2.5s; }
            @-webkit-keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 30px; opacity: 1;} }
            @keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 30px; opacity: 1;} }
            @-webkit-keyframes fadeout { from {bottom: 30px; opacity: 1;} to {bottom: 0; opacity: 0;} }
            @keyframes fadeout { from {bottom: 30px; opacity: 1;} to {bottom: 0; opacity: 0;} }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Vulcan AI Dashboard</h1>
                <div class="ai-toggle">
                    <span>AI Autopilot</span>
                    <label class="switch">
                        <input type="checkbox" id="ai-toggle-switch" onchange="toggleAI()">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>
            <div class="pnl-summary">
                <div class="pnl-card">
                    <h3>Today's PnL</h3>
                    <p id="pnl-today" class="pnl-value">0.00%</p>
                </div>
                <div class="pnl-card">
                    <h3>This Week's PnL</h3>
                    <p id="pnl-week" class="pnl-value">0.00%</p>
                </div>
            </div>
            <div id="pairs-grid" class="pairs-grid"></div>
        </div>
        <div id="toast"></div>

        <script>
            function showToast(message) {
                const toast = document.getElementById("toast");
                toast.textContent = message;
                toast.className = "show";
                setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
            }

            async function toggleAI() {
                const isChecked = document.getElementById('ai-toggle-switch').checked;
                try {
                    const response = await fetch('/api/toggle_ai', { method: 'POST' });
                    const result = await response.json();
                    showToast(result.message);
                    updateDashboard(); 
                } catch (error) {
                    console.error('Error toggling AI:', error);
                    showToast('Error changing AI status');
                }
            }

            async function executeTrade(action, pair) {
                const endpoint = action === 'close' ? '/api/trade/close' : '/api/trade/open';
                const body = { pair: pair };
                if (action !== 'close') body.type = action.toUpperCase();

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const result = await response.json();
                    showToast(result.message);
                    if (result.status === 'success') updateDashboard();
                } catch (error) {
                    console.error('Error executing trade:', error);
                    showToast('Error executing trade');
                }
            }

            function setPnlColor(element, value) {
                element.classList.remove('pnl-positive', 'pnl-negative');
                if (value > 0) element.classList.add('pnl-positive');
                else if (value < 0) element.classList.add('pnl-negative');
            }

            function updateDashboard() {
                fetch('/api/dashboard_data')
                    .then(response => response.json())
                    .then(data => {
                        // Update PnL Summary
                        const pnlTodayEl = document.getElementById('pnl-today');
                        pnlTodayEl.textContent = data.pnl.today.toFixed(2) + '%';
                        setPnlColor(pnlTodayEl, data.pnl.today);
                        
                        const pnlWeekEl = document.getElementById('pnl-week');
                        pnlWeekEl.textContent = data.pnl.week.toFixed(2) + '%';
                        setPnlColor(pnlWeekEl, data.pnl.week);

                        // Update AI Toggle Switch
                        document.getElementById('ai-toggle-switch').checked = data.ai_status;

                        // Update Pair Cards
                        const grid = document.getElementById('pairs-grid');
                        grid.innerHTML = ''; 

                        for (const pairId in data.pairs) {
                            const pair = data.pairs[pairId];
                            
                            let statusHtml;
                            let buttonsHtml;
                            
                            if (pair.open_position) {
                                const pos = pair.open_position;
                                const pnlColorClass = pos.pnl_net > 0 ? 'pnl-positive' : 'pnl-negative';
                                statusHtml = `
                                    <div class="status-box">
                                        <div class="status-line">
                                            <span>Status: <strong>OPEN ${pos.type}</strong></span>
                                            <span>Entry: <strong>${pos.entryPrice.toFixed(4)}</strong></span>
                                        </div>
                                        <div class="status-line" style="margin-top: 5px;">
                                            <span>PnL (Net):</span>
                                            <strong class="${pnlColorClass}">${pos.pnl_net.toFixed(2)}%</strong>
                                        </div>
                                        <div class="reason-box"><strong>Trigger:</strong> ${pos.entryReason.split('\\n')[0]}</div>
                                    </div>
                                `;
                                buttonsHtml = `
                                    <button class="btn btn-long" disabled>Long</button>
                                    <button class="btn btn-short" disabled>Short</button>
                                    <button class="btn btn-close" onclick="executeTrade('close', '${pairId}')">Close</button>
                                `;
                            } else {
                                const fundingColorClass = Math.abs(pair.funding_rate) > 0.05 ? (pair.funding_rate > 0 ? 'pnl-negative' : 'pnl-positive') : '';
                                statusHtml = `
                                    <div class="status-box">
                                        <div class="status-line">
                                            <span>Status: <strong>Waiting for Signal</strong></span>
                                            <span>Funding: <strong class="${fundingColorClass}">${pair.funding_rate.toFixed(4)}%</strong></span>
                                        </div>
                                        <div class="reason-box"><strong>AI Log:</strong> ${pair.last_reason.split('\\n')[0]}</div>
                                    </div>
                                `;
                                buttonsHtml = `
                                    <button class="btn btn-long" onclick="executeTrade('long', '${pairId}')">Long</button>
                                    <button class="btn btn-short" onclick="executeTrade('short', '${pairId}')">Short</button>
                                    <button class="btn btn-close" disabled>Close</button>
                                `;
                            }
                            const priceFormatted = pair.current_price > 0 ? pair.current_price.toFixed(4) : 'Loading...';
                            const cardHtml = \`
                                <div class="pair-card">
                                    <div class="pair-header">
                                        <span class="pair-name">${pairId} <span class="pair-timeframe">(${pair.timeframe})</span></span>
                                        <span class="pair-price">${priceFormatted}</span>
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
                updateDashboard();
                setInterval(updateDashboard, 1500);
            });
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/dashboard_data')
    def get_dashboard_data():
        data = {
            'pnl': { 'today': calculate_todays_pnl(autopilot_trades), 'week': calculate_this_weeks_pnl(autopilot_trades) },
            'ai_status': is_autopilot_running,
            'pairs': {}
        }
        for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
            pair_state = market_state.get(pair_id, {})
            current_price = pair_state.get('candle_data', [{}])[-1].get('close', 0.0)
            open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
            
            pair_data = {
                'timeframe': timeframe, 'current_price': current_price,
                'funding_rate': pair_state.get('funding_rate', 0.0), 'open_position': None, 'last_reason': "Menunggu data..."
            }
            if open_pos:
                pnl_net = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
                pair_data['open_position'] = {
                    'type': open_pos.get('type'), 'entryPrice': open_pos['entryPrice'],
                    'entryReason': open_pos.get('entryReason', 'N/A'), 'pnl_net': pnl_net
                }
            else:
                 relevant_trades = sorted([t for t in autopilot_trades if t['instrumentId'] == pair_id], key=lambda x: x['entryTimestamp'], reverse=True)
                 if relevant_trades:
                     last_reason_full = relevant_trades[0].get('entryReason', 'Menunggu setup...')
                     pair_data['last_reason'] = last_reason_full
            data['pairs'][pair_id] = pair_data
        return jsonify(data)

    @app.route('/api/toggle_ai', methods=['POST'])
    def toggle_ai_status():
        global is_autopilot_running
        is_autopilot_running = not is_autopilot_running
        status_str = "diaktifkan" if is_autopilot_running else "dimatikan"
        return jsonify({'status': 'success', 'message': f'AI Autopilot {status_str}', 'ai_status': is_autopilot_running})

    @app.route('/api/trade/open', methods=['POST'])
    def open_trade():
        req_data = request.get_json()
        pair_id = req_data.get('pair'); trade_type = req_data.get('type')
        if not all([pair_id, trade_type]): return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400
        if next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None):
            return jsonify({'status': 'error', 'message': f'Posisi sudah terbuka untuk {pair_id}'}), 409
        
        current_price = market_state.get(pair_id, {}).get('candle_data', [{}])[-1].get('close')
        if not current_price: return jsonify({'status': 'error', 'message': f'Data harga tidak tersedia'}), 404
        
        new_trade = {
            "id": int(time.time()), "instrumentId": pair_id, "type": trade_type.upper(),
            "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": current_price,
            "entryReason": "Entry Manual (Web)", "status": 'OPEN', "entry_snapshot": None,
            "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
        }
        autopilot_trades.append(new_trade); save_trades()
        return jsonify({'status': 'success', 'message': f'{trade_type} {pair_id} dibuka @ {current_price:.4f}'})

    @app.route('/api/trade/close', methods=['POST'])
    def close_trade():
        pair_id = request.get_json().get('pair')
        if not pair_id: return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400

        open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        if not open_pos: return jsonify({'status': 'error', 'message': f'Tidak ada posisi terbuka'}), 404

        current_price = market_state.get(pair_id, {}).get('candle_data', [{}])[-1].get('close')
        if not current_price: return jsonify({'status': 'error', 'message': f'Data harga tidak tersedia'}), 404

        asyncio.run(manage_trade_closure(open_pos, current_price, "Penutupan Manual (Web)"))
        pnl = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
        return jsonify({'status': 'success', 'message': f'{pair_id} ditutup @ {current_price:.4f} dengan PnL {pnl:.2f}%'})

def run_flask():
    if Flask: app.run(host='0.0.0.0', port=5001)

# --- WEB FLASK INTEGRATION END ---

def main():
    global is_autopilot_running
    load_settings(); load_trades()
    
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
                if not current_settings.get("watched_pairs"): print_colored("Watchlist kosong. Gunakan '!watch <PAIR>'.", Fore.RED); continue
                is_autopilot_running = True
                print_colored("✅ Autopilot AI diaktifkan. Mode dashboard CLI tidak lagi tersedia, gunakan dashboard web.", Fore.GREEN)
                print_colored("Tekan Ctrl+C untuk menonaktifkan AI dan kembali ke menu.", Fore.YELLOW)
                try:
                    while True: time.sleep(1) # Loop to keep CLI busy, web dashboard is the main UI
                except KeyboardInterrupt:
                    is_autopilot_running = False
                    print_colored("\n🛑 Autopilot AI dinonaktifkan.", Fore.RED)
            # Menghapus dashboard CLI manual karena sudah digantikan oleh web
            elif cmd == '!start_manual':
                print_colored("Mode ini telah digantikan oleh dashboard web.", Fore.YELLOW)
                print_colored("Buka http://127.0.0.1:5001 di browser Anda.", Fore.CYAN)
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
            elif cmd in ['!settings', '!set']: print_colored("Pengaturan dipindahkan ke file settings.json", Fore.YELLOW)
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
