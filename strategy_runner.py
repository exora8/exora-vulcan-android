import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math
from flask import Flask, render_template_string, jsonify, redirect, url_for, request

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
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data/v2/"
REFRESH_INTERVAL_SECONDS = 0.5
BACKTEST_FETCH_CHUNK_LIMIT = 1000
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
state_lock = threading.Lock() # Lock untuk thread-safety

# --- INISIALISASI FLASK ---
app = Flask(__name__)
# Menonaktifkan logging request Flask standar agar konsol lebih bersih
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


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
    print_colored("           -- MODERN UI & API EDITION --          ", Fore.YELLOW, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Bot sedang berjalan di latar belakang.", Fore.YELLOW)
    print_colored("Buka browser dan akses:", Fore.GREEN, Style.BRIGHT)
    print_colored("http://127.0.0.1:5000 atau http://[IP_LOKAL_ANDA]:5000", Fore.GREEN, Style.BRIGHT)
    print()


# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 0.20, "fee_pct": 0.1, "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30, "trailing_tp_gap_pct": 0.05,
        "caution_level": 0.5, "target_winrate_pct": 85.0,
        "cryptocompare_api_key": "YOUR_CRYPTOCOMPARE_API_KEY",
        "max_allowed_funding_rate_pct": 0.075, "watched_pairs": {"BTC-USDT": "1H", "ETH-USDT": "1H"}
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
    with state_lock:
        autopilot_trades.sort(key=lambda x: x['entryTimestamp'], reverse=True) # Urutkan terbaru di atas
        if len(autopilot_trades) > MAX_TRADES_IN_HISTORY:
            num_to_trim = len(autopilot_trades) - MAX_TRADES_IN_HISTORY
            autopilot_trades = autopilot_trades[:MAX_TRADES_IN_HISTORY]
            print_colored(f"Riwayat trade dibatasi. {num_to_trim} trade tertua telah dihapus.", Fore.YELLOW)
        try:
            with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)
        except IOError as e: print_colored(f"Error saving trades: {e}", Fore.RED)

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
# ... (Fungsi-fungsi kalkulasi PNL lainnya tetap sama)
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
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date()
                if start_of_week_utc <= exit_date <= today_utc:
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
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date()
                if start_of_last_week_utc <= exit_date <= end_of_last_week_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

# --- OTAK LOCAL AI (Tetap sama, tidak ada perubahan logika) ---
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
        analysis = { "ema9_current": ema9[-1], "ema9_prev": ema9[-2], "ema50": ema50[-1], "ema100": ema100[-1], "current_candle_close": candle_data[-1]['close'], "prev_candle_close": candle_data[-2]['close'], "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING", }
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
                reason = (f"PERINGATAN: Setup ini {similarity_score*25}% mirip dengan loss sebelumnya (ID: {loss['id']}).\n" f"  - Kedua setup memiliki bias {current_analysis['bias']} dan harga di {current_pos_vs_ema50} EMA50.")
                return (True, reason)
        return (False, None)
    def get_decision(self, candle_data, open_position, funding_rate=0.0):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data teknikal tidak cukup untuk analisis."}
        if open_position: return {"action": "HOLD", "reason": "Sudah ada posisi terbuka, sedang memantau."}
        max_funding_rate = self.settings.get("max_allowed_funding_rate_pct", 0.075)
        potential_trade_type = None
        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']: potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']: potential_trade_type = 'SHORT'
        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal LONG diabaikan. Funding rate terlalu tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal SHORT diabaikan. Funding rate terlalu negatif: {funding_rate:.4f}%"}
            is_repeated_mistake, warning_reason = self.check_for_repeated_mistake(analysis)
            if is_repeated_mistake: return {"action": "HOLD", "reason": warning_reason}
            ai_reason = (f"Alasan AI (tinyllama): Entry {potential_trade_type} diambil berdasarkan konfirmasi kelanjutan tren. " f"Pasar menunjukkan bias {analysis['bias']} yang kuat, dan sinyal ini muncul setelah harga melakukan pullback sehat ke EMA9, " "menunjukkan potensi kekuatan untuk melanjutkan pergerakan.")
            avg_solidity = sum(analysis['pre_entry_candle_solidity']) / 3
            bot_details = (f"Detail Bot: \n" f"  - Bias Pasar: {analysis['bias']} (EMA50: {analysis['ema50']:.2f} vs EMA100: {analysis['ema100']:.2f})\n" f"  - Trigger: Candle close ({analysis['current_candle_close']:.2f}) melintasi EMA9 ({analysis['ema9_current']:.2f})\n" f"  - Kondisi Pra-Entry: Arah 3 candle = {analysis['pre_entry_candle_direction']}, Rata-rata Soliditas = {avg_solidity:.2f}")
            full_reason = f"{ai_reason}\n{bot_details}"
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": full_reason, "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias saat ini: {analysis['bias']}."}

# --- LOGIKA TRADING & THREAD WORKERS (Tetap sama) ---
async def analyze_and_close_trade(trade, exit_price, reason, is_backtest=False, exit_timestamp_ms=None):
    with state_lock:
        pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
        exit_dt = datetime.fromtimestamp(exit_timestamp_ms / 1000) if exit_timestamp_ms else datetime.utcnow()
        trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })
        is_profit = (pnl_gross - current_settings.get('fee_pct', 0.1)) > 0
        if is_profit and 'entry_snapshot' in trade:
            try: del trade['entry_snapshot']
            except KeyError: pass
    if not is_backtest:
        save_trades()
        pnl_net = pnl_gross - current_settings.get('fee_pct', 0.1)
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | Trigger: {reason}"
        send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.MAGENTA)
async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3: return
    is_ai_thinking = True
    try:
        with state_lock: open_pos = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        ai = LocalAI(current_settings, relevant_trades)
        funding_rate = pair_state.get("funding_rate", 0.0)
        decision = ai.get_decision(pair_state["candle_data"], open_pos, funding_rate)
        if decision.get('action') in ["BUY", "SELL"] and not open_pos:
            snapshot = decision.get("snapshot", {}); snapshot["funding_rate"] = funding_rate
            new_trade = { "id": int(time.time()), "instrumentId": instrument_id, "type": "LONG" if decision['action'] == "BUY" else "SHORT", "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": pair_state["candle_data"][-1]['close'], "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot, "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0 }
            with state_lock: autopilot_trades.append(new_trade)
            save_trades()
            ai_reason_short = decision.get("reason").split('\n')[0]
            notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {instrument_id}"; notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | {ai_reason_short}"
            send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.GREEN)
    except Exception as e: print_colored(f"Error dalam autopilot analysis: {e}", Fore.RED)
    finally: is_ai_thinking = False
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            for pair_id in list(current_settings.get("watched_pairs", {})): asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)
async def check_realtime_position_management(trade_obj, current_candle_data, is_backtest=False):
    if not trade_obj: return
    with state_lock:
        if trade_obj['type'] == 'LONG':
            pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG'); pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'LONG')
            if pnl_at_high > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_high
            if pnl_at_low < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_low
        elif trade_obj['type'] == 'SHORT':
            pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT'); pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'SHORT')
            if pnl_at_low > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_low
            if pnl_at_high < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_high
        sl_pct = current_settings.get('stop_loss_pct')
        sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
        if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
            await analyze_and_close_trade(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}%", is_backtest, current_candle_data['time']); return
        activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30); gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
        if trade_obj.get("current_tp_checkpoint_level", 0.0) > 0.0:
            ts_price = trade_obj.get('trailing_stop_price')
            if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
                await analyze_and_close_trade(trade_obj, ts_price, f"Trailing TP", is_backtest, current_candle_data['time']); return
        pnl_now = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high' if trade_obj['type'] == 'LONG' else 'low'], trade_obj['type'])
        if pnl_now >= activation_pct:
            current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
            if current_cp == 0.0: current_cp = activation_pct
            steps_passed = math.floor((pnl_now - current_cp) / gap_pct)
            if steps_passed >= 0:
                new_cp = current_cp + (steps_passed * gap_pct); trade_obj['current_tp_checkpoint_level'] = new_cp
                new_ts_level = new_cp - gap_pct; trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + new_ts_level / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 - new_ts_level / 100)
    if not is_backtest: save_trades()
def data_refresh_worker():
    while not stop_event.is_set():
        for pair_id, timeframe in list(current_settings.get("watched_pairs", {}).items()):
            if pair_id not in market_state: market_state[pair_id] = {}
            candle_data = fetch_recent_candles(pair_id, timeframe); funding_rate = fetch_funding_rate(pair_id)
            market_state[pair_id]['funding_rate'] = funding_rate if funding_rate is not None else market_state[pair_id].get('funding_rate', 0.0)
            if candle_data:
                market_state[pair_id]["candle_data"] = candle_data
                with state_lock: open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos: asyncio.run(check_realtime_position_management(open_pos, candle_data[-1]))
            time.sleep(0.5)
        time.sleep(REFRESH_INTERVAL_SECONDS)


# --- TEMPLATE HTML UNTUK DASHBOARD FLASK ---
HTML_SKELETON = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vulcan AI Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #101419;
            --card-color: #1a2028;
            --border-color: #2a3441;
            --text-color: #e1e8f0;
            --text-muted: #8493a6;
            --accent-glow: #00bfff;
            --green: #28a745;
            --red: #dc3545;
            --yellow: #ffc107;
        }
        * { box-sizing: border-box; }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 1.5rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .header h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--accent-glow);
            text-shadow: 0 0 5px rgba(0, 191, 255, 0.5);
        }
        .ai-toggle-btn {
            padding: 0.75rem 1.5rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .ai-toggle-btn.running {
            background-color: var(--green);
            color: white;
            border-color: var(--green);
            box-shadow: 0 0 10px rgba(40, 167, 69, 0.7);
        }
        .ai-toggle-btn.stopped {
            background-color: var(--red);
            color: white;
            border-color: var(--red);
            box-shadow: 0 0 10px rgba(220, 53, 69, 0.7);
        }
        .pnl-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            background-color: var(--card-color);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }
        .stat-item { text-align: center; }
        .stat-item .label { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.25rem; }
        .stat-item .value { font-size: 1.5rem; font-weight: 700; transition: color 0.3s ease; }
        
        .watchlist {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }
        .pair-card {
            background-color: var(--card-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .pair-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
        .pair-card.position-open {
            border-color: var(--accent-glow);
            box-shadow: 0 0 15px rgba(0, 191, 255, 0.3);
        }
        .pair-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1rem; }
        .pair-name { font-size: 1.5rem; font-weight: 600; }
        .pair-price { font-size: 1.25rem; font-weight: 400; }
        
        .pair-info { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem; }
        
        .action-area { margin-top: auto; }
        .manual-trade-btns { display: flex; gap: 1rem; }
        .btn {
            flex-grow: 1; padding: 0.75rem; border-radius: 8px; border: none; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease;
        }
        .btn:hover { transform: scale(1.05); }
        .btn-long { background-color: var(--green); color: white; }
        .btn-short { background-color: var(--red); color: white; }
        .btn-close { background-color: var(--yellow); color: var(--bg-color); }
        
        .position-info { border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; }
        .position-info.long { border-left: 5px solid var(--green); }
        .position-info.short { border-left: 5px solid var(--red); }
        .position-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
        .position-pnl { font-size: 1.75rem; font-weight: 700; text-align: center; margin-bottom: 1rem; }

        .history-section { margin-top: 3rem; }
        .history-table { width: 100%; border-collapse: collapse; }
        .history-table th, .history-table td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }
        .history-table th { font-weight: 600; color: var(--text-muted); font-size: 0.9rem; }
        .history-table tr:last-child td { border-bottom: none; }
        .history-table small { color: var(--text-muted); }
        
        .text-green { color: var(--green); }
        .text-red { color: var(--red); }
        .text-yellow { color: var(--yellow); }

        @keyframes pulse-glow {
            0% { box-shadow: 0 0 5px rgba(0, 191, 255, 0.3); }
            50% { box-shadow: 0 0 15px rgba(0, 191, 255, 0.7); }
            100% { box-shadow: 0 0 5px rgba(0, 191, 255, 0.3); }
        }
        .pulse-animation { animation: pulse-glow 2s infinite ease-in-out; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Vulcan AI</h1>
            <div id="ai-status-container"></div>
        </header>

        <section id="pnl-stats" class="pnl-stats"></section>
        
        <h2 style="margin-bottom: 1.5rem;">Watchlist</h2>
        <section id="watchlist" class="watchlist"></section>

        <section class="history-section">
            <h2>Trade History</h2>
            <div style="overflow-x: auto;">
                <table class="history-table">
                    <thead>
                        <tr><th>Pair</th><th>Type</th><th>Status</th><th>Entry</th><th>Exit</th><th>P/L (Net)</th><th>Reason</th></tr>
                    </thead>
                    <tbody id="trade-history-body"></tbody>
                </table>
            </div>
        </section>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const API_ENDPOINT = '/api/data';
            const REFRESH_INTERVAL_MS = 3000;

            const pnlStatsContainer = document.getElementById('pnl-stats');
            const aiStatusContainer = document.getElementById('ai-status-container');
            const watchlistContainer = document.getElementById('watchlist');
            const historyBody = document.getElementById('trade-history-body');

            function formatPercent(value) {
                if (typeof value !== 'number') return 'N/A';
                return value.toFixed(2) + '%';
            }
            function formatPrice(value) {
                if (typeof value !== 'number') return 'N/A';
                return value.toFixed(4);
            }
            function getColorClass(value) {
                if (value > 0) return 'text-green';
                if (value < 0) return 'text-red';
                return '';
            }

            async function updateDashboard() {
                try {
                    const response = await fetch(API_ENDPOINT);
                    if (!response.ok) {
                        console.error('Failed to fetch data from API');
                        return;
                    }
                    const data = await response.json();

                    // 1. Update PNL Stats
                    pnlStatsContainer.innerHTML = `
                        <div class="stat-item">
                            <div class="label">Today's P/L</div>
                            <div class="value ${getColorClass(data.pnl_today)}">${formatPercent(data.pnl_today)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="label">This Week's P/L</div>
                            <div class="value ${getColorClass(data.pnl_this_week)}">${formatPercent(data.pnl_this_week)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="label">Last Week's P/L</div>
                            <div class="value ${getColorClass(data.pnl_last_week)}">${formatPercent(data.pnl_last_week)}</div>
                        </div>
                    `;

                    // 2. Update AI Status Button
                    aiStatusContainer.innerHTML = `
                        <form action="/toggle-ai" method="POST">
                            <button type="submit" class="ai-toggle-btn ${data.is_ai_running ? 'running' : 'stopped'}">
                                AI ${data.is_ai_running ? 'RUNNING' : 'STOPPED'}
                            </button>
                        </form>
                    `;
                    
                    // 3. Update Watchlist Cards
                    watchlistContainer.innerHTML = ''; // Clear and rebuild
                    for (const pair in data.market_data) {
                        const pairData = data.market_data[pair];
                        const card = document.createElement('div');
                        card.className = 'pair-card';
                        
                        let actionHTML = '';
                        if (pairData.open_position) {
                            card.classList.add('position-open', 'pulse-animation');
                            const pos = pairData.open_position;
                            const pnlColorClass = getColorClass(pairData.pnl);
                            actionHTML = `
                                <div class="position-info ${pos.type.toLowerCase()}">
                                    <div class="position-header">${pos.type} POSITION OPEN</div>
                                    <div class="position-pnl ${pnlColorClass}">${formatPercent(pairData.pnl)}</div>
                                    <div style="font-size: 0.9rem; color: var(--text-muted); text-align: center; margin-bottom: 1rem;">Entry @ ${formatPrice(pos.entryPrice)}</div>
                                    <form action="/trade/close" method="POST" class="d-grid">
                                        <input type="hidden" name="trade_id" value="${pos.id}">
                                        <button type="submit" class="btn btn-close">Close Position (Market)</button>
                                    </form>
                                </div>
                            `;
                        } else {
                            actionHTML = `
                                <div class="manual-trade-btns">
                                    <form action="/trade/manual" method="POST" style="flex-grow: 1;">
                                        <input type="hidden" name="pair" value="${pair}"><input type="hidden" name="type" value="LONG">
                                        <button type="submit" class="btn btn-long w-100">Long</button>
                                    </form>
                                    <form action="/trade/manual" method="POST" style="flex-grow: 1;">
                                        <input type="hidden" name="pair" value="${pair}"><input type="hidden" name="type" value="SHORT">
                                        <button type="submit" class="btn btn-short w-100">Short</button>
                                    </form>
                                </div>
                            `;
                        }
                        
                        const fundingColorClass = pairData.funding > 0.01 ? 'text-red' : (pairData.funding < -0.01 ? 'text-green' : '');

                        card.innerHTML = `
                            <div class="pair-header">
                                <span class="pair-name">${pair}</span>
                                <span class="pair-price">${formatPrice(pairData.price)}</span>
                            </div>
                            <div class="pair-info">
                                <span>Timeframe: <strong>${pairData.timeframe}</strong></span>
                                <span>Funding: <strong class="${fundingColorClass}">${formatPercent(pairData.funding)}</strong></span>
                            </div>
                            <div class="action-area">${actionHTML}</div>
                        `;
                        watchlistContainer.appendChild(card);
                    }

                    // 4. Update Trade History
                    let historyHTML = '';
                    for (const trade of data.trades) {
                        const pnlNet = trade.pl_percent !== null ? trade.pl_percent - 0.1 : null;
                        const pnlClass = trade.status === 'CLOSED' ? getColorClass(pnlNet) : '';
                        const typeClass = trade.type === 'LONG' ? 'text-green' : 'text-red';
                        historyHTML += `
                            <tr>
                                <td><strong>${trade.instrumentId}</strong></td>
                                <td class="${typeClass}">${trade.type}</td>
                                <td>${trade.status}</td>
                                <td>${formatPrice(trade.entryPrice)}</td>
                                <td>${trade.exitPrice !== null ? formatPrice(trade.exitPrice) : 'N/A'}</td>
                                <td class="${pnlClass}">${pnlNet !== null ? formatPercent(pnlNet) : '-'}</td>
                                <td><small>${trade.entryReason.split('\\n')[0]}</small></td>
                            </tr>
                        `;
                    }
                    historyBody.innerHTML = historyHTML;

                } catch (error) {
                    console.error("Error updating dashboard:", error);
                }
            }

            // Initial load and set interval for recurring updates
            updateDashboard();
            setInterval(updateDashboard, REFRESH_INTERVAL_MS);
        });
    </script>
</body>
</html>
"""

# --- RUTE FLASK ---
@app.route('/')
def dashboard():
    """Menyajikan kerangka HTML statis. JavaScript akan menangani sisanya."""
    return render_template_string(HTML_SKELETON)

@app.route('/api/data')
def get_api_data():
    """Endpoint API yang mengembalikan semua data state aplikasi dalam format JSON."""
    with state_lock:
        # Buat salinan data untuk memastikan thread-safety saat membaca
        trades_copy = list(autopilot_trades)
        market_state_copy = dict(market_state)

    market_data_view = {}
    fee_pct = current_settings.get('fee_pct', 0.1)

    for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
        pair_state = market_state_copy.get(pair_id, {})
        current_price = pair_state.get("candle_data", [{}])[-1].get('close', 0.0)
        
        market_data_view[pair_id] = {
            "price": current_price,
            "funding": pair_state.get("funding_rate", 0.0),
            "timeframe": timeframe,
            "open_position": None,
            "pnl": 0.0
        }
        
        open_pos = next((t for t in trades_copy if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        if open_pos:
            market_data_view[pair_id]["open_position"] = open_pos
            if current_price > 0:
                 market_data_view[pair_id]["pnl"] = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - fee_pct

    # Siapkan data untuk respons JSON
    json_response = {
        "is_ai_running": is_autopilot_running,
        "pnl_today": calculate_todays_pnl(trades_copy),
        "pnl_this_week": calculate_this_weeks_pnl(trades_copy),
        "pnl_last_week": calculate_last_weeks_pnl(trades_copy),
        "market_data": market_data_view,
        "trades": trades_copy[:80] # Kirim 80 trade terbaru
    }
    
    return jsonify(json_response)


@app.route('/toggle-ai', methods=['POST'])
def toggle_ai():
    global is_autopilot_running
    with state_lock:
        is_autopilot_running = not is_autopilot_running
        status = "diaktifkan" if is_autopilot_running else "dimatikan"
        print_colored(f"Autopilot {status} melalui Web UI.", Fore.YELLOW)
    return redirect(url_for('dashboard'))

@app.route('/trade/manual', methods=['POST'])
def trade_manual():
    pair = request.form.get('pair')
    trade_type = request.form.get('type')
    if not pair or not trade_type: return redirect(url_for('dashboard'))
    
    current_price = market_state.get(pair, {}).get("candle_data", [{}])[-1].get('close')
    if not current_price:
        print_colored(f"Gagal membuka trade manual untuk {pair}: Harga tidak tersedia.", Fore.RED)
        return redirect(url_for('dashboard'))
        
    with state_lock:
        if any(t for t in autopilot_trades if t['instrumentId'] == pair and t['status'] == 'OPEN'):
            print_colored(f"Gagal membuka trade manual {pair}: Posisi sudah ada.", Fore.YELLOW)
            return redirect(url_for('dashboard'))
        new_trade = {
            "id": int(time.time()), "instrumentId": pair, "type": trade_type,
            "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": current_price,
            "entryReason": "Manual Entry from Web UI", "status": 'OPEN', "entry_snapshot": {},
            "run_up_percent": 0.0, "max_drawdown_percent": 0.0,
            "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0,
            "exitPrice": None, "pl_percent": None
        }
        autopilot_trades.insert(0, new_trade) # Tambahkan ke paling atas
        print_colored(f"Trade Manual {trade_type} {pair} @ {current_price} dibuka.", Fore.BLUE)
    save_trades()
    return redirect(url_for('dashboard'))

@app.route('/trade/close', methods=['POST'])
def trade_close():
    trade_id_str = request.form.get('trade_id')
    if not trade_id_str: return redirect(url_for('dashboard'))
    trade_id = int(trade_id_str)
    
    with state_lock:
        trade_to_close = next((t for t in autopilot_trades if t['id'] == trade_id and t['status'] == 'OPEN'), None)
        if not trade_to_close:
            print_colored(f"Gagal menutup trade ID {trade_id}: Tidak ditemukan atau sudah ditutup.", Fore.RED)
            return redirect(url_for('dashboard'))
        pair = trade_to_close['instrumentId']
        current_price = market_state.get(pair, {}).get("candle_data", [{}])[-1].get('close')
        if not current_price:
            print_colored(f"Gagal menutup trade {pair}: Harga tidak tersedia.", Fore.RED)
            return redirect(url_for('dashboard'))
        pnl_gross = calculate_pnl(trade_to_close['entryPrice'], current_price, trade_to_close.get('type'))
        trade_to_close.update({
            'status': 'CLOSED', 'exitPrice': current_price,
            'exitTimestamp': datetime.utcnow().isoformat() + 'Z', 'pl_percent': pnl_gross
        })
        print_colored(f"Posisi {pair} ditutup manual via Web UI @ {current_price}. P/L: {pnl_gross:.2f}%", Fore.BLUE)
    save_trades()
    return redirect(url_for('dashboard'))


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    load_settings()
    load_trades()
    display_welcome_message()
    
    # Mulai thread latar belakang
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    autopilot_thread.start()
    data_thread.start()
    
    # Jalankan server Flask
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    # Cleanup saat server dihentikan (misalnya dengan Ctrl+C)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)
