import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math
from flask import Flask, render_template_string, jsonify, request

# --- Dummy Colorama for environments where it's not installed ---
class DummyColor:
    def __init__(self): self.BLACK = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.MAGENTA = self.CYAN = self.WHITE = self.RESET = ''
class DummyStyle:
    def __init__(self): self.DIM = self.NORMAL = self.BRIGHT = self.RESET_ALL = ''

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Peringatan: Pustaka 'colorama' tidak ditemukan. Output tidak akan berwarna.")
    Fore = DummyColor(); Style = DummyStyle()

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"

# --- STATE APLIKASI ---
current_settings = {}
trades = []
market_state = {}
is_ai_thinking = False
is_autopilot_running = True
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
state_lock = threading.Lock()

# --- INISIALISASI FLASK ---
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'): print(bright + color + text + Style.RESET_ALL, end=end)
def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        safe_title = title.replace('"', "'"); safe_content = content.replace('"', "'")
        os.system(f'termux-notification --title "{safe_title}" --content "{safe_content}"')
    except Exception as e: print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)
def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("     Strategic AI Analyst (Full Vulcan's Logic)   ", Fore.CYAN, Style.BRIGHT)
    print_colored("       -- WIN/LOSS LEARNING AI EDITION --         ", Fore.YELLOW, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Bot berjalan. Akses dashboard di:", Fore.GREEN, Style.BRIGHT)
    print_colored("http://127.0.0.1:5000 atau http://[IP_LOKAL_ANDA]:5000", Fore.GREEN, Style.BRIGHT)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = { 
        "stop_loss_pct": 0.20, "fee_pct": 0.1, "analysis_interval_sec": 10, 
        "use_trailing_tp": True, "trailing_tp_activation_pct": 0.30, 
        "trailing_tp_gap_pct": 0.05, "caution_level": 0.5, 
        "max_allowed_funding_rate_pct": 0.075, "watched_pairs": {"BTC-USDT": "1H", "ETH-USDT": "1H"},
        "max_trades_in_history": 80, "refresh_interval_seconds": 1, "chart_candle_limit": 80,
        "similarity_threshold_win": 4, "similarity_threshold_loss": 3
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: loaded_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in loaded_settings: loaded_settings[key] = value
            current_settings = loaded_settings
        except (json.JSONDecodeError, IOError): current_settings = default_settings
    else: current_settings = default_settings
    save_settings()
def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)
    except IOError as e: print_colored(f"Error saving settings: {e}", Fore.RED)
def load_trades():
    global trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: trades = json.load(f)
        except (json.JSONDecodeError, IOError): trades = []
    else: trades = []
def save_trades():
    global trades
    with state_lock:
        trades.sort(key=lambda x: x['entryTimestamp'], reverse=True)
        max_trades = current_settings.get("max_trades_in_history", 80)
        if len(trades) > max_trades: trades = trades[:max_trades]
        try:
            with open(TRADES_FILE, 'w') as f: json.dump(trades, f, indent=4)
        except IOError as e: print_colored(f"Error saving trades: {e}", Fore.RED)

# --- FUNGSI API, KALKULASI, AI, THREAD WORKERS ---
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
    except (requests.exceptions.RequestException, Exception): return None
def calculate_pnl(entry_price, current_price, trade_type):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0
def calculate_todays_pnl(all_trades):
    today_utc = datetime.utcnow().date(); total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() == today_utc: total_pnl += (trade.get('pl_percent', 0.0) - (2 * fee_pct))
            except (ValueError, TypeError): continue
    return total_pnl
def calculate_this_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date(); start_of_week_utc = today_utc - timedelta(days=today_utc.weekday())
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date()
                if start_of_week_utc <= exit_date <= today_utc: total_pnl += (trade.get('pl_percent', 0.0) - (2 * fee_pct))
            except (ValueError, TypeError): continue
    return total_pnl
def calculate_last_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date(); start_of_current_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_last_week_utc = start_of_current_week_utc - timedelta(days=1); start_of_last_week_utc = end_of_last_week_utc - timedelta(days=6)
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date()
                if start_of_last_week_utc <= exit_date <= end_of_last_week_utc: total_pnl += (trade.get('pl_percent', 0.0) - (2 * fee_pct))
            except (ValueError, TypeError): continue
    return total_pnl
class LocalAI:
    def __init__(self, settings, past_trades_for_pair): self.settings = settings; self.past_trades = past_trades_for_pair
    def calculate_ema(self, data, period):
        if len(data) < period: return []
        closes = [d['close'] for d in data]; ema_values = [sum(closes[:period]) / period]
        multiplier = 2 / (period + 1)
        for i in range(period, len(closes)): ema_values.append((closes[i] - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values
    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open']); full_range = candle['high'] - candle['low']
        return body / full_range if full_range > 0 else 1.0
    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100 + 3: return None
        ema9 = self.calculate_ema(candle_data, 9); ema50 = self.calculate_ema(candle_data, 50); ema100 = self.calculate_ema(candle_data, 100)
        if len(ema9) < 2 or not ema50 or not ema100: return None
        analysis = { "ema9_current": ema9[-1], "ema9_prev": ema9[-2], "ema50": ema50[-1], "ema100": ema100[-1], "current_candle_close": candle_data[-1]['close'], "prev_candle_close": candle_data[-2]['close'], "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING" }
        pre_entry_candles = candle_data[-4:-1]
        analysis["pre_entry_candle_solidity"] = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        analysis["pre_entry_candle_direction"] = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        return analysis
    def compare_setups(self, current_analysis, past_snapshot):
        if not past_snapshot or not past_snapshot.get('bias') or past_snapshot.get('bias') != current_analysis['bias']: return 0
        similarity_score = 1
        current_pos_vs_ema50 = 'above' if current_analysis['current_candle_close'] > current_analysis['ema50'] else 'below'
        past_pos_vs_ema50 = 'above' if past_snapshot.get('current_candle_close', 0) > past_snapshot.get('ema50', 0) else 'below'
        if current_pos_vs_ema50 == past_pos_vs_ema50: similarity_score += 1
        if 'pre_entry_candle_direction' in current_analysis and current_analysis['pre_entry_candle_direction'] == past_snapshot.get('pre_entry_candle_direction', []): similarity_score += 1
        if 'pre_entry_candle_solidity' in current_analysis and 'pre_entry_candle_solidity' in past_snapshot:
            avg_solidity_current = sum(current_analysis['pre_entry_candle_solidity']) / 3
            past_solidity_list = past_snapshot.get('pre_entry_candle_solidity', [0,0,0]); avg_solidity_past = sum(past_solidity_list) / 3 if past_solidity_list else 0
            if abs(avg_solidity_current - avg_solidity_past) < 0.2: similarity_score += 1
        return similarity_score
    def check_for_winning_setup(self, current_analysis):
        winning_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - (2 * self.settings.get('fee_pct', 0.1))) > 0]
        if not winning_trades: return (False, None, None)
        SIMILARITY_THRESHOLD = self.settings.get("similarity_threshold_win", 4)
        for win in winning_trades:
            score = self.compare_setups(current_analysis, win.get("entry_snapshot"))
            if score >= SIMILARITY_THRESHOLD:
                reason = f"High Confidence: Setup mirip win (ID: {win.get('id', 'N/A')})"
                return (True, reason, win.get('type'))
        return (False, None, None)
    def check_for_repeated_mistake(self, current_analysis):
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - (2 * self.settings.get('fee_pct', 0.1))) < 0]
        if not losing_trades: return (False, None)
        SIMILARITY_THRESHOLD = self.settings.get("similarity_threshold_loss", 3)
        for loss in losing_trades:
            score = self.compare_setups(current_analysis, loss.get("entry_snapshot"))
            if score >= SIMILARITY_THRESHOLD:
                reason = (f"Peringatan: Mirip loss ID {loss.get('id', 'N/A')}. Bias: {current_analysis['bias']}")
                return (True, reason)
        return (False, None)
    def get_decision(self, candle_data, open_position, funding_rate=0.0):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data teknikal tidak cukup."}
        if open_position: return {"action": "HOLD", "reason": "Posisi terbuka."}
        is_win_setup, win_reason, trade_type_from_win = self.check_for_winning_setup(analysis)
        if is_win_setup:
            return {"action": "BUY" if trade_type_from_win == 'LONG' else "SELL", "reason": win_reason, "snapshot": analysis}
        max_funding_rate = self.settings.get("max_allowed_funding_rate_pct", 0.075)
        potential_trade_type = None
        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']: potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']: potential_trade_type = 'SHORT'
        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal LONG batal. Funding rate tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal SHORT batal. Funding rate negatif: {funding_rate:.4f}%"}
            caution_level = self.settings.get("caution_level", 0.5); avg_solidity = sum(analysis.get('pre_entry_candle_solidity', [0])) / 3
            if avg_solidity < caution_level: return {"action": "HOLD", "reason": f"Sinyal batal. Pasar ragu-ragu (Solidity: {avg_solidity:.2f} < Caution: {caution_level:.2f})"}
            is_repeated_mistake, mistake_reason = self.check_for_repeated_mistake(analysis)
            if is_repeated_mistake: return {"action": "HOLD", "reason": mistake_reason}
            ai_reason = (f"AI: {potential_trade_type} berdasarkan konfirmasi tren {analysis['bias']}.")
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": ai_reason, "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias: {analysis['bias']}."}
def close_trade_sync(trade, exit_price, reason):
    with state_lock:
        pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type'))
        exit_dt = datetime.utcnow()
        trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })
    save_trades()
    pnl_net = pnl_gross - (2 * current_settings.get('fee_pct', 0.1))
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | {reason}"
    send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.MAGENTA)
async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking: return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3: return
    is_ai_thinking = True
    try:
        with state_lock: open_pos = next((t for t in trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades_history = [t for t in trades if t['instrumentId'] == instrument_id]
        ai = LocalAI(current_settings, relevant_trades_history)
        funding_rate = pair_state.get("funding_rate", 0.0)
        decision = ai.get_decision(pair_state["candle_data"], open_pos, funding_rate)
        if decision.get('action') in ["BUY", "SELL"] and not open_pos:
            snapshot = decision.get("snapshot", {}); snapshot["funding_rate"] = funding_rate
            new_trade = { "id": int(time.time()), "instrumentId": instrument_id, "type": "LONG" if decision['action'] == "BUY" else "SHORT", "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": pair_state["candle_data"][-1]['close'], "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot, "exitPrice": None, "pl_percent": None }
            with state_lock: trades.insert(0, new_trade)
            save_trades()
            notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {instrument_id}"; notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | {decision.get('reason')}"
            send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.GREEN)
    finally: is_ai_thinking = False
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            for pair_id in list(current_settings.get("watched_pairs", {})): asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data):
    if not trade_obj or not trade_obj.get('type') or trade_obj.get('status') != 'OPEN': return
    trade_type = trade_obj.get('type'); entry_price = trade_obj['entryPrice']
    candle_low = current_candle_data['low']; candle_high = current_candle_data['high']
    sl_pct = current_settings.get('stop_loss_pct', 0)
    if sl_pct > 0:
        if trade_type == 'LONG':
            pnl_at_low = calculate_pnl(entry_price, candle_low, 'LONG')
            if pnl_at_low <= -sl_pct:
                sl_price = entry_price * (1 - sl_pct / 100)
                close_trade_sync(trade_obj, sl_price, f"Stop Loss @ {-sl_pct:.2f}%"); return
        elif trade_type == 'SHORT':
            pnl_at_high = calculate_pnl(entry_price, candle_high, 'SHORT')
            if pnl_at_high <= -sl_pct:
                sl_price = entry_price * (1 + sl_pct / 100)
                close_trade_sync(trade_obj, sl_price, f"Stop Loss @ {-sl_pct:.2f}%"); return
    if current_settings.get('use_trailing_tp', True):
        activation_pct = current_settings.get("trailing_tp_activation_pct", 0); gap_pct = current_settings.get("trailing_tp_gap_pct", 0)
        if activation_pct <= 0 or gap_pct <= 0: return
        pnl_at_best = calculate_pnl(entry_price, candle_high if trade_type == 'LONG' else candle_low, trade_type)
        ts_price = None
        with state_lock:
            if pnl_at_best >= activation_pct:
                current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
                if current_cp == 0.0: current_cp = activation_pct
                steps_passed = math.floor((pnl_at_best - current_cp) / gap_pct)
                if steps_passed >= 0:
                    new_cp = current_cp + (steps_passed * gap_pct); trade_obj['current_tp_checkpoint_level'] = new_cp
                    new_ts_level = new_cp - gap_pct
                    if trade_type == 'LONG': trade_obj['trailing_stop_price'] = entry_price * (1 + new_ts_level / 100)
                    else: trade_obj['trailing_stop_price'] = entry_price * (1 - new_ts_level / 100)
            ts_price = trade_obj.get('trailing_stop_price')
        if ts_price is not None:
            if (trade_type == 'LONG' and candle_low <= ts_price) or (trade_type == 'SHORT' and candle_high >= ts_price):
                close_trade_sync(trade_obj, ts_price, "Trailing TP")
    else:
        static_tp_pct = current_settings.get("trailing_tp_activation_pct", 0)
        if static_tp_pct <= 0: return
        if trade_type == 'LONG':
            tp_price = entry_price * (1 + static_tp_pct / 100)
            if candle_high >= tp_price: close_trade_sync(trade_obj, tp_price, f"Static TP @ {static_tp_pct:.2f}%")
        elif trade_type == 'SHORT':
            tp_price = entry_price * (1 - static_tp_pct / 100)
            if candle_low <= tp_price: close_trade_sync(trade_obj, tp_price, f"Static TP @ {static_tp_pct:.2f}%")

def data_refresh_worker():
    while not stop_event.is_set():
        start_time = time.time()
        watched_pairs_copy = list(current_settings.get("watched_pairs", {}).items())
        for pair_id, timeframe in watched_pairs_copy:
            if pair_id not in market_state: market_state[pair_id] = {}
            candle_data = fetch_recent_candles(pair_id, timeframe)
            funding_rate = fetch_funding_rate(pair_id)
            market_state[pair_id]['funding_rate'] = funding_rate if funding_rate is not None else market_state[pair_id].get('funding_rate', 0.0)
            if candle_data:
                market_state[pair_id]["candle_data"] = candle_data
                with state_lock: open_pos = next((t for t in trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos: asyncio.run(check_realtime_position_management(open_pos, candle_data[-1]))
            time.sleep(0.1) 
        elapsed_time = time.time() - start_time
        sleep_duration = max(0, current_settings.get("refresh_interval_seconds", 1) - elapsed_time)
        stop_event.wait(sleep_duration)

# --- TEMPLATE HTML DENGAN UI PROFESIONAL, ANTI-FLICKER, DAN FUNGSI LENGKAP ---
HTML_SKELETON_TRADINGVIEW = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vulcan AI Dashboard</title>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-color: #0D0D0D; --card-color: #1A1A1A; --border-color: #2A2A2A; --text-color: #EAEAEA; --text-muted: #737373; --green: #34D399; --red: #F87171; --yellow: #FBBF24; --accent-primary: #3B82F6; --shadow-color: rgba(0, 0, 0, 0.4); }
        
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes value-update-green { 0% { color: var(--green); transform: scale(1.1); } 100% { color: var(--green); transform: scale(1); } }
        @keyframes value-update-red { 0% { color: var(--red); transform: scale(1.1); } 100% { color: var(--red); transform: scale(1); } }
        @keyframes value-update-neutral { 0% { color: var(--text-color); transform: scale(1.1); } 100% { color: var(--text-color); transform: scale(1); } }
        
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; font-size: 16px; }
        body { background-color: var(--bg-color); color: var(--text-color); font-family: 'Inter', sans-serif; margin: 0; padding: 1.5rem; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
        .container { max-width: 1400px; margin: 0 auto; }
        
        h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
        h1 { margin: 0; font-size: 2rem; }
        h2 { margin-top: 3.5rem; margin-bottom: 1.5rem; font-size: 1.25rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;}
        h3 { margin-top: 2rem; margin-bottom: 1rem; font-size: 1.1rem; }
        
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.5rem; animation: fadeInUp 0.5s ease-out; }
        .header-actions { display: flex; gap: 1rem; }
        .action-btn { background-color: var(--card-color); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s ease-in-out; }
        .action-btn:hover { background-color: var(--border-color); box-shadow: 0 4px 15px var(--shadow-color); transform: translateY(-2px); }
        .action-btn:active { transform: translateY(0px) scale(0.98); box-shadow: none; }
        .action-btn.ai-status.running { color: var(--green); } .action-btn.ai-status.stopped { color: var(--red); }
        
        .pnl-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; }
        .stat-item { background: var(--card-color); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 12px; transition: all 0.3s ease-in-out; animation: fadeInUp 0.5s ease-out forwards; opacity: 0; }
        .stat-item:hover { transform: translateY(-4px); border-color: var(--accent-primary); }
        .stat-item .label { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.75rem; font-weight: 500; }
        .stat-item .value { font-size: 2.25rem; font-weight: 700; transition: color 0.3s ease; }
        
        .tradingview-widget-container { height: 500px; border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); }
        
        .watchlist { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 1.5rem; }
        .pair-card { opacity: 0; animation: fadeInUp 0.6s ease-out forwards; background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; display: flex; flex-direction: column; transition: all 0.3s ease-in-out; cursor: pointer; }
        .pair-card:hover { transform: translateY(-5px); box-shadow: 0 8px 30px var(--shadow-color); }
        .pair-card.active-chart { border-color: var(--accent-primary); box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); }
        .pair-card.position-open { border-left: 4px solid var(--accent-primary); padding-left: calc(1.5rem - 3px); }
        
        .pair-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; }
        .pair-name-section .pair-name { font-size: 1.75rem; font-weight: 700; line-height: 1; }
        .pair-name-section .pair-tf { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
        
        .timer-container { position: relative; width: 52px; height: 52px; }
        .timer-svg { transform: rotate(-90deg); width: 100%; height: 100%; }
        .timer-circle { fill: none; stroke-linecap: round; }
        .timer-circle-bg { stroke: var(--border-color); stroke-width: 3; }
        .timer-circle-progress { stroke: var(--accent-primary); stroke-width: 4; transition: stroke-dashoffset 0.25s linear; }
        .timer-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1rem; color: var(--text-muted); font-weight: 600; }

        .pair-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1rem; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color); }
        .pair-info-grid strong { color: var(--text-color); font-weight: 600; }

        .btn { flex-grow: 1; padding: 0.85rem; border-radius: 8px; border: none; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s ease-in-out; }
        .btn:hover { filter: brightness(1.15); } .btn:active { transform: scale(0.97); }
        .btn-long { background: var(--green); color: #0D0D0D; } .btn-short { background: var(--red); color: #FFF; } .btn-close { background: var(--yellow); color: #0D0D0D; }
        
        .position-info { border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; text-align: center; margin-top: auto; background: rgba(0,0,0,0.2); }
        .position-header { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; color: var(--text-muted); }
        .position-pnl { font-size: 2.25rem; font-weight: 700; margin-bottom: 0.5rem; }
        .position-entry { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1rem; }

        .history-list { list-style: none; padding: 0; }
        .history-item { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; transition: background-color 0.2s ease; opacity: 0; animation: fadeInUp 0.5s ease-out forwards; }
        .history-main { display: flex; align-items: center; gap: 1rem; }
        .history-type { font-weight: 600; font-size: 1.1rem; }
        .history-pair { color: var(--text-muted); }
        .history-pnl { font-size: 1.25rem; font-weight: 600; text-align: right; }
        .history-details { color: var(--text-muted); font-size: 0.85rem; width: 100%; text-align: left; }
        
        .settings-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(8px); display: none; justify-content: center; align-items: center; z-index: 1000; opacity: 0; transition: opacity 0.3s ease; pointer-events: none; }
        .settings-modal.visible { opacity: 1; pointer-events: all; }
        .settings-modal.visible .settings-content { transform: scale(1) translateY(0); opacity: 1; }
        .settings-content { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 2rem; width: 90%; max-width: 600px; max-height: 90vh; overflow-y: auto; transform: scale(0.95) translateY(20px); opacity: 0; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { color: var(--text-muted); margin-bottom: 0.5rem; font-size: 0.9rem; }
        .form-group input { background-color: var(--bg-color); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.75rem; border-radius: 8px; font-size: 1rem; }
        .form-group.checkbox-group { flex-direction: row; align-items: center; gap: 0.5rem; }
        .form-group.checkbox-group label { margin: 0; }
        .watchlist-manage ul { list-style: none; padding: 0; }
        .watchlist-manage li { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color); }
        .btn-remove { background: none; border: none; color: var(--red); cursor: pointer; font-size: 1.25rem; }

        .text-green { color: var(--green) !important; } .text-red { color: var(--red) !important; } .text-yellow { color: var(--yellow); }
        .update-g { animation: value-update-green 0.5s ease-out; } .update-r { animation: value-update-red 0.5s ease-out; } .update-n { animation: value-update-neutral 0.5s ease-out; }

        @media (max-width: 768px) { .pnl-stats, .watchlist, .form-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header class="header"><h1>Vulcan AI</h1><div class="header-actions"><button id="ai-status-btn" class="action-btn ai-status"></button><button id="settings-btn" class="action-btn">Settings</button></div></header>
        
        <section id="pnl-stats" class="pnl-stats">
            <div class="stat-item" style="animation-delay: 0.1s;"><div class="label">Today's P/L</div><div class="value" id="pnl-today">--%</div></div>
            <div class="stat-item" style="animation-delay: 0.2s;"><div class="label">This Week P/L</div><div class="value" id="pnl-this-week">--%</div></div>
            <div class="stat-item" style="animation-delay: 0.3s;"><div class="label">Last Week P/L</div><div class="value" id="pnl-last-week">--%</div></div>
        </section>
        
        <h2>Charts</h2>
        <div id="tradingview_chart_bybit" class="tradingview-widget-container"></div>
        <h2 style="margin-top: 1rem;"></h2>
        <div id="tradingview_chart_binance" class="tradingview-widget-container"></div>
        
        <h2>Watchlist</h2>
        <section id="watchlist" class="watchlist"></section>
        
        <h2>Recent History</h2>
        <ul id="history-list" class="history-list"></ul>
    </div>
    
    <div id="settings-modal" class="settings-modal">
        <div class="settings-content">
            <div style="display:flex; justify-content:space-between; align-items:center;"><h2>Settings</h2><button id="close-settings-btn" style="background:none; border:none; color:var(--text-color); font-size: 2rem; cursor:pointer;">×</button></div>
            <form id="settings-form">
                <h3>Trading Parameters</h3>
                <div class="form-grid"><div class="form-group"><label>Fee per Transaction (%)</label><input type="number" step="any" name="fee_pct" id="s-fee_pct"></div><div class="form-group"><label>Stop Loss (%)</label><input type="number" step="any" name="stop_loss_pct" id="s-stop_loss_pct"></div><div class="form-group checkbox-group"><input type="checkbox" name="use_trailing_tp" id="s-use_trailing_tp"><label for="s-use_trailing_tp">Enable Trailing TP</label></div><div class="form-group"><label>TP Activation / Static TP (%)</label><input type="number" step="any" name="trailing_tp_activation_pct" id="s-trailing_tp_activation_pct"></div><div class="form-group"><label>TP Gap (for Trailing)</label><input type="number" step="any" name="trailing_tp_gap_pct" id="s-trailing_tp_gap_pct"></div><div class="form-group"><label>Max Funding Rate (%)</label><input type="number" step="any" name="max_allowed_funding_rate_pct" id="s-max_allowed_funding_rate_pct"></div></div>
                <h3>AI Learning Parameters</h3>
                <div class="form-grid"><div class="form-group"><label>Caution Level (0-1)</label><input type="number" step="any" name="caution_level" id="s-caution_level"></div><div class="form-group"><label>Win Similarity Threshold</label><input type="number" step="1" name="similarity_threshold_win" id="s-similarity_threshold_win"></div><div class="form-group"><label>Loss Similarity Threshold</label><input type="number" step="1" name="similarity_threshold_loss" id="s-similarity_threshold_loss"></div></div>
                <h3>System Parameters</h3>
                <div class="form-grid"><div class="form-group"><label>AI Delay (s)</label><input type="number" step="1" name="analysis_interval_sec" id="s-analysis_interval_sec"></div><div class="form-group"><label>Max Trade History</label><input type="number" step="10" name="max_trades_in_history" id="s-max_trades_in_history"></div><div class="form-group"><label>Data Refresh (s)</label><input type="number" step="any" name="refresh_interval_seconds" id="s-refresh_interval_seconds"></div></div>
                <h3>Watchlist</h3>
                <div class="watchlist-manage"><ul id="watchlist-list"></ul><div class="form-group" style="margin-top:1rem;"><label>Add New Pair (e.g., BTC-USDT)</label><div style="display:flex; gap:1rem;"><input type="text" id="new-pair-input" placeholder="Pair" style="flex-grow:1;"><input type="text" id="new-tf-input" value="1H" placeholder="Timeframe" style="width:100px;"><button type="button" id="add-pair-btn" class="action-btn" style="background-color: var(--accent-primary); border:none;">Add</button></div></div></div>
                <button type="submit" class="action-btn" style="width:100%; margin-top: 2rem; padding: 0.75rem; background-color:var(--accent-primary); border:none;">Save Settings</button>
            </form>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', () => {
        const API_ENDPOINT = '/api/data';
        const REFRESH_INTERVAL_MS = {{ current_settings.refresh_interval_seconds * 1000 }};
        
        const formatPercent = v => typeof v === 'number' ? v.toFixed(2) + '%' : 'N/A';
        const formatPrice = v => typeof v === 'number' ? (v < 0.1 ? v.toPrecision(4) : v.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})) : 'N/A';
        const getPnlColorClass = v => v > 0 ? 'text-green' : v < 0 ? 'text-red' : '';
        const postRequest = async (url, data) => { try { const res = await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: new URLSearchParams(data) }); return res.ok; } catch (e) { console.error(`POST to ${url} failed:`, e); return false; }};
        
        let currentChartPair = null; 
        let candleStartTimes = {};
        const TIMEFRAME_SECONDS = { '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1H': 3600, '2H': 7200, '4H': 14400, '1D': 86400, '1W': 604800 };
        const SVG_CIRCLE_RADIUS = 22; // Adjusted for better look
        const SVG_CIRCUMFERENCE = 2 * Math.PI * SVG_CIRCLE_RADIUS;

        // --- UI Helper Functions ---
        const updateText = (element, newText, animationClass = 'update-n') => {
            if (element && element.textContent !== newText) {
                element.textContent = newText;
                element.classList.remove('update-g', 'update-r', 'update-n');
                void element.offsetWidth; // Trigger reflow to restart animation
                element.classList.add(animationClass);
            }
        };

        const createChartWidgets = (pair, timeframe) => {
            if (!pair || !timeframe) return;
            ['tradingview_chart_bybit', 'tradingview_chart_binance'].forEach(id => document.getElementById(id).innerHTML = '');
            const tfMap = { "1m":"1", "3m":"3", "5m":"5", "15m":"15", "30m":"30", "1H":"60", "2H":"120", "4H":"240", "1D":"D", "1W":"W"};
            const interval = tfMap[timeframe] || "60";
            const commonSettings = { autosize: true, interval: interval, timezone: "Etc/UTC", theme: "dark", style: "1", locale: "en", enable_publishing: false, withdateranges: true, hide_side_toolbar: false, allow_symbol_change: true, disabled_features: ["header_widget", "header_symbol_search", "header_resolutions"], studies: [{ id: "MAExp@tv-basicstudies", inputs: { length: 9 } }], overrides: { "study.Moving Average Exponential.plot.color": "#3B82F6", "paneProperties.background": "#1A1A1A", "paneProperties.vertGridProperties.color": "#2A2A2A", "paneProperties.horzGridProperties.color": "#2A2A2A" } };
            new TradingView.widget({ ...commonSettings, symbol: `BYBIT:${pair.replace('-', '')}.P`, container_id: "tradingview_chart_bybit" });
            new TradingView.widget({ ...commonSettings, symbol: `BINANCE:${pair.replace('-', '')}PERP`, container_id: "tradingview_chart_binance" });
        };
        
        const getPairCardHTML = (pair, data) => {
            const safePairId = pair.replace(/[^a-zA-Z0-9]/g, '');
            return `<div class="pair-header"><div class="pair-name-section"><div class="pair-name">${pair}</div><div class="pair-tf" id="tf-${safePairId}">${data.timeframe}</div></div><div class="timer-container"><svg class="timer-svg" viewBox="0 0 52 52"><circle class="timer-circle timer-circle-bg" cx="26" cy="26" r="${SVG_CIRCLE_RADIUS}"></circle><circle class="timer-circle timer-circle-progress" id="timer-progress-${safePairId}" cx="26" cy="26" r="${SVG_CIRCLE_RADIUS}" stroke-dasharray="${SVG_CIRCUMFERENCE}" stroke-dashoffset="0"></circle></svg><div id="timer-text-${safePairId}" class="timer-text">--:--</div></div></div><div class="pair-info-grid"><span>Price</span><strong id="price-${safePairId}">${formatPrice(data.price)}</strong><span>Funding</span><strong id="funding-${safePairId}">${formatPercent(data.funding)}</strong></div><div id="action-area-${safePairId}" class="action-area"></div>`;
        };

        const getActionAreaHTML = (pair, data) => {
            if (data.open_position) {
                return `<div class="position-info"><div class="position-header">${data.open_position.type} POSITION</div><div class="position-pnl" id="pnl-${pair.replace(/[^a-zA-Z0-9]/g, '')}">${formatPercent(data.pnl)}</div><div class="position-entry">Entry @ ${formatPrice(data.open_position.entryPrice)}</div><form class="trade-form" data-url="/trade/close" data-body='{"trade_id":"${data.open_position.id}"}'><button type="submit" class="btn btn-close">Close Position</button></form></div>`;
            } else {
                return `<div style="display:flex; gap:1rem; margin-top:auto;"><form class="trade-form" data-url="/trade/manual" data-body='{"pair":"${pair}","type":"LONG"}'><button type="submit" class="btn btn-long">Long</button></form><form class="trade-form" data-url="/trade/manual" data-body='{"pair":"${pair}","type":"SHORT"}'><button type="submit" class="btn btn-short">Short</button></form></div>`;
            }
        };
        
        const updateUI = (data) => {
            const aiBtn = document.getElementById('ai-status-btn');
            if (aiBtn) {
                aiBtn.className = `action-btn ai-status ${data.is_ai_running ? 'running' : 'stopped'}`;
                aiBtn.textContent = `AI ${data.is_ai_running ? 'Running' : 'Paused'}`;
            }
            updateText(document.getElementById('pnl-today'), formatPercent(data.pnl_today), data.pnl_today >= 0 ? 'update-g' : 'update-r');
            updateText(document.getElementById('pnl-this-week'), formatPercent(data.pnl_this_week), data.pnl_this_week >= 0 ? 'update-g' : 'update-r');
            updateText(document.getElementById('pnl-last-week'), formatPercent(data.pnl_last_week), data.pnl_last_week >= 0 ? 'update-g' : 'update-r');
            document.getElementById('pnl-today').className = 'value ' + getPnlColorClass(data.pnl_today);
            document.getElementById('pnl-this-week').className = 'value ' + getPnlColorClass(data.pnl_this_week);
            document.getElementById('pnl-last-week').className = 'value ' + getPnlColorClass(data.pnl_last_week);
            
            const watchlistEl = document.getElementById('watchlist');
            const incomingPairs = new Set(Object.keys(data.market_data));
            
            Object.entries(data.market_data).forEach(([pair, d], index) => {
                const safePairId = pair.replace(/[^a-zA-Z0-9]/g, '');
                candleStartTimes[pair] = { startTimeMs: d.current_candle_start_time_ms, timeframe: d.timeframe };
                let card = document.getElementById(`card-${safePairId}`);
                if (!card) {
                    card = document.createElement('div');
                    card.id = `card-${safePairId}`;
                    card.dataset.pair = pair;
                    card.className = 'pair-card';
                    card.style.animationDelay = `${index * 50}ms`;
                    watchlistEl.appendChild(card);
                }
                card.innerHTML = getPairCardHTML(pair, d); // Re-render card content to handle position open/close
                
                const actionArea = document.getElementById(`action-area-${safePairId}`);
                actionArea.innerHTML = getActionAreaHTML(pair, d);
                
                card.classList.toggle('position-open', !!d.open_position);
                card.classList.toggle('active-chart', pair === currentChartPair);
            });
            [...watchlistEl.children].forEach(card => { if (!incomingPairs.has(card.dataset.pair)) card.remove(); });
            
            document.getElementById('history-list').innerHTML = data.trades.map((t, i) => {
                const pnlNet = t.status === 'CLOSED' ? (t.pl_percent - (2 * data.settings.fee_pct)) : null;
                return `<li class="history-item" style="animation-delay: ${i*50}ms"><div class="history-main"><span class="history-type ${t.type==='LONG'?'text-green':'text-red'}">${t.type}</span><span class="history-pair">${t.instrumentId}</span></div><div class="history-pnl ${getPnlColorClass(pnlNet)}">${t.status==='CLOSED'?formatPercent(pnlNet):'OPEN'}</div><div class="history-details">Entry @ ${formatPrice(t.entryPrice)} • ${t.entryReason.split('\\n')[0]}</div></li>`;
            }).join('');
            
            Object.entries(data.settings).forEach(([k, v]) => {
                const i = document.getElementById(`s-${k}`);
                if(i && document.activeElement !== i) { if (i.type === 'checkbox') { i.checked = v; } else { i.value = v; } }
                if (k === 'watched_pairs') { 
                    const listEl = document.getElementById('watchlist-list');
                    if (listEl) listEl.innerHTML = Object.entries(v).map(([p,tf])=>`<li><span>${p} (${tf})</span><button class="btn-remove" data-pair="${p}">×</button></li>`).join(''); 
                } 
            });
        };
        
        const updateCountdowns = () => {
            for (const pair in candleStartTimes) {
                const safePairId = pair.replace(/[^a-zA-Z0-9]/g, '');
                const textEl = document.getElementById(`timer-text-${safePairId}`);
                const progressEl = document.getElementById(`timer-progress-${safePairId}`);
                if (!textEl || !progressEl) continue; 
                const { startTimeMs, timeframe } = candleStartTimes[pair];
                const durationSeconds = TIMEFRAME_SECONDS[timeframe];
                if (!startTimeMs || !durationSeconds) continue; 
                const endTimeMs = startTimeMs + (durationSeconds * 1000);
                const remainingMs = Math.max(0, endTimeMs - Date.now());
                const progress = remainingMs / (durationSeconds * 1000);
                progressEl.style.strokeDashoffset = SVG_CIRCUMFERENCE * (1 - progress);
                const totalSeconds = Math.floor(remainingMs / 1000);
                const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
                const seconds = String(totalSeconds % 60).padStart(2, '0');
                textEl.textContent = `${minutes}:${seconds}`;
            }
        };
        
        // --- Event Listeners ---
        const fetchData = async () => { try { const res = await fetch(API_ENDPOINT); if (!res.ok) return; const data = await res.json(); updateUI(data); } catch(e) { console.error("Update failed:", e); } };
        
        document.getElementById('watchlist').addEventListener('click', e => { 
            const card = e.target.closest('.pair-card'); 
            if (card && card.dataset.pair && card.dataset.pair !== currentChartPair) { 
                currentChartPair = card.dataset.pair;
                fetch(API_ENDPOINT).then(res=>res.json()).then(data => {
                    const tf = data.settings.watched_pairs[currentChartPair];
                    createChartWidgets(currentChartPair, tf);
                    document.querySelectorAll('.pair-card').forEach(c => c.classList.remove('active-chart')); 
                    card.classList.add('active-chart'); 
                });
            }
        });
        document.body.addEventListener('submit', async e => { 
            if(e.target.matches('.trade-form')) {
                e.preventDefault(); 
                const f = e.target; 
                await postRequest(f.dataset.url, JSON.parse(f.dataset.body.replace(/'/g, '"')));
                await fetchData();
            } else if (e.target.id === 'settings-form') {
                e.preventDefault();
                await postRequest('/api/settings', Object.fromEntries(new FormData(e.target).entries()));
                window.location.reload();
            }
        });
        
        const modal = document.getElementById('settings-modal');
        document.getElementById('settings-btn').addEventListener('click', () => modal.classList.add('visible'));
        document.getElementById('close-settings-btn').addEventListener('click', () => modal.classList.remove('visible'));
        document.getElementById('ai-status-btn').addEventListener('click', async () => { await postRequest('/toggle-ai', {}); await fetchData(); });
        document.getElementById('add-pair-btn').addEventListener('click', async () => { const p = document.getElementById('new-pair-input').value.toUpperCase(); const tf = document.getElementById('new-tf-input').value; if(p){ await postRequest('/api/watchlist/add', {pair:p, tf:tf}); await fetchData(); document.getElementById('new-pair-input').value = ''; }});
        document.getElementById('watchlist-list').addEventListener('click', async e => { if (e.target.matches('.btn-remove')) { e.preventDefault(); await postRequest('/api/watchlist/remove', {pair: e.target.dataset.pair}); await fetchData(); }});
        
        // --- Initial Load ---
        const initialLoad = async () => {
            try {
                const res = await fetch(API_ENDPOINT);
                const data = await res.json();
                if (Object.keys(data.settings.watched_pairs).length > 0) {
                    currentChartPair = Object.keys(data.settings.watched_pairs)[0];
                    createChartWidgets(currentChartPair, data.settings.watched_pairs[currentChartPair]);
                }
                updateUI(data);
            } catch (e) {
                console.error("Initial load failed:", e);
            }
        };

        initialLoad();
        setInterval(fetchData, REFRESH_INTERVAL_MS);
        setInterval(updateCountdowns, 250);
    });
    </script>
</body>
</html>
"""

# --- RUTE FLASK (Backend) ---
@app.route('/')
def dashboard(): return render_template_string(HTML_SKELETON_TRADINGVIEW, current_settings=current_settings)

@app.route('/api/data')
def get_api_data():
    with state_lock: trades_copy = list(trades); market_state_copy = dict(market_state); settings_copy = dict(current_settings)
    market_data_view = {}
    fee_pct = settings_copy.get('fee_pct', 0.1)
    for pair_id, timeframe in settings_copy.get("watched_pairs", {}).items():
        pair_state = market_state_copy.get(pair_id, {})
        full_candle_data = pair_state.get("candle_data", [])
        current_price = 0.0; current_candle_start_time_ms = 0
        if full_candle_data:
            current_price = full_candle_data[-1].get('close', 0.0)
            current_candle_start_time_ms = full_candle_data[-1].get('time', 0)
        open_pos = next((t for t in trades_copy if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        pnl = 0.0
        if open_pos and current_price > 0: pnl = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - fee_pct
        market_data_view[pair_id] = {
            "price": current_price, "funding": pair_state.get("funding_rate", 0.0),
            "timeframe": timeframe, "open_position": open_pos, "pnl": pnl,
            "current_candle_start_time_ms": current_candle_start_time_ms
        }
    return jsonify({"is_ai_running": is_autopilot_running, "pnl_today": calculate_todays_pnl(trades_copy), "pnl_this_week": calculate_this_weeks_pnl(trades_copy), "pnl_last_week": calculate_last_weeks_pnl(trades_copy), "market_data": market_data_view, "trades": trades_copy, "settings": settings_copy})

@app.route('/toggle-ai', methods=['POST'])
def toggle_ai():
    global is_autopilot_running
    is_autopilot_running = not is_autopilot_running
    print_colored(f"Autopilot {'diaktifkan' if is_autopilot_running else 'dimatikan'} dari Web UI.", Fore.YELLOW)
    return jsonify(success=True)

@app.route('/trade/manual', methods=['POST'])
def trade_manual():
    data = request.form; pair = data.get('pair'); trade_type = data.get('type')
    if not pair or not trade_type: return jsonify(success=False, error="Data tidak lengkap"), 400
    pair_state = market_state.get(pair, {}); candle_data = pair_state.get("candle_data")
    current_price = candle_data[-1].get('close') if candle_data else None
    if not current_price: return jsonify(success=False, error="Harga tidak tersedia"), 400
    entry_snapshot = {}
    if candle_data and len(candle_data) >= 100 + 3:
        with state_lock: relevant_trades_history = [t for t in trades if t['instrumentId'] == pair]
        ai_analyzer = LocalAI(current_settings, relevant_trades_history)
        analysis_result = ai_analyzer.get_market_analysis(candle_data)
        if analysis_result:
            analysis_result["funding_rate"] = pair_state.get("funding_rate", 0.0)
            entry_snapshot = analysis_result
    with state_lock:
        if any(t for t in trades if t['instrumentId'] == pair and t['status'] == 'OPEN'): return jsonify(success=False, error="Posisi sudah ada"), 400
        new_trade = { "id": int(time.time()), "instrumentId": pair, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": current_price, "entryReason": "Manual Entry", "status": 'OPEN', "exitPrice": None, "pl_percent": None, "entry_snapshot": entry_snapshot }
        trades.insert(0, new_trade)
        print_colored(f"Trade Manual {trade_type} {pair} @ {current_price} dibuka.", Fore.BLUE)
    save_trades(); return jsonify(success=True)

@app.route('/trade/close', methods=['POST'])
def trade_close():
    trade_id = int(request.form.get('trade_id'))
    trade_to_close = None
    with state_lock: trade_to_close = next((t for t in trades if t['id'] == trade_id and t['status'] == 'OPEN'), None)
    if not trade_to_close: return jsonify(success=False, error="Trade tidak ditemukan"), 404
    pair = trade_to_close['instrumentId']
    current_price = market_state.get(pair, {}).get("candle_data", [{}])[-1].get('close')
    if not current_price: return jsonify(success=False, error="Harga tidak tersedia"), 400
    close_trade_sync(trade_to_close, current_price, "Manual Close")
    return jsonify(success=True)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global current_settings
    with state_lock:
        current_settings['use_trailing_tp'] = 'use_trailing_tp' in request.form
        for key, value in request.form.items():
            if key in current_settings and isinstance(current_settings[key], (int, float)):
                try:
                    if '.' in value: current_settings[key] = float(value)
                    else: current_settings[key] = int(value)
                except (ValueError, TypeError): print_colored(f"Nilai tidak valid untuk {key}: {value}", Fore.RED)
        save_settings()
    print_colored("Pengaturan diperbarui dari Web UI. Halaman akan dimuat ulang untuk menerapkan interval refresh.", Fore.GREEN)
    return jsonify(success=True)

@app.route('/api/watchlist/add', methods=['POST'])
def add_watchlist():
    pair = request.form.get('pair'); tf = request.form.get('tf', '1H')
    if pair:
        with state_lock: current_settings['watched_pairs'][pair] = tf; save_settings()
        print_colored(f"{pair} ({tf}) ditambahkan ke watchlist.", Fore.GREEN)
    return jsonify(success=True)

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_watchlist():
    pair = request.form.get('pair')
    with state_lock:
        if pair and pair in current_settings['watched_pairs']:
            del current_settings['watched_pairs'][pair]; save_settings()
            print_colored(f"{pair} dihapus dari watchlist.", Fore.YELLOW)
    return jsonify(success=True)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    load_settings(); load_trades(); display_welcome_message()
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    autopilot_thread.start(); data_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)
