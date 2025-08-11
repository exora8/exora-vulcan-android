import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta, timezone # Added timezone
import asyncio
import math
from flask import Flask, render_template_string, jsonify, request
import traceback
import hmac
import hashlib
import random # Added for optimization

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
BINGX_API_URL = "https://open-api.bingx.com" # Base URL
BYBIT_API_URL = "https://api.bybit.com" # URL Bybit untuk backtest

# --- STATE APLIKASI ---
current_settings = {}
trades = []
market_state = {}
is_ai_thinking = False
is_autopilot_running = True
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
state_lock = threading.Lock()
backtest_state = {"is_running": False, "progress": 0, "message": "Idle", "total_trades": 0, "max_trades": 0}
backtest_lock = threading.Lock()

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
        "stop_loss_pct": 0.24, "fee_pct": 0.05, "analysis_interval_sec": 5,
        "use_trailing_tp": False, "trailing_tp_activation_pct": 0.55,
        "trailing_tp_gap_pct": 0.05, "caution_level": 0.55,
        "max_allowed_funding_rate_pct": 0.075, "watched_pairs": {"BRETT-USDT": "5m"},
        "max_trades_in_history": 800, "refresh_interval_seconds": 1, "chart_candle_limit": 80,
        "similarity_threshold_win": 12, "similarity_threshold_loss": 12,
        "cooldown_candles_after_trade": 3,
        "is_real_trading": False,
        "bingx_api_key": "",
        "bingx_api_secret": "",
        "leverage": 10,
        "risk_usdt_per_trade": 5.0,
        "min_win_to_risk_ratio": 0.5,
        "auto_settings_enabled": False # NEW: Auto Settings flag
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

# --- FUNGSI API BINGX (PUBLIC & PRIVATE) ---
def generate_signature(params_str, secret_key):
    return hmac.new(secret_key.encode(), params_str.encode(), hashlib.sha256).hexdigest()

def bingx_request(method, path, params, settings):
    api_key = settings.get("bingx_api_key")
    secret_key = settings.get("bingx_api_secret")
    if not api_key or not secret_key:
        print_colored("API Key/Secret tidak diatur untuk request private.", Fore.RED)
        return None, "API Key/Secret not set"

    params['timestamp'] = int(time.time() * 1000)
    params_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = generate_signature(params_str, secret_key)

    headers = {'X-BX-APIKEY': api_key}
    url = f"{BINGX_API_URL}{path}?{params_str}&signature={signature}"

    try:
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, timeout=15)
        else: # GET
            response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0:
            return data.get('data'), None
        else:
            return None, data.get('msg', 'Unknown BingX Error')
    except Exception as e:
        print_colored(f"Error saat request ke BingX {path}: {e}", Fore.RED)
        return None, str(e)

def place_real_order(symbol, trade_type, quantity, price, settings):
    leverage = settings.get("leverage", 10)
    
    leverage_params = {'symbol': symbol, 'side': trade_type, 'leverage': leverage}
    _, err_leverage = bingx_request('POST', '/openApi/swap/v2/trade/leverage', leverage_params, settings)
    if err_leverage:
        print_colored(f"Gagal mengatur leverage ke {leverage}x untuk {symbol}: {err_leverage}", Fore.RED)
        return None, f"Leverage error: {err_leverage}"

    order_params = {
        'symbol': symbol,
        'side': 'BUY' if trade_type == 'LONG' else 'SELL',
        'positionSide': 'LONG' if trade_type == 'LONG' else 'SHORT',
        'type': 'MARKET',
        'quantity': quantity
    }
    data, error = bingx_request('POST', '/openApi/swap/v2/trade/order', order_params, settings)
    if error:
        print_colored(f"Gagal membuka posisi real {trade_type} {symbol}: {error}", Fore.RED)
        return None, error

    print_colored(f"Order REAL berhasil ditempatkan: {data.get('orderId')}", Fore.GREEN)
    return data.get('orderId'), None

def close_real_order(symbol, trade_type, quantity, settings):
    close_params = {
        'symbol': symbol,
        'side': 'SELL' if trade_type == 'LONG' else 'BUY',
        'positionSide': 'LONG' if trade_type == 'LONG' else 'SHORT',
        'type': 'MARKET',
        'quantity': quantity
    }
    data, error = bingx_request('POST', '/openApi/swap/v2/trade/order', close_params, settings)
    if error:
        print_colored(f"Gagal menutup posisi real {trade_type} {symbol}: {error}", Fore.RED)
        return False, error

    print_colored(f"Posisi REAL {symbol} berhasil ditutup.", Fore.MAGENTA)
    return True, None

def fetch_funding_rate(instId):
    bingx_symbol = instId
    try:
        url = f"{BINGX_API_URL}/openApi/swap/v2/quote/premiumIndex?symbol={bingx_symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0 and 'data' in data and data['data']:
            return float(data['data'][0].get('lastFundingRate', '0')) * 100 # Adjusted key
        return None
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None

def fetch_recent_candles(instId, timeframe, limit=300, end_time_ms=None):
    timeframe_map_str = {'1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m', '1H': '1h', '2H': '2h', '4H': '4h', '1D': '1d', '1W': '1w'}
    bingx_interval = timeframe_map_str.get(timeframe, '5m')
    bingx_symbol = instId
    try:
        url = f"{BINGX_API_URL}/openApi/swap/v2/quote/klines?symbol={bingx_symbol}&interval={bingx_interval}&limit={limit}"
        if end_time_ms: url += f"&endTime={end_time_ms}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0 and 'data' in data:
            candle_list = data['data']
            if not candle_list: return None
            return [{"time": int(d["time"]), "open": float(d["open"]), "high": float(d["high"]), "low": float(d["low"]), "close": float(d["close"]), "volume": float(d["volume"])} for d in candle_list]
        return None
    except (requests.exceptions.RequestException, Exception):
        return None

def fetch_bybit_backtest_candles(instId, timeframe, limit=1000, end_time_ms=None):
    timeframe_map_bybit = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
        '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'
    }
    bybit_interval = timeframe_map_bybit.get(timeframe)
    if not bybit_interval:
        print_colored(f"Timeframe tidak valid untuk Bybit: {timeframe}", Fore.RED)
        return None
    bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/v5/market/kline"
        params = {'category': 'linear', 'symbol': bybit_symbol, 'interval': bybit_interval, 'limit': limit}
        if end_time_ms: params['end'] = end_time_ms
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0 and 'result' in data and data['result'].get('list'):
            candle_list = data['result']['list']
            if not candle_list: return None
            formatted_candles = [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list]
            return formatted_candles
        else:
            print_colored(f"Bybit API Error for {bybit_symbol}: {data.get('retMsg', 'Unknown error')}", Fore.YELLOW)
            return None
    except (requests.exceptions.RequestException, Exception) as e:
        print_colored(f"Error fetching backtest data from Bybit: {e}", Fore.RED)
        return None

# --- FUNGSI KALKULASI & ANALISIS ---

def calculate_pnl(entry_price, current_price, trade_type):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

def is_trade_considered_a_win(trade, settings):
    if not trade or trade.get('status') != 'CLOSED' or trade.get('pl_percent') is None:
        return False
    net_pnl_percent = trade.get('pl_percent', 0.0) - (2 * settings.get('fee_pct', 0.1))
    if net_pnl_percent <= 0:
        return False
    if trade.get('entryReason') == "Manual Entry":
        return True
    stop_loss_pct = settings.get('stop_loss_pct', 1.0)
    min_ratio = settings.get('min_win_to_risk_ratio', 0.5)
    if stop_loss_pct <= 0:
        return True
    required_pnl = stop_loss_pct * min_ratio
    if net_pnl_percent >= required_pnl:
        return True
    return False

def calculate_todays_pnl(all_trades, settings):
    today_utc = datetime.now(timezone.utc).date(); total_pnl = 0.0; fee_pct = settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '+00:00')).date() == today_utc: total_pnl += (trade.get('pl_percent', 0.0) - (2 * fee_pct))
            except (ValueError, TypeError): continue
    return total_pnl

def calculate_this_weeks_pnl(all_trades, settings):
    today_utc = datetime.now(timezone.utc).date(); start_of_week_utc = today_utc - timedelta(days=today_utc.weekday())
    total_pnl = 0.0; fee_pct = settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '+00:00')).date()
                if start_of_week_utc <= exit_date <= today_utc: total_pnl += (trade.get('pl_percent', 0.0) - (2 * fee_pct))
            except (ValueError, TypeError): continue
    return total_pnl

def calculate_last_weeks_pnl(all_trades, settings):
    today_utc = datetime.now(timezone.utc).date(); start_of_current_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_last_week_utc = start_of_current_week_utc - timedelta(days=1); start_of_last_week_utc = end_of_last_week_utc - timedelta(days=6)
    total_pnl = 0.0; fee_pct = settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade and trade.get('pl_percent') is not None:
            try:
                exit_date = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '+00:00')).date()
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
        if len(candle_data) < 100 + 15: return None
        ema9 = self.calculate_ema(candle_data, 9)
        ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100)
        if len(ema9) < 2 or not ema50 or not ema100: return None
        pre_entry_candles = candle_data[-16:-1]; pre_entry_ema9 = ema9[-16:-1]
        analysis = {
            "ema9_current": ema9[-1], "ema9_prev": ema9[-2], "ema50": ema50[-1], "ema100": ema100[-1],
            "current_candle_close": candle_data[-1]['close'], "prev_candle_close": candle_data[-2]['close'],
            "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING",
            "pre_entry_candle_solidity": [self.analyze_candle_solidity(c) for c in pre_entry_candles],
            "pre_entry_candle_direction": ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles],
            "details": {"candles": [{"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]} for c in pre_entry_candles], "ema9": pre_entry_ema9}
        }
        return analysis

    def compare_setups(self, current_analysis, past_snapshot):
        if not past_snapshot or not past_snapshot.get('details'): return 0
        similarity_score = 1
        current_pos_vs_ema50 = 'above' if current_analysis['current_candle_close'] > current_analysis['ema50'] else 'below'
        past_pos_vs_ema50 = 'above' if past_snapshot.get('current_candle_close', 0) > past_snapshot.get('ema50', 0) else 'below'
        if current_pos_vs_ema50 == past_pos_vs_ema50: similarity_score += 1
        current_pos_vs_ema9 = 'above' if current_analysis['current_candle_close'] > current_analysis['ema9_current'] else 'below'
        past_pos_vs_ema9 = 'above' if past_snapshot.get('current_candle_close', 0) > past_snapshot.get('ema9_current', 0) else 'below'
        if current_pos_vs_ema9 == past_pos_vs_ema9: similarity_score += 1
        current_ema9_slope = 'up' if current_analysis['ema9_current'] > current_analysis['ema9_prev'] else 'down'
        past_ema9_slope = 'up' if past_snapshot.get('ema9_current', 0) > past_snapshot.get('ema9_prev', 0) else 'down'
        if current_ema9_slope == past_ema9_slope: similarity_score += 1
        if 'pre_entry_candle_solidity' in current_analysis and 'pre_entry_candle_solidity' in past_snapshot:
            avg_solidity_current = sum(current_analysis['pre_entry_candle_solidity']) / 15
            past_solidity_list = past_snapshot.get('pre_entry_candle_solidity', ([0] * 15)); avg_solidity_past = sum(past_solidity_list) / 15 if past_solidity_list else 0
            if abs(avg_solidity_current - avg_solidity_past) < 0.2: similarity_score += 1
        current_dirs = current_analysis.get('pre_entry_candle_direction'); past_dirs = past_snapshot.get('pre_entry_candle_direction')
        if current_dirs and past_dirs:
            match_count = 0
            for i in range(1, min(len(current_dirs), len(past_dirs)) + 1):
                if current_dirs[-i] == past_dirs[-i]: match_count += 1
                else: break
            similarity_score += match_count
        return similarity_score

    def find_best_match(self, current_analysis, trade_list):
        best_match = None; highest_score = 0
        if not trade_list: return None, 0
        for trade in trade_list:
            snapshot = trade.get("entry_snapshot")
            if not snapshot or not snapshot.get('details'): continue
            score = self.compare_setups(current_analysis, snapshot)
            if score > highest_score:
                highest_score = score
                best_match = trade
        return best_match, highest_score

    def get_decision(self, candle_data, open_position, funding_rate=0.0):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data teknikal tidak cukup."}
        if open_position: return {"action": "HOLD", "reason": "Posisi terbuka."}

        winning_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and is_trade_considered_a_win(t, self.settings)]
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and not is_trade_considered_a_win(t, self.settings)]

        best_win_match, win_score = self.find_best_match(analysis, winning_trades)
        if best_win_match and win_score >= self.settings.get("similarity_threshold_win", 12):
            best_loss_match, loss_score = self.find_best_match(analysis, losing_trades)
            if best_loss_match and loss_score >= self.settings.get("similarity_threshold_loss", 12):
                 return {"action": "HOLD", "reason": f"High Confidence Win (Skor: {win_score}) dibatalkan oleh kemiripan Loss (Skor: {loss_score})"}
            reason = f"High Confidence: Mirip win valid (ID: {best_win_match.get('id', 'N/A')}, Skor: {win_score})"
            return {"action": "BUY" if best_win_match.get('type') == 'LONG' else "SELL", "reason": reason, "snapshot": analysis}

        potential_trade_type = None
        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']: potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']: potential_trade_type = 'SHORT'

        if potential_trade_type:
            current_candle = candle_data[-1]
            ema9_current = analysis['ema9_current']
            candle_touched_ema9 = (current_candle['low'] <= ema9_current <= current_candle['high'])
            if not candle_touched_ema9: return {"action": "HOLD", "reason": f"Sinyal batal. Candle tidak menyentuh EMA 9."}

            best_loss_match, loss_score = self.find_best_match(analysis, losing_trades)
            if best_loss_match and loss_score >= self.settings.get("similarity_threshold_loss", 12):
                return {"action": "HOLD", "reason": f"Peringatan: Mirip loss/lucky win ID {best_loss_match.get('id', 'N/A')}. Skor: {loss_score}"}
            if potential_trade_type == 'LONG' and funding_rate > self.settings.get("max_allowed_funding_rate_pct", 0.075): return {"action": "HOLD", "reason": f"Sinyal LONG batal. Funding rate tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -self.settings.get("max_allowed_funding_rate_pct", 0.075): return {"action": "HOLD", "reason": f"Sinyal SHORT batal. Funding rate negatif: {funding_rate:.4f}%"}
            avg_solidity = sum(analysis.get('pre_entry_candle_solidity', [0])) / 15
            if avg_solidity < self.settings.get("caution_level", 0.5): return {"action": "HOLD", "reason": f"Sinyal batal. Pasar ragu-ragu (Solidity: {avg_solidity:.2f})"}
            ai_reason = (f"AI: {potential_trade_type} berdasarkan konfirmasi tren {analysis['bias']}.")
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": ai_reason, "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias: {analysis['bias']}."}

    def get_similarity_analysis_for_dashboard(self, current_analysis):
        winning_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and is_trade_considered_a_win(t, self.settings)]
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and not is_trade_considered_a_win(t, self.settings)]

        best_win_match, win_score = self.find_best_match(current_analysis, winning_trades)
        best_loss_match, loss_score = self.find_best_match(current_analysis, losing_trades)
        dashboard_data = {"current_details": current_analysis.get('details')}
        if best_win_match:
            dashboard_data['win_match'] = {"id": best_win_match.get('id'), "score": win_score, "details": best_win_match.get('entry_snapshot', {}).get('details')}
        if best_loss_match:
             dashboard_data['loss_match'] = {"id": best_loss_match.get('id'), "score": loss_score, "details": best_loss_match.get('entry_snapshot', {}).get('details')}
        return dashboard_data

def close_trade_sync(trade, exit_price, reason):
    with state_lock:
        if trade.get('is_real'):
            success, error = close_real_order(trade['instrumentId'], trade.get('type'), trade.get('quantity'), current_settings)
            if not success:
                print_colored(f"KRITIS: Gagal menutup posisi real untuk trade ID {trade['id']}. Error: {error}. Tidak mengubah status trade lokal.", Fore.RED, Style.BRIGHT)
                send_termux_notification("‼️ GAGAL TUTUP POSISI REAL ‼️", f"ID: {trade['id']}, Pair: {trade['instrumentId']}, Error: {error}")
                return

        pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type'))
        exit_dt = datetime.now(timezone.utc)
        trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat().replace('+00:00', 'Z'), 'pl_percent': pnl_gross })

        instrument_id = trade['instrumentId']
        cooldown_candles = current_settings.get('cooldown_candles_after_trade', 0)
        if cooldown_candles > 0 and instrument_id in market_state:
            timeframe_str = current_settings.get("watched_pairs", {}).get(instrument_id, "5m")
            tf_map_ms = {'1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000, '1H': 3600000}
            candle_duration_ms = tf_map_ms.get(timeframe_str, 300000)
            cooldown_duration_ms = cooldown_candles * candle_duration_ms
            last_candle_ms = market_state[instrument_id].get("candle_data", [{}])[-1].get("time", int(exit_dt.timestamp() * 1000))
            market_state[instrument_id]['cooldown_until_timestamp'] = last_candle_ms + cooldown_duration_ms
            end_time = datetime.fromtimestamp((last_candle_ms + cooldown_duration_ms) / 1000, timezone.utc)
            print_colored(f"[{instrument_id}] Cooldown diaktifkan untuk {cooldown_candles} lilin. Tidak ada trade baru sampai setelah {end_time.strftime('%H:%M:%S')} UTC.", Fore.YELLOW)

    save_trades()
    pnl_net = pnl_gross - (2 * current_settings.get('fee_pct', 0.1))
    mode = "REAL" if trade.get('is_real') else "DEMO"
    notif_title = f"🔴 Posisi {mode} {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | {reason}"
    send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.MAGENTA)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking: return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 15: return
    current_candle_time = pair_state["candle_data"][-1]['time']
    cooldown_end_time = pair_state.get('cooldown_until_timestamp', 0)
    if current_candle_time < cooldown_end_time: return
    is_ai_thinking = True
    try:
        with state_lock: open_pos = next((t for t in trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades_history = [t for t in trades if t['instrumentId'] == instrument_id]
        ai = LocalAI(current_settings, relevant_trades_history)
        funding_rate = pair_state.get("funding_rate", 0.0)
        decision = ai.get_decision(pair_state["candle_data"], open_pos, funding_rate)
        if decision.get('action') in ["BUY", "SELL"] and not open_pos:
            is_real = current_settings.get("is_real_trading", False)
            entry_price = pair_state["candle_data"][-1]['close']
            trade_type = "LONG" if decision['action'] == "BUY" else "SHORT"
            quantity = 0
            if is_real:
                risk_usdt = float(current_settings.get('risk_usdt_per_trade', 5.0))
                sl_pct = float(current_settings.get('stop_loss_pct', 0.25))
                if risk_usdt <= 0 or sl_pct <= 0:
                    print_colored(f"REAL TRADE DIBATALKAN: Risk per Trade atau Stop Loss harus > 0.", Fore.RED)
                    is_ai_thinking = False; return
                sl_size_in_usdt = entry_price * (sl_pct / 100)
                if sl_size_in_usdt == 0:
                    print_colored(f"REAL TRADE DIBATALKAN: Kalkulasi SL menghasilkan nol.", Fore.RED)
                    is_ai_thinking = False; return
                quantity = risk_usdt / sl_size_in_usdt
                bingx_symbol = instrument_id
                order_id, error = place_real_order(bingx_symbol, trade_type, quantity, entry_price, current_settings)
                if error:
                    send_termux_notification(f"‼️ GAGAL BUKA POSISI REAL ‼️", f"{instrument_id} {trade_type}: {error}")
                    is_ai_thinking = False; return
            snapshot = decision.get("snapshot", {}); snapshot["funding_rate"] = funding_rate
            new_trade = {
                "id": int(time.time()), "instrumentId": instrument_id, "type": trade_type,
                "entryTimestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), "entryPrice": entry_price,
                "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot,
                "exitPrice": None, "pl_percent": None, "is_real": is_real, "quantity": quantity
            }
            with state_lock: trades.insert(0, new_trade)
            save_trades()
            mode = "REAL" if is_real else "DEMO"
            notif_title = f"🟢 Posisi {mode} {trade_type} Dibuka: {instrument_id}"
            notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | Qty: {quantity:.4f} | {decision.get('reason')}"
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
            sl_price = entry_price * (1 - sl_pct / 100)
            if candle_low <= sl_price:
                close_trade_sync(trade_obj, sl_price, f"Stop Loss @ {-sl_pct:.2f}%"); return
        elif trade_type == 'SHORT':
            sl_price = entry_price * (1 + sl_pct / 100)
            if candle_high >= sl_price:
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

# --- NEW: AUTO SETTINGS WORKER AND HELPERS ---

def _run_optimization_backtest_instance(settings_override):
    sim_settings = current_settings.copy()
    sim_settings.update(settings_override)
    
    with state_lock:
        initial_trades = list(trades)

    if not initial_trades:
        return None

    initial_trades.sort(key=lambda x: x['entryTimestamp'])
    oldest_trade = initial_trades[0]
    instrument_id = oldest_trade['instrumentId']
    timeframe = sim_settings.get("watched_pairs", {}).get(instrument_id, "1H")
    start_dt = datetime.fromisoformat(oldest_trade['entryTimestamp'].replace('Z', '+00:00'))
    end_timestamp_ms = int(start_dt.timestamp() * 1000)
    max_trades_limit = sim_settings.get("max_trades_in_history", 800)

    backtest_open_pos = None
    found_trades = []
    
    while True:
        if len(initial_trades) + len(found_trades) >= max_trades_limit:
            break
        
        candle_batch = fetch_bybit_backtest_candles(instrument_id, timeframe, limit=1000, end_time_ms=end_timestamp_ms)
        if not candle_batch or len(candle_batch) <= 1:
            break 
        
        next_end_timestamp_ms = candle_batch[-1]['time']
        candle_batch.reverse()

        for i in range(100 + 15, len(candle_batch)):
            if len(initial_trades) + len(found_trades) >= max_trades_limit:
                break
            
            current_simulation_time_ms = candle_batch[i]['time']
            candle_window = candle_batch[i - (100 + 15) : i + 1]
            current_candle = candle_window[-1]
            
            if backtest_open_pos:
                entry_price = backtest_open_pos['entryPrice']
                trade_type = backtest_open_pos['type']
                exit_price = None
                sl_pct = sim_settings.get('stop_loss_pct', 0)
                if sl_pct > 0:
                    if trade_type == 'LONG' and current_candle['low'] <= entry_price * (1 - sl_pct / 100):
                        exit_price = entry_price * (1 - sl_pct / 100)
                    elif trade_type == 'SHORT' and current_candle['high'] >= entry_price * (1 + sl_pct / 100):
                        exit_price = entry_price * (1 + sl_pct / 100)
                
                if not exit_price and not sim_settings.get('use_trailing_tp', True):
                    static_tp_pct = sim_settings.get("trailing_tp_activation_pct", 0)
                    if static_tp_pct > 0:
                        if trade_type == 'LONG' and current_candle['high'] >= entry_price * (1 + static_tp_pct/100):
                            exit_price = entry_price * (1 + static_tp_pct/100)
                        elif trade_type == 'SHORT' and current_candle['low'] <= entry_price * (1 - static_tp_pct/100):
                            exit_price = entry_price * (1 - static_tp_pct/100)
                
                if exit_price:
                    pnl_gross = calculate_pnl(backtest_open_pos['entryPrice'], exit_price, backtest_open_pos['type'])
                    exit_dt = datetime.fromtimestamp(current_simulation_time_ms / 1000, timezone.utc)
                    backtest_open_pos.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat().replace('+00:00', 'Z'), 'pl_percent': pnl_gross})
                    found_trades.append(backtest_open_pos)
                    backtest_open_pos = None

            if not backtest_open_pos:
                current_trade_history = initial_trades + found_trades
                ai = LocalAI(sim_settings, current_trade_history)
                decision = ai.get_decision(candle_window, None, 0.0)
                if decision.get('action') in ["BUY", "SELL"]:
                    entry_price = current_candle['close']
                    entry_dt = datetime.fromtimestamp(current_simulation_time_ms / 1000, timezone.utc)
                    backtest_open_pos = {
                        "id": int(entry_dt.timestamp()), "instrumentId": instrument_id, 
                        "type": "LONG" if decision['action'] == "BUY" else "SHORT", 
                        "entryTimestamp": entry_dt.isoformat().replace('+00:00', 'Z'), "entryPrice": entry_price, 
                        "entryReason": "Opti-Backtest", "status": 'OPEN', "entry_snapshot": decision.get("snapshot", {})
                    }
        
        end_timestamp_ms = next_end_timestamp_ms
        time.sleep(0.1)

    closed_trades = [t for t in found_trades if t.get('status') == 'CLOSED']
    if not closed_trades:
        return {'trades': [], 'win_rate': 0, 'net_pnl': 0}

    win_count = sum(1 for t in closed_trades if is_trade_considered_a_win(t, sim_settings))
    win_rate = win_count / len(closed_trades) if closed_trades else 0
    fee = sim_settings.get('fee_pct', 0.0)
    net_pnl = sum(t.get('pl_percent', 0.0) - (2 * fee) for t in closed_trades)
    
    return {'trades': found_trades, 'win_rate': win_rate, 'net_pnl': net_pnl}

def calculate_fitness(run_result):
    if not run_result or not run_result.get('trades'):
        return 0

    closed_trades = [t for t in run_result['trades'] if t.get('status') == 'CLOSED']
    num_trades = len(closed_trades)
    net_pnl = run_result.get('net_pnl', 0)

    if num_trades < 20:
        return 0

    gross_profit = sum(t.get('pl_percent', 0) for t in closed_trades if t.get('pl_percent', 0) > 0)
    gross_loss = abs(sum(t.get('pl_percent', 0) for t in closed_trades if t.get('pl_percent', 0) < 0))

    if gross_loss == 0:
        profit_factor = 100
    else:
        profit_factor = gross_profit / gross_loss
    
    if net_pnl <= 0:
        return 0
        
    fitness_score = (math.log(1 + net_pnl) * profit_factor)

    return fitness_score

def auto_settings_worker():
    MAX_ITERATIONS = 100
    MAX_TIME_SECONDS = 7200
    NO_IMPROVE_LIMIT = 15
    
    PARAM_RANGES = {
        'caution_level': (0.3, 0.8), 'min_win_to_risk_ratio': (0.3, 1.5),
        'win_similarity_threshold': (10, 16), 'loss_similarity_threshold': (10, 16),
        'cooldown_candles_after_trade': (0, 8),
        'stop_loss_pct': (0.1, 2.0),
        'trailing_tp_activation_pct': (0.15, 4.0)
    }

    with backtest_lock:
        if backtest_state["is_running"]: return
        backtest_state.update({"is_running": True, "message": "Initializing Auto Settings...", "total_trades": 0, "max_trades": MAX_ITERATIONS})

    start_time = time.time()
    log_data = []
    
    try:
        print_colored("--- Starting Auto-Settings Optimization ---", Fore.CYAN, Style.BRIGHT)
        print_colored("Running baseline test with current settings...", Fore.CYAN)
        with backtest_lock:
            backtest_state["message"] = "Running baseline with current settings..."
        
        baseline_result = _run_optimization_backtest_instance(settings_override={})
        if not baseline_result or not baseline_result['trades']:
            raise Exception("Baseline run failed or found no trades.")

        best_config = {k: current_settings.get(k) for k in PARAM_RANGES.keys()}
        best_fitness = calculate_fitness(baseline_result)
        best_win_rate = baseline_result['win_rate']
        best_pnl = baseline_result['net_pnl']
        no_improve_count = 0
        
        log_data.append({'iteration': 0, 'params': best_config, 'result': baseline_result, 'fitness_score': best_fitness, 'is_baseline': True})
        print_colored(f"Baseline -> Fitness: {best_fitness:.2f}, WR: {best_win_rate:.2%}, PnL: {best_pnl:.2f}%", Fore.GREEN)

        for i in range(1, MAX_ITERATIONS):
            if stop_event.is_set() or (time.time() - start_time) > MAX_TIME_SECONDS:
                print_colored("Optimization stopped: Time limit or stop signal.", Fore.YELLOW); break
            if no_improve_count >= NO_IMPROVE_LIMIT:
                print_colored(f"Stopping: No improvement in {NO_IMPROVE_LIMIT} iterations.", Fore.YELLOW); break

            with backtest_lock:
                backtest_state["message"] = f"Iter {i}/{MAX_ITERATIONS} | Best Fitness: {best_fitness:.2f}"
                backtest_state["total_trades"] = i
            
            sl = round(random.uniform(*PARAM_RANGES['stop_loss_pct']), 3)
            tp = round(random.uniform(sl, PARAM_RANGES['trailing_tp_activation_pct'][1]), 3)

            new_params = {
                'stop_loss_pct': sl,
                'trailing_tp_activation_pct': tp,
                'caution_level': round(random.uniform(*PARAM_RANGES['caution_level']), 3),
                'min_win_to_risk_ratio': round(random.uniform(*PARAM_RANGES['min_win_to_risk_ratio']), 3),
                'win_similarity_threshold': random.randint(*PARAM_RANGES['win_similarity_threshold']),
                'loss_similarity_threshold': random.randint(*PARAM_RANGES['loss_similarity_threshold']),
                'cooldown_candles_after_trade': random.randint(*PARAM_RANGES['cooldown_candles_after_trade'])
            }
            
            run_result = _run_optimization_backtest_instance(settings_override=new_params)
            current_fitness = calculate_fitness(run_result)
            
            if current_fitness == 0:
                print_colored(f"Iter {i}: Skipped (unprofitable or too few trades).", bright=Style.DIM); continue
            
            run_result['fitness_score'] = current_fitness
            log_data.append({'iteration': i, 'params': new_params, 'result': run_result})

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_config = new_params
                best_win_rate = run_result['win_rate']
                best_pnl = run_result['net_pnl']
                no_improve_count = 0
                print_colored(f"Iter {i}: Fitness: {current_fitness:.2f}, WR: {run_result['win_rate']:.2%}, PnL: {run_result['net_pnl']:.2f}% -> New Best!", Fore.GREEN)
            else:
                print_colored(f"Iter {i}: Fitness: {current_fitness:.2f}, WR: {run_result['win_rate']:.2%}, PnL: {run_result['net_pnl']:.2f}%", Fore.WHITE)
                no_improve_count += 1
        
        print_colored("\n--- Optimization Finished ---", Fore.CYAN, Style.BRIGHT)
        print_colored(f"Best Fitness Score: {best_fitness:.2f}", Fore.GREEN, Style.BRIGHT)
        print_colored(f"Best Win Rate: {best_win_rate:.2%}, Best PnL: {best_pnl:.2f}%", Fore.GREEN, Style.BRIGHT)
        print_colored("Best Parameters:", Fore.GREEN)
        for k, v in best_config.items(): print(f"  {k}: {v}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"auto_settings_log.{timestamp}.json"
        with open(log_filename, 'w') as f: json.dump(log_data, f, indent=4)
        print_colored(f"Optimization log saved to {log_filename}", Fore.CYAN)
        
        if os.path.exists(SETTINGS_FILE):
            backup_filename = f"{SETTINGS_FILE}.bak.{timestamp}"
            os.rename(SETTINGS_FILE, backup_filename)
            print_colored(f"Original settings backed up to {backup_filename}", Fore.CYAN)
        
        with state_lock:
            current_settings.update(best_config)
            save_settings()
        
        print_colored(f"SUCCESS: '{SETTINGS_FILE}' has been updated with the best parameters.", Fore.GREEN, Style.BRIGHT)
        with backtest_lock: backtest_state.update({"is_running": False, "message": "Optimization Complete!"})

    except (Exception, KeyboardInterrupt) as e:
        print_colored(f"\nAuto Settings worker failed or was cancelled: {e}", Fore.RED); traceback.print_exc()
        with backtest_lock: backtest_state.update({"is_running": False, "message": f"Error/Cancelled: {e}"})

# --- END NEW FUNCTIONS ---

def backtest_worker():
    with backtest_lock:
        if backtest_state["is_running"]: return
        load_trades()
        with state_lock: trades_copy = list(trades)
        if not trades_copy:
            backtest_state.update({"is_running": False, "message": "Error: No trades in history to start backtest from."})
            return
        backtest_state.update({"is_running": True, "message": "Initializing backtest...", "progress": 0})
    try:
        trades_copy.sort(key=lambda x: x['entryTimestamp'])
        oldest_trade = trades_copy[0]
        instrument_id = oldest_trade['instrumentId']
        timeframe = current_settings.get("watched_pairs", {}).get(instrument_id, "1H")
        start_dt = datetime.fromisoformat(oldest_trade['entryTimestamp'].replace('Z', '+00:00'))
        end_timestamp_ms = int(start_dt.timestamp() * 1000)
        max_trades = current_settings.get("max_trades_in_history", 80)
        with backtest_lock:
            backtest_state["max_trades"] = max_trades
            backtest_state["total_trades"] = len(trades_copy)
        print_colored(f"--- Starting Backtest for {instrument_id} from {start_dt.strftime('%Y-%m-%d %H:%M')} ---", Fore.CYAN)
        backtest_open_pos = None
        newly_found_trades = []
        while True:
            with state_lock: current_total_trades = len(trades)
            if current_total_trades >= max_trades:
                with backtest_lock: backtest_state.update({"is_running": False, "message": "Backtest complete: Max trade history reached."})
                print_colored("Backtest complete: Max trade history reached.", Fore.GREEN)
                break
            candle_batch = None
            max_retries = 5
            for attempt in range(max_retries):
                dt_str = datetime.fromtimestamp(end_timestamp_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')
                print_colored(f"Fetching backtest data from Bybit for {instrument_id} before {dt_str}...", Fore.CYAN, end='\r')
                candle_batch = fetch_bybit_backtest_candles(instrument_id, timeframe, limit=1000, end_time_ms=end_timestamp_ms)
                if candle_batch and len(candle_batch) > 1:
                    print()
                    break
                print_colored(f"\nGagal mengambil data backtest dari Bybit, mencoba lagi dalam 5 detik... (Percobaan {attempt + 1}/{max_retries})", Fore.YELLOW)
                time.sleep(5)
            if not candle_batch or len(candle_batch) <= 1:
                with backtest_lock: backtest_state.update({"is_running": False, "message": "Backtest complete: No more historical data from Bybit."})
                print_colored("\nBacktest complete: No more historical data available from Bybit after retries.", Fore.YELLOW)
                break
            next_end_timestamp_ms = candle_batch[-1]['time']
            candle_batch.reverse()
            for i in range(100 + 15, len(candle_batch)):
                if len(trades) + len(newly_found_trades) >= max_trades: break
                current_simulation_time_ms = candle_batch[i]['time']
                candle_window = candle_batch[i - (100 + 15) : i + 1]
                current_candle = candle_window[-1]
                if backtest_open_pos:
                    entry_price = backtest_open_pos['entryPrice']; trade_type = backtest_open_pos['type']
                    exit_price, exit_reason = None, ""
                    sl_pct = current_settings.get('stop_loss_pct', 0)
                    if sl_pct > 0:
                        if trade_type == 'LONG' and current_candle['low'] <= entry_price * (1 - sl_pct / 100): exit_price, exit_reason = entry_price * (1 - sl_pct / 100), f"Backtest SL @ {-sl_pct:.2f}%"
                        elif trade_type == 'SHORT' and current_candle['high'] >= entry_price * (1 + sl_pct / 100): exit_price, exit_reason = entry_price * (1 + sl_pct / 100), f"Backtest SL @ {-sl_pct:.2f}%"
                    if not exit_price and not current_settings.get('use_trailing_tp', True):
                        static_tp_pct = current_settings.get("trailing_tp_activation_pct", 0)
                        if static_tp_pct > 0:
                            if trade_type == 'LONG' and current_candle['high'] >= entry_price * (1 + static_tp_pct/100): exit_price, exit_reason = entry_price * (1 + static_tp_pct/100), f"Backtest Static TP @ {static_tp_pct:.2f}%"
                            elif trade_type == 'SHORT' and current_candle['low'] <= entry_price * (1 - static_tp_pct/100): exit_price, exit_reason = entry_price * (1 - static_tp_pct/100), f"Backtest Static TP @ {static_tp_pct:.2f}%"
                    if exit_price:
                        pnl_gross = calculate_pnl(backtest_open_pos['entryPrice'], exit_price, backtest_open_pos['type'])
                        exit_dt = datetime.fromtimestamp(current_simulation_time_ms / 1000, timezone.utc)
                        backtest_open_pos.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat().replace('+00:00', 'Z'), 'pl_percent': pnl_gross})
                        print_colored(f"Backtest Close: {exit_reason} at {exit_dt.strftime('%Y-%m-%d %H:%M')}, PnL: {pnl_gross:.2f}%", Fore.MAGENTA)
                        newly_found_trades.append(backtest_open_pos)
                        backtest_open_pos = None
                if not backtest_open_pos:
                    with state_lock: current_trade_history = [t for t in trades if t['instrumentId'] == instrument_id] + newly_found_trades
                    ai = LocalAI(current_settings, current_trade_history)
                    decision = ai.get_decision(candle_window, None, 0.0)
                    if decision.get('action') in ["BUY", "SELL"]:
                        entry_price = current_candle['close']
                        entry_dt = datetime.fromtimestamp(current_simulation_time_ms / 1000, timezone.utc)
                        new_trade = {"id": int(entry_dt.timestamp()), "instrumentId": instrument_id, "type": "LONG" if decision['action'] == "BUY" else "SHORT", "entryTimestamp": entry_dt.isoformat().replace('+00:00', 'Z'), "entryPrice": entry_price, "entryReason": f"Backtest: {decision.get('reason')}", "status": 'OPEN', "entry_snapshot": decision.get("snapshot", {}), "exitPrice": None, "pl_percent": None}
                        backtest_open_pos = new_trade
                        print_colored(f"Backtest Open: {new_trade['type']} @ {entry_price:.4f} on {entry_dt.strftime('%Y-%m-%d %H:%M')}", Fore.GREEN)
            end_timestamp_ms = next_end_timestamp_ms
            with backtest_lock:
                backtest_state["total_trades"] = len(trades) + len(newly_found_trades)
                backtest_state["message"] = f"Backtesting... currently at {datetime.fromtimestamp(end_timestamp_ms / 1000, timezone.utc).strftime('%Y-%m-%d')}"
            if newly_found_trades:
                with state_lock: trades.extend(newly_found_trades)
                save_trades()
                newly_found_trades = []
            time.sleep(0.5)
    except Exception as e:
        print_colored(f"An error occurred during backtest: {e}", Fore.RED); traceback.print_exc()
        with backtest_lock: backtest_state.update({"is_running": False, "message": f"Error: {e}"})
    finally:
        if newly_found_trades:
            with state_lock: trades.extend(newly_found_trades)
            save_trades()
        with backtest_lock:
            if backtest_state["is_running"]: backtest_state.update({"is_running": False, "message": "Backtest stopped."})

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
        :root { --bg-color: #121212; --card-color: #1E1E1E; --border-color: #333; --text-color: #EAEAEA; --text-muted: #888; --green: #34D399; --red: #F87171; --yellow: #FBBF24; --accent-primary: #60A5FA; --blue: #3B82F6;}
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; font-size: 16px; }
        body { background-color: var(--bg-color); color: var(--text-color); font-family: 'Inter', sans-serif; margin: 0; padding: 1rem; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
            overscroll-behavior-y: contain;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { margin: 0; font-size: 1.75rem; }
        h2 { margin-top: 2.5rem; margin-bottom: 1rem; font-size: 1.25rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }
        h1, h2 { font-weight: 600; letter-spacing: -0.5px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
        .header-actions { display: flex; gap: 1rem; }
        .action-btn { background-color: var(--card-color); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; cursor: pointer; transition: background-color 0.2s ease, border-color 0.2s ease; }
        .action-btn:hover:not(:disabled) { background-color: var(--border-color); }
        .action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .action-btn.ai-status.running { color: var(--green); }
        .action-btn.ai-status.stopped { color: var(--red); }
        .action-btn.trade-mode-real { color: var(--red); font-weight: 700; border-color: var(--red); }
        .action-btn.trade-mode-demo { color: var(--green); }
        .pnl-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1.5rem; }
        .stat-item { background-color: var(--card-color); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 12px; transition: transform 0.2s ease; }
        .stat-item:hover { transform: translateY(-3px); }
        .stat-item .label { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem; }
        .stat-item .value { font-size: 1.75rem; font-weight: 700; }
        .chart-wrapper { margin-bottom: 2rem; }
        .tradingview-widget-container { height: 450px; touch-action: none; }
        .watchlist { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.5rem; }
        .pair-card { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; display: flex; flex-direction: column; cursor: pointer; }
        .pair-card.active-chart { border-color: var(--accent-primary); }
        .pair-card.position-open { border-left: 4px solid var(--accent-primary); }
        .pair-card.position-open.is-real { border-left-color: var(--red); }
        .pair-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1rem; }
        .pair-name { font-size: 1.5rem; font-weight: 600; }
        .pair-name .real-badge { font-size: 0.7rem; color: var(--red); border: 1px solid var(--red); padding: 2px 5px; border-radius: 4px; margin-left: 8px; vertical-align: middle;}
        .pair-price { font-size: 1.25rem; color: var(--text-muted); }
        .pair-info { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem; }
        .pair-actions { display: flex; gap: 1rem; margin-top: auto; }
        .btn { flex-grow: 1; padding: 0.75rem; border-radius: 8px; border: none; font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, opacity 0.2s ease; }
        .btn:hover { transform: scale(1.03); opacity: 0.9; }
        .btn-long { background-color: var(--green); color: #fff; }
        .btn-short { background-color: var(--red); color: #fff; }
        .btn-close { background-color: var(--yellow); color: var(--bg-color); }
        .position-info { border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; text-align: center; margin-top: auto;}
        .position-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
        .position-pnl { font-size: 1.75rem; font-weight: 700; margin-bottom: 1rem; }
        .history-list { list-style: none; padding: 0; }
        .history-item { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; }
        .history-main { display: flex; align-items: center; gap: 1rem; }
        .history-type { font-weight: 600; font-size: 1.1rem; }
        .history-type .real-badge { font-size: 0.7rem; color: var(--red); border: 1px solid var(--red); padding: 2px 5px; border-radius: 4px; margin-left: 8px; vertical-align: middle;}
        .history-pair { color: var(--text-muted); }
        .history-pnl { font-size: 1.25rem; font-weight: 600; text-align: right; }
        .history-pnl.status-valid-win { color: var(--green); }
        .history-pnl.status-loss { color: var(--red); }
        .history-pnl.status-lucky-win { color: var(--yellow); }
        .history-pnl.status-open { color: var(--accent-primary); }
        .history-details { color: var(--text-muted); font-size: 0.85rem; width: 100%; text-align: left; }
        .settings-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(5px); display: none; justify-content: center; align-items: center; z-index: 1000; opacity: 0; transition: opacity 0.3s ease; }
        .settings-modal.visible { display: flex; opacity: 1; }
        .settings-content { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 2rem; width: 90%; max-width: 600px; max-height: 90vh; overflow-y: auto; }
        .settings-content h3 { margin-top: 2rem; margin-bottom: 1rem; font-size: 1.1rem; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { color: var(--text-muted); margin-bottom: 0.5rem; font-size: 0.9rem; }
        .form-group input { background-color: var(--bg-color); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.75rem; border-radius: 8px; font-size: 1rem; }
        .form-group.checkbox-group { flex-direction: row; align-items: center; gap: 0.5rem; }
        .form-group.checkbox-group label { margin-bottom: 0; }
        .watchlist-manage ul { list-style: none; padding: 0; }
        .watchlist-manage li { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; }
        .btn-remove { background: none; border: none; color: var(--red); cursor: pointer; font-size: 1.25rem; }
        .text-green { color: var(--green); } .text-red { color: var(--red); } .text-yellow { color: var(--yellow); }
        .fullscreen-btn { background: none; border: none; cursor: pointer; padding: 0.25rem; color: var(--text-muted); transition: color 0.2s ease; }
        .fullscreen-btn:hover { color: var(--text-color); }
        .fullscreen-btn svg { width: 18px; height: 18px; }
        .chart-fullscreen { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 5000; background: var(--bg-color); padding: 1rem; }
        .chart-fullscreen .tradingview-widget-container { height: 100% !important; }
        .is-hidden { display: none !important; }
        #ai-global-analysis-wrapper { margin-bottom: 2rem; }
        .analysis-container { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .analysis-card { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; }
        .analysis-title { font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem 0; }
        .analysis-title.win { color: var(--green); }
        .analysis-title.loss { color: var(--red); }
        .analysis-placeholder { color: var(--text-muted); text-align: center; padding: 2rem; }
        .pa-chart-container { display: flex; position: relative; width: 100%; height: 100px; background-color: rgba(0,0,0,0.2); border-radius: 8px; padding: 5px; }
        .pa-candle { flex: 1; position: relative; margin: 0 1px; }
        .pa-wick { position: absolute; left: 50%; width: 1px; transform: translateX(-50%); background-color: var(--text-muted); }
        .pa-body { position: absolute; left: 25%; width: 50%; }
        .pa-body.green { background-color: var(--green); }
        .pa-body.red { background-color: var(--red); }
        .pa-ema-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: visible; }
        .pa-ema-path { stroke: var(--accent-primary); stroke-width: 1.5; fill: none; }
        progress {-webkit-appearance: none; appearance: none; width: 100%; height: 8px; border: none; border-radius: 4px; background-color: var(--bg-color);}
        progress::-webkit-progress-bar { background-color: var(--bg-color); border-radius: 4px; }
        progress::-webkit-progress-value { background-color: var(--accent-primary); border-radius: 4px; transition: width 0.3s ease;}
        progress::-moz-progress-bar { background-color: var(--accent-primary); border-radius: 4px; transition: width 0.3s ease;}
        .api-warning { background-color: #442020; border: 1px solid var(--red); padding: 1rem; border-radius: 8px; margin-top: 1rem; }

        @media (max-width: 768px) {
            h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; }
            .pnl-stats, .watchlist, .form-grid, .analysis-container { grid-template-columns: 1fr; }
            .header { flex-direction: column; align-items: flex-start; gap: 1rem; }
            .history-item { flex-direction: column; align-items: flex-start; }
            .history-pnl { width: 100%; text-align: left; margin-top: 0.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Vulcan AI</h1>
            <div class="header-actions">
                <button id="ai-status-btn" class="action-btn ai-status"></button>
                <button id="trade-mode-btn" class="action-btn"></button>
                <button id="backtest-btn" class="action-btn">Backtest</button>
                <button id="settings-btn" class="action-btn">Settings</button>
            </div>
        </header>
        <section id="backtest-status-wrapper" class="is-hidden" style="margin-bottom: 2rem; background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem;">
            <h3 id="backtest-title" style="margin:0 0 1rem 0; font-size: 1.1rem; color: var(--accent-primary);">Backtest in Progress</h3>
            <p id="backtest-message" style="margin: 0 0 1rem 0; color: var(--text-muted);"></p>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <progress id="backtest-progress" value="0" max="100"></progress>
                <span id="backtest-trade-count" style="white-space: nowrap; font-weight: 500;">0 / 80</span>
            </div>
        </section>
        <section id="pnl-stats" class="pnl-stats"></section>
        <div class="chart-wrapper" id="bingx-chart-wrapper">
            <h2 id="bingx-chart-title">BingX Perp Chart
                <button class="fullscreen-btn" data-target="#bingx-chart-wrapper" aria-label="Toggle Fullscreen">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m4.5 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" /></svg>
                </button>
            </h2>
            <div id="tradingview_chart_bingx" class="tradingview-widget-container"></div>
        </div>
        <section id="ai-global-analysis-wrapper">
            <h2>AI Pattern Analysis</h2>
            <div id="ai-global-analysis-content"></div>
        </section>
        <h2 id="watchlist-title">Watchlist</h2>
        <section id="watchlist" class="watchlist"></section>
        <h2 id="history-title">Recent History</h2>
        <ul id="history-list" class="history-list"></ul>
    </div>

    <div id="settings-modal" class="settings-modal">
        <div class="settings-content">
            <div style="display:flex; justify-content:space-between; align-items:center;"><h2>Settings</h2><button id="close-settings-btn" style="background:none; border:none; color:var(--text-color); font-size: 2rem; cursor:pointer;">×</button></div>
            <form id="settings-form">
                <h3>Real Trading & API (GUNAKAN DENGAN RISIKO ANDA SENDIRI)</h3>
                <div class="form-grid">
                    <div class="form-group"><label>BingX API Key</label><input type="text" name="bingx_api_key" id="s-bingx_api_key"></div>
                    <div class="form-group"><label>BingX API Secret</label><input type="password" name="bingx_api_secret" id="s-bingx_api_secret"></div>
                    <div class="form-group"><label>Leverage (misal: 10)</label><input type="number" step="1" name="leverage" id="s-leverage"></div>
                    <div class="form-group"><label>Risk per Trade (USDT)</label><input type="number" step="any" name="risk_usdt_per_trade" id="s-risk_usdt_per_trade"></div>
                </div>
                <div class="api-warning"><strong>Peringatan:</strong> Menyimpan API key di sini tidak aman. Pastikan tidak ada yang bisa mengakses file <code>settings.json</code> Anda. Trading otomatis sangat berisiko.</div>
                <h3>Trading Parameters</h3>
                <div class="form-grid">
                    <div class="form-group"><label>Fee per Transaction (%)</label><input type="number" step="any" name="fee_pct" id="s-fee_pct"></div>
                    <div class="form-group"><label>Stop Loss (%)</label><input type="number" step="any" name="stop_loss_pct" id="s-stop_loss_pct"></div>
                    <div class="form-group checkbox-group"><input type="checkbox" name="use_trailing_tp" id="s-use_trailing_tp"><label for="s-use_trailing_tp">Enable Trailing TP</label></div>
                    <div class="form-group"><label>TP Activation / Static TP (%)</label><input type="number" step="any" name="trailing_tp_activation_pct" id="s-trailing_tp_activation_pct"></div>
                    <div class="form-group"><label>TP Gap (for Trailing)</label><input type="number" step="any" name="trailing_tp_gap_pct" id="s-trailing_tp_gap_pct"></div>
                    <div class="form-group"><label>Max Funding Rate (%)</label><input type="number" step="any" name="max_allowed_funding_rate_pct" id="s-max_allowed_funding_rate_pct"></div>
                </div>
                <h3>AI & Security Parameters</h3>
                <div class="form-grid">
                    <div class="form-group"><label>Caution Level (0-1)</label><input type="number" step="any" name="caution_level" id="s-caution_level"></div>
                    <div class="form-group"><label>Min Win/Risk Ratio</label><input type="number" step="any" name="min_win_to_risk_ratio" id="s-min_win_to_risk_ratio" title="Minimum Net PnL / SL % for a win to be 'valid' for the AI. Wins below this are treated as losses."></div>
                    <div class="form-group"><label>Win Similarity Threshold</label><input type="number" step="1" name="similarity_threshold_win" id="s-similarity_threshold_win"></div>
                    <div class="form-group"><label>Loss Similarity Threshold</label><input type="number" step="1" name="similarity_threshold_loss" id="s-similarity_threshold_loss"></div>
                    <div class="form-group"><label>Cooldown (Candles)</label><input type="number" step="1" name="cooldown_candles_after_trade" id="s-cooldown_candles_after_trade"></div>
                </div>
                <!-- NEW: AUTO SETTINGS SECTION -->
                <h3>Backtest & Optimization</h3>
                 <div class="form-grid">
                    <div class="form-group checkbox-group" style="grid-column: 1 / -1;">
                        <input type="checkbox" name="auto_settings_enabled" id="s-auto_settings_enabled">
                        <label for="s-auto_settings_enabled">Enable Auto-Settings Optimization during Backtest</label>
                    </div>
                </div>
                <h3>System Parameters</h3>
                <div class="form-grid">
                    <div class="form-group"><label>AI Delay (s)</label><input type="number" step="1" name="analysis_interval_sec" id="s-analysis_interval_sec"></div>
                    <div class="form-group"><label>Max Trade History</label><input type="number" step="10" name="max_trades_in_history" id="s-max_trades_in_history"></div>
                    <div class="form-group"><label>Data Refresh (s)</label><input type="number" step="any" name="refresh_interval_seconds" id="s-refresh_interval_seconds"></div>
                </div>
                <h3>Watchlist</h3>
                <div class="watchlist-manage"><ul id="watchlist-list"></ul>
                    <div class="form-group" style="margin-top:1rem;"><label>Add New Pair (e.g., BTC-USDT)</label>
                        <div style="display:flex; gap:1rem;">
                            <input type="text" id="new-pair-input" placeholder="Pair" style="flex-grow:1;"><input type="text" id="new-tf-input" value="1H" placeholder="Timeframe" style="width:100px;">
                            <button type="button" id="add-pair-btn" class="action-btn" style="background-color: var(--accent-primary); border:none;">Add</button>
                        </div>
                    </div>
                </div>
                <button type="submit" class="action-btn" style="width:100%; margin-top: 2rem; padding: 0.75rem; background-color:var(--accent-primary); border:none;">Save Settings</button>
            </form>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const API_ENDPOINT_BASE = '/api/data';
            let REFRESH_INTERVAL_MS = {{ current_settings.refresh_interval_seconds * 1000 }};
            const formatPercent = v => typeof v === 'number' ? v.toFixed(2) + '%' : 'N/A';
            const formatPrice = v => typeof v === 'number' ? (v < 1 ? v.toPrecision(4) : v.toFixed(2)) : 'N/A';
            const getTrendColorClass = v => v === 'Bullish' ? 'text-green' : (v === 'Bearish' ? 'text-red' : 'text-yellow');
            const getPnlColorClass = v => v > 0 ? 'text-green' : 'text-red';
            const postRequest = async (url, data) => { try { await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: new URLSearchParams(data) }); } catch (e) { console.error(`POST to ${url} failed:`, e); }};
            let currentChartPair = null; let lastData = {};
            const renderPriceActionChart = (details) => {
                if (!details || !details.candles || !details.ema9 || details.candles.length === 0) return '<div class="analysis-placeholder" style="font-size:0.8rem;">Chart data not available</div>';
                const { candles, ema9 } = details;
                const allPrices = candles.flatMap(c => [c.high, c.low]).concat(ema9);
                const maxPrice = Math.max(...allPrices);
                const minPrice = Math.min(...allPrices);
                const priceRange = maxPrice - minPrice;
                if (priceRange === 0) return '<div class="analysis-placeholder">Price data is flat.</div>';
                const normalize = price => 100 * (maxPrice - price) / priceRange;
                let candlesHtml = '';
                candles.forEach(c => {
                    const top = normalize(Math.max(c.open, c.close));
                    const height = Math.max(0.5, 100 * Math.abs(c.open - c.close) / priceRange);
                    const wickTop = normalize(c.high); const wickHeight = 100 * (c.high - c.low) / priceRange;
                    const colorClass = c.close >= c.open ? 'green' : 'red';
                    candlesHtml += `<div class="pa-candle"><div class="pa-wick" style="top:${wickTop.toFixed(2)}%; height:${wickHeight.toFixed(2)}%;"></div><div class="pa-body ${colorClass}" style="top:${top.toFixed(2)}%; height:${height.toFixed(2)}%;"></div></div>`;
                });
                const candleWidth = 100 / candles.length;
                let pathD = 'M ';
                ema9.forEach((val, i) => { const x = (i + 0.5) * candleWidth; const y = normalize(val); pathD += `${x.toFixed(2)},${y.toFixed(2)} `; });
                const svgHtml = `<svg class="pa-ema-svg" preserveAspectRatio="none" viewBox="0 0 100 100"><path class="pa-ema-path" d="${pathD.trim()}"></path></svg>`;
                return `<div class="pa-chart-container">${candlesHtml}${svgHtml}</div>`;
            };
            const updateGlobalAnalysisPanel = (analysisData) => {
                const container = document.getElementById('ai-global-analysis-content');
                if (!analysisData || !analysisData.current_details) { container.innerHTML = `<div class="analysis-placeholder">No active analysis for ${currentChartPair || 'selected pair'}.</div>`; return; }
                let html = '<div class="analysis-container">';
                html += `<div class="analysis-card"><h3 class="analysis-title">Current Price Action</h3>${renderPriceActionChart(analysisData.current_details)}</div>`;
                let matchCardHtml = '';
                const winMatch = analysisData.win_match; const lossMatch = analysisData.loss_match;
                let bestMatch = null; let isWin = false;
                if (winMatch && lossMatch) { if (winMatch.score >= lossMatch.score) { bestMatch = winMatch; isWin = true; } else { bestMatch = lossMatch; isWin = false; } } 
                else if (winMatch) { bestMatch = winMatch; isWin = true; } else if (lossMatch) { bestMatch = lossMatch; isWin = false; }
                if (bestMatch) { matchCardHtml = `<div class="analysis-card"><h3 class="analysis-title ${isWin ? 'win' : 'loss'}">Best Match: ${isWin ? 'Valid Win' : 'Loss/Lucky Win'} ID #${bestMatch.id} (Score: ${bestMatch.score})</h3>${renderPriceActionChart(bestMatch.details)}</div>`; } 
                else { matchCardHtml = `<div class="analysis-card"><h3 class="analysis-title">No Strong Match Found</h3><div class="analysis-placeholder">Waiting for more history.</div></div>`; }
                html += matchCardHtml + '</div>';
                container.innerHTML = html;
            };
            const createChartWidgets = (pair, timeframe) => {
                document.getElementById('tradingview_chart_bingx').innerHTML = '';
                const tfMap = { "1m":"1", "3m":"3", "5m":"5", "15m":"15", "30m":"30", "1H":"60", "2H":"120", "4H":"240", "1D":"D", "1W":"W"};
                const interval = tfMap[timeframe] || "60";
                const commonSettings = { "autosize": true, "interval": interval, "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en", "enable_publishing": false, "withdateranges": true, "hide_side_toolbar": false, "allow_symbol_change": true, "disabled_features": ["header_widget"], "studies": [{ "id": "MAExp@tv-basicstudies", "inputs": { "length": 9 } }], "overrides": { "study.Moving Average Exponential.plot.color": "#60A5FA" } };
                new TradingView.widget({ ...commonSettings, "symbol": `BINGX:${pair.replace('-', '')}.P`, "container_id": "tradingview_chart_bingx" });
                document.getElementById('bingx-chart-title').childNodes[0].nodeValue = `${pair} BingX Perp Chart `;
            };
            const updateUI = data => {
                if (!currentChartPair && Object.keys(data.settings.watched_pairs).length > 0) {
                    currentChartPair = Object.keys(data.settings.watched_pairs)[0];
                    createChartWidgets(currentChartPair, data.settings.watched_pairs[currentChartPair]);
                }
                document.getElementById('ai-status-btn').className = `action-btn ai-status ${data.is_ai_running ? 'running' : 'stopped'}`;
                document.getElementById('ai-status-btn').textContent = `AI ${data.is_ai_running ? 'Running' : 'Paused'}`;
                const tradeModeBtn = document.getElementById('trade-mode-btn');
                tradeModeBtn.className = `action-btn trade-mode-${data.settings.is_real_trading ? 'real' : 'demo'}`;
                tradeModeBtn.textContent = data.settings.is_real_trading ? '🔴 REAL' : '🟢 DEMO';
                document.getElementById('pnl-stats').innerHTML = `<div class="stat-item"><div class="label">Today's P/L</div><div class="value ${getPnlColorClass(data.pnl_today)}">${formatPercent(data.pnl_today)}</div></div><div class="stat-item"><div class="label">This Week</div><div class="value ${getPnlColorClass(data.pnl_this_week)}">${formatPercent(data.pnl_this_week)}</div></div><div class="stat-item"><div class="label">Last Week</div><div class="value ${getPnlColorClass(data.pnl_last_week)}">${formatPercent(data.pnl_last_week)}</div></div>`;
                const watchlistEl = document.getElementById('watchlist');
                watchlistEl.innerHTML = '';
                Object.entries(data.market_data).forEach(([p, d]) => {
                    const card = document.createElement('div');
                    const isReal = d.open_position && d.open_position.is_real;
                    card.className = `pair-card ${d.open_position ? 'position-open' : ''} ${p === currentChartPair ? 'active-chart' : ''} ${isReal ? 'is-real' : ''}`;
                    card.dataset.pair = p;
                    const actionHTML = d.open_position
                        ? `<div class="position-info"><div class="position-header">${d.open_position.type} POSITION ${isReal ? '<span class="real-badge">REAL</span>' : ''}</div><div class="position-pnl ${getPnlColorClass(d.pnl)}">${formatPercent(d.pnl)}</div><div style="font-size:0.9rem; color:var(--text-muted); margin-bottom:1rem;">Entry @ ${formatPrice(d.open_position.entryPrice)}</div><form class="trade-form" data-url="/trade/close" data-body='{"trade_id":"${d.open_position.id}"}'><button type="submit" class="btn btn-close">Close</button></form></div>`
                        : `<div class="pair-actions"><form class="trade-form" data-url="/trade/manual" data-body='{"pair":"${p}","type":"LONG"}'><button type="submit" class="btn btn-long">Long</button></form><form class="trade-form" data-url="/trade/manual" data-body='{"pair":"${p}","type":"SHORT"}'><button type="submit" class="btn btn-short">Short</button></form></div>`;
                    card.innerHTML = `<div class="pair-header"><span class="pair-name">${p}</span><span class="pair-price">${formatPrice(d.price)}</span></div><div class="pair-info"><span>TF: <strong>${d.timeframe}</strong></span><span>Trend: <strong class="${getTrendColorClass(d.trend)}">${d.trend}</strong></span><span>Funding: <strong class="${d.funding > 0.01 ? 'text-red' : ''}">${formatPercent(d.funding)}</strong></span></div>${actionHTML}`;
                    watchlistEl.appendChild(card);
                });
                updateGlobalAnalysisPanel(data.global_ai_analysis);
                const backtestState = data.backtest_state;
                const backtestWrapper = document.getElementById('backtest-status-wrapper');
                const backtestBtn = document.getElementById('backtest-btn');
                if (backtestState && backtestState.is_running) {
                    backtestWrapper.classList.remove('is-hidden');
                    const isOptimizing = data.settings.auto_settings_enabled;
                    document.getElementById('backtest-title').textContent = isOptimizing ? 'Auto-Settings Optimization in Progress' : 'Backtest in Progress';
                    document.getElementById('backtest-message').textContent = backtestState.message;
                    const progress = backtestState.max_trades > 0 ? (backtestState.total_trades / backtestState.max_trades) * 100 : 0;
                    document.getElementById('backtest-progress').value = progress;
                    const countLabel = isOptimizing ? 'Iteration' : 'Trades';
                    document.getElementById('backtest-trade-count').textContent = `${countLabel}: ${backtestState.total_trades} / ${backtestState.max_trades}`;
                    backtestBtn.disabled = true; backtestBtn.textContent = 'Running...';
                } else {
                    backtestWrapper.classList.add('is-hidden');
                    backtestBtn.disabled = false; backtestBtn.textContent = 'Backtest';
                }
                document.getElementById('history-list').innerHTML = data.trades.map(t => {
                    const realBadge = t.is_real ? '<span class="real-badge">REAL</span>' : '';
                    const pnlText = t.status === 'CLOSED' ? formatPercent(t.pl_percent - (2 * data.settings.fee_pct)) : 'OPEN';
                    const displayStatus = t.display_status || 'open';
                    return `<li class="history-item">
                                <div class="history-main">
                                    <span class="history-type ${t.type==='LONG'?'text-green':'text-red'}">${t.type} ${realBadge}</span>
                                    <span class="history-pair">${t.instrumentId}</span>
                                </div>
                                <div class="history-pnl status-${displayStatus}">${pnlText}</div>
                                <div class="history-details">Entry @ ${formatPrice(t.entryPrice)} • ${t.entryReason.split('\\n')[0]}</div>
                            </li>`
                }).join('');
                Object.entries(data.settings).forEach(([k, v]) => {
                    const i = document.getElementById(`s-${k}`);
                    if(i && document.activeElement !== i) { if (i.type === 'checkbox') { i.checked = v; } else { i.value = v; } }
                    if (k === 'watched_pairs') { document.getElementById('watchlist-list').innerHTML = Object.entries(v).map(([p,tf])=>`<li><span>${p} (${tf})</span><button class="btn-remove" data-pair="${p}">×</button></li>`).join(''); }
                });
            };
            const fetchData = async () => {
                let url = API_ENDPOINT_BASE;
                if (currentChartPair) url += `?active_chart=${currentChartPair}`;
                try {
                    const res = await fetch(url);
                    if (!res.ok) return;
                    const data = await res.json();
                    if(JSON.stringify(data) !== JSON.stringify(lastData)) updateUI(data);
                    lastData = data;
                } catch(e) { console.error("Update failed:", e); }
            };
            const iconExpand = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m4.5 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" /></svg>';
            const iconCollapse = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 9L3.75 3.75M3.75 3.75h4.5m-4.5 0v4.5m11.25 11.25L20.25 20.25M20.25 20.25v-4.5m0 4.5h-4.5M9 15l-5.25 5.25M3.75 20.25v-4.5m0 4.5h4.5m11.25-11.25L15 9m5.25-5.25v4.5m0-4.5h-4.5" /></svg>';
            const UIElementsToHide = ['.header', '#backtest-status-wrapper', '#pnl-stats', '#bingx-chart-wrapper', '#ai-global-analysis-wrapper', '#watchlist-title', '#watchlist', '#history-title', '#history-list'];
            document.querySelectorAll('.fullscreen-btn').forEach(button => {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    const targetWrapper = document.querySelector(button.dataset.target);
                    if (!targetWrapper) return;
                    const isAlreadyFullscreen = targetWrapper.classList.contains('chart-fullscreen');
                    document.querySelectorAll('.chart-fullscreen').forEach(el => el.classList.remove('chart-fullscreen'));
                    UIElementsToHide.forEach(selector => { document.querySelectorAll(selector).forEach(el => el.classList.remove('is-hidden')); });
                    document.querySelectorAll('.fullscreen-btn').forEach(btn => btn.innerHTML = iconExpand);
                    if (!isAlreadyFullscreen) {
                        targetWrapper.classList.add('chart-fullscreen');
                        UIElementsToHide.forEach(selector => { document.querySelectorAll(selector).forEach(el => { if (el !== targetWrapper) { el.classList.add('is-hidden'); } }); });
                        button.innerHTML = iconCollapse;
                    }
                    window.dispatchEvent(new Event('resize'));
                });
            });
            document.getElementById('watchlist').addEventListener('click', e => {
                const card = e.target.closest('.pair-card');
                if (card && card.dataset.pair && card.dataset.pair !== currentChartPair) {
                    currentChartPair = card.dataset.pair;
                    createChartWidgets(currentChartPair, lastData.settings.watched_pairs[currentChartPair]);
                    document.querySelectorAll('.pair-card').forEach(c => c.classList.remove('active-chart'));
                    card.classList.add('active-chart');
                    fetchData();
                }
            });
            document.body.addEventListener('submit', e => { if(e.target.matches('.trade-form')) { e.preventDefault(); const f = e.target; postRequest(f.dataset.url, JSON.parse(f.dataset.body.replace(/'/g, '"'))); }});
            document.getElementById('watchlist-list').addEventListener('click', e => { if (e.target.matches('.btn-remove')) postRequest('/api/watchlist/remove', {pair: e.target.dataset.pair}); });
            const modal=document.getElementById('settings-modal');
            document.getElementById('settings-btn').addEventListener('click',()=>modal.classList.add('visible'));
            document.getElementById('close-settings-btn').addEventListener('click',()=>modal.classList.remove('visible'));
            document.getElementById('ai-status-btn').addEventListener('click',()=>postRequest('/toggle-ai',{}));
            document.getElementById('trade-mode-btn').addEventListener('click', () => {
                const isCurrentlyReal = lastData.settings.is_real_trading;
                const message = isCurrentlyReal ? "Anda akan beralih ke mode DEMO." : "PERINGATAN! Anda akan beralih ke mode REAL. Bot akan menggunakan API key Anda untuk mengeksekusi order sungguhan di BingX. Lanjutkan?";
                if (confirm(message)) { postRequest('/toggle-trade-mode', {}); }
            });
            document.getElementById('backtest-btn').addEventListener('click', () => {
                const autoSettingsEnabled = document.getElementById('s-auto_settings_enabled').checked;
                const message = autoSettingsEnabled 
                    ? 'This will start a parameter optimization based on your existing trades. It can take a long time and will overwrite your settings with the best result found. Continue?'
                    : 'This will start a backtest from your oldest trade. Continue?';
                if (confirm(message)) { postRequest('/start-backtest', {}); }
            });
            document.getElementById('add-pair-btn').addEventListener('click',()=> { const p=document.getElementById('new-pair-input').value.toUpperCase();const tf=document.getElementById('new-tf-input').value; if(p)postRequest('/api/watchlist/add',{pair:p,tf:tf});});
            document.getElementById('settings-form').addEventListener('submit', e => { e.preventDefault(); postRequest('/api/settings', Object.fromEntries(new FormData(e.target).entries())).then(() => window.location.reload()); });
            fetchData();
            setInterval(fetchData, REFRESH_INTERVAL_MS);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_SKELETON_TRADINGVIEW, current_settings=current_settings)

@app.route('/api/data')
def get_api_data():
    with state_lock:
        trades_copy = list(trades)
        market_state_copy = dict(market_state)
        settings_copy = dict(current_settings)
    with backtest_lock:
        backtest_state_copy = dict(backtest_state)

    active_chart_pair = request.args.get('active_chart')
    market_data_view = {}
    global_ai_analysis = None
    fee_pct = settings_copy.get('fee_pct', 0.1)

    for pair_id, timeframe in settings_copy.get("watched_pairs", {}).items():
        pair_state = market_state_copy.get(pair_id, {})
        full_candle_data = pair_state.get("candle_data", [])
        current_price = full_candle_data[-1].get('close', 0.0) if full_candle_data else 0.0
        open_pos = next((t for t in trades_copy if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        pnl = 0.0
        if open_pos and current_price > 0: pnl = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - fee_pct
        trend = "N/A"
        if len(full_candle_data) > 100 + 15:
            relevant_trades_history = [t for t in trades_copy if t['instrumentId'] == pair_id]
            ai_instance = LocalAI(settings_copy, relevant_trades_history)
            analysis_result = ai_instance.get_market_analysis(full_candle_data)
            if analysis_result:
                trend = analysis_result.get('bias', 'N/A').title()
                if pair_id == active_chart_pair and not open_pos: global_ai_analysis = ai_instance.get_similarity_analysis_for_dashboard(analysis_result)
        market_data_view[pair_id] = {"price": current_price, "funding": pair_state.get("funding_rate", 0.0), "timeframe": timeframe, "open_position": open_pos, "pnl": pnl, "trend": trend}

    for trade in trades_copy:
        if trade.get('status') == 'CLOSED':
            if is_trade_considered_a_win(trade, settings_copy):
                trade['display_status'] = 'valid-win'
            else:
                net_pnl = trade.get('pl_percent', 0.0) - (2 * fee_pct)
                trade['display_status'] = 'lucky-win' if net_pnl > 0 else 'loss'
        else:
            trade['display_status'] = 'open'

    return jsonify({
        "is_ai_running": is_autopilot_running,
        "pnl_today": calculate_todays_pnl(trades_copy, settings_copy),
        "pnl_this_week": calculate_this_weeks_pnl(trades_copy, settings_copy),
        "pnl_last_week": calculate_last_weeks_pnl(trades_copy, settings_copy),
        "market_data": market_data_view,
        "trades": trades_copy,
        "settings": settings_copy,
        "global_ai_analysis": global_ai_analysis,
        "backtest_state": backtest_state_copy
    })

@app.route('/toggle-trade-mode', methods=['POST'])
def toggle_trade_mode():
    global current_settings
    with state_lock:
        is_real = current_settings.get('is_real_trading', False)
        current_settings['is_real_trading'] = not is_real
        save_settings()
        mode = "REAL" if not is_real else "DEMO"
        color = Fore.RED if not is_real else Fore.GREEN
        print_colored(f"Mode trading diubah ke {mode} dari Web UI.", color, Style.BRIGHT)
    return jsonify(success=True)

@app.route('/toggle-ai', methods=['POST'])
def toggle_ai():
    global is_autopilot_running
    is_autopilot_running = not is_autopilot_running
    print_colored(f"Autopilot {'diaktifkan' if is_autopilot_running else 'dimatikan'} dari Web UI.", Fore.YELLOW)
    return jsonify(success=True)

# MODIFIED: This endpoint now routes to the correct worker
@app.route('/start-backtest', methods=['POST'])
def start_backtest():
    with backtest_lock:
        if backtest_state["is_running"]:
            return jsonify(success=False, message="Backtest is already running.")

    if current_settings.get("auto_settings_enabled", False):
        auto_settings_thread = threading.Thread(target=auto_settings_worker, daemon=True)
        auto_settings_thread.start()
        return jsonify(success=True, message="Auto Settings optimization started.")
    else:
        backtest_thread = threading.Thread(target=backtest_worker, daemon=True)
        backtest_thread.start()
        return jsonify(success=True, message="Backtest started.")

@app.route('/trade/manual', methods=['POST'])
def trade_manual():
    data = request.form; pair = data.get('pair'); trade_type = data.get('type')
    if not pair or not trade_type: return jsonify(success=False, error="Data tidak lengkap"), 400
    pair_state = market_state.get(pair, {}); candle_data = pair_state.get("candle_data")
    current_price = candle_data[-1].get('close') if candle_data else None
    if not current_price: return jsonify(success=False, error="Harga pasar tidak tersedia"), 400
    is_real = current_settings.get("is_real_trading", False); quantity = 0.0
    if is_real:
        print_colored(f"Mencoba membuka posisi REAL manual {trade_type} {pair}...", Fore.YELLOW)
        try:
            risk_usdt = float(current_settings.get('risk_usdt_per_trade', 5.0))
            sl_pct = float(current_settings.get('stop_loss_pct', 0.25))
            if risk_usdt <= 0 or sl_pct <= 0: return jsonify(success=False, error="Risk/SL harus > 0 untuk Real Trade."), 400
            sl_size_in_usdt = current_price * (sl_pct / 100)
            if sl_size_in_usdt == 0: return jsonify(success=False, error="Kalkulasi SL menghasilkan nol."), 400
            quantity = risk_usdt / sl_size_in_usdt
            order_id, error = place_real_order(pair, trade_type, quantity, current_price, current_settings)
            if error: return jsonify(success=False, error=f"Gagal menempatkan order real: {error}"), 500
        except (ValueError, TypeError) as e: return jsonify(success=False, error=f"Pengaturan trading tidak valid: {e}"), 400
    entry_snapshot = {}
    if candle_data and len(candle_data) >= 100 + 15:
        with state_lock: relevant_trades_history = [t for t in trades if t['instrumentId'] == pair]
        ai_analyzer = LocalAI(current_settings, relevant_trades_history)
        analysis_result = ai_analyzer.get_market_analysis(candle_data)
        if analysis_result:
            analysis_result["funding_rate"] = pair_state.get("funding_rate", 0.0)
            entry_snapshot = analysis_result
    with state_lock:
        if any(t for t in trades if t['instrumentId'] == pair and t['status'] == 'OPEN'): return jsonify(success=False, error="Posisi untuk pair ini sudah terbuka."), 400
        new_trade = {"id": int(time.time()), "instrumentId": pair, "type": trade_type, "entryTimestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), "entryPrice": current_price, "entryReason": "Manual Entry", "status": 'OPEN', "exitPrice": None, "pl_percent": None, "entry_snapshot": entry_snapshot, "is_real": is_real, "quantity": quantity}
        trades.insert(0, new_trade)
        mode_str = "REAL" if is_real else "DEMO"; notif_title = f"🟢 Posisi {mode_str} {trade_type} Dibuka: {pair}"
        notif_content = f"Manual Entry @ {current_price:.4f}"
        if is_real: notif_content += f" | Qty: {quantity:.4f}"
        send_termux_notification(notif_title, notif_content); print_colored(notif_content, Fore.BLUE, Style.BRIGHT)
    save_trades()
    return jsonify(success=True)

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

# MODIFIED: This endpoint now handles the auto_settings_enabled checkbox
@app.route('/api/settings', methods=['POST'])
def update_settings():
    global current_settings
    with state_lock:
        current_settings['use_trailing_tp'] = 'use_trailing_tp' in request.form
        current_settings['auto_settings_enabled'] = 'auto_settings_enabled' in request.form

        for key, value in request.form.items():
            if key in ['use_trailing_tp', 'auto_settings_enabled']:
                continue
            
            if key in current_settings or key.startswith("bingx_") or key in ["leverage", "risk_usdt_per_trade", "min_win_to_risk_ratio"]:
                default_val = current_settings.get(key)
                if default_val is None and key == 'min_win_to_risk_ratio': default_val = 0.0
                
                target_type = type(default_val)
                try:
                    if target_type == float: current_settings[key] = float(value)
                    elif target_type == int: current_settings[key] = int(value)
                    else: current_settings[key] = value
                except (ValueError, TypeError): pass
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
        if pair in current_settings['watched_pairs']:
            del current_settings['watched_pairs'][pair]; save_settings()
            print_colored(f"{pair} dihapus dari watchlist.", Fore.YELLOW)
    return jsonify(success=True)

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