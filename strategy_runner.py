import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math

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
MAX_TRADES_IN_HISTORY = 80

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
    print_colored("PERBAIKAN: Perhitungan PnL & Logika Belajar AI dari Funding.", Fore.GREEN)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk masuk ke Live Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    # ... (Help text remains the same)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade (terbatas 80 terakhir)", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, caution, winrate, cc_key, fr_max)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()


# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 0.20, "fee_pct": 0.1, "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30, "trailing_tp_gap_pct": 0.05,
        "caution_level": 0.5, "target_winrate_pct": 85.0,
        "cryptocompare_api_key": "YOUR_CRYPTOCOMPARE_API_KEY",
        "max_allowed_funding_rate_pct": 0.075, "watched_pairs": {}
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                # Pastikan semua kunci default ada
                for key, value in default_settings.items():
                    if key not in loaded_settings:
                        loaded_settings[key] = value
                current_settings = loaded_settings
        except (json.JSONDecodeError, IOError):
            current_settings = default_settings
    else:
        current_settings = default_settings
    save_settings()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)
    except IOError as e:
        print_colored(f"Error saving settings: {e}", Fore.RED)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
        except (json.JSONDecodeError, IOError):
            autopilot_trades = []
    else:
        autopilot_trades = []
    for trade in autopilot_trades:
        if 'current_tp_checkpoint_level' not in trade:
            trade['current_tp_checkpoint_level'] = 0.0

def save_trades():
    global autopilot_trades
    autopilot_trades.sort(key=lambda x: x['entryTimestamp'])
    if len(autopilot_trades) > MAX_TRADES_IN_HISTORY:
        num_to_trim = len(autopilot_trades) - MAX_TRADES_IN_HISTORY
        autopilot_trades = autopilot_trades[num_to_trim:]
        print_colored(f"Riwayat trade dibatasi. {num_to_trim} trade tertua telah dihapus.", Fore.YELLOW)
    try:
        with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)
    except IOError as e:
        print_colored(f"Error saving trades: {e}", Fore.RED)

def display_history():
    # ... (Fungsi ini sudah benar, tidak perlu diubah)
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
        if trade['status'] == 'CLOSED' and exit_time_str:
            exit_time = datetime.fromisoformat(exit_time_str).strftime('%Y-%m-%d %H:%M')
            pl_percent_gross = trade.get('pl_percent', 0.0)
            pl_percent_net = pl_percent_gross - fee_pct
            is_profit = pl_percent_net > 0
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L (Net): {pl_percent_net:.2f}%", pl_color, Style.BRIGHT)

# --- FUNGSI API (Tidak ada perubahan) ---
def fetch_funding_rate(instId):
    # ... (Sama seperti sebelumnya)
    bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/tickers?category=linear&symbol={bybit_symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}) and data['result']['list']:
            funding_rate_str = data['result']['list'][0].get('fundingRate', '0')
            return float(funding_rate_str) * 100
        else:
            return None
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None

def fetch_recent_candles(instId, timeframe, limit=300):
    # ... (Sama seperti sebelumnya)
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=linear&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: return None
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except requests.exceptions.RequestException: return None
    except Exception: return None

def fetch_historical_candles_backward_from_ts(instId, timeframe, to_ts_seconds, limit_per_request):
    # ... (Sama seperti sebelumnya)
    timeframe_map = {'1m': 'histominute', '1H': 'histohour', '1D': 'histoday'}
    cc_endpoint = timeframe_map.get(timeframe)
    if not cc_endpoint: return [], 0
    try: fsym, tsym = instId.split('-')
    except ValueError: return [], 0
    api_key = current_settings.get("cryptocompare_api_key")
    if not api_key or api_key == "YOUR_CRYPTOCOMPARE_API_KEY": return [], 0
    url = f"{CRYPTOCOMPARE_API_URL}{cc_endpoint}?fsym={fsym}&tsym={tsym}&limit={limit_per_request}&toTs={to_ts_seconds}&api_key={api_key}"
    try:
        response = requests.get(url, timeout=20); response.raise_for_status(); data = response.json()
        if data.get("Response") == "Success" and 'Data' in data.get('Data', {}):
            candles_batch_raw = data['Data']['Data']
            if not candles_batch_raw: return [], 0
            formatted_batch = [{"time": c['time'] * 1000, "open": c['open'], "high": c['high'], "low": c['low'], "close": c['close'], "volume": c['volumefrom']} for c in candles_batch_raw]
            return formatted_batch, candles_batch_raw[0]['time']
        else: return [], 0
    except (requests.exceptions.RequestException, Exception): return [], 0

# --- FUNGSI KALKULASI PNL (Tidak ada perubahan, logika UTC sudah benar) ---
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
    today_utc = datetime.utcnow().date()
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() == today_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

def calculate_this_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date()
    start_of_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_week_utc = start_of_week_utc + timedelta(days=6)
    total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if start_of_week_utc <= datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() <= end_of_week_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - fee_pct)
            except ValueError: continue
    return total_pnl

def calculate_last_weeks_pnl(all_trades):
    today_utc = datetime.utcnow().date()
    start_of_current_week_utc = today_utc - timedelta(days=today_utc.weekday())
    end_of_last_week_utc = start_of_current_week_utc - timedelta(days=1)
    start_of_last_week_utc = end_of_last_week_utc - timedelta(days=6)
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
        # ... (Sama seperti sebelumnya)
        if len(data) < period: return []
        closes = [d['close'] for d in data]
        ema_values = [sum(closes[:period]) / period]
        multiplier = 2 / (period + 1)
        for i in range(period, len(closes)):
            ema = (closes[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        return ema_values

    def analyze_candle_solidity(self, candle):
        # ... (Sama seperti sebelumnya)
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        return body / full_range if full_range > 0 else 1.0

    def get_market_analysis(self, candle_data):
        # ... (Sama seperti sebelumnya)
        if len(candle_data) < 100 + 3: return None
        ema9 = self.calculate_ema(candle_data, 9)
        ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100)
        if len(ema9) < 2 or not ema50 or not ema100: return None
        analysis = {
            "ema9_current": ema9[-1], "ema9_prev": ema9[-2],
            "ema50": ema50[-1], "ema100": ema100[-1],
            "current_candle_close": candle_data[-1]['close'],
            "prev_candle_close": candle_data[-2]['close'],
            "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING",
        }
        pre_entry_candles = candle_data[-4:-1]
        analysis["pre_entry_candle_solidity"] = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        analysis["pre_entry_candle_direction"] = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        return analysis

    def check_for_repeated_mistake(self, current_analysis, funding_rate):
        # DIPERBAIKI: Menambahkan perbandingan funding rate
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - self.settings.get('fee_pct', 0.1)) < 0]
        if not losing_trades: return False

        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or past_snapshot.get('bias') != current_analysis['bias']:
                continue

            # Periksa kemiripan funding rate (jika ada di snapshot lama)
            past_funding_rate = past_snapshot.get("funding_rate")
            if past_funding_rate is not None:
                # Anggap mirip jika selisihnya kurang dari 0.02%
                if abs(funding_rate - past_funding_rate) < 0.02:
                    # Di sini, tambahkan perbandingan teknikal yang lebih detail
                    # Jika teknikal juga mirip, maka itu adalah pengulangan kesalahan
                    print_colored(f"AI Menghindari Trade: Kondisi mirip dengan loss sebelumnya (ID: {loss['id']})", Fore.MAGENTA)
                    return True
        return False

    def get_decision(self, candle_data, open_position, instrument_id, funding_rate=0.0):
        # ... (Logika tidak berubah, tapi pemanggilan check_for_repeated_mistake diperbarui)
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data tidak cukup."}
        if open_position: return {"action": "HOLD", "reason": "Memantau posisi."}

        max_funding_rate = self.settings.get("max_allowed_funding_rate_pct", 0.075)
        potential_trade_type = None

        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']:
            potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']:
            potential_trade_type = 'SHORT'

        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate:
                return {"action": "HOLD", "reason": f"HOLD LONG: Funding {funding_rate:.4f}% > {max_funding_rate}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate:
                return {"action": "HOLD", "reason": f"HOLD SHORT: Funding {funding_rate:.4f}% < {-max_funding_rate}%"}

            if self.check_for_repeated_mistake(analysis, funding_rate): # Passing funding rate
                return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan."}

            reason = "BULLISH: Retrace & close di atas EMA9." if potential_trade_type == 'LONG' else "BEARISH: Retrace & close di bawah EMA9."
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": reason, "snapshot": analysis}

        return {"action": "HOLD", "reason": f"Menunggu setup. Bias: {analysis['bias']}."}

# --- LOGIKA TRADING UTAMA ---
async def analyze_and_close_trade(trade, exit_price, reason, is_backtest=False, exit_timestamp_ms=None):
    # DIPERBAIKI: Menerima timestamp exit untuk backtest
    pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    
    # Gunakan timestamp yang diberikan jika ada (untuk backtest), jika tidak, gunakan waktu sekarang (untuk live)
    exit_dt = datetime.fromtimestamp(exit_timestamp_ms / 1000) if exit_timestamp_ms else datetime.utcnow()
    
    trade.update({
        'status': 'CLOSED', 'exitPrice': exit_price,
        'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross
    })
    is_profit = (pnl_gross - current_settings.get('fee_pct', 0.1)) > 0
    if is_profit and 'entry_snapshot' in trade:
        try: del trade['entry_snapshot']
        except KeyError: pass

    if not is_backtest:
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
        decision = ai.get_decision(
            pair_state["candle_data"], open_pos, instrument_id, funding_rate
        )

        if decision.get('action') in ["BUY", "SELL"] and not open_pos:
            snapshot = decision.get("snapshot", {})
            # BARU: Tambahkan funding rate ke snapshot
            snapshot["funding_rate"] = funding_rate
            
            new_trade = {
                "id": int(time.time()), "instrumentId": instrument_id,
                "type": "LONG" if decision['action'] == "BUY" else "SHORT",
                "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": pair_state["candle_data"][-1]['close'],
                "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot,
                "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
            }
            autopilot_trades.append(new_trade)
            save_trades()
            notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {instrument_id}"
            notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | {new_trade['entryReason']}"
            send_termux_notification(notif_title, notif_content)
    except Exception as e:
        print_colored(f"Error dalam autopilot analysis: {e}", Fore.RED)
    finally:
        is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    # ... (Tidak ada perubahan)
    while not stop_event.is_set():
        if is_autopilot_running:
            for pair_id in list(current_settings.get("watched_pairs", {})):
                asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else:
            time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data, is_backtest=False):
    if not trade_obj: return
    # ... (Logika run-up dan drawdown)
    if trade_obj['type'] == 'LONG':
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG')
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'LONG')
        if pnl_at_high > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_high
        if pnl_at_low < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_low
    elif trade_obj['type'] == 'SHORT':
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT')
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'SHORT')
        if pnl_at_low > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_low
        if pnl_at_high < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_high

    # ... (Logika SL)
    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or \
       (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
        await analyze_and_close_trade(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}%", is_backtest, current_candle_data['time'])
        return

    # ... (Logika Trailing TP)
    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30); gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    if trade_obj.get("current_tp_checkpoint_level", 0.0) > 0.0:
        ts_price = trade_obj.get('trailing_stop_price')
        if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or \
                                      (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
            await analyze_and_close_trade(trade_obj, ts_price, f"Trailing TP", is_backtest, current_candle_data['time'])
            return
    
    # ... (Update Trailing TP)
    pnl_now = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high' if trade_obj['type'] == 'LONG' else 'low'], trade_obj['type'])
    if pnl_now >= activation_pct:
        current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
        if current_cp == 0.0: current_cp = activation_pct
        
        steps_passed = math.floor((pnl_now - current_cp) / gap_pct)
        if steps_passed >= 0:
            new_cp = current_cp + (steps_passed * gap_pct)
            trade_obj['current_tp_checkpoint_level'] = new_cp
            new_ts_level = new_cp - gap_pct
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + new_ts_level / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 - new_ts_level / 100)

    if not is_backtest:
        save_trades()

def data_refresh_worker():
    # ... (Tidak ada perubahan)
    while not stop_event.is_set():
        for pair_id, timeframe in list(current_settings.get("watched_pairs", {}).items()):
            if pair_id not in market_state: market_state[pair_id] = {}
            candle_data = fetch_recent_candles(pair_id, timeframe)
            funding_rate = fetch_funding_rate(pair_id)
            market_state[pair_id]['funding_rate'] = funding_rate if funding_rate is not None else market_state[pair_id].get('funding_rate', 0.0)
            if candle_data:
                market_state[pair_id]["candle_data"] = candle_data
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos:
                    asyncio.run(check_realtime_position_management(open_pos, candle_data[-1]))
            time.sleep(0.5)
        time.sleep(REFRESH_INTERVAL_SECONDS)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    # ... (Tidak ada perubahan)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print_colored(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()

def run_pair_backtest(pair_id, timeframe):
    global autopilot_trades
    print_colored(f"\n🚀 Memulai Backtest untuk {pair_id} ({timeframe})...", Fore.CYAN, Style.BRIGHT)
    autopilot_trades[:] = [t for t in autopilot_trades if t['instrumentId'] != pair_id]
    save_trades() # Simpan state yang bersih
    cumulative_historical_candles = []
    current_to_ts_for_fetch = int(datetime.utcnow().timestamp())
    target_winrate = current_settings.get("target_winrate_pct", 85.0)

    for fetch_iteration in range(1, 101):
        fetched_chunk, earliest_ts = fetch_historical_candles_backward_from_ts(
            pair_id, timeframe, current_to_ts_for_fetch, BACKTEST_FETCH_CHUNK_LIMIT
        )
        if not fetched_chunk: break
        cumulative_historical_candles = fetched_chunk + cumulative_historical_candles
        cumulative_historical_candles.sort(key=lambda x: x['time'])
        current_to_ts_for_fetch = earliest_ts - 1
        
        temp_open_trades = []
        backtested_trades = []
        total_candles = len(cumulative_historical_candles)
        
        for i in range(total_candles):
            data_slice = cumulative_historical_candles[:i+1]
            current_candle = cumulative_historical_candles[i]
            if len(data_slice) < 100 + 3: continue

            for trade in list(temp_open_trades):
                asyncio.run(check_realtime_position_management(trade, current_candle, is_backtest=True))
                if trade['status'] == 'CLOSED':
                    backtested_trades.append(trade)
                    temp_open_trades.remove(trade)
            
            if not temp_open_trades:
                ai_brain = LocalAI(current_settings, backtested_trades)
                decision = ai_brain.get_decision(data_slice, None, pair_id, 0.0) # Funding rate diset 0 untuk backtest
                if decision.get('action') in ["BUY", "SELL"]:
                    snapshot = decision.get("snapshot", {})
                    snapshot["funding_rate"] = 0.0 # Simpan funding rate 0.0 untuk backtest
                    new_trade = {
                        "id": int(current_candle['time'] / 1000), "instrumentId": pair_id,
                        "type": "LONG" if decision['action'] == "BUY" else "SHORT",
                        "entryTimestamp": datetime.fromtimestamp(current_candle['time'] / 1000).isoformat() + 'Z',
                        "entryPrice": current_candle['close'], "entryReason": decision.get("reason"),
                        "status": 'OPEN', "entry_snapshot": snapshot,
                        "run_up_percent": 0.0, "max_drawdown_percent": 0.0,
                        "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
                    }
                    temp_open_trades.append(new_trade)
            print_progress_bar(i + 1, total_candles, prefix=f'  {pair_id} Analisis', suffix='Lengkap')

        for trade in temp_open_trades:
            # DIPERBAIKI: Gunakan timestamp candle terakhir untuk exit
            asyncio.run(analyze_and_close_trade(trade, cumulative_historical_candles[-1]['close'], "Backtest End", True, cumulative_historical_candles[-1]['time']))
            backtested_trades.append(trade)
        
        autopilot_trades.extend(backtested_trades)
        save_trades() # Simpan hasil backtest setelah setiap chunk
        
        winrate = calculate_winrate(backtested_trades, current_settings.get('fee_pct', 0.1))
        if len(backtested_trades) > 0:
            print_colored(f"\n  [DEBUG] Stats: {len(backtested_trades)} trades, Winrate: {winrate:.2f}% (Target: {target_winrate:.2f}%)", Fore.CYAN)
        if winrate >= target_winrate and len(backtested_trades) >= 50:
            print_colored(f"✅ Target Winrate tercapai.", Fore.GREEN); break
    print_colored(f"✅ Backtest untuk {pair_id} selesai.", Fore.GREEN)

def check_and_run_backtests():
    # ... (Tidak ada perubahan)
    watched_pairs = current_settings.get("watched_pairs", {})
    pairs_to_backtest = [ (p, t) for p, t in watched_pairs.items() if not any(tr for tr in autopilot_trades if tr['instrumentId'] == p)]
    if pairs_to_backtest:
        print_colored(f"\nMemerlukan Backtest untuk pembelajaran AI:", Fore.CYAN, Style.BRIGHT)
        for pair_id, timeframe in pairs_to_backtest:
            print_colored(f"- {pair_id} ({timeframe})", Fore.YELLOW)
        for pair_id, timeframe in pairs_to_backtest:
            run_pair_backtest(pair_id, timeframe)
        print_colored("\nBacktest Selesai.", Fore.GREEN); load_trades()
    else:
        print_colored(f"\nTidak ada Backtest yang diperlukan.", Fore.GREEN)

def handle_settings_command(parts):
    # ... (Tidak ada perubahan)
    setting_map = {
        'sl': ('stop_loss_pct', '%'), 'fee': ('fee_pct', '%'), 'delay': ('analysis_interval_sec', 's'),
        'tp_act': ('trailing_tp_activation_pct', '%'), 'tp_gap': ('trailing_tp_gap_pct', '%'),
        'caution': ('caution_level', ''), 'winrate': ('target_winrate_pct', '%'),
        'cc_key': ('cryptocompare_api_key', ''), 'fr_max': ('max_allowed_funding_rate_pct', '%')
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full, unit) in setting_map.items():
            val = current_settings.get(full, 'N/A')
            if full == 'cryptocompare_api_key' and val not in ["YOUR_CRYPTOCOMPARE_API_KEY", "N/A"]:
                val = val[:4] + '...' + val[-4:]
            print_colored(f"{key.capitalize():<10} ({key:<7}) : {val}{unit}", Fore.WHITE)
        return
    if len(parts) == 3 and parts[0] == '!set':
        key, val_str = parts[1].lower(), parts[2]
        if key not in setting_map: print_colored(f"Kunci '{key}' tidak dikenal.", Fore.RED); return
        try:
            full, unit = setting_map[key]
            current_settings[full] = float(val_str) if key != 'cc_key' else val_str
            save_settings()
            print_colored(f"Pengaturan '{full}' diubah menjadi {current_settings[full]}{unit}.", Fore.GREEN)
        except ValueError: print_colored(f"Nilai '{val_str}' tidak valid untuk '{key}'.", Fore.RED)

def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="")
            print_colored("--- VULCAN'S EDITION LIVE DASHBOARD ---", Fore.CYAN, Style.BRIGHT)
            
            # DIPERBAIKI: Semua fungsi PnL dipanggil lagi
            todays_pnl = calculate_todays_pnl(autopilot_trades)
            this_weeks_pnl = calculate_this_weeks_pnl(autopilot_trades)
            last_weeks_pnl = calculate_last_weeks_pnl(autopilot_trades)
            
            pnl_color_today = Fore.GREEN if todays_pnl > 0 else Fore.RED if todays_pnl < 0 else Fore.WHITE
            pnl_color_week = Fore.GREEN if this_weeks_pnl > 0 else Fore.RED if this_weeks_pnl < 0 else Fore.WHITE
            pnl_color_last_week = Fore.GREEN if last_weeks_pnl > 0 else Fore.RED if last_weeks_pnl < 0 else Fore.WHITE

            print_colored(f"Today's P/L: ", end=""); print_colored(f"{todays_pnl:.2f}%", pnl_color_today, Style.BRIGHT, end="")
            print_colored(f" | This Week: ", end=""); print_colored(f"{this_weeks_pnl:.2f}%", pnl_color_week, Style.BRIGHT, end="")
            print_colored(f" | Last Week: ", end=""); print_colored(f"{last_weeks_pnl:.2f}%", pnl_color_last_week, Style.BRIGHT)

            print_colored("="*80, Fore.CYAN)
            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Tekan Ctrl+C dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                # ... (Sisa dashboard tidak berubah)
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
                    print_colored(f"  Status: Searching... | Funding Rate: ", end=""); print_colored(f"{funding_rate:.4f}%", funding_color)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def main():
    global is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    check_and_run_backtests()
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
                    print_colored("Watchlist kosong. Gunakan '!watch <PAIR>'.", Fore.RED)
                    continue
                is_autopilot_running = True
                print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN)
                run_dashboard_mode()
                is_autopilot_running = False
                print_colored("\n🛑 Live Dashboard ditutup.", Fore.RED)
            elif cmd == '!watch':
                if len(parts) >= 2:
                    pair_id = parts[1].upper()
                    tf = parts[2] if len(parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"{pair_id} ({tf}) ditambahkan. Jalankan ulang bot untuk memulai backtest.", Fore.GREEN)
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
