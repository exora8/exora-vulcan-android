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

# --- PERUBAHAN: Integrasi PyBit ---
try:
    from pybit.unified_trading import HTTP
except ImportError:
    print(f"{Fore.RED}Peringatan: Pustaka 'pybit' tidak ditemukan. Fitur integrasi Bybit tidak akan berfungsi.")
    print(f"{Fore.YELLOW}Silakan install dengan: pip install pybit")
    HTTP = None

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
bybit_session = None # PERUBAHAN: Menambahkan variabel sesi Bybit

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
    print_colored("                🔥 BYBIT INTEGRATED 🔥              ", Fore.YELLOW, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("PERBAIKAN: Logika Kolaborasi AI & Bot, 'Loss Memory' Cerdas.", Fore.GREEN)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk masuk ke Live Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!leverage <PAIR> <X>  - Atur leverage untuk pair di Bybit (e.g., !leverage BTC-USDT 10)", Fore.GREEN)
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
        "bybit_api_key": "YOUR_BYBIT_API_KEY", # PERUBAHAN: Menambahkan API key
        "bybit_api_secret": "YOUR_BYBIT_API_SECRET", # PERUBAHAN: Menambahkan API secret
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
                    if key not in loaded_settings: loaded_settings[key] = value
                current_settings = loaded_settings
        except (json.JSONDecodeError, IOError):
            current_settings = default_settings
    else:
        current_settings = default_settings

    # PERUBAHAN: Memastikan struktur watched_pairs baru
    for pair, config in list(current_settings.get("watched_pairs", {}).items()):
        if isinstance(config, str): # Konversi format lama
            current_settings["watched_pairs"][pair] = {"timeframe": config, "leverage": 10}
    
    save_settings()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)
    except IOError as e: print_colored(f"Error saving settings: {e}", Fore.RED)

# --- PERUBAHAN: Fungsi untuk inisialisasi sesi Bybit ---
def initialize_bybit_client():
    global bybit_session
    if not HTTP:
        print_colored("Sesi Bybit tidak dapat diinisialisasi karena pustaka 'pybit' tidak ada.", Fore.RED)
        return False
        
    api_key = current_settings.get("bybit_api_key")
    api_secret = current_settings.get("bybit_api_secret")

    if not api_key or api_key == "YOUR_BYBIT_API_KEY" or not api_secret or api_secret == "YOUR_BYBIT_API_SECRET":
        print_colored("Kunci API Bybit belum diatur di 'settings.json'. Fitur akun tidak akan berfungsi.", Fore.YELLOW)
        return False

    try:
        bybit_session = HTTP(api_key=api_key, api_secret=api_secret)
        print_colored("✅ Sesi Bybit berhasil diinisialisasi.", Fore.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Gagal menginisialisasi sesi Bybit: {e}", Fore.RED)
        return False

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
        except (json.JSONDecodeError, IOError): autopilot_trades = []
    else: autopilot_trades = []
    # Memastikan trade lama memiliki nilai leverage default
    for trade in autopilot_trades:
        if 'leverage' not in trade: trade['leverage'] = 10
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
        leverage = trade.get('leverage', 10) # PERUBAHAN: Mengambil leverage
        
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Leverage: {leverage}x | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        
        if trade.get('entryReason'):
            reason_lines = trade['entryReason'].split('\n')
            for line in reason_lines:
                print_colored(f"    {line}", Fore.WHITE)

        if trade['status'] == 'CLOSED' and exit_time_str:
            exit_time = datetime.fromisoformat(exit_time_str).strftime('%Y-%m-%d %H:%M')
            pl_percent_gross = trade.get('pl_percent', 0.0)
            pl_percent_net = pl_percent_gross - (fee_pct * leverage) # PERUBAHAN: Fee dikali leverage
            is_profit = pl_percent_net > 0
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  RoE (Net): {pl_percent_net:.2f}%", pl_color, Style.BRIGHT)

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

# --- PERUBAHAN: Fungsi untuk mengambil saldo Bybit ---
def fetch_bybit_balance():
    if not bybit_session:
        return "N/A (API Key Error)"
    try:
        balance_data = bybit_session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if balance_data and balance_data.get('retCode') == 0:
            unified_balance_info = balance_data['result']['list'][0]
            wallet_balance = unified_balance_info.get('walletBalance', '0')
            return f"{float(wallet_balance):.2f} USDT"
        else:
            error_msg = balance_data.get('retMsg', 'Unknown Error')
            return f"N/A ({error_msg})"
    except Exception as e:
        return f"N/A (Error: {e})"

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
# PERUBAHAN: Kalkulasi PNL sekarang menyertakan leverage
def calculate_pnl(entry_price, current_price, trade_type, leverage=1.0):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': 
        return ((current_price - entry_price) / entry_price) * 100 * leverage
    elif trade_type == 'SHORT': 
        return ((entry_price - current_price) / entry_price) * 100 * leverage
    return 0

def calculate_winrate(trades_list, fee_pct):
    closed_trades = [t for t in trades_list if t.get('status') == 'CLOSED']
    if not closed_trades: return 0.0
    profitable_trades = sum(1 for t in closed_trades if (t.get('pl_percent', 0) - (fee_pct * t.get('leverage', 1))) > 0)
    return (profitable_trades / len(closed_trades)) * 100

def calculate_todays_pnl(all_trades):
    today_utc = datetime.utcnow().date(); total_pnl = 0.0; fee_pct = current_settings.get('fee_pct', 0.1)
    for trade in all_trades:
        if trade.get('status') == 'CLOSED' and 'exitTimestamp' in trade:
            try:
                if datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).date() == today_utc:
                    total_pnl += (trade.get('pl_percent', 0.0) - (fee_pct * trade.get('leverage', 1)))
            except ValueError: continue
    return total_pnl

# --- OTAK LOCAL AI (TIDAK DIUBAH, SESUAI PERMINTAAN) ---
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
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - (self.settings.get('fee_pct', 0.1) * t.get('leverage', 1))) < 0]
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
        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']: potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']: potential_trade_type = 'SHORT'
        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal LONG diabaikan. Funding rate terlalu tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate: return {"action": "HOLD", "reason": f"Sinyal SHORT diabaikan. Funding rate terlalu negatif: {funding_rate:.4f}%"}
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
async def analyze_and_close_trade(trade, exit_price, reason, is_backtest=False, exit_timestamp_ms=None):
    leverage = trade.get('leverage', 1.0) # PERUBAHAN: Menggunakan leverage untuk kalkulasi PNL
    pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'), leverage)
    exit_dt = datetime.fromtimestamp(exit_timestamp_ms / 1000) if exit_timestamp_ms else datetime.utcnow()
    
    trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })
    
    is_profit = (pnl_gross - (current_settings.get('fee_pct', 0.1) * leverage)) > 0
    if is_profit and 'entry_snapshot' in trade:
        try: del trade['entry_snapshot']
        except KeyError: pass

    if not is_backtest:
        save_trades()
        pnl_net = pnl_gross - (current_settings.get('fee_pct', 0.1) * leverage)
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"RoE (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | Trigger: {reason}"
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
            
            # PERUBAHAN: Mengambil leverage yang di-set untuk pair ini
            leverage = current_settings.get("watched_pairs", {}).get(instrument_id, {}).get("leverage", 10)

            new_trade = {
                "id": int(time.time()), "instrumentId": instrument_id,
                "type": "LONG" if decision['action'] == "BUY" else "SHORT",
                "leverage": leverage, # PERUBAHAN: Menyimpan leverage dalam trade
                "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": pair_state["candle_data"][-1]['close'],
                "entryReason": decision.get("reason"), "status": 'OPEN', "entry_snapshot": snapshot,
                "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
            }
            autopilot_trades.append(new_trade)
            save_trades()
            
            ai_reason_short = decision.get("reason").split('\n')[0]
            notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {instrument_id} ({leverage}x)"
            notif_content = f"Entry @ {new_trade['entryPrice']:.4f} | {ai_reason_short}"
            send_termux_notification(notif_title, notif_content)
    except Exception as e:
        print_colored(f"Error dalam autopilot analysis: {e}", Fore.RED)
    finally:
        is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            # PERUBAHAN: Menggunakan .keys() karena value adalah dict
            for pair_id in list(current_settings.get("watched_pairs", {}).keys()):
                asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else:
            time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data, is_backtest=False):
    if not trade_obj: return
    leverage = trade_obj.get('leverage', 1.0) # PERUBAHAN: leverage
    
    # Logika run-up dan drawdown
    if trade_obj['type'] == 'LONG':
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'LONG', leverage)
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'LONG', leverage)
        if pnl_at_high > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_high
        if pnl_at_low < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_low
    elif trade_obj['type'] == 'SHORT':
        pnl_at_low = calculate_pnl(trade_obj['entryPrice'], current_candle_data['low'], 'SHORT', leverage)
        pnl_at_high = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high'], 'SHORT', leverage)
        if pnl_at_low > trade_obj.get('run_up_percent', 0.0): trade_obj['run_up_percent'] = pnl_at_low
        if pnl_at_high < trade_obj.get('max_drawdown_percent', 0.0): trade_obj['max_drawdown_percent'] = pnl_at_high

    # Logika SL
    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or \
       (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
        await analyze_and_close_trade(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct) * leverage:.2f}% RoE", is_backtest, current_candle_data['time'])
        return

    # Logika Trailing TP
    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30) * leverage
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05) * leverage
    
    # Hitung pnl saat ini dengan leverage
    pnl_now = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high' if trade_obj['type'] == 'LONG' else 'low'], trade_obj['type'], leverage)
    
    if trade_obj.get("current_tp_checkpoint_level", 0.0) > 0.0:
        ts_price = trade_obj.get('trailing_stop_price')
        if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or \
                                      (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
            await analyze_and_close_trade(trade_obj, ts_price, f"Trailing TP", is_backtest, current_candle_data['time'])
            return
    
    if pnl_now >= activation_pct:
        current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
        if current_cp == 0.0: current_cp = activation_pct
        
        steps_passed = math.floor((pnl_now - current_cp) / gap_pct)
        if steps_passed >= 0:
            new_cp = current_cp + (steps_passed * gap_pct)
            trade_obj['current_tp_checkpoint_level'] = new_cp
            new_ts_level_unleveraged = (new_cp - gap_pct) / leverage
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + new_ts_level_unleveraged / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 - new_ts_level_unleveraged / 100)

    if not is_backtest: save_trades()

def data_refresh_worker():
    while not stop_event.is_set():
        # PERUBAHAN: Menyesuaikan loop dengan struktur setting baru
        for pair_id, pair_config in list(current_settings.get("watched_pairs", {}).items()):
            timeframe = pair_config["timeframe"]
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

# --- FUNGSI BARU: Menangani perintah leverage ---
def handle_leverage_command(parts):
    if not bybit_session:
        print_colored("Tidak bisa mengatur leverage, sesi Bybit tidak aktif (cek API key).", Fore.RED)
        return
    if len(parts) != 3:
        print_colored("Format salah. Gunakan: !leverage <PAIR> <NILAI>", Fore.RED)
        print_colored("Contoh: !leverage BTC-USDT 10", Fore.YELLOW)
        return
    
    pair_id = parts[1].upper()
    bybit_symbol = pair_id.replace('-', '')
    
    if pair_id not in current_settings.get("watched_pairs", {}):
        print_colored(f"Pair {pair_id} tidak ada di watchlist. Tambahkan dulu dengan !watch.", Fore.RED)
        return
        
    try:
        leverage_val = float(parts[2])
        if not (0 < leverage_val <= 100): # Bybit biasanya maks 100x
            raise ValueError("Leverage harus antara 1 dan 100.")
            
        print_colored(f"Mengatur leverage untuk {pair_id} menjadi {leverage_val}x di Bybit...", Fore.YELLOW)
        
        # Panggil API untuk set leverage
        response = bybit_session.set_leverage(
            category="linear",
            symbol=bybit_symbol,
            buyLeverage=str(leverage_val),
            sellLeverage=str(leverage_val)
        )
        
        if response and response.get('retCode') == 0:
            current_settings["watched_pairs"][pair_id]["leverage"] = leverage_val
            save_settings()
            print_colored(f"✅ Leverage untuk {pair_id} berhasil diatur menjadi {leverage_val}x.", Fore.GREEN)
        else:
            error_msg = response.get('retMsg', 'Unknown Bybit Error')
            print_colored(f"❌ Gagal mengatur leverage di Bybit: {error_msg}", Fore.RED)

    except ValueError as e:
        print_colored(f"Nilai leverage tidak valid: {e}", Fore.RED)
    except Exception as e:
        print_colored(f"Terjadi error saat komunikasi dengan Bybit: {e}", Fore.RED)


def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="") # Clear screen
            print_colored("--- VULCAN'S EDITION LIVE DASHBOARD (BYBIT CONNECTED) ---", Fore.CYAN, Style.BRIGHT)
            
            # PERUBAHAN: Menampilkan saldo
            bybit_balance = fetch_bybit_balance()
            print_colored(f"Bybit Balance: ", end=""); print_colored(f"{bybit_balance}", Fore.YELLOW, Style.BRIGHT)

            todays_pnl = calculate_todays_pnl(autopilot_trades)
            pnl_color_today = Fore.GREEN if todays_pnl > 0 else Fore.RED if todays_pnl < 0 else Fore.WHITE
            print_colored(f"Today's Realized RoE: ", end=""); print_colored(f"{todays_pnl:.2f}%", pnl_color_today, Style.BRIGHT)

            print_colored("="*80, Fore.CYAN)
            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Tekan Ctrl+C dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            
            # PERUBAHAN: Menyesuaikan loop dengan struktur setting baru
            for pair_id, pair_config in current_settings.get("watched_pairs", {}).items():
                timeframe = pair_config["timeframe"]
                leverage = pair_config.get("leverage", 10)
                print_colored(f"\n⦿ {pair_id} ({timeframe}) | Leverage: {leverage}x", Fore.WHITE, Style.BRIGHT)
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                pair_state = market_state.get(pair_id, {})
                if open_pos:
                    price = pair_state.get('candle_data', [{}])[-1].get('close', open_pos['entryPrice'])
                    # PERUBAHAN: Gunakan leverage dari trade object untuk kalkulasi PNL
                    pos_leverage = open_pos.get('leverage', leverage)
                    pnl_unleveraged = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'), 1)
                    fee_impact = current_settings.get('fee_pct', 0.1) * pos_leverage
                    roe_net = (pnl_unleveraged * pos_leverage) - fee_impact
                    
                    pnl_color = Fore.GREEN if roe_net > 0 else Fore.RED
                    print_colored(f"  Status: OPEN {open_pos.get('type')} | Entry: {open_pos['entryPrice']:.4f} | RoE(Net): ", end="")
                    print_colored(f"{roe_net:.2f}%", pnl_color, Style.BRIGHT)
                    if open_pos.get("current_tp_checkpoint_level", 0.0) > 0:
                        cp_level_roe = open_pos["current_tp_checkpoint_level"]
                        ts_price = open_pos.get("trailing_stop_price", 0)
                        print_colored(f"  TP Checkpoint: Aktif @ {cp_level_roe:.2f}% RoE ({ts_price:.4f})", Fore.MAGENTA)
                else:
                    funding_rate = pair_state.get('funding_rate', 0.0)
                    funding_color = Fore.RED if funding_rate > 0.01 else Fore.GREEN if funding_rate < -0.01 else Fore.WHITE
                    print_colored(f"  Status: Waiting | Funding: ", end=""); print_colored(f"{funding_rate:.4f}%", funding_color)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def main():
    global is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    
    # PERUBAHAN: Inisialisasi sesi Bybit di awal
    initialize_bybit_client()

    # NOTE: Backtest tidak diubah, tetap berjalan dengan simulasi
    # check_and_run_backtests() # Anda bisa uncomment ini jika perlu
    
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
                print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN)
                run_dashboard_mode()
                is_autopilot_running = False
                print_colored("\n🛑 Live Dashboard ditutup.", Fore.RED)
            elif cmd == '!watch':
                if len(parts) >= 2:
                    pair_id = parts[1].upper(); tf = parts[2] if len(parts) > 2 else '1H'
                    # PERUBAHAN: Menyimpan dalam format baru
                    if pair_id not in current_settings['watched_pairs']:
                         current_settings['watched_pairs'][pair_id] = {"timeframe": tf, "leverage": 10} # Default leverage 10x
                    else: # Jika sudah ada, hanya update timeframe
                         current_settings['watched_pairs'][pair_id]['timeframe'] = tf
                    save_settings()
                    print_colored(f"{pair_id} ({tf}) ditambahkan/diperbarui. Gunakan '!leverage' untuk mengubah leverage.", Fore.GREEN)
                else: print_colored("Format: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!leverage': # PERUBAHAN: Menambahkan handler baru
                handle_leverage_command(parts)
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
                    # PERUBAHAN: Menampilkan dengan format baru
                    for pair, config in watched.items(): 
                        print_colored(f"- {pair} (TF: {config['timeframe']}, Leverage: {config.get('leverage', 'N/A')}x)", Fore.WHITE)
            elif cmd == '!history': display_history()
            # PERUBAHAN: handle_settings_command tidak diubah karena tidak menangani leverage
            elif cmd in ['!settings', '!set']: print_colored("Gunakan !help untuk melihat perintah yang tersedia.", Fore.YELLOW)
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
