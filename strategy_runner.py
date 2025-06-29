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
REFRESH_INTERVAL_SECONDS = 0.5
MAX_TRADES_IN_HISTORY = 800

# --- STATE APLIKASI ---
current_settings = {}
manual_trades = []
market_state = {}
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
    print_colored("      Manual Trade Data Generator for AI Analyst    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Mode: Trading Manual. Setiap trade akan dicatat untuk analisis AI.", Fore.GREEN)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk masuk ke Manual Trading Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Manual Trading Dashboard", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade (terbatas 80 terakhir)", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, tp_act, tp_gap)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()


# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 0.20, "fee_pct": 0.1,
        "trailing_tp_activation_pct": 0.30, "trailing_tp_gap_pct": 0.05,
        "watched_pairs": {}
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
    global manual_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: manual_trades = json.load(f)
        except (json.JSONDecodeError, IOError): manual_trades = []
    else: manual_trades = []
    # Compatibility check for older data
    for trade in manual_trades:
        if 'current_tp_checkpoint_level' not in trade: trade['current_tp_checkpoint_level'] = 0.0

def save_trades():
    global manual_trades
    manual_trades.sort(key=lambda x: x['entryTimestamp'])
    if len(manual_trades) > MAX_TRADES_IN_HISTORY:
        num_to_trim = len(manual_trades) - MAX_TRADES_IN_HISTORY
        manual_trades = manual_trades[num_to_trim:]
        print_colored(f"Riwayat trade dibatasi. {num_to_trim} trade tertua telah dihapus.", Fore.YELLOW)
    try:
        with open(TRADES_FILE, 'w') as f: json.dump(manual_trades, f, indent=4)
    except IOError as e: print_colored(f"Error saving trades: {e}", Fore.RED)

def display_history():
    if not manual_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    fee_pct = current_settings.get('fee_pct', 0.1)
    # Sort trades by entry time, newest first
    for trade in sorted(manual_trades, key=lambda x: x['entryTimestamp'], reverse=True):
        entry_time_str = trade['entryTimestamp'].replace('Z', '')
        exit_time_str = trade.get('exitTimestamp', '').replace('Z', '')
        entry_time = datetime.fromisoformat(entry_time_str).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        trade_type = trade.get('type', 'LONG'); type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)

        if trade.get('entryReason'):
            print_colored("  Alasan Entry:", Fore.YELLOW)
            reason_lines = trade['entryReason'].split('\n')
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

# --- OTAK LOCAL AI (Untuk Analisis & Snapshot) ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair):
        self.settings = settings
        self.past_trades = past_trades_for_pair # Disimpan untuk potensi penggunaan di masa depan

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
        """Fungsi ini hanya menganalisis dan mengembalikan snapshot pasar, tanpa membuat keputusan."""
        if len(candle_data) < 100 + 3: return None
        ema9 = self.calculate_ema(candle_data, 9)
        ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100)
        if not ema9 or not ema50 or not ema100: return None

        analysis = {
            "ema9_current": ema9[-1], "ema50": ema50[-1], "ema100": ema100[-1],
            "current_candle_close": candle_data[-1]['close'],
            "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING",
        }
        pre_entry_candles = candle_data[-4:-1]
        analysis["pre_entry_candle_solidity"] = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        analysis["pre_entry_candle_direction"] = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        return analysis

# --- LOGIKA TRADING MANUAL & MANAJEMEN POSISI ---
async def close_trade_logic(trade, exit_price, reason):
    pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    exit_dt = datetime.utcnow()
    trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })

    # Simpan snapshot hanya jika trade mengalami kerugian, untuk pembelajaran
    is_profit = (pnl_gross - current_settings.get('fee_pct', 0.1)) > 0
    if is_profit and 'entry_snapshot' in trade:
        try: del trade['entry_snapshot']
        except KeyError: pass

    save_trades()
    pnl_net = pnl_gross - current_settings.get('fee_pct', 0.1)
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | Trigger: {reason}"
    send_termux_notification(notif_title, notif_content)
    print_colored(f"\n{notif_title}\n{notif_content}", Fore.YELLOW)

def handle_manual_open(pair_id, trade_type):
    open_pos = next((t for t in manual_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
    if open_pos:
        print_colored(f"Sudah ada posisi {open_pos['type']} terbuka untuk {pair_id}. Tutup dulu untuk membuka yang baru.", Fore.RED)
        return

    pair_state = market_state.get(pair_id)
    if not pair_state or not pair_state.get("candle_data"):
        print_colored(f"Data pasar untuk {pair_id} belum siap. Mohon tunggu sebentar.", Fore.YELLOW)
        return

    candle_data = pair_state["candle_data"]
    if len(candle_data) < 100 + 3:
        print_colored(f"Data candle untuk {pair_id} tidak cukup untuk membuat snapshot AI.", Fore.RED)
        return

    # Buat snapshot pasar saat ini untuk AI
    ai_analyzer = LocalAI(current_settings, [])
    market_snapshot = ai_analyzer.get_market_analysis(candle_data)
    if not market_snapshot:
        print_colored(f"Gagal membuat snapshot analisis untuk {pair_id}.", Fore.RED)
        return

    # Tambahkan info tambahan ke snapshot
    market_snapshot["funding_rate"] = pair_state.get("funding_rate", 0.0)
    
    entry_price = candle_data[-1]['close']
    new_trade = {
        "id": int(time.time()),
        "instrumentId": pair_id,
        "type": trade_type,
        "entryTimestamp": datetime.utcnow().isoformat() + 'Z',
        "entryPrice": entry_price,
        "entryReason": "Manual Entry by User",
        "status": 'OPEN',
        "entry_snapshot": market_snapshot, # Ini kuncinya: menyimpan kondisi pasar saat entry
        "run_up_percent": 0.0,
        "max_drawdown_percent": 0.0,
        "trailing_stop_price": None,
        "current_tp_checkpoint_level": 0.0
    }
    manual_trades.append(new_trade)
    save_trades()

    notif_title = f"🟢 Posisi {new_trade['type']} Dibuka: {pair_id}"
    notif_content = f"Manual Entry @ {new_trade['entryPrice']:.4f}"
    send_termux_notification(notif_title, notif_content)
    print_colored(f"\n{notif_title}\n{notif_content}", Fore.GREEN)

def handle_manual_close(pair_id):
    trade_to_close = next((t for t in manual_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
    if not trade_to_close:
        print_colored(f"Tidak ada posisi terbuka untuk {pair_id}.", Fore.RED)
        return

    pair_state = market_state.get(pair_id)
    if not pair_state or not pair_state.get("candle_data"):
        print_colored(f"Tidak bisa mendapatkan harga close untuk {pair_id}. Coba lagi.", Fore.YELLOW)
        return

    exit_price = pair_state["candle_data"][-1]['close']
    asyncio.run(close_trade_logic(trade_to_close, exit_price, "Manual Close by User"))


async def check_realtime_position_management(trade_obj, current_candle_data):
    if not trade_obj: return

    # Logika run-up dan drawdown
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

    # Logika SL
    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or \
       (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
        await close_trade_logic(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}%")
        return

    # Logika Trailing TP
    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30); gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    if trade_obj.get("current_tp_checkpoint_level", 0.0) > 0.0:
        ts_price = trade_obj.get('trailing_stop_price')
        if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or \
                                      (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
            await close_trade_logic(trade_obj, ts_price, f"Trailing TP")
            return
    
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
                open_pos = next((t for t in manual_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos:
                    asyncio.run(check_realtime_position_management(open_pos, candle_data[-1]))
            time.sleep(0.5)
        time.sleep(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'), 'fee': ('fee_pct', '%'),
        'tp_act': ('trailing_tp_activation_pct', '%'), 'tp_gap': ('trailing_tp_gap_pct', '%')
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

def run_manual_trading_dashboard():
    while True:
        try:
            print("\033[H\033[J", end="") # Clear screen
            print_colored("--- MANUAL TRADING DASHBOARD ---", Fore.CYAN, Style.BRIGHT)
            print_colored("="*80, Fore.CYAN)

            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Keluar (ketik 'exit') dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            
            for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                pair_state = market_state.get(pair_id, {})
                candle_data = pair_state.get('candle_data')
                
                if not candle_data:
                    print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                    print_colored(f"  Memuat data harga...", Fore.YELLOW)
                    continue

                current_price = candle_data[-1]['close']
                price_color = Fore.GREEN if current_price >= candle_data[-2]['close'] else Fore.RED
                
                print_colored(f"\n⦿ {pair_id} ({timeframe}) | Harga: ", Fore.WHITE, Style.BRIGHT, end="")
                print_colored(f"{current_price:.4f}", price_color, Style.BRIGHT)

                open_pos = next((t for t in manual_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                
                if open_pos:
                    pnl_net = calculate_pnl(open_pos['entryPrice'], current_price, open_pos.get('type')) - current_settings.get('fee_pct', 0.1)
                    pnl_color = Fore.GREEN if pnl_net > 0 else Fore.RED
                    print_colored(f"  Status: OPEN {open_pos.get('type')} | Entry: {open_pos['entryPrice']:.4f} | PnL(Net): ", end="")
                    print_colored(f"{pnl_net:.2f}%", pnl_color, Style.BRIGHT)
                    if open_pos.get("current_tp_checkpoint_level", 0.0) > 0:
                        cp_level = open_pos["current_tp_checkpoint_level"]
                        ts_price = open_pos.get("trailing_stop_price", 0)
                        print_colored(f"  TP Checkpoint: Aktif @ {cp_level:.2f}% (SL di {ts_price:.4f})", Fore.MAGENTA)
                else:
                    funding_rate = pair_state.get('funding_rate', 0.0)
                    funding_color = Fore.RED if funding_rate > 0.01 else Fore.GREEN if funding_rate < -0.01 else Fore.WHITE
                    print_colored(f"  Status: Waiting for command | Funding: ", end=""); print_colored(f"{funding_rate:.4f}%", funding_color)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("Perintah: long <PAIR>, short <PAIR>, close <PAIR>, exit", Fore.YELLOW)
            
            user_input = input("[Manual Trade] > ").strip().lower()
            if not user_input: continue

            parts = user_input.split()
            cmd = parts[0]

            if cmd == 'exit':
                break
            elif cmd in ['long', 'short']:
                if len(parts) == 2:
                    pair = parts[1].upper()
                    if pair in current_settings.get("watched_pairs", {}):
                        handle_manual_open(pair, cmd.upper())
                        time.sleep(2) # Beri jeda agar user bisa membaca pesan
                    else:
                        print_colored(f"Pair '{pair}' tidak ada di watchlist.", Fore.RED)
                        time.sleep(2)
                else:
                    print_colored("Format salah. Contoh: long BTC-USDT", Fore.RED)
                    time.sleep(2)
            elif cmd == 'close':
                if len(parts) == 2:
                    pair = parts[1].upper()
                    if pair in current_settings.get("watched_pairs", {}):
                        handle_manual_close(pair)
                        time.sleep(2)
                    else:
                        print_colored(f"Pair '{pair}' tidak ada di watchlist.", Fore.RED)
                        time.sleep(2)
                else:
                    print_colored("Format salah. Contoh: close BTC-USDT", Fore.RED)
                    time.sleep(2)
            else:
                print_colored(f"Perintah '{cmd}' tidak dikenal.", Fore.RED)
                time.sleep(2)

        except (KeyboardInterrupt, EOFError):
            break

def main():
    load_settings(); load_trades(); display_welcome_message()
    
    # Hanya thread data yang diperlukan sekarang
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
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
                print_colored("✅ Memasuki Manual Trading Dashboard...", Fore.GREEN)
                run_manual_trading_dashboard()
                print_colored("\n🛑 Keluar dari Manual Trading Dashboard.", Fore.RED)
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
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
