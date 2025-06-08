import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
import asyncio

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
# BARU: File untuk menyimpan "Buku Jurnal" AI
STRATEGY_BOOK_FILE = 'strategy_book.json'
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
strategy_book = {} # "Buku Jurnal" AI akan dimuat di sini
current_instrument_id = None
current_candle_data = []
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        os.system(f'termux-notification --title "{title.replace("\"", "\'")}" --content "{content.replace("\"", "\'")}"')
    except Exception as e:
        print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("       Strategic AI Analyst (Local AI Edition)      ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini 100% lokal dan belajar dari setiap trade.", Fore.YELLOW)
    print_colored("Gunakan '!start' untuk memulai proses belajar.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot AI", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot AI", Fore.GREEN)
    print_colored("!pair <PAIR> [TF]   - Ganti pair dan timeframe", Fore.GREEN)
    print_colored("!status               - Tampilkan status saat ini", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!book                 - Tampilkan 'Buku Jurnal' / strategi AI", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan saat ini", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings, current_instrument_id
    default_settings = {
        "take_profit_pct": 1.5, "stop_loss_pct": 0.8,
        "fee_pct": 0.1, "analysis_interval_sec": 30, "confidence_threshold": 2.0,
        "last_pair": None, "last_timeframe": "1H"
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else:
        current_settings = default_settings; save_settings()
    current_instrument_id = current_settings.get("last_pair")

def save_settings():
    current_settings["last_pair"] = current_instrument_id
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_data():
    global autopilot_trades, strategy_book
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
    if os.path.exists(STRATEGY_BOOK_FILE):
        with open(STRATEGY_BOOK_FILE, 'r') as f: strategy_book = json.load(f)

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

def save_strategy_book():
    with open(STRATEGY_BOOK_FILE, 'w') as f: json.dump(strategy_book, f, indent=4)

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades[-10:]): # Tampilkan 10 terakhir
        # ... logika display history ...
        pass # Disembunyikan untuk keringkasan

def display_strategy_book():
    print_colored("\n--- Buku Jurnal & Strategi AI ---", Fore.CYAN, Style.BRIGHT)
    if not strategy_book:
        print_colored("Buku jurnal masih kosong. AI belum belajar apa-apa.", Fore.YELLOW)
        return
    
    sorted_strategies = sorted(strategy_book.items(), key=lambda item: item[1]['score'], reverse=True)
    for pattern, stats in sorted_strategies:
        score = stats['score']
        score_color = Fore.GREEN if score > 0 else Fore.RED if score < 0 else Fore.YELLOW
        print_colored(f"Pola: {pattern}", Fore.WHITE)
        print_colored(f"  Skor: {score:.2f}", score_color, Style.BRIGHT)
        print_colored(f"  Total Trades: {stats['trades_taken']} (Menang: {stats.get('wins', 0)})", Fore.WHITE)
        print_colored(f"  Net P/L: {stats.get('net_pnl', 0):.2f}%", Fore.WHITE)
        print("-" * 20)

# --- FUNGSI API & PERHITUNGAN LOKAL ---
def fetch_okx_candle_data(instId, timeframe):
    # ... fungsi ini tidak berubah ...
    pass

# BARU: Fungsi untuk menghitung indikator tanpa library eksternal
def calculate_sma(data, period):
    if len(data) < period: return None
    return sum(data[-period:]) / period

def calculate_rsi(data, period=14):
    if len(data) < period + 1: return None
    changes = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]

    avg_gain = sum(gains[:period]) / period if len(gains) >= period else 0
    avg_loss = sum(losses[:period]) / period if len(losses) >= period else 1e-10

    for i in range(period, len(changes)):
        change = changes[i]
        gain = change if change > 0 else 0
        loss = -change if change < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# --- LOGIKA INTI AI ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# DIUBAH TOTAL: Fungsi ini sekarang hanya memperbarui 'Buku Jurnal'
def update_strategy_book_on_close(trade, exit_price):
    pattern_key = trade.get('pattern_key')
    if not pattern_key: return # Tidak bisa belajar jika tidak tahu polanya

    print_colored(f"\nAI sedang belajar dari trade {trade['id']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_win = pnl > fee

    # Ambil atau buat entri baru di buku jurnal
    strategy = strategy_book.setdefault(pattern_key, {'score': 0, 'trades_taken': 0, 'wins': 0, 'net_pnl': 0})
    
    # Update statistik
    strategy['trades_taken'] += 1
    strategy['net_pnl'] = strategy.get('net_pnl', 0) + pnl
    if is_win:
        strategy['score'] += 1.0
        strategy['wins'] = strategy.get('wins', 0) + 1
        print_colored(f"Pelajaran: Pola '{pattern_key}' berhasil. Skor naik menjadi {strategy['score']:.2f}", Fore.GREEN)
    else:
        # Hukuman untuk loss lebih berat
        strategy['score'] -= 1.5
        print_colored(f"Pelajaran: Pola '{pattern_key}' gagal. Skor turun menjadi {strategy['score']:.2f}", Fore.RED)
        
    save_strategy_book()

# DIUBAH TOTAL: Logika autopilot sekarang berbasis aturan dan skor
def run_local_autopilot_analysis():
    global is_autopilot_running, current_candle_data
    if not is_autopilot_running or len(current_candle_data) < 22: # Perlu data cukup untuk indikator
        time.sleep(1)
        return

    open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
    
    # --- Analisis Penutupan Posisi ---
    if open_position:
        current_price = current_candle_data[-1]['close']
        pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type', 'LONG'))
        tp_pct = current_settings.get('take_profit_pct')
        sl_pct = current_settings.get('stop_loss_pct')
        close_reason = None
        if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
        elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
        
        if close_reason:
            print_colored(f"\n🔴 MENUTUP POSISI: {close_reason}", Fore.YELLOW, Style.BRIGHT)
            update_strategy_book_on_close(open_position, current_price)
            open_position.update({'status': 'CLOSED', 'exitPrice': current_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
            save_trades()
            send_termux_notification(f"🔴 Posisi Ditutup: {current_instrument_id}", f"PnL: {pnl:.2f}% | Alasan: {close_reason}")
            return
    
    # --- Analisis Pembukaan Posisi Baru (Hanya jika tidak ada posisi terbuka) ---
    if not open_position:
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI sedang menganalisis pola di {current_instrument_id}...", Fore.MAGENTA)
        
        close_prices = [c['close'] for c in current_candle_data]
        sma_fast = calculate_sma(close_prices, 9)
        sma_slow = calculate_sma(close_prices, 21)
        rsi = calculate_rsi(close_prices, 14)
        current_price = close_prices[-1]

        if not all([sma_fast, sma_slow, rsi]):
            print_colored("Data tidak cukup untuk analisis.", Fore.YELLOW)
            return

        # 1. Tentukan Pola Tren
        trend_pattern = "Ranging"
        if current_price > sma_fast > sma_slow: trend_pattern = "Uptrend Kuat"
        elif sma_slow > sma_fast > current_price: trend_pattern = "Downtrend Kuat"
        elif current_price > sma_slow and current_price < sma_fast: trend_pattern = "Koreksi Bearish"
        elif current_price < sma_slow and current_price > sma_fast: trend_pattern = "Koreksi Bullish"

        # 2. Tentukan Pola Momentum
        rsi_pattern = "Normal"
        if rsi > 70: rsi_pattern = "Overbought"
        elif rsi < 30: rsi_pattern = "Oversold"

        # 3. Gabungkan menjadi Kunci Pola yang Unik
        pattern_key = f"{trend_pattern} | RSI {rsi_pattern}"
        
        # 4. Ambil Keputusan Berdasarkan Buku Jurnal
        strategy = strategy_book.get(pattern_key, {'score': 0}) # Ambil skor, default 0 jika pola baru
        score = strategy['score']
        confidence_threshold = current_settings.get('confidence_threshold', 2.0)

        action = "HOLD"
        trade_type = None

        if score >= confidence_threshold: # Skor cukup tinggi untuk percaya diri
            if "Uptrend" in trend_pattern or "Bullish" in trend_pattern:
                action = "BUY"
                trade_type = "LONG"
            elif "Downtrend" in trend_pattern or "Bearish" in trend_pattern:
                action = "SELL"
                trade_type = "SHORT"
        
        if score <= -confidence_threshold: # Skor sangat buruk, AI "takut" pada pola ini
            print_colored(f"AI Menghindari Pola Buruk: '{pattern_key}' (Skor: {score:.2f})", Fore.RED)
            action = "HOLD"

        # 5. Eksekusi
        if action in ["BUY", "SELL"]:
            new_trade = {
                "id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type,
                "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price,
                "entryReason": f"Pola: {pattern_key} (Skor: {score:.2f})", "status": 'OPEN',
                "pattern_key": pattern_key # Simpan pola untuk proses belajar nanti
            }
            autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if action == "BUY" else Fore.RED
            print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
            print_colored(f"   Alasan: {new_trade['entryReason']}", Fore.WHITE)
            save_trades()
            send_termux_notification(f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka", f"Pair: {current_instrument_id}\nEntry: {current_price:.4f}\nAlasan: {pattern_key}")
        else:
            print_colored(f"⚪️ HOLD: Pola '{pattern_key}' (Skor: {score:.2f}) tidak memenuhi syarat.", Fore.CYAN)

# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        run_local_autopilot_analysis()
        current_delay = current_settings.get("analysis_interval_sec", 30)
        time.sleep(current_delay) # Ganti wait dengan sleep biasa

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            # ... logika fetch data ...
            pass
        time.sleep(REFRESH_INTERVAL_SECONDS)

def main():
    load_settings()
    load_data()
    display_welcome_message()
    
    # Jalankan worker di background
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    data_thread.start()
    
    while True:
        try:
            # ... logika command handler (tidak berubah secara signifikan)
            # tambahkan command '!book' untuk memanggil display_strategy_book()
            pass
        except KeyboardInterrupt:
            break
            
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    # Simplified main loop for clarity
    print("This is a conceptual script. Running main loop is disabled for review.")
    # To run, you would uncomment the following line and fill in the missing parts from previous scripts.
    # main()
