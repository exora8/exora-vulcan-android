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
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 5
# Cooldown dalam jam untuk strategi yang gagal
STRATEGY_COOLDOWN_HOURS = 4 

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
# BARU: Dictionary untuk melacak cooldown strategi
strategy_cooldowns = {}

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        safe_title = title.replace('"', "'")
        safe_content = content.replace('"', "'")
        command = f'termux-notification --title "{safe_title}" --content "{safe_content}"'
        os.system(command)
    except Exception as e:
        print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("     Strategic AI Analyst (Local Engine Edition)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini berjalan 100% lokal tanpa API eksternal.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot Engine", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot Engine", Fore.GREEN)
    print_colored("!pair <PAIR> [TF]   - Ganti pair dan timeframe", Fore.GREEN)
    print_colored("!status               - Tampilkan status saat ini", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan saat ini", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (contoh: !set tp 1.5)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings, current_instrument_id
    default_settings = {
        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "fee_pct": 0.1,
        "analysis_interval_sec": 30, "last_pair": None, "last_timeframe": "1H"
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else:
        current_settings = default_settings
        save_settings()
    current_instrument_id = current_settings.get("last_pair")

def save_settings():
    current_settings["last_pair"] = current_instrument_id
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

# --- FUNGSI API (Hanya OKX) ---
def fetch_okx_candle_data(instId, timeframe):
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4])} for d in data["data"]][::-1]
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return []
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return []

# --- BAGIAN INTI "LOCAL AI": PERHITUNGAN INDIKATOR ---
def calculate_sma(data, period):
    if len(data) < period: return None
    return sum(data[-period:]) / period

def calculate_rsi(data, period=14):
    if len(data) < period + 1: return None
    closes = data
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0: gains.append(change); losses.append(0)
        else: losses.append(abs(change)); gains.append(0)
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, period=20, std_dev=2):
    if len(data) < period: return None, None, None
    closes = data[-period:]
    sma = sum(closes) / period
    variance = sum([(x - sma) ** 2 for x in closes]) / period
    stdev = math.sqrt(variance)
    upper_band = sma + (stdev * std_dev)
    lower_band = sma - (stdev * std_dev)
    return sma, upper_band, lower_band

# --- LOGIKA TRADING ENGINE ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# DIUBAH: Fungsi ini sekarang menerapkan "hukuman" cooldown
def handle_trade_closure(trade, exit_price, close_trigger_reason):
    global strategy_cooldowns
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_loss = pnl < fee
    
    # "Sistem Belajar": Jika trade loss, berikan hukuman cooldown pada strategi yg digunakan
    if is_loss and 'strategy' in trade:
        strategy_name = trade['strategy']
        cooldown_end = datetime.now() + timedelta(hours=STRATEGY_COOLDOWN_HOURS)
        if current_instrument_id not in strategy_cooldowns:
            strategy_cooldowns[current_instrument_id] = {}
        strategy_cooldowns[current_instrument_id][strategy_name] = cooldown_end
        lesson = f"Strategi '{strategy_name}' gagal & dihukum cooldown selama {STRATEGY_COOLDOWN_HOURS} jam."
        print_colored(f"   Pelajaran Baru: {lesson}", Fore.MAGENTA, Style.BRIGHT)
    else:
        lesson = "Analisis mandiri: Periksa chart untuk alasan profit/loss."

    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z",'pl_percent': pnl, 'exitReason': lesson})
    
    pnl_text = f"PROFIT: +{pnl:.2f}%" if not is_loss else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if not is_loss else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)

# DIUBAH: INI ADALAH "LOCAL AI" ENGINE UTAMA
def run_local_sniper_analysis():
    global is_autopilot_running
    if not is_autopilot_running: return

    open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
    
    # 1. Cek Penutupan Posisi (TP/SL)
    if open_position and current_candle_data:
        current_price = current_candle_data[-1]['close']
        pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type', 'LONG'))
        tp_pct = current_settings.get('take_profit_pct')
        sl_pct = current_settings.get('stop_loss_pct')
        close_reason = None
        if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
        elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
        if close_reason: handle_trade_closure(open_position, current_price, close_reason); return

    # 2. Hanya cari posisi baru jika tidak ada yang terbuka
    if open_position: return

    print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local Engine sedang berburu di {current_instrument_id}...", Fore.MAGENTA)
    
    # 3. Persiapan Data & Indikator
    if len(current_candle_data) < 21: # Butuh data yg cukup
        print_colored("Data tidak cukup untuk analisis.", Fore.YELLOW)
        return
        
    closes = [c['close'] for c in current_candle_data]
    current_price = closes[-1]
    rsi = calculate_rsi(closes)
    sma20, upper_bb, lower_bb = calculate_bollinger_bands(closes)
    
    if rsi is None or sma20 is None:
        print_colored("Gagal menghitung indikator.", Fore.YELLOW)
        return

    # 4. Daftar Strategi "Sniper"
    strategies = [
        {
            'name': 'RSI Oversold Bounce', 'type': 'LONG',
            'check': lambda: rsi < 30 and current_price <= lower_bb,
            'reason': f'RSI Oversold ({rsi:.2f}) & menyentuh Lower Bollinger Band.'
        },
        {
            'name': 'RSI Overbought Rejection', 'type': 'SHORT',
            'check': lambda: rsi > 70 and current_price >= upper_bb,
            'reason': f'RSI Overbought ({rsi:.2f}) & menyentuh Upper Bollinger Band.'
        },
        {
            'name': 'Bullish Momentum', 'type': 'LONG',
            'check': lambda: current_price > sma20 and closes[-2] <= sma20, # Baru saja menyebrang ke atas
            'reason': f'Harga menyebrang ke atas SMA-20, indikasi momentum bullish.'
        },
        {
            'name': 'Bearish Momentum', 'type': 'SHORT',
            'check': lambda: current_price < sma20 and closes[-2] >= sma20, # Baru saja menyebrang ke bawah
            'reason': f'Harga menyebrang ke bawah SMA-20, indikasi momentum bearish.'
        }
    ]

    # 5. Eksekusi Strategi
    for strat in strategies:
        # "Sistem Belajar": Cek apakah strategi ini sedang dihukum cooldown
        cooldown_until = strategy_cooldowns.get(current_instrument_id, {}).get(strat['name'])
        if cooldown_until and datetime.now() < cooldown_until:
            print_colored(f"Strategi '{strat['name']}' dilewati karena sedang cooldown.", Fore.YELLOW)
            continue

        if strat['check']():
            trade_type = strat['type']
            reason = strat['reason']
            new_trade = {
                "id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, 
                "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, 
                "entryReason": reason, "status": 'OPEN', 'strategy': strat['name'] # Simpan strategi yg digunakan
            }
            autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
            icon = '🟢' if trade_type == 'LONG' else '🔴'
            print_colored(f"\n{icon} ACTION: {trade_type} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
            print_colored(f"   Strategi: {strat['name']} | Alasan: {reason}", Fore.WHITE)
            save_trades()
            notif_title = f"{icon} Posisi {trade_type} Dibuka: {current_instrument_id}"
            notif_content = f"Entry pada harga {current_price:.4f}. Alasan: {reason}"
            send_termux_notification(notif_title, notif_content)
            return # Keluar setelah menemukan satu sinyal valid

# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        run_local_sniper_analysis()
        current_delay = current_settings.get("analysis_interval_sec", 30)
        time.sleep(current_delay)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: current_candle_data = data
        time.sleep(REFRESH_INTERVAL_SECONDS)

def main():
    global current_instrument_id, current_candle_data, is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    if current_instrument_id:
        print_colored(f"Memuat pair terakhir: {current_instrument_id} ({current_settings.get('last_timeframe', '1H')})...", Fore.CYAN)
        current_candle_data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
        if current_candle_data: print_colored("Data berhasil dimuat.", Fore.GREEN)
        else: print_colored("Gagal memuat data terakhir.", Fore.RED)
    
    # DIUBAH: worker tidak lagi async, karena tidak ada panggilan API AI
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()

    while True:
        try:
            prompt_text = f"[{current_instrument_id or 'No Pair'}] > "
            user_input = input(prompt_text)
            command_parts = user_input.split()
            if not command_parts: continue
            cmd = command_parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan.", Fore.YELLOW)
                elif not current_instrument_id: print_colored("Error: Pilih pair dulu dengan '!pair'.", Fore.RED)
                else: is_autopilot_running = True; print_colored("✅ Local Engine diaktifkan. Perburuan dimulai...", Fore.GREEN, Style.BRIGHT)
            elif cmd == '!stop':
                if not is_autopilot_running: print_colored("Autopilot sudah tidak aktif.", Fore.YELLOW)
                else: is_autopilot_running = False; print_colored("🛑 Local Engine dinonaktifkan.", Fore.RED, Style.BRIGHT)
            # ... Sisa command handler tetap sama ...

        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)

    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
