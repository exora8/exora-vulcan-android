import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
import pandas as pd
import pandas_ta as ta

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'local_ai_settings.json'
TRADES_FILE = 'local_ai_trades.json'
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = pd.DataFrame() # Menggunakan DataFrame Pandas
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI TAMPILAN & UTILITAS ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        os.system(f'termux-notification --title "{title.replace("\"", "")}" --content "{content.replace("\"", "")}"')
    except Exception as e:
        print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("      Local AI Expert System (Rule-Based)       ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini berjalan 100% lokal tanpa API Key.", Fore.YELLOW)
    print_colored("Gunakan '!start' untuk memulai.", Fore.YELLOW)
    print()

def display_help():
    # Menghapus perintah yang tidak relevan lagi (API key)
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot", Fore.GREEN)
    print_colored("!pair <PAIR> [TF]     - Ganti pair dan timeframe", Fore.GREEN)
    print_colored("!status               - Tampilkan status saat ini", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (tp, sl, fee, delay)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()


# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings, current_instrument_id
    default_settings = {
        "take_profit_pct": 1.2, "stop_loss_pct": 0.7, "fee_pct": 0.1,
        "analysis_interval_sec": 10, # Bisa lebih cepat karena lokal
        "last_pair": None, "last_timeframe": "15m"
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

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades):
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        trade_type = trade.get('type', 'LONG')
        type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type}", type_color, Style.BRIGHT)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        print_colored(f"  Alasan (Aturan): {trade.get('entryReason', 'N/A')}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0)
            is_profit = pl_percent > current_settings.get('fee_pct', 0.1)
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
        print()


# --- FUNGSI INTI ---
def fetch_and_prepare_data(instId, timeframe):
    global current_candle_data
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'vol']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            # --- Menghitung Indikator ---
            df.ta.rsi(length=14, append=True)
            df.ta.ema(length=50, append=True, col_names=('EMA_50'))
            df.ta.ema(length=200, append=True, col_names=('EMA_200'))
            df.ta.bbands(length=20, append=True)
            current_candle_data = df
            return True
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return False
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return False

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# DIUBAH TOTAL: Logika autopilot sekarang berbasis aturan, bukan LLM
def run_autopilot_analysis():
    if current_candle_data.empty: return

    open_position = next((t for t in autopilot_trades if t['status'] == 'OPEN'), None)
    
    # 1. Manajemen Posisi Terbuka (TP/SL)
    if open_position:
        current_price = current_candle_data['close'].iloc[-1]
        pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position['type'])
        tp_pct = current_settings.get('take_profit_pct')
        sl_pct = current_settings.get('stop_loss_pct')
        close_reason = None
        if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
        elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
        if close_reason:
            trade = open_position
            trade.update({'status': 'CLOSED', 'exitPrice': current_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
            print_colored(f"🔴 Posisi {trade['type']} Ditutup: {close_reason} | P/L: {pnl:.2f}%", Fore.YELLOW, Style.BRIGHT)
            save_trades(); send_termux_notification(f"🔴 Posisi Ditutup: {trade['instrumentId']}", f"PnL: {pnl:.2f}% | Alasan: TP/SL")
        return

    # 2. Analisis untuk Entry Baru (jika tidak ada posisi terbuka)
    print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI menganalisis {current_instrument_id}...", Fore.MAGENTA)
    
    # Ambil data candle terakhir untuk analisis
    last = current_candle_data.iloc[-1]
    prev = current_candle_data.iloc[-2]
    
    action = 'HOLD'
    reason = 'Tidak ada setup yang cocok dengan aturan.'
    
    # --- KUMPULAN ATURAN (RULES ENGINE) ---
    # Aturan #1: Golden Cross / Death Cross Sederhana
    if prev.EMA_50 < prev.EMA_200 and last.EMA_50 > last.EMA_200:
        action = 'BUY'
        reason = "Golden Cross (EMA 50 memotong ke atas EMA 200)"
    elif prev.EMA_50 > prev.EMA_200 and last.EMA_50 < last.EMA_200:
        action = 'SELL'
        reason = "Death Cross (EMA 50 memotong ke bawah EMA 200)"

    # Aturan #2: RSI Overbought/Oversold dengan konfirmasi tren
    if last.RSI_14 < 30 and last.close > last.EMA_200: # Oversold dalam tren naik
        action = 'BUY'
        reason = f"RSI Oversold ({last.RSI_14:.1f}) dalam tren Bullish (di atas EMA 200)"
    elif last.RSI_14 > 70 and last.close < last.EMA_200: # Overbought dalam tren turun
        action = 'SELL'
        reason = f"RSI Overbought ({last.RSI_14:.1f}) dalam tren Bearish (di bawah EMA 200)"
        
    # Aturan #3: Bounce dari Bollinger Bands
    if last.close < last.BBL_20_2_0 and last.RSI_14 < 35:
        action = 'BUY'
        reason = "Harga menyentuh Bollinger Band Bawah & RSI rendah"
    elif last.close > last.BBU_20_2_0 and last.RSI_14 > 65:
        action = 'SELL'
        reason = "Harga menyentuh Bollinger Band Atas & RSI tinggi"

    # --- Eksekusi ---
    if action != 'HOLD':
        current_price = last.close
        trade_type = "LONG" if action == "BUY" else "SHORT"
        new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN'}
        autopilot_trades.append(new_trade)
        action_color = Fore.GREEN if action == "BUY" else Fore.RED
        print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
        print_colored(f"   Aturan Pemicu: {reason}", Fore.WHITE)
        save_trades()
        notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka: {current_instrument_id}"
        notif_content = f"Entry @ {current_price:.4f}. Aturan: {reason}"
        send_termux_notification(notif_title, notif_content)
    else:
        print_colored(f"⚪️ HOLD: {reason}", Fore.CYAN)


# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            run_autopilot_analysis()
        current_delay = current_settings.get("analysis_interval_sec", 10)
        time.sleep(current_delay)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            fetch_and_prepare_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
        time.sleep(REFRESH_INTERVAL_SECONDS)

def main():
    global current_instrument_id, is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    if current_instrument_id:
        print_colored(f"Memuat data untuk pair terakhir: {current_instrument_id}...", Fore.CYAN)
        if not fetch_and_prepare_data(current_instrument_id, current_settings.get('last_timeframe', '1H')):
            print_colored("Gagal memuat data terakhir.", Fore.RED)
        else:
            print_colored("Data berhasil dimuat dan dianalisis.", Fore.GREEN)
    
    threading.Thread(target=autopilot_worker, daemon=True).start()
    threading.Thread(target=data_refresh_worker, daemon=True).start()
    
    while True:
        try:
            user_input = input(f"[{current_instrument_id or 'No Pair'}] > ")
            parts = user_input.split()
            if not parts: continue
            cmd = parts[0].lower()

            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if not current_instrument_id: print_colored("Pilih pair dulu dengan '!pair'.", Fore.RED)
                else: is_autopilot_running = True; print_colored("✅ Autopilot diaktifkan.", Fore.GREEN, Style.BRIGHT)
            elif cmd == '!stop': is_autopilot_running = False; print_colored("🛑 Autopilot dinonaktifkan.", Fore.RED, Style.BRIGHT)
            elif cmd == '!status':
                price = current_candle_data['close'].iloc[-1] if not current_candle_data.empty else 'N/A'
                print_colored(f"\n--- Status Saat Ini ---", Fore.CYAN, Style.BRIGHT)
                ap_status, ap_color = ("Aktif", Fore.GREEN) if is_autopilot_running else ("Tidak Aktif", Fore.RED)
                print_colored(f"Autopilot Status  : {ap_status}", ap_color, Style.BRIGHT)
                print_colored(f"Pair              : {current_instrument_id}, TF: {current_settings['last_timeframe']}", Fore.WHITE)
                print_colored(f"Harga Terkini     : {price}", Fore.WHITE)
            elif cmd == '!history': display_history()
            elif cmd in ['!settings', '!set']:
                # Logika settings...
                pass
            elif cmd == '!pair':
                if len(parts) >= 2:
                    current_instrument_id = parts[1].upper()
                    tf = parts[2] if len(parts) > 2 else '15m'
                    current_settings['last_timeframe'] = tf
                    print_colored(f"Mengganti ke {current_instrument_id} TF {tf}...", Fore.CYAN)
                    fetch_and_prepare_data(current_instrument_id, tf)
                    save_settings()
                else: print_colored("Format salah. Gunakan: !pair NAMA-PAIR [TIMEFRAME]", Fore.RED)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error: {e}", Fore.RED)

    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    
if __name__ == "__main__":
    main()
