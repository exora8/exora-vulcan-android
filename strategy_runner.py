import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
import math # Diperlukan untuk kalkulasi Standard Deviation

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'pure_ai_settings.json'
TRADES_FILE = 'pure_ai_trades.json'
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = [] # Sekarang hanya list of dicts biasa
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
    print_colored("     Pure Python Local AI (Pandas-Free)      ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini 100% Python murni, super ringan untuk Termux.", Fore.YELLOW)
    print_colored("Gunakan '!start' untuk memulai.", Fore.YELLOW)
    print()

def display_help():
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
        "analysis_interval_sec": 10, 
        "last_pair": None, "last_timeframe": "15m"
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

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat trade.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades):
        # ... Logika display tetap sama
        pass

# --- PERHITUNGAN INDIKATOR MURNI (TANPA PANDAS) ---
def calculate_indicators(data):
    """
    Menghitung indikator teknikal (EMA, RSI, Bollinger Bands) pada list of dicts.
    Fungsi ini memodifikasi 'data' secara langsung dengan menambahkan kunci baru.
    """
    closes = [d['close'] for d in data]

    # --- EMA Calculation ---
    def ema(period):
        if len(closes) < period: return [None] * len(closes)
        ema_values = []
        # SMA pertama sebagai nilai awal
        sma = sum(closes[:period]) / period
        ema_values.extend([None] * (period - 1) + [sma])
        multiplier = 2 / (period + 1)
        for price in closes[period:]:
            ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        return ema_values

    # --- RSI Calculation ---
    def rsi(period):
        if len(closes) < period + 1: return [None] * len(closes)
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [None] * period
        for i in range(period, len(changes)):
            if avg_loss == 0: rs = 100
            else: rs = 100 - (100 / (1 + (avg_gain / avg_loss)))
            rsi_values.append(rs)
            
            # Smoothed average
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        rsi_values.insert(0, None) # Menyesuaikan panjang
        return rsi_values

    # --- Bollinger Bands Calculation ---
    def bbands(period, std_dev_mult):
        if len(closes) < period: return ([None]*len(closes), [None]*len(closes), [None]*len(closes))
        sma_list, upper_list, lower_list = [None] * (period - 1), [None] * (period - 1), [None] * (period - 1)
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1 : i + 1]
            sma = sum(window) / period
            std_dev = math.sqrt(sum([(x - sma) ** 2 for x in window]) / period)
            sma_list.append(sma)
            upper_list.append(sma + (std_dev * std_dev_mult))
            lower_list.append(sma - (std_dev * std_dev_mult))
        return sma_list, upper_list, lower_list

    # Menjalankan kalkulasi
    ema50_values = ema(50)
    ema200_values = ema(200)
    rsi14_values = rsi(14)
    sma20, bbu, bbl = bbands(20, 2.0)

    # Menambahkan hasil ke data asli
    for i, d in enumerate(data):
        d['EMA_50'] = ema50_values[i]
        d['EMA_200'] = ema200_values[i]
        d['RSI_14'] = rsi14_values[i]
        d['BBM_20_2.0'] = sma20[i]
        d['BBU_20_2.0'] = bbu[i]
        d['BBL_20_2.0'] = bbl[i]
    return data


# --- FUNGSI INTI ---
def fetch_and_prepare_data(instId, timeframe):
    global current_candle_data
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            # Mengubah menjadi format list of dicts yang kita inginkan
            raw_data = []
            for d in reversed(data['data']): # Reversed agar data tertua di awal
                raw_data.append({
                    'timestamp': datetime.fromtimestamp(int(d[0]) / 1000),
                    'open': float(d[1]), 'high': float(d[2]),
                    'low': float(d[3]), 'close': float(d[4]), 'vol': float(d[5])
                })
            # Menghitung indikator pada data mentah
            current_candle_data = calculate_indicators(raw_data)
            return True
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return False
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return False

def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

def run_autopilot_analysis():
    if not current_candle_data: return

    open_position = next((t for t in autopilot_trades if t['status'] == 'OPEN'), None)
    
    if open_position:
        current_price = current_candle_data[-1]['close']
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

    print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Pure Python AI menganalisis {current_instrument_id}...", Fore.MAGENTA)
    
    last = current_candle_data[-1]
    prev = current_candle_data[-2]
    
    # Memastikan semua data indikator ada sebelum membuat keputusan
    if any(k not in last for k in ['EMA_50', 'EMA_200', 'RSI_14', 'BBL_20_2.0', 'BBU_20_2.0']) or \
       any(k not in prev for k in ['EMA_50', 'EMA_200']):
        print_colored(f"⚪️ HOLD: Menunggu data indikator yang cukup...", Fore.CYAN)
        return

    action = 'HOLD'
    reason = 'Tidak ada setup yang cocok dengan aturan.'
    
    # --- KUMPULAN ATURAN (RULES ENGINE) ---
    if prev['EMA_50'] < prev['EMA_200'] and last['EMA_50'] > last['EMA_200']:
        action = 'BUY'; reason = "Golden Cross (EMA 50 > EMA 200)"
    elif prev['EMA_50'] > prev['EMA_200'] and last['EMA_50'] < last['EMA_200']:
        action = 'SELL'; reason = "Death Cross (EMA 50 < EMA 200)"

    if last['RSI_14'] < 30 and last['close'] > last['EMA_200']:
        action = 'BUY'; reason = f"RSI Oversold ({last['RSI_14']:.1f}) dalam tren Bullish"
    elif last['RSI_14'] > 70 and last['close'] < last['EMA_200']:
        action = 'SELL'; reason = f"RSI Overbought ({last['RSI_14']:.1f}) dalam tren Bearish"
        
    if last['close'] < last['BBL_20_2.0'] and last['RSI_14'] < 35:
        action = 'BUY'; reason = "Harga menyentuh Bollinger Band Bawah"
    elif last['close'] > last['BBU_20_2.0'] and last['RSI_14'] > 65:
        action = 'SELL'; reason = "Harga menyentuh Bollinger Band Atas"

    if action != 'HOLD':
        current_price = last['close']
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
        time.sleep(current_settings.get("analysis_interval_sec", 10))

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
            # ... Logika perintah lainnya (help, start, stop, status, etc.) tetap sama
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error: {e}", Fore.RED)

    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    
if __name__ == "__main__":
    main()
