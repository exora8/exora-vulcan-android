import json
import os
import time
import threading
from datetime import datetime
from colorama import init, Fore, Style
import asyncio

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'local_ai_settings.json' # File settings baru
TRADES_FILE = 'local_ai_trades.json' # File trades baru
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
# BARU: Parameter pembelajaran lokal
risk_aversion = 0 # 0 = Normal, > 0 = Lebih penakut/konservatif

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
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
    print_colored("         Strategic AI Analyst (Local AI)          ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini berjalan 100% lokal, tanpa API Key.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot AI", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot AI", Fore.GREEN)
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
        "take_profit_pct": 1.5, "stop_loss_pct": 0.8,
        "fee_pct": 0.1, "analysis_interval_sec": 30, 
        "last_pair": None, "last_timeframe": "1H",
        "entry_score_threshold": 4 # Skor minimum untuk entry
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
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        trade_type = trade.get('type', 'LONG')
        type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Status: {trade['status']}", Fore.WHITE)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        print_colored(f"  Alasan Entry: {trade.get('entryReason', 'N/A')}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0)
            is_profit = pl_percent > current_settings.get('fee_pct', 0.1)
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
        print()

# --- FUNGSI API (HANYA OKX) ---
def fetch_okx_candle_data(instId, timeframe):
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=100" # Butuh 100 candle untuk TA
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4])} for d in data["data"]][::-1]
        else: print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return []
    except requests.exceptions.RequestException as e: print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return []

# --- LOGIKA INTI LOCAL AI ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# BARU: Fungsi untuk menghitung indikator teknikal sederhana
def calculate_technical_signals(candles):
    if len(candles) < 50: return None # Butuh data yang cukup
    
    signals = {}
    closes = [c['close'] for c in candles]
    
    # 1. Simple Moving Average (SMA) untuk tren
    sma_50 = sum(closes[-50:]) / 50
    current_price = closes[-1]
    signals['sma_trend'] = 'bullish' if current_price > sma_50 else 'bearish'
    
    # 2. Support & Resistance sederhana
    recent_low = min(c['low'] for c in candles[-20:])
    recent_high = max(c['high'] for c in candles[-20:])
    signals['support'] = recent_low
    signals['resistance'] = recent_high
    
    # 3. Momentum Candlestick
    last_candle = candles[-1]
    body_size = abs(last_candle['close'] - last_candle['open'])
    range_size = last_candle['high'] - last_candle['low']
    if range_size == 0: range_size = 0.0001
    is_bullish_candle = last_candle['close'] > last_candle['open']
    
    if body_size / range_size > 0.7: # Body candle dominan
        signals['momentum'] = 'strong_bull' if is_bullish_candle else 'strong_bear'
    else:
        signals['momentum'] = 'weak'
        
    return signals

# BARU: Fungsi ini menggantikan analyze_and_close_trade versi API
def close_trade_local(trade, exit_price, close_trigger_reason):
    global risk_aversion
    print_colored(f"\nMenutup trade {trade['id']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    # MEKANISME BELAJAR: Sesuaikan risk_aversion
    if is_profit:
        risk_aversion = max(0, risk_aversion - 1) # Kurangi "rasa takut" jika profit
        print_colored("Hasil: PROFIT. Tingkat kepercayaan diri AI pulih.", Fore.GREEN)
    else:
        risk_aversion += 2 # Tambah "rasa takut" secara signifikan jika loss
        print_colored("Hasil: LOSS. AI akan lebih berhati-hati selanjutnya.", Fore.RED)

    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z",'pl_percent': pnl})
    pnl_text = f"PROFIT: +{pnl:.2f}%" if is_profit else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if is_profit else Fore.RED
    print_colored(f"🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)

# BARU: Ini adalah otak dari Local AI kita
def run_local_ai_analysis():
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
    
    if not current_candle_data: return
    current_price = current_candle_data[-1]['close']
    
    # Periksa TP/SL untuk posisi yang terbuka
    if open_position:
        pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type', 'LONG'))
        tp_pct = current_settings.get('take_profit_pct')
        sl_pct = current_settings.get('stop_loss_pct')
        close_reason = None
        if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
        elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
        if close_reason: close_trade_local(open_position, current_price, close_reason); return
    
    # Jika tidak ada posisi terbuka, lakukan analisis untuk entry baru
    if not open_position:
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI menganalisis {current_instrument_id}...", Fore.MAGENTA)
        signals = calculate_technical_signals(current_candle_data)
        if not signals: print_colored("Data tidak cukup untuk analisis.", Fore.YELLOW); return
        
        bull_score = 0
        bear_score = 0
        reasons = []

        # 1. Analisis Tren SMA
        if signals['sma_trend'] == 'bullish': bull_score += 2; reasons.append("tren bullish (di atas SMA50)")
        else: bear_score += 2; reasons.append("tren bearish (di bawah SMA50)")

        # 2. Analisis Momentum
        if signals['momentum'] == 'strong_bull': bull_score += 2; reasons.append("momentum candle kuat naik")
        elif signals['momentum'] == 'strong_bear': bear_score += 2; reasons.append("momentum candle kuat turun")

        # 3. Analisis Support/Resistance
        price_to_support_dist = abs(current_price - signals['support'])
        price_to_res_dist = abs(current_price - signals['resistance'])
        if price_to_support_dist < price_to_res_dist: # Lebih dekat ke support
            bull_score += 1; reasons.append("dekat dengan support")
        else: # Lebih dekat ke resistance
            bear_score += 1; reasons.append("dekat dengan resistance")
        
        # Hitung skor total
        final_score = bull_score - bear_score
        
        # MEKANISME BELAJAR: Gunakan risk_aversion
        entry_threshold = current_settings.get('entry_score_threshold', 4) + risk_aversion
        print_colored(f"[INFO] Skor Analisis: {final_score}. Threshold Entry: {entry_threshold} (Risk Aversion: {risk_aversion})", Fore.BLUE)
        
        action = "HOLD"
        if final_score >= entry_threshold: action = "BUY"
        elif final_score <= -entry_threshold: action = "SELL"

        # Eksekusi keputusan
        if action in ["BUY", "SELL"]:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            reason_text = " & ".join(reasons)
            new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason_text, "status": 'OPEN'}
            autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if action == "BUY" else Fore.RED
            print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
            print_colored(f"   Alasan: {reason_text}", Fore.WHITE)
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka: {current_instrument_id}"
            notif_content = f"Entry pada harga {current_price:.4f}. Skor: {final_score}"
            send_termux_notification(notif_title, notif_content)
        else:
            print_colored("⚪️ HOLD: Tidak ada sinyal yang cukup kuat.", Fore.CYAN)

# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            run_local_ai_analysis()
        current_delay = current_settings.get("analysis_interval_sec", 30)
        stop_event.wait(current_delay)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    # DIUBAH: Menambahkan entry_score_threshold ke settings
    setting_map = {'tp': ('take_profit_pct', '%'),'sl': ('stop_loss_pct', '%'),'fee': ('fee_pct', '%'),'delay': ('analysis_interval_sec', 's'),'score': ('entry_score_threshold', ' pts')}
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full_key, unit) in setting_map.items():
            display_key = key.capitalize().ljust(10)
            print_colored(f"{display_key} ({key:<10}) : {current_settings[full_key]}{unit}", Fore.WHITE)
        print(); return
    if len(parts) == 3 and parts[0] == '!set':
        key_short = parts[1].lower()
        if key_short not in setting_map: print_colored(f"Error: Kunci '{key_short}' tidak dikenal.", Fore.RED); return
        try: value = float(parts[2])
        except ValueError: print_colored(f"Error: Nilai '{parts[2]}' harus angka.", Fore.RED); return
        key_full, unit = setting_map[key_short]
        current_settings[key_full] = value; save_settings()
        print_colored(f"Pengaturan '{key_full}' diubah menjadi {value}{unit}.", Fore.GREEN, Style.BRIGHT); return
    print_colored("Format salah. Gunakan '!settings' atau '!set <key> <value>'.", Fore.RED)

def main():
    global current_instrument_id, current_candle_data, is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    if current_instrument_id:
        print_colored(f"Memuat pair terakhir: {current_instrument_id} ({current_settings.get('last_timeframe', '1H')})...", Fore.CYAN)
        current_candle_data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
        if current_candle_data: print_colored("Data berhasil dimuat.", Fore.GREEN)
        else: print_colored("Gagal memuat data terakhir.", Fore.RED)
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()
    while True:
        try:
            user_input = input(f"[{current_instrument_id or 'No Pair'}] > ")
            command_parts = user_input.split();
            if not command_parts: continue
            cmd = command_parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan.", Fore.YELLOW)
                elif not current_instrument_id: print_colored("Error: Pilih pair dulu dengan '!pair'.", Fore.RED)
                else: is_autopilot_running = True; print_colored("✅ Autopilot Lokal diaktifkan.", Fore.GREEN, Style.BRIGHT)
            elif cmd == '!stop':
                if not is_autopilot_running: print_colored("Autopilot sudah tidak aktif.", Fore.YELLOW)
                else: is_autopilot_running = False; print_colored("🛑 Autopilot Lokal dinonaktifkan.", Fore.RED, Style.BRIGHT)
            elif cmd == '!status':
                if not current_instrument_id: print_colored("Pilih pair dulu.", Fore.YELLOW)
                else:
                    price = current_candle_data[-1]['close'] if current_candle_data else 'N/A'
                    print_colored(f"\n--- Status Saat Ini ---", Fore.CYAN, Style.BRIGHT)
                    ap_status, ap_color = ("Aktif", Fore.GREEN) if is_autopilot_running else ("Tidak Aktif", Fore.RED)
                    print_colored(f"Autopilot Status  : {ap_status}", ap_color, Style.BRIGHT)
                    print_colored(f"Risk Aversion     : {risk_aversion} (AI lebih berhati-hati)", Fore.YELLOW) if risk_aversion > 0 else print_colored(f"Risk Aversion     : {risk_aversion} (Normal)", Fore.WHITE)
                    print_colored(f"Pair              : {current_instrument_id}, TF: {current_settings['last_timeframe']}", Fore.WHITE)
                    print_colored(f"Harga Terkini     : {price}", Fore.WHITE)
                    open_pos = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
                    if open_pos and isinstance(price, float):
                        pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type', 'LONG'))
                        print_colored(f"Posisi Terbuka    : {open_pos.get('type')} @ {open_pos['entryPrice']:.4f}, P/L: {pnl:.2f}%", Fore.GREEN if pnl>0 else Fore.RED)
                    else: print_colored("Posisi Terbuka    : Tidak ada", Fore.WHITE)
                    print()
            elif cmd == '!history': display_history()
            elif cmd in ['!settings', '!set']: handle_settings_command(command_parts)
            elif cmd == '!pair':
                if len(command_parts) >= 2:
                    current_instrument_id = command_parts[1].upper()
                    tf = command_parts[2] if len(command_parts) > 2 else '1H'
                    current_settings['last_timeframe'] = tf
                    print_colored(f"Mengganti pair ke {current_instrument_id} TF {tf}. Memuat data...", Fore.CYAN)
                    current_candle_data = fetch_okx_candle_data(current_instrument_id, tf)
                    if current_candle_data: print_colored("Data berhasil dimuat.", Fore.GREEN)
                    else: print_colored("Gagal memuat data.", Fore.RED)
                    save_settings()
                else: print_colored("Format salah. Gunakan: !pair NAMA-PAIR [TIMEFRAME]", Fore.RED)
            elif user_input.strip():
                print_colored("Mode chat tidak tersedia di versi Local AI.", Fore.YELLOW)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
