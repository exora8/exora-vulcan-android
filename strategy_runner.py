import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
import asyncio
import math
import sys
import select
import tty
import termios

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
OKX_API_URL = "https://www.okx.com/api/v5"
REFRESH_INTERVAL_SECONDS = 3

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_ai_thinking = False
is_autopilot_in_cooldown = False
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ

# --- INISIALISASI ---
init(autoreset=True)

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
    print_colored("     Strategic AI Analyst (Real-time Live Log)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored(f"Setelah '!start', P/L akan update setiap {REFRESH_INTERVAL_SECONDS} detik.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot AI & Live Log", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot AI & Live Log", Fore.GREEN)
    print_colored("!pair <PAIR> [TF]   - Ganti pair dan timeframe", Fore.GREEN)
    print_colored("!status               - Tampilkan status detail (satu kali)", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (contoh: !set tp 1.5)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings, current_instrument_id
    default_settings = { "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "fee_pct": 0.1, "analysis_interval_sec": 30, "last_pair": None, "last_timeframe": "1H" }
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

# --- FUNGSI API (HANYA OKX) ---
def fetch_okx_candle_data(instId, timeframe):
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        response = requests.get(url, timeout=10)
        response.raise_for_status(); data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4])} for d in data["data"]][::-1]
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return []
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return []

# --- OTAK LOCAL AI ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair):
        self.settings = settings
        self.past_trades = past_trades_for_pair

    def calculate_ema(self, data, period):
        if len(data) < period: return None
        multiplier = 2 / (period + 1)
        ema = sum(d['close'] for d in data[:period]) / period
        for d in data[period:]: ema = (d['close'] - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, data, period=14):
        if len(data) <= period: return 50
        closes = [d['close'] for d in data]
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0: gains.append(change); losses.append(0)
            else: losses.append(abs(change)); gains.append(0)
        avg_gain = sum(gains[-period:]) / period; avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100: return None
        ema9 = self.calculate_ema(candle_data, 9); ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100); rsi = self.calculate_rsi(candle_data, 14)
        bias = "RANGING"
        if ema9 > ema50 and ema50 > ema100: bias = "BULLISH_STRONG"
        elif ema9 < ema50 and ema50 < ema100: bias = "BEARISH_STRONG"
        return {"ema9": ema9, "ema50": ema50, "ema100": ema100, "rsi": rsi, "bias": bias}

    def check_for_repeated_mistake(self, current_analysis):
        losing_trades = [t for t in self.past_trades if t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        if not losing_trades: return False
        for loss in losing_trades[-3:]:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot: continue
            bias_same = current_analysis['bias'] == past_snapshot.get('bias')
            rsi_similar = abs(current_analysis['rsi'] - past_snapshot.get('rsi', 50)) < 15
            if bias_same and rsi_similar:
                print_colored(f"\n[LEARNING] Menghindari posisi karena mirip dengan loss trade #{loss['id']}", Fore.MAGENTA, end='\n')
                return True
        return False

    def get_decision(self, candle_data, open_position):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data tidak cukup untuk analisis."}
        current_price = candle_data[-1]['close']
        if open_position:
            pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type'))
            trade_type = open_position.get('type')
            if trade_type == 'LONG' and (analysis['rsi'] > 78 or current_price < analysis['ema9']): return {"action": "CLOSE", "reason": f"Sinyal exit (RSI:{analysis['rsi']:.0f}/<EMA9)."}
            if trade_type == 'SHORT' and (analysis['rsi'] < 22 or current_price > analysis['ema9']): return {"action": "CLOSE", "reason": f"Sinyal exit (RSI:{analysis['rsi']:.0f}/>EMA9)."}
            return {"action": "HOLD", "reason": f"Holding {trade_type}, P/L: {pnl:.2f}%"}
        if self.check_for_repeated_mistake(analysis): return {"action": "HOLD", "reason": "Menghindari pengulangan kesalahan."}
        bias, rsi = analysis['bias'], analysis['rsi']
        if bias == "BULLISH_STRONG" and rsi < 55 and current_price > analysis['ema50']: return {"action": "BUY", "reason": f"Tren Bullish Kuat & RSI pullback ({rsi:.0f}).", "snapshot": analysis}
        if bias == "BEARISH_STRONG" and rsi > 45 and current_price < analysis['ema50']: return {"action": "SELL", "reason": f"Tren Bearish Kuat & RSI rally ({rsi:.0f}).", "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias: {bias}, RSI: {rsi:.0f}."}

# --- LOGIKA TRADING UTAMA ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, entry_snapshot=None):
    print_colored(f"\nMenutup trade {trade['id']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1); is_profit = pnl > fee
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
    if not is_profit and entry_snapshot:
        trade['entry_snapshot'] = entry_snapshot
        print_colored(f"   [LEARNING] Menyimpan snapshot kegagalan trade #{trade['id']}", Fore.MAGENTA)
    pnl_text = f"PROFIT: +{pnl:.2f}%" if is_profit else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if is_profit else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or not current_candle_data or is_autopilot_in_cooldown: return
    is_ai_thinking = True
    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        current_price = current_candle_data[-1]['close']
        if open_position:
            pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type'))
            tp_pct = current_settings.get('take_profit_pct'); sl_pct = current_settings.get('stop_loss_pct')
            close_reason = None
            if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
            elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
            if close_reason: await analyze_and_close_trade(open_position, current_price, close_reason, open_position.get("entry_snapshot")); is_ai_thinking = False; return
        
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI sedang menganalisis {current_instrument_id}...", Fore.MAGENTA, end='\n')
        local_brain = LocalAI(current_settings, [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id])
        decision = local_brain.get_decision(current_candle_data, open_position)
        action, reason = decision.get('action', 'HOLD').upper(), decision.get('reason', 'No reason provided.')
        
        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN', "entry_snapshot": decision.get("snapshot")}
            autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if action == "BUY" else Fore.RED
            print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT, end='\n')
            print_colored(f"   Reason: {reason}", Fore.WHITE, end='\n')
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"; notif_content = f"Entry @ {current_price:.4f} | Alasan: {reason}"
            send_termux_notification(notif_title, notif_content)
        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"Local AI Decision: {reason}", open_position.get("entry_snapshot"))
        else:
            print_colored(f"\n⚪️ HOLD: {reason}", Fore.CYAN, end='\n')
    except Exception as e:
        print_colored(f"\nAutopilot Error: {e}. Cooldown...", Fore.RED, end='\n')
        is_autopilot_in_cooldown = True; await asyncio.sleep(5); is_autopilot_in_cooldown = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            asyncio.run(run_autopilot_analysis())
            stop_event.wait(current_settings.get("analysis_interval_sec", 30))
        else: time.sleep(1)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

# --- FUNGSI BARU UNTUK TAMPILAN DAN INPUT ---
def get_live_log_line():
    """Membangun string untuk live log."""
    if not current_instrument_id or not current_candle_data:
        return "Autopilot tidak aktif. Pilih pair dengan '!pair <PAIR>' dan mulai dengan '!start'."

    price = current_candle_data[-1]['close']
    price_str = f"| {current_instrument_id}: {price:.4f} "
    
    open_pos = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
    
    pos_str = "| Posisi: Tidak ada"
    if open_pos:
        pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
        pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
        type_str = f"{Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED}{open_pos.get('type')}{Style.RESET_ALL}"
        pos_str = f"| Posisi: {type_str} @ {open_pos['entryPrice']:.4f} | P/L: {pnl_color}{pnl:+.2f}%{Style.RESET_ALL} "
        
    return f"Live Log {price_str}{pos_str}"

def handle_command(command):
    """Memproses perintah yang sudah lengkap."""
    global is_autopilot_running, current_instrument_id
    parts = command.split()
    cmd = parts[0].lower() if parts else ""
    print() # Pindah baris setelah perintah dieksekusi

    if cmd == '!exit': return False # Sinyal untuk keluar dari loop utama
    elif cmd == '!help': display_help()
    elif cmd == '!start':
        if is_autopilot_running: print_colored("Autopilot sudah berjalan.", Fore.YELLOW)
        elif not current_instrument_id: print_colored("Error: Pilih pair dulu dengan '!pair'.", Fore.RED)
        else: is_autopilot_running = True; print_colored("✅ Autopilot diaktifkan. Live Log dimulai...", Fore.GREEN, Style.BRIGHT)
    elif cmd == '!stop':
        if not is_autopilot_running: print_colored("Autopilot sudah tidak aktif.", Fore.YELLOW)
        else: is_autopilot_running = False; print_colored("🛑 Autopilot dinonaktifkan. Live Log berhenti.", Fore.RED, Style.BRIGHT)
    elif cmd == '!status':
        # ... bisa ditambahkan fungsi status detail di sini
        pass
    elif cmd == '!history': display_history()
    elif cmd in ['!settings', '!set']:
        # ... bisa ditambahkan fungsi settings detail di sini
        pass
    elif cmd == '!pair':
        if len(parts) >= 2:
            current_instrument_id = parts[1].upper()
            tf = parts[2] if len(parts) > 2 else '1H'
            current_settings['last_timeframe'] = tf
            print_colored(f"Mengganti pair ke {current_instrument_id} TF {tf}. Memuat data...", Fore.CYAN)
            current_candle_data = fetch_okx_candle_data(current_instrument_id, tf)
            if current_candle_data: print_colored("Data berhasil dimuat.", Fore.GREEN)
            else: print_colored("Gagal memuat data.", Fore.RED)
            save_settings()
        else: print_colored("Format salah. Gunakan: !pair NAMA-PAIR [TIMEFRAME]", Fore.RED)
    elif cmd: print_colored(f"Perintah tidak dikenal: {cmd}", Fore.YELLOW)
    
    return True

# --- LOOP UTAMA REAL-TIME ---
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

    # Simpan pengaturan terminal asli
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Atur terminal ke mode RAW untuk input per karakter
        tty.setcbreak(sys.stdin.fileno())
        
        user_command_buffer = ""
        last_display_time = 0
        
        while True:
            # 1. Proses Input Pengguna (Non-blocking)
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                if char == '\n': # Enter
                    handle_command(user_command_buffer)
                    user_command_buffer = ""
                elif ord(char) == 127: # Backspace
                    user_command_buffer = user_command_buffer[:-1]
                else:
                    user_command_buffer += char

            # 2. Render Tampilan
            if is_autopilot_running:
                # Update tampilan setiap 3 detik
                if time.time() - last_display_time > REFRESH_INTERVAL_SECONDS:
                    live_line = get_live_log_line()
                    # Gerakkan kursor ke awal baris, hapus baris, cetak, lalu kembalikan prompt
                    sys.stdout.write('\r\033[K' + live_line)
                    last_display_time = time.time()
            
            # Tampilkan prompt dan buffer perintah saat ini
            prompt = f"\n> {user_command_buffer}" if not is_autopilot_running else f"\r> {user_command_buffer}"
            sys.stdout.write(prompt)
            sys.stdout.flush()

            time.sleep(0.1) # Jeda singkat untuk tidak membebani CPU

    except KeyboardInterrupt:
        print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    finally:
        # Sangat penting untuk mengembalikan pengaturan terminal ke normal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        stop_event.set()
        autopilot_thread.join(); data_thread.join()
        print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
