import json
import os
import time
import threading
import requests
import subprocess
from datetime import datetime
from colorama import init, Fore, Style
import asyncio

# --- KONFIGURASI GLOBAL & STATE ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
OKX_API_URL = "https://www.okx.com/api/v5"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MIN_PROFIT_PERCENTAGE = 0.1
DEFAULT_TIMEFRAME = '1H'
# DIUBAH: Konstanta ini tidak lagi digunakan, nilainya akan diambil dari settings.
# ANALYSIS_INTERVAL_SECONDS = 20 
REFRESH_INTERVAL_SECONDS = 5

# State
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_ai_thinking = False
is_autopilot_in_cooldown = False
stop_event = threading.Event()
termux_api_warning_shown = False

# Inisialisasi
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---

def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def send_termux_notification(title, content):
    # ... (fungsi ini tidak berubah)
    global termux_api_warning_shown
    try:
        subprocess.run(['termux-notification','--title', title,'--content', content,'--group', 'StrategicAI'], check=True, timeout=5)
    except FileNotFoundError:
        if not termux_api_warning_shown:
            print_colored("\n[PERINGATAN] 'termux-api' tidak terpasang. Notifikasi dinonaktifkan.", Fore.YELLOW)
            termux_api_warning_shown = True
    except Exception as e:
        print_colored(f"\n[ERROR] Gagal mengirim notifikasi: {e}", Fore.RED)


def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("   Strategic AI Analyst (Configurable Edition)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Gunakan '!config delay <detik>' untuk mengubah interval analisis.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!pair <PAIR> [TF]      - Ganti pair dan timeframe (misal: !pair BTC-USDT 15m)", Fore.GREEN)
    print_colored("!config delay <detik>  - Ubah interval analisis Autopilot (misal: !config delay 30)", Fore.GREEN) # DIUBAH
    print_colored("!status                - Tampilkan status saat ini", Fore.GREEN)
    print_colored("!history               - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!exit                  - Keluar dari aplikasi", Fore.GREEN)
    print_colored("Teks lain              - Kirim pesan ke Analyst AI (chat)", Fore.GREEN)
    print()


# --- MANAJEMEN DATA & PENGATURAN ---

# DIUBAH: Menambahkan 'analysis_interval_seconds' ke default settings
def load_settings():
    global current_settings, current_instrument_id, DEFAULT_TIMEFRAME
    default_settings = {
        "groq_api_key": "",
        "take_profit_pct": 1.0,
        "stop_loss_pct": 0.5,
        "min_confidence": 7,
        "last_pair": None,
        "last_timeframe": "1H",
        "analysis_interval_seconds": 20  # Nilai default untuk delay
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            # Memastikan semua key ada, jika tidak ada, tambahkan dari default
            for key, value in default_settings.items():
                if key not in current_settings:
                    current_settings[key] = value
    else:
        current_settings = default_settings
        key = input("Masukkan Groq API Key Anda: ")
        current_settings["groq_api_key"] = key
        save_settings()
    if not current_settings.get("groq_api_key"):
        print_colored("ERROR: Groq API Key kosong.", Fore.RED)
        exit()
    current_instrument_id = current_settings.get("last_pair")
    DEFAULT_TIMEFRAME = current_settings.get("last_timeframe")

def save_settings():
    # ... (fungsi ini tidak berubah, tapi sekarang juga menyimpan delay)
    current_settings["last_pair"] = current_instrument_id
    current_settings["last_timeframe"] = DEFAULT_TIMEFRAME
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(current_settings, f, indent=4)

# ... (Fungsi load_trades, save_trades, display_history, fetch_okx_candle_data, get_groq_completion tidak berubah)
def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)
def display_history():
    if not autopilot_trades: print_colored("Belum ada riwayat.", Fore.YELLOW); return
    for trade in reversed(autopilot_trades):
        # ...
        pass
def fetch_okx_candle_data(instId, timeframe):
    # ...
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        r = requests.get(url, timeout=10); r.raise_for_status(); data = r.json()
        if data.get("code") == "0": return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4])} for d in data["data"]][::-1]
        else: print_colored(f"OKX API Error: {data.get('msg')}", Fore.RED); return []
    except requests.exceptions.RequestException as e: print_colored(f"OKX Network Error: {e}", Fore.RED); return []
def get_groq_completion(system_prompt, user_content, model='llama3-70b-8192', is_json=False):
    # ...
    headers = {"Authorization": f"Bearer {current_settings['groq_api_key']}", "Content-Type": "application/json"}
    payload = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],"model": model}
    if is_json: payload["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=40); r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e: print_colored(f"Groq Error: {e}", Fore.RED); return None

# --- LOGIKA INTI AI ---
# ... (Fungsi analyze_and_close_trade dan run_autopilot_analysis tidak perlu diubah, karena mereka tidak mengatur delay)
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    # ...
    pass
async def run_autopilot_analysis():
    # ...
    pass

# --- THREAD WORKERS & MAIN LOOP ---

# DIUBAH: Worker sekarang membaca delay dari settings
def autopilot_worker():
    while not stop_event.is_set():
        asyncio.run(run_autopilot_analysis())
        # Mengambil nilai delay dari settings, dengan default 20 jika tidak ada
        delay = current_settings.get('analysis_interval_seconds', 20)
        stop_event.wait(delay)

def data_refresh_worker():
    # ... (fungsi ini tidak berubah)
    global current_candle_data
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def main():
    global current_instrument_id, DEFAULT_TIMEFRAME, current_candle_data
    load_settings()
    load_trades()
    display_welcome_message()
    if current_instrument_id:
        print_colored(f"Memuat pair terakhir: {current_instrument_id}...", Fore.CYAN)
        current_candle_data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
        if current_candle_data: print_colored("Data dimuat.", Fore.GREEN)
    
    threading.Thread(target=autopilot_worker, daemon=True).start()
    threading.Thread(target=data_refresh_worker, daemon=True).start()

    while True:
        try:
            prompt_text = f"[{current_instrument_id or 'No Pair'}] > "
            user_input = input(prompt_text)

            if user_input.lower() == '!exit':
                break
            
            elif user_input.lower() == '!help':
                display_help()
            
            # DIUBAH: Menambahkan handler untuk perintah !config
            elif user_input.lower().startswith('!config '):
                parts = user_input.split()
                if len(parts) == 3 and parts[1].lower() == 'delay':
                    try:
                        new_delay = int(parts[2])
                        if new_delay < 5:
                            print_colored("Error: Delay tidak boleh kurang dari 5 detik untuk menghindari rate limit API.", Fore.RED)
                        else:
                            current_settings['analysis_interval_seconds'] = new_delay
                            save_settings() # Simpan perubahan secara permanen
                            print_colored(f"Interval analisis Autopilot diubah menjadi {new_delay} detik.", Fore.GREEN)
                    except ValueError:
                        print_colored("Error: Harap masukkan angka (detik) yang valid.", Fore.RED)
                else:
                    print_colored("Format salah. Gunakan: !config delay <detik>", Fore.YELLOW)
            
            elif user_input.lower().startswith('!pair '):
                # ... (logika !pair tidak berubah)
                parts = user_input.split();
                if len(parts) >= 2:
                    current_instrument_id = parts[1].upper()
                    DEFAULT_TIMEFRAME = parts[2] if len(parts) > 2 else '1H'
                    print_colored(f"Ganti ke {current_instrument_id} TF {DEFAULT_TIMEFRAME}...", Fore.CYAN)
                    current_candle_data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
                    save_settings()
                else: print_colored("Format salah.", Fore.RED)

            elif user_input.strip():
                # ... (logika chat tidak berubah)
                asyncio.run(handle_chat_message(user_input)) # Placeholder

        except KeyboardInterrupt:
            break
    
    print_colored("\nMenutup...", Fore.YELLOW)
    stop_event.set()

# Dummy functions for completeness
async def handle_chat_message(text): pass
# The full logic for these functions should be copied from the previous answer
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    print_colored(f"\nMenganalisis hasil trade {trade['id']}...", Fore.CYAN)
    pnl = ((exit_price - trade['entryPrice']) / trade['entryPrice']) * 100
    outcome = "PROFIT" if pnl > MIN_PROFIT_PERCENTAGE else "LOSS"
    system_prompt = "You are a concise trading analyst..."
    user_content = f"Analyze this trade:\n- Outcome: {outcome} ({pnl:.2f}%) ..."
    exit_reason_analysis = get_groq_completion(system_prompt, user_content, model='llama3-8b-8192')
    trade['status'] = 'CLOSED'; trade['exitPrice'] = exit_price; trade['exitTimestamp'] = datetime.utcnow().isoformat() + "Z"
    trade['pl_percent'] = pnl; trade['exitReason'] = exit_reason_analysis or f"Auto-closed: {close_trigger_reason}"
    pnl_text = f"PROFIT: +{pnl:.2f}%" if pnl > MIN_PROFIT_PERCENTAGE else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if pnl > MIN_PROFIT_PERCENTAGE else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    print_colored(f"   Pelajaran Baru: {trade['exitReason']}", Fore.MAGENTA, Style.BRIGHT)
    save_trades()
    notif_title = f"✅ TRADE PROFIT: {trade['instrumentId']}" if pnl > MIN_PROFIT_PERCENTAGE else f"❌ TRADE LOSS: {trade['instrumentId']}"
    notif_content = f"P/L: {pnl:.2f}%\nEntry: {trade['entryPrice']:.4f} -> Exit: {exit_price:.4f}\nPelajaran: {trade['exitReason']}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or not current_candle_data or is_autopilot_in_cooldown: return
    is_ai_thinking = True
    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        current_price = current_candle_data[-1]['close']
        if open_position:
            pnl = ((current_price - open_position['entryPrice']) / open_position['entryPrice']) * 100
            if (current_settings.get('take_profit_pct') and pnl >= current_settings['take_profit_pct']) or (current_settings.get('stop_loss_pct') and pnl <= -current_settings['stop_loss_pct']):
                await analyze_and_close_trade(open_position, current_price, f"TP/SL Hit ({pnl:.2f}%)"); is_ai_thinking = False; return
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Autopilot menganalisis...", Fore.MAGENTA)
        past_trades = [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'CLOSED'][-3:]
        learning_context = "History:\n" + "\n".join([f"- {t.get('exitReason')}" for t in past_trades]) if past_trades else "No history."
        system_prompt = f"You are a self-correcting trading bot... Learn from this: {learning_context} ..."
        user_content = f"Analyze for {current_instrument_id} ..."
        ai_response_str = get_groq_completion(system_prompt, user_content, is_json=True)
        if not ai_response_str: raise Exception("AI response empty.")
        ai_response = json.loads(ai_response_str)
        action = ai_response.get('action', 'HOLD').upper()
        if action == "BUY" and not open_position:
            confidence = ai_response.get('confidence', 0)
            if confidence >= current_settings.get("min_confidence", 1):
                reason = ai_response.get('reason', 'N/A')
                new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": 'LONG', "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN'}
                autopilot_trades.append(new_trade)
                print_colored(f"\n🟢 ACTION: BUY {current_instrument_id} @ {current_price}", Fore.GREEN, Style.BRIGHT)
                print_colored(f"   Confidence: {confidence}/10 | Reason: {reason}", Fore.WHITE)
                save_trades()
                notif_title = f"🚀 POSISI DIBUKA: {current_instrument_id}"
                notif_content = f"Harga Entry: {current_price:.4f}\nConfidence: {confidence}/10\nAlasan: {reason}"
                send_termux_notification(notif_title, notif_content)
        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"AI Decision: {ai_response.get('reason', 'N/A')}")
        else: print_colored(f"⚪️ HOLD: {ai_response.get('reason', 'N/A')}", Fore.CYAN)
    except Exception as e:
        print_colored(f"Autopilot Error: {e}", Fore.RED); is_autopilot_in_cooldown = True; await asyncio.sleep(5); is_autopilot_in_cooldown = False
    finally: is_ai_thinking = False
    
if __name__ == "__main__":
    main()
