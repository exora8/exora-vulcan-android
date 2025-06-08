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
ANALYSIS_INTERVAL_SECONDS = 20 # DIUBAH: Interval Autopilot menjadi 20 detik
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
    global termux_api_warning_shown
    try:
        subprocess.run(
            ['termux-notification', '--title', title, '--content', content, '--group', 'StrategicAI'],
            check=True, timeout=5
        )
    except FileNotFoundError:
        if not termux_api_warning_shown:
            print_colored("\n[PERINGATAN] 'termux-api' tidak terpasang. Notifikasi dinonaktifkan.", Fore.YELLOW)
            termux_api_warning_shown = True
    except Exception as e:
        print_colored(f"\n[ERROR] Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("    Strategic AI Analyst (Final Perfected Ver)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Interval Analisis: 20 detik. Notifikasi Aktif.", Fore.YELLOW)
    print_colored("Pastikan 'termux-api' sudah terpasang.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Perintah ---\n!pair <PAIR> [TF]\n!status\n!history\n!exit\n<teks> (untuk chat)", Fore.CYAN)

# --- MANAJEMEN DATA & PENGATURAN ---

def load_settings():
    global current_settings, current_instrument_id, DEFAULT_TIMEFRAME
    default_settings = {
        "groq_api_key": "", "take_profit_pct": 1.0, "stop_loss_pct": 0.5,
        "min_confidence": 7, "last_pair": None, "last_timeframe": "1H"
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
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
    current_settings["last_pair"] = current_instrument_id
    current_settings["last_timeframe"] = DEFAULT_TIMEFRAME
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)

def save_trades():
    with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

def display_history():
    if not autopilot_trades:
        print_colored("Belum ada riwayat.", Fore.YELLOW)
        return
    for trade in reversed(autopilot_trades):
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        print_colored(f"--- ID: {trade['id']} | Pair: {trade['instrumentId']} ---", Fore.CYAN)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f} | Alasan: {trade.get('entryReason', 'N/A')}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0)
            pl_color = Fore.GREEN if pl_percent > MIN_PROFIT_PERCENTAGE else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f} | P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
            print_colored(f"  Pelajaran: {trade.get('exitReason', 'N/A')}", Fore.MAGENTA)
        else:
            print_colored(f"  Status: {trade['status']}", Fore.YELLOW)
        print()

# --- FUNGSI API (OKX & GROQ) ---

def fetch_okx_candle_data(instId, timeframe):
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4])} for d in data["data"]][::-1]
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED)
            return []
    except requests.exceptions.RequestException as e:
        print_colored(f"OKX Network Error: {e}", Fore.RED)
        return []

def get_groq_completion(system_prompt, user_content, model='llama3-70b-8192', is_json=False):
    headers = {"Authorization": f"Bearer {current_settings['groq_api_key']}", "Content-Type": "application/json"}
    payload = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],"model": model}
    if is_json:
        payload["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print_colored(f"Groq Network Error: {e}", Fore.RED)
        return None
    except Exception as e:
        response_text = r.text if 'r' in locals() else 'No response'
        print_colored(f"Groq Response Error: {e}. Resp: {response_text}", Fore.RED)
        return None

# --- LOGIKA INTI AI ---

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    print_colored(f"\nMenganalisis hasil trade {trade['id']}...", Fore.CYAN)
    pnl = ((exit_price - trade['entryPrice']) / trade['entryPrice']) * 100
    outcome = "TRUE PROFIT" if pnl > MIN_PROFIT_PERCENTAGE else "LOSS"
    system_prompt = "You are a concise, brutally honest trading analyst... (prompt diringkas)"
    user_content = f"Analyze this trade:\n- Outcome: {outcome} ({pnl:.2f}%)\n... (prompt diringkas)"
    exit_reason_analysis = get_groq_completion(system_prompt, user_content, model='llama3-8b-8192')
    
    trade.update({
        'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z",
        'pl_percent': pnl, 'exitReason': exit_reason_analysis or f"Auto-closed: {close_trigger_reason}"
    })

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
            tp_hit = current_settings.get('take_profit_pct') and pnl >= current_settings['take_profit_pct']
            sl_hit = current_settings.get('stop_loss_pct') and pnl <= -current_settings['stop_loss_pct']
            if tp_hit or sl_hit:
                close_reason = f"TP Hit ({pnl:.2f}%)" if tp_hit else f"SL Hit ({pnl:.2f}%)"
                await analyze_and_close_trade(open_position, current_price, close_reason)
                is_ai_thinking = False; return
        
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Autopilot menganalisis...", Fore.MAGENTA)
        past_trades = [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'CLOSED'][-3:]
        learning_context = "No history."
        if past_trades:
            learning_context = "Learn from past trades:\n" + "\n".join([f"- Trade outcome: {'PROFIT' if t.get('pl_percent',0) > MIN_PROFIT_PERCENTAGE else 'LOSS'}. Lesson: {t.get('exitReason')}" for t in past_trades])
        
        system_prompt = f"You are a self-correcting trading bot. Your goal is sniper trades. {learning_context} CRITICAL RULE: Avoid setups that resemble past losses. Your response MUST be JSON..."
        user_content = f"Analyze {current_instrument_id} at {current_price}... (prompt diringkas)"
        
        ai_response_str = get_groq_completion(system_prompt, user_content, is_json=True)
        if not ai_response_str: raise Exception("AI response was empty.")
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
            reason = ai_response.get('reason', 'AI decided to close.')
            await analyze_and_close_trade(open_position, current_price, f"AI Decision: {reason}")
        else:
            print_colored(f"⚪️ HOLD: {ai_response.get('reason', 'N/A')}", Fore.CYAN)
    except Exception as e:
        print_colored(f"Autopilot Error: {e}", Fore.RED)
        is_autopilot_in_cooldown = True
        await asyncio.sleep(5)
        is_autopilot_in_cooldown = False
    finally:
        is_ai_thinking = False

# --- THREAD WORKERS & MAIN LOOP ---

async def handle_chat_message(user_text): pass
def autopilot_worker():
    while not stop_event.is_set():
        asyncio.run(run_autopilot_analysis())
        stop_event.wait(ANALYSIS_INTERVAL_SECONDS)
def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def main():
    global current_instrument_id, DEFAULT_TIMEFRAME, current_candle_data
    load_settings(); load_trades(); display_welcome_message()
    if current_instrument_id:
        print_colored(f"Memuat pair terakhir: {current_instrument_id}...", Fore.CYAN)
        current_candle_data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
        if current_candle_data: print_colored("Data dimuat.", Fore.GREEN)
    
    threading.Thread(target=autopilot_worker, daemon=True).start()
    threading.Thread(target=data_refresh_worker, daemon=True).start()

    while True:
        try:
            user_input = input(f"[{current_instrument_id or 'No Pair'}] > ")
            if user_input.lower() == '!exit': break
            elif user_input.lower() == '!help': display_help()
            elif user_input.lower() == '!status':
                 if not current_instrument_id: print_colored("Pilih pair dulu.", Fore.YELLOW)
                 else:
                     # ... (kode status diringkas)
                     print("Menampilkan status...")
            elif user_input.lower() == '!history': display_history()
            elif user_input.lower().startswith('!pair '):
                parts = user_input.split(); current_instrument_id = parts[1].upper()
                DEFAULT_TIMEFRAME = parts[2] if len(parts) > 2 else '1H'
                print_colored(f"Ganti ke {current_instrument_id} TF {DEFAULT_TIMEFRAME}...", Fore.CYAN)
                current_candle_data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
                save_settings()
            elif user_input.strip(): asyncio.run(handle_chat_message(user_input))
        except KeyboardInterrupt: break
    print_colored("\nMenutup...", Fore.YELLOW); stop_event.set()

if __name__ == "__main__":
    main()
