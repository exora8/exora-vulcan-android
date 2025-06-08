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
OKX_API_URL = "https://www.okx.com/api/v5"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
REFRESH_INTERVAL_SECONDS = 5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_ai_thinking = False
is_autopilot_in_cooldown = False
# BARU: State untuk mengontrol Autopilot secara manual
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
current_api_key_index = 0

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
    print_colored("      Strategic AI Analyst (Manual Control)       ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Gunakan '!start' untuk memulai Autopilot.", Fore.YELLOW)
    if IS_TERMUX:
        print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
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
    print_colored("Teks lain             - Kirim pesan ke Analyst AI (chat)", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---

def load_settings():
    global current_settings, current_instrument_id
    default_settings = {
        "groq_api_keys": [], "take_profit_pct": 1.5, "stop_loss_pct": 0.8,
        "fee_pct": 0.1, "analysis_interval_sec": 30, "min_confidence": 7, 
        "last_pair": None, "last_timeframe": "1H"
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings:
                    current_settings[key] = value
    else:
        current_settings = default_settings

    if not current_settings.get("groq_api_keys"):
        print_colored("Setup Awal: Silakan masukkan Groq API Key Anda.", Fore.YELLOW)
        print_colored("Anda bisa memasukkan lebih dari satu. Tekan Enter pada baris kosong untuk selesai.", Fore.YELLOW)
        keys = []
        while True:
            key = input(f"Masukkan API Key #{len(keys) + 1} (atau Enter untuk selesai): ")
            if not key: break
            keys.append(key)
        
        if not keys:
            print_colored("Tidak ada API Key yang dimasukkan. Aplikasi tidak bisa berjalan.", Fore.RED, Style.BRIGHT); exit()
            
        current_settings["groq_api_keys"] = keys
        save_settings()

    current_instrument_id = current_settings.get("last_pair")

def save_settings():
    current_settings["last_pair"] = current_instrument_id
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            autopilot_trades = json.load(f)

def save_trades():
    with open(TRADES_FILE, 'w') as f:
        json.dump(autopilot_trades, f, indent=4)

def display_history():
    if not autopilot_trades:
        print_colored("Belum ada riwayat trade.", Fore.YELLOW)
        return
        
    for trade in reversed(autopilot_trades):
        entry_time = datetime.fromisoformat(trade['entryTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        print_colored(f"  Alasan Entry: {trade.get('entryReason', 'N/A')}", Fore.WHITE)
        
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0)
            is_profit = pl_percent > current_settings.get('fee_pct', 0.1)
            pl_color = Fore.GREEN if is_profit else Fore.RED
            
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
            print_colored(f"  Analisis AI (Pelajaran): {trade.get('exitReason', 'N/A')}", Fore.MAGENTA, Style.BRIGHT)
        print()


# --- FUNGSI API (OKX & GROQ) ---
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

def get_groq_completion(system_prompt, user_content, model='llama3-70b-8192', is_json=False):
    global current_api_key_index
    api_key_to_use = current_settings["groq_api_keys"][current_api_key_index]
    key_display_index = current_api_key_index + 1
    print_colored(f"[INFO] Menggunakan Groq API Key #{key_display_index}...", Fore.BLUE)
    current_api_key_index = (current_api_key_index + 1) % len(current_settings["groq_api_keys"])

    headers = {"Authorization": f"Bearer {api_key_to_use}", "Content-Type": "application/json"}
    payload = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],"model": model,"temperature": 0.7,"max_tokens": 700}
    if is_json: payload["response_format"] = {"type": "json_object"}
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=40)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e: print_colored(f"Groq API Network Error dengan Key #{key_display_index}: {e}", Fore.RED); return None
    except (KeyError, IndexError, json.JSONDecodeError) as e: print_colored(f"Groq API response format error dengan Key #{key_display_index}: {e}. Resp: {response.text}", Fore.RED); return None


# --- LOGIKA INTI AI ---

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    print_colored(f"\nMenganalisis hasil trade {trade['id']} untuk pembelajaran...", Fore.CYAN)
    pnl = ((exit_price - trade['entryPrice']) / trade['entryPrice']) * 100
    
    fee = current_settings.get('fee_pct', 0.1)
    outcome = "TRUE PROFIT" if pnl > fee else "BREAK-EVEN/FEES" if pnl >= 0 else "CLEAR LOSS"
    
    system_prompt = """You are a concise, brutally honest trading analyst. Your one and only task is to provide a brief, insightful, one-sentence analysis of *why* the trade resulted in its outcome. Focus on market structure, momentum, or confirmation signals that were missed or correctly identified. Start your response directly with the analysis. This analysis will be used to teach the AI for the next trade."""
    
    user_content = f"""Analyze the following completed trade for {trade['instrumentId']}:
- Outcome: {outcome} ({pnl:.2f}%) vs Fee Threshold: {fee}%
- Entry Reason: "{trade['entryReason']}"
- How it was closed: {close_trigger_reason}
- Chart Data around Exit (last 50 candles):\n{json.dumps(current_candle_data[-50:])}"""

    exit_reason_analysis = get_groq_completion(system_prompt, user_content, model='llama3-8b-8192')
    
    trade['status'] = 'CLOSED'
    trade['exitPrice'] = exit_price
    trade['exitTimestamp'] = datetime.utcnow().isoformat() + "Z"
    trade['pl_percent'] = pnl
    trade['exitReason'] = exit_reason_analysis or f"Auto-closed: {close_trigger_reason}"

    pnl_text = f"PROFIT: +{pnl:.2f}%" if pnl > fee else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if pnl > fee else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    print_colored(f"   Pelajaran Baru: {trade['exitReason']}", Fore.MAGENTA, Style.BRIGHT)
    
    save_trades()
    notification_title = f"🔴 Posisi Ditutup: {trade['instrumentId']}"
    notification_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notification_title, notification_content)


async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or not current_candle_data or is_autopilot_in_cooldown: return
    is_ai_thinking = True

    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        current_price = current_candle_data[-1]['close']
        
        if open_position:
            pnl = ((current_price - open_position['entryPrice']) / open_position['entryPrice']) * 100
            close_reason = None
            if current_settings.get('take_profit_pct') and pnl >= current_settings['take_profit_pct']:
                close_reason = f"Take Profit @ {current_settings['take_profit_pct']}% tercapai."
            elif current_settings.get('stop_loss_pct') and pnl <= -current_settings['stop_loss_pct']:
                close_reason = f"Stop Loss @ {current_settings['stop_loss_pct']}% tercapai."

            if close_reason:
                await analyze_and_close_trade(open_position, current_price, close_reason)
                is_ai_thinking = False; return

        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Autopilot menganalisis {current_instrument_id}...", Fore.MAGENTA)
        
        past_trades = [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'CLOSED'][-3:]
        learning_context = "There is no trading history for this pair yet. Perform your best analysis based on the current market structure."
        if past_trades:
            learning_context = "Here is the analysis of your last few trades for this pair. Learn from your mistakes and successes to make a sharper decision.\n\n"
            for i, pt in enumerate(past_trades):
                is_profit_internal = pt.get('pl_percent', 0) > current_settings.get('fee_pct', 0.1)
                outcome = "PROFIT" if is_profit_internal else "LOSS"
                learning_context += f"- Trade #{i+1} ({outcome}):\n"
                learning_context += f"  - Entry Reason: \"{pt['entryReason']}\"\n"
                learning_context += f"  - The Lesson Learned (Exit Analysis): \"{pt['exitReason']}\"\n\n"
        
        system_prompt = f"""You are a strategic, data-driven, and self-correcting crypto trading bot. Your objective is to find 'sniper' trades.

**CONTEXT & RULES:**
- Current Fee Threshold for a profitable trade: **{current_settings.get('fee_pct', 0.1)}%**. A trade is only a success if P/L is greater than this.
- Your memory of past trades is provided below. Learn from it.
{learning_context}
- **CRITICAL RULE:** Avoid setups that mimic past losing trades. If a setup looks like a past loss, you MUST "HOLD".

**YOUR TASK:**
Analyze the provided real-time market data. Combine it with your historical learnings. Decide whether to "BUY", "CLOSE" (if a position is open), or "HOLD". A "BUY" action is only for high-probability setups that do NOT repeat past mistakes.

**RESPONSE FORMAT:**
You MUST respond with a valid JSON object: {{"action": "...", "reason": "...", "confidence": ... (only for BUY)}}."""

        user_content = f"Instrument: {current_instrument_id}, Timeframe: {current_settings['last_timeframe']}, Current Price: {current_price}\n"
        if open_position:
            pnl = ((current_price - open_position['entryPrice']) / open_position['entryPrice']) * 100
            user_content += f"CURRENT OPEN POSITION: LONG from {open_position['entryPrice']}. Current P/L: {pnl:.2f}%. Analyze if momentum is weakening or if it's strong enough to continue.\n"
        else:
            user_content += "CURRENT OPEN POSITION: None. Analyze for a high-quality entry.\n"
        user_content += f"LATEST CANDLESTICK DATA (last 75 candles):\n{json.dumps(current_candle_data[-75:])}"

        ai_response_str = get_groq_completion(system_prompt, user_content, is_json=True)
        if not ai_response_str: raise Exception("AI response was empty.")
        
        ai_response = json.loads(ai_response_str)
        action = ai_response.get('action', 'HOLD').upper()
        reason = ai_response.get('reason', 'No reason provided.')
        confidence = ai_response.get('confidence', 0)

        if action == "BUY" and not open_position:
            if confidence >= current_settings.get("min_confidence", 7):
                new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": 'LONG', "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN'}
                autopilot_trades.append(new_trade)
                print_colored(f"\n🟢 ACTION: BUY {current_instrument_id} @ {current_price}", Fore.GREEN, Style.BRIGHT)
                print_colored(f"   Confidence: {confidence}/10 | Reason: {reason}", Fore.WHITE)
                save_trades()
                
                notification_title = f"🟢 Posisi Dibuka: {current_instrument_id}"
                notification_content = f"Entry pada harga {current_price:.4f}. Alasan: {reason}"
                send_termux_notification(notification_title, notification_content)
            else:
                print_colored(f"⚪️ HOLD: Sinyal BUY terdeteksi (Conf: {confidence}) tetapi di bawah ambang batas ({current_settings['min_confidence']}).", Fore.YELLOW)
        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"AI Decision: {reason}")
        else:
            print_colored(f"⚪️ HOLD: {reason}", Fore.CYAN)

    except Exception as e:
        print_colored(f"Autopilot Error: {e}. Cooldown 5 detik...", Fore.RED)
        is_autopilot_in_cooldown = True
        await asyncio.sleep(5)
        is_autopilot_in_cooldown = False
    finally: is_ai_thinking = False

async def handle_chat_message(user_text):
    # Chat logic tidak perlu diubah secara signifikan
    pass


# --- THREAD WORKERS & MAIN LOOP ---

# DIUBAH: Worker sekarang memeriksa state 'is_autopilot_running'
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            asyncio.run(run_autopilot_analysis())
            current_delay = current_settings.get("analysis_interval_sec", 30)
            stop_event.wait(current_delay)
        else:
            # Jika Autopilot tidak aktif, cukup tunggu sebentar lalu cek lagi
            time.sleep(1)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {
        'tp': ('take_profit_pct', '%'), 'sl': ('stop_loss_pct', '%'),
        'fee': ('fee_pct', '%'), 'delay': ('analysis_interval_sec', ' detik'),
        'confidence': ('min_confidence', '')
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        print_colored(f"Take Profit       (tp)  : {current_settings['take_profit_pct']}%", Fore.WHITE)
        print_colored(f"Stop Loss         (sl)  : {current_settings['stop_loss_pct']}%", Fore.WHITE)
        print_colored(f"Fee Trading       (fee) : {current_settings['fee_pct']}%", Fore.WHITE)
        print_colored(f"Delay Autopilot (delay) : {current_settings['analysis_interval_sec']} detik", Fore.WHITE)
        print_colored(f"Min. Confidence (conf)  : {current_settings['min_confidence']}", Fore.WHITE)
        print()
        return

    if len(parts) == 3 and parts[0] == '!set':
        key_short = parts[1].lower()
        if key_short not in setting_map:
            print_colored(f"Error: Kunci '{key_short}' tidak dikenal.", Fore.RED); return
        try:
            value = float(parts[2])
            if value < 0:
                print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
        except ValueError:
            print_colored(f"Error: Nilai '{parts[2]}' harus berupa angka.", Fore.RED); return
            
        key_full, unit = setting_map[key_short]
        current_settings[key_full] = value
        save_settings()
        print_colored(f"Pengaturan '{key_full}' berhasil diubah menjadi {value}{unit}.", Fore.GREEN, Style.BRIGHT)
        return
        
    print_colored("Format salah. Gunakan '!settings' atau '!set <key> <value>'.", Fore.RED)

def main():
    global current_instrument_id, current_candle_data, is_autopilot_running
    
    load_settings()
    load_trades()
    display_welcome_message()

    if current_instrument_id:
        print_colored(f"Memuat pair terakhir: {current_instrument_id} ({current_settings.get('last_timeframe', '1H')})...", Fore.CYAN)
        current_candle_data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
        if current_candle_data: print_colored("Data berhasil dimuat.", Fore.GREEN)
        else: print_colored("Gagal memuat data terakhir.", Fore.RED)

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

            # BARU: Menangani perintah !start dan !stop
            elif cmd == '!start':
                if is_autopilot_running:
                    print_colored("Autopilot sudah berjalan.", Fore.YELLOW)
                elif not current_instrument_id:
                    print_colored("Error: Harap pilih pair terlebih dahulu dengan perintah '!pair'.", Fore.RED)
                else:
                    is_autopilot_running = True
                    print_colored("✅ Autopilot diaktifkan. Analisis akan dimulai...", Fore.GREEN, Style.BRIGHT)
            
            elif cmd == '!stop':
                if not is_autopilot_running:
                    print_colored("Autopilot sudah tidak aktif.", Fore.YELLOW)
                else:
                    is_autopilot_running = False
                    print_colored("🛑 Autopilot dinonaktifkan.", Fore.RED, Style.BRIGHT)

            elif cmd == '!status':
                if not current_instrument_id: print_colored("Pilih pair dulu.", Fore.YELLOW)
                else:
                    price = current_candle_data[-1]['close'] if current_candle_data else 'N/A'
                    print_colored(f"\n--- Status Saat Ini ---", Fore.CYAN, Style.BRIGHT)
                    # DIUBAH: Menambahkan status Autopilot
                    ap_status = "Aktif" if is_autopilot_running else "Tidak Aktif"
                    ap_color = Fore.GREEN if is_autopilot_running else Fore.RED
                    print_colored(f"Autopilot Status  : {ap_status}", ap_color, Style.BRIGHT)
                    print_colored(f"Pair              : {current_instrument_id}, TF: {current_settings['last_timeframe']}", Fore.WHITE)
                    print_colored(f"Harga Terkini     : {price}", Fore.WHITE)
                    
                    open_pos = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
                    if open_pos and isinstance(price, float):
                        pnl = ((price - open_pos['entryPrice']) / open_pos['entryPrice']) * 100
                        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                        print_colored(f"Posisi Terbuka    : Entry @ {open_pos['entryPrice']:.4f}, P/L Saat Ini: {pnl:.2f}%", pnl_color)
                    else: print_colored("Posisi Terbuka    : Tidak ada", Fore.WHITE)
                    print()

            elif cmd == '!history': display_history()
            elif cmd in ['!settings', '!set']:
                handle_settings_command(command_parts)
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
                asyncio.run(handle_chat_message(user_input))
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)

    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
