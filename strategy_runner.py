import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
from groq import Groq

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
OKX_API_URL = "https://www.okx.com/api/v5"
MIN_PROFIT_PERCENTAGE = 0.1 # Ambang batas profit di atas biaya trading
DEFAULT_TIMEFRAME = '1H'
ANALYSIS_INTERVAL_SECONDS = 20 # Interval analisis Autopilot
REFRESH_INTERVAL_SECONDS = 5    # Interval refresh data chart

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
current_instrument_id = None
current_candle_data = []
is_ai_thinking = False
is_autopilot_in_cooldown = False
stop_event = threading.Event()
client = None # Klien Groq API

# --- INISIALISASI WARNA ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---

def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def display_welcome_message():
    print_colored("==============================================", Fore.CYAN, Style.BRIGHT)
    print_colored("   Strategic AI Analyst for Termux (v1.0)   ", Fore.CYAN, Style.BRIGHT)
    print_colored("==============================================", Fore.CYAN, Style.BRIGHT)
    print_colored("Data dari OKX, AI ditenagai oleh Groq.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!pair <PAIR> [TIMEFRAME] - Ganti pair dan timeframe (misal: !pair BTC-USDT 15m)", Fore.GREEN)
    print_colored("!status                   - Tampilkan status saat ini", Fore.GREEN)
    print_colored("!history                  - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!exit                     - Keluar dari aplikasi", Fore.GREEN)
    print_colored("Teks apa pun selain itu   - Mengirim pesan ke Analyst AI (chat)", Fore.GREEN)
    print()

# --- MANAJEMEN PENGATURAN & DATA ---

def load_settings():
    global current_settings, client
    default_settings = {
        "groq_api_key": "",
        "take_profit_pct": 1.0,
        "stop_loss_pct": 0.5,
        "min_confidence": 7,
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
    else:
        current_settings = default_settings
        key = input("Masukkan Groq API Key Anda: ")
        current_settings["groq_api_key"] = key
        save_settings()

    if not current_settings.get("groq_api_key"):
        print_colored("ERROR: Groq API Key tidak ditemukan. Harap edit 'settings.json'.", Fore.RED, Style.BRIGHT)
        exit()
        
    client = Groq(api_key=current_settings["groq_api_key"])


def save_settings():
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
        entry_time = datetime.fromisoformat(trade['entryTimestamp'][:-1]).strftime('%Y-%m-%d %H:%M')
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        pl_color = Fore.WHITE
        if trade.get('pl_percent') is not None:
             pl_color = Fore.GREEN if trade['pl_percent'] > MIN_PROFIT_PERCENTAGE else Fore.RED
        
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        print_colored(f"  Alasan Entry: {trade.get('entryReason', 'N/A')}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'][:-1]).strftime('%Y-%m-%d %H:%M')
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {trade.get('pl_percent', 0.0):.2f}%", pl_color, Style.BRIGHT)
            print_colored(f"  Analisis Exit: {trade.get('exitReason', 'N/A')}", Fore.WHITE)
        print()


# --- PENGAMBILAN DATA ---

def fetch_okx_candle_data(instId, timeframe):
    try:
        url = f"{OKX_API_URL}/market/history-candles?instId={instId}&bar={timeframe}&limit=300"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and isinstance(data.get("data"), list):
            formatted_data = [
                {
                    "time": int(d[0]),
                    "open": float(d[1]),
                    "high": float(d[2]),
                    "low": float(d[3]),
                    "close": float(d[4]),
                }
                for d in data["data"]
            ]
            return formatted_data[::-1] # Reverse to get oldest first
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED)
            return []
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error: {e}", Fore.RED)
        return []

# --- LOGIKA INTI AI ---

def get_groq_completion(system_prompt, user_content, model='llama3-70b-8192', is_json=False):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response_format = {"type": "json_object"} if is_json else None
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=700,
            response_format=response_format
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print_colored(f"Groq API Error: {e}", Fore.RED)
        return None

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason):
    print_colored(f"Menganalisis hasil trade {trade['id']}...", Fore.CYAN)
    pnl = ((exit_price - trade['entryPrice']) / trade['entryPrice']) * 100

    outcome = "TRUE PROFIT" if pnl > MIN_PROFIT_PERCENTAGE else "BREAK-EVEN/FEES" if pnl >= 0 else "CLEAR LOSS"
    
    system_prompt = """You are a concise trading analyst. You will receive details of a completed trade. Your task is to provide a brief, insightful, one-sentence analysis of *why* the trade resulted in its outcome. Be brutally honest. Focus on market structure, momentum, or confirmation signals.
Example for Profit: "The entry correctly identified the bounce from support, and the bullish momentum carried the price to the target."
Example for Loss: "The entry was premature, buying into resistance before a confirmed breakout, leading to a reversal."
Start your response directly with the analysis."""
    
    user_content = f"""Analyze the following completed trade for {trade['instrumentId']}:
- Outcome: {outcome} ({pnl:.2f}%)
- Entry Reason: "{trade['entryReason']}"
- Close Reason: {close_trigger_reason}
- Chart Data around Exit:\n{json.dumps(current_candle_data[-50:])}"""

    exit_reason_analysis = get_groq_completion(system_prompt, user_content, model='llama3-8b-8192')
    
    trade['status'] = 'CLOSED'
    trade['exitPrice'] = exit_price
    trade['exitTimestamp'] = datetime.utcnow().isoformat() + "Z"
    trade['pl_percent'] = pnl
    trade['exitReason'] = exit_reason_analysis or f"Auto-closed: {close_trigger_reason}"

    pnl_text = f"PROFIT: +{pnl:.2f}%" if pnl > MIN_PROFIT_PERCENTAGE else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if pnl > MIN_PROFIT_PERCENTAGE else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    print_colored(f"   Analisis: {trade['exitReason']}", Fore.WHITE)
    
    save_trades()

async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or not current_candle_data or is_autopilot_in_cooldown:
        return
    
    is_ai_thinking = True

    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        current_price = current_candle_data[-1]['close']
        
        # 1. Hard Rules Check (TP/SL)
        if open_position:
            pnl = ((current_price - open_position['entryPrice']) / open_position['entryPrice']) * 100
            close_reason = None
            if current_settings.get('take_profit_pct') and pnl >= current_settings['take_profit_pct']:
                close_reason = f"Take Profit @ {current_settings['take_profit_pct']}% tercapai."
            elif current_settings.get('stop_loss_pct') and pnl <= -current_settings['stop_loss_pct']:
                close_reason = f"Stop Loss @ {current_settings['stop_loss_pct']}% tercapai."

            if close_reason:
                await analyze_and_close_trade(open_position, current_price, close_reason)
                is_ai_thinking = False
                return

        # 2. AI Decision Making
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Autopilot menganalisis {current_instrument_id}...", Fore.MAGENTA)
        
        past_trades = [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'CLOSED'][-3:]
        learning_context = "There is no trading history for this pair yet."
        if past_trades:
            learning_context = "Here is the analysis of your last few trades. Learn from mistakes and successes.\n"
            for pt in past_trades:
                outcome = "PROFIT" if pt['pl_percent'] > MIN_PROFIT_PERCENTAGE else "LOSS"
                learning_context += f"- Trade ({outcome}):\n  - Entry Reason: \"{pt['entryReason']}\"\n  - Exit Analysis: \"{pt['exitReason']}\"\n"

        system_prompt = f"""You are a strategic, data-driven, and self-correcting crypto trading bot. Your goal is to identify only high-quality trades.

**MANDATORY THREE-STEP ANALYSIS PROCESS:**

**STEP 1: LEARN FROM PAST TRADES**
Review the provided trade history.
{learning_context}
- **CRITICAL RULE:** If the current market strongly resembles a previous losing trade, you MUST choose "HOLD".

**STEP 2: ANALYZE CURRENT MARKET**
Analyze the provided real-time candlestick data. Identify trend, support, resistance, and momentum.

**STEP 3: SYNTHESIZE AND EXECUTE**
Combine historical learnings with real-time analysis.
- You are ONLY PERMITTED to issue a "BUY" command if the setup does NOT repeat a past mistake AND there's a clear path to profit.
- If a position is open, your valid actions are "CLOSE" or "HOLD".
- Your response MUST be a valid JSON: {{"action": "...", "reason": "...", "confidence": ... (only for BUY)}}."""

        user_content = f"Instrument: {current_instrument_id}, Timeframe: {DEFAULT_TIMEFRAME}, Current Price: {current_price}\n"
        if open_position:
            pnl = ((current_price - open_position['entryPrice']) / open_position['entryPrice']) * 100
            user_content += f"CURRENT OPEN POSITION: LONG from {open_position['entryPrice']}. Current P/L: {pnl:.2f}%. Analyze if momentum is strong enough to continue or if it's better to close.\n"
        else:
            user_content += "CURRENT OPEN POSITION: None.\n"
        user_content += f"LATEST CANDLESTICK DATA (75 candles):\n{json.dumps(current_candle_data[-75:])}"

        ai_response_str = get_groq_completion(system_prompt, user_content, is_json=True)
        if not ai_response_str:
            raise Exception("AI response was empty.")
        
        ai_response = json.loads(ai_response_str)
        action = ai_response.get('action', 'HOLD').upper()
        reason = ai_response.get('reason', 'No reason provided.')
        confidence = ai_response.get('confidence', 0)

        if action == "BUY" and not open_position:
            if confidence >= current_settings.get("min_confidence", 1):
                new_trade = {
                    "id": int(time.time()), "instrumentId": current_instrument_id, "type": 'LONG',
                    "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price,
                    "entryReason": reason, "status": 'OPEN',
                }
                autopilot_trades.append(new_trade)
                print_colored(f"\n🟢 ACTION: BUY {current_instrument_id} @ {current_price}", Fore.GREEN, Style.BRIGHT)
                print_colored(f"   Confidence: {confidence}/10 | Reason: {reason}", Fore.WHITE)
                save_trades()
            else:
                print_colored(f"⚪️ HOLD: Sinyal BUY terdeteksi (Conf: {confidence}) tetapi di bawah ambang batas ({current_settings['min_confidence']}).", Fore.YELLOW)

        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"AI Decision: {reason}")
        
        else: # HOLD
            print_colored(f"⚪️ HOLD: {reason}", Fore.CYAN)

    except Exception as e:
        print_colored(f"Autopilot Error: {e}. Cooldown 5 detik...", Fore.RED)
        is_autopilot_in_cooldown = True
        time.sleep(5)
        is_autopilot_in_cooldown = False
    finally:
        is_ai_thinking = False

async def handle_chat_message(user_text):
    global is_ai_thinking
    if is_ai_thinking:
        print_colored("AI sedang sibuk, harap tunggu...", Fore.YELLOW)
        return
    if not current_instrument_id or not current_candle_data:
        print_colored("Harap pilih pair terlebih dahulu dengan '!pair'.", Fore.YELLOW)
        return

    is_ai_thinking = True
    print_colored("\nAnalyst AI sedang berpikir...", Fore.MAGENTA)

    try:
        system_prompt = """You are "ChartWise", an expert crypto chart analyst. Your personality is professional and insightful. Use the provided context to answer accurately. Focus on price action (support, resistance, trends). Do not give financial advice. Start your response directly."""
        
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        user_content = f"Candlestick Data (Pair: {current_instrument_id}, Timeframe: {DEFAULT_TIMEFRAME}):\n{json.dumps(current_candle_data[-75:])}\n\n"
        if open_position:
            pnl = ((current_candle_data[-1]['close'] - open_position['entryPrice']) / open_position['entryPrice']) * 100
            user_content += f"CONTEXT: Autopilot has an OPEN LONG position, entered at {open_position['entryPrice']} because \"{open_position['entryReason']}\". Current P/L is {pnl:.2f}%.\n\n"
        else:
            user_content += "CONTEXT: Autopilot has NO open positions for this pair.\n\n"
        user_content += f"User Question: \"{user_text}\""

        ai_response = get_groq_completion(system_prompt, user_content)
        if ai_response:
            print_colored("\n--- ChartWise AI Analyst ---", Fore.CYAN, Style.BRIGHT)
            print(ai_response)
            print_colored("--------------------------", Fore.CYAN, Style.BRIGHT)

    except Exception as e:
        print_colored(f"Chat Error: {e}", Fore.RED)
    finally:
        is_ai_thinking = False

# --- THREAD WORKERS ---
import asyncio

def autopilot_worker():
    while not stop_event.is_set():
        asyncio.run(run_autopilot_analysis())
        stop_event.wait(ANALYSIS_INTERVAL_SECONDS)

def data_refresh_worker():
    global current_candle_data
    while not stop_event.is_set():
        if current_instrument_id:
            # print_colored(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing data for {current_instrument_id}...", Fore.BLUE)
            data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
            if data:
                current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)


# --- FUNGSI UTAMA ---

def main():
    global current_instrument_id, DEFAULT_TIMEFRAME, current_candle_data
    
    load_settings()
    load_trades()
    display_welcome_message()

    # Mulai thread di latar belakang
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    autopilot_thread.start()
    
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    data_thread.start()

    while True:
        try:
            prompt_text = f"[{current_instrument_id or 'No Pair'}] > "
            user_input = input(prompt_text)

            if user_input.lower() == '!exit':
                print_colored("Menutup aplikasi...", Fore.YELLOW)
                stop_event.set()
                break
            
            elif user_input.lower() == '!help':
                display_help()

            elif user_input.lower() == '!status':
                if not current_instrument_id:
                    print_colored("Belum ada pair yang dipilih. Gunakan '!pair NAMA-PAIR'.", Fore.YELLOW)
                else:
                    price = current_candle_data[-1]['close'] if current_candle_data else 'N/A'
                    print_colored(f"\n--- Status Saat Ini ---", Fore.CYAN, Style.BRIGHT)
                    print_colored(f"Pair      : {current_instrument_id}", Fore.WHITE)
                    print_colored(f"Timeframe : {DEFAULT_TIMEFRAME}", Fore.WHITE)
                    print_colored(f"Harga     : {price}", Fore.WHITE)
                    open_pos = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
                    if open_pos:
                        pnl = ((price - open_pos['entryPrice']) / open_pos['entryPrice']) * 100
                        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                        print_colored("Posisi Terbuka: ", Fore.YELLOW, Style.BRIGHT)
                        print_colored(f"  Entry @ {open_pos['entryPrice']:.4f}", Fore.WHITE)
                        print_colored(f"  P/L Saat Ini: {pnl:.2f}%", pnl_color)
                    else:
                        print_colored("Posisi Terbuka: Tidak ada", Fore.WHITE)
                    print()
            
            elif user_input.lower() == '!history':
                display_history()

            elif user_input.lower().startswith('!pair '):
                parts = user_input.split()
                if len(parts) >= 2:
                    current_instrument_id = parts[1].upper()
                    DEFAULT_TIMEFRAME = parts[2] if len(parts) > 2 else '1H'
                    print_colored(f"Mengganti pair ke {current_instrument_id} dengan timeframe {DEFAULT_TIMEFRAME}. Memuat data...", Fore.CYAN)
                    current_candle_data = fetch_okx_candle_data(current_instrument_id, DEFAULT_TIMEFRAME)
                    if current_candle_data:
                        print_colored("Data berhasil dimuat.", Fore.GREEN)
                    else:
                        print_colored("Gagal memuat data.", Fore.RED)
                else:
                    print_colored("Format salah. Gunakan: !pair NAMA-PAIR [TIMEFRAME]", Fore.RED)

            else: # Dianggap sebagai chat
                asyncio.run(handle_chat_message(user_input))

        except KeyboardInterrupt:
            print_colored("\nMenutup aplikasi (Ctrl+C)...", Fore.YELLOW)
            stop_event.set()
            break
        except Exception as e:
            print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)

    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)


if __name__ == "__main__":
    main()
