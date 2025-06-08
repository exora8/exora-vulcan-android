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
REFRESH_INTERVAL_SECONDS = 5

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
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL):
    print(bright + color + text)

def send_termux_notification(title, content):
    if not IS_TERMUX: return
    try:
        safe_title = title.replace('"', "'"); safe_content = content.replace('"', "'")
        command = f'termux-notification --title "{safe_title}" --content "{safe_content}"'
        os.system(command)
    except Exception as e: print_colored(f"Gagal mengirim notifikasi: {e}", Fore.RED)

def display_welcome_message():
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("      Strategic AI Analyst (Local AI Edition)     ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini berjalan 100% lokal, tanpa API eksternal.", Fore.YELLOW)
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
        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "fee_pct": 0.1,
        "analysis_interval_sec": 30, "min_confidence_trades": 3,
        "last_pair": None, "last_timeframe": "1H",
        "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
        "sma_short_period": 20, "sma_long_period": 50
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else: current_settings = default_settings; save_settings()
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
    # ... Fungsi ini tidak berubah ...
    pass

# --- PERHITUNGAN INDIKATOR LOKAL (TANPA PANDAS) ---
def calculate_sma(data_points, period):
    if len(data_points) < period: return None
    return sum(data_points[-period:]) / period

def calculate_rsi(close_prices, period=14):
    if len(close_prices) <= period: return None
    gains, losses = [], []
    for i in range(1, len(close_prices)):
        change = close_prices[i] - close_prices[i-1]
        if change > 0: gains.append(change); losses.append(0)
        else: gains.append(0); losses.append(abs(change))
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(candle_data):
    close_prices = [c['close'] for c in candle_data]
    if not close_prices: return {}
    
    sma_short = calculate_sma(close_prices, current_settings.get('sma_short_period', 20))
    sma_long = calculate_sma(close_prices, current_settings.get('sma_long_period', 50))
    rsi = calculate_rsi(close_prices, current_settings.get('rsi_period', 14))

    trend = 'RANGING'
    if sma_short and sma_long:
        if sma_short > sma_long: trend = 'BULLISH'
        else: trend = 'BEARISH'
        
    return {
        "rsi": rsi,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "trend": trend
    }

# --- LOGIKA INTI AI ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

# DIUBAH: Fungsi ini tidak lagi memanggil AI, hanya mencatat
def close_trade_and_log(trade, exit_price, close_trigger_reason):
    print_colored(f"\nMenutup trade {trade['id']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    
    trade.update({
        'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z",
        'pl_percent': pnl, 'exitReason': close_trigger_reason, 'is_profit': is_profit
    })
    
    pnl_text = f"PROFIT: +{pnl:.2f}%" if is_profit else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if is_profit else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    print_colored(f"   Alasan Tutup: {close_trigger_reason}", Fore.WHITE)
    
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)

# DIUBAH TOTAL: Ini sekarang adalah inti dari Local AI
async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or len(current_candle_data) < current_settings['sma_long_period']: return
    is_ai_thinking = True
    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        current_price = current_candle_data[-1]['close']
        
        # 1. Manajemen Posisi Terbuka (TP/SL/Exit)
        if open_position:
            pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type', 'LONG'))
            tp_pct = current_settings.get('take_profit_pct'); sl_pct = current_settings.get('stop_loss_pct')
            close_reason = None
            if tp_pct and pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
            elif sl_pct and pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
            
            # Aturan Exit berdasarkan kondisi pasar
            current_indicators = calculate_indicators(current_candle_data)
            trade_type = open_position.get('type')
            if trade_type == 'LONG' and current_indicators.get('trend') == 'BEARISH':
                close_reason = "Exit: Tren berbalik menjadi Bearish (Death Cross)."
            elif trade_type == 'SHORT' and current_indicators.get('trend') == 'BULLISH':
                close_reason = "Exit: Tren berbalik menjadi Bullish (Golden Cross)."

            if close_reason:
                close_trade_and_log(open_position, current_price, close_reason); is_ai_thinking = False; return

        # 2. Analisis untuk Entry Baru (jika tidak ada posisi terbuka)
        else:
            print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI sedang menganalisis {current_instrument_id}...", Fore.MAGENTA)
            
            # Hitung kondisi pasar saat ini
            indicators = calculate_indicators(current_candle_data)
            if not all(k in indicators for k in ['rsi', 'trend']):
                print_colored("Data indikator tidak lengkap, skip analisis.", Fore.YELLOW)
                is_ai_thinking = False; return

            rsi = indicators['rsi']
            trend = indicators['trend']
            
            # --- FASE PEMBELAJARAN (STATISTICAL LOOKBACK) ---
            past_trades_for_pair = [t for t in autopilot_trades if t.get('instrumentId') == current_instrument_id and t.get('status') == 'CLOSED']
            
            oversold_trades = [t for t in past_trades_for_pair if t.get('entry_indicators', {}).get('rsi_condition') == 'OVERSOLD']
            overbought_trades = [t for t in past_trades_for_pair if t.get('entry_indicators', {}).get('rsi_condition') == 'OVERBOUGHT']
            
            oversold_wins = sum(1 for t in oversold_trades if t.get('is_profit'))
            overbought_wins = sum(1 for t in overbought_trades if t.get('is_profit'))

            # Hitung win rate, hindari pembagian dengan nol
            min_trades = current_settings.get('min_confidence_trades', 3)
            oversold_win_rate = (oversold_wins / len(oversold_trades)) if len(oversold_trades) >= min_trades else -1
            overbought_win_rate = (overbought_wins / len(overbought_trades)) if len(overbought_trades) >= min_trades else -1
            
            # --- FASE PENGAMBILAN KEPUTUSAN (RULES-BASED) ---
            action = 'HOLD'
            reason = 'Tidak ada setup sniper yang terdeteksi.'
            entry_indicators_to_log = {}

            # Aturan untuk LONG (BUY)
            if trend == 'BULLISH' and rsi < current_settings.get('rsi_oversold', 30):
                if oversold_win_rate == -1 or oversold_win_rate > 0.5: # Jika tidak ada data atau win rate > 50%
                    action = 'BUY'
                    reason = f"Setup LONG: RSI Oversold ({rsi:.1f}) dalam tren Bullish. Win rate historis: {'N/A' if oversold_win_rate == -1 else f'{oversold_win_rate:.0%}'}"
                    entry_indicators_to_log = {'rsi_condition': 'OVERSOLD', 'trend_condition': trend}
            
            # Aturan untuk SHORT (SELL)
            elif trend == 'BEARISH' and rsi > current_settings.get('rsi_overbought', 70):
                 if overbought_win_rate == -1 or overbought_win_rate > 0.5: # Jika tidak ada data atau win rate > 50%
                    action = 'SELL'
                    reason = f"Setup SHORT: RSI Overbought ({rsi:.1f}) dalam tren Bearish. Win rate historis: {'N/A' if overbought_win_rate == -1 else f'{overbought_win_rate:.0%}'}"
                    entry_indicators_to_log = {'rsi_condition': 'OVERBOUGHT', 'trend_condition': trend}

            # Eksekusi Aksi
            if action in ['BUY', 'SELL']:
                trade_type = "LONG" if action == "BUY" else "SHORT"
                new_trade = {
                    "id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, 
                    "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, 
                    "entryReason": reason, "status": 'OPEN',
                    "entry_indicators": entry_indicators_to_log # Simpan kondisi saat entry untuk pembelajaran
                }
                autopilot_trades.append(new_trade)
                action_color = Fore.GREEN if action == "BUY" else Fore.RED
                print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
                print_colored(f"   Reason: {reason}", Fore.WHITE)
                save_trades()
                notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka: {current_instrument_id}"
                notif_content = f"Entry pada harga {current_price:.4f}."
                send_termux_notification(notif_title, notif_content)
            else:
                print_colored(f"⚪️ HOLD: {reason}", Fore.CYAN)

    except Exception as e:
        print_colored(f"Autopilot Error: {e}. Cooldown 5 detik...", Fore.RED)
        is_autopilot_in_cooldown = True
        await asyncio.sleep(5)
        is_autopilot_in_cooldown = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            asyncio.run(run_autopilot_analysis())
            current_delay = current_settings.get("analysis_interval_sec", 30)
            stop_event.wait(current_delay)
        else: time.sleep(1)

def data_refresh_worker():
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: current_candle_data = data
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

# Fungsi lain (main, handle_settings, dll.) tetap sama
def handle_settings_command(parts):
    # ... (tidak ada perubahan di sini)
    pass

def main():
    global current_instrument_id, current_candle_data, is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    # ... Sisa fungsi main tetap sama ...
    pass
# Dummy main loop untuk kelengkapan
if __name__ == "__main__":
    main()
