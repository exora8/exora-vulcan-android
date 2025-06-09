import json
import os
import time
import threading
import requests
from datetime import datetime
from colorama import init, Fore, Style
import asyncio
import math

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
    print_colored("     Strategic AI Analyst (God-Tier Local AI)     ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI ini menggunakan Volume, Bollinger Bands, Pivot Points, EMA & RSI.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot AI", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot AI", Fore.GREEN)
    print_colored("!pair <PAIR> [TF]   - Ganti pair dan timeframe", Fore.GREEN)
    print_colored("!status               - Tampilkan status dan analisis teknikal singkat", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan saat ini", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (contoh: !set tp 1.5)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings, current_instrument_id
    default_settings = {"take_profit_pct": 1.5, "stop_loss_pct": 0.8, "fee_pct": 0.1, "analysis_interval_sec": 30, "last_pair": None, "last_timeframe": "1H"}
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
            # OKX volume ada di index 5 ('volCcy')
            return [{"time": int(d[0]),"open": float(d[1]),"high": float(d[2]),"low": float(d[3]),"close": float(d[4]), "volume": float(d[5])} for d in data["data"]][::-1]
        else:
            print_colored(f"OKX API Error: {data.get('msg', 'Data tidak valid')}", Fore.RED); return []
    except requests.exceptions.RequestException as e:
        print_colored(f"Network Error saat fetch data OKX: {e}", Fore.RED); return []

# --- OTAK LOCAL AI (DENGAN SEMUA LOGIKA BARU) ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair):
        self.settings = settings
        self.past_trades = past_trades_for_pair

    # --- Kalkulator Indikator ---
    def calculate_ema(self, data, period):
        if len(data) < period: return None
        closes = [d['close'] for d in data]
        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, data, period=14):
        if len(data) <= period: return 50
        closes = [d['close'] for d in data]
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

    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        if len(data) < period: return None
        closes = [d['close'] for d in data[-period:]]
        sma = sum(closes) / period
        variance = sum([(price - sma) ** 2 for price in closes]) / period
        stdev = math.sqrt(variance)
        upper = sma + std_dev * stdev
        lower = sma - std_dev * stdev
        return {"middle": sma, "upper": upper, "lower": lower, "width": (upper - lower) / sma}

    def calculate_lookback_pivots(self, data, period=100):
        if len(data) < period: return None
        relevant_data = data[-period:]
        high = max(d['high'] for d in relevant_data)
        low = min(d['low'] for d in relevant_data)
        close = relevant_data[-1]['close']
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        return {"p": pivot, "s1": s1, "r1": r1}

    # --- Analisis Pasar Komprehensif ---
    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100: return None
        
        analysis = {
            "ema9": self.calculate_ema(candle_data, 9),
            "ema50": self.calculate_ema(candle_data, 50),
            "ema100": self.calculate_ema(candle_data, 100),
            "rsi": self.calculate_rsi(candle_data, 14),
            "bb": self.calculate_bollinger_bands(candle_data, 20),
            "pivots": self.calculate_lookback_pivots(candle_data, 100),
            "avg_volume": sum(d['volume'] for d in candle_data[-20:]) / 20
        }
        
        # Tentukan Bias Tren berdasarkan EMA
        bias = "RANGING"
        if analysis["ema50"] > analysis["ema100"]: bias = "BULLISH"
        elif analysis["ema50"] < analysis["ema100"]: bias = "BEARISH"
        analysis["bias"] = bias
        
        # Tentukan Kondisi Volatilitas
        volatility = "NORMAL"
        if analysis["bb"]["width"] < 0.03: volatility = "SQUEEZE" # Lebar BB < 3% dari harga
        analysis["volatility"] = volatility

        return analysis

    # --- Sistem Belajar yang Diperkuat ---
    def check_for_repeated_mistake(self, current_analysis, trade_type):
        losing_trades = [t for t in self.past_trades if t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        if not losing_trades: return False
        
        for loss in losing_trades[-3:]:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or loss.get("type") != trade_type: continue
            
            # Kondisi kemiripan
            bias_same = current_analysis['bias'] == past_snapshot.get('bias')
            volatility_same = current_analysis['volatility'] == past_snapshot.get('volatility')
            rsi_similar = abs(current_analysis['rsi'] - past_snapshot.get('rsi', 50)) < 15
            
            # Jika semua kondisi mirip, ini adalah pengulangan kesalahan!
            if bias_same and volatility_same and rsi_similar:
                # Logika Bobot: Jika loss sebelumnya parah, jangan ambil risiko.
                if loss.get('pl_percent', 0) < -2.0: # Jika loss lebih dari 2%
                    print_colored(f"[LEARNING] Menolak keras! Mirip dengan loss fatal #{loss['id']}", Fore.RED, Style.BRIGHT)
                    return True
                print_colored(f"[LEARNING] Menghindari posisi karena mirip dengan loss trade #{loss['id']}", Fore.MAGENTA)
                return True
        return False

    # --- Logika Keputusan Inti ---
    def get_decision(self, candle_data, open_position):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data tidak cukup untuk analisis."}
        
        current_price = candle_data[-1]['close']
        
        if open_position: # Logika untuk menutup posisi
            pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type'))
            trade_type = open_position.get('type')
            if trade_type == 'LONG' and (current_price < analysis['ema9'] or current_price > analysis['bb']['upper']):
                return {"action": "CLOSE", "reason": "Sinyal exit (cross EMA9 / sentuh Upper BB)."}
            if trade_type == 'SHORT' and (current_price > analysis['ema9'] or current_price < analysis['bb']['lower']):
                return {"action": "CLOSE", "reason": "Sinyal exit (cross EMA9 / sentuh Lower BB)."}
            return {"action": "HOLD", "reason": f"Holding {trade_type}, P/L: {pnl:.2f}%"}

        # --- Logika Entry "God-Tier" ---
        
        # 1. Sabar saat Volatilitas Rendah (Squeeze)
        if analysis['volatility'] == 'SQUEEZE':
            return {"action": "HOLD", "reason": "Bollinger Bands Squeeze, menunggu breakout."}
        
        # 2. Periksa Kondisi Entry LONG
        if analysis['bias'] == 'BULLISH':
            is_near_support = current_price < analysis['bb']['middle'] or current_price < analysis['pivots']['p']
            is_volume_confirmed = candle_data[-1]['volume'] > analysis['avg_volume'] * 1.2
            is_not_overbought = analysis['rsi'] < 70
            
            if is_near_support and is_volume_confirmed and is_not_overbought:
                if not self.check_for_repeated_mistake(analysis, "LONG"):
                    return {"action": "BUY", "reason": f"Bullish, pullback ke support dinamis dengan konfirmasi volume.", "snapshot": analysis}
        
        # 3. Periksa Kondisi Entry SHORT
        if analysis['bias'] == 'BEARISH':
            is_near_resistance = current_price > analysis['bb']['middle'] or current_price > analysis['pivots']['p']
            is_volume_confirmed = candle_data[-1]['volume'] > analysis['avg_volume'] * 1.2
            is_not_oversold = analysis['rsi'] > 30

            if is_near_resistance and is_volume_confirmed and is_not_oversold:
                if not self.check_for_repeated_mistake(analysis, "SHORT"):
                    return {"action": "SELL", "reason": f"Bearish, rally ke resistance dinamis dengan konfirmasi volume.", "snapshot": analysis}
            
        return {"action": "HOLD", "reason": f"Menunggu setup presisi. Bias: {analysis['bias']}, Vol: {analysis['volatility']}."}


# --- LOGIKA TRADING UTAMA ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, entry_snapshot=None):
    global is_ai_thinking
    is_ai_thinking = True
    print_colored(f"\nMenutup trade {trade['id']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
    if not is_profit and entry_snapshot:
        trade['entry_snapshot'] = entry_snapshot
        print_colored(f"   [LEARNING] Menyimpan snapshot kegagalan trade #{trade['id']}", Fore.MAGENTA)
    pnl_text = f"PROFIT: +{pnl:.2f}%" if is_profit else f"LOSS: {pnl:.2f}%"
    pnl_color = Fore.GREEN if is_profit else Fore.RED
    print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
    # Menampilkan run-up
    run_up = trade.get('run_up_percent', pnl)
    print_colored(f"   Profit Tertinggi (Run-up): {run_up:.2f}%", Fore.YELLOW)
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)
    is_ai_thinking = False

async def run_autopilot_analysis():
    global is_ai_thinking, is_autopilot_in_cooldown
    if is_ai_thinking or not current_instrument_id or not current_candle_data or is_autopilot_in_cooldown: return
    is_ai_thinking = True
    try:
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
        print_colored(f"\n[{datetime.now().strftime('%H:%M:%S')}] Local AI sedang menganalisis {current_instrument_id}...", Fore.MAGENTA)
        local_brain = LocalAI(current_settings, [t for t in autopilot_trades if t['instrumentId'] == current_instrument_id])
        decision = local_brain.get_decision(current_candle_data, open_position)
        action = decision.get('action', 'HOLD').upper()
        reason = decision.get('reason', 'No reason provided.')
        current_price = current_candle_data[-1]['close']
        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {"id": int(time.time()), "instrumentId": current_instrument_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN', "entry_snapshot": decision.get("snapshot"), "run_up_percent": 0.0}
            autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if action == "BUY" else Fore.RED
            print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {current_instrument_id} @ {current_price}", action_color, Style.BRIGHT)
            print_colored(f"   Reason: {reason}", Fore.WHITE)
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
            notif_content = f"Entry @ {current_price:.4f} | Alasan: {reason}"
            send_termux_notification(notif_title, notif_content)
        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"Local AI Decision: {reason}", open_position.get("entry_snapshot"))
        else:
            print_colored(f"⚪️ HOLD: {reason}", Fore.CYAN)
    except Exception as e:
        print_colored(f"Autopilot Error: {e}. Cooldown 5 detik...", Fore.RED)
        is_autopilot_in_cooldown = True; await asyncio.sleep(5); is_autopilot_in_cooldown = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS & MAIN LOOP ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            asyncio.run(run_autopilot_analysis())
            stop_event.wait(current_settings.get("analysis_interval_sec", 30))
        else: time.sleep(1)

async def check_realtime_tp_sl_and_runup(latest_price):
    global is_ai_thinking
    if is_ai_thinking or not is_autopilot_running: return
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    
    # Update Run-up
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
        
    # Check TP/SL
    tp_pct = current_settings.get('take_profit_pct'); sl_pct = current_settings.get('stop_loss_pct')
    close_reason = None
    if tp_pct and current_pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
    elif sl_pct and current_pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
    if close_reason:
        await analyze_and_close_trade(open_position, latest_price, close_reason, open_position.get("entry_snapshot"))

def data_refresh_worker():
    global current_candle_data
    while not stop_event.is_set():
        if current_instrument_id:
            data = fetch_okx_candle_data(current_instrument_id, current_settings.get('last_timeframe', '1H'))
            if data: 
                current_candle_data = data
                latest_price = data[-1]['close']
                asyncio.run(check_realtime_tp_sl_and_runup(latest_price))
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

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
                    print_colored(f"Pair              : {current_instrument_id}, TF: {current_settings['last_timeframe']}", Fore.WHITE)
                    print_colored(f"Harga Terkini     : {price}", Fore.WHITE)
                    open_pos = next((t for t in autopilot_trades if t['instrumentId'] == current_instrument_id and t['status'] == 'OPEN'), None)
                    if open_pos and isinstance(price, float):
                        pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
                        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                        type_color = Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED
                        print_colored(f"Posisi Terbuka    : ", Fore.WHITE, end="")
                        print_colored(f"{open_pos.get('type')} ", type_color, Style.BRIGHT, end="")
                        print_colored(f"Entry @ {open_pos['entryPrice']:.4f}, P/L: {pnl:.2f}%", pnl_color)
                        run_up = open_pos.get('run_up_percent', 0.0)
                        print_colored(f"  Profit Tertinggi Sejauh Ini (Run-up): {run_up:.2f}%", Fore.YELLOW)
                    else: print_colored("Posisi Terbuka    : Tidak ada", Fore.WHITE)
                    print()
            elif cmd == '!history':
                # ...
                pass
            elif cmd in ['!settings', '!set']:
                # ...
                pass
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
                print_colored("Fungsi chat tidak tersedia di versi Local AI.", Fore.YELLOW)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
