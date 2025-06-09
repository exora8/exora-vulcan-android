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
BYBIT_API_URL = "https://api.bybit.com/v5/market"
# DIUBAH: Interval refresh sekarang menjadi 1 detik untuk pengalaman real-time
REFRESH_INTERVAL_SECONDS = 1

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
monitored_pairs = {}
data_lock = threading.Lock()
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
is_ai_thinking = False
is_autopilot_in_cooldown = False

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
    print_colored("    Strategic AI Analyst (Real-time Dashboard)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored(f"Dashboard & PnL diperbarui setiap {REFRESH_INTERVAL_SECONDS} detik.", Fore.YELLOW)
    print_colored("PERINGATAN: Jangan memantau terlalu banyak pair untuk menghindari limit API.", Fore.RED)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Mengaktifkan Autopilot AI & Live Dashboard", Fore.GREEN)
    print_colored("!stop                 - Menonaktifkan Autopilot AI & Live Dashboard", Fore.GREEN)
    print_colored("!add <PAIR> [TF]      - Tambah pair ke pantauan", Fore.GREEN)
    print_colored("!remove <PAIR>        - Hapus pair dari pantauan", Fore.GREEN)
    print_colored("!status               - Tampilkan status semua pair saat ini", Fore.GREEN)
    print_colored("!history [PAIR]       - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {"take_profit_pct": 1.5, "stop_loss_pct": 0.8, "fee_pct": 0.1, "analysis_interval_sec": 30, "monitored_pairs_list": []}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else:
        current_settings = default_settings; save_settings()

def save_settings():
    with data_lock:
        current_settings["monitored_pairs_list"] = list(monitored_pairs.keys())
    with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)

def save_trades():
    with data_lock:
        with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)

# --- FUNGSI API (BYBIT) ---
def fetch_bybit_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in data['result']['list']][::-1]
        else: return []
    except requests.exceptions.RequestException: return []
    except (KeyError, IndexError): return []

# --- OTAK LOCAL AI ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair): self.settings = settings; self.past_trades = past_trades_for_pair
    def calculate_ema(self, data, period):
        if len(data) < period: return None
        closes = [d['close'] for d in data]; multiplier = 2 / (period + 1); ema = sum(closes[:period]) / period
        for price in closes[period:]: ema = (price - ema) * multiplier + ema
        return ema
    def calculate_rsi(self, data, period=14):
        if len(data) <= period: return 50
        closes = [d['close'] for d in data]; gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0: gains.append(change); losses.append(0)
            else: losses.append(abs(change)); gains.append(0)
        avg_gain = sum(gains[-period:]) / period; avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100
        rs = avg_gain / avg_loss; return 100 - (100 / (1 + rs))
    def calculate_lookback_pivots(self, data, period=100):
        if len(data) < period: return None
        relevant_data = data[-period:]; high = max(d['high'] for d in relevant_data); low = min(d['low'] for d in relevant_data); close = relevant_data[-1]['close']
        pivot = (high + low + close) / 3; s1 = (2 * pivot) - high; r1 = (2 * pivot) - low
        return {"p": pivot, "s1": s1, "r1": r1}
    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100: return None
        analysis = {"ema9": self.calculate_ema(candle_data, 9), "ema50": self.calculate_ema(candle_data, 50), "ema100": self.calculate_ema(candle_data, 100), "rsi": self.calculate_rsi(candle_data, 14), "pivots": self.calculate_lookback_pivots(candle_data, 100)}
        bias = "RANGING";
        if analysis["ema50"] > analysis["ema100"]: bias = "BULLISH"
        elif analysis["ema50"] < analysis["ema100"]: bias = "BEARISH"
        analysis["bias"] = bias; return analysis
    def check_for_repeated_mistake(self, current_analysis, trade_type):
        losing_trades = [t for t in self.past_trades if t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        if not losing_trades: return False
        for loss in losing_trades[-3:]:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or loss.get("type") != trade_type: continue
            bias_same = current_analysis['bias'] == past_snapshot.get('bias')
            rsi_similar = abs(current_analysis['rsi'] - past_snapshot.get('rsi', 50)) < 15
            if bias_same and rsi_similar: print_colored(f"[LEARNING] Menghindari posisi {loss.get('instrumentId')} karena mirip dengan loss trade #{loss['id']}", Fore.MAGENTA); return True
        return False
    def get_decision(self, candle_data, open_position):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data tidak cukup untuk analisis."}
        current_price = candle_data[-1]['close']
        if open_position:
            pnl = calculate_pnl(open_position['entryPrice'], current_price, open_position.get('type'))
            trade_type = open_position.get('type')
            if trade_type == 'LONG' and current_price < analysis['ema9']: return {"action": "CLOSE", "reason": f"Harga cross ke bawah EMA9, sinyal exit."}
            if trade_type == 'SHORT' and current_price > analysis['ema9']: return {"action": "CLOSE", "reason": f"Harga cross ke atas EMA9, sinyal exit."}
            return {"action": "HOLD", "reason": f"Holding {trade_type}, P/L: {pnl:.2f}%"}
        if self.check_for_repeated_mistake(analysis, "LONG"): return {"action": "HOLD", "reason": "Menghindari pengulangan kesalahan masa lalu."}
        if analysis['bias'] == 'BULLISH' and current_price < analysis['pivots']['p'] and analysis['rsi'] < 70:
            return {"action": "BUY", "reason": f"Tren Bullish & pullback ke area Pivot. RSI: {analysis['rsi']:.0f}", "snapshot": analysis}
        if self.check_for_repeated_mistake(analysis, "SHORT"): return {"action": "HOLD", "reason": "Menghindari pengulangan kesalahan masa lalu."}
        if analysis['bias'] == 'BEARISH' and current_price > analysis['pivots']['p'] and analysis['rsi'] > 30:
            return {"action": "SELL", "reason": f"Tren Bearish & rally ke area Pivot. RSI: {analysis['rsi']:.0f}", "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup presisi. Bias: {analysis['bias']}, RSI: {analysis['rsi']:.0f}."}

# --- LOGIKA TRADING UTAMA ---
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, entry_snapshot=None):
    with data_lock:
        if trade.get('status') != 'OPEN': return
        trade['status'] = 'CLOSING'
    print_colored(f"\nMenutup trade {trade['id']} untuk {trade['instrumentId']}...", Fore.CYAN)
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1); is_profit = pnl > fee
    with data_lock:
        trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
        if not is_profit and entry_snapshot:
            trade['entry_snapshot'] = entry_snapshot
            print_colored(f"   [LEARNING] Menyimpan snapshot kegagalan untuk {trade['instrumentId']} #{trade['id']}", Fore.MAGENTA)
        pnl_text = f"PROFIT: +{pnl:.2f}%" if is_profit else f"LOSS: {pnl:.2f}%"
        pnl_color = Fore.GREEN if is_profit else Fore.RED
        print_colored(f"\n🔴 TRADE CLOSED: {pnl_text}", pnl_color, Style.BRIGHT)
        run_up = trade.get('run_up_percent', pnl)
        print_colored(f"   Profit Tertinggi (Run-up): {run_up:.2f}%", Fore.YELLOW)
        save_trades()
        notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
        notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
        send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(pair_id):
    global is_ai_thinking, is_autopilot_in_cooldown
    with data_lock:
        if is_ai_thinking or is_autopilot_in_cooldown: return
        is_ai_thinking = True
    try:
        with data_lock:
            pair_data = monitored_pairs.get(pair_id)
            if not pair_data or not pair_data.get('candle_data'): return
            candle_data = pair_data['candle_data']
            open_position = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
        local_brain = LocalAI(current_settings, [t for t in autopilot_trades if t['instrumentId'] == pair_id])
        decision = local_brain.get_decision(candle_data, open_position)
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close']
        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {"id": int(time.time()), "instrumentId": pair_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN', "entry_snapshot": decision.get("snapshot"), "run_up_percent": 0.0}
            with data_lock: autopilot_trades.append(new_trade)
            action_color = Fore.GREEN if action == "BUY" else Fore.RED
            print_colored(f"\n{'🟢' if action == 'BUY' else '🔴'} ACTION: {action} {pair_id} @ {current_price}", action_color, Style.BRIGHT)
            print_colored(f"   Reason: {reason}", Fore.WHITE)
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka: {pair_id}"
            notif_content = f"Entry @ {current_price:.4f} | Alasan: {reason}"
            send_termux_notification(notif_title, notif_content)
        elif action == "CLOSE" and open_position:
            await analyze_and_close_trade(open_position, current_price, f"Local AI Decision: {reason}", open_position.get("entry_snapshot"))
    except Exception as e:
        print_colored(f"Autopilot Error pada {pair_id}: {e}", Fore.RED)
        is_autopilot_in_cooldown = True; await asyncio.sleep(5); is_autopilot_in_cooldown = False
    finally:
        with data_lock: is_ai_thinking = False

# --- THREAD WORKERS & MAIN LOOP ---
def print_live_dashboard():
    with data_lock:
        os.system('clear')
        if not monitored_pairs:
            print_colored("Tidak ada pair yang dipantau. Gunakan '!add <PAIR>' untuk memulai.", Fore.YELLOW); return
        ap_status, ap_color = ("AKTIF", Fore.GREEN) if is_autopilot_running else ("TIDAK AKTIF", Fore.RED)
        print_colored(f"--- Live Dashboard @ {datetime.now().strftime('%H:%M:%S')} | Autopilot: {Style.BRIGHT}{ap_color}{ap_status}{Style.RESET_ALL} ---", Fore.YELLOW, Style.BRIGHT)
        print_colored("-" * 80, Fore.YELLOW)
        for pair_id, pair_data in monitored_pairs.items():
            line = f"{Style.BRIGHT}{pair_id.ljust(12)}{Style.RESET_ALL} | "
            if not pair_data.get('candle_data'):
                line += Fore.RED + "Menunggu Data..."; print(line); continue
            candle_data = pair_data['candle_data']; price = candle_data[-1]['close']
            ai = LocalAI(current_settings, []); analysis = ai.get_market_analysis(candle_data)
            if not analysis:
                line += Fore.RED + "Analisis Gagal (data kurang)"; print(line); continue
            bias = analysis['bias']; rsi = analysis['rsi']
            bias_color = Fore.GREEN if bias == "BULLISH" else Fore.RED if bias == "BEARISH" else Fore.YELLOW
            line += f"Harga: {price:<9.4f} | Tren: {Style.BRIGHT}{bias_color}{bias:<8}{Style.RESET_ALL} | RSI: {rsi:<5.1f} | "
            open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
            if open_pos:
                pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
                pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                type_color = Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED
                line += f"Posisi: {type_color}{open_pos.get('type'):<5}{Style.RESET_ALL} P/L: {pnl_color}{pnl:>6.2f}%{Style.RESET_ALL}"
            else:
                line += Fore.CYAN + "Status: Standby"
            print(line)
        print_colored("-" * 80, Fore.YELLOW)
        print_colored("Ketik perintah (atau !stop) lalu Enter untuk berinteraksi...", Fore.WHITE)

def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            with data_lock: pairs_to_analyze = list(monitored_pairs.keys())
            for pair_id in pairs_to_analyze:
                asyncio.run(run_autopilot_analysis(pair_id))
                time.sleep(1)
            stop_event.wait(current_settings.get("analysis_interval_sec", 30))
        else: time.sleep(1)

async def check_realtime_tp_sl_and_runup(pair_id, latest_price):
    with data_lock:
        if not is_autopilot_running: return
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    with data_lock:
        if current_pnl > open_position.get('run_up_percent', 0.0):
            open_position['run_up_percent'] = current_pnl
    tp_pct = current_settings.get('take_profit_pct'); sl_pct = current_settings.get('stop_loss_pct')
    close_reason = None
    if tp_pct and current_pnl >= tp_pct: close_reason = f"Take Profit @ {tp_pct}% tercapai."
    elif sl_pct and current_pnl <= -sl_pct: close_reason = f"Stop Loss @ {sl_pct}% tercapai."
    if close_reason:
        await analyze_and_close_trade(open_position, latest_price, close_reason, open_position.get("entry_snapshot"))

def data_refresh_worker():
    while not stop_event.is_set():
        with data_lock: pairs_to_refresh = list(monitored_pairs.keys())
        
        if is_autopilot_running:
            print_live_dashboard()
        
        if not pairs_to_refresh: time.sleep(REFRESH_INTERVAL_SECONDS); continue
        
        for pair_id in pairs_to_refresh:
            tf = monitored_pairs[pair_id]['timeframe']
            data = fetch_bybit_candle_data(pair_id, tf)
            if data: 
                with data_lock: monitored_pairs[pair_id]['candle_data'] = data
                latest_price = data[-1]['close']
                asyncio.run(check_realtime_tp_sl_and_runup(pair_id, latest_price))
            time.sleep(0.2)
        
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def main():
    global is_autopilot_running
    load_settings(); load_trades()
    for pair_id in current_settings.get("monitored_pairs_list", []):
        monitored_pairs[pair_id] = {"timeframe": "1H", "candle_data": []}
    display_welcome_message()
    for pair_id, data in monitored_pairs.items():
        print_colored(f"Memuat data awal untuk {pair_id}...", Fore.CYAN)
        candle_data = fetch_bybit_candle_data(pair_id, data['timeframe'])
        if candle_data:
            data['candle_data'] = candle_data; print_colored(f"Data {pair_id} berhasil dimuat.", Fore.GREEN)
        else: print_colored(f"Gagal memuat data untuk {pair_id}.", Fore.RED)
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()
    while True:
        try:
            if not is_autopilot_running:
                prompt_text = f"[{len(monitored_pairs)} Pairs] > "
                user_input = input(prompt_text)
            else:
                user_input = input() # Input tetap berjalan, tapi tanpa prompt
            command_parts = user_input.split()
            if not command_parts: continue
            cmd = command_parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan.", Fore.YELLOW)
                elif not monitored_pairs: print_colored("Error: Tidak ada pair yang dipantau. Gunakan '!add'.", Fore.RED)
                else: is_autopilot_running = True; print_live_dashboard()
            elif cmd == '!stop':
                if not is_autopilot_running: print_colored("Autopilot sudah tidak aktif.", Fore.YELLOW)
                else: is_autopilot_running = False; os.system('clear'); print_colored("🛑 Autopilot Lokal & Live Dashboard dinonaktifkan.", Fore.RED, Style.BRIGHT)
            elif cmd == '!status': print_live_dashboard()
            elif cmd == '!history': 
                # ... 
                pass
            elif cmd in ['!settings', '!set']: 
                # ...
                pass
            elif cmd == '!add':
                if len(command_parts) >= 2:
                    pair_id = command_parts[1].upper()
                    tf = command_parts[2] if len(command_parts) > 2 else '1H'
                    with data_lock:
                        if pair_id in monitored_pairs: print_colored(f"Error: {pair_id} sudah dipantau.", Fore.RED)
                        else:
                            print_colored(f"Menambahkan {pair_id} ({tf}) ke pantauan...", Fore.CYAN)
                            monitored_pairs[pair_id] = {"timeframe": tf, "candle_data": []}
                            data = fetch_bybit_candle_data(pair_id, tf)
                            if data:
                                monitored_pairs[pair_id]['candle_data'] = data
                                print_colored(f"Data awal untuk {pair_id} berhasil dimuat.", Fore.GREEN)
                            else:
                                print_colored(f"Gagal memuat data awal untuk {pair_id}, pair mungkin tidak valid.", Fore.RED); del monitored_pairs[pair_id]
                    save_settings()
                else: print_colored("Format salah. Gunakan: !add PAIR-USDT [TIMEFRAME]", Fore.RED)
            elif cmd == '!remove':
                if len(command_parts) == 2:
                    pair_id = command_parts[1].upper()
                    with data_lock:
                        if pair_id in monitored_pairs:
                            del monitored_pairs[pair_id]
                            print_colored(f"{pair_id} berhasil dihapus dari pantauan.", Fore.YELLOW)
                        else: print_colored(f"Error: {pair_id} tidak ada dalam daftar pantauan.", Fore.RED)
                    save_settings()
                else: print_colored("Format salah. Gunakan: !remove PAIR-USDT", Fore.RED)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga: {e}", Fore.RED)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
