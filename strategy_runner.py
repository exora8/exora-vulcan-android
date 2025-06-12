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
REFRESH_INTERVAL_SECONDS = 0.5

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
market_state = {} 
is_ai_thinking = False
is_autopilot_in_cooldown = {} 
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
    print_colored("    Strategic AI Analyst (Trailing TP Edition)    ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("AI fokus pada entry, Take Profit dikunci oleh Trailing Stop.", Fore.YELLOW)
    if IS_TERMUX: print_colored("Notifikasi Termux diaktifkan.", Fore.GREEN)
    print_colored("Gunakan '!start' untuk masuk ke Live Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "stop_loss_pct": 1.0, 
        "fee_pct": 0.1, 
        "analysis_interval_sec": 10, 
        "trailing_tp_activation_pct": 0.3,
        "trailing_tp_gap_pct": 0.1,
        "watched_pairs": {}
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            current_settings = json.load(f)
            for key, value in default_settings.items():
                if key not in current_settings: current_settings[key] = value
    else:
        current_settings = default_settings; save_settings()

def save_settings():
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
        status_color = Fore.YELLOW if trade['status'] == 'OPEN' else Fore.WHITE
        trade_type = trade.get('type', 'LONG'); type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
        print_colored(f"--- Trade ID: {trade['id']} ---", Fore.CYAN)
        print_colored(f"  Pair: {trade['instrumentId']} | Tipe: {trade_type} | Status: {trade['status']}", status_color)
        print_colored(f"  Entry: {entry_time} @ {trade['entryPrice']:.4f}", Fore.WHITE)
        if trade['status'] == 'CLOSED':
            exit_time = datetime.fromisoformat(trade['exitTimestamp'].replace('Z', '')).strftime('%Y-%m-%d %H:%M')
            pl_percent = trade.get('pl_percent', 0.0); is_profit = pl_percent > current_settings.get('fee_pct', 0.1)
            pl_color = Fore.GREEN if is_profit else Fore.RED
            print_colored(f"  Exit: {exit_time} @ {trade['exitPrice']:.4f}", Fore.WHITE)
            print_colored(f"  P/L: {pl_percent:.2f}%", pl_color, Style.BRIGHT)
            run_up = trade.get('run_up_percent', pl_percent)
            drawdown = trade.get('max_drawdown_percent', 0.0)
            print_colored(f"  Run-up: {run_up:.2f}%", Fore.YELLOW, end='')
            print_colored(f" / Drawdown: {drawdown:.2f}%", Fore.RED)
            if 'entry_snapshot' in trade and not is_profit:
                snapshot = trade['entry_snapshot']
                print_colored(f"  Pelajaran (Snapshot): Bias={snapshot.get('bias', 'N/A')}, RSI={snapshot.get('rsi', 0):.0f}", Fore.MAGENTA)
        print()


# --- FUNGSI API (BYBIT) ---
def fetch_bybit_candle_data(instId, timeframe):
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=spot&symbol={bybit_symbol}&interval={bybit_interval}&limit=300"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        else: return None
    except Exception: return None

# DIPINDAHKAN KE ATAS: Fungsi ini sekarang didefinisikan sebelum digunakan oleh kelas LocalAI
def calculate_pnl(entry_price, current_price, trade_type):
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

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
        relevant_data = data[-period:]; high = max(d['high'] for d in relevant_data)
        low = min(d['low'] for d in relevant_data); close = relevant_data[-1]['close']
        pivot = (high + low + close) / 3; s1 = (2 * pivot) - high; r1 = (2 * pivot) - low
        return {"p": pivot, "s1": s1, "r1": r1}
    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100: return None
        analysis = {"ema9": self.calculate_ema(candle_data, 9), "ema50": self.calculate_ema(candle_data, 50), "ema100": self.calculate_ema(candle_data, 100), "rsi": self.calculate_rsi(candle_data, 14), "pivots": self.calculate_lookback_pivots(candle_data, 100)}
        bias = "RANGING";
        if analysis["ema50"] > analysis["ema100"]: bias = "BULLISH"
        elif analysis["ema50"] < analysis["ema100"]: bias = "BEARISH"
        analysis["bias"] = bias; return analysis
    def check_for_repeated_mistake(self, current_analysis, trade_type, instrument_id):
        losing_trades = [t for t in self.past_trades if t.get('pl_percent', 0) < self.settings.get('fee_pct', 0.1)]
        if not losing_trades: return False
        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or loss.get("type") != trade_type: continue
            bias_same = current_analysis['bias'] == past_snapshot.get('bias')
            rsi_similar = abs(current_analysis['rsi'] - past_snapshot.get('rsi', 50)) < 15
            if bias_same and rsi_similar: return True
        return False
    def get_decision(self, candle_data, open_position, instrument_id):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data tidak cukup untuk analisis."}
        if open_position:
            return {"action": "HOLD", "reason": "Memantau posisi terbuka..."}
        if self.check_for_repeated_mistake(analysis, "LONG", instrument_id):
            return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan Long di {instrument_id}."}
        if self.check_for_repeated_mistake(analysis, "SHORT", instrument_id):
             return {"action": "HOLD", "reason": f"Menghindari pengulangan kesalahan Short di {instrument_id}."}
        current_price = candle_data[-1]['close']
        if analysis['bias'] == 'BULLISH':
            if current_price < analysis['pivots']['p'] and analysis['rsi'] < 70:
                return {"action": "BUY", "reason": f"Tren Bullish & pullback ke area Pivot. RSI: {analysis['rsi']:.0f}", "snapshot": analysis}
        if analysis['bias'] == 'BEARISH':
            if current_price > analysis['pivots']['p'] and analysis['rsi'] > 30:
                return {"action": "SELL", "reason": f"Tren Bearish & rally ke area Pivot. RSI: {analysis['rsi']:.0f}", "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup presisi. Bias: {analysis['bias']}, RSI: {analysis['rsi']:.0f}."}

# --- LOGIKA TRADING UTAMA ---
async def analyze_and_close_trade(trade, exit_price, close_trigger_reason, entry_snapshot=None):
    pnl = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    fee = current_settings.get('fee_pct', 0.1)
    is_profit = pnl > fee
    trade.update({'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': datetime.utcnow().isoformat() + "Z", 'pl_percent': pnl})
    if not is_profit and entry_snapshot:
        trade['entry_snapshot'] = entry_snapshot
    save_trades()
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL: {pnl:.2f}% | Entry: {trade['entryPrice']:.4f} | Exit: {exit_price:.4f}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking or is_autopilot_in_cooldown.get(instrument_id): return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data"): return
    is_ai_thinking = True
    try:
        candle_data = pair_state["candle_data"]
        open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        local_brain = LocalAI(current_settings, relevant_trades)
        decision = local_brain.get_decision(candle_data, open_position, instrument_id)
        action = decision.get('action', 'HOLD').upper(); reason = decision.get('reason', 'No reason provided.')
        current_price = candle_data[-1]['close']
        if action in ["BUY", "SELL"] and not open_position:
            trade_type = "LONG" if action == "BUY" else "SHORT"
            new_trade = {"id": int(time.time()), "instrumentId": instrument_id, "type": trade_type, "entryTimestamp": datetime.utcnow().isoformat() + "Z", "entryPrice": current_price, "entryReason": reason, "status": 'OPEN', "entry_snapshot": decision.get("snapshot"), "run_up_percent": 0.0, "max_drawdown_percent": 0.0, "trailing_stop_price": None}
            autopilot_trades.append(new_trade)
            save_trades()
            notif_title = f"{'🟢' if action == 'BUY' else '🔴'} Posisi {trade_type} Dibuka"
            notif_content = f"{instrument_id}: Entry @ {current_price:.4f} | {reason}"
            send_termux_notification(notif_title, notif_content)
    except Exception:
        is_autopilot_in_cooldown[instrument_id] = True; await asyncio.sleep(60); is_autopilot_in_cooldown[instrument_id] = False
    finally: is_ai_thinking = False

# --- THREAD WORKERS ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running:
            watched_pairs = list(current_settings.get("watched_pairs", {}).keys())
            if watched_pairs:
                for pair_id in watched_pairs:
                    asyncio.run(run_autopilot_analysis(pair_id))
                    time.sleep(1) 
            stop_event.wait(current_settings.get("analysis_interval_sec", 10))
        else: time.sleep(1)

async def check_realtime_position_management(instrument_id, latest_price):
    open_position = next((t for t in autopilot_trades if t['instrumentId'] == instrument_id and t['status'] == 'OPEN'), None)
    if not open_position: return
    
    current_pnl = calculate_pnl(open_position['entryPrice'], latest_price, open_position.get('type'))
    if current_pnl > open_position.get('run_up_percent', 0.0):
        open_position['run_up_percent'] = current_pnl
    if current_pnl < open_position.get('max_drawdown_percent', 0.0):
        open_position['max_drawdown_percent'] = current_pnl

    sl_pct = current_settings.get('stop_loss_pct')
    if sl_pct and current_pnl <= -sl_pct: 
        global is_ai_thinking
        if not is_ai_thinking:
            is_ai_thinking = True
            await analyze_and_close_trade(open_position, latest_price, f"Stop Loss @ {sl_pct}% tercapai.", open_position.get("entry_snapshot"))
            is_ai_thinking = False
        return

    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.3)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.1)
    
    if open_position.get("trailing_stop_price") is None and current_pnl >= activation_pct:
        if open_position['type'] == 'LONG':
            lock_in_profit_pct = current_pnl - gap_pct
            open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 + lock_in_profit_pct / 100)
        else:
            lock_in_profit_pct = current_pnl - gap_pct
            open_position['trailing_stop_price'] = open_position['entryPrice'] * (1 - lock_in_profit_pct / 100)
    
    if open_position.get("trailing_stop_price") is not None:
        if open_position['type'] == 'LONG':
            new_trailing_stop = latest_price * (1 - gap_pct / 100)
            if new_trailing_stop > open_position["trailing_stop_price"]:
                open_position["trailing_stop_price"] = new_trailing_stop
        else:
            new_trailing_stop = latest_price * (1 + gap_pct / 100)
            if new_trailing_stop < open_position["trailing_stop_price"]:
                open_position["trailing_stop_price"] = new_trailing_stop
        
        if open_position['type'] == 'LONG' and latest_price <= open_position["trailing_stop_price"]:
            if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], "Trailing Take Profit tercapai.", open_position.get("entry_snapshot"))
                is_ai_thinking = False
        elif open_position['type'] == 'SHORT' and latest_price >= open_position["trailing_stop_price"]:
             if not is_ai_thinking:
                is_ai_thinking = True
                await analyze_and_close_trade(open_position, open_position["trailing_stop_price"], "Trailing Take Profit tercapai.", open_position.get("entry_snapshot"))
                is_ai_thinking = False

def data_refresh_worker():
    global market_state
    while not stop_event.is_set():
        watched_pairs = current_settings.get("watched_pairs", {})
        if watched_pairs:
            for pair_id, timeframe in watched_pairs.items():
                data = fetch_bybit_candle_data(pair_id, timeframe)
                if data: 
                    analysis = LocalAI(current_settings, []).get_market_analysis(data)
                    market_state[pair_id] = {"candle_data": data, "analysis": analysis}
                    if is_autopilot_running:
                        latest_price = data[-1]['close']
                        asyncio.run(check_realtime_position_management(pair_id, latest_price))
                time.sleep(0.5)
        stop_event.wait(REFRESH_INTERVAL_SECONDS)

def handle_settings_command(parts):
    setting_map = {'sl': ('stop_loss_pct', '%'),'fee': ('fee_pct', '%'),'delay': ('analysis_interval_sec', ' detik'), 'tp_act': ('trailing_tp_activation_pct', '%'), 'tp_gap': ('trailing_tp_gap_pct', '%')}
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full_key, unit) in setting_map.items():
            display_key = key.capitalize().ljust(10)
            print_colored(f"{display_key} ({key:<10}) : {current_settings[full_key]}{unit}", Fore.WHITE)
        print(); return
    if len(parts) == 3 and parts[0] == '!set':
        key_short = parts[1].lower()
        if key_short not in setting_map: print_colored(f"Error: Kunci '{key_short}' tidak dikenal.", Fore.RED); return
        try:
            value = float(parts[2])
            if value < 0: print_colored("Error: Nilai tidak boleh negatif.", Fore.RED); return
        except ValueError: print_colored(f"Error: Nilai '{parts[2]}' harus berupa angka.", Fore.RED); return
        key_full, unit = setting_map[key_short]
        current_settings[key_full] = value; save_settings()
        print_colored(f"Pengaturan '{key_full}' berhasil diubah menjadi {value}{unit}.", Fore.GREEN, Style.BRIGHT); return
    print_colored("Format salah. Gunakan '!settings' atau '!set <key> <value>'.", Fore.RED)

def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="")
            print_colored("--- VULCAN'S EDITION LIVE DASHBOARD ---", Fore.CYAN, Style.BRIGHT)
            print_colored(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Refresh: {REFRESH_INTERVAL_SECONDS}s | AI Cycle: {current_settings.get('analysis_interval_sec')}s", Fore.WHITE)
            print_colored("="*60, Fore.CYAN)
            watched_pairs = current_settings.get("watched_pairs", {})
            if not watched_pairs:
                print_colored("\nWatchlist kosong. Tekan Ctrl+C untuk kembali dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            for pair_id, timeframe in watched_pairs.items():
                print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                open_pos = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                pair_state = market_state.get(pair_id)
                if open_pos:
                    price = pair_state['candle_data'][-1]['close'] if pair_state and pair_state.get('candle_data') else open_pos['entryPrice']
                    pnl = calculate_pnl(open_pos['entryPrice'], price, open_pos.get('type'))
                    pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                    type_color = Fore.GREEN if open_pos.get('type') == 'LONG' else Fore.RED
                    print_colored("  Status    : ", end=''); print_colored("POSITION OPEN", Fore.YELLOW, Style.BRIGHT)
                    print_colored("  Tipe      : ", end=''); print_colored(f"{open_pos.get('type')}", type_color, Style.BRIGHT)
                    print_colored("  Entry     : ", end=''); print_colored(f"{open_pos['entryPrice']:.4f}", Fore.WHITE)
                    print_colored("  PnL       : ", end=''); print_colored(f"{pnl:.2f}%", pnl_color, Style.BRIGHT)
                    if open_pos.get("trailing_stop_price") is not None:
                        tsp_price = open_pos.get("trailing_stop_price")
                        print_colored("  Trailing TP: ", end=''); print_colored(f"Aktif @ {tsp_price:.4f}", Fore.MAGENTA)
                    else:
                        print_colored("  Trailing TP: ", end=''); print_colored(f"Menunggu {current_settings.get('trailing_tp_activation_pct')}%", Fore.WHITE)
                else:
                    print_colored("  Status    : ", end=''); print_colored("Searching for setup...", Fore.BLUE)
                    if pair_state and pair_state.get("analysis"):
                        analysis = pair_state["analysis"]
                        bias_color = Fore.GREEN if analysis['bias'] == 'BULLISH' else Fore.RED if analysis['bias'] == 'BEARISH' else Fore.YELLOW
                        print_colored("  Trend     : ", end=''); print_colored(analysis['bias'], bias_color)
                        print_colored("  RSI       : ", end=''); print_colored(f"{analysis['rsi']:.2f}", Fore.WHITE)
                    else:
                        print_colored("  Trend     : Menunggu data...", Fore.WHITE)
            print_colored("\n"+"="*60, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard dan kembali ke command prompt.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def main():
    global is_autopilot_running
    load_settings(); load_trades(); display_welcome_message()
    
    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True); autopilot_thread.start()
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True); data_thread.start()
    while True:
        try:
            prompt_text = f"[Command] > "
            user_input = input(prompt_text)
            command_parts = user_input.split()
            if not command_parts: continue
            cmd = command_parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if is_autopilot_running: print_colored("Autopilot sudah berjalan. Dashboard aktif.", Fore.YELLOW)
                elif not current_settings.get("watched_pairs"): print_colored("Error: Watchlist kosong. Gunakan '!watch <PAIR>' dulu.", Fore.RED)
                else: 
                    is_autopilot_running = True
                    print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN, Style.BRIGHT)
                    time.sleep(1); run_dashboard_mode()
                    is_autopilot_running = False
                    print_colored("\n🛑 Live Dashboard ditutup. Autopilot dinonaktifkan.", Fore.RED, Style.BRIGHT)
                    print_colored("Ketik '!start' untuk masuk lagi atau '!exit' untuk keluar.", Fore.YELLOW)
            elif cmd == '!stop': print_colored("Gunakan '!start' untuk masuk ke dashboard, lalu Ctrl+C untuk berhenti.", Fore.YELLOW)
            elif cmd == '!watchlist':
                watched = current_settings.get("watched_pairs", {})
                if not watched: print_colored("Watchlist kosong.", Fore.YELLOW)
                else:
                    print_colored("\n--- Watchlist ---", Fore.CYAN, Style.BRIGHT)
                    for pair, tf in watched.items(): print_colored(f"- {pair} (Timeframe: {tf})", Fore.WHITE)
            elif cmd == '!watch':
                if len(command_parts) >= 2:
                    pair_id = command_parts[1].upper()
                    tf = command_parts[2] if len(command_parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"Pair {pair_id} dengan TF {tf} ditambahkan ke watchlist.", Fore.GREEN)
                else: print_colored("Format salah. Gunakan: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(command_parts) == 2:
                    pair_id = command_parts[1].upper()
                    if pair_id in current_settings['watched_pairs']:
                        del current_settings['watched_pairs'][pair_id]
                        if pair_id in market_state: del market_state[pair_id]
                        save_settings()
                        print_colored(f"Pair {pair_id} dihapus dari watchlist.", Fore.YELLOW)
                    else: print_colored(f"Error: Pair {pair_id} tidak ada di watchlist.", Fore.RED)
                else: print_colored("Format salah. Gunakan: !unwatch <PAIR>", Fore.RED)
            elif cmd == '!history':
                display_history()
            elif cmd in ['!settings', '!set']:
                handle_settings_command(command_parts)
            else:
                print_colored(f"Perintah '{user_input}' tidak dikenal. Ketik '!help'.", Fore.RED)
        except KeyboardInterrupt: break
        except Exception as e: print_colored(f"\nTerjadi error tak terduga di main loop: {e}", Fore.RED)
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join(); data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
