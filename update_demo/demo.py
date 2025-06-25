# --- PENTING: SKRIP INI MELAKUKAN PERDAGANGAN NYATA ---
# --- GUNAKAN DI AKUN TESTNET TERLEBIH DAHULU ---
# --- RISIKO DITANGGUNG PENGGUNA ---

import json
import os
import time
import threading
import requests
from datetime import datetime, timedelta
import asyncio
import math
import hmac
import hashlib

# Coba impor pustaka yang diperlukan
try:
    from pybit.unified_trading import HTTP
except ImportError:
    print("Peringatan: Pustaka 'pybit' tidak ditemukan. Silakan install dengan 'pip install pybit'")
    exit()

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Peringatan: Pustaka 'colorama' tidak ditemukan. Output tidak akan berwarna.")
    class DummyColor:
        def __init__(self): self.BLACK = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.MAGENTA = self.CYAN = self.WHITE = self.RESET = ''
    class DummyStyle:
        def __init__(self): self.DIM = self.NORMAL = self.BRIGHT = self.RESET_ALL = ''
    Fore = DummyColor()
    Style = DummyStyle()

# --- KONFIGURASI GLOBAL ---
SETTINGS_FILE = 'settings.json'
TRADES_FILE = 'trades.json'
BYBIT_API_URL = "https://api.bybit.com/v5/market"
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data/v2/"
REFRESH_INTERVAL_SECONDS = 0.5
BACKTEST_FETCH_CHUNK_LIMIT = 1000
MAX_TRADES_IN_HISTORY = 80

# --- STATE APLIKASI ---
current_settings = {}
autopilot_trades = []
market_state = {}
is_ai_thinking = False
is_autopilot_running = False
stop_event = threading.Event()
IS_TERMUX = 'TERMUX_VERSION' in os.environ
bybit_session = None # Akan diinisialisasi nanti

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
    print_colored("  Strategic AI Analyst (Bybit Integrated Edition) ", Fore.CYAN, Style.BRIGHT)
    print_colored("==================================================", Fore.CYAN, Style.BRIGHT)
    print_colored("PERINGATAN: Skrip ini melakukan trade nyata. Gunakan dengan hati-hati.", Fore.RED, Style.BRIGHT)
    if current_settings.get("use_testnet", True):
        print_colored("Mode Testnet Aktif. Tidak ada dana nyata yang digunakan.", Fore.GREEN)
    else:
        print_colored("Mode Live Aktif. Dana nyata akan digunakan.", Fore.YELLOW, Style.BRIGHT)
    print_colored("Gunakan '!start' untuk masuk ke Live Dashboard.", Fore.YELLOW)
    print_colored("Ketik '!help' untuk daftar perintah.", Fore.YELLOW)
    print()

def display_help():
    print_colored("\n--- Daftar Perintah (Command Mode) ---", Fore.CYAN, Style.BRIGHT)
    print_colored("!start                - Masuk ke Live Dashboard & aktifkan Autopilot", Fore.GREEN)
    print_colored("!watch <PAIR> [TF]    - Tambah pair ke watchlist (e.g., BTC-USDT)", Fore.GREEN)
    print_colored("!unwatch <PAIR>       - Hapus pair dari watchlist", Fore.GREEN)
    print_colored("!watchlist            - Tampilkan semua pair yang dipantau", Fore.GREEN)
    print_colored("!history              - Tampilkan riwayat trade (terbatas 80 terakhir)", Fore.GREEN)
    print_colored("!settings             - Tampilkan semua pengaturan global", Fore.GREEN)
    print_colored("!set <key> <value>    - Ubah pengaturan (key: sl, fee, delay, tp_act, tp_gap, leverage, trade_size, cc_key, bybit_key, bybit_secret)", Fore.GREEN)
    print_colored("!exit                 - Keluar dari aplikasi", Fore.GREEN)
    print()

# --- MANAJEMEN DATA & PENGATURAN ---
def load_settings():
    global current_settings
    default_settings = {
        "bybit_api_key": "YOUR_API_KEY", "bybit_api_secret": "YOUR_API_SECRET", "use_testnet": True,
        "stop_loss_pct": 0.20, "fee_pct": 0.055, "analysis_interval_sec": 10,
        "trailing_tp_activation_pct": 0.30, "trailing_tp_gap_pct": 0.05,
        "target_winrate_pct": 85.0, "cryptocompare_api_key": "YOUR_CRYPTOCOMPARE_API_KEY",
        "max_allowed_funding_rate_pct": 0.075, "leverage": 10, "trade_size_percent_of_balance": 5.0,
        "watched_pairs": {}
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                current_settings = {**default_settings, **loaded_settings}
        except (json.JSONDecodeError, IOError):
            current_settings = default_settings
    else:
        current_settings = default_settings
    save_settings()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(current_settings, f, indent=4)
    except IOError as e: print_colored(f"Error saving settings: {e}", Fore.RED)

def load_trades():
    global autopilot_trades
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f: autopilot_trades = json.load(f)
        except (json.JSONDecodeError, IOError): autopilot_trades = []
    else: autopilot_trades = []

def save_trades():
    global autopilot_trades
    autopilot_trades.sort(key=lambda x: x['entryTimestamp'])
    if len(autopilot_trades) > MAX_TRADES_IN_HISTORY:
        num_to_trim = len(autopilot_trades) - MAX_TRADES_IN_HISTORY
        autopilot_trades = autopilot_trades[num_to_trim:]
    try:
        with open(TRADES_FILE, 'w') as f: json.dump(autopilot_trades, f, indent=4)
    except IOError as e: print_colored(f"Error saving trades: {e}", Fore.RED)

# --- FUNGSI INTEGRASI BYBIT ---
def initialize_bybit_session():
    global bybit_session
    api_key = current_settings.get("bybit_api_key")
    api_secret = current_settings.get("bybit_api_secret")
    if not api_key or api_key == "YOUR_API_KEY" or not api_secret or api_secret == "YOUR_API_SECRET":
        print_colored("Kunci API Bybit belum diatur di settings.json. Fungsionalitas trade akan dinonaktifkan.", Fore.RED, Style.BRIGHT)
        return False
    try:
        bybit_session = HTTP(
            testnet=current_settings.get("use_testnet", True),
            api_key=api_key,
            api_secret=api_secret
        )
        # Test connection
        res = bybit_session.get_server_time()
        if res.get('retCode') == 0:
            print_colored("Koneksi ke Bybit berhasil.", Fore.GREEN)
            return True
        else:
            print_colored(f"Gagal koneksi ke Bybit: {res.get('retMsg')}", Fore.RED)
            return False
    except Exception as e:
        print_colored(f"Error saat inisialisasi sesi Bybit: {e}", Fore.RED)
        return False

def get_bybit_balance(coin="USDT"):
    if not bybit_session: return 0.0
    try:
        res = bybit_session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if res['retCode'] == 0 and res['result']['list']:
            return float(res['result']['list'][0]['coin'][0]['walletBalance'])
        return 0.0
    except Exception:
        return 0.0

def get_symbol_info(symbol):
    if not bybit_session: return None
    try:
        res = bybit_session.get_instruments_info(category="linear", symbol=symbol)
        if res.get('retCode') == 0 and res['result']['list']:
            return res['result']['list'][0]['lotSizeFilter']
    except Exception:
        return None

def adjust_qty_to_step(quantity, qty_step):
    qty_step_float = float(qty_step)
    return math.floor(quantity / qty_step_float) * qty_step_float

def calculate_order_qty(symbol, price):
    balance = get_bybit_balance()
    leverage = current_settings.get('leverage', 10)
    trade_percent = current_settings.get('trade_size_percent_of_balance', 5.0)
    if balance == 0 or price == 0: return 0

    capital_to_use = balance * (trade_percent / 100)
    notional_value = capital_to_use * leverage
    quantity = notional_value / price

    info = get_symbol_info(symbol)
    if info:
        return adjust_qty_to_step(quantity, info['qtyStep'])
    return round(quantity, 8) # Fallback

def place_bybit_order(symbol, side, reason=""):
    if not bybit_session: return None
    price = market_state.get(symbol.replace('-', ''), {}).get('candle_data', [{}])[-1].get('close')
    if not price: return None
        
    qty = calculate_order_qty(symbol.replace('-', ''), price)
    if qty <= 0:
        print_colored(f"Gagal membuka posisi: Kuantitas trade terlalu kecil (Qty: {qty})", Fore.RED)
        return None
        
    trade_type = "LONG" if side == "Buy" else "SHORT"
    try:
        res = bybit_session.place_order(
            category="linear",
            symbol=symbol.replace('-', ''),
            side=side,
            orderType="Market",
            qty=str(qty)
        )
        if res.get('retCode') == 0:
            order_id = res['result']['orderId']
            print_colored(f"✅ Posisi {trade_type} {symbol} berhasil dibuka (Qty: {qty}). Order ID: {order_id}", Fore.GREEN, Style.BRIGHT)
            # Buat catatan trade lokal untuk PNL tracking & history
            new_trade = {
                "id": order_id, "instrumentId": symbol, "type": trade_type,
                "entryTimestamp": datetime.utcnow().isoformat() + 'Z', "entryPrice": price, "qty": qty,
                "entryReason": reason, "status": 'OPEN', "run_up_percent": 0.0, "max_drawdown_percent": 0.0,
                "trailing_stop_price": None, "current_tp_checkpoint_level": 0.0
            }
            autopilot_trades.append(new_trade)
            save_trades()
            notif_title = f"🟢 Posisi {trade_type} Dibuka: {symbol}"
            notif_content = f"Entry @ {price:.4f} | Qty: {qty}"
            send_termux_notification(notif_title, notif_content)
            return new_trade
        else:
            print_colored(f"❌ Gagal membuka posisi {symbol}: {res.get('retMsg')}", Fore.RED)
            return None
    except Exception as e:
        print_colored(f"Error saat place_bybit_order: {e}", Fore.RED)
        return None


def close_bybit_position(trade, close_price, reason):
    if not bybit_session or 'qty' not in trade: return
    symbol = trade['instrumentId'].replace('-', '')
    side = "Sell" if trade['type'] == 'LONG' else "Buy" # Arah berlawanan untuk menutup
    try:
        res = bybit_session.place_order(
            category="linear", symbol=symbol, side=side, orderType="Market",
            qty=str(trade['qty']), reduceOnly=True
        )
        if res.get('retCode') == 0:
            print_colored(f"✅ Posisi {symbol} berhasil ditutup karena: {reason}", Fore.GREEN)
            asyncio.run(analyze_and_close_trade_local(trade, close_price, reason))
        else:
            # Jika gagal (misal posisi sudah tertutup manual), tetap update state lokal
            print_colored(f"⚠️ Gagal menutup posisi {symbol} via API: {res.get('retMsg')}. Mungkin sudah tertutup.", Fore.YELLOW)
            asyncio.run(analyze_and_close_trade_local(trade, close_price, reason))
    except Exception as e:
        print_colored(f"Error saat close_bybit_position: {e}", Fore.RED)

def get_bybit_position(symbol):
    if not bybit_session: return None
    try:
        res = bybit_session.get_positions(category="linear", symbol=symbol.replace('-', ''))
        if res.get('retCode') == 0 and res['result']['list']:
            pos = res['result']['list'][0]
            if float(pos.get('size', 0)) > 0:
                return {
                    'symbol': pos['symbol'],
                    'side': 'LONG' if pos['side'] == 'Buy' else 'SHORT',
                    'size': float(pos['size']),
                    'avgPrice': float(pos['avgPrice']),
                    'unrealisedPnl': float(pos.get('unrealisedPnl', 0))
                }
    except Exception:
        return None
    return None

def set_leverage_for_pair(symbol, leverage):
    if not bybit_session: return
    try:
        res = bybit_session.set_leverage(
            category="linear",
            symbol=symbol.replace('-', ''),
            buyLeverage=str(leverage),
            sellLeverage=str(leverage)
        )
        if res.get('retCode') == 0:
            print_colored(f"Leverage untuk {symbol} berhasil diatur ke {leverage}x.", Fore.GREEN)
        else:
            # Kode 110045 berarti leverage tidak diubah, bukan error.
            if res.get('retCode') == 110045:
                print_colored(f"Leverage untuk {symbol} sudah {leverage}x.", Fore.CYAN)
            else:
                print_colored(f"Gagal mengatur leverage untuk {symbol}: {res.get('retMsg')}", Fore.RED)
    except Exception as e:
        print_colored(f"Error saat set_leverage: {e}", Fore.RED)


# --- API Fetching (Tidak ada perubahan signifikan) ---
def fetch_funding_rate(instId):
    # (Fungsi ini tidak diubah)
    bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/tickers?category=linear&symbol={bybit_symbol}"
        response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}) and data['result']['list']:
            return float(data['result']['list'][0].get('fundingRate', '0')) * 100
        return None
    except (requests.exceptions.RequestException, ValueError, KeyError): return None

def fetch_recent_candles(instId, timeframe, limit=300):
    # (Fungsi ini tidak diubah)
    timeframe_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1H': '60', '2H': '120', '4H': '240', '1D': 'D', '1W': 'W'}
    bybit_interval = timeframe_map.get(timeframe, '60'); bybit_symbol = instId.replace('-', '')
    try:
        url = f"{BYBIT_API_URL}/kline?category=linear&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
        if data.get("retCode") == 0 and 'list' in data.get('result', {}):
            candle_list = data['result']['list']
            if len(candle_list) < 100 + 3: return None
            return [{"time": int(d[0]), "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])} for d in candle_list][::-1]
        return None
    except requests.exceptions.RequestException: return None
    except Exception: return None

# --- OTAK LOCAL AI (Tidak ada perubahan) ---
class LocalAI:
    def __init__(self, settings, past_trades_for_pair):
        self.settings = settings
        self.past_trades = past_trades_for_pair

    def calculate_ema(self, data, period):
        if len(data) < period: return []
        closes = [d['close'] for d in data]
        ema_values = [sum(closes[:period]) / period]
        multiplier = 2 / (period + 1)
        for i in range(period, len(closes)):
            ema = (closes[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        return ema_values

    def analyze_candle_solidity(self, candle):
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        return body / full_range if full_range > 0 else 1.0

    def get_market_analysis(self, candle_data):
        if len(candle_data) < 100 + 3: return None
        ema9 = self.calculate_ema(candle_data, 9)
        ema50 = self.calculate_ema(candle_data, 50)
        ema100 = self.calculate_ema(candle_data, 100)
        if len(ema9) < 2 or not ema50 or not ema100: return None
        
        analysis = {
            "ema9_current": ema9[-1], "ema9_prev": ema9[-2], "ema50": ema50[-1], "ema100": ema100[-1],
            "current_candle_close": candle_data[-1]['close'], "prev_candle_close": candle_data[-2]['close'],
            "bias": "BULLISH" if ema50[-1] > ema100[-1] else "BEARISH" if ema50[-1] < ema100[-1] else "RANGING",
        }
        pre_entry_candles = candle_data[-4:-1]
        analysis["pre_entry_candle_solidity"] = [self.analyze_candle_solidity(c) for c in pre_entry_candles]
        analysis["pre_entry_candle_direction"] = ['UP' if c['close'] > c['open'] else 'DOWN' for c in pre_entry_candles]
        return analysis

    def check_for_repeated_mistake(self, current_analysis):
        losing_trades = [t for t in self.past_trades if t.get('status') == 'CLOSED' and (t.get('pl_percent', 0) - self.settings.get('fee_pct', 0.1)) < 0]
        if not losing_trades: return (False, None)
        SIMILARITY_THRESHOLD = 3
        for loss in losing_trades:
            past_snapshot = loss.get("entry_snapshot")
            if not past_snapshot or past_snapshot.get('bias') != current_analysis['bias']: continue
            similarity_score = 1
            current_pos_vs_ema50 = 'above' if current_analysis['current_candle_close'] > current_analysis['ema50'] else 'below'
            past_pos_vs_ema50 = 'above' if past_snapshot['current_candle_close'] > past_snapshot['ema50'] else 'below'
            if current_pos_vs_ema50 == past_pos_vs_ema50: similarity_score += 1
            if current_analysis['pre_entry_candle_direction'] == past_snapshot.get('pre_entry_candle_direction', []): similarity_score += 1
            avg_solidity_current = sum(current_analysis['pre_entry_candle_solidity']) / 3
            past_solidity_list = past_snapshot.get('pre_entry_candle_solidity', [0,0,0])
            avg_solidity_past = sum(past_solidity_list) / 3 if past_solidity_list else 0
            if abs(avg_solidity_current - avg_solidity_past) < 0.2: similarity_score += 1
            if similarity_score >= SIMILARITY_THRESHOLD:
                reason = (f"PERINGATAN: Setup ini {similarity_score*25}% mirip dengan loss sebelumnya (ID: {loss['id']}).\n"
                          f"  - Kedua setup memiliki bias {current_analysis['bias']} dan harga di {current_pos_vs_ema50} EMA50.")
                return (True, reason)
        return (False, None)

    def get_decision(self, candle_data, open_position, funding_rate=0.0):
        analysis = self.get_market_analysis(candle_data)
        if not analysis: return {"action": "HOLD", "reason": "Data teknikal tidak cukup untuk analisis."}
        if open_position: return {"action": "HOLD", "reason": "Sudah ada posisi terbuka, sedang memantau."}
        max_funding_rate = self.settings.get("max_allowed_funding_rate_pct", 0.075)
        potential_trade_type = None
        if analysis['bias'] == 'BULLISH' and analysis['prev_candle_close'] <= analysis['ema9_prev'] and analysis['current_candle_close'] > analysis['ema9_current']:
            potential_trade_type = 'LONG'
        elif analysis['bias'] == 'BEARISH' and analysis['prev_candle_close'] >= analysis['ema9_prev'] and analysis['current_candle_close'] < analysis['ema9_current']:
            potential_trade_type = 'SHORT'
        if potential_trade_type:
            if potential_trade_type == 'LONG' and funding_rate > max_funding_rate:
                return {"action": "HOLD", "reason": f"Sinyal LONG diabaikan. Funding rate terlalu tinggi: {funding_rate:.4f}%"}
            if potential_trade_type == 'SHORT' and funding_rate < -max_funding_rate:
                return {"action": "HOLD", "reason": f"Sinyal SHORT diabaikan. Funding rate terlalu negatif: {funding_rate:.4f}%"}
            is_repeated_mistake, warning_reason = self.check_for_repeated_mistake(analysis)
            if is_repeated_mistake:
                return {"action": "HOLD", "reason": warning_reason}
            ai_reason = (f"Alasan AI (tinyllama): Entry {potential_trade_type} diambil berdasarkan konfirmasi kelanjutan tren. "
                         f"Pasar menunjukkan bias {analysis['bias']} yang kuat, dan sinyal ini muncul setelah harga melakukan pullback sehat ke EMA9, "
                         "menunjukkan potensi kekuatan untuk melanjutkan pergerakan.")
            avg_solidity = sum(analysis['pre_entry_candle_solidity']) / 3
            bot_details = (f"Detail Bot: \n"
                           f"  - Bias Pasar: {analysis['bias']} (EMA50: {analysis['ema50']:.2f} vs EMA100: {analysis['ema100']:.2f})\n"
                           f"  - Trigger: Candle close ({analysis['current_candle_close']:.2f}) melintasi EMA9 ({analysis['ema9_current']:.2f})\n"
                           f"  - Kondisi Pra-Entry: Arah 3 candle = {analysis['pre_entry_candle_direction']}, Rata-rata Soliditas = {avg_solidity:.2f}")
            full_reason = f"{ai_reason}\n{bot_details}"
            return {"action": "BUY" if potential_trade_type == 'LONG' else "SELL", "reason": full_reason, "snapshot": analysis}
        return {"action": "HOLD", "reason": f"Menunggu setup. Bias saat ini: {analysis['bias']}."}


# --- LOGIKA TRADING (DIMODIFIKASI UNTUK REAL TRADING) ---
def calculate_pnl(entry_price, current_price, trade_type):
    if entry_price == 0: return 0.0
    if trade_type == 'LONG': return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT': return ((entry_price - current_price) / entry_price) * 100
    return 0

async def analyze_and_close_trade_local(trade, exit_price, reason):
    # Fungsi ini HANYA mengupdate state lokal setelah posisi di Bybit ditutup
    pnl_gross = calculate_pnl(trade['entryPrice'], exit_price, trade.get('type', 'LONG'))
    exit_dt = datetime.utcnow()
    
    trade.update({ 'status': 'CLOSED', 'exitPrice': exit_price, 'exitTimestamp': exit_dt.isoformat() + 'Z', 'pl_percent': pnl_gross })
    
    is_profit = (pnl_gross - current_settings.get('fee_pct', 0.1)) > 0
    if is_profit and 'entry_snapshot' in trade:
        try: del trade['entry_snapshot']
        except KeyError: pass

    save_trades()
    pnl_net = pnl_gross - current_settings.get('fee_pct', 0.1)
    notif_title = f"🔴 Posisi {trade.get('type')} Ditutup: {trade['instrumentId']}"
    notif_content = f"PnL (Net): {pnl_net:.2f}% | Exit: {exit_price:.4f} | Trigger: {reason}"
    send_termux_notification(notif_title, notif_content)

async def run_autopilot_analysis(instrument_id):
    global is_ai_thinking
    if is_ai_thinking: return
    pair_state = market_state.get(instrument_id)
    if not pair_state or not pair_state.get("candle_data") or len(pair_state["candle_data"]) < 100 + 3: return

    is_ai_thinking = True
    try:
        # Cek posisi real di Bybit
        open_pos_bybit = get_bybit_position(instrument_id)
        
        relevant_trades = [t for t in autopilot_trades if t['instrumentId'] == instrument_id]
        ai = LocalAI(current_settings, relevant_trades)
        funding_rate = pair_state.get("funding_rate", 0.0)
        
        decision = ai.get_decision(pair_state["candle_data"], open_pos_bybit, funding_rate)

        if decision.get('action') in ["BUY", "SELL"] and not open_pos_bybit:
            snapshot = decision.get("snapshot", {})
            snapshot["funding_rate"] = funding_rate
            
            # Eksekusi order di Bybit
            place_bybit_order(
                instrument_id, 
                "Buy" if decision['action'] == "BUY" else "Sell", 
                reason=decision.get("reason")
            )
    except Exception as e:
        print_colored(f"Error dalam autopilot analysis: {e}", Fore.RED)
    finally:
        is_ai_thinking = False

# --- THREAD WORKERS (DIMODIFIKASI) ---
def autopilot_worker():
    while not stop_event.is_set():
        if is_autopilot_running and bybit_session:
            for pair_id in list(current_settings.get("watched_pairs", {})):
                asyncio.run(run_autopilot_analysis(pair_id))
            time.sleep(current_settings.get("analysis_interval_sec", 10))
        else:
            time.sleep(1)

async def check_realtime_position_management(trade_obj, current_candle_data):
    if not trade_obj or trade_obj['status'] != 'OPEN': return

    # Logika SL
    sl_pct = current_settings.get('stop_loss_pct')
    sl_price = trade_obj['entryPrice'] * (1 - abs(sl_pct) / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 + abs(sl_pct) / 100)
    if (trade_obj['type'] == 'LONG' and current_candle_data['low'] <= sl_price) or \
       (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= sl_price):
        close_bybit_position(trade_obj, sl_price, f"Stop Loss @ {-abs(sl_pct):.2f}%")
        return # Hentikan proses lebih lanjut untuk trade ini

    # Logika Trailing TP
    activation_pct = current_settings.get("trailing_tp_activation_pct", 0.30)
    gap_pct = current_settings.get("trailing_tp_gap_pct", 0.05)
    
    # Cek apakah trailing stop di-trigger
    ts_price = trade_obj.get('trailing_stop_price')
    if ts_price is not None and ((trade_obj['type'] == 'LONG' and current_candle_data['low'] <= ts_price) or \
                                  (trade_obj['type'] == 'SHORT' and current_candle_data['high'] >= ts_price)):
        close_bybit_position(trade_obj, ts_price, "Trailing TP")
        return

    # Update level Trailing TP
    pnl_now = calculate_pnl(trade_obj['entryPrice'], current_candle_data['high' if trade_obj['type'] == 'LONG' else 'low'], trade_obj['type'])
    if pnl_now >= activation_pct:
        current_cp = trade_obj.get('current_tp_checkpoint_level', 0.0)
        if current_cp == 0.0: current_cp = activation_pct
        
        steps_passed = math.floor((pnl_now - current_cp) / gap_pct)
        if steps_passed > 0:
            new_cp = current_cp + (steps_passed * gap_pct)
            trade_obj['current_tp_checkpoint_level'] = new_cp
            new_ts_level = new_cp - gap_pct
            trade_obj['trailing_stop_price'] = trade_obj['entryPrice'] * (1 + new_ts_level / 100) if trade_obj['type'] == 'LONG' else trade_obj['entryPrice'] * (1 - new_ts_level / 100)
            save_trades() # Simpan state trailing stop yang baru

def data_refresh_worker():
    while not stop_event.is_set():
        # Hapus trade lokal yang sudah 'CLOSED' tapi posisinya sudah tidak ada di Bybit
        open_local_trades = [t for t in autopilot_trades if t['status'] == 'OPEN']
        for trade in open_local_trades:
            if not get_bybit_position(trade['instrumentId']):
                 print_colored(f"Posisi lokal {trade['instrumentId']} tidak ditemukan di Bybit. Mungkin ditutup manual. Status diupdate.", Fore.YELLOW)
                 # Gunakan harga close terakhir sebagai harga exit
                 last_price = market_state.get(trade['instrumentId'],{}).get('candle_data',[{}])[-1].get('close', trade['entryPrice'])
                 asyncio.run(analyze_and_close_trade_local(trade, last_price, "Posisi tidak ditemukan di Bybit"))

        for pair_id, timeframe in list(current_settings.get("watched_pairs", {}).items()):
            if pair_id not in market_state: market_state[pair_id] = {}
            candle_data = fetch_recent_candles(pair_id, timeframe)
            funding_rate = fetch_funding_rate(pair_id)
            market_state[pair_id]['funding_rate'] = funding_rate if funding_rate is not None else market_state[pair_id].get('funding_rate', 0.0)
            if candle_data:
                market_state[pair_id]["candle_data"] = candle_data
                # Cari trade lokal yang masih OPEN untuk manajemen SL/TP
                open_pos_local = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                if open_pos_local:
                    asyncio.run(check_realtime_position_management(open_pos_local, candle_data[-1]))
            time.sleep(0.5)
        time.sleep(REFRESH_INTERVAL_SECONDS)


# --- FUNGSI UTAMA & DASHBOARD (DIMODIFIKASI) ---
def handle_settings_command(parts):
    setting_map = {
        'sl': ('stop_loss_pct', '%'), 'fee': ('fee_pct', '%'), 'delay': ('analysis_interval_sec', 's'),
        'tp_act': ('trailing_tp_activation_pct', '%'), 'tp_gap': ('trailing_tp_gap_pct', '%'),
        'leverage': ('leverage', 'x'), 'trade_size': ('trade_size_percent_of_balance', '%'),
        'cc_key': ('cryptocompare_api_key', ''), 'bybit_key': ('bybit_api_key', ''), 'bybit_secret': ('bybit_api_secret', '')
    }
    if len(parts) == 1 and parts[0] == '!settings':
        print_colored("\n--- Pengaturan Saat Ini ---", Fore.CYAN, Style.BRIGHT)
        for key, (full, unit) in setting_map.items():
            val = current_settings.get(full, 'N/A')
            if 'key' in key or 'secret' in key:
                if isinstance(val, str) and len(val) > 8: val = val[:4] + '...' + val[-4:]
            print_colored(f"{key.capitalize():<12} ({key:<12}) : {val}{unit}", Fore.WHITE)
        return
    if len(parts) == 3 and parts[0] == '!set':
        key, val_str = parts[1].lower(), parts[2]
        if key not in setting_map: print_colored(f"Kunci '{key}' tidak dikenal.", Fore.RED); return
        try:
            full, unit = setting_map[key]
            current_settings[full] = float(val_str) if key not in ['cc_key', 'bybit_key', 'bybit_secret'] else val_str
            save_settings(); print_colored(f"Pengaturan '{full}' diubah menjadi {current_settings[full]}{unit}.", Fore.GREEN)
            if key == 'leverage':
                print_colored("Mengupdate leverage untuk semua pair di watchlist...", Fore.YELLOW)
                for pair_id in current_settings.get("watched_pairs", {}):
                    set_leverage_for_pair(pair_id, float(val_str))
        except ValueError: print_colored(f"Nilai '{val_str}' tidak valid untuk '{key}'.", Fore.RED)

def run_dashboard_mode():
    try:
        while True:
            print("\033[H\033[J", end="") # Clear screen
            print_colored("--- VULCAN'S LIVE DASHBOARD (BYBIT INTEGRATED) ---", Fore.CYAN, Style.BRIGHT)
            
            # TAMPILKAN BALANCE & LEVERAGE
            balance = get_bybit_balance()
            leverage = current_settings.get('leverage', 'N/A')
            balance_color = Fore.GREEN if balance > 0 else Fore.RED
            print_colored("Bybit Balance: ", end=""); print_colored(f"{balance:.2f} USDT", balance_color, Style.BRIGHT, end="")
            print_colored(" | Leverage Setting: ", end=""); print_colored(f"{leverage}x", Fore.YELLOW, Style.BRIGHT)

            print_colored("="*80, Fore.CYAN)
            if not current_settings.get("watched_pairs"):
                print_colored("\nWatchlist kosong. Tekan Ctrl+C dan gunakan '!watch <PAIR>'.", Fore.YELLOW)
            
            for pair_id, timeframe in current_settings.get("watched_pairs", {}).items():
                print_colored(f"\n⦿ {pair_id} ({timeframe})", Fore.WHITE, Style.BRIGHT)
                
                # Gunakan data posisi real dari Bybit untuk tampilan
                open_pos_bybit = get_bybit_position(pair_id)
                
                if open_pos_bybit:
                    pnl_color = Fore.GREEN if open_pos_bybit['unrealisedPnl'] > 0 else Fore.RED
                    print_colored(f"  Status: OPEN {open_pos_bybit['side']} | Qty: {open_pos_bybit['size']}", end="")
                    print_colored(f" | Entry: {open_pos_bybit['avgPrice']:.4f}", end="")
                    print_colored(f" | PnL: ", end=""); print_colored(f"{open_pos_bybit['unrealisedPnl']:.2f} USDT", pnl_color, Style.BRIGHT)
                    
                    # Cek info trailing stop dari data lokal
                    local_trade_info = next((t for t in autopilot_trades if t['instrumentId'] == pair_id and t['status'] == 'OPEN'), None)
                    if local_trade_info and local_trade_info.get("trailing_stop_price"):
                         ts_price = local_trade_info.get("trailing_stop_price", 0)
                         print_colored(f"  Trailing Stop Aktif pada harga: {ts_price:.4f}", Fore.MAGENTA)
                else:
                    funding_rate = market_state.get(pair_id, {}).get('funding_rate', 0.0)
                    funding_color = Fore.RED if funding_rate > 0.01 else Fore.GREEN if funding_rate < -0.01 else Fore.WHITE
                    print_colored(f"  Status: Waiting | Funding: ", end=""); print_colored(f"{funding_rate:.4f}%", funding_color)
                    
                    # AI Log dari trade terakhir
                    last_trade = next((t for t in sorted(autopilot_trades, key=lambda x: x['entryTimestamp'], reverse=True) if t['instrumentId'] == pair_id), None)
                    last_reason = "Mencari setup..."
                    if last_trade and last_trade.get('entryReason'):
                        last_reason = last_trade['entryReason'].split('\n')[0]
                    print_colored(f"  AI Log: {last_reason}", Fore.YELLOW)

            print_colored("\n" + "="*80, Fore.CYAN)
            print_colored("Tekan Ctrl+C untuk keluar dari dashboard.", Fore.YELLOW)
            time.sleep(1)
    except KeyboardInterrupt:
        return

def main():
    global is_autopilot_running
    load_settings()
    load_trades()
    display_welcome_message()

    if not initialize_bybit_session():
        print_colored("Tidak dapat melanjutkan tanpa koneksi Bybit. Periksa kunci API Anda.", Fore.RED)
        return

    # Backtesting tidak diubah, namun sekarang hanya untuk tujuan pembelajaran AI, bukan eksekusi
    # check_and_run_backtests() # Jika Anda masih ingin menjalankannya

    autopilot_thread = threading.Thread(target=autopilot_worker, daemon=True)
    data_thread = threading.Thread(target=data_refresh_worker, daemon=True)
    autopilot_thread.start()
    data_thread.start()

    while True:
        try:
            user_input = input("[Command] > ").strip()
            if not user_input: continue
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd == '!exit': break
            elif cmd == '!help': display_help()
            elif cmd == '!start':
                if not current_settings.get("watched_pairs"):
                    print_colored("Watchlist kosong. Gunakan '!watch <PAIR>'.", Fore.RED); continue
                is_autopilot_running = True
                print_colored("✅ Autopilot diaktifkan. Memasuki Live Dashboard...", Fore.GREEN)
                run_dashboard_mode()
                is_autopilot_running = False
                print_colored("\n🛑 Live Dashboard ditutup.", Fore.RED)
            elif cmd == '!watch':
                if len(parts) >= 2:
                    pair_id = parts[1].upper(); tf = parts[2] if len(parts) > 2 else '1H'
                    current_settings['watched_pairs'][pair_id] = tf
                    save_settings()
                    print_colored(f"{pair_id} ({tf}) ditambahkan ke watchlist.", Fore.GREEN)
                    # Set leverage untuk pair yang baru ditambahkan
                    set_leverage_for_pair(pair_id, current_settings.get('leverage'))
                else: print_colored("Format: !watch <PAIR> [TIMEFRAME]", Fore.RED)
            elif cmd == '!unwatch':
                if len(parts) == 2:
                    pair_id = parts[1].upper()
                    if current_settings['watched_pairs'].pop(pair_id, None):
                        save_settings(); print_colored(f"{pair_id} dihapus.", Fore.YELLOW)
                    else: print_colored(f"{pair_id} tidak ditemukan.", Fore.RED)
                else: print_colored("Format: !unwatch <PAIR>", Fore.RED)
            elif cmd == '!watchlist':
                watched = current_settings.get("watched_pairs", {})
                if not watched: print_colored("Watchlist kosong.", Fore.YELLOW)
                else:
                    print_colored("\n--- Watchlist ---", Fore.CYAN, Style.BRIGHT)
                    for pair, tf in watched.items(): print_colored(f"- {pair} ({tf})", Fore.WHITE)
            # !history tidak diubah
            elif cmd in ['!settings', '!set']: handle_settings_command(parts)
            else: print_colored(f"Perintah '{cmd}' tidak dikenal.", Fore.RED)
        except (KeyboardInterrupt, EOFError): break
        except Exception as e: print_colored(f"\nError di main loop: {e}", Fore.RED)
    
    print_colored("\nMenutup aplikasi...", Fore.YELLOW)
    stop_event.set()
    autopilot_thread.join()
    data_thread.join()
    print_colored("Aplikasi berhasil ditutup.", Fore.CYAN)

if __name__ == "__main__":
    main()
