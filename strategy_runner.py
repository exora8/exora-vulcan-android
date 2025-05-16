import requests
# pandas diganti dengan list of dictionaries
import time
import json
import os
import logging
from datetime import datetime # datetime sudah diimport dengan benar
import smtplib # Untuk email
from email.mime.text import MIMEText # Untuk email
import sys # Untuk cek platform (beep)

# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m' # Warning / Late FIB
    RED = '\033[91m'    # Error / SL
    ENDC = '\033[0m'    # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[96m'

# --- KONFIGURASI LOGGING ---
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter(f'%(asctime)s - {AnsiColors.BOLD}%(levelname)s{AnsiColors.ENDC} - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = logging.FileHandler("trading_log.txt", mode='a')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(console_formatter)
logger.addHandler(ch)


SETTINGS_FILE = "settings.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999 # Maksimum limit data dari CryptoCompare API per request

# --- FUNGSI BEEP ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else: # Linux, macOS, Termux
            print('\a', end='', flush=True)
            time.sleep(0.2)
            print('\a', end='', flush=True)
    except Exception as e:
        logging.warning(f"Tidak bisa memainkan suara notifikasi: {e}")

# --- FUNGSI EMAIL ---
def send_email_notification(subject, body_text, settings):
    if not settings.get("enable_email_notifications", False):
        return

    sender_email = settings.get("email_sender_address")
    sender_password = settings.get("email_sender_app_password")
    receiver_email = settings.get("email_receiver_address")

    if not all([sender_email, sender_password, receiver_email]):
        logging.warning("Konfigurasi email tidak lengkap. Notifikasi email dilewati.")
        return

    msg = MIMEText(body_text)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        logging.info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}")
    except Exception as e:
        logging.error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}")


# --- FUNGSI PENGATURAN ---
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logging.error("Error membaca settings.json. Menggunakan default.")
    return {
        "api_key": "YOUR_API_KEY_HERE", "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,
        "left_strength": 50, "right_strength": 150,
        "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0,
        "emergency_sl_percent": 10.0, "enable_secure_fib": True, "secure_fib_check_price": "Close",
        "enable_email_notifications": False,
        "email_sender_address": "pengirim@gmail.com",
        "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address": "penerima@example.com"
    }

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
    logging.info(f"{AnsiColors.CYAN}Pengaturan disimpan ke settings.json{AnsiColors.ENDC}")

def settings_menu(current_settings):
    print(f"\n{AnsiColors.HEADER}--- Menu Pengaturan ---{AnsiColors.ENDC}")
    new_settings = current_settings.copy()
    try:
        new_settings["api_key"] = input(f"API Key CryptoCompare [{current_settings.get('api_key','')}]: ") or current_settings.get('api_key','')
        new_settings["symbol"] = (input(f"Simbol Crypto Dasar (misal BTC) [{current_settings.get('symbol','BTC')}]: ") or current_settings.get('symbol','BTC')).upper()
        new_settings["currency"] = (input(f"Simbol Mata Uang Quote (misal USDT, USD) [{current_settings.get('currency','USD')}]: ") or current_settings.get('currency','USD')).upper()
        new_settings["exchange"] = (input(f"Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{current_settings.get('exchange','CCCAGG')}]: ") or current_settings.get('exchange','CCCAGG'))
        tf_input = (input(f"Timeframe (minute/hour/day) [{current_settings.get('timeframe','hour')}]: ") or current_settings.get('timeframe','hour')).lower()
        if tf_input in ['minute', 'hour', 'day']: new_settings["timeframe"] = tf_input
        else: print("Timeframe tidak valid."); new_settings["timeframe"] = current_settings.get('timeframe','hour')
        new_settings["refresh_interval_seconds"] = int(input(f"Interval Refresh (detik) [{current_settings.get('refresh_interval_seconds',60)}]: ") or current_settings.get('refresh_interval_seconds',60))

        print(f"\n{AnsiColors.HEADER}-- Parameter Pivot --{AnsiColors.ENDC}")
        new_settings["left_strength"] = int(input(f"Left Strength [{current_settings.get('left_strength',50)}]: ") or current_settings.get('left_strength',50))
        new_settings["right_strength"] = int(input(f"Right Strength [{current_settings.get('right_strength',150)}]: ") or current_settings.get('right_strength',150))

        print(f"\n{AnsiColors.HEADER}-- Parameter Trading --{AnsiColors.ENDC}")
        new_settings["profit_target_percent_activation"] = float(input(f"Profit % Aktivasi Trailing TP [{current_settings.get('profit_target_percent_activation',5.0)}]: ") or current_settings.get('profit_target_percent_activation',5.0))
        new_settings["trailing_stop_gap_percent"] = float(input(f"Gap Trailing TP % [{current_settings.get('trailing_stop_gap_percent',5.0)}]: ") or current_settings.get('trailing_stop_gap_percent',5.0))
        new_settings["emergency_sl_percent"] = float(input(f"Emergency SL % [{current_settings.get('emergency_sl_percent',10.0)}]: ") or current_settings.get('emergency_sl_percent',10.0))
        
        print(f"\n{AnsiColors.HEADER}-- Fitur Secure FIB --{AnsiColors.ENDC}")
        enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{current_settings.get('enable_secure_fib',True)}]: ").lower()
        new_settings["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else current_settings.get('enable_secure_fib',True))
        secure_fib_price_input = (input(f"Harga Cek Secure FIB (Close/High) [{current_settings.get('secure_fib_check_price','Close')}]: ") or current_settings.get('secure_fib_check_price','Close')).capitalize()
        if secure_fib_price_input in ["Close", "High"]: new_settings["secure_fib_check_price"] = secure_fib_price_input
        else: print("Pilihan harga Secure FIB tidak valid."); new_settings["secure_fib_check_price"] = current_settings.get('secure_fib_check_price','Close')

        print(f"\n{AnsiColors.HEADER}-- Notifikasi Email (Gmail) --{AnsiColors.ENDC}")
        email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{current_settings.get('enable_email_notifications',False)}]: ").lower()
        new_settings["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else current_settings.get('enable_email_notifications',False))
        new_settings["email_sender_address"] = input(f"Email Pengirim (Gmail) [{current_settings.get('email_sender_address','')}]: ") or current_settings.get('email_sender_address','')
        new_settings["email_sender_app_password"] = input(f"App Password Email Pengirim [{current_settings.get('email_sender_app_password','')}]: ") or current_settings.get('email_sender_app_password','')
        new_settings["email_receiver_address"] = input(f"Email Penerima [{current_settings.get('email_receiver_address','')}]: ") or current_settings.get('email_receiver_address','')
        
        save_settings(new_settings)
        return new_settings
    except ValueError:
        print(f"{AnsiColors.RED}Input tidak valid. Pengaturan tidak diubah.{AnsiColors.ENDC}")
        return current_settings

# --- FUNGSI PENGAMBILAN DATA --- (MODIFIED: NO PANDAS)
def fetch_candles(symbol, currency, limit, exchange_name, api_key, timeframe="hour"):
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"
    else: api_endpoint = "histohour" # default ke hour
    
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    params = {"fsym": symbol, "tsym": currency, "limit": limit, "api_key": api_key}
    if exchange_name and exchange_name.upper() != "CCCAGG":
        params["e"] = exchange_name
    
    try:
        logging.debug(f"Fetching data from: {url} with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status() 
        data = response.json()

        if data.get('Response') == 'Error':
            logging.error(f"{AnsiColors.RED}API Error CryptoCompare: {data.get('Message', 'N/A')}{AnsiColors.ENDC} (Params: fsym={symbol}, tsym={currency}, exch={exchange_name or 'CCCAGG'}, lim={limit}, tf={timeframe})")
            return [] # Return empty list
        
        if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
            logging.info("Tidak ada data candle dari API atau format data tidak sesuai.")
            return [] # Return empty list
            
        candles_list = []
        for item in data['Data']['Data']:
            candle = {
                'timestamp': datetime.fromtimestamp(item['time']), # Konversi UNIX timestamp ke datetime
                'open': item.get('open'),
                'high': item.get('high'),
                'low': item.get('low'),
                'close': item.get('close'),
                'volume': item.get('volumefrom') # 'volumefrom' adalah nama kolom volume dari API
            }
            # Memastikan semua nilai OHLCV ada, jika tidak ada akan None dari .get()
            # Strategi harus bisa menangani None jika API tidak lengkap (meski jarang untuk OHLC)
            candles_list.append(candle)
        
        return candles_list # Return list of dictionaries

    except requests.exceptions.RequestException as e:
        logging.error(f"{AnsiColors.RED}Kesalahan koneksi: {e}{AnsiColors.ENDC}")
        return [] # Return empty list
    except Exception as e:
        logging.error(f"{AnsiColors.RED}Error fetch_candles: {e}{AnsiColors.ENDC}")
        return [] # Return empty list


# --- LOGIKA STRATEGI --- (MODIFIED: NO PANDAS)
strategy_state = {
    "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
    "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None,
    "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None,
    "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
    "emergency_sl_level_custom": None, "position_size": 0,
}

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1:
        return pivots
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        # Cek candle di kiri
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None or series_list[i] is None: # Handle None values
                is_pivot = False; break
            if is_high:
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # is_low
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue
        # Cek candle di kanan
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None or series_list[i] is None: # Handle None values
                is_pivot = False; break
            if is_high:
                if series_list[i] < series_list[i+j]: is_pivot = False; break 
            else: # is_low
                if series_list[i] > series_list[i+j]: is_pivot = False; break 
        if is_pivot:
            pivots[i] = series_list[i] 
    return pivots

def run_strategy_logic(candles_history, settings): # candles_history adalah list of dict
    global strategy_state 
    
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None
    
    left_strength = settings['left_strength']
    right_strength = settings['right_strength']
    
    required_keys = ['high', 'low', 'open', 'close']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history[0]):
        logging.warning(f"{AnsiColors.ORANGE}Data candle kosong/kurang kunci di run_strategy_logic.{AnsiColors.ENDC}")
        return

    # Ekstrak series harga High dan Low dari list of dictionaries
    high_prices = [c['high'] for c in candles_history]
    low_prices = [c['low'] for c in candles_history]

    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, is_high=True)
    raw_pivot_lows  = find_pivots(low_prices,  left_strength, right_strength, is_high=False)
    
    current_bar_index_in_list = len(candles_history) - 1 
    if current_bar_index_in_list < 0 : return # Seharusnya sudah dicek oleh 'not candles_history'

    # Index di mana event pivot terkonfirmasi (setelah right_strength bar)
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    raw_pivot_high_price_at_event = None
    if 0 <= idx_pivot_event_high < len(raw_pivot_highs):
        raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high]

    idx_pivot_event_low = current_bar_index_in_list - right_strength
    raw_pivot_low_price_at_event = None
    if 0 <= idx_pivot_event_low < len(raw_pivot_lows):
        raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low]

    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["last_signal_type"] = 1
        pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
        logging.info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}")
        
    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        logging.info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}")

    current_candle = candles_history[current_bar_index_in_list] # Ini adalah dictionary candle terakhir
    # Pastikan semua harga ada di current_candle
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        logging.warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp')}. Skip evaluasi.")
        return

    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high
        if strategy_state["active_fib_level"] is not None:
            logging.debug("Resetting active FIB due to new High.")
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0
                is_fib_late = False
                if settings["enable_secure_fib"]:
                    price_to_check_key = settings["secure_fib_check_price"].lower() # 'close' atau 'high'
                    if price_to_check_key not in current_candle or current_candle[price_to_check_key] is None:
                        price_to_check_key = 'close' # Fallback
                    
                    price_val_current_candle = current_candle[price_to_check_key]
                    if price_val_current_candle is not None and price_val_current_candle > calculated_fib_level:
                        is_fib_late = True
                
                if is_fib_late:
                    logging.info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({settings['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}")
                    strategy_state["active_fib_level"] = None; strategy_state["active_fib_line_start_index"] = None
                else:
                    logging.info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})")
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = current_low_bar_index
                strategy_state["high_price_for_fib"] = None; strategy_state["high_bar_index_for_fib"] = None

    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]
        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: 
                strategy_state["position_size"] = 1 
                entry_px = current_candle['close']
                strategy_state["entry_price_custom"] = entry_px
                strategy_state["highest_price_for_trailing"] = entry_px 
                strategy_state["trailing_tp_active_custom"] = False
                strategy_state["current_trailing_stop_level"] = None
                emerg_sl = entry_px * (1 - settings["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl
                
                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                logging.info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}")
                play_notification_sound()
                email_subject = f"BUY Signal: {settings['symbol']}-{settings['currency']}"
                email_body = (f"New BUY signal triggered for {settings['symbol']}-{settings['currency']} on {settings['exchange']}.\n\n"
                             f"Entry Price: {entry_px:.5f}\n"
                             f"FIB Level: {strategy_state['active_fib_level']:.5f}\n"
                             f"Emergency SL: {emerg_sl:.5f}\n"
                             f"Timestamp: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, settings)
            
            strategy_state["active_fib_level"] = None; strategy_state["active_fib_line_start_index"] = None

    if strategy_state["position_size"] > 0:
        # Pastikan highest_price_for_trailing adalah float, bukan None
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle['high'])
        if current_high_for_trailing is None: current_high_for_trailing = current_candle['high'] # Fallback if None

        strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])
        
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            profit_percent = 0
            if strategy_state["entry_price_custom"] != 0: # Hindari ZeroDivisionError
                 profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= settings["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                logging.info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}")

        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"] is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (settings["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                logging.debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}")
        
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE 
        
        if final_stop_for_exit is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price = min(current_candle['open'], final_stop_for_exit) 
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                 pnl = (exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"] * 100.0
            
            if exit_comment == "Trailing Stop" and pnl < 0: exit_color = AnsiColors.RED

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            logging.info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}")
            play_notification_sound()
            email_subject = f"Trade Closed: {settings['symbol']}-{settings['currency']} ({exit_comment})"
            email_body = (f"Trade closed for {settings['symbol']}-{settings['currency']} on {settings['exchange']}.\n\n"
                         f"Exit Price: {exit_price:.5f}\n"
                         f"Reason: {exit_comment}\n"
                         f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\n"
                         f"PnL: {pnl:.2f}%\n"
                         f"Timestamp: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, settings)

            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
    
    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            emergency_sl = strategy_state.get("emergency_sl_level_custom")
            current_trailing_sl = strategy_state.get("current_trailing_stop_level")
            if emergency_sl is not None and current_trailing_sl is not None and current_trailing_sl > emergency_sl: plot_stop_level = current_trailing_sl
            elif current_trailing_sl is not None and emergency_sl is None: plot_stop_level = current_trailing_sl
        
        entry_price_display = strategy_state.get('entry_price_custom', 0)
        sl_display_str = f'{plot_stop_level:.5f}' if plot_stop_level is not None else 'N/A'
        logging.debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, Current SL: {sl_display_str}")


# --- FUNGSI UTAMA TRADING LOOP --- (MODIFIED: NO PANDAS)
def start_trading(settings):
    display_pair = f"{settings.get('symbol','N/A')}-{settings.get('currency','N/A')}"
    display_exchange = settings.get('exchange','N/A')
    display_timeframe = settings.get('timeframe','N/A')
    refresh_interval = settings.get('refresh_interval_seconds',0)
    
    logging.info(f"{AnsiColors.HEADER}================ STRATEGY START ================{AnsiColors.ENDC}")
    logging.info(f"Pair: {AnsiColors.BOLD}{display_pair}{AnsiColors.ENDC} | Exchange: {AnsiColors.BOLD}{display_exchange}{AnsiColors.ENDC} | TF: {AnsiColors.BOLD}{display_timeframe}{AnsiColors.ENDC} | Refresh: {AnsiColors.BOLD}{refresh_interval}s{AnsiColors.ENDC}")
    # ... (logging info lainnya)
    logging.info(f"Params: LeftStr={settings.get('left_strength',0)}, RightStr={settings.get('right_strength',0)}, TrailActiv={settings.get('profit_target_percent_activation',0.0)}%, TrailGap={settings.get('trailing_stop_gap_percent',0.0)}%, EmergSL={settings.get('emergency_sl_percent',0.0)}%")
    logging.info(f"SecureFIB: {settings.get('enable_secure_fib',False)}, CheckPrice: {settings.get('secure_fib_check_price','N/A')}")
    if settings.get('enable_email_notifications'):
        logging.info(f"Email Notif: {AnsiColors.GREEN}Aktif{AnsiColors.ENDC} (Ke: {settings.get('email_receiver_address')})")
    else:
        logging.info(f"Email Notif: {AnsiColors.ORANGE}Nonaktif{AnsiColors.ENDC}")
    logging.info(f"{AnsiColors.HEADER}==============================================={AnsiColors.ENDC}")


    if settings.get('api_key',"") == "YOUR_API_KEY_HERE" or not settings.get('api_key',""):
        logging.error(f"{AnsiColors.RED}API Key belum diatur! Atur via menu Settings.{AnsiColors.ENDC}")
        return

    global strategy_state 
    strategy_state = { # Reset state
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None,
        "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
    }
    
    all_candles_list = fetch_candles(settings.get('symbol'), settings.get('currency'), CRYPTOCOMPARE_MAX_LIMIT, 
                                     settings.get('exchange'), settings.get('api_key'), settings.get('timeframe'))

    if not all_candles_list:
        logging.error(f"{AnsiColors.RED}Tidak ada data awal. Periksa setting & koneksi. Menghentikan.{AnsiColors.ENDC}")
        return

    logging.info(f"Memproses {max(0, len(all_candles_list) - 1)} candle historis awal untuk inisialisasi state...")
    # Indeks awal untuk pemrosesan pemanasan, butuh cukup data untuk pivot
    start_warmup_processing_idx = settings.get('left_strength',50) + settings.get('right_strength',150)
    # Pemanasan: proses data historis kecuali candle terakhir (untuk disimulasikan sebagai 'live' pertama)
    for i in range(start_warmup_processing_idx, len(all_candles_list) -1): 
        historical_slice = all_candles_list[:i+1] # Slice dari awal hingga candle ke-i
        if len(historical_slice) < start_warmup_processing_idx +1 : continue # Slice terlalu pendek
        run_strategy_logic(historical_slice, settings)
        if strategy_state["position_size"] > 0: # Reset jika ada trade saat pemanasan
            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None 
            strategy_state["emergency_sl_level_custom"] = None # Reset SL juga
            # Reset state lainnya yang relevan dengan posisi
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            
    logging.info(f"{AnsiColors.CYAN}Inisialisasi state selesai.{AnsiColors.ENDC}")
    logging.info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ----------{AnsiColors.ENDC}")

    try:
        while True:
            current_loop_time = datetime.now()
            logging.info(f"\n{AnsiColors.BOLD}--- Analisa Candle Baru ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) ---{AnsiColors.ENDC}")

            last_candle_ts_before_fetch = all_candles_list[-1]['timestamp'] if all_candles_list else None
            
            new_candles_list = fetch_candles(settings.get('symbol'), settings.get('currency'), CRYPTOCOMPARE_MAX_LIMIT, 
                                             settings.get('exchange'), settings.get('api_key'), settings.get('timeframe'))
            
            if not new_candles_list:
                logging.warning(f"{AnsiColors.ORANGE}Gagal mengambil data baru atau tidak ada data baru saat ini. Mencoba lagi...{AnsiColors.ENDC}")
                time.sleep(settings.get('refresh_interval_seconds',15))
                continue
            
            # Gabungkan data lama dan baru, hapus duplikat berdasarkan timestamp (ambil yang terbaru), lalu urutkan
            merged_candles_dict = {c['timestamp']: c for c in all_candles_list} # Data lama ke dict
            for candle in new_candles_list: # Timpa/tambah dengan data baru
                merged_candles_dict[candle['timestamp']] = candle
            all_candles_list = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp']) # Kembalikan ke list terurut

            # Tentukan indeks awal untuk pemrosesan di all_candles_list yang sudah digabung
            processing_start_index = 0
            if last_candle_ts_before_fetch:
                idx_of_last_before_fetch = -1
                for i_loop, candle in enumerate(all_candles_list):
                    if candle['timestamp'] == last_candle_ts_before_fetch:
                        idx_of_last_before_fetch = i_loop
                        break
                if idx_of_last_before_fetch != -1:
                    processing_start_index = idx_of_last_before_fetch + 1 # Mulai dari candle setelahnya
                else:
                    logging.warning(f"{AnsiColors.ORANGE}Timestamp {last_candle_ts_before_fetch} tidak ditemukan setelah merge. Fallback: proses beberapa candle terakhir.{AnsiColors.ENDC}")
                    # Fallback: proses 5 candle terakhir, tapi pastikan ada cukup histori
                    min_hist_len = settings.get('left_strength',50) + settings.get('right_strength',150)
                    processing_start_index = max(min_hist_len, len(all_candles_list) - 5)
                    processing_start_index = max(0, processing_start_index) # Pastikan tidak negatif
            else:
                # Jika tidak ada last_candle_ts_before_fetch (misal, all_candles_list awalnya kosong)
                # Proses dari indeks yang cukup untuk histori pivot
                idx = max(0, len(all_candles_list) - int(CRYPTOCOMPARE_MAX_LIMIT / 2))
                min_hist_needed_from_start = settings.get('left_strength',50) + settings.get('right_strength',150)
                processing_start_index = max(idx, min_hist_needed_from_start)


            if processing_start_index >= len(all_candles_list):
                ts_display = all_candles_list[-1]['timestamp'].strftime('%H:%M:%S') if all_candles_list else "N/A"
                logging.info(f"Tidak ada candle baru untuk diproses (data terakhir @ {ts_display}). Menunggu {settings.get('refresh_interval_seconds',15)} detik...")
            else:
                num_candles_to_process = len(all_candles_list) - processing_start_index
                start_ts_str = all_candles_list[processing_start_index]['timestamp'].strftime('%H:%M')
                end_ts_str = all_candles_list[-1]['timestamp'].strftime('%H:%M')
                logging.info(f"Memproses {AnsiColors.BOLD}{num_candles_to_process}{AnsiColors.ENDC} candle (dari indeks {processing_start_index}: {start_ts_str} hingga {end_ts_str}).")
                
                for i_slice_end_idx in range(processing_start_index, len(all_candles_list)):
                    # Slice adalah dari awal list hingga candle ke i_slice_end_idx
                    current_processing_slice = all_candles_list[:i_slice_end_idx + 1] 
                    
                    min_len_for_pivots = settings.get('left_strength',50) + settings.get('right_strength',150) + 1
                    if len(current_processing_slice) < min_len_for_pivots: 
                        # logging.debug(f"Slice ke {i_slice_end_idx} terlalu pendek ({len(current_processing_slice)}/{min_len_for_pivots}). Skip.")
                        continue 
                    # logging.debug(f"Menganalisa slice berakhir di: {current_processing_slice[-1]['timestamp'].strftime('%H:%M')} (Close: {current_processing_slice[-1]['close']:.5f})")
                    run_strategy_logic(current_processing_slice, settings)
            
            # Batasi ukuran all_candles_list agar tidak terlalu besar seiring waktu
            # Pertahankan CRYPTOCOMPARE_MAX_LIMIT * 1.5 atau 2x candle, misalnya
            max_retained_candles = int(CRYPTOCOMPARE_MAX_LIMIT * 1.5)
            if len(all_candles_list) > max_retained_candles:
                all_candles_list = all_candles_list[-max_retained_candles:]
                logging.debug(f"Ukuran all_candles_list dipangkas menjadi {len(all_candles_list)} candle.")

            logging.info(f"{AnsiColors.BOLD}--- Selesai Loop Analisa. Menunggu {settings.get('refresh_interval_seconds',15)} detik ---{AnsiColors.ENDC}")
            time.sleep(settings.get('refresh_interval_seconds',15))

    except KeyboardInterrupt:
        logging.info(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}")
    except Exception as e:
        logging.exception(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}")
    finally:
        logging.info(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    while True:
        display_pair = f"{settings.get('symbol','N/A')}-{settings.get('currency','N/A')}"
        display_exchange = settings.get('exchange','N/A')
        display_timeframe = settings.get('timeframe','N/A')
        refresh_interval = settings.get('refresh_interval_seconds',0)
        print(f"\n{AnsiColors.HEADER}========= Crypto Strategy Runner ========={AnsiColors.ENDC}")
        print(f"Pair: {AnsiColors.BOLD}{display_pair}{AnsiColors.ENDC} | Exch: {AnsiColors.BOLD}{display_exchange}{AnsiColors.ENDC} | TF: {AnsiColors.BOLD}{display_timeframe}{AnsiColors.ENDC} | Int: {AnsiColors.BOLD}{refresh_interval}s{AnsiColors.ENDC}")
        print("--------------------------------------")
        print(f"1. {AnsiColors.GREEN}Mulai Analisa Realtime{AnsiColors.ENDC}")
        print(f"2. {AnsiColors.ORANGE}Pengaturan{AnsiColors.ENDC}")
        print(f"3. {AnsiColors.RED}Keluar{AnsiColors.ENDC}")
        choice = input("Pilihan Anda: ")

        if choice == '1':
            start_trading(settings)
        elif choice == '2':
            settings = settings_menu(settings)
        elif choice == '3':
            logging.info("Aplikasi ditutup.")
            break
        else:
            print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")

if __name__ == "__main__":
    main_menu()
