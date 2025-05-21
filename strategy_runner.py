import requests
import time
import json
import os
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import sys
import uuid
from pick import pick
import subprocess # Ditambahkan untuk termux-notification

# CHART_INTEGRATION_START
import threading
import copy # Untuk deep copy data agar thread-safe (dasar)
try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    print("Flask tidak terinstal. Charting tidak akan tersedia. Lanjutkan dengan: pip install Flask")
    # Tidak keluar dari skrip jika hanya untuk charting opsional di Termux
    # sys.exit(1) 
    pass # Biarkan skrip jalan, tapi charting mungkin tidak berfungsi
# CHART_INTEGRATION_END

# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m' # Warning / GR1
    RED = '\033[91m'    # Error / SL
    ENDC = '\033[0m'    # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[96m'
    MAGENTA = '\033[35m'
    YELLOW_BG = '\033[43m' # GR2

# --- ANIMATION HELPER FUNCTIONS ---
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True):
    for char in text:
        sys.stdout.write(color + char + AnsiColors.ENDC if color else char)
        sys.stdout.flush()
        time.sleep(delay)
    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing..."):
    spinner_chars = ['-', '\\', '|', '/']
    start_time = time.time()
    idx = 0
    sys.stdout.write(AnsiColors.MAGENTA)
    term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            pass

    while (time.time() - start_time) < duration_seconds:
        display_message = message[:term_width - 5] # Truncate message if too long
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") # Clear spinner line
    sys.stdout.write(AnsiColors.ENDC)
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            pass

    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    sys.stdout.write(progress_line[:term_width]) # Truncate if too long
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Default INFO, bisa diubah ke DEBUG jika perlu
if logger.hasHandlers():
    logger.handlers.clear()

log_file_name = "trading_log_ema_gr.txt" # Nama log file baru
fh = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
console_formatter_template = '%(asctime)s - {bold}%(levelname)s{endc} - {cyan}[%(pair_name)s]{endc} - %(message)s'
console_formatter = logging.Formatter(
    console_formatter_template.format(bold=AnsiColors.BOLD, endc=AnsiColors.ENDC, cyan=AnsiColors.CYAN)
)
ch.setFormatter(console_formatter)
logger.addHandler(ch)

class AddPairNameFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'pair_name'):
            record.pair_name = 'SYSTEM'
        return True
logger.addFilter(AddPairNameFilter())

def log_info(message, pair_name="SYSTEM"): logger.info(message, extra={'pair_name': pair_name})
def log_warning(message, pair_name="SYSTEM"): logger.warning(message, extra={'pair_name': pair_name})
def log_error(message, pair_name="SYSTEM"): logger.error(message, extra={'pair_name': pair_name})
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name})
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})


SETTINGS_FILE = "settings_ema_gr_strategy.json" # Nama file setting baru
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500 
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY":
            self.keys.append(primary_key)
        if recovery_keys_list:
            self.keys.extend([k for k in recovery_keys_list if k])

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}

        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys: return None
        return self.keys[self.current_index] if self.current_index < len(self.keys) else None

    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # (Email notifikasi pergantian key seperti di skrip asli)
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL!{AnsiColors.ENDC}")
            # (Email notifikasi kritis seperti di skrip asli)
            return None

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            try: # Coba Termux Vibrate dulu
                subprocess.run(['termux-vibrate', '-d', '300'], timeout=0.5, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.TimeoutExpired): # Jika gagal, coba bell
                print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False): return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")
    pair_name_ctx = settings_for_email.get('pair_name', 'GLOBAL_EMAIL')

    if not all([sender_email, sender_password, receiver_email]):
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=pair_name_ctx)
        return

    msg = MIMEText(body_text)
    msg['Subject'] = subject; msg['From'] = sender_email; msg['To'] = receiver_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)

def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    if not global_settings.get("api_settings", {}).get("enable_termux_notifications", False): return
    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,
        "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0,
        "emergency_sl_percent": 10.0,
        "enable_email_notifications": False,
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": "",
        "ema_length": 200, "gr2_drop_perc": 15.0,
    }

def load_settings():
    default_api_settings = {
        "primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "", "email_sender_app_password": "",
        "email_receiver_address_admin": "", "enable_termux_notifications": False
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            # Merge API settings with defaults
            settings["api_settings"] = {**default_api_settings, **settings.get("api_settings", {})}
            # Merge individual crypto configs with defaults
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list): settings["cryptos"] = []
            
            updated_cryptos = []
            for crypto_cfg in settings["cryptos"]:
                default_pair_cfg = get_default_crypto_config()
                # Pastikan semua field dari default ada di config yang dimuat
                # Ini juga akan menambahkan field baru seperti ema_length jika file setting lama
                merged_cfg = {**default_pair_cfg, **crypto_cfg}
                if "id" not in merged_cfg or not merged_cfg["id"]: merged_cfg["id"] = str(uuid.uuid4()) # Pastikan ID ada
                updated_cryptos.append(merged_cfg)
            settings["cryptos"] = updated_cryptos
            return settings
        except json.JSONDecodeError:
            log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan: {e}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): # Versi lebih ringkas
    clear_screen_animated()
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    def get_input(prompt_text, current_value, type_converter=str, color=AnsiColors.BLUE):
        val_str = input(f"{color}{prompt_text} [{current_value}]: {AnsiColors.ENDC}").strip()
        if not val_str: return current_value
        try: return type_converter(val_str)
        except ValueError:
            print(f"{AnsiColors.RED}Input tidak valid. Menggunakan nilai sebelumnya: {current_value}{AnsiColors.ENDC}")
            return current_value

    enabled_input = input(f"Aktifkan pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    if enabled_input == 'true': new_config["enabled"] = True
    elif enabled_input == 'false': new_config["enabled"] = False

    new_config["symbol"] = get_input("Simbol Crypto (misal BTC)", new_config.get('symbol','BTC')).upper()
    new_config["currency"] = get_input("Mata Uang Quote (misal USDT)", new_config.get('currency','USD')).upper()
    new_config["exchange"] = get_input("Exchange (CCCAGG untuk agregat)", new_config.get('exchange','CCCAGG'))
    
    tf_input = get_input("Timeframe (minute/hour/day)", new_config.get('timeframe','hour')).lower()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}")

    new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, get_input("Interval Refresh (detik)", new_config.get('refresh_interval_seconds',60), int))

    animated_text_display("\n-- Parameter Trading (TP/SL) --", color=AnsiColors.HEADER, delay=0.01)
    new_config["profit_target_percent_activation"] = get_input("Profit % Aktivasi Trailing TP", new_config.get('profit_target_percent_activation',5.0), float)
    new_config["trailing_stop_gap_percent"] = get_input("Gap Trailing TP %", new_config.get('trailing_stop_gap_percent',5.0), float)
    new_config["emergency_sl_percent"] = get_input("Emergency SL %", new_config.get('emergency_sl_percent',10.0), float, color=AnsiColors.RED)

    animated_text_display("\n-- Parameter Strategi EMA GR1/GR2 --", color=AnsiColors.HEADER, delay=0.01)
    new_config["ema_length"] = get_input("Panjang EMA", new_config.get('ema_length',200), int)
    new_config["gr2_drop_perc"] = get_input("GR2 - Penurunan % dari GR1", new_config.get('gr2_drop_perc',15.0), float)

    # ... (bagian email tetap sama seperti di skrip asli)
    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    # ...
    return new_config

def settings_menu(current_settings): # (Tetap sama seperti di skrip asli, hanya UI)
    # ... (Implementasi settings_menu dari skrip asli bisa disalin ke sini) ...
    # Pastikan _prompt_crypto_config yang baru dipanggil
    # Tidak akan saya sertakan ulang di sini agar tidak terlalu panjang, tapi strukturnya sama
    # Hanya saja prompt akan memanggil _prompt_crypto_config yang sudah dimodifikasi
    # Ini adalah contoh singkatnya, asumsikan sisanya dari skrip asli
    while True:
        clear_screen_animated()
        # ... (Tampilkan menu utama pengaturan) ...
        # Contoh:
        options = ["Atur API", "Kelola Crypto", "Kembali"] # Disederhanakan
        _text, choice = pick(options, "Pengaturan:", indicator="=>")

        if choice == 0: # Atur API
            # ... (logika atur API Key, email global, termux notif)
            pass
        elif choice == 1: # Kelola Crypto
            while True:
                # ... (Tampilkan daftar crypto, opsi Tambah/Ubah/Hapus)
                crypto_options = ["Tambah", "Ubah", "Hapus", "Kembali"]
                _ctext, cchoice = pick(crypto_options, "Kelola Crypto Pairs:", indicator="=>")
                if cchoice == 0: # Tambah
                    new_conf = get_default_crypto_config()
                    current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(new_conf))
                    save_settings(current_settings)
                elif cchoice == 1: # Ubah (pilih dulu)
                    if current_settings.get("cryptos"):
                        # ... (logika pilih crypto untuk diubah)
                        # current_settings["cryptos"][idx_ubah] = _prompt_crypto_config(current_settings["cryptos"][idx_ubah])
                        # save_settings(current_settings)
                        pass
                elif cchoice == 2: # Hapus (pilih dulu)
                    # ... (logika pilih crypto untuk dihapus)
                    pass
                elif cchoice == 3: break # Kembali ke menu pengaturan utama
        elif choice == 2: break # Kembali ke menu utama skrip
    return current_settings


# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    # (Fungsi fetch_candles dari skrip asli bisa langsung dipakai di sini, tidak perlu diubah karena tidak tergantung pandas)
    # ... (Salin implementasi fetch_candles dari skrip asli) ...
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None # Timestamp tertua yang sudah diambil, untuk request berikutnya
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 20 # Anggap >20 sebagai large fetch untuk logging

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)

    # Tampilkan progress bar jika fetch besar dan butuh beberapa panggilan API
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch :
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=30)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # CryptoCompare punya limit max 2000 per request.
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)

        if limit_for_this_api_call <= 0: break # Sudah cukup

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts # Ambil data SEBELUM timestamp ini

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                 key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                 log_debug(f"Fetching batch (Key: {key_display}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20) # Timeout 20 detik

            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass # abaikan jika response bukan json
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() # Akan raise error untuk status 4xx atau 5xx
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'Pesan error tidak tersedia dari API')
                key_related_error_messages = [ # Daftar kata kunci error terkait API Key
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active", "call_limit_reached",
                    "you are over your rate limit", "please pass an API key", "api_key not found"
                ]
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else: # Error API lain
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Hentikan loop untuk pair ini jika ada error API non-key

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format tidak sesuai. Total: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break # Tidak ada data lagi

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: # Seharusnya sudah ditangani di atas
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                required_ohlcv_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in required_ohlcv_keys):
                    log_warning(f"Candle tidak lengkap dari API @ ts {item.get('time', 'N/A')}. Dilewati. Data: {item}", pair_name=pair_name)
                    continue
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom')
                }
                batch_candles_list.append(candle)

            # Hapus candle duplikat jika ada (karena toTs inklusif)
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                # batch_candles_list diurutkan dari tua ke muda (API mengembalikan begitu)
                # all_accumulated_candles diurutkan dari tua ke muda juga (karena kita prepend)
                # Yang paling muda di batch_candles_list adalah batch_candles_list[-1]
                # Yang paling tua di all_accumulated_candles adalah all_accumulated_candles[0]
                # Jika kita fetch data LAMA, maka candle TERAKHIR dari batch_candles_list (paling muda di batch itu)
                # bisa jadi sama dengan candle PERTAMA dari all_accumulated_candles (paling tua di list utama)
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                     if is_large_fetch: log_debug(f"Overlap candle: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                     batch_candles_list.pop() # Hapus yang paling muda dari batch baru


            if not batch_candles_list and current_to_ts is not None : # Jika batch jadi kosong setelah overlap
                if is_large_fetch: log_info("Batch kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles # Prepend (tambahkan di depan)

            if raw_candles_from_api: # Jika ada data di batch ini
                current_to_ts = raw_candles_from_api[0]['time'] # Ambil timestamp PALING TUA dari batch ini untuk request berikutnya
            else: # Seharusnya tidak terjadi jika cek di atas benar
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired):
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=30)

            if len(raw_candles_from_api) < limit_for_this_api_call: # API mengembalikan kurang dari yang diminta
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori.", pair_name=pair_name)
                break # Akhir dari data historis

            if len(all_accumulated_candles) >= total_limit_desired: break # Target tercapai

            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                time.sleep(0.3) # Delay kecil antar batch besar

        except APIKeyError: # Tangkap dan lempar lagi agar bisa ditangani di loop utama per pair
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Hentikan loop untuk pair ini
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error Fetch Candles (batch loop):", pair_name=pair_name)
            break

    if len(all_accumulated_candles) > total_limit_desired: # Pastikan tidak kelebihan
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Selesaikan progress bar jika ada
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=30)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
        "in_get_ready_1": False, "in_get_ready_2": False, "price_at_gr1_close": None,
    }

# Fungsi EMA Manual (tanpa Pandas)
def calculate_ema_manual(prices, length):
    if not prices or length <= 0 or len(prices) < 1:
        return []
    alpha = 2 / (length + 1)
    ema_values = [prices[0]] # Inisialisasi dengan harga pertama
    for i in range(1, len(prices)):
        current_ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(current_ema)
    return ema_values


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    required_keys = ['high', 'low', 'open', 'close', 'timestamp']

    if not candles_history or len(candles_history) < 2:
        log_debug(f"Data candle ({len(candles_history)}) < 2, skip logic.", pair_name=pair_name)
        return strategy_state
        
    if not all(key in candles_history[-1] for key in required_keys if candles_history[-1]):
        log_warning(f"Data candle terbaru tidak lengkap.", pair_name=pair_name)
        return strategy_state
    
    current_candle = candles_history[-1]
    previous_actual_candle = candles_history[-2] # Candle aktual sebelumnya

    if any(current_candle.get(k) is None for k in required_keys) or \
       any(previous_actual_candle.get(k) is None for k in required_keys):
        log_warning(f"Data OHLC tidak lengkap untuk candle saat ini atau sebelumnya. Skip.", pair_name=pair_name)
        return strategy_state

    ema_length = crypto_config.get('ema_length', 200)
    gr2_drop_perc_config = crypto_config.get('gr2_drop_perc', 15.0) / 100.0

    current_close = current_candle['close']
    current_low = current_candle['low']
    previous_actual_close_for_cross = previous_actual_candle['close']
    
    close_prices_for_ema_calc = [c['close'] for c in candles_history if c and 'close' in c and c['close'] is not None]
    
    if len(close_prices_for_ema_calc) < max(2, ema_length): # Butuh cukup data untuk EMA yang "stabil" dan untuk cross
        log_debug(f"Data harga ({len(close_prices_for_ema_calc)}) belum cukup untuk EMA {ema_length} atau cross. Min: {max(2, ema_length)}", pair_name=pair_name)
        return strategy_state
    
    ema_series = calculate_ema_manual(close_prices_for_ema_calc, ema_length)

    if len(ema_series) < 2: # Perlu setidaknya 2 nilai EMA (current dan previous) dari seri yang dihitung
        log_debug(f"Seri EMA ({len(ema_series)}) tidak cukup untuk cross (min 2).", pair_name=pair_name)
        return strategy_state

    ema_value_current_candle = ema_series[-1]
    ema_value_previous_candle = ema_series[-2] # EMA dari bar sebelumnya relatif terhadap seri EMA yang baru dihitung

    is_cross_under_ema = previous_actual_close_for_cross > ema_value_previous_candle and current_close < ema_value_current_candle
    is_cross_over_ema = previous_actual_close_for_cross < ema_value_previous_candle and current_close > ema_value_current_candle

    # --- GR1 Condition ---
    if (is_cross_under_ema and strategy_state["position_size"] == 0 and
        not strategy_state.get("in_get_ready_1") and not strategy_state.get("in_get_ready_2")):
        strategy_state["in_get_ready_1"] = True
        strategy_state["in_get_ready_2"] = False
        strategy_state["price_at_gr1_close"] = current_close
        log_info(f"{AnsiColors.ORANGE}GR1 Aktif. EMA xUnder. Harga GR1: {current_close:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

    # --- GR2 Condition ---
    if (strategy_state.get("in_get_ready_1") and not strategy_state.get("in_get_ready_2")):
        price_at_gr1 = strategy_state.get("price_at_gr1_close")
        if price_at_gr1 and price_at_gr1 > 0:
            current_drop = (price_at_gr1 - current_low) / price_at_gr1
            if current_drop >= gr2_drop_perc_config:
                strategy_state["in_get_ready_2"] = True
                strategy_state["in_get_ready_1"] = False
                log_info(f"{AnsiColors.YELLOW_BG}{AnsiColors.BLUE}GR2 Aktif. Drop {current_drop*100:.2f}% dari GR1 ({price_at_gr1:.5f}) ke Low ({current_low:.5f}).{AnsiColors.ENDC}", pair_name=pair_name)

    # --- Entry Condition ---
    if (strategy_state.get("in_get_ready_2") and is_cross_over_ema and strategy_state["position_size"] == 0):
        entry_px = current_close 
        strategy_state["position_size"] = 1
        strategy_state["entry_price_custom"] = entry_px
        strategy_state["highest_price_for_trailing"] = entry_px # Reset for new trade
        strategy_state["trailing_tp_active_custom"] = False
        strategy_state["current_trailing_stop_level"] = None

        sl_perc = crypto_config.get("emergency_sl_percent", 10.0)
        emerg_sl = entry_px * (1 - sl_perc / 100.0)
        strategy_state["emergency_sl_level_custom"] = emerg_sl

        log_msg = f"BUY ENTRY @ {entry_px:.5f} (EMA xOver pasca GR2). SL: {emerg_sl:.5f}"
        log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
        play_notification_sound()
        # (Notifikasi Termux & Email seperti sebelumnya)
        termux_title = f"BUY Signal: {pair_name}"
        termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}. EMA GR1/GR2 Logic."
        send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)
        # ... (email notif)

        strategy_state["in_get_ready_2"] = False # Reset GR2 setelah entry

    # --- Exit Logic (SL/TP - dari Exora asli) ---
    if strategy_state["position_size"] > 0:
        # Update highest price
        strategy_state["highest_price_for_trailing"] = max(strategy_state.get("highest_price_for_trailing", current_candle['high']), current_candle['high'])
        
        # Activate trailing TP
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"]:
            profit_perc = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if profit_perc >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit {profit_perc:.2f}%{AnsiColors.ENDC}", pair_name=pair_name)

        # Update trailing SL
        if strategy_state["trailing_tp_active_custom"]:
            new_trail_sl = strategy_state["highest_price_for_trailing"] * (1 - crypto_config["trailing_stop_gap_percent"] / 100.0)
            if strategy_state["current_trailing_stop_level"] is None or new_trail_sl > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = new_trail_sl
                log_debug(f"Trailing SL update: {new_trail_sl:.5f}", pair_name=pair_name)
        
        # Determine SL for this bar
        final_sl = strategy_state["emergency_sl_level_custom"]
        exit_reason = "Emergency SL"
        exit_clr = AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"]:
            if final_sl is None or strategy_state["current_trailing_stop_level"] > final_sl:
                final_sl = strategy_state["current_trailing_stop_level"]
                exit_reason = "Trailing Stop"
                exit_clr = AnsiColors.BLUE

        # Check SL hit
        if final_sl and current_candle['low'] <= final_sl:
            exit_px = final_sl # Assume executed at SL level
            pnl_val = ((exit_px - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if exit_reason == "Trailing Stop" and pnl_val < 0: exit_clr = AnsiColors.RED; exit_reason = "Trailing Stop (Loss)"

            log_msg = f"EXIT @ {exit_px:.5f} by {exit_reason}. PnL: {pnl_val:.2f}%"
            log_info(f"{exit_clr}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()
            # (Notifikasi Termux & Email)
            termux_title_exit = f"EXIT Signal: {pair_name}"
            termux_content_exit = f"{exit_reason} @ {exit_px:.5f}. PnL: {pnl_val:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)
            # ... (email notif)

            # Reset state
            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
            strategy_state["in_get_ready_1"] = False; strategy_state["in_get_ready_2"] = False
            strategy_state["price_at_gr1_close"] = None
            log_info("Status GR1/GR2 direset setelah trade close.", pair_name=pair_name)
        elif strategy_state["position_size"] > 0: # Masih dalam posisi
             log_debug(f"Posisi Aktif. Entry: {strategy_state['entry_price_custom']:.5f}, SL: {final_sl:.5f} ({exit_reason})", pair_name=pair_name)

    return strategy_state


# CHART_INTEGRATION_START (Fungsi chart sama, hanya pastikan data yang dikirim sesuai)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot):
    if pair_id_to_display not in current_data_manager_snapshot: return None
    pair_data = current_data_manager_snapshot[pair_id_to_display]
    candles_hist = pair_data.get("all_candles_list", [])
    strat_state = pair_data.get("strategy_state", {})
    pair_cfg = pair_data.get("config", {})

    chart_candles = candles_hist[-300:] # Max 300 candle di chart
    ohlc_pts = []
    close_prices_chart = []
    if not chart_candles:
        return {"ohlc": [], "ema_line": [], "annotations_yaxis": [], "annotations_points": [], 
                "pair_name": pair_cfg.get('pair_name', pair_id_to_display), "last_updated_tv": None,
                "in_gr1": False, "in_gr2": False}

    for c in chart_candles:
        if all(k in c and c[k] is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ohlc_pts.append({'x': c['timestamp'].timestamp() * 1000, 'y': [c['open'], c['high'], c['low'], c['close']]})
            close_prices_chart.append(c['close'])
    
    ema_line_chart = []
    ema_len_chart = pair_cfg.get('ema_length', 200)
    if len(close_prices_chart) >= 1:
        ema_vals_chart = calculate_ema_manual(close_prices_chart, ema_len_chart)
        for i in range(len(ema_vals_chart)):
            if i < len(ohlc_pts): ema_line_chart.append({'x': ohlc_pts[i]['x'], 'y': ema_vals_chart[i]})
    
    ann_yaxis = []
    # Entry & SL annotations (sama seperti sebelumnya)
    if strat_state.get("position_size", 0) > 0 and strat_state.get("entry_price_custom"):
        # ... (Tambahkan annotasi entry & SL)
        pass
    # GR1 Price annotation
    gr1_active = strat_state.get("in_get_ready_1", False)
    gr2_active = strat_state.get("in_get_ready_2", False)
    gr1_price_val = strat_state.get("price_at_gr1_close")
    if (gr1_active or gr2_active) and gr1_price_val: # Tampilkan jika GR1 atau GR2 aktif
        ann_yaxis.append({
            'y': gr1_price_val, 'borderColor': '#FFA500', 'strokeDashArray': 2, # Orange
            'label': {'borderColor': '#FFA500', 'style': {'color': '#000', 'background': '#FFA500'}, 'text': f'GR1 Ref: {gr1_price_val:.5f}'}
        })

    return {
        "ohlc": ohlc_pts, "ema_line": ema_line_chart, "annotations_yaxis": ann_yaxis, "annotations_points": [],
        "pair_name": pair_cfg.get('pair_name', pair_id_to_display),
        "last_updated_tv": chart_candles[-1]['timestamp'].timestamp() * 1000 if chart_candles else None,
        "in_gr1": gr1_active, "in_gr2": gr2_active
    }

flask_app_instance = None
if 'Flask' in sys.modules: flask_app_instance = Flask(__name__)
# HTML_CHART_TEMPLATE (sama, hanya pastikan Javascript menghandle `ema_line`, `in_gr1`, `in_gr2` dari payload)
# ... (Salin HTML_CHART_TEMPLATE dari skrip asli, pastikan di JS:
#      - Ada series untuk 'ema_line'
#      - Ada tampilan status untuk 'in_gr1' / 'in_gr2' di #strategyStateLabel
# )
# Fungsi Flask routes (get_available_pairs, get_chart_data, run_flask_server_thread) sama

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    # (Fungsi start_trading dari skrip asli bisa dipakai, penyesuaian utama ada di warm-up logic)
    # ... (Salin implementasi start_trading dari skrip asli) ...
    # Bagian WARM-UP STRATEGY STATE perlu disesuaikan:
    # (Contoh penyesuaian warm-up)
    # if initial_candles:
    #     min_len_for_logic = max(2, config.get('ema_length', 200))
    #     if len(initial_candles) >= min_len_for_logic:
    #         log_info(f"Warm-up state untuk {config['pair_name']}...")
    #         # Reset state GR sebelum warm-up
    #         local_crypto_data_manager[pair_id]["strategy_state"]["in_get_ready_1"] = False
    #         local_crypto_data_manager[pair_id]["strategy_state"]["in_get_ready_2"] = False
    #         local_crypto_data_manager[pair_id]["strategy_state"]["price_at_gr1_close"] = None
    #         for i in range(min_len_for_logic - 1, len(initial_candles)):
    #             historical_slice = initial_candles[:i+1]
    #             # Buat salinan state untuk warm-up, tapi hanya update state GR, bukan posisi
    #             temp_state = local_crypto_data_manager[pair_id]["strategy_state"].copy()
    #             temp_state["position_size"] = 0 # Jangan buka posisi saat warm-up
    #             
    #             warmed_up_state_slice = run_strategy_logic(historical_slice, config, temp_state, global_settings_dict)
    #             # Update state GR dari hasil warm-up slice ini
    #             local_crypto_data_manager[pair_id]["strategy_state"]["in_get_ready_1"] = warmed_up_state_slice.get("in_get_ready_1", False)
    #             local_crypto_data_manager[pair_id]["strategy_state"]["in_get_ready_2"] = warmed_up_state_slice.get("in_get_ready_2", False)
    #             local_crypto_data_manager[pair_id]["strategy_state"]["price_at_gr1_close"] = warmed_up_state_slice.get("price_at_gr1_close")
    #         
    #         # Pastikan tidak ada posisi terbuka setelah warm-up
    #         local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0 
    #         log_info(f"Warm-up {config['pair_name']} selesai.")
    # ... (Sisa start_trading tetap sama) ...
    pass # Placeholder, salin dari skrip asli dan sesuaikan warm-up


# --- MENU UTAMA ---
def main_menu():
    # (Fungsi main_menu dari skrip asli bisa dipakai)
    # ... (Salin implementasi main_menu) ...
    pass # Placeholder, salin dari skrip asli


if __name__ == "__main__":
    # (Bagian if __name__ == "__main__": dari skrip asli bisa dipakai)
    # ... (Salin implementasi __main__) ...
    # Contoh sederhana:
    settings = load_settings()
    if not settings.get("api_settings",{}).get("primary_key") or settings.get("api_settings",{}).get("primary_key") in ["YOUR_PRIMARY_KEY", "YOUR_API_KEY_HERE"]:
        settings = settings_menu(settings) # Paksa ke menu setting jika API key belum diatur
    
    flask_thread = None
    if flask_app_instance:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
        flask_thread.start()

    while True: # Loop menu utama
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (EMA GR1/GR2 - No Pandas) =========", color=AnsiColors.HEADER)
        # Tampilkan info ringkas di sini
        # ...
        options = ["Mulai Analisa", "Pengaturan", "Keluar"]
        try:
            _text, choice = pick(options, "Pilih Opsi:", indicator="=>")
            if choice == 0:
                settings = load_settings() # Reload settings before starting
                start_trading(settings, shared_crypto_data_manager, shared_data_lock)
            elif choice == 1:
                settings = settings_menu(settings)
            elif choice == 2:
                log_info("Aplikasi ditutup.", "SYSTEM")
                break
        except Exception as e_pick: # Handle jika pick error (misal di environment aneh)
            log_warning(f"Gagal menggunakan menu interaktif 'pick': {e_pick}. Gunakan input angka.")
            for i, opt in enumerate(options): print(f"{i+1}. {opt}")
            try:
                num_choice = int(input("Pilih nomor: ")) - 1
                if num_choice == 0: start_trading(load_settings(), shared_crypto_data_manager, shared_data_lock)
                elif num_choice == 1: settings = settings_menu(settings)
                elif num_choice == 2: log_info("Aplikasi ditutup.", "SYSTEM"); break
                else: print("Pilihan tidak valid.")
            except ValueError: print("Input angka tidak valid.")
            time.sleep(1)
