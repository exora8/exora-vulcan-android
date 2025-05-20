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

# --- TAMBAHAN UNTUK WEB SERVER ---
try:
    from flask import Flask, render_template_string, Response, jsonify
    import threading
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("PERINGATAN: Flask tidak terinstal. Fitur web chart tidak akan tersedia. pip install Flask")
# ----------------------------------

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
    MAGENTA = '\033[35m'
    YELLOW_BG = '\033[43m'

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
        display_message = message[:term_width - 5]
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r")
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
    sys.stdout.write(progress_line[:term_width])
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
console_formatter_template = '%(asctime)s - {bold}%(levelname)s{endc} - {cyan}[%(pair_name)s]{endc} - %(message)s'
ch.setFormatter(logging.Formatter(
    console_formatter_template.format(bold=AnsiColors.BOLD, endc=AnsiColors.ENDC, cyan=AnsiColors.CYAN)
))
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

SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500 # Anda bisa kurangi ini untuk testing chart lebih cepat, misal 200-300
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# --- WEB SERVER CONFIG ---
WEB_SERVER_PORT = 5005 # Ganti port jika perlu
MAX_CHART_CANDLES = 200 # Jumlah candle maksimal di chart
if FLASK_AVAILABLE:
    app_flask = Flask(__name__)
    chart_data_store = {} # { "pair_id": {"candles": [], "indicators": {}, "last_update": timestamp}, ... }
    chart_data_lock = threading.Lock()
    active_trading_pairs_for_web = [] # List of pair_ids being traded

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None): #init diubah ke __init__
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
        if not self.keys:
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None

    def switch_to_next_key(self):
        if not self.keys: return None

        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
                email_body = (f"Skrip trading telah secara otomatis mengganti API key CryptoCompare.\n\n"
                              f"API Key sebelumnya mungkin telah mencapai limit atau tidak valid.\n"
                              f"Sekarang menggunakan API key dengan index: {self.current_index}\n"
                              f"Key: ...{new_key_display[-8:] if len(new_key_display) > 8 else new_key_display} (bagian akhir ditampilkan untuk identifikasi)\n\n"
                              f"Harap periksa status API key Anda di CryptoCompare.")
                dummy_email_cfg = {
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi pergantian API key (APIKeyManager).")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "KRITIS: SEMUA API Key CryptoCompare Gagal!"
                email_body = (f"Skrip trading telah mencoba semua API key CryptoCompare yang tersedia (primary dan recovery) dan semuanya gagal.\n\n"
                              f"Skrip tidak dapat lagi mengambil data harga.\n"
                              f"Harap segera periksa akun CryptoCompare Anda dan konfigurasi API key di skrip.")
                dummy_email_cfg = {
                     "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal (APIKeyManager).")
            return None

    def has_valid_keys(self):
        return bool(self.keys)

    def total_keys(self):
        return len(self.keys)

    def get_current_key_index(self):
        return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a', end='', flush=True) # Beep for Linux/macOS/Termux
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('pair_name',
                                             settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=pair_name_ctx)
        return

    msg = MIMEText(body_text)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e:
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)

def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    api_settings = global_settings.get("api_settings", {})
    if not api_settings.get("enable_termux_notifications", False):
        return

    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal dan paket termux-api sudah diinstal di Termux (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,
        "left_strength": 50, "right_strength": 150,
        "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0,
        "emergency_sl_percent": 10.0, "enable_secure_fib": True, "secure_fib_check_price": "Close",
        "enable_email_notifications": False,
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }

def load_settings():
    default_api_settings = {
        "primary_key": "YOUR_PRIMARY_KEY",
        "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com",
        "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com",
        "enable_termux_notifications": False,
        "enable_web_server": True, # Pengaturan baru untuk web server
        "web_server_port": WEB_SERVER_PORT # Pengaturan baru
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            try:
                settings = json.load(f)
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v

                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = []
                for crypto_cfg in settings["cryptos"]:
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                return settings
            except json.JSONDecodeError:
                log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
                return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")


def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()

    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}");

    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        refresh_input = int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60)
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, refresh_input)
    except ValueError:
        print(f"{AnsiColors.RED}Input interval refresh tidak valid. Menggunakan default: {new_config.get('refresh_interval_seconds',60)}{AnsiColors.ENDC}")
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))

    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))
    except ValueError:
        print(f"{AnsiColors.RED}Input strength tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["left_strength"] = new_config.get('left_strength',50)
        new_config["right_strength"] = new_config.get('right_strength',150)

    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter trading tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',5.0)
        new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0)
        new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',10.0)

    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower().strip()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))

    secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid. Menggunakan default: {new_config.get('secure_fib_check_price','Close')}{AnsiColors.ENDC}");

    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global dari API Settings (jika notif global aktif).{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))

    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()

    return new_config

def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]

        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len([k for k in recovery_keys if k])
        termux_notif_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        web_server_status = "Aktif" if api_s.get("enable_web_server", True) and FLASK_AVAILABLE else "Nonaktif"
        web_server_port_val = api_s.get("web_server_port", WEB_SERVER_PORT)


        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
        pick_title_settings += f"Web Server Live Chart: {web_server_status} (Port: {web_server_port_val})\n" # Info Web Server
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"


        if not current_settings.get("cryptos"):
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

        original_options_structure = [
            ("header", "--- Pengaturan API & Global ---"),
            ("option", "Atur Primary API Key"),
            ("option", "Kelola Recovery API Keys"),
            ("option", "Atur Email Global untuk Notifikasi Sistem"),
            ("option", "Aktifkan/Nonaktifkan Notifikasi Termux Realtime"),
            ("option", "Pengaturan Web Server Live Chart"), # Opsi baru
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"),
            ("option", "Ubah Konfigurasi Crypto"),
            ("option", "Hapus Konfigurasi Crypto"),
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")
        ]

        selectable_options = [text for type, text in original_options_structure if type == "option"]

        try:
            option_text, index = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick:
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings)
            for idx, opt_text in enumerate(selectable_options):
                print(f"  {idx + 1}. {opt_text}")
            try:
                choice = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice < len(selectable_options):
                    index = choice
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue

        action_choice = index

        try:
            clear_screen_animated()
            if action_choice == 0: # Atur Primary API Key
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 1: # Kelola Recovery API Keys
                while True:
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k]
                    api_s['recovery_keys'] = current_recovery

                    if not current_recovery:
                        recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"

                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]

                    try:
                        rec_option_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e_pick_rec:
                        log_error(f"Error dengan library 'pick' di menu recovery: {e_pick_rec}. Gunakan input manual.")
                        print(recovery_pick_title)
                        for idx_rec, opt_text_rec in enumerate(recovery_options_plain):
                             print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice = int(input("Pilih nomor opsi: ")) -1
                            if 0 <= rec_choice < len(recovery_options_plain):
                                rec_index = rec_choice
                            else:
                                print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                                show_spinner(1, "Kembali...")
                                continue
                        except ValueError:
                            print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                            show_spinner(1, "Kembali...")
                            continue

                    clear_screen_animated()

                    if rec_index == 0: # Tambah Recovery Key
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else:
                            print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 1: # Hapus Recovery Key
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
                            show_spinner(1, "Kembali...")
                            continue
                        try:
                            for i_del, r_key_del in enumerate(current_recovery):
                                r_key_del_display = r_key_del[:5] + "..." + r_key_del[-3:] if len(r_key_del) > 8 else r_key_del
                                print(f"  {i_del+1}. {r_key_del_display}")
                            idx_del_str = input("Nomor recovery key yang akan dihapus: ").strip()
                            if not idx_del_str:
                                print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                                show_spinner(1, "Kembali...")
                                continue
                            idx_del = int(idx_del_str) - 1
                            if 0 <= idx_del < len(current_recovery):
                                removed = current_recovery.pop(idx_del)
                                api_s['recovery_keys'] = current_recovery
                                save_settings(current_settings)
                                print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                            else:
                                print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                        except ValueError:
                            print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 2: # Kembali
                        break
            elif action_choice == 2: # Atur Email Global
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global (API Key switch, dll)? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))

                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")

            elif action_choice == 3: # Atur Notifikasi Termux
                animated_text_display("-- Pengaturan Notifikasi Termux Realtime --", color=AnsiColors.HEADER)
                current_status = api_s.get('enable_termux_notifications', False)
                new_status_input = input(f"Aktifkan Notifikasi Termux? (true/false) [{current_status}]: ").lower().strip()
                if new_status_input == 'true':
                    api_s['enable_termux_notifications'] = True
                    print(f"{AnsiColors.GREEN}Notifikasi Termux diaktifkan.{AnsiColors.ENDC}")
                    print(f"{AnsiColors.ORANGE}Pastikan Termux:API terinstal dan `pkg install termux-api` sudah dijalankan di Termux.{AnsiColors.ENDC}")
                elif new_status_input == 'false':
                    api_s['enable_termux_notifications'] = False
                    print(f"{AnsiColors.GREEN}Notifikasi Termux dinonaktifkan.{AnsiColors.ENDC}")
                else:
                    print(f"{AnsiColors.ORANGE}Input tidak valid. Status Notifikasi Termux tidak berubah: {current_status}.{AnsiColors.ENDC}")

                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(2, "Menyimpan & Kembali...")
            elif action_choice == 4: # Pengaturan Web Server
                animated_text_display("-- Pengaturan Web Server Live Chart --", color=AnsiColors.HEADER)
                if not FLASK_AVAILABLE:
                    print(f"{AnsiColors.RED}Flask tidak terinstal. Fitur ini tidak dapat diaktifkan.{AnsiColors.ENDC}")
                    print(f"{AnsiColors.ORANGE}Silakan instal Flask dengan: pip install Flask{AnsiColors.ENDC}")
                    api_s['enable_web_server'] = False
                else:
                    current_web_status = api_s.get('enable_web_server', True)
                    web_status_input = input(f"Aktifkan Web Server Live Chart? (true/false) [{current_web_status}]: ").lower().strip()
                    if web_status_input == 'true':
                        api_s['enable_web_server'] = True
                        print(f"{AnsiColors.GREEN}Web server diaktifkan.{AnsiColors.ENDC}")
                    elif web_status_input == 'false':
                        api_s['enable_web_server'] = False
                        print(f"{AnsiColors.GREEN}Web server dinonaktifkan.{AnsiColors.ENDC}")
                    else:
                        print(f"{AnsiColors.ORANGE}Input tidak valid. Status web server tidak berubah: {current_web_status}.{AnsiColors.ENDC}")

                current_web_port = api_s.get('web_server_port', WEB_SERVER_PORT)
                web_port_input = input(f"Masukkan port untuk Web Server [{current_web_port}]: ").strip()
                if web_port_input:
                    try:
                        api_s['web_server_port'] = int(web_port_input)
                        print(f"{AnsiColors.GREEN}Port Web Server diatur ke: {api_s['web_server_port']}{AnsiColors.ENDC}")
                    except ValueError:
                        print(f"{AnsiColors.RED}Input port tidak valid. Menggunakan port sebelumnya: {current_web_port}{AnsiColors.ENDC}")
                        api_s['web_server_port'] = current_web_port
                else:
                    api_s['web_server_port'] = current_web_port


                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(2, "Menyimpan & Kembali...")

            elif action_choice == 5: # Tambah Konfigurasi Crypto Baru
                new_crypto_conf = get_default_crypto_config()
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf)
                current_settings.setdefault("cryptos", []).append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 6: # Ubah Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali...");
                    continue
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                crypto_options = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]
                if not crypto_options:
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue

                try:
                    _sel_text, idx_choice = pick(crypto_options, "Pilih konfigurasi untuk diubah:", indicator="=>")
                except Exception as e_pick_crypto_edit:
                    log_error(f"Error pick di ubah crypto: {e_pick_crypto_edit}. Input manual.")
                    for i, crypto_conf_text in enumerate(crypto_options): print(f"  {i+1}. {crypto_conf_text}")
                    idx_choice_str = input("Nomor konfigurasi crypto yang akan diubah: ").strip()
                    if not idx_choice_str: print(f"{AnsiColors.RED}Input kosong.{AnsiColors.ENDC}"); show_spinner(1,"..."); continue
                    try: idx_choice = int(idx_choice_str) - 1
                    except ValueError: print(f"{AnsiColors.RED}Input bukan angka.{AnsiColors.ENDC}"); show_spinner(1,"..."); continue

                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")

            elif action_choice == 7: # Hapus Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali...");
                    continue
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                crypto_options_del = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]
                if not crypto_options_del:
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
                try:
                    _sel_text_del, idx_choice_del = pick(crypto_options_del, "Pilih konfigurasi untuk dihapus:", indicator="=>")
                except Exception as e_pick_crypto_del:
                    log_error(f"Error pick di hapus crypto: {e_pick_crypto_del}. Input manual.")
                    for i, crypto_conf_text_del in enumerate(crypto_options_del): print(f"  {i+1}. {crypto_conf_text_del}")
                    idx_choice_str_del = input("Nomor konfigurasi crypto yang akan dihapus: ").strip()
                    if not idx_choice_str_del: print(f"{AnsiColors.RED}Input kosong.{AnsiColors.ENDC}"); show_spinner(1,"..."); continue
                    try: idx_choice_del = int(idx_choice_str_del) - 1
                    except ValueError: print(f"{AnsiColors.RED}Input bukan angka.{AnsiColors.ENDC}"); show_spinner(1,"..."); continue

                if 0 <= idx_choice_del < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_choice_del]['symbol']}-{current_settings['cryptos'][idx_choice_del]['currency']}"
                    current_settings["cryptos"].pop(idx_choice_del)
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")

            elif action_choice == 8: # Kembali ke Menu Utama
                break
        except ValueError:
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e:
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:")
            show_spinner(1.5, "Error, kembali...")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)

    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT :
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)

        if current_to_ts is not None and candles_still_needed > 1 :
            limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)

        if limit_for_this_api_call <= 0: break

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20)

            if response.status_code in [401, 403, 429]:
                error_data = {}
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    pass
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit"
                ]
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api:
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom')
                }
                batch_candles_list.append(candle)

            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop()

            if not batch_candles_list and current_to_ts is not None :
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles

            if raw_candles_from_api:
                current_to_ts = raw_candles_from_api[0]['time']
            else:
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired):
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break

            if len(all_accumulated_candles) >= total_limit_desired: break

            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3)

        except APIKeyError:
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error:", pair_name=pair_name)
            break

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- FUNGSI UNTUK UPDATE CHART DATA STORE ---
if FLASK_AVAILABLE:
    def update_chart_store_candles(pair_id, candles_history):
        with chart_data_lock:
            if pair_id not in chart_data_store:
                chart_data_store[pair_id] = {"candles": [], "indicators": {}, "last_update": 0}

            chart_candles = []
            # Ambil MAX_CHART_CANDLES terakhir atau semua jika lebih sedikit
            source_candles = candles_history[-MAX_CHART_CANDLES:] if len(candles_history) > MAX_CHART_CANDLES else candles_history
            for candle in source_candles:
                if all(k in candle and candle[k] is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
                    chart_candles.append({
                        "t": int(candle['timestamp'].timestamp() * 1000), # ms untuk Chart.js
                        "o": candle['open'],
                        "h": candle['high'],
                        "l": candle['low'],
                        "c": candle['close']
                    })
            chart_data_store[pair_id]["candles"] = chart_candles
            chart_data_store[pair_id]["last_update"] = time.time()

    def update_chart_store_indicators(pair_id, strategy_state_param, current_candle_for_ts=None):
        with chart_data_lock:
            if pair_id not in chart_data_store:
                chart_data_store[pair_id] = {"candles": [], "indicators": {}, "last_update": 0}

            indicators = chart_data_store[pair_id].get("indicators", {})

            # Pivots (diasumsikan hanya yang terbaru yang relevan untuk tampilan langsung)
            # Jika Anda ingin menyimpan histori pivot di chart, Anda perlu logika akumulasi
            if strategy_state_param.get("final_pivot_high_price_confirmed") is not None and current_candle_for_ts:
                 # Cari candle yang sesuai dengan high_bar_index_for_fib untuk timestamp pivot
                idx_ph = strategy_state_param.get("high_bar_index_for_fib") # Ini index di history, bukan di candle_for_ts
                # Untuk simple, kita bisa asumsikan pivot terjadi beberapa candle lalu relatif thd current_candle_for_ts
                # Atau, kita butuh timestamp asli dari pivot. Mari kita sederhanakan untuk sekarang.
                # Jika strategy_state memiliki timestamp pivot, itu lebih baik.
                # Kita akan menandai pivot high/low terbaru.
                indicators["pivot_high"] = {
                    "price": strategy_state_param["final_pivot_high_price_confirmed"],
                    # "timestamp": butuh timestamp aktual pivot, untuk saat ini pakai candle terakhir - right_strength
                }
            if strategy_state_param.get("final_pivot_low_price_confirmed") is not None and current_candle_for_ts:
                indicators["pivot_low"] = {
                    "price": strategy_state_param["final_pivot_low_price_confirmed"],
                    # "timestamp": butuh timestamp aktual pivot
                }

            # FIB Level
            if strategy_state_param.get("active_fib_level") is not None:
                indicators["fib_0_5"] = {"price": strategy_state_param["active_fib_level"], "active": True}
            elif "fib_0_5" in indicators: # Hapus jika tidak aktif lagi
                 indicators["fib_0_5"]["active"] = False


            # Entry (perlu timestamp dari candle saat entry)
            if strategy_state_param.get("entry_price_custom") is not None and current_candle_for_ts:
                # Untuk entry/exit, kita bisa menambahkan ke list agar semua terlihat
                if "entries" not in indicators: indicators["entries"] = []
                # Hindari duplikasi jika event yang sama dikirim berkali-kali
                # Cek berdasarkan timestamp dan harga (atau ID trade jika ada)
                is_new_entry = True
                entry_ts = int(current_candle_for_ts['timestamp'].timestamp() * 1000)
                for e in indicators["entries"]:
                    if e["x"] == entry_ts and e["y"] == strategy_state_param["entry_price_custom"]:
                        is_new_entry = False
                        break
                if is_new_entry: # Hanya tambah jika ini entry baru (misal, baru terdeteksi)
                    indicators["entries"].append({
                        "x": entry_ts,
                        "y": strategy_state_param["entry_price_custom"]
                    })
                    # Batasi jumlah entri yang ditampilkan jika perlu
                    indicators["entries"] = indicators["entries"][-10:]


            # Exit (perlu timestamp dari candle saat exit)
            # Ini akan lebih rumit karena exit membersihkan state entry.
            # Mungkin lebih baik menangkap event exit saat terjadi di run_strategy_logic
            # dan mengirimkannya secara eksplisit ke update_chart_store_indicators.

            # Current SL
            sl_level = strategy_state_param.get("emergency_sl_level_custom")
            sl_type = "Emergency SL"
            if strategy_state_param.get("trailing_tp_active_custom") and strategy_state_param.get("current_trailing_stop_level") is not None:
                if sl_level is None or strategy_state_param.get("current_trailing_stop_level") > sl_level:
                    sl_level = strategy_state_param.get("current_trailing_stop_level")
                    sl_type = "Trailing SL"

            if sl_level is not None:
                indicators["current_sl"] = {"price": sl_level, "type": sl_type, "active": True}
            elif "current_sl" in indicators: # Hapus jika tidak ada posisi
                indicators["current_sl"]["active"] = False


            chart_data_store[pair_id]["indicators"] = indicators
            chart_data_store[pair_id]["last_update"] = time.time()

    def add_chart_event(pair_id, event_type, price, timestamp_dt, text=""):
        """ Helper untuk menambah event seperti EXIT ke chart store """
        with chart_data_lock:
            if pair_id not in chart_data_store:
                chart_data_store[pair_id] = {"candles": [], "indicators": {}, "last_update": 0}
            
            indicators = chart_data_store[pair_id].get("indicators", {})
            if event_type not in indicators:
                indicators[event_type] = []

            event_ts_ms = int(timestamp_dt.timestamp() * 1000)
            
            # Hindari duplikasi event yang sama
            is_new_event = True
            for ev in indicators[event_type]:
                if ev["x"] == event_ts_ms and ev["y"] == price:
                    is_new_event = False
                    break
            
            if is_new_event:
                indicators[event_type].append({"x": event_ts_ms, "y": price, "text": text})
                indicators[event_type] = indicators[event_type][-10:] # Batasi histori event

            chart_data_store[pair_id]["indicators"] = indicators
            chart_data_store[pair_id]["last_update"] = time.time()

# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0,
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None,
        "high_bar_index_for_fib": None, # Index dalam list candle histori
        "low_bar_index_for_fib": None,  # Index dalam list candle histori
        "active_fib_level": None,
        "active_fib_line_start_index": None,
        "entry_price_custom": None,
        "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None,
        "position_size": 0,
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1:
        return pivots

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue

        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break
            if is_high:
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else:
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue

        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break
            if is_high:
                if series_list[i] < series_list[i+j]: is_pivot = False; break
            else:
                if series_list[i] > series_list[i+j]: is_pivot = False; break

        if is_pivot:
            pivots[i] = series_list[i]
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    pair_id = f"{pair_name}_{crypto_config['timeframe']}" # Untuk web server key

    # Reset pivots yang terkonfirmasi di state untuk deteksi ulang pada candle saat ini
    # State pivot yang dikonfirmasi adalah per kejadian, bukan persisten kecuali untuk kalkulasi FIB
    # strategy_state["final_pivot_high_price_confirmed"] = None # Ini menyebabkan FIB tidak pernah terbentuk
    # strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True) : update_chart_store_indicators(pair_id, strategy_state)
        return strategy_state

    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]

    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    current_bar_index_in_list = len(candles_history) - 1
    if current_bar_index_in_list < 0 :
        if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True) : update_chart_store_indicators(pair_id, strategy_state)
        return strategy_state

    # Index di mana pivot dikonfirmasi (candle saat ini adalah right_strength bar setelah pivot aktual)
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength

    # Cek apakah ada pivot high baru yang terkonfirmasi PADA BAR INI
    if 0 <= idx_pivot_event_high < len(raw_pivot_highs) and raw_pivot_highs[idx_pivot_event_high] is not None:
        # Hanya update jika ini adalah sinyal pivot type yang berbeda atau pivot high baru
        if strategy_state["last_signal_type"] != 1 or strategy_state.get("final_pivot_high_price_confirmed") != raw_pivot_highs[idx_pivot_event_high]:
            strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_highs[idx_pivot_event_high]
            strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high # Simpan index dari candle pivot high
            strategy_state["last_signal_type"] = 1
            pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
            log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)
            # Reset state FIB jika pivot high baru terdeteksi, karena ini memulai sekuens baru
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None
            strategy_state["final_pivot_low_price_confirmed"] = None # Reset low juga untuk sekuens H-L baru
            strategy_state["low_bar_index_for_fib"] = None


    # Cek apakah ada pivot low baru yang terkonfirmasi PADA BAR INI
    if 0 <= idx_pivot_event_low < len(raw_pivot_lows) and raw_pivot_lows[idx_pivot_event_low] is not None:
        # Hanya update jika ini adalah sinyal pivot type yang berbeda atau pivot low baru
        if strategy_state["last_signal_type"] != -1 or strategy_state.get("final_pivot_low_price_confirmed") != raw_pivot_lows[idx_pivot_event_low]:
            strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_lows[idx_pivot_event_low]
            strategy_state["low_bar_index_for_fib"] = idx_pivot_event_low # Simpan index dari candle pivot low
            strategy_state["last_signal_type"] = -1
            pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
            log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    current_candle = candles_history[current_bar_index_in_list]

    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=pair_name)
        if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True) : update_chart_store_indicators(pair_id, strategy_state, current_candle)
        return strategy_state

    # Logika FIB setelah Pivot Low terkonfirmasi dan ada Pivot High sebelumnya
    if strategy_state["final_pivot_low_price_confirmed"] is not None and \
       strategy_state["final_pivot_high_price_confirmed"] is not None and \
       strategy_state["high_bar_index_for_fib"] is not None and \
       strategy_state["low_bar_index_for_fib"] is not None:

        # Pastikan Pivot Low terjadi SETELAH Pivot High untuk sekuens H-L yang valid
        if strategy_state["low_bar_index_for_fib"] > strategy_state["high_bar_index_for_fib"]:
            high_for_fib = strategy_state["final_pivot_high_price_confirmed"]
            low_for_fib = strategy_state["final_pivot_low_price_confirmed"]

            if high_for_fib is None or low_for_fib is None:
                log_warning("Harga untuk kalkulasi FIB tidak valid (None).", pair_name=pair_name)
            else:
                calculated_fib_level = (high_for_fib + low_for_fib) / 2.0
                is_fib_late = False
                if crypto_config["enable_secure_fib"]:
                    price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                    if price_val_current_candle is not None and calculated_fib_level is not None and price_val_current_candle > calculated_fib_level:
                        is_fib_late = True

                if is_fib_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state["active_fib_level"] = None
                    strategy_state["active_fib_line_start_index"] = None
                elif calculated_fib_level is not None :
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {high_for_fib:.2f}, L: {low_for_fib:.2f})", pair_name=pair_name)
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = strategy_state["low_bar_index_for_fib"] # Pivot low adalah start fib

            # Setelah FIB digunakan (atau gagal karena terlambat), reset pivot high dan low untuk sekuens berikutnya
            # Ini penting agar FIB tidak terus aktif dari pivot lama
            strategy_state["final_pivot_high_price_confirmed"] = None
            strategy_state["high_bar_index_for_fib"] = None
            # strategy_state["final_pivot_low_price_confirmed"] = None # Jangan reset low, karena bisa jadi High baru
            # strategy_state["low_bar_index_for_fib"] = None

    # Logika Entry
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        if current_candle.get('close') is None or current_candle.get('open') is None:
            log_warning("Nilai close atau open tidak ada di candle saat ini. Skip entry check.", pair_name=pair_name)
            if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True) : update_chart_store_indicators(pair_id, strategy_state, current_candle)
            return strategy_state

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

                emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl

                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()

                termux_title = f"BUY Signal: {pair_name}"
                termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}"
                send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)

                email_subject = f"BUY Signal: {pair_name}"
                email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\n"
                              f"Entry Price: {entry_px:.5f}\n"
                              f"FIB Level: {strategy_state['active_fib_level']:.5f}\n"
                              f"Emergency SL: {emerg_sl:.5f}\n"
                              f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, crypto_config)

            # Setelah entry, reset FIB agar tidak re-entry di candle berikutnya dengan FIB yang sama
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None
            # Reset juga pivot agar tidak langsung membentuk FIB baru dari pivot yang sama
            # strategy_state["final_pivot_high_price_confirmed"] = None # Sebaiknya tidak direset di sini
            # strategy_state["final_pivot_low_price_confirmed"] = None  # agar bisa lanjut ke pivot baru


    # Logika Trailing Stop dan Exit
    if strategy_state["position_size"] > 0:
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None).", pair_name=pair_name)
        else:
             strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0:
                profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None:
                profit_percent = 0.0
                log_warning("highest_price_for_trailing is None saat kalkulasi profit.", pair_name=pair_name)
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE

        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price_open = current_candle.get('open')
            if exit_price_open is None:
                log_warning("Harga open candle tidak ada untuk exit. Menggunakan SL sebagai harga exit.", pair_name=pair_name)
                exit_price = final_stop_for_exit
            else:
                exit_price = min(exit_price_open, final_stop_for_exit) # Harga exit adalah SL atau open (mana yg lebih dulu tercapai)

            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if exit_comment == "Trailing Stop" and pnl < 0: # Trailing stop tapi rugi (misal harga drop cepat)
                exit_color = AnsiColors.RED

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

            if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True):
                 add_chart_event(pair_id, "exits", exit_price, current_candle['timestamp'], exit_comment)


            termux_title_exit = f"EXIT Signal: {pair_name}"
            termux_content_exit = f"{exit_comment} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)

            email_subject = f"Trade Closed: {pair_name} ({exit_comment})"
            email_body = (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\n"
                          f"Reason: {exit_comment}\n"
                          f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\n"
                          f"PnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, crypto_config)

            # Reset state setelah exit
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            # strategy_state["active_fib_level"] = None # Sudah direset saat entry
            # strategy_state["final_pivot_high_price_confirmed"] = None # Reset untuk sekuens baru
            # strategy_state["final_pivot_low_price_confirmed"] = None

    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        stop_type_info = "Emergency SL"
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state.get("current_trailing_stop_level")
                stop_type_info = "Trailing SL"

        entry_price_display = strategy_state.get('entry_price_custom', 0)
        sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)

    if FLASK_AVAILABLE and global_settings.get("api_settings",{}).get("enable_web_server", True) :
        update_chart_store_indicators(pair_id, strategy_state, current_candle)
    return strategy_state

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    if FLASK_AVAILABLE and api_settings.get("enable_web_server", True):
        global active_trading_pairs_for_web
        active_trading_pairs_for_web = [
            f"{cfg['symbol']}-{cfg['currency']}_{cfg['timeframe']}" for cfg in all_crypto_configs
        ]
        log_info(f"Web server akan aktif. Akses di http://localhost:{api_settings.get('web_server_port', WEB_SERVER_PORT)}", "SYSTEM")


    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = current_key_display_val[:5] + "..." + current_key_display_val[-3:] if len(current_key_display_val) > 8 else current_key_display_val

    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_name = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        pair_id = f"{pair_name}_{config.get('timeframe','DEF')}" # Key untuk web server
        config['pair_name'] = pair_name


        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True,
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min,
            "data_fetch_failed_consecutively": 0
        }

        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True
            except APIKeyError:
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break
                retries_done_initial +=1
            except requests.exceptions.RequestException as e:
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                break
            except Exception as e_gen:
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now()
            continue

        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])

        if FLASK_AVAILABLE and api_settings.get("enable_web_server", True):
            update_chart_store_candles(pair_id, initial_candles)


        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])

                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # Proses hingga candle kedua terakhir
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue

                    temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Pastikan tidak ada posisi aktif saat warm-up

                    # Jalankan logika, tapi jangan kirim notifikasi atau lakukan trade sungguhan
                    # Kita hanya ingin state internal (pivots, fib) ter-update
                    # Global settings bisa di-clone dan dinonaktifkan notifnya untuk warm-up jika perlu
                    dummy_global_settings_warmup = json.loads(json.dumps(global_settings_dict)) # deep copy
                    if "api_settings" in dummy_global_settings_warmup:
                        dummy_global_settings_warmup["api_settings"]["enable_termux_notifications"] = False
                    
                    # Matikan notif email per pair juga untuk warm-up
                    dummy_config_warmup = config.copy()
                    dummy_config_warmup["enable_email_notifications"] = False


                    crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice, dummy_config_warmup, temp_state_for_warmup, dummy_global_settings_warmup
                    )

                    # Jika posisi terbuka selama warm-up, segera tutup untuk state berikutnya
                    if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0:
                        crypto_data_manager[pair_id]["strategy_state"] = {
                            **crypto_data_manager[pair_id]["strategy_state"],
                            **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None,
                               "highest_price_for_trailing":None, "trailing_tp_active_custom":False,
                               "current_trailing_stop_level":None}
                        }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
                # Update chart store indicators setelah warm-up
                if FLASK_AVAILABLE and api_settings.get("enable_web_server", True):
                    update_chart_store_indicators(pair_id, crypto_data_manager[pair_id]["strategy_state"], initial_candles[-1] if initial_candles else None)

            else:
                log_warning(f"Data awal ({len(initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else:
            log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])

        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(f"Data Downloading Complete: {config['pair_name']}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", config)
                crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])

    animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]
                pair_name = config['pair_name']

                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 :
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600:
                        log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name)
                        continue
                    else:
                        data["data_fetch_failed_consecutively"] = 0
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()

                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    if config.get('timeframe') == "minute": required_interval_for_this_pair = 55
                    elif config.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8
                    else: required_interval_for_this_pair = 3580
                else:
                    required_interval_for_this_pair = config.get('refresh_interval_seconds', 60)

                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue

                log_info(f"Memproses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data["all_candles_list"])

                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False

                max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_for_this_pair_update = 0

                while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt:
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {pair_name}.", pair_name=pair_name)
                        break

                    limit_fetch = 3 # Ambil beberapa candle terakhir untuk update
                    if data["big_data_collection_phase_active"]:
                        limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"])
                        if limit_fetch_needed <=0 :
                             fetch_update_successful_for_this_pair = True
                             new_candles_batch = []
                             break
                        limit_fetch = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT)
                        limit_fetch = max(limit_fetch, 1)

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch,
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0
                        any_data_fetched_this_cycle = True

                    except APIKeyError:
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1

                        if not api_key_manager.switch_to_next_key():
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break
                        retries_done_for_this_pair_update += 1

                    except requests.exceptions.RequestException as e:
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break
                    except Exception as e_gen_update:
                        log_error(f"Error umum saat mengambil update {pair_name}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name)
                        log_exception("Traceback Error Update Fetch:", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break

                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data["last_attempt_after_all_keys_failed"] = datetime.now()
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data["big_data_collection_phase_active"]:
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {pair_name} meskipun fetch (dianggap) berhasil.{AnsiColors.ENDC}", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    continue

                # Merge new candles with existing
                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Check if content is different
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1

                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                     if FLASK_AVAILABLE and api_settings.get("enable_web_server", True):
                        update_chart_store_candles(pair_id, data["all_candles_list"])
                elif new_candles_batch : # Fetch berhasil tapi tidak ada candle baru/update
                     log_info("Tidak ada candle dengan timestamp baru atau update konten. Data terakhir mungkin identik.", pair_name=pair_name)


                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!{AnsiColors.ENDC}", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES:
                            data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                            if FLASK_AVAILABLE and api_settings.get("enable_web_server", True): # Update chart store jika dipotong
                                update_chart_store_candles(pair_id, data["all_candles_list"])


                        if not data["big_data_email_sent"]:
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", config)
                            data["big_data_email_sent"] = True

                        data["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------{AnsiColors.ENDC}", pair_name=pair_name)
                else: # Fase live analysis
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Jaga agar tidak terlalu banyak
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                        if FLASK_AVAILABLE and api_settings.get("enable_web_server", True): # Update chart store jika dipotong
                            update_chart_store_candles(pair_id, data["all_candles_list"])


                min_len_for_pivots = config.get('left_strength',50) + config.get('right_strength',150) + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    # Proses logika jika ada candle baru, ATAU jika baru selesai big data, ATAU jika masih big data dan ada tambahan
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         (data["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) )

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"], global_settings_dict)
                    elif not data["big_data_collection_phase_active"]: # Live, tapi tidak ada candle baru
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                         # Walaupun tidak ada candle baru, state indikator mungkin perlu di-refresh ke chart jika belum ada posisi
                         # Ini ditangani di akhir run_strategy_logic
                else:
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)

                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)

            # Sleep logic
            sleep_duration = 15 # Default sleep

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None:
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600
            elif active_cryptos_still_in_big_data_collection > 0:
                min_big_data_interval = float('inf')
                for pid_loop, pdata_loop in crypto_data_manager.items():
                    if pdata_loop["big_data_collection_phase_active"]:
                        pconfig_loop = pdata_loop["config"]
                        interval_bd = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_loop.get('timeframe') == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval_bd)

                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30)
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua pair sudah live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM")
                else: # Fallback jika min_overall_next_refresh_seconds tidak terhitung
                    default_refresh_from_config = 60 # Default umum
                    if all_crypto_configs : # Ambil dari config pair pertama jika ada
                        default_refresh_from_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)

                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama).", pair_name="SYSTEM")


            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: # Jika sleep_duration 0 atau negatif, sleep minimal
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM")
                time.sleep(1)


    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error:", pair_name="SYSTEM")
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()

# --- WEB SERVER FLASK ---
if FLASK_AVAILABLE:
    HOME_PAGE_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Chart Analysis - Select Pair</title>
        <style>
            body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #d4d4d4; display: flex; flex-direction: column; align-items: center; padding-top: 20px; }
            h1 { color: #4CAF50; text-align: center; }
            .container { background-color: #2a2a2a; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.5); width: 90%; max-width: 600px; }
            ul { list-style-type: none; padding: 0; }
            li { margin-bottom: 10px; }
            a {
                display: block;
                padding: 15px;
                background-color: #333;
                color: #4CAF50;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s ease;
                text-align: center;
                font-weight: bold;
            }
            a:hover { background-color: #45a049; color: #fff; }
            p { text-align: center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Select Crypto Pair for Live Chart</h1>
            {% if pairs %}
                <ul>
                    {% for pair_id_str in pairs %}
                        <li><a href="{{ url_for('chart_page', pair_id_url=pair_id_str) }}">{{ pair_id_str.replace('_', ' ') }}</a></li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>No trading pairs are currently active or configured for web view.</p>
                <p>Please start the trading analysis from the main script menu.</p>
            {% endif %}
        </div>
    </body>
    </html>
    """

    CHART_PAGE_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Live Chart: {{ pair_id_display }}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.0/dist/chartjs-chart-financial.min.js"></script>
        <style>
            body { font-family: sans-serif; margin: 0; padding: 0; background-color: #121212; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; }
            h1 { text-align: center; color: #4CAF50; margin: 10px 0; font-size: 1.5em; }
            #chartContainer { flex-grow: 1; display: flex; justify-content: center; align-items: center; padding: 5px; }
            canvas { max-width: 100%; max-height: 100%; }
            .status { text-align: center; padding: 5px; background-color: #333; font-size: 0.9em; }
            a.back-button { display: inline-block; margin: 10px; padding: 8px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <a href="{{ url_for('home_page') }}" class="back-button">&laquo; Back to Pair List</a>
        <h1>Live Chart: {{ pair_id_display }}</h1>
        <div id="chartContainer">
            <canvas id="liveChart"></canvas>
        </div>
        <div class="status" id="statusMessage">Connecting to data stream...</div>

        <script>
            const pairId = "{{ pair_id_raw }}";
            const statusMessage = document.getElementById('statusMessage');
            const ctx = document.getElementById('liveChart').getContext('2d');
            let chart;

            function createChart(initialData) {
                const candlestickData = initialData.candles || [];
                
                chart = new Chart(ctx, {
                    type: 'candlestick', // from chartjs-chart-financial
                    data: {
                        datasets: [{
                            label: pairId.replace('_', ' '),
                            data: candlestickData,
                            borderColor: 'rgba(0,0,0,0)', // Hide main line border for candlestick
                            color: { // from chartjs-chart-financial for up/down colors
                                up: 'rgba(76, 175, 80, 1)',    // Green for up
                                down: 'rgba(244, 67, 54, 1)',  // Red for down
                                unchanged: 'rgba(158, 158, 158, 1)' // Grey for unchanged
                            }
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 0 }, // Disable animation for real-time updates
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'minute', // Adjust based on timeframe, e.g. 'hour', 'day'
                                    tooltipFormat: 'MMM dd, HH:mm',
                                    displayFormats: { minute: 'HH:mm', hour: 'MMM dd, HH:mm' }
                                },
                                ticks: { color: '#ccc', maxRotation: 0, autoSkip: true, autoSkipPadding: 15 },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            },
                            y: {
                                ticks: { color: '#ccc' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            }
                        },
                        plugins: {
                            legend: { display: true, labels: { color: '#ccc'} },
                            tooltip: {
                                enabled: true,
                                mode: 'index',
                                intersect: false,
                            },
                            annotation: { // For FIB, SL, Entry, Exit lines/points
                                annotations: {} // Will be populated by updateAnnotations
                            }
                        }
                    }
                });
                 if (initialData.indicators) {
                    updateAnnotations(initialData.indicators);
                }
            }
            
            function updateAnnotations(indicators) {
                if (!chart || !indicators) return;
                const annotations = {};
                let annotationIndex = 0;

                if (indicators.fib_0_5 && indicators.fib_0_5.active && indicators.fib_0_5.price) {
                    annotations['fibLine' + annotationIndex++] = {
                        type: 'line',
                        yMin: indicators.fib_0_5.price,
                        yMax: indicators.fib_0_5.price,
                        borderColor: 'rgba(255, 165, 0, 0.7)', // Orange
                        borderWidth: 2,
                        borderDash: [5, 5],
                        label: {
                            content: 'FIB 0.5: ' + indicators.fib_0_5.price.toFixed(5),
                            enabled: true,
                            position: 'start',
                            backgroundColor: 'rgba(255, 165, 0, 0.7)',
                            color: 'black',
                            font: { size: 10 }
                        }
                    };
                }

                if (indicators.current_sl && indicators.current_sl.active && indicators.current_sl.price) {
                    annotations['slLine' + annotationIndex++] = {
                        type: 'line',
                        yMin: indicators.current_sl.price,
                        yMax: indicators.current_sl.price,
                        borderColor: 'rgba(220, 53, 69, 0.7)', // Reddish
                        borderWidth: 2,
                        label: {
                            content: (indicators.current_sl.type || 'SL') + ': ' + indicators.current_sl.price.toFixed(5),
                            enabled: true,
                            position: 'start',
                            backgroundColor: 'rgba(220, 53, 69, 0.7)',
                            color: 'white',
                            font: { size: 10 }
                        }
                    };
                }
                
                (indicators.entries || []).forEach((entry, i) => {
                     annotations['entryPoint' + annotationIndex++] = {
                        type: 'point',
                        xValue: entry.x,
                        yValue: entry.y,
                        backgroundColor: 'rgba(0, 255, 0, 0.8)', // Bright Green
                        borderColor: 'rgba(0,128,0,1)',
                        borderWidth: 1,
                        radius: 6,
                        label: {
                            content: 'BUY: ' + entry.y.toFixed(5),
                            enabled: true,
                            position: 'top',
                            backgroundColor: 'rgba(0, 200, 0, 0.7)',
                            color: 'white',
                            font: {size: 9},
                            yAdjust: -10
                        }
                    };
                });

                (indicators.exits || []).forEach((exit, i) => {
                    annotations['exitPoint' + annotationIndex++] = {
                        type: 'point',
                        xValue: exit.x,
                        yValue: exit.y,
                        backgroundColor: 'rgba(255, 0, 0, 0.8)', // Bright Red
                        borderColor: 'rgba(128,0,0,1)',
                        borderWidth: 1,
                        radius: 6,
                         label: {
                            content: (exit.text || 'SELL') + ': ' + exit.y.toFixed(5),
                            enabled: true,
                            position: 'bottom',
                            backgroundColor: 'rgba(200, 0, 0, 0.7)',
                            color: 'white',
                            font: {size: 9},
                            yAdjust: 10
                        }
                    };
                });
                
                // Pivots (draw as points)
                // This needs more logic if you store history of pivots.
                // For simplicity, if a pivot is in indicators, draw it near the latest candles.
                // A better way is to have timestamp for pivots.
                // For now, let's assume it's a horizontal line for the latest confirmed pivot.
                if (indicators.pivot_high && indicators.pivot_high.price && chart.data.datasets[0].data.length > 0) {
                    annotations['pivotHighLine' + annotationIndex++] = {
                        type: 'line',
                        yMin: indicators.pivot_high.price,
                        yMax: indicators.pivot_high.price,
                        // xMin: chart.data.datasets[0].data[Math.max(0, chart.data.datasets[0].data.length - 10)].x, // Draw near last few candles
                        // xMax: chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1].x,
                        borderColor: 'rgba(255, 0, 255, 0.5)', // Magenta
                        borderWidth: 1,
                        borderDash: [2,2],
                        label: { content: 'PH: ' + indicators.pivot_high.price.toFixed(5), enabled: true, position: 'end', font: {size:9}, color: 'magenta'}
                    };
                }
                 if (indicators.pivot_low && indicators.pivot_low.price && chart.data.datasets[0].data.length > 0) {
                    annotations['pivotLowLine' + annotationIndex++] = {
                        type: 'line',
                        yMin: indicators.pivot_low.price,
                        yMax: indicators.pivot_low.price,
                        borderColor: 'rgba(0, 255, 255, 0.5)', // Cyan
                        borderWidth: 1,
                        borderDash: [2,2],
                        label: { content: 'PL: ' + indicators.pivot_low.price.toFixed(5), enabled: true, position: 'end', font: {size:9}, color: 'cyan'}
                    };
                }


                chart.options.plugins.annotation.annotations = annotations;
            }


            // Fetch initial data to create the chart, then connect to SSE
            fetch(`/initial-chart-data/${pairId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.candles && data.candles.length > 0) {
                        createChart(data);
                        connectSSE();
                    } else {
                        statusMessage.textContent = "No initial data available for this pair. Waiting for data...";
                        // Try to connect to SSE anyway, maybe data will arrive
                        connectSSE();
                         // Create an empty chart
                        createChart({candles:[], indicators:{}});

                    }
                })
                .catch(error => {
                    console.error("Error fetching initial data:", error);
                    statusMessage.textContent = "Error fetching initial data. Check console.";
                     createChart({candles:[], indicators:{}}); // Create empty chart on error
                });


            function connectSSE() {
                const eventSource = new EventSource(`/sse-data/${pairId}`);
                statusMessage.textContent = `Connected to ${pairId} stream. Waiting for updates...`;

                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    statusMessage.textContent = `Data received at ${new Date().toLocaleTimeString()} for ${pairId}`;

                    if (chart && data.candles) {
                        chart.data.datasets[0].data = data.candles;
                        if (data.indicators) {
                           updateAnnotations(data.indicators);
                        }
                        chart.update('none'); // 'none' for no animation
                    } else if (!chart && data.candles && data.candles.length > 0) {
                        // If chart wasn't created due to no initial data, create it now
                        createChart(data);
                    }
                };

                eventSource.onerror = function(err) {
                    console.error("EventSource failed:", err);
                    statusMessage.textContent = "Stream connection error. Retrying or check server.";
                    eventSource.close();
                    // Optional: Implement a retry mechanism
                    setTimeout(connectSSE, 5000); // Retry after 5 seconds
                };
            }
        </script>
    </body>
    </html>
    """

    @app_flask.route('/')
    def home_page():
        global active_trading_pairs_for_web
        # active_trading_pairs_for_web should be updated by start_trading
        return render_template_string(HOME_PAGE_HTML, pairs=active_trading_pairs_for_web)

    @app_flask.route('/chart/<pair_id_url>')
    def chart_page(pair_id_url):
        # pair_id_url is like BTC-USD_hour
        display_name = pair_id_url.replace('_', ' ')
        return render_template_string(CHART_PAGE_HTML, pair_id_raw=pair_id_url, pair_id_display=display_name)

    @app_flask.route('/initial-chart-data/<pair_id_url>')
    def initial_chart_data(pair_id_url):
        with chart_data_lock:
            if pair_id_url in chart_data_store:
                return jsonify(chart_data_store[pair_id_url])
            else:
                return jsonify({"candles": [], "indicators": {}}) # Send empty if not found

    @app_flask.route('/sse-data/<pair_id_url>')
    def sse_data(pair_id_url):
        def generate_data():
            last_sent_update_time = 0
            while True:
                with chart_data_lock:
                    pair_data = chart_data_store.get(pair_id_url)
                    current_update_time = pair_data.get("last_update", 0) if pair_data else 0

                if pair_data and current_update_time > last_sent_update_time:
                    # Hanya kirim data lengkap, bukan delta, untuk kesederhanaan
                    data_to_send = {
                        "candles": pair_data.get("candles", []),
                        "indicators": pair_data.get("indicators", {})
                    }
                    yield f"data: {json.dumps(data_to_send)}\n\n"
                    last_sent_update_time = current_update_time
                else:
                    # Send a comment to keep the connection alive if no new data
                    yield ": keep-alive\n\n"
                time.sleep(1) # Cek update setiap 1 detik
        return Response(generate_data(), mimetype='text/event-stream')

    def run_flask_app(port):
        log_info(f"Starting Flask web server on port {port}...", "WEB_SERVER")
        try:
            # Werkzeug adalah development server, untuk produksi mungkin perlu Gunicorn/Waitress
            # tapi untuk Termux & localhost, ini cukup.
            app_flask.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        except Exception as e:
            log_error(f"Could not start Flask web server: {e}", "WEB_SERVER")

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    flask_thread = None

    api_s_for_web = settings.get("api_settings", {})
    if FLASK_AVAILABLE and api_s_for_web.get("enable_web_server", True):
        port = api_s_for_web.get("web_server_port", WEB_SERVER_PORT)
        flask_thread = threading.Thread(target=run_flask_app, args=(port,), daemon=True)
        flask_thread.start()
    elif not FLASK_AVAILABLE and api_s_for_web.get("enable_web_server", True):
        log_warning("Web server diaktifkan di pengaturan tapi Flask tidak tersedia.", "SYSTEM")


    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Web Chart) =========", color=AnsiColors.HEADER, delay=0.005)

        pick_title_main = ""
        active_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs)}) ---\n"
            for i, cfg in enumerate(active_configs):
                pick_title_main += f"  {i+1}. {cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')} (TF: {cfg.get('timeframe','N/A')}, Exch: {cfg.get('exchange','N/A')})\n"
        else:
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s = settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
             primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len([k for k in api_s.get('recovery_keys',[]) if k])
        termux_notif_main_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        web_server_main_status = "Aktif" if api_s.get("enable_web_server", True) and FLASK_AVAILABLE else "Nonaktif"
        web_server_main_port = api_s.get("web_server_port", WEB_SERVER_PORT)


        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display} | Recovery Keys: {num_recovery_keys}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status}\n"
        if FLASK_AVAILABLE:
            pick_title_main += f"Web Server Chart: {AnsiColors.GREEN if web_server_main_status == 'Aktif' else AnsiColors.RED}{web_server_main_status}{AnsiColors.ENDC} (Port: {web_server_main_port})\n"
            if web_server_main_status == 'Aktif':
                 pick_title_main += f"   Akses di: {AnsiColors.CYAN}http://localhost:{web_server_main_port}{AnsiColors.ENDC} atau {AnsiColors.CYAN}http://127.0.0.1:{web_server_main_port}{AnsiColors.ENDC}\n"
        else:
            pick_title_main += f"Web Server Chart: {AnsiColors.RED}Nonaktif (Flask tidak tersedia){AnsiColors.ENDC}\n"


        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"

        options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]

        selected_index = -1
        try:
            _option_text, selected_index = pick(options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception as e_pick_main:
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main)
            for idx_main, opt_text_main in enumerate(options_plain):
                print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice_main < len(options_plain):
                    selected_index = choice_main
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue

        if selected_index == 0:
            start_trading(settings)
        elif selected_index == 1:
            settings = settings_menu(settings)
            # Jika port web server berubah, restart Flask thread (jika aktif)
            # Ini adalah penyederhanaan. Idealnya, server Flask dihentikan dengan benar.
            api_s_recheck = settings.get("api_settings", {})
            if FLASK_AVAILABLE and api_s_recheck.get("enable_web_server", True):
                new_port = api_s_recheck.get("web_server_port", WEB_SERVER_PORT)
                current_port_from_thread_args = flask_thread.args[0] if flask_thread and flask_thread.args else None
                
                if not flask_thread or not flask_thread.is_alive():
                    log_info("Flask thread tidak aktif atau belum ada. Memulai ulang jika perlu...", "SYSTEM")
                    flask_thread = threading.Thread(target=run_flask_app, args=(new_port,), daemon=True)
                    flask_thread.start()
                elif current_port_from_thread_args != new_port:
                    log_warning(f"Port web server diubah ke {new_port}. Untuk menerapkan, silakan restart skrip.", "SYSTEM")
                    # Untuk menghentikan thread Flask dengan aman itu kompleks.
                    # Cara paling mudah adalah meminta pengguna restart skrip.
            elif FLASK_AVAILABLE and not api_s_recheck.get("enable_web_server", True) and flask_thread and flask_thread.is_alive():
                 log_warning("Web server dinonaktifkan. Untuk menghentikan server yang sedang berjalan, silakan restart skrip.", "SYSTEM")


        elif selected_index == 2:
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
            show_spinner(0.5, "Exiting")
            # Tidak perlu join flask_thread secara eksplisit karena daemon=True
            break

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        log_info("Skrip Selesai.", "SYSTEM")
