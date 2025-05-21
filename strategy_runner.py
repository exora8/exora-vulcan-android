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
    print("Flask tidak terinstal. Silakan install dengan: pip install Flask")
    sys.exit(1)
# CHART_INTEGRATION_END

# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m' # Warning / Late FIB / GR1
    RED = '\033[91m'    # Error / SL
    ENDC = '\033[0m'    # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[96m'
    MAGENTA = '\033[35m'
    YELLOW_BG = '\033[43m'
    YELLOW_FG = '\033[33m' # Untuk GR2


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

log_file_name = "trading_log.txt"
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


SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500 # Minimal 200 + beberapa buffer untuk EMA
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
            print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))

    if not all([sender_email, sender_password, receiver_email]):
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
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)

def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    api_settings = global_settings.get("api_settings", {})
    if not api_settings.get("enable_termux_notifications", False):
        return

    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal dan paket termux-api sudah diinstal di Termux (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config(): # MODIFIED
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,
        # Parameter lama (pivot/fib) akan kita abaikan di logic baru, tapi biarkan di config untuk kompatibilitas jika ada yg masih pakai versi lama
        "left_strength": 50, "right_strength": 150, 
        "enable_secure_fib": True, "secure_fib_check_price": "Close", # Ini juga terkait FIB lama
        
        # Parameter Baru untuk EMA Strategy
        "ema_length": 200,
        "gr2_drop_percent": 15.0,

        # Parameter SL/TP tetap sama
        "profit_target_percent_activation": 10.0, # Sesuai Pine Script TP Activation
        "trailing_stop_gap_percent": 5.0,      # Sesuai Pine Script TP Trailing Gap
        "emergency_sl_percent": 5.0,           # Sesuai Pine Script SL Percentage

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
        "enable_termux_notifications": False
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            if "api_settings" not in settings:
                settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]:
                        settings["api_settings"][k] = v

            if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                settings["cryptos"] = []
            
            default_pair_cfg_for_migration = get_default_crypto_config() # MODIFIED: get new defaults
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                # MODIFIED: Add new fields if missing
                if "ema_length" not in crypto_cfg: crypto_cfg["ema_length"] = default_pair_cfg_for_migration["ema_length"]
                if "gr2_drop_percent" not in crypto_cfg: crypto_cfg["gr2_drop_percent"] = default_pair_cfg_for_migration["gr2_drop_percent"]
                # Ensure SL/TP parameters from default are also considered if missing
                if "profit_target_percent_activation" not in crypto_cfg: crypto_cfg["profit_target_percent_activation"] = default_pair_cfg_for_migration["profit_target_percent_activation"]
                if "trailing_stop_gap_percent" not in crypto_cfg: crypto_cfg["trailing_stop_gap_percent"] = default_pair_cfg_for_migration["trailing_stop_gap_percent"]
                if "emergency_sl_percent" not in crypto_cfg: crypto_cfg["emergency_sl_percent"] = default_pair_cfg_for_migration["emergency_sl_percent"]


            return settings
        except json.JSONDecodeError:
            log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
        except Exception as e:
            log_error(f"Error tidak diketahui saat load_settings: {e}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): # MODIFIED
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

    # --- Parameter EMA Strategy BARU ---
    animated_text_display("\n-- Parameter EMA Strategy --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["ema_length"] = int(input(f"{AnsiColors.BLUE}Panjang EMA [{new_config.get('ema_length',200)}]: {AnsiColors.ENDC}").strip() or new_config.get('ema_length',200))
        new_config["gr2_drop_percent"] = float(input(f"{AnsiColors.BLUE}GR2 - Persen Drop dari GR1 (%) [{new_config.get('gr2_drop_percent',15.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('gr2_drop_percent',15.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter EMA tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["ema_length"] = new_config.get('ema_length',200)
        new_config["gr2_drop_percent"] = new_config.get('gr2_drop_percent',15.0)
    
    # Parameter Pivot/FIB lama bisa disembunyikan atau diberi catatan bahwa ini tidak dipakai oleh EMA strategy
    # animated_text_display("\n-- Parameter Pivot (Tidak digunakan EMA Strategy) --", color=AnsiColors.ORANGE, delay=0.01)
    # try:
    #     new_config["left_strength"] = int(input(f"Left Strength [{new_config.get('left_strength',50)}]: ").strip() or new_config.get('left_strength',50))
    #     new_config["right_strength"] = int(input(f"Right Strength [{new_config.get('right_strength',150)}]: ").strip() or new_config.get('right_strength',150))
    # except ValueError:
    #     new_config["left_strength"] = new_config.get('left_strength',50)
    #     new_config["right_strength"] = new_config.get('right_strength',150)

    animated_text_display("\n-- Parameter Trading (SL/TP) --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}TP - Aktivasi Trailing (%) [{new_config.get('profit_target_percent_activation',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',10.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}TP - Gap Trailing (%) [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}SL - Fixed Percentage (%) [{new_config.get('emergency_sl_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',5.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter trading tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',10.0)
        new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0)
        new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',5.0)

    # Secure FIB tidak relevan untuk EMA strategy
    # animated_text_display("\n-- Fitur Secure FIB (Tidak digunakan EMA Strategy) --", color=AnsiColors.ORANGE, delay=0.01)
    # enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower().strip()
    # new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))
    # secure_fib_price_input = (input(f"Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: ").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
    # if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input

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

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
        pick_title_settings += "Strategi Aktif: EMA GR1/GR2\n" # MODIFIED: Info strategy
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"

        if not current_settings.get("cryptos"):
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                # MODIFIED: Tampilkan info EMA Length
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} (TF: {crypto_conf.get('timeframe','N/A')}, EMA: {crypto_conf.get('ema_length', 'N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

        original_options_structure = [
            ("header", "--- Pengaturan API & Global ---"),
            ("option", "Atur Primary API Key"),
            ("option", "Kelola Recovery API Keys"),
            ("option", "Atur Email Global untuk Notifikasi Sistem"),
            ("option", "Aktifkan/Nonaktifkan Notifikasi Termux Realtime"),
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"),
            ("option", "Ubah Konfigurasi Crypto"),
            ("option", "Hapus Konfigurasi Crypto"),
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")
        ]

        selectable_options = [text for type, text in original_options_structure if type == "option"]
        
        selected_option_text = None
        action_choice = -1

        try:
            selected_option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick:
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings)
            for idx, opt_text in enumerate(selectable_options):
                print(f"  {idx + 1}. {opt_text}")
            try:
                choice_input = input("Pilih nomor opsi: ").strip()
                if not choice_input: continue
                choice = int(choice_input) -1
                if 0 <= choice < len(selectable_options):
                    action_choice = choice
                    selected_option_text = selectable_options[choice]
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue
        
        if selected_option_text is not None and action_choice < 0:
             pass
        elif action_choice < 0 and selected_option_text is None:
            continue

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
                    
                    rec_selected_text = None
                    rec_index = -1
                    try:
                        rec_selected_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e_pick_rec:
                        log_debug(f"Error dengan library 'pick' di menu recovery: {e_pick_rec}. Gunakan input manual.")
                        print(recovery_pick_title)
                        for idx_rec, opt_text_rec in enumerate(recovery_options_plain):
                            print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice_input = input("Pilih nomor opsi: ").strip()
                            if not rec_choice_input: continue
                            rec_choice_val = int(rec_choice_input) -1
                            if 0 <= rec_choice_val < len(recovery_options_plain):
                                rec_index = rec_choice_val
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
                            show_spinner(1, "Kembali..."); continue
                        try:
                            del_options = []
                            for i_del, r_key_del in enumerate(current_recovery):
                                r_key_del_display = r_key_del[:5] + "..." + r_key_del[-3:] if len(r_key_del) > 8 else r_key_del
                                del_options.append(f"{r_key_del_display}")
                            del_options.append("Batal")

                            del_title = "Pilih recovery key yang akan dihapus:"
                            _del_text, idx_del_pick = pick(del_options, del_title, indicator='=>')
                            
                            if idx_del_pick == len(del_options) -1 : # Batal
                                 show_spinner(0.5, "Dibatalkan..."); continue

                            if 0 <= idx_del_pick < len(current_recovery):
                                removed = current_recovery.pop(idx_del_pick)
                                api_s['recovery_keys'] = current_recovery
                                save_settings(current_settings)
                                print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                            else:
                                print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}")
                        except Exception as e_pick_del:
                            log_debug(f"Pick untuk hapus recovery key dibatalkan atau error: {e_pick_del}")
                            print(f"{AnsiColors.ORANGE}Penghapusan dibatalkan.{AnsiColors.ENDC}")
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
            elif action_choice == 3: # Notifikasi Termux
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
            elif action_choice == 4: # Tambah Konfigurasi Crypto
                new_crypto_conf = get_default_crypto_config()
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf)
                current_settings.setdefault("cryptos", []).append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 5: # Ubah Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue
                
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                edit_options = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]
                edit_options.append("Batal")
                
                edit_title = "Pilih konfigurasi crypto yang akan diubah:"
                _edit_text, idx_choice_pick = pick(edit_options, edit_title, indicator='=>')

                if idx_choice_pick == len(edit_options) -1 : # Batal
                     show_spinner(0.5, "Dibatalkan..."); continue

                if 0 <= idx_choice_pick < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice_pick] = _prompt_crypto_config(current_settings["cryptos"][idx_choice_pick])
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice_pick]['symbol']}-{current_settings['cryptos'][idx_choice_pick]['currency']} diubah.")
                else: print(f"{AnsiColors.RED}Pilihan ubah tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")

            elif action_choice == 6: # Hapus Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue

                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                del_crypto_options = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]
                del_crypto_options.append("Batal")
                
                del_crypto_title = "Pilih konfigurasi crypto yang akan dihapus:"
                _del_c_text, idx_del_c_pick = pick(del_crypto_options, del_crypto_title, indicator='=>')

                if idx_del_c_pick == len(del_crypto_options) - 1: # Batal
                    show_spinner(0.5, "Dibatalkan..."); continue

                if 0 <= idx_del_c_pick < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_del_c_pick]['symbol']}-{current_settings['cryptos'][idx_del_c_pick]['currency']}"
                    current_settings["cryptos"].pop(idx_del_c_pick)
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                else: print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 7: # Kembali ke Menu Utama
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
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: {key_display}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20)

            if response.status_code in [401, 403, 429]: # Termasuk 429 (Rate Limit) sebagai APIKeyError
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error/Rate Limit (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded", # double check
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit", # double check
                    "please pass an API key", "api_key not found"
                ]
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error/Rate Limit (JSON): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
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
                required_ohlcv_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in required_ohlcv_keys):
                    log_warning(f"Candle tidak lengkap dari API @ ts {item.get('time', 'N/A')}. Dilewati. Data: {item}", pair_name=pair_name)
                    continue

                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom'),
                    'ema': None # MODIFIED: Tambah field ema
                }
                batch_candles_list.append(candle)

            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                     if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih (overlap): {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                     batch_candles_list.pop()


            if not batch_candles_list and current_to_ts is not None :
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data atau hanya ada 1 candle overlap.", pair_name=pair_name)
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
                time.sleep(0.3) # Small delay for very large fetches

        except APIKeyError: # Propagate this error
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Break from while loop for this pair for this fetch attempt
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles (batch loop): {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error Fetch Candles (batch loop):", pair_name=pair_name)
            break # Break from while loop

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state(): # MODIFIED
    return {
        # State lama (beberapa mungkin tidak terpakai lagi, tapi tidak apa-apa ada)
        "last_signal_type": 0, # Tidak dipakai di EMA GR1/GR2, tapi bisa di-nol-kan
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None,
        "last_pivot_low_display_info": None,
        "high_price_for_fib": None,
        "high_bar_index_for_fib": None,
        "active_fib_level": None,
        "active_fib_line_start_index": None,
        
        # State untuk Trading (SL/TP) - tetap dipakai
        "entry_price_custom": None,
        "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None,
        "position_size": 0,

        # State BARU untuk EMA GR1/GR2 Strategy
        "inGetReady1": False,
        "inGetReady2": False,
        "priceAtGR1Close": None,
        # EMA value tidak disimpan di state, tapi dihitung per candle
    }

# FUNGSI BARU: Helper untuk kalkulasi EMA
def calculate_emas(candles, period):
    if not candles or len(candles) < 1: # Cukup ada 1 candle untuk mulai
        for c in candles: c['ema'] = None
        return

    # Pastikan semua candle punya key 'ema', default None jika belum ada
    for c in candles:
        if 'ema' not in c:
            c['ema'] = None

    if len(candles) < period:
        # Tidak cukup data untuk SMA awal, semua EMA jadi None
        for i in range(len(candles)):
            candles[i]['ema'] = None
        return
    
    # Calculate SMA for the first EMA value
    # Hanya hitung SMA jika EMA di candle[period-1] belum ada (misal, dari proses sebelumnya)
    if candles[period-1]['ema'] is None:
        closes_for_sma = [c['close'] for c in candles[0:period] if c['close'] is not None]
        if len(closes_for_sma) == period :
            sma_sum = sum(closes_for_sma)
            candles[period-1]['ema'] = sma_sum / period
        else: # Gagal hitung SMA jika ada data close yang None
            for i in range(len(candles)): candles[i]['ema'] = None # Reset semua jika ada masalah
            return


    # Calculate EMA for the rest
    multiplier = 2 / (period + 1)
    for i in range(period, len(candles)):
        # Hanya hitung EMA baru jika EMA sebelumnya valid dan close saat ini valid
        if candles[i-1].get('ema') is not None and candles[i].get('close') is not None:
            candles[i]['ema'] = (candles[i]['close'] * multiplier) + \
                                (candles[i-1]['ema'] * (1 - multiplier))
        # Jika EMA sebelumnya tidak valid, EMA saat ini juga tidak bisa dihitung (kecuali ini adalah titik awal lagi)
        # Jika candle saat ini 'close' nya None, EMA juga None
        else:
            candles[i]['ema'] = None # Biarkan atau set None jika tidak bisa dihitung
    return candles # Kembalikan list candles yang sudah diupdate


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings): # MODIFIED TOTAL
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    ema_length = crypto_config.get('ema_length', 200)
    gr2_required_drop_percentage = crypto_config.get('gr2_drop_percent', 15.0) / 100.0

    # 1. Validasi data candle dasar
    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"Data candle kosong atau kunci OHLC tidak lengkap.", pair_name=pair_name)
        return strategy_state
    
    if len(candles_history) < 2: # Butuh minimal 2 candle untuk cek cross (current dan previous)
        log_debug(f"Tidak cukup candle (butuh min 2, ada {len(candles_history)}) untuk proses logika.", pair_name=pair_name)
        return strategy_state

    # 2. Hitung EMA untuk semua candle di history
    # Fungsi calculate_emas akan memodifikasi candles_history secara langsung
    candles_history = calculate_emas(candles_history, ema_length)

    # 3. Dapatkan data candle saat ini dan sebelumnya
    current_candle_idx = len(candles_history) - 1
    previous_candle_idx = len(candles_history) - 2

    current_candle = candles_history[current_candle_idx]
    previous_candle = candles_history[previous_candle_idx]

    # Validasi kelengkapan data OHLC dan EMA setelah kalkulasi
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp', 'ema']) or \
       any(previous_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp', 'ema']):
        log_debug(f"Data OHLC atau EMA tidak lengkap untuk candle terbaru/sebelumnya. Skip evaluasi. Current EMA: {current_candle.get('ema')}, Prev EMA: {previous_candle.get('ema')}", pair_name=pair_name)
        return strategy_state

    current_close = current_candle['close']
    previous_close = previous_candle['close']
    current_ema = current_candle['ema']
    previous_ema = previous_candle['ema']
    current_low = current_candle['low'] # Untuk GR2 check

    # 4. Deteksi Crossover dan Crossunder EMA
    is_cross_under_ema = previous_close > previous_ema and current_close < current_ema
    is_cross_over_ema  = previous_close < previous_ema and current_close > current_ema
    
    # --- LOGIKA INTI STRATEGI BARU (EMA GR1/GR2) ---

    # Cek apakah ada trade yang baru saja ditutup untuk mereset GR states
    # Ini dilakukan di bagian exit logic jika position_size berubah jadi 0
    
    # Kondisi GR1 (Get Ready 1)
    if is_cross_under_ema and strategy_state["position_size"] == 0 and \
       not strategy_state["inGetReady1"] and not strategy_state["inGetReady2"]:
        strategy_state["inGetReady1"] = True
        strategy_state["inGetReady2"] = False # Reset GR2 jika GR1 baru terpicu
        strategy_state["priceAtGR1Close"] = current_close
        log_info(f"{AnsiColors.ORANGE}GR1 Aktif: Harga cross DI BAWAH EMA. Harga GR1: {current_close:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

    # Kondisi GR2 (Get Ready 2)
    if strategy_state["inGetReady1"] and not strategy_state["inGetReady2"]:
        price_at_gr1 = strategy_state["priceAtGR1Close"]
        if price_at_gr1 is not None and price_at_gr1 > 0: # Pastikan price_at_gr1 valid
            current_drop_from_gr1_price_actual = (price_at_gr1 - current_low) / price_at_gr1
            if current_drop_from_gr1_price_actual >= gr2_required_drop_percentage:
                strategy_state["inGetReady2"] = True
                strategy_state["inGetReady1"] = False # GR1 terpenuhi, sekarang di GR2
                drop_perc_display = current_drop_from_gr1_price_actual * 100
                log_info(f"{AnsiColors.YELLOW_FG}GR2 Aktif: Harga turun {drop_perc_display:.2f}% dari harga GR1 ({price_at_gr1:.5f}) ke low {current_low:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

    # Kondisi Entry (Long)
    if strategy_state["inGetReady2"] and is_cross_over_ema and strategy_state["position_size"] == 0:
        entry_px = current_close
        strategy_state["position_size"] = 1 # Masuk posisi
        strategy_state["entry_price_custom"] = entry_px
        strategy_state["highest_price_for_trailing"] = entry_px # Inisialisasi untuk trailing TP
        strategy_state["trailing_tp_active_custom"] = False
        strategy_state["current_trailing_stop_level"] = None

        # Hitung Emergency SL berdasarkan harga entry
        emerg_sl_val = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
        strategy_state["emergency_sl_level_custom"] = emerg_sl_val
        
        # Reset GR2 setelah entry
        strategy_state["inGetReady2"] = False 
        # GR1 juga sudah false dari saat GR2 aktif

        log_msg = f"BUY ENTRY @ {entry_px:.5f} (EMA Crossover setelah GR2). Emerg SL: {emerg_sl_val:.5f}"
        log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
        play_notification_sound()
        
        termux_title = f"BUY Signal (EMA GR1/GR2): {pair_name}"
        termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl_val:.5f}"
        send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)

        email_subject = f"BUY Signal (EMA GR1/GR2): {pair_name}"
        email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']} (EMA GR1/GR2 Strategy).\n\n"
                      f"Entry Price: {entry_px:.5f}\n"
                      f"EMA ({ema_length}) Value: {current_ema:.5f}\n"
                      f"Emergency SL: {emerg_sl_val:.5f}\n"
                      f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})

    # --- MANAJEMEN POSISI (SL/TP - Kode lama dipertahankan) ---
    if strategy_state["position_size"] > 0:
        current_high_for_trailing_update = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing_update is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None) saat update.", pair_name=pair_name)
        else:
            strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing_update , current_candle['high'])

        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: # Hindari ZeroDivisionError
                profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None:
                profit_percent = 0.0
                log_warning("highest_price_for_trailing is None saat kalkulasi profit untuk aktivasi Trailing TP.", pair_name=pair_name)
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High Sejak Entry: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=pair_name)

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
            exit_price_open_candle = current_candle.get('open') # Exit di open candle berikutnya, atau SL jika gap down
            if exit_price_open_candle is None:
                log_warning("Harga open candle tidak ada untuk kalkulasi exit. Menggunakan SL sebagai harga exit.", pair_name=pair_name)
                exit_price = final_stop_for_exit
            else: # Harga exit adalah harga open candle jika lebih buruk dari SL, atau SL itu sendiri jika harga open lebih baik.
                exit_price = min(exit_price_open_candle, final_stop_for_exit)


            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if exit_comment == "Trailing Stop" and pnl < 0: # Jika trailing stop kena rugi
                exit_color = AnsiColors.RED

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

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
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})

            # Reset state trading
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            
            # MODIFIED: Reset state GR1/GR2 setelah trade ditutup
            log_info(f"Resetting GR1/GR2 states for {pair_name} after trade closure.", pair_name=pair_name)
            strategy_state["inGetReady1"] = False
            strategy_state["inGetReady2"] = False
            strategy_state["priceAtGR1Close"] = None

        elif strategy_state["position_size"] > 0 : # Jika masih dalam posisi, log status SL
            plot_stop_level = strategy_state.get("emergency_sl_level_custom")
            stop_type_info = "Emergency SL"
            if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
                current_trailing_sl = strategy_state.get("current_trailing_stop_level")
                if plot_stop_level is None or (current_trailing_sl is not None and current_trailing_sl > plot_stop_level):
                    plot_stop_level = current_trailing_sl
                    stop_type_info = "Trailing SL"

            entry_price_display = strategy_state.get('entry_price_custom', 0)
            sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
            # Log detail posisi aktif (opsional, bisa jadi terlalu verbose)
            # log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}, High: {strategy_state.get('highest_price_for_trailing',0):.5f}", pair_name=pair_name)

    return strategy_state

# CHART_INTEGRATION_START
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot): # MODIFIED for EMA series
    if pair_id_to_display not in current_data_manager_snapshot:
        log_warning(f"Data untuk pair {pair_id_to_display} tidak ditemukan di snapshot untuk chart.", pair_name="SYSTEM_CHART")
        return None

    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]
    candles_full_history_orig = pair_specific_data.get("all_candles_list", []) # Ini mungkin belum ada EMA nya
    current_strategy_state = pair_specific_data.get("strategy_state", {})
    pair_config = pair_specific_data.get("config", {})
    
    # Penting: Kalkulasi ulang EMA di sini untuk data chart jika tidak ada di `all_candles_list`
    # atau pastikan `all_candles_list` di `shared_crypto_data_manager` selalu punya EMA.
    # Untuk konsistensi, mari kita asumsikan `all_candles_list` sudah memiliki EMA dari `run_strategy_logic`
    # yang diperbarui oleh thread utama.
    candles_full_history = copy.deepcopy(candles_full_history_orig) # Salin agar tidak modifikasi data asli di thread lain
    
    # Jika EMA belum ada di candles_full_history, hitung di sini
    # (Ini bisa terjadi jika `run_strategy_logic` tidak selalu menyimpan EMA ke `all_candles_list` di shared manager)
    # Cara yang lebih baik adalah memastikan thread utama yang menjalankan `run_strategy_logic`
    # mengupdate `all_candles_list` di `shared_crypto_data_manager` dengan EMA terpasang.
    # Untuk sementara, kita bisa panggil `calculate_emas` di sini lagi pada `candles_full_history`
    # jika 'ema' tidak ada di candle pertama (sebagai indikasi). Ini kurang ideal karena duplikasi kalkulasi.
    
    # Cek jika 'ema' ada, jika tidak, hitung (darurat, idealnya sudah ada)
    if candles_full_history and 'ema' not in candles_full_history[0]:
        log_warning(f"EMA tidak ditemukan di data candle untuk chart pair {pair_id_to_display}. Menghitung ulang EMA untuk chart.", pair_name="SYSTEM_CHART")
        candles_full_history = calculate_emas(candles_full_history, pair_config.get('ema_length', 200))


    candles_for_chart_display = candles_full_history[-TARGET_BIG_DATA_CANDLES:]


    ohlc_data_points = []
    ema_data_points = [] # MODIFIED: Tambah untuk EMA
    if not candles_for_chart_display:
        log_warning(f"Tidak ada candle di `candles_for_chart_display` untuk {pair_id_to_display}.", pair_name="SYSTEM_CHART")
        # MODIFIED: Return series_data
        return {"series_data": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None}


    for candle in candles_for_chart_display:
        required_candle_keys = ['timestamp', 'open', 'high', 'low', 'close']
        if all(k in candle and candle[k] is not None for k in required_candle_keys):
            ohlc_data_points.append({
                'x': candle['timestamp'].timestamp() * 1000,
                'y': [candle['open'], candle['high'], candle['low'], candle['close']]
            })
            # MODIFIED: Tambahkan data EMA jika ada
            if candle.get('ema') is not None:
                ema_data_points.append({
                    'x': candle['timestamp'].timestamp() * 1000,
                    'y': candle['ema']
                })
        else:
            log_debug(f"Skipping incomplete candle for chart: {candle.get('timestamp')}", pair_name="SYSTEM_CHART")

    # MODIFIED: Struktur series untuk ApexCharts
    chart_series_data = [
        {"name": "Candlestick", "type": "candlestick", "data": ohlc_data_points}
    ]
    if ema_data_points:
        chart_series_data.append({
            "name": f"EMA({pair_config.get('ema_length', 200)})", 
            "type": "line", 
            "data": ema_data_points,
            "color": "#00FFFF" # Cyan color for EMA line
        })


    chart_annotations_yaxis = []
    chart_annotations_points = []

    # Anotasi untuk GR1 (jika aktif)
    if current_strategy_state.get("inGetReady1") and current_strategy_state.get("priceAtGR1Close") is not None:
        price_gr1 = current_strategy_state.get("priceAtGR1Close")
        chart_annotations_yaxis.append({
            'y': price_gr1,
            'borderColor': AnsiColors.ORANGE.replace('\033[', '#').replace('m',''), # Gunakan kode hex jika memungkinkan, atau warna standar
            'strokeDashArray': 2,
            'label': {
                'borderColor': '#FFA500', # Orange
                'style': {'color': '#000', 'background': '#FFA500', 'fontSize': '10px'},
                'text': f'GR1 Price: {price_gr1:.5f}'
            }
        })
    
    # Anotasi untuk GR2 (jika aktif) - Mungkin lebih baik sebagai background color, tapi yaxis line sementara
    if current_strategy_state.get("inGetReady2"):
        # Tidak ada level harga spesifik untuk GR2, ini adalah *state*
        # Bisa ditandai dengan cara lain, misal teks di chart
        pass


    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") is not None:
        entry_price_val = current_strategy_state.get("entry_price_custom")
        
        if ohlc_data_points: # Hanya tambah jika ada data candle utama
             chart_annotations_yaxis.append({
                'y': entry_price_val,
                'borderColor': '#2698FF', # Biru
                'strokeDashArray': 4,
                'label': {
                    'borderColor': '#2698FF',
                    'style': {'color': '#fff', 'background': '#2698FF', 'fontSize': '10px'},
                    'text': f'Entry: {entry_price_val:.5f}'
                }
            })

        sl_level_val = current_strategy_state.get("emergency_sl_level_custom")
        sl_type_text = "Emerg. SL"
        if current_strategy_state.get("trailing_tp_active_custom") and current_strategy_state.get("current_trailing_stop_level") is not None:
            current_trailing_sl_val = current_strategy_state.get("current_trailing_stop_level")
            if sl_level_val is None or (current_trailing_sl_val is not None and current_trailing_sl_val > sl_level_val):
                sl_level_val = current_trailing_sl_val
                sl_type_text = "Trail. SL"
        
        if sl_level_val and ohlc_data_points:
            chart_annotations_yaxis.append({
                'y': sl_level_val,
                'borderColor': '#FF4560', # Merah
                'label': {
                    'borderColor': '#FF4560',
                    'style': {'color': '#fff', 'background': '#FF4560', 'fontSize': '10px'},
                    'text': f'{sl_type_text}: {sl_level_val:.5f}'
                }
            })
    
    # Pivot high/low annotations (jika masih relevan atau ingin dipertahankan dari skrip lama untuk visualisasi umum)
    # Saya akan biarkan ini untuk sekarang, karena fokus ke EMA strategy.
    # Jika logic pivot dihapus total dari run_strategy_logic, maka last_pivot_high/low_display_info tidak akan terupdate.
    # Jadi, ini mungkin tidak akan muncul lagi.

    return { # MODIFIED: "series_data"
        "series_data": chart_series_data,
        "annotations_yaxis": chart_annotations_yaxis,
        "annotations_points": chart_annotations_points, # Mungkin kosong jika pivot tidak dipakai
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": candles_for_chart_display[-1]['timestamp'].timestamp() * 1000 if candles_for_chart_display else None
    }

flask_app_instance = Flask(__name__)

HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crypto Chart - EMA GR1/GR2 Strategy</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;}
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width:100%; max-width: 1200px; }
        #controls label { font-size: 0.9em; }
        select, button { padding: 8px 12px; font-size:0.9em; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; }
        button:hover { background-color: #444; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        h1 { text-align: center; color: #00bcd4; margin-top: 0; margin-bottom:15px; font-size:1.5em; }
        #lastUpdatedLabel { font-size: 0.8em; color: #aaa; margin-left: auto; /* Aligns to the right */ padding-right: 10px; }
        .apexcharts-tooltip-candlestick { background: #333 !important; color: #fff !important; border: 1px solid #555 !important;}
        .apexcharts-tooltip-candlestick .value { font-weight: bold; }
        .apexcharts-marker-inverted .apexcharts-marker-poly { transform: rotate(180deg); transform-origin: center; } /* For inverted triangle */
    </style>
</head>
<body>
    <h1>Live Strategy Chart (EMA GR1/GR2)</h1>
    <div id="controls">
        <label for="pairSelector">Pilih Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh Manual</button>
        <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container">
        <div id="chart"></div>
    </div>

    <script>
        let activeChart;
        let currentSelectedPairId = '';
        let lastKnownDataTimestamp = null;
        let autoRefreshIntervalId = null;
        let isLoadingData = false; 

        const initialChartOptions = {
            // MODIFIED: series is now an array, can contain multiple series objects
            series: [], 
            chart: { 
                type: 'candlestick', // Default type, but series can override
                height: 550,
                id: 'mainCandlestickChart',
                background: '#2a2a2a',
                animations: { enabled: true, easing: 'easeinout', speed: 500, animateGradually: { enabled: false } },
                toolbar: { show: true, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } }
            },
            theme: { mode: 'dark' },
            title: { text: 'Memuat Data Pair...', align: 'left', style: { color: '#e0e0e0', fontSize: '16px'} },
            xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
            yaxis: { 
                tooltip: { enabled: true }, 
                labels: { style: { colors: '#aaa'}, formatter: function (value) { return typeof value === 'number' ? value.toFixed(5) : value; } } 
            },
            grid: { borderColor: '#444' },
            annotations: { yaxis: [], points: [] },
            stroke: { // MODIFIED: Define stroke width for lines like EMA
                 width: [0, 2], // 0 for candlestick (no border essentially), 2 for line series (EMA)
                 curve: 'smooth'
            },
            markers: { // MODIFIED: Ensure markers for line series are small or off
                size: [0, 0] // No markers for candlestick, small/no markers for EMA line (optional)
            },
            tooltip: { 
                theme: 'dark', 
                shared: true, // Set to false if you want separate tooltips per series
                intersect: false, // Tooltip will show for nearest data point on hover, good for multiple series
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    let html = '<div class="apexcharts-tooltip-title" style="padding:5px 10px;">' + 
                               new Date(w.globals.seriesX[seriesIndex][dataPointIndex]).toLocaleString() + 
                               '</div>' +
                               '<div class="apexcharts-tooltip-series-group" style="padding:5px 10px;">';
                    
                    // Iterate over all series to display their info at this dataPointIndex
                    w.globals.series.forEach((s, i) => {
                        if (w.globals.series[i][dataPointIndex] !== undefined && w.globals.seriesNames[i]) {
                             const seriesName = w.globals.seriesNames[i];
                             if (seriesName.toLowerCase().includes('candlestick') || seriesName.toLowerCase().includes('ohlc')) {
                                const o = w.globals.seriesCandleO[i][dataPointIndex];
                                const h = w.globals.seriesCandleH[i][dataPointIndex];
                                const l = w.globals.seriesCandleL[i][dataPointIndex];
                                const c = w.globals.seriesCandleC[i][dataPointIndex];
                                if ([o,h,l,c].every(val => typeof val === 'number')) {
                                    html += '<div>'+seriesName+':&nbsp;</div>' +
                                            '<div>O: <span class="value">' + o.toFixed(5) + '</span></div>' +
                                            '<div>H: <span class="value">' + h.toFixed(5) + '</span></div>' +
                                            '<div>L: <span class="value">' + l.toFixed(5) + '</span></div>' +
                                            '<div>C: <span class="value">' + c.toFixed(5) + '</span></div><hr style="border-color:#555; margin:3px 0;">';
                                }
                             } else if (seriesName.toLowerCase().includes('ema')) { // For line series like EMA
                                const val = w.globals.series[i][dataPointIndex];
                                if (typeof val === 'number') {
                                    html += '<div>'+seriesName+': <span class="value">' + val.toFixed(5) + '</span></div>';
                                }
                             }
                        }
                    });
                    html += '</div>';
                    return html;
                }
            },
            noData: { text: 'Tidak ada data untuk ditampilkan.', align: 'center', verticalAlign: 'middle', style: { color: '#ccc', fontSize: '14px' } }
        };

        async function fetchAvailablePairs() {
            try {
                const response = await fetch('/api/available_pairs');
                if (!response.ok) throw new Error(`Gagal memuat daftar pair: ${response.status}`);
                const pairs = await response.json();
                const selectorElement = document.getElementById('pairSelector');
                selectorElement.innerHTML = ''; 
                if (pairs.length > 0) {
                    pairs.forEach(pair => {
                        const optionEl = document.createElement('option');
                        optionEl.value = pair.id;
                        optionEl.textContent = pair.name;
                        selectorElement.appendChild(optionEl);
                    });
                    currentSelectedPairId = selectorElement.value || pairs[0].id; 
                    loadChartDataForCurrentPair(); 
                } else {
                     selectorElement.innerHTML = '<option value="">Tidak ada pair aktif</option>';
                     if(activeChart) activeChart.destroy();
                     activeChart = null; 
                     document.getElementById('chart').innerHTML = '<p style="text-align:center; color:#aaa;">Tidak ada pair aktif yang dikonfigurasi di server.</p>';
                     document.getElementById('lastUpdatedLabel').textContent = "Tidak ada pair";
                }
            } catch (error) {
                console.error("Error fetching pair list:", error);
                document.getElementById('pairSelector').innerHTML = '<option value="">Error memuat pair</option>';
                if(activeChart) activeChart.destroy();
                activeChart = null;
                document.getElementById('chart').innerHTML = `<p style="text-align:center; color:red;">Error: ${error.message}</p>`;
                document.getElementById('lastUpdatedLabel').textContent = "Error";
            }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById('pairSelector').value;
            lastKnownDataTimestamp = null; 
            loadChartDataForCurrentPair();
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId) {
                console.log("Tidak ada pair ID yang dipilih.");
                if(activeChart) activeChart.updateOptions(initialChartOptions);
                document.getElementById('lastUpdatedLabel').textContent = "Pilih pair";
                return;
            }
            if (isLoadingData) {
                console.log(`Sinkronisasi data untuk ${currentSelectedPairId} sedang berjalan. Lewati sementara.`);
                return;
            }

            isLoadingData = true;
            document.getElementById('lastUpdatedLabel').textContent = `Sinkronisasi ${currentSelectedPairId}...`;
            
            try {
                const fetchResponse = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!fetchResponse.ok) {
                     let errorMsgText = `Gagal mengambil data chart: ${fetchResponse.status}`;
                     try { const errorData = await fetchResponse.json(); errorMsgText = errorData.error || errorMsgText; } catch(e){}
                     throw new Error(errorMsgText);
                }
                const chartDataPayload = await fetchResponse.json();
                
                // MODIFIED: Check chartDataPayload.series_data
                if (!chartDataPayload || !chartDataPayload.series_data || chartDataPayload.series_data.length === 0 || 
                    !chartDataPayload.series_data.find(s => (s.name.toLowerCase().includes('candlestick') || s.name.toLowerCase().includes('ohlc')) && s.data && s.data.length > 0) ) {
                    console.warn(`Data OHLC tidak diterima atau kosong untuk ${currentSelectedPairId}. Payload:`, chartDataPayload);
                    const pairDisplayName = chartDataPayload.pair_name || currentSelectedPairId;
                    const noDataOpts = {
                        ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${pairDisplayName} - Tidak Ada Data Candle` },
                        series: [], // Kosongkan series
                        annotations: { yaxis: [], points: [] },
                        noData: { text: 'Tidak ada data candle terbaru dari server.' }
                    };
                    if (!activeChart) {
                        activeChart = new ApexCharts(document.querySelector("#chart"), noDataOpts);
                        activeChart.render();
                    } else {
                        activeChart.updateOptions(noDataOpts, true, true); // Redraw, animate
                    }
                    lastKnownDataTimestamp = chartDataPayload.last_updated_tv || null;
                    document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data (kosong) @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Tidak ada data";
                    isLoadingData = false; 
                    return; 
                }

                if (chartDataPayload.last_updated_tv && chartDataPayload.last_updated_tv === lastKnownDataTimestamp) {
                    console.log("Data chart tidak berubah, tidak perlu update render.");
                    document.getElementById('lastUpdatedLabel').textContent = `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                    isLoadingData = false; 
                    return;
                }
                lastKnownDataTimestamp = chartDataPayload.last_updated_tv;
                document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "N/A";

                const newChartOptions = {
                    ...initialChartOptions, 
                    title: { ...initialChartOptions.title, text: `${chartDataPayload.pair_name} Candlestick & EMA` },
                    series: chartDataPayload.series_data, // MODIFIED: Gunakan series_data dari payload
                    annotations: { 
                        yaxis: chartDataPayload.annotations_yaxis || [], 
                        points: chartDataPayload.annotations_points || [] 
                    },
                    // Pastikan stroke dan markers disesuaikan dengan jumlah series
                    stroke: {
                        width: chartDataPayload.series_data.map(s => s.type === 'line' ? 2 : 0),
                        curve: 'smooth'
                    },
                    markers: {
                        size: chartDataPayload.series_data.map(s => s.type === 'line' ? 0 : 0) // No markers for lines
                    }
                };
                
                if (!activeChart) {
                    activeChart = new ApexCharts(document.querySelector("#chart"), newChartOptions);
                    activeChart.render();
                } else {
                    activeChart.updateOptions(newChartOptions, true, true); // Redraw, animate
                }

            } catch (error) {
                console.error("Error loading chart data:", error);
                const pairDisplayNameError = currentSelectedPairId || "Chart"; 
                if (activeChart) { 
                    activeChart.destroy();
                    activeChart = null; 
                }
                const errorChartOpts = { 
                    ...initialChartOptions, 
                    title: { ...initialChartOptions.title, text: `Error: ${pairDisplayNameError}` }, 
                    series: [], 
                    noData: { text: `Gagal memuat data: ${error.message}` } 
                };
                activeChart = new ApexCharts(document.querySelector("#chart"), errorChartOpts);
                activeChart.render();
                document.getElementById('lastUpdatedLabel').textContent = "Error update";
            } finally {
                isLoadingData = false; 
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            if (!activeChart) {
                 activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions);
                 activeChart.render();
            }
            fetchAvailablePairs(); 
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId); 
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) { 
                    console.log(`Auto-refresh chart untuk ${currentSelectedPairId}`);
                    await loadChartDataForCurrentPair();
                }
            }, 30000); // Refresh tiap 30 detik jika tab aktif dan tidak sedang loading
        });

    </script>
</body>
</html>
"""

@flask_app_instance.route('/')
def serve_index_page():
    return render_template_string(HTML_CHART_TEMPLATE)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs():
    with shared_data_lock:
        # Lakukan deep copy agar aman dari modifikasi oleh thread lain saat iterasi
        data_manager_view = copy.deepcopy(shared_crypto_data_manager)
    
    active_pairs_info = []
    for pair_identifier, pair_data_item in data_manager_view.items():
        config_item = pair_data_item.get("config", {})
        if config_item.get("enabled", True): # Hanya pair yang enabled
             active_pairs_info.append({
                "id": pair_identifier, # Gunakan ID unik (symbol-currency_timeframe)
                "name": config_item.get('pair_name', pair_identifier) # Tampilkan symbol-currency
            })
    return jsonify(active_pairs_info)


@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request):
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager:
             return jsonify({"error": f"Data untuk pair {pair_id_from_request} tidak ditemukan di server."}), 404
        
        # Deep copy untuk thread safety saat data diproses
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))

    if not pair_data_snapshot: # Jika pair_id ada tapi datanya kosong (seharusnya tidak terjadi jika ada di shared_crypto_data_manager)
        return jsonify({"error": f"Data untuk pair {pair_id_from_request} tidak ditemukan (snapshot kosong)."}), 404

    # Buat struktur sementara yang dibutuhkan oleh prepare_chart_data_for_pair
    temp_data_manager_for_prep = {pair_id_from_request: pair_data_snapshot}
    
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_data_manager_for_prep)

    # MODIFIED: Check prepared_data.series_data dan keberadaan candlestick
    if not prepared_data or not prepared_data.get("series_data") or \
       not prepared_data["series_data"].find(s => (s.name.toLowerCase().includes('candlestick') || s.name.toLowerCase().includes('ohlc')) && s.data && s.data.length > 0):
        log_warning(f"Tidak ada data OHLC yang cukup untuk ditampilkan untuk {pair_id_from_request} setelah diproses.", pair_name="SYSTEM_CHART")
        # Kembalikan struktur yang diharapkan frontend meskipun kosong
        return jsonify({
            "error": f"Tidak ada data candle yang cukup untuk ditampilkan untuk {pair_id_from_request}.", 
            "series_data": [], 
            "annotations_yaxis": [], 
            "annotations_points": [], 
            "pair_name": prepared_data.get("pair_name", pair_id_from_request) if prepared_data else pair_id_from_request, 
            "last_updated_tv": prepared_data.get("last_updated_tv") if prepared_data else None
        }), 200 # Return 200 OK dengan pesan error di body jika logicnya begitu
    
    return jsonify(prepared_data)


def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001 (atau IP Termux-mu)", pair_name="SYSTEM_CHART")
    try:
        # Menonaktifkan logging bawaan Flask yang ke console agar tidak duplikat dengan logger utama
        flask_log = logging.getLogger('werkzeug')
        flask_log.setLevel(logging.ERROR) # Hanya log error dari Flask/Werkzeug
        
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask:
        log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
        log_exception("Traceback Error Flask Server:", pair_name="SYSTEM_CHART")

# CHART_INTEGRATION_END


# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref): # MODIFIED
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Pass a reference to global settings for email notifications from APIKeyManager
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

    animated_text_display("================ MULTI-CRYPTO STRATEGY (EMA GR1/GR2) START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = ("..." + current_key_display_val[-3:]) if len(current_key_display_val) > 8 else current_key_display_val
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {} # Data manager lokal untuk thread ini

    # Inisialisasi setiap pair
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}" # ID unik
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}" # Nama tampilan

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')} | EMA: {config.get('ema_length', 'N/A')}", color=AnsiColors.MAGENTA, delay=0.01)

        local_crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [], # Akan diisi data candle OHLCV + EMA
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True,
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min,
            "data_fetch_failed_consecutively": 0 # Untuk menghitung kegagalan fetch beruntun per pair
        }
        
        # Segera update shared manager dengan state awal (kosong) agar UI bisa lihat pairnya
        with lock_ref:
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])

        # Pengambilan data awal (BIG DATA)
        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles_raw = [] # Raw dari API
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key_for_init = api_key_manager.get_current_key()
            if not current_api_key_for_init: # Jika semua key habis secara global
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break # Hentikan upaya untuk pair ini

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles_raw = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key_for_init, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Fetch berhasil, keluar loop retry
            except APIKeyError: # API Key gagal (invalid, limit, dll)
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break # Jika tidak ada key lagi, hentikan
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e: # Error jaringan
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key, akan coba lagi nanti jika loop utama mengizinkan.", pair_name=config['pair_name'])
                break # Hentikan fetch awal untuk pair ini, mungkin masalah sementara
            except Exception as e_gen: # Error lain
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break
        
        if not initial_candles_raw and not initial_fetch_successful : # Jika setelah semua retry tetap gagal
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Anggap gagal kumpulkan big data
            local_crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Tandai waktu fetch terakhir
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue # Lanjut ke pair berikutnya

        # Simpan data candle awal yang berhasil diambil
        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles_raw
        log_info(f"BIG DATA: {len(initial_candles_raw)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        # Warm-up strategy state dengan data historis (tanpa eksekusi trade aktual)
        if initial_candles_raw:
            # Hitung EMA untuk data historis sebelum warm-up
            log_info(f"BIG DATA: Menghitung EMA awal untuk {len(initial_candles_raw)} candle {config['pair_name']}...", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["all_candles_list"] = calculate_emas(
                local_crypto_data_manager[pair_id]["all_candles_list"], 
                config.get('ema_length', 200)
            )

            min_len_for_logic = config.get('ema_length', 200) + 2 # EMA length + 2 untuk prev/current check
            if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= min_len_for_logic:
                log_info(f"BIG DATA: Memproses {max(0, len(local_crypto_data_manager[pair_id]['all_candles_list']) - 1)} candle historis awal untuk inisialisasi state {config['pair_name']}...", pair_name=config['pair_name'])
                
                # Iterasi melalui data historis untuk membangun state (GR1, GR2, dll.)
                # Mulai dari index yang cukup untuk EMA dan pengecekan (misal, ema_length + 1)
                for i in range(min_len_for_logic - 1, len(local_crypto_data_manager[pair_id]["all_candles_list"])): # Loop sampai candle terakhir
                    historical_slice_for_warmup = local_crypto_data_manager[pair_id]["all_candles_list"][:i+1]
                    if len(historical_slice_for_warmup) < min_len_for_logic: continue # Pastikan slice cukup panjang

                    # Gunakan temporary state untuk warm-up, jangan biarkan trade terbuka
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Pastikan tidak ada posisi saat warm-up
                    
                    # Jalankan logic pada slice historis, hasilnya akan update temp_state_for_warmup
                    # yang kemudian menjadi strategy_state utama untuk pair ini
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice_for_warmup, config, temp_state_for_warmup, global_settings_dict
                    )
                    
                    # Pastikan tidak ada "sisa" posisi dari warm-up (seharusnya sudah dihandle di run_strategy_logic jika position_size=0)
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        log_warning(f"Warm-up menghasilkan posisi terbuka untuk {config['pair_name']}, ini seharusnya tidak terjadi. Mereset posisi.", pair_name=config['pair_name'])
                        local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0
                        # Reset juga state trading terkait lainnya
                        local_crypto_data_manager[pair_id]["strategy_state"]["entry_price_custom"] = None
                        local_crypto_data_manager[pair_id]["strategy_state"]["emergency_sl_level_custom"] = None
                        # GR state (inGetReady1, inGetReady2, priceAtGR1Close) akan diupdate oleh run_strategy_logic itu sendiri
                log_info(f"{AnsiColors.CYAN}BIG DATA: Inisialisasi state (warm-up) untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"BIG DATA: Data awal ({len(initial_candles_raw)}) untuk {config['pair_name']} tidak cukup untuk warm-up EMA logic (min: {min_len_for_logic}). State mungkin tidak optimal.", pair_name=config['pair_name'])
        else:
            log_warning(f"BIG DATA: Tidak ada data awal untuk warm-up state {config['pair_name']}.", pair_name=config['pair_name'])

        # Cek apakah target BIG DATA tercapai
        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']} setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(
                    f"Data Downloading Complete: {config['pair_name']}", 
                    f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", 
                    {**config, 'pair_name': config['pair_name']} # Kirim config pair untuk konteks email
                )
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(local_crypto_data_manager[pair_id]['all_candles_list'])} candles) untuk {config['pair_name']} ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        # Update shared manager dengan data terbaru setelah inisialisasi
        with lock_ref:
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])


    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    # Loop utama trading
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False # Untuk cek apakah ada fetch berhasil di siklus ini

            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                # Cooldown jika semua key gagal untuk pair ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        log_debug(f"Pair {pair_name_for_log} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name_for_log)
                        continue # Skip pair ini untuk siklus ini
                    else: # Cooldown selesai
                        data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal
                        log_info(f"Cooldown 1 jam untuk {pair_name_for_log} selesai. Mencoba fetch lagi.", pair_name=pair_name_for_log)


                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()

                # Tentukan interval refresh berdasarkan fase (big data atau live)
                required_interval_for_this_pair = 0
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Interval lebih agresif saat kumpulkan big data
                    if config_for_pair.get('timeframe') == "minute": required_interval_for_this_pair = 55 # Hampir tiap menit
                    elif config_for_pair.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 # Hampir tiap hari
                    else: required_interval_for_this_pair = 3580 # Hampir tiap jam
                else: # Fase live
                    required_interval_for_this_pair = config_for_pair.get('refresh_interval_seconds', 60) 

                # Cek apakah sudah waktunya refresh untuk pair ini
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Belum waktunya, skip ke pair lain

                # Waktunya proses pair ini
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval_for_this_pair}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time # Update waktu fetch terakhir
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])

                if data_per_pair["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA {pair_name_for_log} ({len(data_per_pair['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005, new_line=True)
                else:
                    animated_text_display(f"\n--- ANALISA LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {len(data_per_pair['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005, new_line=True)

                new_candles_batch_raw = []
                fetch_update_successful_for_this_pair = False
                
                # Tentukan berapa candle yang akan di-fetch
                limit_fetch_for_update = 3 # Default untuk update live (ambil beberapa candle terakhir untuk jaga-jaga)
                if data_per_pair["big_data_collection_phase_active"]:
                    limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data_per_pair["all_candles_list"])
                    if limit_fetch_needed <=0 : # Seharusnya sudah false jika ini terjadi, tapi sbg pengaman
                         fetch_update_successful_for_this_pair = True # Anggap berhasil, tidak perlu fetch
                         new_candles_batch_raw = []
                    else:
                        limit_fetch_for_update = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) # Ambil sebanyak yg dibutuhkan atau max limit API
                        limit_fetch_for_update = max(limit_fetch_for_update, 1) # Minimal 1

                # Lakukan fetch jika perlu
                if limit_fetch_for_update > 0 or data_per_pair["big_data_collection_phase_active"]: # Selalu fetch jika masih big data phase
                    max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    retries_done_for_this_pair_update = 0
                    original_api_key_index_at_start_of_pair_processing = api_key_manager.get_current_key_index() # Catat key index awal

                    while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                        current_api_key_for_attempt = api_key_manager.get_current_key()
                        if not current_api_key_for_attempt: # Semua key habis global
                            log_error(f"Semua API key habis (global) saat mencoba mengambil update untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            break # Hentikan upaya untuk pair ini

                        log_info(f"Mengambil {limit_fetch_for_update} candle untuk {pair_name_for_log} (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch_raw = fetch_candles(
                                config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_for_update, 
                                config_for_pair['exchange'], current_api_key_for_attempt, config_for_pair['timeframe'],
                                pair_name=pair_name_for_log
                            )
                            fetch_update_successful_for_this_pair = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal karena berhasil
                            any_data_fetched_this_cycle = True # Tandai ada data berhasil di-fetch di siklus ini
                            # Jika key berubah selama retry dan berhasil, log info
                            if api_key_manager.get_current_key_index() != original_api_key_index_at_start_of_pair_processing :
                                log_info(f"Fetch berhasil dengan key index {api_key_manager.get_current_key_index()} setelah retry untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                        
                        except APIKeyError: # Key gagal
                            log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name_for_log}. Mencoba key berikutnya.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            
                            if not api_key_manager.switch_to_next_key(): # Coba ganti key global
                                log_error(f"Tidak ada lagi API key tersedia (global) setelah kegagalan pada {pair_name_for_log}.", pair_name=pair_name_for_log)
                                break # Hentikan loop retry untuk pair ini
                            retries_done_for_this_pair_update += 1 

                        except requests.exceptions.RequestException as e_req: # Error jaringan
                            log_error(f"Error jaringan saat mengambil update {pair_name_for_log}: {e_req}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            break # Hentikan loop retry untuk pair ini
                        except Exception as e_gen_update: # Error lain
                            log_error(f"Error umum saat mengambil update {pair_name_for_log}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            log_exception("Traceback Error Update Fetch:", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            break # Hentikan loop retry untuk pair ini
                
                # Jika setelah semua retry masih gagal untuk pair ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 : # Ditambah 1 karena index 0
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() # Catat waktu untuk cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name_for_log} di siklus ini. Akan masuk cooldown.", pair_name=pair_name_for_log)

                # Proses data baru jika fetch berhasil
                if not fetch_update_successful_for_this_pair or not new_candles_batch_raw:
                    if fetch_update_successful_for_this_pair and not new_candles_batch_raw and not data_per_pair["big_data_collection_phase_active"]:
                        log_info(f"Tidak ada data candle baru diterima untuk {pair_name_for_log} (fetch berhasil tapi batch kosong).", pair_name=pair_name_for_log)
                    elif not fetch_update_successful_for_this_pair: # Gagal fetch setelah semua upaya
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair) # Update shared manager
                    continue # Lanjut ke pair berikutnya

                # Gabungkan candle baru dengan yang lama, hindari duplikasi, dan urutkan
                merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0

                for candle in new_candles_batch_raw: # Candle baru dari API (sudah ada field 'ema': None)
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: # Candle benar-benar baru
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts]['close'] != candle['close'] or \
                         merged_candles_dict[ts]['open'] != candle['open'] or \
                         merged_candles_dict[ts]['high'] != candle['high'] or \
                         merged_candles_dict[ts]['low'] != candle['low']: # Candle lama tapi datanya berubah (misal, candle saat ini)
                        merged_candles_dict[ts] = candle # Update dengan data terbaru
                        updated_count_this_batch +=1
                
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                data_per_pair["all_candles_list"] = all_candles_list_temp # Update list candle utama

                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate untuk {pair_name_for_log}. Total: {len(data_per_pair['all_candles_list'])}.", pair_name=pair_name_for_log)
                elif new_candles_batch_raw : # Fetch berhasil tapi tidak ada yg baru/berubah
                     log_info(f"Tidak ada candle dengan timestamp baru atau konten berbeda untuk {pair_name_for_log}. Data terakhir mungkin identik.", pair_name=pair_name_for_log)

                # Hitung ulang EMA untuk seluruh set data setelah penggabungan
                # Ini penting agar EMA konsisten
                if actual_new_or_updated_count > 0 or not data_per_pair.get("_ema_calculated_at_least_once", False):
                    log_debug(f"Menghitung ulang EMA untuk {len(data_per_pair['all_candles_list'])} candle pada {pair_name_for_log}...", pair_name=pair_name_for_log)
                    data_per_pair["all_candles_list"] = calculate_emas(data_per_pair["all_candles_list"], config_for_pair.get('ema_length', 200))
                    data_per_pair["_ema_calculated_at_least_once"] = True


                # Cek lagi apakah BIG DATA tercapai setelah update
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Pangkas jika berlebih
                            data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data_per_pair["big_data_email_sent"]:
                            send_email_notification(
                                f"Data Downloading Complete: {pair_name_for_log}", 
                                f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name_for_log}.", 
                                {**config_for_pair, 'pair_name': pair_name_for_log}
                            )
                            data_per_pair["big_data_email_sent"] = True
                        
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) # Kurangi counter
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data_per_pair['all_candles_list'])} candles) untuk {pair_name_for_log} ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                else: # Jika sudah live, pastikan tidak melebihi TARGET_BIG_DATA_CANDLES terlalu banyak
                    if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 50: # Beri sedikit buffer
                        data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 50):]

                # Jalankan logika strategi jika ada data baru/update atau baru masuk fase live
                min_len_for_logic_run = config_for_pair.get('ema_length',200) + 2 # ema_length + current + previous
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run:
                    # Jalankan jika ada candle baru/update ATAU jika baru selesai big data phase
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data_per_pair["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or # Baru selesai big data
                                         (data_per_pair["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) ) # Masih big data tapi ada tambahan

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi untuk {pair_name_for_log} dengan {len(data_per_pair['all_candles_list'])} candle...", pair_name=pair_name_for_log)
                         # Kirim slice data yang relevan jika terlalu panjang, misal 2x EMA length + buffer
                         # Slice untuk logic, bukan untuk `all_candles_list`
                         slice_for_logic = data_per_pair["all_candles_list"][- (config_for_pair.get('ema_length', 200) * 3) : ] # Ambil 3x EMA length untuk jaga2, atau seluruhnya jika < itu

                         data_per_pair["strategy_state"] = run_strategy_logic(
                             slice_for_logic, # Gunakan slice untuk efisiensi jika perlu
                             config_for_pair, 
                             data_per_pair["strategy_state"],
                             global_settings_dict
                         )
                    elif not data_per_pair["big_data_collection_phase_active"]: # Live tapi tidak ada update
                         last_c_time_str = data_per_pair["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data_per_pair["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name_for_log}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name_for_log)
                else:
                    log_info(f"Data ({len(data_per_pair['all_candles_list'])}) untuk {pair_name_for_log} belum cukup utk analisa EMA logic (min: {min_len_for_logic_run}).", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
            
                # Update shared data manager untuk UI chart
                with lock_ref:
                    shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair) # Update dengan semua data terbaru

            # Tentukan durasi tidur sebelum siklus berikutnya
            sleep_duration = 15 # Default sleep jika tidak ada interval lain

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None:
                # Jika semua key gagal secara GLOBAL dan tidak ada data berhasil di-fetch di siklus ini
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch di siklus ini. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600 # Tidur 1 jam
            elif active_cryptos_still_in_big_data_collection > 0:
                # Jika masih ada yg kumpulkan big data, gunakan interval yg lebih pendek
                min_big_data_interval_for_sleep = float('inf')
                for _pid_loop, pdata_loop_item in local_crypto_data_manager.items():
                    if pdata_loop_item["big_data_collection_phase_active"]:
                        pconfig_loop = pdata_loop_item["config"]
                        interval_bd_sleep = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_loop.get('timeframe') == "day" else 3580)
                        min_big_data_interval_for_sleep = min(min_big_data_interval_for_sleep, interval_bd_sleep)
                
                sleep_duration = min(min_big_data_interval_for_sleep if min_big_data_interval_for_sleep != float('inf') else 30, 30) # Max 30s jika masih big data
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua sudah live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    # Sesuaikan sleep_duration dengan MIN_REFRESH_INTERVAL_AFTER_BIG_DATA
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya (min_next_refresh: {min_overall_next_refresh_seconds:.0f}s).", pair_name="SYSTEM")
                else: # Fallback jika tidak ada interval terhitung
                    default_refresh_from_first_config = 60
                    if all_crypto_configs : # Ambil dari config pair pertama
                        default_refresh_from_first_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_first_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama setelah live).", pair_name="SYSTEM")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s hingga siklus berikutnya...")
            else: # Pengaman jika sleep_duration 0 atau negatif
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM")
                time.sleep(1)

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna (Ctrl+C).{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01, new_line=True)
    except Exception as e_main_loop:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error Loop Trading Utama:", pair_name="SYSTEM")
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005, new_line=True)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings() # Load settings saat aplikasi pertama kali jalan

    # Jalankan Flask server di thread terpisah
    flask_server_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
    flask_server_thread.start()

    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Chart) =========", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("--- Current Strategy: EMA GR1/GR2 ---", color=AnsiColors.CYAN, delay=0.005)


        pick_title_main = ""
        active_configs_list = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs_list:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs_list)}) ---\n"
            for i, cfg_item in enumerate(active_configs_list):
                pick_title_main += f"  {i+1}. {cfg_item.get('symbol','N/A')}-{cfg_item.get('currency','N/A')} (TF: {cfg_item.get('timeframe','N/A')}, Exch: {cfg_item.get('exchange','N/A')}, EMA: {cfg_item.get('ema_length','N/A')})\n"
        else:
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s_main = settings.get("api_settings", {})
        primary_key_display_main = api_s_main.get('primary_key', 'BELUM DIATUR')
        if primary_key_display_main and len(primary_key_display_main) > 10 and primary_key_display_main not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display_main = "..." + primary_key_display_main[-5:] # Tampilkan bagian belakang saja
        num_recovery_keys_main = len([k for k in api_s_main.get('recovery_keys',[]) if k])
        termux_notif_main_status = "Aktif" if api_s_main.get("enable_termux_notifications", False) else "Nonaktif"

        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display_main} | Recovery Keys: {num_recovery_keys_main}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status}\n"
        pick_title_main += f"Chart Server: http://localhost:5001 (atau IP Termux-mu)\n"
        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"

        main_menu_options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]
        
        selected_main_option_text = None
        selected_main_index = -1

        try:
            selected_main_option_text, selected_main_index = pick(main_menu_options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception as e_pick_main: # Fallback jika 'pick' gagal (misal, environment non-interaktif)
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main)
            for idx_main, opt_text_main in enumerate(main_menu_options_plain):
                print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main_input = input("Pilih nomor opsi: ").strip()
                if not choice_main_input: continue # Jika hanya Enter, loop lagi
                choice_main_val = int(choice_main_input) -1 # Konversi ke index 0-based
                if 0 <= choice_main_val < len(main_menu_options_plain):
                    selected_main_index = choice_main_val
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue
        
        if selected_main_index == 0: # Mulai Trading
            # Pastikan shared_crypto_data_manager di-clear atau di-reset sebelum memulai sesi baru
            # agar tidak ada data sisa dari sesi sebelumnya jika skrip tidak di-restart total.
            with shared_data_lock:
                shared_crypto_data_manager.clear()
            start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif selected_main_index == 1: # Pengaturan
            settings = settings_menu(settings) # settings_menu akan return settings yg mungkin sudah diubah
            # Tidak perlu clear shared_crypto_data_manager di sini, karena start_trading yg akan melakukannya
        elif selected_main_index == 2: # Keluar
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA, new_line=True)
            show_spinner(0.5, "Exiting")
            break # Keluar dari loop while True


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa dari level utama (Ctrl+C). Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01, new_line=True)
    except Exception as e_global:
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}TERJADI ERROR KRITIKAL DI LEVEL UTAMA:{AnsiColors.ENDC}")
        print(f"{AnsiColors.RED}{str(e_global)}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        print(f"\n{AnsiColors.ORANGE}Detail error telah dicatat di: {log_file_name}{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        # Pastikan semua output ter-flush sebelum exit, terutama di Termux
        sys.stdout.flush()
        sys.stderr.flush()
