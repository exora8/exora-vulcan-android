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

# --- ANSI COLOR CODES --- (Tetap sama)
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

# --- ANIMATION HELPER FUNCTIONS --- (Tetap sama)
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True):
    for char in text:
        sys.stdout.write(color + char + AnsiColors.ENDC if color else char)
        sys.stdout.flush()
        time.sleep(delay)
    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing..."):
    spinner_chars = ['-', '\\', '|', '/'] # CHART_MOD: Spinner lebih jelas
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

# --- CUSTOM EXCEPTION --- (Tetap sama)
class APIKeyError(Exception):
    pass

# --- KONFIGURASI LOGGING --- (Tetap sama, mungkin perlu sedikit penyesuaian format jika ada error dari Flask)
logger = logging.getLogger()
logger.setLevel(logging.INFO) # CHART_MOD: Mungkin ingin INFO atau DEBUG untuk Flask
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler
log_file_name = "trading_log.txt"
fh = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# Console Handler
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
            record.pair_name = 'SYSTEM' # Default pair_name
        return True
logger.addFilter(AddPairNameFilter())

def log_info(message, pair_name="SYSTEM"): logger.info(message, extra={'pair_name': pair_name})
def log_warning(message, pair_name="SYSTEM"): logger.warning(message, extra={'pair_name': pair_name})
def log_error(message, pair_name="SYSTEM"): logger.error(message, extra={'pair_name': pair_name})
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name})
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})


SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999 # CryptoCompare API limit per request
TARGET_BIG_DATA_CANDLES = 2500 # Jumlah candle yang diinginkan
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 # Detik, interval minimal setelah 2500 candle terkumpul

# --- FUNGSI CLEAR SCREEN --- (Tetap sama)
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER --- (Tetap sama)
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None): # CHART_MOD: init method
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

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION --- (Tetap sama)
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a', end='', flush=True) # Standard bell for Linux/macOS/Termux
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL')) # CHART_MOD

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
    """Mengirim notifikasi menggunakan termux-notification jika diaktifkan."""
    api_settings = global_settings.get("api_settings", {}) # CHART_MOD: Defensive get
    if not api_settings.get("enable_termux_notifications", False):
        return

    try:
        # Menggunakan timeout untuk termux-notification agar tidak hang jika ada masalah
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False, # Jangan error jika command gagal, log saja
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10) # CHART_MOD: timeout
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal dan paket termux-api sudah diinstal di Termux (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired: # CHART_MOD
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN --- (Umumnya tetap sama, mungkin ada sedikit penyesuaian UI jika diperlukan)
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60, # Default setelah big data
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
        "enable_termux_notifications": False
    }
    if os.path.exists(SETTINGS_FILE):
        try: # CHART_MOD: Try-except block
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Pastikan semua kunci default ada
            if "api_settings" not in settings:
                settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]:
                        settings["api_settings"][k] = v

            if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                settings["cryptos"] = []
            for crypto_cfg in settings["cryptos"]: # Pastikan ID dan enabled ada
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
            return settings
        except json.JSONDecodeError:
            log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
            # Buat struktur default jika file korup
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
        except Exception as e: # CHART_MOD: General exception
            log_error(f"Error tidak diketahui saat load_settings: {e}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    try: # CHART_MOD: Try-except block
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

# _prompt_crypto_config dan settings_menu tetap sama seperti yang kamu berikan
# ... (kode _prompt_crypto_config dan settings_menu dari pertanyaan, tidak diubah) ...
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

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
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
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"),
            ("option", "Ubah Konfigurasi Crypto"),
            ("option", "Hapus Konfigurasi Crypto"),
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")
        ]

        selectable_options = [text for type, text in original_options_structure if type == "option"]
        
        selected_option_text = None # CHART_MOD
        action_choice = -1 # CHART_MOD

        try:
            selected_option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick:
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings) # Tampilkan title lagi karena pick gagal
            for idx, opt_text in enumerate(selectable_options):
                print(f"  {idx + 1}. {opt_text}")
            try:
                choice_input = input("Pilih nomor opsi: ").strip()
                if not choice_input: continue # Kembali jika input kosong
                choice = int(choice_input) -1
                if 0 <= choice < len(selectable_options):
                    action_choice = choice
                    selected_option_text = selectable_options[choice] # CHART_MOD
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue
        
        # Konversi selected_option_text ke action_choice jika pick berhasil (pick mengembalikan teks, bukan indeks asli)
        if selected_option_text is not None and action_choice < 0: # action_choice dari pick adalah index di selectable_options
             pass # action_choice sudah benar dari pick
        elif action_choice < 0 and selected_option_text is None: # Jika gagal input manual juga
            continue


        # Pemetaan dari teks opsi ke action_choice jika menggunakan input manual dan selected_option_text diisi
        # Atau jika pick() mengembalikan teks dan kita butuh indeks yang konsisten dengan struktur lama.
        # Namun, karena pick() mengembalikan indeks dari selectable_options, ini seharusnya sudah OK.
        # Cukup pastikan action_choice adalah indeks yang valid.
        # Indeks dari pick sudah benar untuk `if action_choice == ...`

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
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k] # Filter out empty strings
                    api_s['recovery_keys'] = current_recovery # Update dengan list yang bersih

                    if not current_recovery:
                        recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"

                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]
                    
                    rec_selected_text = None # CHART_MOD
                    rec_index = -1 # CHART_MOD
                    try:
                        rec_selected_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e_pick_rec:
                        log_error(f"Error dengan library 'pick' di menu recovery: {e_pick_rec}. Gunakan input manual.")
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
                    
                    # Jika pick berhasil, rec_index adalah index dari recovery_options_plain
                    # Jika input manual, rec_index juga sudah diset dengan benar.

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
                            else: # Seharusnya tidak terjadi jika pick digunakan dengan benar
                                print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}")
                        except Exception as e_pick_del: # Termasuk jika pick dibatalkan dengan Ctrl+C
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
                else: print(f"{AnsiColors.RED}Pilihan ubah tidak valid.{AnsiColors.ENDC}") # Seharusnya tidak terjadi
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
                else: print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}") # Seharusnya tidak terjadi
                show_spinner(1, "Kembali...")
            elif action_choice == 7: # Kembali ke Menu Utama
                break
        except ValueError: # Misal dari input manual untuk nomor jika pick gagal
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e: # Kesalahan umum dalam blok aksi
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:") # CHART_MOD: Log traceback
            show_spinner(1.5, "Error, kembali...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA --- (Sedikit modifikasi untuk logging dan penanganan error)
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None # Untuk pagination mundur
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10 # Anggap lebih dari 10 adalah "besar" untuk logging

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)

    # Progress bar hanya jika lebih dari max limit per call
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT :
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)

        # Jika mem-paginate mundur dan butuh > 1 candle, minta satu ekstra untuk overlap check (jika perlu)
        # CryptoCompare API v2 mengembalikan data hingga toTs, jadi tidak perlu +1 untuk overlap.
        # Cukup ambil limit yang dibutuhkan.
        # if current_to_ts is not None and candles_still_needed > 1 :
        #     limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT) # Ambil limit yang pas

        if limit_for_this_api_call <= 0: break # Sudah cukup atau tidak butuh lagi

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call, # API mengembalikan `limit` candle SEBELUM `toTs`
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # CHART_MOD: log detail
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: {key_display}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20) # Timeout 20 detik

            if response.status_code in [401, 403, 429]: # Kemungkinan error API Key
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}") # Raise untuk ditangkap oleh pemanggil

            response.raise_for_status() # Untuk error HTTP lain (5xx, 404, dll)
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                # Cek apakah error terkait API key
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit", # Sudah ada
                    "please pass an API key", "api_key not found"
                ]
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}") # Raise untuk ditangkap
                else:
                    # Error lain dari CryptoCompare, mungkin data tidak ada
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Hentikan loop fetch untuk pair ini

            # Cek data yang valid
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break # Tidak ada data lagi

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: # API mengembalikan list kosong
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                # Pastikan semua field ada, jika tidak, skip candle ini tapi log
                required_ohlcv_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in required_ohlcv_keys):
                    log_warning(f"Candle tidak lengkap dari API @ ts {item.get('time', 'N/A')}. Dilewati. Data: {item}", pair_name=pair_name)
                    continue

                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom') # 'volumefrom' adalah volume base currency
                }
                batch_candles_list.append(candle)

            # Hapus overlap jika ada. CryptoCompare mengembalikan data *sebelum* toTs.
            # Jika kita mengambil data secara berurutan ke masa lalu, candle pertama dari batch baru
            # (yang merupakan candle tertua di batch itu) adalah toTs dari request sebelumnya.
            # Data dari API sudah diurutkan dari yang tertua ke terbaru dalam satu batch.
            # Jadi, batch_candles_list[0] adalah yang tertua dalam batch.
            # all_accumulated_candles di-prepend, jadi all_accumulated_candles[0] adalah yang terbaru dari akumulasi sebelumnya.

            # Logika overlap: jika ada data terakumulasi dan batch baru,
            # dan timestamp candle *terbaru* di batch baru sama dengan timestamp *tertua* dari data yang sudah ada,
            # maka itu overlap. Tapi karena kita prepend, candle tertua di batch baru adalah batch_candles_list[0]
            # dan candle terbaru di akumulasi adalah all_accumulated_candles[0] (jika ada).
            # Kita prepend: all_accumulated_candles = batch_candles_list + all_accumulated_candles
            # Jadi, candle terakhir dari batch_candles_list (batch_candles_list[-1]) bisa jadi sama dengan
            # candle pertama dari all_accumulated_candles (all_accumulated_candles[0]) jika ada cycle data aneh.
            # CryptoCompare biasanya OK, jadi ini mungkin tidak terlalu sering terjadi.
            # Namun, untuk fetch_candles yang mengambil candle terbaru (bukan historis),
            # kita perlu hati-hati agar tidak duplikat candle saat ini.
            # Skrip ini menggunakan fetch_candles untuk historis (mundur) dan update terbaru (maju).
            # Logika ini adalah untuk pagination mundur (mengisi data historis).
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                # Jika candle tertua dari batch baru sama dengan candle terbaru dari akumulasi sebelumnya, itu aneh.
                # Yang lebih mungkin: candle terbaru dari batch baru (batch_candles_list[-1])
                # sama dengan candle tertua dari akumulasi sebelumnya (all_accumulated_candles[0]) jika terjadi
                # permintaan yang tumpang tindih.
                # Karena kita prepend, kita harus cek all_accumulated_candles[0] (yang paling baru dari sebelumnya)
                # dengan batch_candles_list[-1] (yang paling baru dari batch saat ini).
                # Seharusnya tidak terjadi jika toTs digunakan dengan benar.
                # Untuk data historis yang di-fetch mundur: `all_accumulated_candles` diisi dengan `batch_list + all_accumulated_candles`.
                # `toTs` adalah timestamp dari candle *pertama* (`[0]`) dari batch sebelumnya.
                # Jadi `batch_candles_list` yang baru seharusnya berakhir *sebelum* `toTs` tersebut.
                # Jika `batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']`, maka ada duplikasi.
                # Hapus dari `batch_candles_list` sebelum digabung.
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                     if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih (overlap): {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                     batch_candles_list.pop()


            if not batch_candles_list and current_to_ts is not None : # Batch jadi kosong setelah overlap removal
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data atau hanya ada 1 candle overlap.", pair_name=pair_name)
                # Jika ini terjadi, dan kita butuh banyak data, berarti ada masalah.
                # Jika hanya butuh sedikit (misal 1-2 candle update), ini bisa jadi normal.
                # Untuk large fetch historis, ini berarti akhir data.
                break # Berhenti jika batch kosong, tidak ada data baru

            all_accumulated_candles = batch_candles_list + all_accumulated_candles # Prepend

            if raw_candles_from_api: # Set toTs untuk request berikutnya
                current_to_ts = raw_candles_from_api[0]['time'] # Waktu dari candle tertua di batch saat ini
            else: # Seharusnya tidak sampai sini jika raw_candles_from_api kosong karena sudah break di atas
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): # Update progress bar
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

            # Jika API mengembalikan kurang dari yang diminta, berarti akhir data
            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break # Akhir data tercapai

            if len(all_accumulated_candles) >= total_limit_desired: break # Target tercapai

            # Delay kecil antar batch besar untuk tidak membebani API
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3) # Jeda antar request batch besar

        except APIKeyError: # Dilempar dari dalam try block
            raise # Biarkan pemanggil yang menangani penggantian key
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            # Tidak raise APIKeyError, karena ini masalah jaringan, bukan key.
            # Pemanggil mungkin mau retry dengan key yang sama.
            # Untuk fetch_candles, kita break saja loop fetch batchnya.
            break
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles (batch loop): {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error Fetch Candles (batch loop):", pair_name=pair_name) # CHART_MOD
            break # Hentikan loop fetch untuk pair ini

    # Pastikan jumlah candle tidak melebihi yang diinginkan (jika ada overlap atau API memberi lebih)
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:] # Ambil N terakhir

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Final progress bar update
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 1 for high, -1 for low
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        # CHART_MOD: Tambahkan info timestamp untuk pivot yang akan di-plot
        "last_pivot_high_display_info": None, # Berisi {'price': X, 'timestamp_ms': Y}
        "last_pivot_low_display_info": None,  # Berisi {'price': X, 'timestamp_ms': Y}

        "high_price_for_fib": None, # Harga tertinggi dari Pivot High sebelumnya untuk kalkulasi FIB
        "high_bar_index_for_fib": None, # Index bar dari Pivot High di atas

        "active_fib_level": None, # Level FIB 0.5 yang sedang aktif
        "active_fib_line_start_index": None, # Index bar dari Pivot Low yang membentuk FIB ini

        "entry_price_custom": None, # Harga entry kustom
        "highest_price_for_trailing": None, # Harga tertinggi sejak entry untuk trailing TP
        "trailing_tp_active_custom": False, # Apakah trailing TP sudah aktif
        "current_trailing_stop_level": None, # Level trailing stop loss saat ini
        "emergency_sl_level_custom": None, # Level emergency stop loss kustom

        "position_size": 0, # 0 = no position, 1 = long position (atau jumlah kontrak)
        # CHART_MOD: Untuk plotting historis, Anda mungkin butuh list of trades, bukan hanya state saat ini.
        # "trade_history": [] # contoh: [{'type': 'buy', 'price': X, 'time': Y, 'sl': Z}, {'type':'sell', ...}]
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1:
        return pivots # Tidak cukup data

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue # Skip jika data harga tidak ada

        # Cek ke kiri
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break
            if is_high: # Pivot High
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # Pivot Low
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue

        # Cek ke kanan
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break
            if is_high: # Pivot High
                if series_list[i] < series_list[i+j]: is_pivot = False; break # Harus lebih tinggi dari N bar ke kanan
            else: # Pivot Low
                if series_list[i] > series_list[i+j]: is_pivot = False; break # Harus lebih rendah dari N bar ke kanan
        
        if is_pivot:
            pivots[i] = series_list[i]
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings): # CHART_MOD: global_settings untuk notif
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}" # Nama pair untuk logging

    # Reset pivot confirmed flags di awal setiap run, karena kita re-evaluasi berdasarkan data saat ini
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None
    # strategy_state["last_pivot_high_display_info"] = None # Jangan reset ini agar chart bisa lihat yang terakhir
    # strategy_state["last_pivot_low_display_info"] = None  # Cukup di-overwrite jika ada yang baru

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]): # Cek candle pertama saja
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state

    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]

    # Cari semua pivot high dan low potensial di seluruh histori yang diberikan
    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    # Pivot dianggap "terkonfirmasi" setelah `right_strength` candle berlalu sejak titik pivot.
    # Jadi, kita lihat `right_strength` candle ke belakang dari candle saat ini.
    current_bar_index_in_list = len(candles_history) - 1 # Index candle terbaru
    if current_bar_index_in_list < 0 : return strategy_state # Tidak ada candle

    # Index di mana event pivot (jika ada) akan terkonfirmasi SEKARANG
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength

    # Cek apakah ada Pivot High yang terkonfirmasi pada `idx_pivot_event_high`
    raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high] if 0 <= idx_pivot_event_high < len(raw_pivot_highs) else None
    raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low] if 0 <= idx_pivot_event_low < len(raw_pivot_lows) else None

    # --- Logika Pivot High Terkonfirmasi ---
    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1: # Ada PH baru & sinyal sebelumnya bukan PH
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["last_signal_type"] = 1 # Set sinyal terakhir ke Pivot High
        pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)
        
        # CHART_MOD: Simpan info untuk plotting
        strategy_state["last_pivot_high_display_info"] = {
            'price': strategy_state['final_pivot_high_price_confirmed'],
            'timestamp_ms': pivot_timestamp.timestamp() * 1000
        }

        # Jika PH terkonfirmasi, ini menjadi kandidat High untuk FIB berikutnya
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high

        # Jika ada FIB aktif sebelumnya, PH baru ini me-resetnya (karena struktur pasar berubah)
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None
            # Juga reset state trading jika FIB di-reset dan belum entry? Tergantung aturan.
            # Untuk saat ini, hanya reset FIB-nya.

    # --- Logika Pivot Low Terkonfirmasi & Kalkulasi FIB ---
    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1: # Ada PL baru & sinyal sebelumnya bukan PL
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1 # Set sinyal terakhir ke Pivot Low
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

        # CHART_MOD: Simpan info untuk plotting
        strategy_state["last_pivot_low_display_info"] = {
            'price': strategy_state['final_pivot_low_price_confirmed'],
            'timestamp_ms': pivot_timestamp.timestamp() * 1000
        }

        # Jika ada Pivot High valid sebelumnya dan Pivot Low ini muncul SETELAHNYA, hitung FIB
        if strategy_state["high_price_for_fib"] is not None and \
           strategy_state["high_bar_index_for_fib"] is not None and \
           idx_pivot_event_low > strategy_state["high_bar_index_for_fib"]: # PL harus setelah PH

            current_low_price_for_fib = strategy_state["final_pivot_low_price_confirmed"]
            
            # Pastikan high_price_for_fib dan current_low_price_for_fib tidak None
            if strategy_state["high_price_for_fib"] is None or current_low_price_for_fib is None:
                 log_warning("Harga untuk kalkulasi FIB tidak valid (None). Tidak bisa menghitung FIB.", pair_name=pair_name)
            else:
                calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price_for_fib) / 2.0

                # Cek Secure FIB jika diaktifkan
                is_fib_late = False
                current_candle_for_fib_check = candles_history[current_bar_index_in_list] # Candle saat ini
                if crypto_config["enable_secure_fib"]:
                    price_key_for_secure_fib = crypto_config["secure_fib_check_price"].lower() # 'close' atau 'high'
                    price_val_current_candle = current_candle_for_fib_check.get(price_key_for_secure_fib)

                    if price_val_current_candle is None:
                        log_warning(f"Harga '{price_key_for_secure_fib}' tidak ada di candle saat ini untuk Secure FIB. Menggunakan close.", pair_name=pair_name)
                        price_val_current_candle = current_candle_for_fib_check.get('close')

                    if price_val_current_candle is not None and calculated_fib_level is not None and \
                       price_val_current_candle > calculated_fib_level:
                        is_fib_late = True

                if is_fib_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state["active_fib_level"] = None 
                    strategy_state["active_fib_line_start_index"] = None
                elif calculated_fib_level is not None : 
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.5f}, L: {current_low_price_for_fib:.5f})", pair_name=pair_name)
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = idx_pivot_event_low # Index PL yang membentuk FIB ini

            # Reset high_price_for_fib setelah digunakan (atau diabaikan karena telat)
            # agar menunggu Pivot High baru berikutnya.
            strategy_state["high_price_for_fib"] = None
            strategy_state["high_bar_index_for_fib"] = None
    
    # --- Logika Entry & Manajemen Posisi ---
    current_candle = candles_history[current_bar_index_in_list] # Candle terbaru yang lengkap
    
    # Validasi data candle saat ini
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi trading.", pair_name=pair_name)
        return strategy_state

    # Cek Entry jika ada FIB Aktif dan belum ada posisi
    if strategy_state["active_fib_level"] is not None and \
       strategy_state["active_fib_line_start_index"] is not None and \
       strategy_state["position_size"] == 0: # Hanya jika belum ada posisi

        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            strategy_state["position_size"] = 1 # Atau jumlah lot/kontrak sesuai aturan
            entry_px = current_candle['close'] # Entry di close candle sinyal
            strategy_state["entry_price_custom"] = entry_px
            strategy_state["highest_price_for_trailing"] = entry_px # Inisialisasi harga tertinggi
            strategy_state["trailing_tp_active_custom"] = False # Trailing TP belum aktif
            strategy_state["current_trailing_stop_level"] = None # Belum ada trailing SL

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
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name}) # CHART_MOD: pass pair_name

            # Reset FIB setelah entry agar tidak re-entry di FIB yang sama
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    # Manajemen Posisi Aktif (jika ada)
    if strategy_state["position_size"] > 0:
        # Update harga tertinggi untuk trailing stop
        current_high_for_trailing_update = strategy_state.get("highest_price_for_trailing", current_candle.get('high')) # Default ke entry jika baru masuk
        if current_high_for_trailing_update is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None) saat update.", pair_name=pair_name)
        else:
            strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing_update , current_candle['high'])

        # Aktivasi Trailing TP
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: # Hindari ZeroDivisionError
                profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None: # Harga tertinggi belum diset
                profit_percent = 0.0
                log_warning("highest_price_for_trailing is None saat kalkulasi profit untuk aktivasi Trailing TP.", pair_name=pair_name)
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High Sejak Entry: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        # Update Trailing Stop Level jika Trailing TP aktif
        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        # Cek Exit (Emergency SL atau Trailing SL)
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            # Jika trailing SL lebih tinggi dari emergency SL, gunakan trailing SL
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE # Biru untuk Trailing TP (bisa jadi profit)

        # Jika low candle saat ini menyentuh atau melewati level stop
        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price_open_candle = current_candle.get('open')
            if exit_price_open_candle is None:
                log_warning("Harga open candle tidak ada untuk kalkulasi exit. Menggunakan SL sebagai harga exit.", pair_name=pair_name)
                exit_price = final_stop_for_exit # Harga exit adalah level SL itu sendiri
            else:
                # Exit di open candle berikutnya (disimulasikan sebagai open candle ini jika SL tembus),
                # atau di level SL jika open lebih buruk dari SL (slippage)
                exit_price = min(exit_price_open_candle, final_stop_for_exit)

            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if exit_comment == "Trailing Stop" and pnl < 0: # Jika trailing stop tapi rugi
                exit_color = AnsiColors.RED # Merah juga

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
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name}) # CHART_MOD

            # Reset state posisi
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            # Sinyal terakhir direset agar bisa deteksi pivot baru setelah exit
            # strategy_state["last_signal_type"] = 0 # Atau biarkan agar tidak langsung re-entry jika kondisi masih sama

        elif strategy_state["position_size"] > 0 : # Jika masih ada posisi dan tidak exit
            plot_stop_level = strategy_state.get("emergency_sl_level_custom")
            stop_type_info = "Emergency SL"
            if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
                # Pilih SL yang lebih relevan (lebih tinggi/protektif) untuk display
                current_trailing_sl = strategy_state.get("current_trailing_stop_level")
                if plot_stop_level is None or (current_trailing_sl is not None and current_trailing_sl > plot_stop_level):
                    plot_stop_level = current_trailing_sl
                    stop_type_info = "Trailing SL"

            entry_price_display = strategy_state.get('entry_price_custom', 0)
            sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
            log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)

    return strategy_state

# CHART_INTEGRATION_START
# Global data store untuk Flask dan lock-nya
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

# Fungsi untuk mempersiapkan data untuk chart ApexCharts
def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot):
    if pair_id_to_display not in current_data_manager_snapshot:
        log_warning(f"Data untuk pair {pair_id_to_display} tidak ditemukan di snapshot untuk chart.", pair_name="SYSTEM_CHART")
        return None

    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]
    candles_full_history = pair_specific_data.get("all_candles_list", [])
    current_strategy_state = pair_specific_data.get("strategy_state", {})
    pair_config = pair_specific_data.get("config", {})

    # Ambil 2500 candle terbaru untuk chart
    candles_for_chart_display = candles_full_history[-TARGET_BIG_DATA_CANDLES:]

    ohlc_data_points = []
    if not candles_for_chart_display:
        log_warning(f"Tidak ada candle di `candles_for_chart_display` untuk {pair_id_to_display}.", pair_name="SYSTEM_CHART")
        return {"ohlc": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None}


    for candle in candles_for_chart_display:
        required_candle_keys = ['timestamp', 'open', 'high', 'low', 'close']
        if all(k in candle and candle[k] is not None for k in required_candle_keys):
            ohlc_data_points.append({
                'x': candle['timestamp'].timestamp() * 1000, # ApexCharts butuh timestamp milidetik
                'y': [candle['open'], candle['high'], candle['low'], candle['close']]
            })
        else:
            log_debug(f"Skipping incomplete candle for chart: {candle.get('timestamp')}", pair_name="SYSTEM_CHART")


    chart_annotations_yaxis = [] # Untuk garis horizontal (misal FIB, SL)
    chart_annotations_points = [] # Untuk titik spesifik (misal Pivot, Entry)

    # Anotasi untuk Active FIB Level
    active_fib_val = current_strategy_state.get("active_fib_level")
    if active_fib_val and current_strategy_state.get("active_fib_line_start_index") is not None:
        if ohlc_data_points: # Hanya jika ada data untuk dirender
            chart_annotations_yaxis.append({
                'y': active_fib_val,
                'borderColor': '#00E396', # Hijau
                'label': {
                    'borderColor': '#00E396',
                    'style': {'color': '#fff', 'background': '#00E396', 'fontSize': '10px', 'padding': {'left': '3px', 'right': '3px', 'top':'1px', 'bottom':'1px'}},
                    'text': f'FIB 0.5: {active_fib_val:.5f}'
                }
            })
    
    # Anotasi untuk Posisi Aktif (Entry & SL)
    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") is not None:
        entry_price_val = current_strategy_state.get("entry_price_custom")
        
        # Untuk titik entry, idealnya kita punya timestamp entry.
        # Karena tidak disimpan, kita tandai saja di candle terakhir jika posisi masih aktif.
        # Atau, kita bisa juga buat garis horizontal untuk entry price.
        if ohlc_data_points:
             chart_annotations_yaxis.append({ # Garis horizontal untuk entry
                'y': entry_price_val,
                'borderColor': '#2698FF', # Biru
                'strokeDashArray': 4, # Garis putus-putus
                'label': {
                    'borderColor': '#2698FF',
                    'style': {'color': '#fff', 'background': '#2698FF', 'fontSize': '10px', 'padding': {'left': '3px', 'right': '3px', 'top':'1px', 'bottom':'1px'}},
                    'text': f'Entry: {entry_price_val:.5f}'
                }
            })

        # Stop Loss Level
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
                    'style': {'color': '#fff', 'background': '#FF4560', 'fontSize': '10px', 'padding': {'left': '3px', 'right': '3px', 'top':'1px', 'bottom':'1px'}},
                    'text': f'{sl_type_text}: {sl_level_val:.5f}'
                }
            })

    # Anotasi untuk Pivot High/Low Terbaru yang terkonfirmasi (jika ada)
    # Ini akan menampilkan *hanya* pivot terakhir yang disimpan di state.
    # Untuk histori pivot, `run_strategy_logic` perlu diubah untuk menyimpan list pivot.
    first_candle_ts_ms_on_chart = ohlc_data_points[0]['x'] if ohlc_data_points else 0
    
    last_ph_info = current_strategy_state.get("last_pivot_high_display_info")
    if last_ph_info and last_ph_info['timestamp_ms'] >= first_candle_ts_ms_on_chart: # Hanya jika pivot ada di chart saat ini
        chart_annotations_points.append({
            'x': last_ph_info['timestamp_ms'],
            'y': last_ph_info['price'],
            'marker': {'size': 7, 'fillColor': '#FF0000', 'strokeColor': '#FF0000', 'shape': 'triangle', 'radius':0},
            'label': {'borderColor': '#FF0000','offsetY': -18, 'style': {'color': '#fff', 'background': '#FF0000', 'fontSize': '10px'}, 'text': 'PH'}
        })

    last_pl_info = current_strategy_state.get("last_pivot_low_display_info")
    if last_pl_info and last_pl_info['timestamp_ms'] >= first_candle_ts_ms_on_chart:
        chart_annotations_points.append({
            'x': last_pl_info['timestamp_ms'],
            'y': last_pl_info['price'],
            'marker': {'size': 7, 'fillColor': '#00CD00', 'strokeColor': '#00CD00', 'shape': 'triangle', 'radius':0, 'cssClass': 'apexcharts-marker-inverted'}, # Inverted triangle
            'label': {'borderColor': '#00CD00','offsetY': 10, 'style': {'color': '#fff', 'background': '#00CD00', 'fontSize': '10px'}, 'text': 'PL'}
        })

    return {
        "ohlc": ohlc_data_points,
        "annotations_yaxis": chart_annotations_yaxis,
        "annotations_points": chart_annotations_points,
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": candles_for_chart_display[-1]['timestamp'].timestamp() * 1000 if candles_for_chart_display else None
    }

# Flask App Instance
flask_app_instance = Flask(__name__)

# Template HTML untuk chart (disimpan sebagai string agar mudah)
HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crypto Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;}
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width:100%; max-width: 1200px; }
        #controls label { font-size: 0.9em; }
        select, button { padding: 8px 12px; font-size:0.9em; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; }
        button:hover { background-color: #444; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        h1 { text-align: center; color: #00bcd4; margin-top: 0; margin-bottom:15px; font-size:1.5em; }
        #lastUpdatedLabel { font-size: 0.8em; color: #aaa; }
        .apexcharts-tooltip-candlestick { background: #333 !important; color: #fff !important; border: 1px solid #555 !important;}
        .apexcharts-tooltip-candlestick .value { font-weight: bold; }
        .apexcharts-marker-inverted .apexcharts-marker-poly { transform: rotate(180deg); transform-origin: center; } /* For inverted triangle */
    </style>
</head>
<body>
    <h1>Live Strategy Chart</h1>
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

        const initialChartOptions = {
            series: [{ name: 'Candlestick', data: [] }],
            chart: { 
                type: 'candlestick', 
                height: 550, // Adjusted height
                id: 'mainCandlestickChart',
                background: '#2a2a2a', // Chart background
                animations: { enabled: true, easing: 'easeinout', speed: 500, animateGradually: { enabled: false } },
                toolbar: { show: true, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } }
            },
            theme: { mode: 'dark' },
            title: { text: 'Memuat Data Pair...', align: 'left', style: { color: '#e0e0e0', fontSize: '16px'} },
            xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: '#aaa'}, formatter: function (value) { return value.toFixed(5); } } },
            grid: { borderColor: '#444' },
            annotations: { yaxis: [], points: [] }, // Init empty annotations
            tooltip: { theme: 'dark', shared: true, 
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    if (w.globals.seriesCandleO && w.globals.seriesCandleO[seriesIndex] && w.globals.seriesCandleO[seriesIndex][dataPointIndex] !== undefined) {
                        const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
                        const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
                        const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
                        const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
                        return (
                            '<div class="apexcharts-tooltip-candlestick" style="padding:5px 10px;">' +
                            '<div>O: <span class="value">' + o.toFixed(5) + '</span></div>' +
                            '<div>H: <span class="value">' + h.toFixed(5) + '</span></div>' +
                            '<div>L: <span class="value">' + l.toFixed(5) + '</span></div>' +
                            '<div>C: <span class="value">' + c.toFixed(5) + '</span></div>' +
                            '</div>'
                        );
                    } return '';
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
                selectorElement.innerHTML = ''; // Hapus opsi lama
                if (pairs.length > 0) {
                    pairs.forEach(pair => {
                        const optionEl = document.createElement('option');
                        optionEl.value = pair.id;
                        optionEl.textContent = pair.name;
                        selectorElement.appendChild(optionEl);
                    });
                    currentSelectedPairId = selectorElement.value || pairs[0].id; // Set pair pertama sebagai default atau yang terpilih
                    loadChartDataForCurrentPair(); // Muat data untuk pair yang terpilih
                } else {
                     selectorElement.innerHTML = '<option value="">Tidak ada pair aktif</option>';
                     if(activeChart) activeChart.destroy();
                     document.getElementById('chart').innerHTML = '<p style="text-align:center; color:#aaa;">Tidak ada pair aktif yang dikonfigurasi di server.</p>';
                }
            } catch (error) {
                console.error("Error fetching pair list:", error);
                document.getElementById('pairSelector').innerHTML = '<option value="">Error memuat pair</option>';
                if(activeChart) activeChart.destroy();
                document.getElementById('chart').innerHTML = `<p style="text-align:center; color:red;">Error: ${error.message}</p>`;
            }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById('pairSelector').value;
            lastKnownDataTimestamp = null; // Reset timestamp agar data baru selalu diambil
            loadChartDataForCurrentPair();
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId) {
                console.log("Tidak ada pair ID yang dipilih.");
                if(activeChart) activeChart.updateOptions(initialChartOptions); // Reset ke state awal
                return;
            }
            console.log(`Memuat data chart untuk: ${currentSelectedPairId}`);
            try {
                const fetchResponse = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!fetchResponse.ok) {
                     let errorMsg = `Gagal mengambil data chart: ${fetchResponse.status}`;
                     try { const errorData = await fetchResponse.json(); errorMsg = errorData.error || errorMsg; } catch(e){}
                     throw new Error(errorMsg);
                }
                const chartDataPayload = await fetchResponse.json();

                if (chartDataPayload.last_updated_tv && chartDataPayload.last_updated_tv === lastKnownDataTimestamp) {
                    console.log("Data chart tidak berubah, tidak perlu update render.");
                    document.getElementById('lastUpdatedLabel').textContent = `Data terakhir (server): ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                    return;
                }
                lastKnownDataTimestamp = chartDataPayload.last_updated_tv;
                document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data terakhir (server): ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "N/A";

                const newChartOptions = {
                    ...initialChartOptions, // Ambil basis dari initial options
                    title: { ...initialChartOptions.title, text: `${chartDataPayload.pair_name} Candlestick` },
                    series: [{ name: 'Candlestick', data: chartDataPayload.ohlc || [] }],
                    annotations: { 
                        yaxis: chartDataPayload.annotations_yaxis || [], 
                        points: chartDataPayload.annotations_points || [] 
                    }
                };
                
                if (!activeChart) {
                    activeChart = new ApexCharts(document.querySelector("#chart"), newChartOptions);
                    activeChart.render();
                } else {
                    activeChart.updateOptions(newChartOptions);
                }

            } catch (error) {
                console.error("Error loading chart data:", error);
                if (activeChart) { // Hancurkan chart jika error agar tidak tampilkan data lama
                    activeChart.destroy();
                    activeChart = null; 
                }
                 // Buat ulang chart dengan pesan error
                const errorChartOptions = { ...initialChartOptions, title: { ...initialChartOptions.title, text: `Error: ${chartDataPayload.pair_name || currentSelectedPairId}` }, series: [], noData: { text: error.message } };
                activeChart = new ApexCharts(document.querySelector("#chart"), errorChartOptions);
                activeChart.render();
                document.getElementById('lastUpdatedLabel').textContent = "Error update";
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            fetchAvailablePairs(); // Muat daftar pair saat halaman siap
            
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId); // Hapus interval lama jika ada
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible') { // Hanya refresh jika tab terlihat
                    console.log(`Auto-refresh chart untuk ${currentSelectedPairId}`);
                    await loadChartDataForCurrentPair();
                }
            }, 30000); // Refresh setiap 30 detik
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
        # Buat salinan dangkal, karena kita hanya baca konfigurasi pair
        data_manager_view = shared_crypto_data_manager.copy()
    
    active_pairs_info = []
    for pair_identifier, pair_data_item in data_manager_view.items():
        config_item = pair_data_item.get("config", {})
        if config_item.get("enabled", True): # Hanya pair yang aktif
             active_pairs_info.append({
                "id": pair_identifier, 
                "name": config_item.get('pair_name', pair_identifier)
            })
    return jsonify(active_pairs_info)


@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request):
    with shared_data_lock:
        # Salinan dalam (deep copy) untuk `pair_specific_data` penting jika `prepare_chart_data_for_pair`
        # atau sub-fungsinya memodifikasi data. Jika hanya read-only, shallow copy cukup.
        # Mengingat `all_candles_list` bisa besar, `deepcopy` seluruh `shared_crypto_data_manager` bisa lambat.
        # Jadi kita ambil pair spesifik saja dan deepcopy itu.
        if pair_id_from_request not in shared_crypto_data_manager:
             return jsonify({"error": f"Data untuk pair {pair_id_from_request} tidak ditemukan di server."}), 404
        
        # Deepcopy hanya data pair yang diminta untuk thread safety saat memproses
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))

    if not pair_data_snapshot: # Jika setelah get ternyata kosong (meski sudah dicek)
        return jsonify({"error": f"Data untuk pair {pair_id_from_request} tidak ditemukan (snapshot kosong)."}), 404

    # Buat snapshot tiruan dari data manager hanya dengan pair yang diminta
    # agar prepare_chart_data_for_pair tidak perlu diubah signature-nya.
    temp_data_manager_for_prep = {pair_id_from_request: pair_data_snapshot}
    
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_data_manager_for_prep)

    if not prepared_data or not prepared_data.get("ohlc"): # Cek jika OHLC kosong
        log_warning(f"Tidak ada data OHLC yang cukup untuk ditampilkan untuk {pair_id_from_request} setelah diproses.", pair_name="SYSTEM_CHART")
        return jsonify({"error": f"Tidak ada data candle yang cukup untuk ditampilkan untuk {pair_id_from_request} (setelah proses).", "ohlc": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_id_from_request, "last_updated_tv": None}), 200 # Kirim 200 agar JS bisa handle noData
    
    return jsonify(prepared_data)


def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001 (atau IP Termux-mu)", pair_name="SYSTEM_CHART")
    try:
        # use_reloader=False penting saat Flask dijalankan di thread non-main atau produksi
        # debug=False untuk produksi/Termux
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask:
        log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
        log_exception("Traceback Error Flask Server:", pair_name="SYSTEM_CHART")

# CHART_INTEGRATION_END


# --- FUNGSI UTAMA TRADING LOOP ---
# CHART_MOD: Tambahkan argumen shared_data_lock dan shared_crypto_data_manager_ref
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Untuk notifikasi email global saat ganti key
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

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = ("..." + current_key_display_val[-3:]) if len(current_key_display_val) > 8 else current_key_display_val # CHART_MOD: Disingkat
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    # Inisialisasi crypto_data_manager lokal untuk loop trading
    local_crypto_data_manager = {}

    for config in all_crypto_configs:
        # Buat ID unik untuk setiap pair berdasarkan simbol, currency, dan timeframe
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}" # Tambahkan nama pair ke config untuk logging mudah

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        local_crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, # Fase pengumpulan data besar awal
            "big_data_email_sent": False, # Flag email notifikasi data besar
            "last_candle_fetch_time": datetime.min, # Waktu fetch terakhir
            "data_fetch_failed_consecutively": 0 # Counter kegagalan fetch berturut-turut
        }
        
        # CHART_INTEGRATION_START: Update shared manager untuk Flask
        with lock_ref:
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id]) # Deepcopy state awal
        # CHART_INTEGRATION_END

        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        # Loop untuk mengambil data awal, coba semua API key jika perlu
        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break # Tidak ada key lagi

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Jika fetch_candles tidak error, anggap berhasil
            except APIKeyError: # Gagal karena API key (limit, invalid)
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break # Tidak ada key lagi untuk dicoba
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e: # Error jaringan
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key, akan coba lagi nanti jika loop utama mengizinkan.", pair_name=config['pair_name'])
                # Berhenti mencoba untuk pair ini di fase inisialisasi jika jaringan error parah
                break 
            except Exception as e_gen: # Error umum lainnya
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break # Error tidak terduga, hentikan inisialisasi pair ini
        
        if not initial_candles and not initial_fetch_successful : # Jika semua upaya gagal
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Anggap gagal kumpul data besar
            local_crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Tandai waktu agar tidak langsung retry
            # CHART_INTEGRATION_START: Update shared manager untuk Flask
            with lock_ref:
                shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            # CHART_INTEGRATION_END
            continue # Lanjut ke pair berikutnya

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        # Warm-up state strategi dengan data historis (jika ada cukup data)
        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state {config['pair_name']}...", pair_name=config['pair_name'])
                
                # Iterasi dari candle ke-(min_len_for_pivots -1) hingga sebelum candle terakhir
                # untuk membangun state pivot dan FIB tanpa memicu trade.
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # -1 agar tidak proses candle paling baru untuk warm-up sinyal
                    historical_slice = initial_candles[:i+1] # Slice dari awal hingga candle ke-i
                    if len(historical_slice) < min_len_for_pivots: continue # Pastikan cukup data untuk pivot

                    # Buat state sementara untuk warm-up, jangan simpan posisi/trade
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Pastikan tidak ada posisi aktif selama warm-up
                    
                    # Jalankan logika strategi hanya untuk update internal state (pivot, fib)
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict
                    )
                    
                    # Pastikan setelah warm-up, state trading direset (jika run_strategy_logic sempat set)
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        local_crypto_data_manager[pair_id]["strategy_state"] = {
                            **local_crypto_data_manager[pair_id]["strategy_state"], # Ambil state pivot/fib
                            **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                               "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                               "current_trailing_stop_level":None} # Reset state trading
                        }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) untuk {config['pair_name']} tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}). State mungkin tidak optimal.", pair_name=config['pair_name'])
        else: # Tidak ada data awal sama sekali
            log_warning(f"Tidak ada data awal untuk warm-up state {config['pair_name']}.", pair_name=config['pair_name'])

        # Cek apakah target data besar tercapai setelah pengambilan awal
        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']} setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]: # Kirim notif jika belum
                send_email_notification(
                    f"Data Downloading Complete: {config['pair_name']}", 
                    f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", 
                    {**config, 'pair_name': config['pair_name']} # CHART_MOD: pastikan pair_name ada di konteks email
                )
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(local_crypto_data_manager[pair_id]['all_candles_list'])} candles) untuk {config['pair_name']} ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        # CHART_INTEGRATION_START: Update shared manager untuk Flask setelah inisialisasi pair
        with lock_ref:
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        # CHART_INTEGRATION_END

    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    # Loop Trading Utama
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf') # Untuk menentukan sleep optimal
            any_data_fetched_this_cycle = False # Flag apakah ada data baru di-fetch di siklus ini

            # Iterasi semua pair yang dikelola
            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                # Cooldown jika semua API key gagal untuk pair ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        log_debug(f"Pair {pair_name_for_log} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name_for_log)
                        continue # Skip pair ini untuk sementara
                    else: # Cooldown selesai
                        data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal
                        log_info(f"Cooldown 1 jam untuk {pair_name_for_log} selesai. Mencoba fetch lagi.", pair_name=pair_name_for_log)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()

                # Tentukan interval refresh yang dibutuhkan untuk pair ini
                required_interval_for_this_pair = 0
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Interval lebih cepat saat kumpul data besar, tergantung timeframe
                    if config_for_pair.get('timeframe') == "minute": required_interval_for_this_pair = 55 # Hampir tiap menit
                    elif config_for_pair.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 # Hampir tiap hari
                    else: required_interval_for_this_pair = 3580 # Hampir tiap jam
                else: # Fase live trading
                    required_interval_for_this_pair = config_for_pair.get('refresh_interval_seconds', 60) 

                # Jika belum waktunya refresh untuk pair ini, skip
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Lanjut ke pair berikutnya

                # Waktunya proses pair ini
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval_for_this_pair}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time # Update waktu fetch terakhir
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])

                if data_per_pair["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA {pair_name_for_log} ({len(data_per_pair['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005, new_line=True)
                else:
                    animated_text_display(f"\n--- ANALISA LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {len(data_per_pair['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005, new_line=True)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                # Tentukan berapa banyak candle yang akan di-fetch
                limit_fetch_for_update = 3 # Default untuk update live (ambil beberapa candle terakhir untuk jaga-jaga)
                if data_per_pair["big_data_collection_phase_active"]:
                    limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data_per_pair["all_candles_list"])
                    if limit_fetch_needed <=0 : # Seharusnya sudah selesai fase big data jika ini terjadi
                         fetch_update_successful_for_this_pair = True # Anggap berhasil, tidak perlu fetch
                         new_candles_batch = [] # Tidak ada candle baru
                         # Lanjutkan ke bawah untuk memastikan fase big data di-nonaktifkan
                    else:
                        limit_fetch_for_update = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) # Ambil sebanyak mungkin hingga max limit
                        limit_fetch_for_update = max(limit_fetch_for_update, 1) # Minimal 1

                if limit_fetch_for_update > 0 or data_per_pair["big_data_collection_phase_active"]: # Hanya fetch jika perlu
                    # Loop retry dengan API key berbeda jika gagal
                    max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    retries_done_for_this_pair_update = 0
                    original_api_key_index = api_key_manager.get_current_key_index() # Simpan index key awal

                    while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                        current_api_key_for_attempt = api_key_manager.get_current_key()
                        if not current_api_key_for_attempt:
                            log_error(f"Semua API key habis (global) saat mencoba mengambil update untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            break # Tidak ada key lagi

                        log_info(f"Mengambil {limit_fetch_for_update} candle untuk {pair_name_for_log} (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(
                                config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_for_update, 
                                config_for_pair['exchange'], current_api_key_for_attempt, config_for_pair['timeframe'],
                                pair_name=pair_name_for_log
                            )
                            fetch_update_successful_for_this_pair = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal jika berhasil
                            any_data_fetched_this_cycle = True # Tandai ada data di-fetch
                            if api_key_manager.get_current_key_index() != original_api_key_index : # Jika key diganti dan berhasil
                                log_info(f"Fetch berhasil dengan key index {api_key_manager.get_current_key_index()} setelah retry.", pair_name=pair_name_for_log)
                        
                        except APIKeyError:
                            log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name_for_log}. Mencoba key berikutnya.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            
                            if not api_key_manager.switch_to_next_key(): # Coba ganti key
                                log_error(f"Tidak ada lagi API key tersedia (global) setelah kegagalan pada {pair_name_for_log}.", pair_name=pair_name_for_log)
                                break # Berhenti retry jika tidak ada key lagi
                            retries_done_for_this_pair_update += 1 

                        except requests.exceptions.RequestException as e_req: # Error jaringan, jangan ganti key
                            log_error(f"Error jaringan saat mengambil update {pair_name_for_log}: {e_req}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            # Hentikan retry untuk pair ini di siklus ini jika jaringan error
                            break 
                        except Exception as e_gen_update: # Error umum lain
                            log_error(f"Error umum saat mengambil update {pair_name_for_log}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            log_exception("Traceback Error Update Fetch:", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            break # Hentikan retry

                # Jika semua upaya fetch gagal untuk pair ini di siklus ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() # Tandai waktu untuk cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name_for_log} di siklus ini. Akan masuk cooldown.", pair_name=pair_name_for_log)

                # Proses data baru jika fetch berhasil
                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data_per_pair["big_data_collection_phase_active"]:
                        log_info(f"Tidak ada data candle baru diterima untuk {pair_name_for_log} (fetch dianggap berhasil tapi batch kosong).", pair_name=pair_name_for_log)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    # Jangan proses lebih lanjut jika tidak ada candle baru atau fetch gagal
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) # Hitung untuk sleep
                    # CHART_INTEGRATION_START: Update shared manager bahkan jika tidak ada candle baru, untuk state
                    with lock_ref:
                        shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                    # CHART_INTEGRATION_END
                    continue # Lanjut ke pair berikutnya

                # Gabungkan candle baru dengan yang lama, hindari duplikasi
                # Gunakan dictionary untuk merge berdasarkan timestamp untuk efisiensi
                merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 # Jika candle dengan timestamp sama di-update (misal close berubah)

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: # Candle baru
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Candle lama tapi isinya berubah (misal candle saat ini di-update)
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1

                # Urutkan kembali semua candle berdasarkan timestamp
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                data_per_pair["all_candles_list"] = all_candles_list_temp # Update list candle di data manager lokal

                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate untuk {pair_name_for_log}. Total: {len(data_per_pair['all_candles_list'])}.", pair_name=pair_name_for_log)
                elif new_candles_batch : # Fetch berhasil tapi tidak ada timestamp baru/update
                     log_info(f"Tidak ada candle dengan timestamp baru atau konten berbeda untuk {pair_name_for_log}. Data terakhir mungkin identik.", pair_name=pair_name_for_log)

                # Cek lagi apakah target big data tercapai setelah update (jika masih dalam fase itu)
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        # Pangkas jika melebihi target
                        if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                            data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data_per_pair["big_data_email_sent"]: # Kirim notif email
                            send_email_notification(
                                f"Data Downloading Complete: {pair_name_for_log}", 
                                f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name_for_log}.", 
                                {**config_for_pair, 'pair_name': pair_name_for_log} # CHART_MOD
                            )
                            data_per_pair["big_data_email_sent"] = True
                        
                        data_per_pair["big_data_collection_phase_active"] = False # Selesai fase big data
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) # Kurangi counter
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data_per_pair['all_candles_list'])} candles) untuk {pair_name_for_log} ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                else: # Jika sudah live trading, pastikan tidak melebihi target (misal karena API memberi lebih)
                    if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]

                # Jalankan logika strategi jika ada cukup data
                min_len_for_pivots_logic = config_for_pair.get('left_strength',50) + config_for_pair.get('right_strength',150) + 1
                if len(data_per_pair["all_candles_list"]) >= min_len_for_pivots_logic:
                    # Hanya proses logika jika ada candle baru/diupdate, ATAU
                    # jika baru selesai fase big data, ATAU
                    # jika masih fase big data dan ada tambahan baru (untuk warm-up berkelanjutan)
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data_per_pair["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or # Transisi dari big data ke live
                                         (data_per_pair["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) )

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi untuk {pair_name_for_log} dengan {len(data_per_pair['all_candles_list'])} candle...", pair_name=pair_name_for_log)
                         data_per_pair["strategy_state"] = run_strategy_logic(
                             data_per_pair["all_candles_list"], 
                             config_for_pair, 
                             data_per_pair["strategy_state"],
                             global_settings_dict # CHART_MOD: Pass global_settings untuk notif termux
                         )
                    elif not data_per_pair["big_data_collection_phase_active"]: # Live tapi tidak ada candle baru
                         last_c_time_str = data_per_pair["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data_per_pair["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name_for_log}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name_for_log)
                else: # Tidak cukup data untuk analisa pivot
                    log_info(f"Data ({len(data_per_pair['all_candles_list'])}) untuk {pair_name_for_log} belum cukup utk analisa pivot (min: {min_len_for_pivots_logic}).", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) # Update untuk sleep
            
                # CHART_INTEGRATION_START: Update shared manager untuk Flask setelah pair diproses
                with lock_ref:
                    shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                # CHART_INTEGRATION_END

            # Tentukan durasi sleep sebelum siklus berikutnya
            sleep_duration = 15 # Default sleep jika tidak ada kalkulasi lain

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: # Semua key habis global & tidak ada data
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch di siklus ini. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600 # Cooldown 1 jam
            elif active_cryptos_still_in_big_data_collection > 0: # Masih ada yang kumpul big data
                min_big_data_interval_for_sleep = float('inf')
                for _pid_loop, pdata_loop_item in local_crypto_data_manager.items(): # Iterasi manager lokal
                    if pdata_loop_item["big_data_collection_phase_active"]:
                        pconfig_loop = pdata_loop_item["config"]
                        interval_bd_sleep = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_loop.get('timeframe') == "day" else 3580)
                        min_big_data_interval_for_sleep = min(min_big_data_interval_for_sleep, interval_bd_sleep)
                
                # Sleep berdasarkan interval big data terpendek, tapi tidak terlalu cepat
                sleep_duration = min(min_big_data_interval_for_sleep if min_big_data_interval_for_sleep != float('inf') else 30, 30) 
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua pair sudah live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    # Sleep hingga waktu refresh pair berikutnya, tapi tidak kurang dari batas minimal
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya (min_overall_next_refresh_seconds: {min_overall_next_refresh_seconds:.0f}s).", pair_name="SYSTEM")
                else: # Fallback jika min_overall_next_refresh_seconds tidak terhitung
                    default_refresh_from_first_config = 60 # Default umum
                    if all_crypto_configs : # Ambil dari config pair pertama jika ada
                        default_refresh_from_first_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_first_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama setelah live).", pair_name="SYSTEM")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s hingga siklus berikutnya...")
            else: # Jaga-jaga jika sleep_duration jadi 0 atau negatif
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM")
                time.sleep(1)

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna (Ctrl+C).{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01, new_line=True)
    except Exception as e_main_loop:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error Loop Trading Utama:", pair_name="SYSTEM") # CHART_MOD
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005, new_line=True)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()

    # CHART_INTEGRATION_START: Jalankan Flask server di thread terpisah
    flask_server_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
    flask_server_thread.start()
    # CHART_INTEGRATION_END

    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Chart) =========", color=AnsiColors.HEADER, delay=0.005) # CHART_MOD: Title

        pick_title_main = ""
        active_configs_list = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)] # Ambil yang aktif
        if active_configs_list:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs_list)}) ---\n"
            for i, cfg_item in enumerate(active_configs_list): # Tampilkan beberapa info
                pick_title_main += f"  {i+1}. {cfg_item.get('symbol','N/A')}-{cfg_item.get('currency','N/A')} (TF: {cfg_item.get('timeframe','N/A')}, Exch: {cfg_item.get('exchange','N/A')})\n"
        else:
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s_main = settings.get("api_settings", {})
        primary_key_display_main = api_s_main.get('primary_key', 'BELUM DIATUR')
        if primary_key_display_main and len(primary_key_display_main) > 10 and primary_key_display_main not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display_main = "..." + primary_key_display_main[-5:] # CHART_MOD: Disingkat
        num_recovery_keys_main = len([k for k in api_s_main.get('recovery_keys',[]) if k])
        termux_notif_main_status = "Aktif" if api_s_main.get("enable_termux_notifications", False) else "Nonaktif"

        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display_main} | Recovery Keys: {num_recovery_keys_main}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status}\n"
        # CHART_MOD: Info Chart Server
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
        except Exception as e_pick_main: # Fallback ke input manual jika `pick` error
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main) # Tampilkan title lagi
            for idx_main, opt_text_main in enumerate(main_menu_options_plain):
                print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main_input = input("Pilih nomor opsi: ").strip()
                if not choice_main_input: continue # Kembali jika input kosong
                choice_main_val = int(choice_main_input) -1
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
        
        # `selected_main_index` sudah merupakan index dari `main_menu_options_plain`
        if selected_main_index == 0: # Mulai Analisa
            # CHART_MOD: Pass shared data manager dan lock ke start_trading
            start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif selected_main_index == 1: # Pengaturan
            settings = settings_menu(settings) # settings_menu mengembalikan settings yang mungkin diubah
            # CHART_MOD: Setelah kembali dari settings, shared_crypto_data_manager mungkin perlu di-clear atau di-update
            # jika konfigurasi pair berubah drastis. Untuk saat ini, kita biarkan.
            # Jika user menonaktifkan pair, chart akan tetap menampilkannya sampai restart/refresh data.
            # Atau, kita bisa panggil clear di sini:
            with shared_data_lock:
                # Jika ada perubahan config, data lama mungkin tidak relevan lagi.
                # Ini akan memaksa chart mengambil data baru jika pair masih ada dan aktif.
                # Atau, bisa juga lebih selektif menghapus pair yang diubah/dinonaktifkan.
                # Untuk kesederhanaan, kita bisa refresh semua atau biarkan.
                # shared_crypto_data_manager.clear() # Ini akan membersihkan semua data chart
                # log_info("Data chart di-reset karena perubahan pengaturan.", pair_name="SYSTEM_CHART")
                # Pendekatan yang lebih baik: start_trading akan meng-overwrite/menginisialisasi ulang data pair yang aktif.
                pass


        elif selected_main_index == 2: # Keluar
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA, new_line=True)
            show_spinner(0.5, "Exiting")
            break # Keluar dari loop while True -> keluar dari main_menu


if __name__ == "__main__":
    try:
        # Pastikan direktori log ada (jika diperlukan oleh logger)
        # if not os.path.exists("logs"): os.makedirs("logs") # Contoh
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa dari level utama (Ctrl+C). Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01, new_line=True)
    except Exception as e_global: # Tangkap error global yang mungkin tidak tertangani
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}TERJADI ERROR KRITIKAL DI LEVEL UTAMA:{AnsiColors.ENDC}")
        print(f"{AnsiColors.RED}{str(e_global)}{AnsiColors.ENDC}")
        # Gunakan logger yang sudah dikonfigurasi untuk traceback yang lebih baik
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        print(f"\n{AnsiColors.ORANGE}Detail error telah dicatat di: {log_file_name}{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        # Pastikan semua output di-flush
        sys.stdout.flush()
        sys.stderr.flush()
