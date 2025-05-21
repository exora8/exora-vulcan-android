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
TARGET_BIG_DATA_CANDLES = 2500 # Minimal EMA length + beberapa buffer
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
def get_default_crypto_config():
    # Menggabungkan parameter dari strategi PineScript
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,

        # Parameter dari PineScript EMA GR1/GR2
        "ema_length": 200,
        "gr2_drop_percentage": 15.0, # % penurunan dari GR1 untuk aktivasi GR2

        # Parameter SL/TP yang sudah ada di Python, dipetakan dari PineScript
        "profit_target_percent_activation": 10.0, # Mirip tp_activation_perc_input
        "trailing_stop_gap_percent": 5.0,        # Mirip tp_trailing_gap_perc_input
        "emergency_sl_percent": 5.0,             # Mirip sl_perc_input

        # Parameter lama dari skrip Python (jika masih ada guna untuk fitur lain, jika tidak bisa dihapus)
        "left_strength": 50, # Tidak digunakan oleh strategi EMA GR1/GR2
        "right_strength": 150,# Tidak digunakan oleh strategi EMA GR1/GR2
        "enable_secure_fib": True, # Tidak digunakan oleh strategi EMA GR1/GR2
        "secure_fib_check_price": "Close", # Tidak digunakan oleh strategi EMA GR1/GR2

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
            
            # Pastikan semua config crypto memiliki semua field default
            default_crypto_template = get_default_crypto_config()
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg:
                        crypto_cfg[key] = default_value
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

def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy() # Mulai dengan nilai yang ada
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

    animated_text_display("\n-- Parameter Strategi EMA GR1/GR2 --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["ema_length"] = int(input(f"{AnsiColors.BLUE}Panjang EMA [{new_config.get('ema_length',200)}]: {AnsiColors.ENDC}").strip() or new_config.get('ema_length',200))
        new_config["gr2_drop_percentage"] = float(input(f"{AnsiColors.BLUE}GR2 - Min. Drop % dari GR1 [{new_config.get('gr2_drop_percentage',15.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('gr2_drop_percentage',15.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter EMA/GR2 tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["ema_length"] = new_config.get('ema_length',200)
        new_config["gr2_drop_percentage"] = new_config.get('gr2_drop_percentage',15.0)

    animated_text_display("\n-- Parameter Stop Loss & Take Profit --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Stop Loss Tetap % [{new_config.get('emergency_sl_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',5.0))
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}TP - Aktivasi Trailing % [{new_config.get('profit_target_percent_activation',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',10.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}TP - Jarak Trailing % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter SL/TP tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',5.0)
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',10.0)
        new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0)

    # Parameter lama yang tidak digunakan oleh strategi EMA GR1/GR2 dapat disembunyikan atau ditandai sebagai tidak relevan
    # Untuk saat ini, saya akan membiarkannya agar struktur save/load tidak pecah, tapi Anda bisa menghapusnya jika mau.
    # animated_text_display("\n-- Parameter Pivot (Tidak digunakan oleh EMA GR1/GR2) --", color=AnsiColors.ORANGE, delay=0.01)
    # ... (input left_strength, right_strength, enable_secure_fib, secure_fib_check_price bisa di-skip/dihapus)

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
        pick_title_settings += "Strategi Aktif: EMA GR1/GR2 Entry\n"
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
                new_crypto_conf = get_default_crypto_config() # Ini sudah berisi parameter EMA GR1/GR2
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

            if response.status_code in [401, 403, 429]:
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit",
                    "please pass an API key", "api_key not found"
                ]
                key_display = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: {key_display}", pair_name=pair_name)
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
                    'volume': item.get('volumefrom')
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
                time.sleep(0.3)

        except APIKeyError:
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles (batch loop): {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error Fetch Candles (batch loop):", pair_name=pair_name)
            break

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        # State untuk EMA GR1/GR2
        "inGetReady1": False,
        "inGetReady2": False,
        "priceAtGR1Close": None,
        "last_ema_value_for_chart": None, # Untuk plotting EMA di chart

        # State yang sudah ada untuk manajemen posisi
        "entry_price_custom": None,
        "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None,
        "position_size": 0, # 0 = no position, 1 = long position (atau qty jika diubah)
        
        # State lama (tidak digunakan oleh EMA GR1/GR2, bisa dihapus jika tidak ada strategi lain)
        "last_signal_type": 0,
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None,
        "last_pivot_low_display_info": None,
        "high_price_for_fib": None,
        "high_bar_index_for_fib": None,
        "active_fib_level": None,
        "active_fib_line_start_index": None,
    }

# Fungsi find_pivots (tidak digunakan oleh EMA GR1/GR2, tapi dibiarkan sesuai permintaan "yg lain jangan diubah")
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

def calculate_ema(prices, period):
    if not prices or not any(p is not None for p in prices): # Cek jika semua None
        return [None] * len(prices)
    
    valid_prices_indices = [i for i, p in enumerate(prices) if p is not None]
    if not valid_prices_indices:
        return [None] * len(prices)

    first_valid_index = valid_prices_indices[0]
    
    # Hanya proses dari harga valid pertama
    effective_prices = [p for p in prices[first_valid_index:] if p is not None]

    if len(effective_prices) < period:
        # Pad None di awal, dan None untuk sisa yang tidak cukup data
        return [None] * first_valid_index + [None] * len(effective_prices) + [None] * (len(prices) - first_valid_index - len(effective_prices))


    ema_values_effective = [None] * len(effective_prices)
    
    # Hitung SMA untuk nilai EMA pertama
    sma = sum(effective_prices[:period]) / period
    ema_values_effective[period - 1] = sma
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(effective_prices)):
        if ema_values_effective[i-1] is None: # Jika EMA sebelumnya tidak bisa dihitung
             ema_values_effective[i] = None # Maka EMA saat ini juga tidak bisa
        else:
            ema_values_effective[i] = (effective_prices[i] - ema_values_effective[i-1]) * multiplier + ema_values_effective[i-1]

    # Gabungkan dengan None di awal jika ada
    final_ema_values = [None] * first_valid_index + ema_values_effective
    # Pad dengan None di akhir jika panjang prices asli lebih besar
    if len(final_ema_values) < len(prices):
        final_ema_values.extend([None] * (len(prices) - len(final_ema_values)))
        
    return final_ema_values


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"

    # --- Parameter Strategi ---
    ema_length = crypto_config.get('ema_length', 200)
    gr2_required_drop_percent = crypto_config.get('gr2_drop_percentage', 15.0) / 100.0 # Jadi desimal
    sl_percentage = crypto_config.get('emergency_sl_percent', 5.0) / 100.0
    tp_activation_percentage = crypto_config.get('profit_target_percent_activation', 10.0) / 100.0
    tp_trailing_gap_percentage = crypto_config.get('trailing_stop_gap_percent', 5.0) / 100.0

    # --- Pastikan data cukup ---
    if len(candles_history) < max(2, ema_length): # Butuh min 2 candle untuk cross, dan ema_length untuk EMA
        log_debug(f"Data candle tidak cukup ({len(candles_history)}) untuk EMA {ema_length}. Skip logic.", pair_name=pair_name)
        return strategy_state

    # --- Ambil data candle & hitung EMA ---
    close_prices = [c.get('close') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history] # Untuk GR2
    
    ema_values = calculate_ema(close_prices, ema_length)

    current_candle_idx = len(candles_history) - 1
    prev_candle_idx = current_candle_idx - 1

    if current_candle_idx < 0: return strategy_state # Seharusnya tidak terjadi jika len_check di atas benar
    
    current_candle = candles_history[current_candle_idx]
    # Pastikan semua field penting ada di candle terbaru
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=pair_name)
        return strategy_state

    current_close = current_candle['close']
    current_low = current_candle['low']
    current_ema = ema_values[current_candle_idx]
    strategy_state["last_ema_value_for_chart"] = current_ema # Simpan untuk chart

    # Butuh data sebelumnya untuk deteksi cross
    if prev_candle_idx < 0 or ema_values[prev_candle_idx] is None or candles_history[prev_candle_idx].get('close') is None:
        log_debug("Tidak cukup data candle/EMA sebelumnya untuk deteksi cross. Skip logic cross.", pair_name=pair_name)
        # Jalankan exit logic jika dalam posisi
        if strategy_state["position_size"] > 0:
            pass # Exit logic akan jalan di bawah
        else:
            return strategy_state # Jika tidak dalam posisi dan tidak bisa cek cross, keluar

    prev_close = candles_history[prev_candle_idx]['close']
    prev_ema = ema_values[prev_candle_idx]

    # --- Deteksi Cross ---
    is_cross_under_ema = False
    is_cross_over_ema = False

    if prev_close is not None and prev_ema is not None and current_close is not None and current_ema is not None:
        is_cross_under_ema = prev_close > prev_ema and current_close < current_ema
        is_cross_over_ema = prev_close < prev_ema and current_close > current_ema

    # --- Logika Reset State (jika trade baru saja ditutup) ---
    # Ini dilakukan di awal agar state bersih sebelum evaluasi kondisi baru
    # Deteksi trade ditutup: jika posisi_size adalah 0 SEKARANG, tapi di state SEBELUM fungsi ini dipanggil adalah > 0
    # Ini sedikit rumit karena kita memodifikasi state_strategy secara langsung.
    # Cara PineScript: if strategy.closedtrades > strategy.closedtrades[1]
    # Di Python, kita bisa cek jika posisi jadi 0 di akhir blok 'if strategy_state["position_size"] > 0'
    # Untuk sekarang, reset akan ada di dalam blok exit.

    # --- Get Ready 1 (GR1) Condition ---
    if is_cross_under_ema and \
       strategy_state["position_size"] == 0 and \
       not strategy_state["inGetReady1"] and \
       not strategy_state["inGetReady2"]:
        
        strategy_state["inGetReady1"] = True
        strategy_state["inGetReady2"] = False # Reset GR2 jika sinyal GR1 baru
        strategy_state["priceAtGR1Close"] = current_close
        log_info(f"{AnsiColors.ORANGE}GR1 AKTIF: EMA ({current_ema:.5f}) dilintasi KE BAWAH. Harga GR1: {current_close:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
        # Notifikasi Termux/Email untuk GR1 (opsional, sesuai preferensi)
        # send_termux_notification(f"GR1: {pair_name}", f"EMA Crossed DOWN @ {current_close:.5f}", global_settings, pair_name)

    # --- Get Ready 2 (GR2) Condition ---
    if strategy_state["inGetReady1"] and \
       not strategy_state["inGetReady2"] and \
       strategy_state["priceAtGR1Close"] is not None and \
       current_low is not None:
        
        price_at_gr1 = strategy_state["priceAtGR1Close"]
        if price_at_gr1 > 0: # Hindari pembagian dengan nol
            current_drop_from_gr1_price_abs = price_at_gr1 - current_low
            current_drop_percentage = current_drop_from_gr1_price_abs / price_at_gr1
            
            if current_drop_percentage >= gr2_required_drop_percent:
                strategy_state["inGetReady2"] = True
                strategy_state["inGetReady1"] = False # GR1 terpenuhi, sekarang di GR2
                log_info(f"{AnsiColors.YELLOW_BG}{AnsiColors.RED}GR2 AKTIF: Harga turun {current_drop_percentage*100:.2f}% dari GR1 ({price_at_gr1:.5f}) ke {current_low:.5f}. EMA: {current_ema:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
                # send_termux_notification(f"GR2: {pair_name}", f"Drop {current_drop_percentage*100:.2f}% from GR1. Price: {current_low:.5f}", global_settings, pair_name)


    # --- Entry Condition ---
    if strategy_state["inGetReady2"] and \
       is_cross_over_ema and \
       strategy_state["position_size"] == 0:
        
        entry_price = current_close
        strategy_state["position_size"] = 1 # Atau qty berdasarkan equity jika diimplementasikan
        strategy_state["entry_price_custom"] = entry_price
        
        # Hitung SL dan inisialisasi TP
        strategy_state["emergency_sl_level_custom"] = entry_price * (1 - sl_percentage)
        strategy_state["highest_price_for_trailing"] = entry_price
        strategy_state["trailing_tp_active_custom"] = False
        strategy_state["current_trailing_stop_level"] = None
        
        strategy_state["inGetReady2"] = False # Reset GR2 setelah entry

        log_msg = f"BUY ENTRY @ {entry_price:.5f} (EMA: {current_ema:.5f}). SL: {strategy_state['emergency_sl_level_custom']:.5f}"
        log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
        play_notification_sound()
        
        termux_title = f"BUY Signal: {pair_name}"
        termux_content = f"Entry @ {entry_price:.5f}. EMA: {current_ema:.5f}. SL: {strategy_state['emergency_sl_level_custom']:.5f}"
        send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)

        email_subject = f"BUY Signal: {pair_name}"
        email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\n"
                      f"Entry Price: {entry_price:.5f}\n"
                      f"EMA ({ema_length}): {current_ema:.5f}\n"
                      f"Stop Loss: {strategy_state['emergency_sl_level_custom']:.5f}\n"
                      f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})

    # --- Exit Conditions (Menggunakan Logika SL/TP yang sudah ada di Python) ---
    if strategy_state["position_size"] > 0:
        entry_price_val = strategy_state["entry_price_custom"]
        
        # Update harga tertinggi untuk trailing
        if current_candle.get('high') is not None:
            if strategy_state["highest_price_for_trailing"] is None: # Harusnya sudah di-set saat entry
                 strategy_state["highest_price_for_trailing"] = current_candle['high']
            else:
                 strategy_state["highest_price_for_trailing"] = max(strategy_state["highest_price_for_trailing"], current_candle['high'])

        # Aktivasi Trailing TP
        if not strategy_state["trailing_tp_active_custom"] and entry_price_val is not None and entry_price_val > 0:
            highest_price = strategy_state["highest_price_for_trailing"]
            if highest_price is not None:
                profit_perc = ((highest_price - entry_price_val) / entry_price_val) # Sudah desimal
                if profit_perc >= tp_activation_percentage:
                    strategy_state["trailing_tp_active_custom"] = True
                    log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_perc*100:.2f}%, High Sejak Entry: {highest_price:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        # Update Trailing Stop Level
        if strategy_state["trailing_tp_active_custom"]:
            highest_price = strategy_state["highest_price_for_trailing"]
            if highest_price is not None:
                potential_new_stop = highest_price * (1 - tp_trailing_gap_percentage)
                if strategy_state["current_trailing_stop_level"] is None or potential_new_stop > strategy_state["current_trailing_stop_level"]:
                    strategy_state["current_trailing_stop_level"] = potential_new_stop
                    log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        # Tentukan SL yang berlaku (Emergency SL atau Trailing SL)
        effective_sl_level = strategy_state["emergency_sl_level_custom"]
        exit_reason = "Stop Loss"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if effective_sl_level is None or strategy_state["current_trailing_stop_level"] > effective_sl_level:
                effective_sl_level = strategy_state["current_trailing_stop_level"]
                exit_reason = "Trailing TP"
                exit_color = AnsiColors.BLUE # Biru untuk Trailing TP yang profit

        # Cek jika SL/Trailing TP kena
        if effective_sl_level is not None and current_low is not None and current_low <= effective_sl_level:
            # Untuk exit, kita bisa asumsikan dieksekusi pada open candle berikutnya atau pada level SL itu sendiri.
            # PineScript strategy.exit biasanya di bar berikutnya, atau di harga SL jika itu yang kena duluan.
            # Kita pakai harga SL sebagai harga exit untuk simplisitas.
            exit_price = effective_sl_level 
            
            pnl = 0.0
            if entry_price_val is not None and entry_price_val != 0:
                pnl = ((exit_price - entry_price_val) / entry_price_val) * 100.0 # PnL dalam %

            if exit_reason == "Trailing TP" and pnl < 0: # Jika Trailing TP tapi rugi (misal gap besar)
                exit_color = AnsiColors.RED
                exit_reason = "Trailing SL (Loss)"


            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_reason}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

            termux_title_exit = f"EXIT Signal: {pair_name}"
            termux_content_exit = f"{exit_reason} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)

            email_subject = f"Trade Closed: {pair_name} ({exit_reason})"
            email_body = (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\n"
                          f"Reason: {exit_reason}\n"
                          f"Entry Price: {entry_price_val:.5f}\n"
                          f"PnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})
            
            # Reset state posisi
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None

            # Reset state GR1/GR2 karena trade sudah selesai
            strategy_state["inGetReady1"] = False
            strategy_state["inGetReady2"] = False
            strategy_state["priceAtGR1Close"] = None
            log_info("Status GR1/GR2 direset setelah trade ditutup.", pair_name=pair_name)
            
        elif strategy_state["position_size"] > 0 : # Jika masih dalam posisi & SL/TP tidak kena
            entry_display = strategy_state.get('entry_price_custom', 0)
            sl_display_str = f'{effective_sl_level:.5f} ({exit_reason})' if effective_sl_level is not None else 'N/A'
            log_debug(f"Posisi Aktif. Entry: {entry_display:.5f}, SL Saat Ini: {sl_display_str}, High: {strategy_state.get('highest_price_for_trailing',0):.5f}", pair_name=pair_name)

    return strategy_state

# CHART_INTEGRATION_START
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot):
    if pair_id_to_display not in current_data_manager_snapshot:
        log_warning(f"Data untuk pair {pair_id_to_display} tidak ditemukan di snapshot untuk chart.", pair_name="SYSTEM_CHART")
        return None

    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]
    candles_full_history = pair_specific_data.get("all_candles_list", [])
    current_strategy_state = pair_specific_data.get("strategy_state", {})
    pair_config = pair_specific_data.get("config", {})

    candles_for_chart_display = candles_full_history[-TARGET_BIG_DATA_CANDLES:]

    ohlc_data_points = []
    ema_series_data = [] # Untuk plot EMA
    
    if not candles_for_chart_display:
        log_warning(f"Tidak ada candle di `candles_for_chart_display` untuk {pair_id_to_display}.", pair_name="SYSTEM_CHART")
        return {"ohlc": [], "ema_series": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None, "background_colors": []}


    # Hitung ulang EMA untuk data yang akan ditampilkan di chart
    chart_close_prices = [c.get('close') for c in candles_for_chart_display]
    chart_ema_length = pair_config.get('ema_length', 200)
    chart_ema_values = calculate_ema(chart_close_prices, chart_ema_length)

    for i, candle in enumerate(candles_for_chart_display):
        required_candle_keys = ['timestamp', 'open', 'high', 'low', 'close']
        if all(k in candle and candle[k] is not None for k in required_candle_keys):
            ts_ms = candle['timestamp'].timestamp() * 1000
            ohlc_data_points.append({
                'x': ts_ms,
                'y': [candle['open'], candle['high'], candle['low'], candle['close']]
            })
            if chart_ema_values[i] is not None:
                ema_series_data.append({'x': ts_ms, 'y': chart_ema_values[i]})
        else:
            log_debug(f"Skipping incomplete candle for chart: {candle.get('timestamp')}", pair_name="SYSTEM_CHART")


    chart_annotations_yaxis = []
    chart_annotations_points = [] # Tidak ada anotasi titik dari strategi EMA GR1/GR2 (seperti Pivot)
    
    # Anotasi untuk SL & Entry Price
    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") is not None:
        entry_price_val = current_strategy_state.get("entry_price_custom")
        
        if ohlc_data_points: # Hanya tambah jika ada data candle
             chart_annotations_yaxis.append({
                'y': entry_price_val,
                'borderColor': '#2698FF', # Biru untuk entry
                'strokeDashArray': 4,
                'label': {
                    'borderColor': '#2698FF',
                    'style': {'color': '#fff', 'background': '#2698FF', 'fontSize': '10px', 'padding': {'left': '3px', 'right': '3px', 'top':'1px', 'bottom':'1px'}},
                    'text': f'Entry: {entry_price_val:.5f}'
                }
            })

        sl_level_val = current_strategy_state.get("emergency_sl_level_custom")
        sl_type_text = "SL"
        if current_strategy_state.get("trailing_tp_active_custom") and current_strategy_state.get("current_trailing_stop_level") is not None:
            current_trailing_sl_val = current_strategy_state.get("current_trailing_stop_level")
            if sl_level_val is None or (current_trailing_sl_val is not None and current_trailing_sl_val > sl_level_val):
                sl_level_val = current_trailing_sl_val
                sl_type_text = "Trail.SL"
        
        if sl_level_val and ohlc_data_points:
            chart_annotations_yaxis.append({
                'y': sl_level_val,
                'borderColor': '#FF4560', # Merah untuk SL
                'label': {
                    'borderColor': '#FF4560',
                    'style': {'color': '#fff', 'background': '#FF4560', 'fontSize': '10px', 'padding': {'left': '3px', 'right': '3px', 'top':'1px', 'bottom':'1px'}},
                    'text': f'{sl_type_text}: {sl_level_val:.5f}'
                }
            })
    
    # Background colors for GR1/GR2 (simplified, just for the last known state if active)
    # Ini memerlukan modifikasi signifikan di frontend untuk render per bar.
    # Untuk simplifikasi, kita bisa kirim state GR1/GR2 dan biarkan frontend menampilkannya sebagai label, bukan background.
    # Atau, kita bisa tandai bar terakhir jika GR1/GR2 aktif.
    # PineScript: bgcolor(inGetReady1 ? color.new(color.orange, 80) : na, title="GR1 Active")
    # PineScript: bgcolor(inGetReady2 ? color.new(color.yellow, 70) : na, title="GR2 Active")
    # Ini akan saya skip untuk saat ini karena kompleksitas di frontend. Label status GR1/GR2 bisa ditambahkan di judul chart atau info teks.

    last_updated_tv_val = None
    if candles_for_chart_display and candles_for_chart_display[-1].get('timestamp'):
        last_updated_tv_val = candles_for_chart_display[-1]['timestamp'].timestamp() * 1000


    return {
        "ohlc": ohlc_data_points,
        "ema_series": ema_series_data, # Data EMA untuk diplot sebagai garis
        "ema_length_for_label": chart_ema_length,
        "annotations_yaxis": chart_annotations_yaxis,
        "annotations_points": chart_annotations_points,
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": last_updated_tv_val,
        "strategy_state_info": { # Info tambahan untuk ditampilkan di chart
            "inGetReady1": current_strategy_state.get("inGetReady1", False),
            "inGetReady2": current_strategy_state.get("inGetReady2", False),
            "priceAtGR1Close": current_strategy_state.get("priceAtGR1Close")
        }
    }

flask_app_instance = Flask(__name__)

HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crypto Strategy Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;}
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width:100%; max-width: 1200px; }
        #controls label { font-size: 0.9em; }
        select, button { padding: 8px 12px; font-size:0.9em; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; }
        button:hover { background-color: #444; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        h1 { text-align: center; color: #00bcd4; margin-top: 0; margin-bottom:10px; font-size:1.5em; }
        #lastUpdatedLabel { font-size: 0.8em; color: #aaa; margin-left: auto; padding-right: 5px; }
        #strategyStateLabel { font-size: 0.8em; color: #FFD700; /* Gold */ margin-left: 10px; }
        .apexcharts-tooltip-candlestick { background: #333 !important; color: #fff !important; border: 1px solid #555 !important;}
        .apexcharts-tooltip-candlestick .value { font-weight: bold; }
        .apexcharts-marker-inverted .apexcharts-marker-poly { transform: rotate(180deg); transform-origin: center; }
    </style>
</head>
<body>
    <h1>EMA GR1/GR2 Strategy Chart</h1>
    <div id="controls">
        <label for="pairSelector">Pilih Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh Manual</button>
        <span id="strategyStateLabel">Status: -</span>
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
            series: [
                { name: 'Candlestick', type: 'candlestick', data: [] },
                { name: 'EMA', type: 'line', data: [] }
            ],
            chart: { 
                type: 'candlestick', // Default, akan di-override oleh series type
                height: 550,
                id: 'mainStrategyChart',
                background: '#2a2a2a',
                animations: { enabled: true, easing: 'easeinout', speed: 500, animateGradually: { enabled: false } },
                toolbar: { show: true, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } }
            },
            theme: { mode: 'dark' },
            title: { text: 'Memuat Data Pair...', align: 'left', style: { color: '#e0e0e0', fontSize: '16px'} },
            xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
            yaxis: { 
                tooltip: { enabled: true }, 
                labels: { style: { colors: '#aaa'}, formatter: function (value) { return value ? value.toFixed(5) : ''; } } 
            },
            stroke: { // Untuk garis EMA
              width: [1, 2], // Lebar stroke untuk candlestick dan EMA
              curve: 'smooth'
            },
            markers: { // Untuk garis EMA (jika diinginkan marker di tiap titik)
                size: 0 // Sembunyikan marker default untuk line
            },
            colors: ['#FEB019', '#008FFB'], // Warna untuk Candlestick (tidak berlaku langsung) dan EMA (biru)
            grid: { borderColor: '#444' },
            annotations: { yaxis: [], points: [] },
            tooltip: { 
                theme: 'dark', 
                shared: true, 
                intersect: false, // Biar tooltip muncul walau tidak tepat di atas candle/line
                y: {
                    formatter: function(value, { series, seriesIndex, dataPointIndex, w }) {
                        if (w.config.series[seriesIndex].type === 'candlestick') {
                            // Ini tidak akan terpanggil karena custom tooltip di bawah
                            return value ? value.toFixed(5) : '';
                        }
                        return value ? value.toFixed(5) : ''; // Untuk EMA dan y-axis annotations
                    }
                },
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    let o, h, l, c;
                    let emaVal;
                    const candlestickSeriesIndex = w.globals.series.findIndex(s => s.type === 'candlestick' || w.config.series.findIndex(cs => cs.name === 'Candlestick'));
                    const emaSeriesIndex = w.globals.series.findIndex(s => s.name === 'EMA');

                    if (candlestickSeriesIndex !== -1 && w.globals.seriesCandleO[candlestickSeriesIndex] && w.globals.seriesCandleO[candlestickSeriesIndex][dataPointIndex] !== undefined) {
                        o = w.globals.seriesCandleO[candlestickSeriesIndex][dataPointIndex];
                        h = w.globals.seriesCandleH[candlestickSeriesIndex][dataPointIndex];
                        l = w.globals.seriesCandleL[candlestickSeriesIndex][dataPointIndex];
                        c = w.globals.seriesCandleC[candlestickSeriesIndex][dataPointIndex];
                    }
                    if (emaSeriesIndex !== -1 && series[emaSeriesIndex] && series[emaSeriesIndex][dataPointIndex] !== undefined) {
                         // Cek jika data point ada di series EMA
                         if (w.globals.series[emaSeriesIndex] && w.globals.series[emaSeriesIndex][dataPointIndex] !== undefined) {
                            emaVal = w.globals.series[emaSeriesIndex][dataPointIndex];
                         }
                    }
                    
                    let html = '<div class="apexcharts-tooltip-candlestick" style="padding:5px 10px;">';
                    if (o !== undefined) {
                        html += '<div>O: <span class="value">' + o.toFixed(5) + '</span></div>' +
                                '<div>H: <span class="value">' + h.toFixed(5) + '</span></div>' +
                                '<div>L: <span class="value">' + l.toFixed(5) + '</span></div>' +
                                '<div>C: <span class="value">' + c.toFixed(5) + '</span></div>';
                    }
                    if (emaVal !== undefined) {
                        html += '<div>EMA: <span class="value">' + emaVal.toFixed(5) + '</span></div>';
                    }
                    html += '</div>';
                    return (o !== undefined || emaVal !== undefined) ? html : '';
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
                     document.getElementById('strategyStateLabel').textContent = "Status: -";
                }
            } catch (error) {
                console.error("Error fetching pair list:", error);
                document.getElementById('pairSelector').innerHTML = '<option value="">Error memuat pair</option>';
                if(activeChart) activeChart.destroy();
                activeChart = null;
                document.getElementById('chart').innerHTML = `<p style="text-align:center; color:red;">Error: ${error.message}</p>`;
                document.getElementById('lastUpdatedLabel').textContent = "Error";
                document.getElementById('strategyStateLabel').textContent = "Status: Error";
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
                if(activeChart) activeChart.updateOptions(initialChartOptions); // Reset ke opsi awal
                document.getElementById('lastUpdatedLabel').textContent = "Pilih pair";
                document.getElementById('strategyStateLabel').textContent = "Status: Pilih Pair";
                return;
            }
            if (isLoadingData) {
                console.log(`Sinkronisasi data untuk ${currentSelectedPairId} sedang berjalan. Lewati sementara.`);
                return;
            }

            isLoadingData = true;
            document.getElementById('lastUpdatedLabel').textContent = `Sinkronisasi ${currentSelectedPairId}...`;
            document.getElementById('strategyStateLabel').textContent = "Status: Memuat...";
            
            try {
                const fetchResponse = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!fetchResponse.ok) {
                     let errorMsgText = `Gagal mengambil data chart: ${fetchResponse.status}`;
                     try { const errorData = await fetchResponse.json(); errorMsgText = errorData.error || errorMsgText; } catch(e){}
                     throw new Error(errorMsgText);
                }
                const chartDataPayload = await fetchResponse.json();

                if (!chartDataPayload || !chartDataPayload.ohlc || chartDataPayload.ohlc.length === 0) {
                    console.warn(`Data OHLC tidak diterima atau kosong untuk ${currentSelectedPairId}.`);
                    const pairDisplayName = chartDataPayload.pair_name || currentSelectedPairId;
                    const noDataOpts = {
                        ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${pairDisplayName} - Tidak Ada Data Candle` },
                        series: [{ name: 'Candlestick', type: 'candlestick', data: [] }, { name: 'EMA', type: 'line', data: [] }],
                        annotations: { yaxis: [], points: [] },
                        noData: { text: 'Tidak ada data candle terbaru dari server.' }
                    };
                    if (!activeChart) {
                        activeChart = new ApexCharts(document.querySelector("#chart"), noDataOpts);
                        activeChart.render();
                    } else {
                        activeChart.updateOptions(noDataOpts);
                    }
                    lastKnownDataTimestamp = chartDataPayload.last_updated_tv || null;
                    document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data (kosong) @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Tidak ada data";
                    document.getElementById('strategyStateLabel').textContent = "Status: Data Kosong";
                    isLoadingData = false; 
                    return; 
                }

                if (chartDataPayload.last_updated_tv && chartDataPayload.last_updated_tv === lastKnownDataTimestamp) {
                    console.log("Data chart tidak berubah, tidak perlu update render.");
                    document.getElementById('lastUpdatedLabel').textContent = `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                    // Update status GR1/GR2 meskipun data candle tidak berubah
                    const stateInfo = chartDataPayload.strategy_state_info || {};
                    let statusText = "Status: ";
                    if (stateInfo.inGetReady2) statusText += "GR2 Aktif";
                    else if (stateInfo.inGetReady1) statusText += "GR1 Aktif";
                    else statusText += "Idle";
                    if (stateInfo.priceAtGR1Close) statusText += ` (GR1 Price: ${stateInfo.priceAtGR1Close.toFixed(5)})`;
                    document.getElementById('strategyStateLabel').textContent = statusText;
                    isLoadingData = false; 
                    return;
                }
                lastKnownDataTimestamp = chartDataPayload.last_updated_tv;
                document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "N/A";

                const stateInfo = chartDataPayload.strategy_state_info || {};
                let statusText = "Status: ";
                if (stateInfo.inGetReady2) statusText += "GR2 Aktif";
                else if (stateInfo.inGetReady1) statusText += "GR1 Aktif";
                else statusText += "Idle";
                if (stateInfo.priceAtGR1Close) statusText += ` (GR1 @ ${stateInfo.priceAtGR1Close.toFixed(5)})`;
                document.getElementById('strategyStateLabel').textContent = statusText;

                const emaLabel = chartDataPayload.ema_length_for_label ? `EMA (${chartDataPayload.ema_length_for_label})` : 'EMA';

                const newChartOptions = {
                    ...initialChartOptions, 
                    title: { ...initialChartOptions.title, text: `${chartDataPayload.pair_name} - EMA GR1/GR2 Strategy` },
                    series: [
                        { name: 'Candlestick', type: 'candlestick', data: chartDataPayload.ohlc || [] },
                        { name: emaLabel, type: 'line', data: chartDataPayload.ema_series || [] }
                    ],
                    annotations: { 
                        yaxis: chartDataPayload.annotations_yaxis || [], 
                        points: chartDataPayload.annotations_points || [] 
                    },
                    colors: ['#FEB019', '#008FFB'] // Warna default, bisa diatur lebih spesifik
                };
                
                if (!activeChart) {
                    activeChart = new ApexCharts(document.querySelector("#chart"), newChartOptions);
                    activeChart.render();
                } else {
                    activeChart.updateOptions(newChartOptions);
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
                    series: [{ name: 'Candlestick', type: 'candlestick', data: [] }, { name: 'EMA', type: 'line', data: [] }], 
                    noData: { text: `Gagal memuat data: ${error.message}` } 
                };
                activeChart = new ApexCharts(document.querySelector("#chart"), errorChartOpts);
                activeChart.render();
                document.getElementById('lastUpdatedLabel').textContent = "Error update";
                document.getElementById('strategyStateLabel').textContent = "Status: Error Update";
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
            }, 15000); // Refresh chart lebih sering, misal 15 detik
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
        data_manager_view = shared_crypto_data_manager.copy()
    
    active_pairs_info = []
    for pair_identifier, pair_data_item in data_manager_view.items():
        config_item = pair_data_item.get("config", {})
        if config_item.get("enabled", True):
             active_pairs_info.append({
                "id": pair_identifier, 
                "name": config_item.get('pair_name', pair_identifier)
            })
    return jsonify(active_pairs_info)


@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request):
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager:
             return jsonify({"error": f"Data untuk pair {pair_id_from_request} tidak ditemukan di server."}), 404
        
        # Deep copy untuk thread safety saat memproses
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))

    if not pair_data_snapshot: # Jika pair_id ada tapi datanya kosong (misal setelah gagal fetch awal)
        return jsonify({
            "error": f"Data untuk pair {pair_id_from_request} kosong (snapshot kosong).",
            "ohlc": [], "ema_series": [], "annotations_yaxis": [], "annotations_points": [],
            "pair_name": pair_id_from_request, "last_updated_tv": None,
            "strategy_state_info": get_initial_strategy_state() # Kirim state default
        }), 200 # Kirim 200 agar frontend bisa handle sebagai "tidak ada data"

    # Buat struktur sementara yang diharapkan oleh prepare_chart_data_for_pair
    temp_data_manager_for_prep = {pair_id_from_request: pair_data_snapshot}
    
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_data_manager_for_prep)

    if not prepared_data: # Jika prepare_chart_data_for_pair return None
         return jsonify({
            "error": f"Gagal memproses data chart untuk {pair_id_from_request}.",
            "ohlc": [], "ema_series": [], "annotations_yaxis": [], "annotations_points": [],
            "pair_name": pair_id_from_request, "last_updated_tv": None,
            "strategy_state_info": get_initial_strategy_state()
        }), 500

    # Jika prepare_chart_data_for_pair berhasil tapi tidak ada ohlc (misal histori pendek)
    if not prepared_data.get("ohlc"):
        log_warning(f"Tidak ada data OHLC yang cukup untuk ditampilkan untuk {pair_id_from_request} setelah diproses.", pair_name="SYSTEM_CHART")
        # Kirim data parsial agar frontend bisa menampilkan judul dan status "tidak ada data"
        return jsonify({
            "error": f"Tidak ada data candle yang cukup untuk ditampilkan untuk {pair_id_from_request} (setelah proses).",
            **prepared_data # Kirim apa pun yang berhasil disiapkan
        }), 200 
    
    return jsonify(prepared_data)


def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001 (atau IP Termux-mu)", pair_name="SYSTEM_CHART")
    try:
        # Menonaktifkan logging standar Flask karena kita punya logging sendiri
        flask_log = logging.getLogger('werkzeug')
        flask_log.setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask:
        log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
        log_exception("Traceback Error Flask Server:", pair_name="SYSTEM_CHART")

# CHART_INTEGRATION_END


# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
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

    animated_text_display("=========== EMA GR1/GR2 STRATEGY START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = ("..." + current_key_display_val[-3:]) if len(current_key_display_val) > 8 else current_key_display_val
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {}

    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        local_crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(), # Ini sudah berisi state untuk EMA GR1/GR2
            "big_data_collection_phase_active": True,
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min,
            "data_fetch_failed_consecutively": 0
        }
        
        with lock_ref: # Update shared manager untuk chart
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])

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
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle, EMA {config.get('ema_length',200)}) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True
            except APIKeyError:
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e:
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key, akan coba lagi nanti jika loop utama mengizinkan.", pair_name=config['pair_name'])
                break 
            except Exception as e_gen:
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break
        
        if not initial_candles and not initial_fetch_successful :
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Tandai gagal agar tidak terus mencoba big data
            local_crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Update waktu agar tidak langsung dicoba lagi
            with lock_ref:
                shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue # Lanjut ke pair berikutnya

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        # Warm-up state dengan data historis (jalankan logic tanpa trading)
        if initial_candles:
            min_len_for_logic = max(2, config.get('ema_length', 200)) # Minimal data untuk EMA dan cross
            if len(initial_candles) >= min_len_for_logic:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state {config['pair_name']}...", pair_name=config['pair_name'])
                
                # Jalankan logic untuk setiap bar historis, tapi hanya update state, jangan trade
                for i in range(min_len_for_logic -1, len(initial_candles) - 1): # -1 karena logic jalan di bar terakhir slice
                    historical_slice = initial_candles[:i+1] # Slice sampai bar ke-i (inklusif)
                    if len(historical_slice) < min_len_for_logic: continue

                    # Buat temporary state, pastikan tidak ada posisi
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 
                    # Reset state lain yang mungkin terpengaruh oleh "trade" palsu
                    temp_state_for_warmup["entry_price_custom"] = None
                    temp_state_for_warmup["emergency_sl_level_custom"] = None
                    temp_state_for_warmup["highest_price_for_trailing"] = None
                    temp_state_for_warmup["trailing_tp_active_custom"] = False
                    temp_state_for_warmup["current_trailing_stop_level"] = None
                    # GR1/GR2 akan di-update oleh logic itu sendiri
                    
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict
                    )
                    
                    # Pastikan tidak ada posisi yang "terbuka" selama warm-up
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        log_debug(f"Warm-up: Posisi terdeteksi @ {historical_slice[-1]['timestamp']}, direset.", pair_name=config['pair_name'])
                        local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0
                        local_crypto_data_manager[pair_id]["strategy_state"]["entry_price_custom"] = None
                        # Reset state GR juga jika entry terpicu, agar siap untuk data live
                        local_crypto_data_manager[pair_id]["strategy_state"]["inGetReady2"] = False


                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) untuk {config['pair_name']} tidak cukup untuk warm-up EMA (min: {min_len_for_logic}). State mungkin tidak optimal.", pair_name=config['pair_name'])
        else:
            log_warning(f"Tidak ada data awal untuk warm-up state {config['pair_name']}.", pair_name=config['pair_name'])

        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']} setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(
                    f"Data Downloading Complete: {config['pair_name']}", 
                    f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", 
                    {**config, 'pair_name': config['pair_name']}
                )
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(local_crypto_data_manager[pair_id]['all_candles_list'])} candles) untuk {config['pair_name']} ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        with lock_ref: # Update shared manager untuk chart setelah warm-up
            shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])

    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                # Cooldown jika semua key gagal untuk pair ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        log_debug(f"Pair {pair_name_for_log} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name_for_log)
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600) # Perhitungkan cooldown
                        continue
                    else: # Cooldown selesai
                        data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal
                        log_info(f"Cooldown 1 jam untuk {pair_name_for_log} selesai. Mencoba fetch lagi.", pair_name=pair_name_for_log)


                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()

                required_interval_for_this_pair = 0
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Selama big data, fetch lebih agresif, misal tiap ~1 menit untuk TF menit, ~jam untuk TF jam/hari
                    if config_for_pair.get('timeframe') == "minute": required_interval_for_this_pair = 55 
                    elif config_for_pair.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 2 # Tiap 2 jam untuk TF hari
                    else: required_interval_for_this_pair = 3580 # Hampir 1 jam untuk TF jam
                else: # Setelah big data, gunakan interval dari config
                    required_interval_for_this_pair = config_for_pair.get('refresh_interval_seconds', 60) 

                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Belum waktunya refresh pair ini

                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval_for_this_pair}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time # Update waktu fetch SEKARANG
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])

                if data_per_pair["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA {pair_name_for_log} ({len(data_per_pair['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005, new_line=True)
                else:
                    animated_text_display(f"\n--- ANALISA LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {len(data_per_pair['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005, new_line=True)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                limit_fetch_for_update = 3 # Default untuk update live (ambil beberapa candle terakhir)
                if data_per_pair["big_data_collection_phase_active"]:
                    limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data_per_pair["all_candles_list"])
                    if limit_fetch_needed <=0 : # Harusnya sudah ditangani saat inisialisasi
                         fetch_update_successful_for_this_pair = True # Tidak perlu fetch lagi untuk big data
                         new_candles_batch = [] # Tidak ada candle baru dari fetch
                    else: # Masih butuh candle untuk big data
                        limit_fetch_for_update = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT)
                        limit_fetch_for_update = max(limit_fetch_for_update, 1) # Minimal 1

                if limit_fetch_for_update > 0 or data_per_pair["big_data_collection_phase_active"]: # Hanya fetch jika perlu
                    max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    retries_done_for_this_pair_update = 0
                    original_api_key_index_for_this_fetch = api_key_manager.get_current_key_index() # Simpan index awal untuk fetch ini

                    while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                        current_api_key_for_attempt = api_key_manager.get_current_key()
                        if not current_api_key_for_attempt:
                            log_error(f"Semua API key habis (global) saat mencoba mengambil update untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            break # Keluar dari loop retry untuk pair ini

                        log_info(f"Mengambil {limit_fetch_for_update} candle untuk {pair_name_for_log} (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(
                                config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_for_update, 
                                config_for_pair['exchange'], current_api_key_for_attempt, config_for_pair['timeframe'],
                                pair_name=pair_name_for_log
                            )
                            fetch_update_successful_for_this_pair = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter gagal jika berhasil
                            any_data_fetched_this_cycle = True # Tandai bahwa ada fetch berhasil di siklus ini
                            if api_key_manager.get_current_key_index() != original_api_key_index_for_this_fetch :
                                log_info(f"Fetch berhasil dengan key index {api_key_manager.get_current_key_index()} setelah retry untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                        
                        except APIKeyError:
                            log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name_for_log}. Mencoba key berikutnya.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            
                            if not api_key_manager.switch_to_next_key(): # Coba ganti key global
                                log_error(f"Tidak ada lagi API key tersedia (global) setelah kegagalan pada {pair_name_for_log}.", pair_name=pair_name_for_log)
                                break # Keluar dari loop retry untuk pair ini
                            retries_done_for_this_pair_update += 1 

                        except requests.exceptions.RequestException as e_req:
                            log_error(f"Error jaringan saat mengambil update {pair_name_for_log}: {e_req}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            break # Keluar dari loop retry untuk pair ini, coba lagi di siklus berikutnya
                        except Exception as e_gen_update:
                            log_error(f"Error umum saat mengambil update {pair_name_for_log}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name_for_log)
                            log_exception("Traceback Error Update Fetch:", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] = data_per_pair.get("data_fetch_failed_consecutively", 0) + 1
                            break # Keluar dari loop retry

                # Jika semua key gagal untuk pair ini di siklus ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() # Catat waktu agar bisa cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name_for_log} di siklus ini. Akan masuk cooldown.", pair_name=pair_name_for_log)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data_per_pair["big_data_collection_phase_active"]:
                        log_info(f"Tidak ada data candle baru diterima untuk {pair_name_for_log} (fetch dianggap berhasil tapi batch kosong).", pair_name=pair_name_for_log)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    with lock_ref: # Update shared manager (misal, last_candle_fetch_time, data_fetch_failed_consecutively)
                        shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                    continue # Lanjut ke pair berikutnya jika fetch gagal atau tidak ada candle baru


                # Merge candle baru ke list yang ada
                merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Jika timestamp sama tapi konten beda (misal candle belum final)
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1

                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                data_per_pair["all_candles_list"] = all_candles_list_temp

                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate untuk {pair_name_for_log}. Total: {len(data_per_pair['all_candles_list'])}.", pair_name=pair_name_for_log)
                elif new_candles_batch : # Jika fetch mengembalikan candle tapi tidak ada yang baru/berbeda
                     log_info(f"Tidak ada candle dengan timestamp baru atau konten berbeda untuk {pair_name_for_log}. Data terakhir mungkin identik.", pair_name=pair_name_for_log)

                # Handle Big Data Collection
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Trim jika kelebihan
                            data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data_per_pair["big_data_email_sent"]:
                            send_email_notification(
                                f"Data Downloading Complete: {pair_name_for_log}", 
                                f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name_for_log}.", 
                                {**config_for_pair, 'pair_name': pair_name_for_log}
                            )
                            data_per_pair["big_data_email_sent"] = True
                        
                        data_per_pair["big_data_collection_phase_active"] = False # Selesai big data
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) # Kurangi counter
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data_per_pair['all_candles_list'])} candles) untuk {pair_name_for_log} ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                else: # Jika sudah live, pastikan tidak melebihi target (misal karena fetch awal yang banyak)
                    if len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]

                # Jalankan Logika Strategi
                min_len_for_logic_run = max(2, config_for_pair.get('ema_length', 200))
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run:
                    # Proses logic jika: ada candle baru/update ATAU baru selesai big data ATAU big data & ada candle baru
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data_per_pair["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or # Transisi dari big data ke live
                                         (data_per_pair["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) ) # Masih big data tapi ada tambahan

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi EMA GR1/GR2 untuk {pair_name_for_log} dengan {len(data_per_pair['all_candles_list'])} candle...", pair_name=pair_name_for_log)
                         data_per_pair["strategy_state"] = run_strategy_logic(
                             data_per_pair["all_candles_list"], 
                             config_for_pair, 
                             data_per_pair["strategy_state"],
                             global_settings_dict
                         )
                    elif not data_per_pair["big_data_collection_phase_active"]: # Jika sudah live dan tidak ada candle baru
                         last_c_time_str = data_per_pair["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data_per_pair["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses (strategi) untuk {pair_name_for_log}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name_for_log)
                else:
                    log_info(f"Data ({len(data_per_pair['all_candles_list'])}) untuk {pair_name_for_log} belum cukup utk analisa EMA (min: {min_len_for_logic_run}).", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
            
                with lock_ref: # Update shared manager untuk chart
                    shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)

            # --- Penentuan Waktu Tidur ---
            sleep_duration = 15 # Default sleep jika tidak ada info lain

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: # Semua key gagal global
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch di siklus ini. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600
            elif active_cryptos_still_in_big_data_collection > 0:
                # Jika masih ada yang big data, sleep interval terpendek dari yang big data
                min_big_data_interval_for_sleep = float('inf')
                for _pid_loop, pdata_loop_item in local_crypto_data_manager.items():
                    if pdata_loop_item["big_data_collection_phase_active"]:
                        pconfig_loop = pdata_loop_item["config"]
                        # Interval saat big data
                        interval_bd_sleep = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 2 if pconfig_loop.get('timeframe') == "day" else 3580)
                        min_big_data_interval_for_sleep = min(min_big_data_interval_for_sleep, interval_bd_sleep)
                
                # Ambil nilai minimal antara interval big data dan interval refresh normal terpendek
                effective_min_refresh = min_overall_next_refresh_seconds if min_overall_next_refresh_seconds != float('inf') else float('inf')
                sleep_duration = min(min_big_data_interval_for_sleep if min_big_data_interval_for_sleep != float('inf') else 30, effective_min_refresh if effective_min_refresh > 0 else 30, 30) 
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep ~{sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua sudah live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya (min_next_refresh: {min_overall_next_refresh_seconds:.0f}s).", pair_name="SYSTEM")
                else: # Fallback jika min_overall_next_refresh_seconds tidak valid
                    default_refresh_from_first_config = 60
                    if all_crypto_configs :
                        default_refresh_from_first_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_first_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama setelah live).", pair_name="SYSTEM")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s hingga siklus berikutnya...")
            else: # Jika sleep_duration 0 atau negatif, beri jeda minimal
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
    settings = load_settings() # Memuat settings di awal

    # Jalankan Flask server di thread terpisah
    flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
    flask_thread.start()

    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (EMA GR1/GR2) =========", color=AnsiColors.HEADER, delay=0.005)

        pick_title_main = ""
        active_configs_list = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs_list:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs_list)}) ---\n"
            for i, cfg_item in enumerate(active_configs_list):
                pick_title_main += f"  {i+1}. {cfg_item.get('symbol','N/A')}-{cfg_item.get('currency','N/A')} (TF: {cfg_item.get('timeframe','N/A')}, Exch: {cfg_item.get('exchange','N/A')})\n"
        else:
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s_main = settings.get("api_settings", {})
        primary_key_display_main = api_s_main.get('primary_key', 'BELUM DIATUR')
        if primary_key_display_main and len(primary_key_display_main) > 10 and primary_key_display_main not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display_main = "..." + primary_key_display_main[-5:]
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
        except Exception as e_pick_main:
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main)
            for idx_main, opt_text_main in enumerate(main_menu_options_plain):
                print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main_input = input("Pilih nomor opsi: ").strip()
                if not choice_main_input: continue
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
        
        if selected_main_index == 0: # Mulai Trading
            # Pastikan settings yang terbaru digunakan
            settings = load_settings() # Reload settings terbaru sebelum memulai
            start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif selected_main_index == 1: # Pengaturan
            settings = settings_menu(settings) # settings_menu akan menyimpan jika ada perubahan
        elif selected_main_index == 2: # Keluar
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA, new_line=True)
            show_spinner(0.5, "Exiting")
            # Tidak perlu sys.exit() eksplisit jika Flask thread adalah daemon
            break


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
        sys.stdout.flush()
        sys.stderr.flush()
        # os._exit(0) # Bisa digunakan untuk memastikan semua thread child (termasuk Flask jika non-daemon)
