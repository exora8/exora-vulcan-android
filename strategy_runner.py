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

# --- Imports for Live Chart ---
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from threading import Thread
import gevent # Diperlukan untuk async_mode='gevent' di SocketIO

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
    spinner_chars = ['-', '\\', '|', '/'] # Lebih baik untuk terminal non-unicode
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
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") # Clear line
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

logger = logging.getLogger(__name__) # Menggunakan __name__ adalah praktik yang baik
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler
fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(filename)s:%(lineno)d - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# Console Handler
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
CRYPTOCOMPARE_MAX_LIMIT = 1999 # Cryptocompare limit per request
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 # Detik

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
            # ... (notifikasi email untuk switch key)
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            # ... (notifikasi email untuk semua key gagal)
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
        elif sys.platform.startswith('linux') or sys.platform == 'darwin': # Termux is linux-based
            try: # Try termux-vibrate first for subtle notification
                subprocess.run(['termux-vibrate', '-d', '300', '-f'], timeout=0.5, check=False)
            except FileNotFoundError:
                print('\a', end='', flush=True) # Fallback to bell
            except subprocess.TimeoutExpired:
                print('\a', end='', flush=True) # Fallback to bell if vibrate hangs
            except Exception:
                print('\a', end='', flush=True) # Fallback for other errors
        else: # Other OS
            print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")


def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    # Cek jika menggunakan set global
    api_s = settings_for_email.get("api_settings", {}) # Untuk kasus ketika 'settings_for_email' adalah config global
    if not sender_email and api_s.get("enable_global_email_notifications_for_key_switch"):
        sender_email = api_s.get("email_sender_address")
        sender_password = api_s.get("email_sender_app_password")
        # receiver_email untuk notif global (seperti key switch) harusnya admin
        # jika ini dipanggil dari config pair, receiver_email dari pair config sudah pas

    pair_name_ctx = settings_for_email.get('pair_name',
                                           settings_for_email.get('symbol', 'GLOBAL_EMAIL'))

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
                       check=False, # Don't raise exception on non-zero exit
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal dan paket termux-api sudah diinstal di Termux (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN --- (Tidak banyak diubah, kecuali inisialisasi 'id' jika tidak ada)

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
        "enable_termux_notifications": False
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
                
                # Pastikan setiap config crypto punya 'id'
                for crypto_cfg in settings["cryptos"]:
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True # default enabled
                return settings
            except json.JSONDecodeError:
                log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
                new_settings = {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
                save_settings(new_settings) # Simpan struktur default jika file korup
                return new_settings
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy()
    # Jika ini config baru, pastikan ada ID
    if "id" not in new_config or not new_config["id"]:
        new_config["id"] = str(uuid.uuid4())

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
    # ... (Fungsi settings_menu tetap sama, tidak perlu diubah untuk charting)
    # Pastikan saja ia mengembalikan `current_settings` yang sudah diupdate.
    # Kode fungsi settings_menu Anda yang panjang ada di sini...
    # Untuk brevity, saya akan singkat, tapi di implementasi penuh, kode Anda akan ada di sini
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
        
        try:
            option_text, index = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick: # Fallback jika pick gagal
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings)
            for idx, opt_text in enumerate(selectable_options):
                print(f"  {idx + 1}. {opt_text}")
            try:
                choice = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice < len(selectable_options):
                    index = choice # Ini adalah index yang benar untuk if/elif di bawah
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue

        action_choice = index # index dari pick() atau input manual

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
                    api_s['recovery_keys'] = current_recovery # Pastikan list bersih dari None atau string kosong

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
                    except Exception as e_pick_rec: # Fallback
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
                            if not idx_del_str: # Handle empty input
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
            
            elif action_choice == 3: # Aktifkan/Nonaktifkan Notifikasi Termux
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
                else: # Input tidak valid
                    print(f"{AnsiColors.ORANGE}Input tidak valid. Status Notifikasi Termux tidak berubah: {current_status}.{AnsiColors.ENDC}")

                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(2, "Menyimpan & Kembali...")
            elif action_choice == 4: # Tambah Konfigurasi Crypto Baru
                new_crypto_conf = get_default_crypto_config() # ID sudah dibuat di sini
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf)
                current_settings.setdefault("cryptos", []).append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 5: # Ubah Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); 
                    continue
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                    print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")
                
                idx_choice_str = input("Nomor konfigurasi crypto yang akan diubah: ").strip()
                if not idx_choice_str: # Handle empty input
                    print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue
                try:
                    idx_choice = int(idx_choice_str) - 1
                    if 0 <= idx_choice < len(current_settings["cryptos"]):
                        current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                        save_settings(current_settings)
                        log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                    else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                except ValueError:
                     print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 6: # Hapus Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali...");
                    continue
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                    print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")

                idx_choice_str = input("Nomor konfigurasi crypto yang akan dihapus: ").strip()
                if not idx_choice_str: # Handle empty input
                    print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue
                try:
                    idx_choice = int(idx_choice_str) - 1
                    if 0 <= idx_choice < len(current_settings["cryptos"]):
                        removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                        current_settings["cryptos"].pop(idx_choice)
                        save_settings(current_settings)
                        log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                    else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                except ValueError:
                    print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 7: # Kembali ke Menu Utama
                break
        except ValueError: # General ValueError for the menu's int conversions
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e: # General error catch for safety
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:")
            show_spinner(1.5, "Error, kembali...")

    return current_settings
# --- FUNGSI PENGAMBILAN DATA --- (Tidak banyak diubah)

def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = "histohour" # default
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    # Construct base URL; CCCAGG is default, no explicit exchange param needed if it's CCCAGG
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10 # Simple heuristic for logging/progress bar

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
    
    # Progress bar untuk fetch besar
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : # Hanya tampilkan jika multi-request
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # Jika kita sudah punya data dan butuh lebih, ambil N+1 lalu buang yang overlap
        # Ini membantu memastikan kita tidak terjebak jika 'toTs' ada di tengah candle.
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        if current_to_ts is not None and candles_still_needed > 1 : # Jika bukan request pertama & butuh >1
            # Minta N+1 untuk overlap check, tapi jangan melebihi MAX_LIMIT
            limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)


        if limit_for_this_api_call <= 0: break # Sudah cukup atau tidak butuh lagi

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": # Hanya tambahkan 'e' jika bukan CCCAGG
            params["e"] = exchange_name
        if current_to_ts is not None: # Untuk pagination mundur
            params["toTs"] = current_to_ts

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Log detail jika fetch besar
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20) # Timeout 20 detik

            # Check for API key related HTTP errors first
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}
                try: # Coba parse JSON error jika ada
                    error_data = response.json()
                except json.JSONDecodeError:
                    pass # Tidak masalah jika tidak ada body JSON
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() # Raise HTTPError untuk status 4xx/5xx lainnya
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                # Daftar keyword yang lebih spesifik untuk error terkait API key dari JSON response
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit" # Tambahan dari skrip Anda
                ]
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    # Error API lain yang tidak terkait key, misal pair tidak ada
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Hentikan loop fetch untuk pair ini jika error non-key

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break # Tidak ada data atau format salah

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: # API mengembalikan list kosong
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break
            
            # Konversi ke format yang kita inginkan
            batch_candles_list = []
            for item in raw_candles_from_api:
                # Pastikan semua field ada, jika tidak, candle mungkin tidak valid
                if all(k in item for k in ['time', 'open', 'high', 'low', 'close']):
                    candle = {
                        'timestamp': datetime.fromtimestamp(item['time']),
                        'open': item.get('open'), 'high': item.get('high'),
                        'low': item.get('low'), 'close': item.get('close'),
                        'volume': item.get('volumefrom') # atau 'volume' tergantung API
                    }
                    batch_candles_list.append(candle)
                else:
                    if is_large_fetch: log_warning(f"Skipping malformed candle: {item}", pair_name=pair_name)

            # Hapus candle overlap jika bukan request pertama dan ada data baru
            # Candle terakhir dari batch_candles_list (yang paling baru) harusnya yang paling tua dari API (krn toTs)
            # Candle pertama dari all_accumulated_candles (yang paling tua)
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                # API mengembalikan data dari [TimeFrom, TimeTo], inklusif.
                # Jika toTs dari request sebelumnya adalah timestamp candle pertama dari batch baru ini,
                # berarti candle itu sudah kita proses.
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop() # Hapus yang paling baru dari batch ini (yang sebenarnya paling tua dari API)

            # Jika setelah overlap removal, batch jadi kosong, berarti sudah tidak ada data baru lagi
            if not batch_candles_list and current_to_ts is not None : # Hanya jika bukan request pertama
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break

            # Prepend batch baru ke list utama (karena kita fetch mundur)
            all_accumulated_candles = batch_candles_list + all_accumulated_candles 

            # Update toTs untuk request berikutnya (candle paling tua di batch ini)
            # Pastikan raw_candles_from_api tidak kosong sebelum akses index 0
            if raw_candles_from_api: # Seharusnya selalu ada jika batch_candles_list ada isinya
                current_to_ts = raw_candles_from_api[0]['time'] # timestamp dari candle paling tua di batch ini
            else: # Jika raw_candles_from_api kosong tapi loop belum break (misal batch_candles_list masih ada), ini aneh
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): # Update progress bar tiap 2 request atau jika sudah selesai
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

            # Jika API mengembalikan data lebih sedikit dari yang diminta, berarti sudah di ujung histori
            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break # Keluar loop, sudah dapat semua yang ada

            if len(all_accumulated_candles) >= total_limit_desired: break # Sudah cukup

            # Delay kecil antar request jika fetch besar untuk menghindari rate limit yang terlalu agresif
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3) # 300ms delay

        except APIKeyError: # Dilempar dari dalam blok try di atas
            raise # Lempar lagi agar ditangani oleh loop pemanggil di start_trading
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Hentikan loop fetch untuk pair ini jika ada error koneksi
        except Exception as e: # Error tak terduga lainnya
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error:", pair_name=pair_name) # Log traceback
            break
    
    # Potong jika data yang didapat lebih banyak dari yang diminta (misal karena pembulatan limit)
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Final update progress bar
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles

# --- LOGIKA STRATEGI --- (Perlu sedikit modifikasi untuk mengirim data ke chart)

def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 0: None, 1: High, -1: Low
        "final_pivot_high_price_confirmed": None,
        "final_pivot_high_time_confirmed": None, # Tambahkan timestamp untuk pivot
        "final_pivot_low_price_confirmed": None,
        "final_pivot_low_time_confirmed": None, # Tambahkan timestamp untuk pivot
        "high_price_for_fib": None,
        "high_bar_index_for_fib": None,
        "active_fib_level": None,
        "active_fib_line_start_index": None,
        "active_fib_line_start_time": None, # Tambahkan timestamp untuk FIB start
        "entry_price_custom": None,
        "entry_time": None, # Tambahkan timestamp untuk entry
        "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None,
        "position_size": 0, # 0: No position, 1: Long
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1:
        return pivots # Tidak cukup data

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue # Skip jika data tengah kosong

        # Cek kiri
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break # Data kiri kosong
            if is_high: # Pivot High
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # Pivot Low
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue # Lanjut ke candle berikutnya jika bukan pivot dari sisi kiri

        # Cek kanan
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break # Data kanan kosong
            if is_high: # Pivot High
                # Untuk pivot high yang sedang terbentuk, candle di kanan *harus* lebih rendah
                # Jika right_strength = 0, ini akan jadi simple highest()
                if series_list[i] < series_list[i+j]: is_pivot = False; break
            else: # Pivot Low
                if series_list[i] > series_list[i+j]: is_pivot = False; break
        
        if is_pivot:
            pivots[i] = series_list[i]
            
    return pivots


# Variabel global untuk menyimpan data chart yang akan dikirim
# Ini akan diupdate oleh run_strategy_logic dan dibaca oleh fungsi WebSocket
# Strukturnya: {'pair_id': chart_event_data_object}
# Kita tidak akan menggunakan ini secara global, tapi akan dikembalikan oleh run_strategy_logic

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    
    # Data yang akan dikirim ke chart untuk update ini
    chart_event_data = {
        'pair_id': pair_name, # Untuk identifikasi di frontend jika ada multiple charts
        'candle': None,
        'pivot_high': None, 'pivot_low': None,
        'fib_level': None, 'entry': None, 'stop_loss': None,
        'exit_signal': None, # Untuk menandai exit
        'clear_fib': False, 'clear_trade_markers': False,
        'current_sl_type': None # 'emergency' atau 'trailing'
    }

    # Reset sinyal transient di strategy_state (yang hanya berlaku per-candle)
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_high_time_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None
    strategy_state["final_pivot_low_time_confirmed"] = None
    # Jangan reset active_fib_level di sini, reset saat pivot high baru atau saat entry

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state, chart_event_data # Kembalikan juga chart_event_data kosong

    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]

    # Pivot dihitung pada semua data historis yang tersedia
    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    current_bar_index_in_list = len(candles_history) - 1
    if current_bar_index_in_list < 0 : return strategy_state, chart_event_data

    # Pivot High/Low dikonfirmasi setelah `right_strength` bar berlalu
    # Jadi, jika pivot terjadi di bar `i`, ia dikonfirmasi di bar `i + right_strength`
    # Saat kita berada di `current_bar_index_in_list`, pivot yang bisa dikonfirmasi adalah yang terjadi di `current_bar_index_in_list - right_strength`
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength

    # Ambil harga pivot dari array `raw_pivot_highs/lows` pada index di mana pivot tersebut terjadi
    raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high] if 0 <= idx_pivot_event_high < len(raw_pivot_highs) else None
    raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low] if 0 <= idx_pivot_event_low < len(raw_pivot_lows) else None

    current_candle = candles_history[current_bar_index_in_list]
    # Kirim candle terbaru ke chart
    chart_event_data['candle'] = {
        'time': int(current_candle['timestamp'].timestamp()), # Lightweight Charts butuh epoch seconds
        'open': current_candle['open'],
        'high': current_candle['high'],
        'low': current_candle['low'],
        'close': current_candle['close']
    }

    # Deteksi Pivot High
    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["final_pivot_high_time_confirmed"] = candles_history[idx_pivot_event_high]['timestamp']
        strategy_state["last_signal_type"] = 1
        
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {strategy_state['final_pivot_high_time_confirmed'].strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)
        chart_event_data['pivot_high'] = {'time': int(strategy_state['final_pivot_high_time_confirmed'].timestamp()), 'price': strategy_state['final_pivot_high_price_confirmed']}
        
        # Jika ada pivot high baru, ini menjadi kandidat H untuk FIB
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high # index di `candles_history`

        # Reset FIB aktif jika ada pivot high baru (karena H berubah)
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None
            strategy_state["active_fib_line_start_time"] = None
            chart_event_data['clear_fib'] = True


    # Deteksi Pivot Low
    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["final_pivot_low_time_confirmed"] = candles_history[idx_pivot_event_low]['timestamp']
        strategy_state["last_signal_type"] = -1

        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {strategy_state['final_pivot_low_time_confirmed'].strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)
        chart_event_data['pivot_low'] = {'time': int(strategy_state['final_pivot_low_time_confirmed'].timestamp()), 'price': strategy_state['final_pivot_low_price_confirmed']}

        # Jika ada pivot low baru DAN kita sudah punya H dari pivot high sebelumnya
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low # index di `candles_history`

            # Pastikan Low terjadi SETELAH High
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                if strategy_state["high_price_for_fib"] is None or current_low_price is None:
                     log_warning("Harga untuk kalkulasi FIB tidak valid (None).", pair_name=pair_name)
                else:
                    calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0
                    is_fib_late = False
                    if crypto_config["enable_secure_fib"]:
                        price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                        if price_val_current_candle is not None and calculated_fib_level is not None and price_val_current_candle > calculated_fib_level:
                            is_fib_late = True
                    
                    if is_fib_late:
                        log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                        strategy_state["active_fib_level"] = None 
                        strategy_state["active_fib_line_start_index"] = None
                        strategy_state["active_fib_line_start_time"] = None
                        chart_event_data['clear_fib'] = True # Pastikan FIB lama dihapus jika ada
                    elif calculated_fib_level is not None : 
                        log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=pair_name)
                        strategy_state["active_fib_level"] = calculated_fib_level
                        strategy_state["active_fib_line_start_index"] = current_low_bar_index
                        strategy_state["active_fib_line_start_time"] = candles_history[current_low_bar_index]['timestamp']
                        chart_event_data['fib_level'] = {'price': strategy_state["active_fib_level"], 'start_time': int(strategy_state["active_fib_line_start_time"].timestamp())}
                
                # Setelah FIB dihitung (atau gagal), reset H agar tidak digunakan lagi sampai ada Pivot High baru
                strategy_state["high_price_for_fib"] = None
                strategy_state["high_bar_index_for_fib"] = None


    # Cek Entry
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        if current_candle.get('close') is None or current_candle.get('open') is None:
            log_warning("Nilai close atau open tidak ada di candle saat ini. Skip entry check.", pair_name=pair_name)
        else:
            is_bullish_candle = current_candle['close'] > current_candle['open']
            is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

            if is_bullish_candle and is_closed_above_fib:
                if strategy_state["position_size"] == 0: # Jika belum ada posisi
                    strategy_state["position_size"] = 1 # Masuk posisi Long
                    entry_px = current_candle['close'] 
                    strategy_state["entry_price_custom"] = entry_px
                    strategy_state["entry_time"] = current_candle['timestamp']
                    strategy_state["highest_price_for_trailing"] = entry_px # Inisialisasi untuk trailing
                    strategy_state["trailing_tp_active_custom"] = False 
                    strategy_state["current_trailing_stop_level"] = None

                    emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                    strategy_state["emergency_sl_level_custom"] = emerg_sl

                    log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                    log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                    play_notification_sound()
                    
                    # Data untuk chart
                    chart_event_data['entry'] = {'time': int(strategy_state["entry_time"].timestamp()), 'price': entry_px, 'type': 'buy'}
                    chart_event_data['stop_loss'] = {'price': emerg_sl, 'type': 'emergency'}
                    chart_event_data['current_sl_type'] = 'emergency'


                    termux_title = f"BUY Signal: {pair_name}"
                    termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}"
                    send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)
                    # ... (notifikasi email) ...
                    
                    # Setelah entry, FIB dianggap selesai (tidak aktif lagi untuk entry berikutnya)
                    strategy_state["active_fib_level"] = None
                    strategy_state["active_fib_line_start_index"] = None
                    strategy_state["active_fib_line_start_time"] = None
                    chart_event_data['clear_fib'] = True # Hapus FIB dari chart

    # Manajemen Posisi Aktif (Trailing Stop, Emergency SL)
    if strategy_state["position_size"] > 0: # Jika sedang dalam posisi
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None).", pair_name=pair_name)
        else:
            strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        # Aktivasi Trailing TP
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None: profit_percent = 0.0
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=pair_name)
                # Tidak ada update chart khusus di sini, kecuali SL berubah

        # Update Trailing Stop Level jika Trailing TP aktif
        sl_updated_this_candle = False
        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)
                sl_updated_this_candle = True
        
        # Tentukan SL yang akan digunakan untuk exit (Emergency atau Trailing)
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        current_sl_type_for_chart = 'emergency'

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                current_sl_type_for_chart = 'trailing'
        
        # Kirim update SL ke chart jika ada SL aktif atau baru diupdate
        if final_stop_for_exit is not None:
             chart_event_data['stop_loss'] = {'price': final_stop_for_exit, 'type': current_sl_type_for_chart}
             chart_event_data['current_sl_type'] = current_sl_type_for_chart


        # Cek Exit
        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price_open = current_candle.get('open')
            if exit_price_open is None: exit_price = final_stop_for_exit 
            else: exit_price = min(exit_price_open, final_stop_for_exit) # Keluar di SL atau open candle (mana yg lebih dulu)
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            exit_color = AnsiColors.RED if pnl < 0 else AnsiColors.BLUE # Merah jika rugi, biru jika profit by trailing
            if exit_comment == "Emergency SL": exit_color = AnsiColors.RED # Selalu merah untuk emergency SL

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

            chart_event_data['exit_signal'] = {'time': int(current_candle['timestamp'].timestamp()), 'price': exit_price, 'reason': exit_comment, 'pnl': pnl}
            chart_event_data['clear_trade_markers'] = True # Hapus entry dan SL dari chart

            # ... (notifikasi Termux dan email untuk exit) ...
            termux_title_exit = f"EXIT Signal: {pair_name}"
            termux_content_exit = f"{exit_comment} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)

            # Reset state setelah exit
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["entry_time"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            # Biarkan last_signal_type agar tidak langsung re-entry jika kondisi masih sama
        
        # Jika tidak exit, log status posisi
        elif strategy_state.get("entry_price_custom") is not None :
            entry_price_display = strategy_state.get('entry_price_custom', 0)
            sl_display_str = f'{final_stop_for_exit:.5f} ({current_sl_type_for_chart})' if final_stop_for_exit is not None else 'N/A'
            log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)


    return strategy_state, chart_event_data

# --- LIVE CHART SERVER ---
app_flask = Flask(__name__)
# Ganti secret_key jika perlu, tapi untuk penggunaan lokal sederhana tidak terlalu krusial
app_flask.config['SECRET_KEY'] = os.urandom(24) 
# Gunakan gevent untuk async mode jika tersedia, lebih baik untuk performa I/O bound
# Jika gevent tidak ada, Flask-SocketIO akan fallback ke mode threading.
# Pastikan `pip install gevent`
socketio = SocketIO(app_flask, async_mode='gevent', cors_allowed_origins="*")

# Global dictionary untuk menyimpan data candle dan state strategi per pair
# Ini akan diisi oleh `start_trading`
# Format: crypto_data_store[pair_id_config] = {'candles': [], 'strategy_state': {}}
# Sebenarnya kita akan menggunakan `crypto_data_manager` dari `start_trading` secara langsung.
# Variabel global ini untuk kemudahan akses dari handler Flask.
# Jika `start_trading` berjalan di thread berbeda, perlu cara aman untuk share data.
# Untuk saat ini, asumsikan `start_trading` mengisi variabel yang bisa diakses.
# Kita akan gunakan referensi ke `crypto_data_manager` yang ada di `start_trading`.
# Ini agak tricky jika `start_trading` tidak di-pass ke sini.
# Solusi: Buat `crypto_data_manager` dapat diakses secara global atau pass instance.
# Untuk simplicity, kita akan buat `crypto_data_manager` global di scope main,
# dan diakses dari sini. Ini bukan best practice tapi paling mudah diintegrasikan.
# **Revisi**: Lebih baik `start_trading` memanggil fungsi emit dari SocketIO secara langsung.
# Tidak perlu `shared_crypto_data_manager` global.

# HTML Template untuk chart
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Crypto Chart</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body, html { margin: 0; padding: 0; width: 100%; height: 100%; background-color: #131722; color: #d1d4dc;}
        #chartContainer { width: 98%; height: 550px; margin: 10px auto; }
        #controls { text-align: center; padding: 10px; background-color: #1e222d; }
        label, select, button { margin: 0 5px; color: #d1d4dc; background-color: #2a2e39; border: 1px solid #4E5C6E; padding: 5px;}
        #log { width: 98%; height: 100px; margin: 10px auto; overflow-y: scroll; border: 1px solid #333; padding: 5px; font-family: monospace; font-size: 0.8em; background-color: #1e222d;}
    </style>
</head>
<body>
    <div id="controls">
        <label for="pairSelector">Select Pair:</label>
        <select id="pairSelector"></select>
        <button id="loadPairButton">Load/Refresh Pair</button>
        <span id="connectionStatus" style="margin-left: 20px;">Disconnected</span>
    </div>
    <div id="chartContainer"></div>
    <div id="log"></div>

    <script>
        const chartContainer = document.getElementById('chartContainer');
        const logDiv = document.getElementById('log');
        const pairSelector = document.getElementById('pairSelector');
        const loadPairButton = document.getElementById('loadPairButton');
        const connectionStatusSpan = document.getElementById('connectionStatus');
        
        let chart = null;
        let candlestickSeries = null;
        let fibLine = null; // Untuk PriceLine Fibonacci
        let entryMarkerSeries = null; // Series terpisah untuk marker agar mudah di-clear
        let slLine = null;    // Untuk PriceLine Stop Loss

        let currentChartPairId = null;

        function addLog(message) {
            const time = new Date().toLocaleTimeString();
            logDiv.innerHTML = `[${time}] ${message}<br>` + logDiv.innerHTML;
            if (logDiv.children.length > 50) { // Batasi jumlah log
                logDiv.removeChild(logDiv.lastChild);
            }
        }

        function createOrUpdateChart(pairId) {
            if (chart) {
                chart.remove(); // Hapus chart lama jika ada
            }
            chart = LightweightCharts.createChart(chartContainer, {
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
                layout: {
                    backgroundColor: '#131722',
                    textColor: '#d1d4dc',
                },
                grid: {
                    vertLines: { color: '#334158' },
                    horzLines: { color: '#334158' },
                },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                priceScale: { borderColor: '#485c7b' },
                timeScale: { borderColor: '#485c7b', timeVisible: true, secondsVisible: false },
            });
            candlestickSeries = chart.addCandlestickSeries({
                upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            });
            entryMarkerSeries = chart.addCandlestickSeries({}); // Dummy series for markers
            currentChartPairId = pairId;
            addLog(`Chart created/reset for ${pairId}. Requesting initial data...`);
            socket.emit('request_initial_data', { pair_id: pairId });
        }
        
        const socket = io();

        socket.on('connect', () => {
            addLog('Connected to backend.');
            connectionStatusSpan.textContent = 'Connected';
            connectionStatusSpan.style.color = 'lightgreen';
            socket.emit('request_available_pairs'); // Minta daftar pair yang tersedia
        });

        socket.on('disconnect', () => {
            addLog('Disconnected from backend.');
            connectionStatusSpan.textContent = 'Disconnected';
            connectionStatusSpan.style.color = 'red';
        });

        socket.on('available_pairs', (data) => {
            addLog(`Available pairs received: ${data.pairs.join(', ')}`);
            pairSelector.innerHTML = ''; // Clear existing options
            if (data.pairs && data.pairs.length > 0) {
                data.pairs.forEach(pairId => {
                    const option = document.createElement('option');
                    option.value = pairId;
                    option.textContent = pairId;
                    pairSelector.appendChild(option);
                });
                // Otomatis load pair pertama jika belum ada yang dipilih
                if (!currentChartPairId && pairSelector.options.length > 0) {
                    pairSelector.value = pairSelector.options[0].value; // Pilih yang pertama
                    createOrUpdateChart(pairSelector.value);
                }
            } else {
                addLog("No active pairs found from backend.");
            }
        });

        socket.on('initial_chart_data', (data) => {
            if (!chart || !candlestickSeries || data.pair_id !== currentChartPairId) {
                if (data.pair_id !== currentChartPairId) {
                    addLog(`Received initial data for ${data.pair_id}, but current chart is for ${currentChartPairId}. Ignoring.`);
                } else {
                    addLog("Chart not ready for initial data.");
                }
                return;
            }
            addLog(`Received initial ${data.candles.length} candles for ${data.pair_id}.`);
            candlestickSeries.setData(data.candles);
            
            // Hapus semua marker/line lama sebelum menggambar yang baru dari initial state
            if (fibLine) { fibLine.applyOptions({price: 0}); fibLine = null; } // Hacky remove
            if (slLine) { slLine.applyOptions({price: 0}); slLine = null; }
            entryMarkerSeries.setMarkers([]);


            if (data.strategy_markers) {
                const markers = data.strategy_markers;
                if (markers.pivot_highs && markers.pivot_highs.length > 0) {
                    const phMarkers = markers.pivot_highs.map(m => ({ time: m.time, position: 'aboveBar', color: 'red', shape: 'arrowDown', text: `PH ${m.price.toFixed(2)}`}));
                    candlestickSeries.setMarkers(candlestickSeries.markers().concat(phMarkers)); // Gabung, jangan timpa semua
                }
                if (markers.pivot_lows && markers.pivot_lows.length > 0) {
                     const plMarkers = markers.pivot_lows.map(m => ({ time: m.time, position: 'belowBar', color: 'lime', shape: 'arrowUp', text: `PL ${m.price.toFixed(2)}`}));
                     candlestickSeries.setMarkers(candlestickSeries.markers().concat(plMarkers));
                }
                if (markers.active_fib_level) {
                    if (fibLine) chart.removePriceLine(fibLine);
                    fibLine = candlestickSeries.createPriceLine({ price: markers.active_fib_level.price, color: 'blue', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: `FIB 0.5 ${markers.active_fib_level.price.toFixed(2)}`});
                }
                if (markers.entry) {
                    const entryMarkers = [{ time: markers.entry.time, position: 'belowBar', color: 'green', shape: 'arrowUp', text: `BUY @ ${markers.entry.price.toFixed(2)}`}];
                    entryMarkerSeries.setMarkers(entryMarkers);
                }
                if (markers.stop_loss) {
                    if (slLine) chart.removePriceLine(slLine);
                    slLine = candlestickSeries.createPriceLine({ price: markers.stop_loss.price, color: markers.stop_loss.type === 'trailing' ? 'orange' : 'red', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: `${markers.stop_loss.type.toUpperCase()} SL ${markers.stop_loss.price.toFixed(2)}`});
                }
            }
            chart.timeScale().fitContent();
        });

        socket.on('chart_update', (data) => {
            if (!chart || !candlestickSeries || data.pair_id !== currentChartPairId) return;

            if (data.candle) {
                candlestickSeries.update(data.candle);
            }

            // Handle markers and lines
            if (data.pivot_high) {
                candlestickSeries.setMarkers(candlestickSeries.markers().filter(m => m.shape !== 'arrowDown').concat([{ time: data.pivot_high.time, position: 'aboveBar', color: 'red', shape: 'arrowDown', text: `PH ${data.pivot_high.price.toFixed(2)}` }]));
            }
            if (data.pivot_low) {
                 candlestickSeries.setMarkers(candlestickSeries.markers().filter(m => m.shape !== 'arrowUp' || m.position !== 'belowBar').concat([{ time: data.pivot_low.time, position: 'belowBar', color: 'lime', shape: 'arrowUp', text: `PL ${data.pivot_low.price.toFixed(2)}` }]));
            }

            if (data.clear_fib) {
                if (fibLine) {
                    try { candlestickSeries.removePriceLine(fibLine); } catch(e) { /* ignore if already removed */ }
                    fibLine = null;
                }
            } else if (data.fib_level) {
                if (fibLine) candlestickSeries.removePriceLine(fibLine);
                fibLine = candlestickSeries.createPriceLine({ price: data.fib_level.price, color: 'blue', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: `FIB 0.5 ${data.fib_level.price.toFixed(2)}` });
            }

            if (data.clear_trade_markers) {
                entryMarkerSeries.setMarkers([]);
                if (slLine) {
                    try { candlestickSeries.removePriceLine(slLine); } catch(e) { /* ignore */ }
                    slLine = null;
                }
            }
            
            if (data.entry) {
                entryMarkerSeries.setMarkers([{ time: data.entry.time, position: 'belowBar', color: 'green', shape: 'arrowUp', text: `BUY @ ${data.entry.price.toFixed(2)}` }]);
            }
            
            if (data.stop_loss) {
                if (slLine) candlestickSeries.removePriceLine(slLine);
                slLine = candlestickSeries.createPriceLine({ price: data.stop_loss.price, color: data.current_sl_type === 'trailing' ? 'orange' : 'red', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: `${data.current_sl_type.toUpperCase()} SL ${data.stop_loss.price.toFixed(2)}` });
            }

            if (data.exit_signal) {
                entryMarkerSeries.setMarkers(entryMarkerSeries.markers().concat([{ time: data.exit_signal.time, position: data.exit_signal.pnl >= 0 ? 'aboveBar' : 'aboveBar', color: data.exit_signal.pnl >= 0 ? 'aqua' : 'fuchsia', shape: data.exit_signal.pnl >= 0 ? 'circle' : 'circle', text: `EXIT ${data.exit_signal.reason} ${data.exit_signal.price.toFixed(2)} PNL: ${data.exit_signal.pnl.toFixed(1)}%` }]));
                // Clear SL line on exit
                if (slLine) {
                    try { candlestickSeries.removePriceLine(slLine); } catch(e) { /* ignore */ }
                    slLine = null;
                }
            }
        });

        loadPairButton.addEventListener('click', () => {
            const selectedPairId = pairSelector.value;
            if (selectedPairId) {
                createOrUpdateChart(selectedPairId);
            } else {
                addLog("No pair selected.");
            }
        });
        
        // Resize chart on window resize
        window.addEventListener('resize', () => {
            if (chart) {
                chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
            }
        });

    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app_flask.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Global reference to the main trading data structure, to be set by start_trading
# Ini adalah cara yang kurang ideal, tapi paling mudah untuk integrasi cepat.
# Seharusnya, event SocketIO diemit langsung dari dalam `start_trading` loop.
_g_crypto_data_manager = None
_g_all_crypto_configs = None # Untuk daftar pairs

@socketio.on('connect')
def handle_connect():
    log_info("Chart client connected to WebSocket.", pair_name="CHART_SERVER")
    emit('message', {'data': 'Connected to Python backend!'})

@socketio.on('request_available_pairs')
def handle_request_available_pairs():
    global _g_all_crypto_configs # Akses konfigurasi global
    if _g_all_crypto_configs:
        pair_ids = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in _g_all_crypto_configs if cfg.get("enabled", True)]
        emit('available_pairs', {'pairs': pair_ids})
        log_debug(f"Sent available pairs: {pair_ids}", pair_name="CHART_SERVER")
    else:
        emit('available_pairs', {'pairs': []})
        log_debug("No crypto_configs available to send.", pair_name="CHART_SERVER")


@socketio.on('request_initial_data')
def handle_request_initial_data(message):
    global _g_crypto_data_manager # Akses data manager global
    pair_id_req = message.get('pair_id')
    log_info(f"Chart client requested initial data for {pair_id_req}", pair_name="CHART_SERVER")

    if not _g_crypto_data_manager or not pair_id_req:
        log_warning(f"Cannot provide initial data: crypto_data_manager not set or pair_id missing ({pair_id_req})", pair_name="CHART_SERVER")
        emit('initial_chart_data', {'pair_id': pair_id_req, 'candles': [], 'strategy_markers': {}})
        return

    # Cari data untuk pair_id yang diminta
    # Ingat, key di _g_crypto_data_manager mungkin menyertakan timeframe
    # Kita perlu mencocokkan bagian symbol-currency
    data_for_pair = None
    matched_key_in_manager = None
    for key, data_val in _g_crypto_data_manager.items():
        if key.startswith(pair_id_req + "_"): # Contoh: "BTC-USD_hour"
            data_for_pair = data_val
            matched_key_in_manager = key
            break
    
    if not data_for_pair: # Jika tidak ada timeframe, coba match langsung
         if pair_id_req in _g_crypto_data_manager:
            data_for_pair = _g_crypto_data_manager[pair_id_req]
            matched_key_in_manager = pair_id_req


    if data_for_pair and data_for_pair.get("all_candles_list"):
        candles_to_send = data_for_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
        formatted_candles = []
        for c in candles_to_send:
            if all(k in c for k in ['timestamp', 'open', 'high', 'low', 'close']):
                 formatted_candles.append({
                    'time': int(c['timestamp'].timestamp()),
                    'open': c['open'], 'high': c['high'],
                    'low': c['low'], 'close': c['close']
                })
            else:
                log_warning(f"Skipping malformed candle in initial_data for {pair_id_req}: {c.get('timestamp')}", pair_name="CHART_SERVER")


        # Ambil state strategi saat ini untuk marker awal
        # Ini perlu diekstrak dari `strategy_state` yang ada
        s_state = data_for_pair.get("strategy_state", {})
        initial_markers = {
            'pivot_highs': [], # Ini perlu diisi dari histori jika mau ditampilkan
            'pivot_lows': [],  # Ini perlu diisi dari histori jika mau ditampilkan
            'active_fib_level': None,
            'entry': None,
            'stop_loss': None
        }
        if s_state.get("active_fib_level") and s_state.get("active_fib_line_start_time"):
            initial_markers['active_fib_level'] = {'price': s_state["active_fib_level"], 'start_time': int(s_state["active_fib_line_start_time"].timestamp())}
        
        if s_state.get("position_size", 0) > 0 and s_state.get("entry_price_custom") and s_state.get("entry_time"):
            initial_markers['entry'] = {'time': int(s_state["entry_time"].timestamp()), 'price': s_state["entry_price_custom"]}
            
            sl_price = s_state.get("current_trailing_stop_level") if s_state.get("trailing_tp_active_custom") else s_state.get("emergency_sl_level_custom")
            sl_type = 'trailing' if s_state.get("trailing_tp_active_custom") and s_state.get("current_trailing_stop_level") else 'emergency'
            if sl_price:
                initial_markers['stop_loss'] = {'price': sl_price, 'type': sl_type}

        # Untuk pivot, kita idealnya menyimpan histori pivot yang dikonfirmasi.
        # Skrip saat ini hanya menyimpan yang paling baru di `strategy_state` transient.
        # Untuk kesederhanaan, kita tidak akan mengirim histori pivot saat ini.
        # Hanya pivot yang terdeteksi secara live akan muncul.

        log_info(f"Sending {len(formatted_candles)} initial candles and markers for {pair_id_req}.", pair_name="CHART_SERVER")
        emit('initial_chart_data', {'pair_id': pair_id_req, 'candles': formatted_candles, 'strategy_markers': initial_markers})
    else:
        log_warning(f"No candle data found for {pair_id_req} in crypto_data_manager or key not matched: {matched_key_in_manager}", pair_name="CHART_SERVER")
        emit('initial_chart_data', {'pair_id': pair_id_req, 'candles': [], 'strategy_markers': {}})

@socketio.on('disconnect')
def handle_disconnect():
    log_info("Chart client disconnected.", pair_name="CHART_SERVER")

# Fungsi untuk menjalankan Flask app di thread terpisah
def run_flask_app():
    log_info("Starting Flask-SocketIO server for charts on http://0.0.0.0:5000", pair_name="CHART_SERVER")
    # socketio.run(app_flask, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    # Menggunakan server gevent secara eksplisit
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app_flask, handler_class=WebSocketHandler)
    log_info("gevent WSGI server for Flask-SocketIO is running.", pair_name="CHART_SERVER")
    try:
        server.serve_forever()
    except Exception as e:
        log_error(f"Flask-SocketIO server crashed: {e}", pair_name="CHART_SERVER")
        log_exception("Flask-SocketIO server exception details:", pair_name="CHART_SERVER")


# --- FUNGSI UTAMA TRADING LOOP ---

def start_trading(global_settings_dict):
    global _g_crypto_data_manager, _g_all_crypto_configs # Set variabel global
    
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Pass entire api_settings for global email config
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs_from_settings = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    _g_all_crypto_configs = all_crypto_configs_from_settings # Set untuk chart server

    if not all_crypto_configs_from_settings:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    # ... (log API key) ...

    crypto_data_manager = {} # Ini akan menjadi _g_crypto_data_manager
    _g_crypto_data_manager = crypto_data_manager # Set referensi global

    for config in all_crypto_configs_from_settings:
        # Gunakan ID unik dari config jika ada, atau buat berdasarkan pair & TF
        # ID dari config lebih baik jika user bisa edit/hapus/tambah config
        config_id = config.get("id") # ID ini harusnya sudah ada dari load_settings / _prompt_crypto_config
        if not config_id: # Fallback jika ID tidak ada (seharusnya tidak terjadi)
            config_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
            log_warning(f"Config crypto tidak memiliki ID, menggunakan fallback ID: {config_id}", pair_name=config.get('pair_name', 'UNKNOWN_PAIR'))
        
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}" # Untuk logging

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        # pair_id_for_manager adalah ID unik untuk data manager, bisa pakai config_id
        pair_id_for_manager = config_id 
        # Namun, untuk chart, kita sering menggunakan symbol-currency. Jadi kita perlu keduanya.
        # Key di crypto_data_manager akan menggunakan symbol-currency_timeframe untuk kemudahan debug
        # tapi kita juga simpan config_id di dalamnya.
        # REVISI: Mari gunakan Symbol-Currency_Timeframe sebagai key utama di crypto_data_manager
        # karena ini yang paling sering digunakan untuk logika dan identifikasi.
        # ID dari config.json akan disimpan di dalam value dict.
        
        # Key untuk crypto_data_manager (lebih deskriptif)
        manager_key = f"{config['pair_name']}_{config['timeframe']}"


        crypto_data_manager[manager_key] = {
            "config_id": config_id, # Simpan ID asli dari settings.json
            "config": config, # Seluruh config pair
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, 
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, 
            "data_fetch_failed_consecutively": 0,
            "last_attempt_after_all_keys_failed": datetime.min # Timestamp retry setelah semua key gagal
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
                break # Tidak ada key lagi

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Berhasil fetch, keluar loop retry
            except APIKeyError: # Gagal karena API Key
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break # Tidak ada key lagi, keluar loop retry
                retries_done_initial +=1 # Tambah counter retry
            except requests.exceptions.RequestException as e: # Gagal karena jaringan/koneksi
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key. Coba lagi nanti.", pair_name=config['pair_name'])
                # Tidak increment retries_done_initial di sini, anggap ini masalah sementara,
                # loop utama akan retry nanti. Atau, bisa juga break. Untuk fetch awal, lebih baik break.
                break # Keluar loop retry, masalah koneksi
            except Exception as e_gen: # Error umum lainnya
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break # Keluar loop retry

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[manager_key]["big_data_collection_phase_active"] = False # Tandai agar tidak coba fetch besar lagi
            crypto_data_manager[manager_key]["last_candle_fetch_time"] = datetime.now() # Agar tidak langsung di-fetch di loop utama
            continue # Lanjut ke config pair berikutnya

        crypto_data_manager[manager_key]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])

        # Warm-up strategy state
        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1 
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
                
                # Proses semua candle historis kecuali yang terakhir (yang akan diproses live)
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # -1 agar tidak proses candle terakhir
                    historical_slice = initial_candles[:i+1] # Slice dari awal sampai candle ke-i
                    if len(historical_slice) < min_len_for_pivots: continue # Lewati jika slice terlalu pendek

                    # Saat warm-up, kita tidak mau ada sinyal trade aktual, hanya update state pivot/fib
                    temp_state_for_warmup = crypto_data_manager[manager_key]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Pastikan tidak ada posisi
                    
                    # Panggil run_strategy_logic, tapi abaikan chart_event_data-nya
                    updated_state, _ = run_strategy_logic(historical_slice, config, temp_state_for_warmup, global_settings_dict)
                    crypto_data_manager[manager_key]["strategy_state"] = updated_state
                    
                    # Jika ada "posisi" terbuka selama warm-up (seharusnya tidak jika position_size=0), reset lagi
                    if crypto_data_manager[manager_key]["strategy_state"]["position_size"] > 0: 
                        crypto_data_manager[manager_key]["strategy_state"] = {
                            **crypto_data_manager[manager_key]["strategy_state"], # Ambil state pivot/fib
                            **{"position_size":0, "entry_price_custom":None, "entry_time": None, 
                               "emergency_sl_level_custom":None, "highest_price_for_trailing":None, 
                               "trailing_tp_active_custom":False, "current_trailing_stop_level":None}
                        }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else:
            log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])

        if len(crypto_data_manager[manager_key]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[manager_key]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not crypto_data_manager[manager_key]["big_data_email_sent"]:
                # ... (send email notif) ...
                crypto_data_manager[manager_key]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[manager_key]['all_candles_list'])} candles) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])

    animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            # Iterasi melalui copy dari keys jika kita mungkin memodifikasi dict (meski di sini tidak)
            for manager_key_loop, data in list(crypto_data_manager.items()):
                config = data["config"] # Ambil config dari data yang disimpan
                pair_name = config['pair_name'] # Ini "BTC-USD", dll.
                
                # Cek cooldown jika semua key gagal sebelumnya
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : # +1 untuk toleransi
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name)
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600) # Pertimbangkan cooldown ini
                        continue # Lewati pair ini untuk siklus ini
                    else:
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter setelah cooldown
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()

                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Interval lebih cepat saat kumpulkan big data, tapi tidak terlalu cepat
                    if config.get('timeframe') == "minute": required_interval_for_this_pair = 55 # Hampir tiap menit
                    elif config.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 # Hampir tiap hari
                    else: required_interval_for_this_pair = 3580 # Hampir tiap jam
                else: # Fase live trading
                    required_interval_for_this_pair = config.get('refresh_interval_seconds', 60) 

                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Belum waktunya refresh untuk pair ini

                log_info(f"Memproses {pair_name} ({manager_key_loop})...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time # Catat waktu mulai proses
                num_candles_before_fetch = len(data["all_candles_list"])

                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_for_this_pair_update = 0 # Counter retry untuk siklus ini

                while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt:
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {pair_name}.", pair_name=pair_name)
                        break # Tidak ada key lagi

                    limit_fetch = 3 # Default untuk update live (ambil beberapa candle terakhir)
                    if data["big_data_collection_phase_active"]:
                        limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"])
                        if limit_fetch_needed <=0 : # Seharusnya sudah selesai, tapi jaga-jaga
                             fetch_update_successful_for_this_pair = True # Anggap berhasil (tidak perlu fetch)
                             new_candles_batch = []
                             break
                        limit_fetch = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) # Ambil sebanyak sisa atau max limit
                        limit_fetch = max(limit_fetch, 1) # Minimal 1

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch, 
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter gagal jika berhasil
                        any_data_fetched_this_cycle = True # Tandai ada data di-fetch di siklus ini
                    
                    except APIKeyError: # Gagal karena API Key
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        
                        if not api_key_manager.switch_to_next_key(): # Coba ganti key
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break # Keluar loop retry jika tidak ada key lagi
                        retries_done_for_this_pair_update += 1 # Tambah counter retry

                    except requests.exceptions.RequestException as e: # Gagal karena jaringan
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key. Akan dicoba lagi nanti.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1 # Tetap hitung sbg kegagalan
                        # Tidak break loop retry di sini, biarkan loop utama yang handle sleep
                        break # Keluar dari while loop retry untuk pair ini, akan coba lagi di siklus berikutnya
                    except Exception as e_gen_update: # Error umum lainnya
                        log_error(f"Error umum saat mengambil update {pair_name}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name)
                        log_exception("Traceback Error Update Fetch:", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break # Keluar dari while loop retry

                # Setelah loop retry, cek apakah semua key sudah dicoba dan gagal
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data["last_attempt_after_all_keys_failed"] = datetime.now() # Catat waktu untuk cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data["big_data_collection_phase_active"]:
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {pair_name} meskipun fetch (dianggap) berhasil.{AnsiColors.ENDC}", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) # Jadwalkan refresh berikutnya
                    continue # Lanjut ke pair berikutnya jika fetch gagal atau tidak ada data baru

                # Gabungkan candle baru dengan yang lama, hindari duplikat berdasarkan timestamp
                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 # Untuk candle yang mungkin sama timestamp tapi beda OHLC (misal candle live)

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: # Candle baru
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Candle ada tapi isinya beda (update)
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1

                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                elif new_candles_batch : # Ada batch tapi tidak ada yg baru/update (data identik)
                     log_info("Tidak ada candle dengan timestamp baru atau update konten. Data terakhir mungkin identik.", pair_name=pair_name)


                # Handle fase big data
                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!{AnsiColors.ENDC}", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Pangkas jika kelebihan
                            data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data["big_data_email_sent"]:
                            # ... (send email notif) ...
                            data["big_data_email_sent"] = True
                        
                        data["big_data_collection_phase_active"] = False # Selesai fase big data
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------{AnsiColors.ENDC}", pair_name=pair_name)
                else: # Fase live, pastikan tidak melebihi target
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]

                # Jalankan logika strategi jika ada data baru atau jika baru selesai big data
                min_len_for_pivots = config.get('left_strength',50) + config.get('right_strength',150) + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    # Proses jika ada candle baru/update, ATAU jika baru beralih dari big data ke live,
                    # ATAU jika masih big data tapi ada penambahan candle baru.
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or # Baru selesai big data
                                         (data["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) ) # Masih big data tapi ada progress

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         updated_strategy_state, chart_update_payload = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"], global_settings_dict)
                         data["strategy_state"] = updated_strategy_state
                         
                         # Kirim update ke chart melalui WebSocket
                         # Pastikan chart_update_payload memiliki pair_id yang benar (symbol-currency)
                         # manager_key_loop adalah "BTC-USD_hour", chart butuh "BTC-USD"
                         chart_update_payload['pair_id'] = pair_name # Seharusnya sudah diisi oleh run_strategy_logic
                         if chart_update_payload.get('candle') or any(v for k, v in chart_update_payload.items() if k not in ['pair_id', 'candle'] and v is not None and v != False): # Ada sesuatu untuk dikirim
                            socketio.emit('chart_update', chart_update_payload)
                            log_debug(f"Chart update sent for {pair_name}: Cand? {'Yes' if chart_update_payload.get('candle') else 'No'}, PivH? {'Yes' if chart_update_payload.get('pivot_high') else 'No'}, PivL? {'Yes' if chart_update_payload.get('pivot_low') else 'No'}", pair_name="CHART_SERVER")


                    elif not data["big_data_collection_phase_active"]: # Live, tapi tidak ada candle baru
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: # Belum cukup data untuk analisa pivot
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
            
            # --- End of loop for crypto_data_manager.items() ---

            # Tentukan durasi tidur
            sleep_duration = 15 # Default jika tidak ada kalkulasi lain

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None:
                # Semua API key gagal global DAN tidak ada data berhasil di-fetch di siklus ini
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600 # Tunggu 1 jam
            elif active_cryptos_still_in_big_data_collection > 0:
                # Jika masih ada yang kumpulkan big data, interval tidur lebih pendek
                min_big_data_interval = float('inf')
                for pid_loop, pdata_loop in crypto_data_manager.items(): # Cek interval big data
                    if pdata_loop["big_data_collection_phase_active"]:
                        pconfig_loop = pdata_loop["config"]
                        interval_bd = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_loop.get('timeframe') == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval_bd)
                
                # Tidur sesuai interval big data terpendek, atau max 30 detik agar responsif
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) 
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua sudah live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    # Ambil nilai minimal refresh dari semua pair, tapi tidak kurang dari batas bawah
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM")
                else: # Fallback jika min_overall_next_refresh_seconds tidak terhitung
                    default_refresh_from_config = 60 # Default umum
                    if all_crypto_configs_from_settings : # Ambil dari config pair pertama jika ada
                        default_refresh_from_config = all_crypto_configs_from_settings[0].get('refresh_interval_seconds', 60)

                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama).", pair_name="SYSTEM")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: # Jika sleep_duration 0 atau negatif (seharusnya tidak terjadi)
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM")
                time.sleep(1) # Minimal sleep

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error:", pair_name="SYSTEM")
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        # _g_crypto_data_manager = None # Clear referensi global saat keluar
        # _g_all_crypto_configs = None
        input()


# --- MENU UTAMA ---

def main_menu():
    global _g_crypto_data_manager, _g_all_crypto_configs # Deklarasi untuk modifikasi

    settings = load_settings()
    _g_all_crypto_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]


    # Mulai Flask-SocketIO server di thread terpisah
    # Cek apakah thread sudah berjalan untuk menghindari multiple server
    # Ini lebih baik dihandle dengan flag atau mengecek thread yang ada.
    # Untuk kesederhanaan, kita anggap hanya ada satu instance main_menu.
    flask_thread = Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    log_info("Flask-SocketIO server thread started.", pair_name="SYSTEM")


    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Live Chart) =========", color=AnsiColors.HEADER, delay=0.005)
        
        # Perbarui _g_all_crypto_configs jika settings diubah
        _g_all_crypto_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]


        pick_title_main = "" 
        active_configs = _g_all_crypto_configs # Gunakan yang sudah difilter
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

        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Chart Server: http://localhost:5000\n" # Info chart server
        pick_title_main += f"Primary API Key: {primary_key_display} | Recovery Keys: {num_recovery_keys}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status}\n"
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
        except Exception as e_pick_main: # Fallback jika pick gagal
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main) # Tampilkan title lagi
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
            # Sebelum memulai trading, pastikan _g_crypto_data_manager direset atau disiapkan
            _g_crypto_data_manager = {} # Akan diisi oleh start_trading
            start_trading(settings)
        elif selected_index == 1:
            settings = settings_menu(settings) # settings_menu mengembalikan settings yang diupdate
             # Setelah kembali dari pengaturan, update _g_all_crypto_configs
            _g_all_crypto_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
            # Kirim ulang daftar pair ke client chart jika ada yang terhubung
            if _g_all_crypto_configs:
                pair_ids_update = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in _g_all_crypto_configs]
                socketio.emit('available_pairs', {'pairs': pair_ids_update})


        elif selected_index == 2:
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
            show_spinner(0.5, "Exiting")
            # Tidak perlu mematikan thread Flask secara eksplisit karena daemon=True
            break # Keluar dari loop while True

if __name__ == "__main__":
    try:
        # Pastikan library 'pick' ada atau berikan fallback
        try:
            import pick
        except ImportError:
            print(f"{AnsiColors.RED}WARNING: Library 'pick' tidak ditemukan. Menu akan menggunakan input angka standar.{AnsiColors.ENDC}")
            print(f"{AnsiColors.ORANGE}Anda bisa menginstalnya dengan: pip install pick{AnsiColors.ENDC}")
            time.sleep(3) # Beri waktu untuk membaca warning

        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        # Menggunakan logger yang sudah dikonfigurasi untuk traceback yang lebih baik
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        log_info("Skrip Selesai.", pair_name="SYSTEM_FINAL")
