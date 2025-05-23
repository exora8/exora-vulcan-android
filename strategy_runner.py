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
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
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


SETTINGS_FILE = "settings_supertrend.json" # Nama file setting disesuaikan
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 250 # Cukup untuk Supertrend dan display
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER (Sama seperti sebelumnya) ---
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
        if self.current_index < len(self.keys): return self.keys[self.current_index]
        return None

    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # ... (logika email notifikasi API key switch, sama seperti sebelumnya) ...
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
            # ... (logika email notifikasi KRITIS API key habis, sama seperti sebelumnya) ...
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

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION (Sama seperti sebelumnya) ---
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
    if not settings_for_email.get("enable_email_notifications", False): return
    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")
    pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
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
    api_settings = global_settings.get("api_settings", {})
    if not api_settings.get("enable_termux_notifications", False): return
    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,

        # Parameter Supertrend
        "atr_length": 10,
        "factor": 3.0, # Biasanya float, misal 3.0 atau 2.0

        # Parameter Exits
        "emergency_sl_percent": 2.0,             # Fixed Stop Loss %
        "profit_target_percent_activation": 2.0, # Trail Activation Profit %
        "trailing_stop_gap_percent": 1.0,        # Trail Gap %
        # "use_percentage_trailing_tp": True, # Implisit True karena pakai persentase

        # Parameter lama (disimpan untuk kompatibilitas file, tidak dipakai logika Supertrend)
        "ma_length": 50, "stoch_length": 14, "stoch_smooth_k": 3, "stoch_smooth_d": 3,
        "stoch_overbought": 80, "stoch_oversold": 20,
        "left_strength": 50, "right_strength": 150, "enable_secure_fib": True, "secure_fib_check_price": "Close",

        "enable_email_notifications": False,
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }

def load_settings():
    default_api_settings = {
        "primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com", "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com",
        "enable_termux_notifications": False
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            if "api_settings" not in settings: settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]: settings["api_settings"][k] = v
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list): settings["cryptos"] = []
            default_crypto_template = get_default_crypto_config()
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
            return settings
        except Exception as e:
            log_error(f"Error membaca {SETTINGS_FILE}: {e}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    # Input standar (symbol, currency, exchange, timeframe, refresh_interval)
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}");
    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        refresh_input = int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60)
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, refresh_input)
    except ValueError:
        print(f"{AnsiColors.RED}Input interval refresh tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))

    animated_text_display("\n-- Parameter Supertrend --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["atr_length"] = int(input(f"{AnsiColors.BLUE}Supertrend - ATR Length [{new_config.get('atr_length',10)}]: {AnsiColors.ENDC}").strip() or new_config.get('atr_length',10))
        new_config["factor"] = float(input(f"{AnsiColors.BLUE}Supertrend - Factor [{new_config.get('factor',3.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('factor',3.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter Supertrend tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        default_cfg_temp = get_default_crypto_config()
        new_config["atr_length"] = new_config.get('atr_length', default_cfg_temp['atr_length'])
        new_config["factor"] = new_config.get('factor', default_cfg_temp['factor'])

    animated_text_display("\n-- Parameter Exits --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Fixed Stop Loss % [{new_config.get('emergency_sl_percent',2.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',2.0))
        animated_text_display("\n-- Trailing Take Profit (Persentase) --", color=AnsiColors.CYAN, delay=0.01)
        # Checkbox "Use Percentage Trailing TP?" adalah implisit True dengan parameter berikut
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Trail Activation Profit % [{new_config.get('profit_target_percent_activation',2.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',2.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Trail Gap % [{new_config.get('trailing_stop_gap_percent',1.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',1.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter Exits tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        default_cfg_temp = get_default_crypto_config()
        for k_exit in ["emergency_sl_percent", "profit_target_percent_activation", "trailing_stop_gap_percent"]:
            new_config[k_exit] = new_config.get(k_exit, default_cfg_temp[k_exit])

    # Input notifikasi email (Sama seperti sebelumnya)
    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global.{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()

    return new_config

def settings_menu(current_settings): # Fungsi settings_menu tetap sama strukturnya
    # ... (Struktur menu pick sama seperti sebelumnya, hanya judul strategi yang diubah di dalamnya) ...
    # (Penting: Saat memanggil _prompt_crypto_config, ia akan menggunakan versi baru yang fokus Supertrend)
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len([k for k in api_s.get('recovery_keys', []) if k])
        termux_notif_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Strategi Aktif: Supertrend Entry\n" # Diubah
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
            ("option", "Atur Primary API Key"), ("option", "Kelola Recovery API Keys"),
            ("option", "Atur Email Global untuk Notifikasi Sistem"), ("option", "Aktifkan/Nonaktifkan Notifikasi Termux Realtime"),
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"), ("option", "Ubah Konfigurasi Crypto"),
            ("option", "Hapus Konfigurasi Crypto"),
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")
        ]
        selectable_options = [text for type, text in original_options_structure if type == "option"]
        selected_option_text, action_choice = None, -1
        try:
            selected_option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception: # Fallback jika pick gagal
            print(pick_title_settings)
            for idx, opt_text in enumerate(selectable_options): print(f"  {idx + 1}. {opt_text}")
            try:
                choice = int(input("Pilih nomor opsi: ").strip()) -1
                if 0 <= choice < len(selectable_options): action_choice = choice
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
        
        # Logika untuk setiap pilihan menu (sama seperti sebelumnya, _prompt_crypto_config akan menangani input parameter Supertrend)
        try:
            clear_screen_animated()
            if action_choice == 0: # Atur Primary API Key
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s
            elif action_choice == 1: # Kelola Recovery API Keys (Loop internal sama)
                while True:
                    # ... (logika kelola recovery key, tidak berubah)
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k]
                    api_s['recovery_keys'] = current_recovery
                    if not current_recovery: recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"
                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]
                    rec_selected_text, rec_index = None, -1
                    try: rec_selected_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception: # Fallback
                        print(recovery_pick_title)
                        for idx_rec, opt_text_rec in enumerate(recovery_options_plain): print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice_val = int(input("Pilih nomor opsi: ").strip()) -1
                            if 0 <= rec_choice_val < len(recovery_options_plain): rec_index = rec_choice_val
                            else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                        except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                    
                    clear_screen_animated()
                    if rec_index == 0: # Tambah
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key: current_recovery.append(new_r_key); api_s['recovery_keys'] = current_recovery; print(f"{AnsiColors.GREEN}Ditambahkan.{AnsiColors.ENDC}")
                        else: print(f"{AnsiColors.RED}Input kosong.{AnsiColors.ENDC}")
                    elif rec_index == 1: # Hapus
                        if not current_recovery: print(f"{AnsiColors.ORANGE}Tidak ada untuk dihapus.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                        del_options = [f"{r_key[:5]}...{r_key[-3:]}" for r_key in current_recovery] + ["Batal"]
                        _del_text, idx_del_pick = pick(del_options, "Pilih yang akan dihapus:", indicator='=>')
                        if idx_del_pick < len(current_recovery) : removed = current_recovery.pop(idx_del_pick); api_s['recovery_keys'] = current_recovery; print(f"{AnsiColors.GREEN}Dihapus.{AnsiColors.ENDC}")
                    elif rec_index == 2: break # Kembali
                    save_settings(current_settings) # Simpan setelah tambah/hapus
                    show_spinner(1, "Kembali...")
            elif action_choice == 2: # Atur Email Global
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s
            elif action_choice == 3: # Notifikasi Termux
                animated_text_display("-- Pengaturan Notifikasi Termux Realtime --", color=AnsiColors.HEADER)
                current_status = api_s.get('enable_termux_notifications', False)
                new_status_input = input(f"Aktifkan Notifikasi Termux? (true/false) [{current_status}]: ").lower().strip()
                if new_status_input == 'true': api_s['enable_termux_notifications'] = True; print(f"{AnsiColors.GREEN}Diaktifkan.{AnsiColors.ENDC}")
                elif new_status_input == 'false': api_s['enable_termux_notifications'] = False; print(f"{AnsiColors.GREEN}Dinonaktifkan.{AnsiColors.ENDC}")
                else: print(f"{AnsiColors.ORANGE}Input tidak valid. Status tidak berubah.{AnsiColors.ENDC}")
                current_settings["api_settings"] = api_s
            elif action_choice == 4: # Tambah Konfigurasi Crypto
                new_crypto_conf = get_default_crypto_config()
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) # Ini akan memanggil prompt Supertrend
                current_settings.setdefault("cryptos", []).append(new_crypto_conf)
            elif action_choice == 5: # Ubah Konfigurasi Crypto
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada untuk diubah.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                edit_options = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]] + ["Batal"]
                _edit_text, idx_choice_pick = pick(edit_options, "Pilih yang akan diubah:", indicator='=>')
                if idx_choice_pick < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice_pick] = _prompt_crypto_config(current_settings["cryptos"][idx_choice_pick]) # Prompt Supertrend
            elif action_choice == 6: # Hapus Konfigurasi Crypto
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada untuk dihapus.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                del_crypto_options = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]] + ["Batal"]
                _del_c_text, idx_del_c_pick = pick(del_crypto_options, "Pilih yang akan dihapus:", indicator='=>')
                if idx_del_c_pick < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_del_c_pick]['symbol']}-{current_settings['cryptos'][idx_del_c_pick]['currency']}"
                    current_settings["cryptos"].pop(idx_del_c_pick)
                    log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
            elif action_choice == 7: # Kembali ke Menu Utama
                break
            save_settings(current_settings) # Simpan perubahan setelah setiap aksi utama (kecuali kembali)
            show_spinner(1, "Menyimpan & Kembali...")
        except Exception as e_settings:
            log_error(f"Error di menu pengaturan: {e_settings}")
            show_spinner(1.5, "Error, kembali...")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA (Sama seperti sebelumnya) ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    # ... (Isi fungsi fetch_candles sama persis seperti skrip Anda sebelumnya) ...
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10
    if is_large_fetch:
        log_info(f"Pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
            simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        if limit_for_this_api_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_for_this_api_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429]: # Key error atau rate limit
                error_data = {}; 
                try: error_data = response.json()
                except: pass
                msg = error_data.get('Message', f"HTTP Error {response.status_code}")
                raise APIKeyError(f"HTTP {response.status_code}: {msg}")
            response.raise_for_status()
            data = response.json()
            if data.get('Response') == 'Error':
                msg = data.get('Message', 'N/A')
                # Pesan error terkait API Key
                key_err_msgs = ["api key is invalid", "apikey_is_missing", "your_monthly_calls_are_over_the_limit", "rate limit exceeded"]
                if any(k_err.lower() in msg.lower() for k_err in key_err_msgs):
                    raise APIKeyError(f"JSON Error: {msg}")
                else: # Error API lain
                    log_error(f"API Error CryptoCompare: {msg}", pair_name=pair_name); break
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API. Total: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break
            raw_candles_from_api = data['Data']['Data']
            if not raw_candles_from_api: break
            batch_candles_list = []
            for item in raw_candles_from_api:
                req_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in req_keys): continue # Skip candle tak lengkap
                batch_candles_list.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list and batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                batch_candles_list.pop() # Hapus overlap
            if not batch_candles_list and current_to_ts is not None: break
            all_accumulated_candles = batch_candles_list + all_accumulated_candles
            if raw_candles_from_api: current_to_ts = raw_candles_from_api[0]['time']
            else: break
            fetch_loop_count += 1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
            if len(raw_candles_from_api) < limit_for_this_api_call: break # Akhir histori
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.3)
        except APIKeyError: raise
        except requests.exceptions.RequestException as e: log_error(f"Kesalahan koneksi: {e}", pair_name=pair_name); break
        except Exception as e: log_error(f"Error fetch_candles: {e}", pair_name=pair_name); break
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
         simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai', length=40)
    if is_large_fetch: log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)}.", pair_name=pair_name)
    return all_accumulated_candles

# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        # State untuk Supertrend
        "last_supertrend_value": None,
        "supertrend_direction": 0, # 1 untuk Bullish (ST di bawah harga), -1 untuk Bearish (ST di atas harga)
        "last_atr_value_for_chart": None, # Bisa ditambahkan ke chart jika mau

        # State manajemen posisi (sama)
        "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
        
        # State lama (tidak dipakai, tapi biarkan di struktur untuk save/load settings lama)
        "last_ma_value_for_chart": None, "last_stoch_k_value_for_chart": None, "last_stoch_d_value_for_chart": None,
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None, "last_pivot_low_display_info": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None, "active_fib_line_start_index": None,
    }

def calculate_atr(high_prices, low_prices, close_prices, period):
    if len(close_prices) < period or period <= 0:
        return [None] * len(close_prices)
    
    tr_values = [None] * len(close_prices)
    atr_values = [None] * len(close_prices)

    # Hitung True Range (TR)
    for i in range(len(close_prices)):
        if high_prices[i] is None or low_prices[i] is None or close_prices[i] is None:
            continue
        
        high_low = high_prices[i] - low_prices[i]
        if i > 0 and close_prices[i-1] is not None:
            high_close_prev = abs(high_prices[i] - close_prices[i-1])
            low_close_prev = abs(low_prices[i] - close_prices[i-1])
            tr_values[i] = max(high_low, high_close_prev, low_close_prev)
        elif i == 0 : # Untuk candle pertama, TR adalah High - Low
             tr_values[i] = high_low
        # else: tr_values[i] akan tetap None jika close_prices[i-1] None setelah candle 0

    # Hitung ATR
    # Cari index pertama TR yang valid
    first_valid_tr_index = -1
    for i, tr_val in enumerate(tr_values):
        if tr_val is not None:
            first_valid_tr_index = i
            break
    
    if first_valid_tr_index == -1: return atr_values # Tidak ada TR valid

    # Cek apakah ada cukup data TR valid setelah index pertama
    num_valid_trs = sum(1 for tr_val in tr_values[first_valid_tr_index:] if tr_val is not None)

    if num_valid_trs < period : return atr_values # Tidak cukup TR valid untuk memulai ATR

    # Hitung ATR pertama (SMA dari TR)
    sum_first_trs = 0
    count_first_trs = 0
    atr_start_index = -1
    
    temp_idx_walk = first_valid_tr_index
    while count_first_trs < period and temp_idx_walk < len(tr_values):
        if tr_values[temp_idx_walk] is not None:
            sum_first_trs += tr_values[temp_idx_walk]
            count_first_trs +=1
        if count_first_trs == period:
            atr_start_index = temp_idx_walk
            break
        temp_idx_walk +=1
    
    if atr_start_index != -1 and count_first_trs == period:
        atr_values[atr_start_index] = sum_first_trs / period
        # Hitung ATR berikutnya menggunakan Wilder's Smoothing
        for i in range(atr_start_index + 1, len(close_prices)):
            if tr_values[i] is not None and atr_values[i-1] is not None:
                atr_values[i] = (atr_values[i-1] * (period - 1) + tr_values[i]) / period
            # Jika tr_values[i] atau atr_values[i-1] adalah None, atr_values[i] akan tetap None
    
    return atr_values


def calculate_supertrend(high_prices, low_prices, close_prices, atr_length, factor):
    num_candles = len(close_prices)
    if num_candles < atr_length + 1: # Butuh setidaknya atr_length + 1 untuk supertrend pertama
        return ([None] * num_candles, [0] * num_candles) # supertrend_values, direction_values

    atr_values = calculate_atr(high_prices, low_prices, close_prices, atr_length)
    
    supertrend_values = [None] * num_candles
    direction_values = [0] * num_candles # 1 for uptrend, -1 for downtrend

    # Inisialisasi Supertrend
    # Cari candle pertama dimana ATR dan harga close valid
    first_valid_candle_idx = -1
    for i in range(atr_length, num_candles): # Mulai dari atr_length karena ATR butuh itu
        if close_prices[i] is not None and high_prices[i] is not None and low_prices[i] is not None and atr_values[i] is not None:
            first_valid_candle_idx = i
            break
    
    if first_valid_candle_idx == -1: # Tidak ada candle yang cukup valid untuk memulai
        return (supertrend_values, direction_values)

    # Untuk candle valid pertama, tentukan Supertrend awal secara sederhana
    # Ini adalah simplifikasi. Implementasi yang lebih kompleks mungkin memproyeksikan dari awal.
    # Kita akan anggap candle pertama selalu sebagai permulaan tren baru (misal uptrend).
    # Ini akan dikoreksi pada iterasi berikutnya.
    
    # Hitung basic bands untuk candle valid pertama
    basic_upper_band_init = ((high_prices[first_valid_candle_idx] + low_prices[first_valid_candle_idx]) / 2) + factor * atr_values[first_valid_candle_idx]
    basic_lower_band_init = ((high_prices[first_valid_candle_idx] + low_prices[first_valid_candle_idx]) / 2) - factor * atr_values[first_valid_candle_idx]

    # Default Supertrend awal (misal, bearish jika harga di bawah tengah, atau bullish jika di atas)
    mid_price_init = (high_prices[first_valid_candle_idx] + low_prices[first_valid_candle_idx]) / 2
    if close_prices[first_valid_candle_idx] <= mid_price_init :
        supertrend_values[first_valid_candle_idx] = basic_upper_band_init
        direction_values[first_valid_candle_idx] = -1 # Downtrend
    else:
        supertrend_values[first_valid_candle_idx] = basic_lower_band_init
        direction_values[first_valid_candle_idx] = 1  # Uptrend


    # Iterasi untuk candle berikutnya
    for i in range(first_valid_candle_idx + 1, num_candles):
        if close_prices[i] is None or high_prices[i] is None or low_prices[i] is None or atr_values[i] is None or close_prices[i-1] is None or supertrend_values[i-1] is None:
            # Propagate None jika data tidak cukup atau ST sebelumnya tidak valid
            supertrend_values[i] = None 
            direction_values[i] = direction_values[i-1] # Bawa arah sebelumnya
            continue

        basic_upper_band = ((high_prices[i] + low_prices[i]) / 2) + factor * atr_values[i]
        basic_lower_band = ((high_prices[i] + low_prices[i]) / 2) - factor * atr_values[i]
        
        prev_supertrend = supertrend_values[i-1]
        prev_direction = direction_values[i-1]

        current_supertrend = None
        current_direction = prev_direction

        if prev_direction == 1: # Previous trend was UP
            current_supertrend = max(basic_lower_band, prev_supertrend)
            if close_prices[i] < current_supertrend: # Price crossed below new ST
                current_direction = -1
                current_supertrend = basic_upper_band # Flip to upper band
        elif prev_direction == -1: # Previous trend was DOWN
            current_supertrend = min(basic_upper_band, prev_supertrend)
            if close_prices[i] > current_supertrend: # Price crossed above new ST
                current_direction = 1
                current_supertrend = basic_lower_band # Flip to lower band
        else: # prev_direction was 0 (initial state, should not happen after first_valid_candle_idx)
             # Re-evaluate based on current close vs mid (sama seperti inisialisasi)
            mid_price_curr = (high_prices[i] + low_prices[i]) / 2
            if close_prices[i] <= mid_price_curr :
                current_supertrend = basic_upper_band
                current_direction = -1
            else:
                current_supertrend = basic_lower_band
                current_direction = 1 
        
        supertrend_values[i] = current_supertrend
        direction_values[i] = current_direction
        
    return supertrend_values, direction_values


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"

    # Parameter Supertrend
    atr_len = crypto_config.get('atr_length', 10)
    st_factor = crypto_config.get('factor', 3.0)
    
    # Parameter Exits
    sl_percentage = crypto_config.get('emergency_sl_percent', 2.0) / 100.0
    tp_activation_percentage = crypto_config.get('profit_target_percent_activation', 2.0) / 100.0
    tp_trailing_gap_percentage = crypto_config.get('trailing_stop_gap_percent', 1.0) / 100.0

    min_data_needed = atr_len + 2 # Perkiraan minimal untuk Supertrend + cross
    if len(candles_history) < min_data_needed:
        log_debug(f"Data candle ({len(candles_history)}) tidak cukup untuk Supertrend (ATR: {atr_len}). Min: ~{min_data_needed}. Skip.", pair_name=pair_name)
        return strategy_state

    close_prices = [c.get('close') for c in candles_history]
    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]
    
    supertrend_line, supertrend_direction_signal = calculate_supertrend(
        high_prices, low_prices, close_prices, atr_len, st_factor
    )

    current_candle_idx = len(candles_history) - 1
    prev_candle_idx = current_candle_idx - 1

    current_candle = candles_history[current_candle_idx]
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru. Skip.", pair_name=pair_name)
        return strategy_state

    current_close = current_candle['close']
    current_low = current_candle['low']
    current_high = current_candle['high']
    current_st_value = supertrend_line[current_candle_idx]
    current_st_direction = supertrend_direction_signal[current_candle_idx]

    strategy_state["last_supertrend_value"] = current_st_value
    strategy_state["supertrend_direction"] = current_st_direction
    # strategy_state["last_atr_value_for_chart"] = calculate_atr(high_prices, low_prices, close_prices, atr_len)[-1]


    # Data sebelumnya untuk deteksi cross
    prev_close = None
    prev_st_value = None
    prev_st_direction = 0 # Default jika tidak ada data sebelumnya

    if prev_candle_idx >= 0:
        prev_close_val = candles_history[prev_candle_idx].get('close')
        if prev_close_val is not None: prev_close = prev_close_val
        if supertrend_line[prev_candle_idx] is not None: prev_st_value = supertrend_line[prev_candle_idx]
        prev_st_direction = supertrend_direction_signal[prev_candle_idx]


    # --- Kondisi Entry & Exit ---
    if current_st_value is None or prev_st_value is None or prev_close is None:
        log_debug("Data Supertrend tidak cukup untuk evaluasi cross. Skip.", pair_name=pair_name)
    else:
        # Entry Long Condition: Harga close memotong ke atas Supertrend
        # dan Supertrend saat ini menunjukkan arah bullish (ST di bawah harga)
        crossed_above_st = prev_close <= prev_st_value and current_close > current_st_value
        # Sinyal menjadi lebih kuat jika ST juga baru flip ke bullish atau sudah bullish
        entry_long_condition = crossed_above_st and current_st_direction == 1

        # Exit Long / Sinyal Jual Potensial: Harga close memotong ke bawah Supertrend
        crossed_below_st = prev_close >= prev_st_value and current_close < current_st_value
        exit_long_condition = crossed_below_st # Tidak perlu cek arah ST untuk exit

        # Jika dalam posisi Long
        if strategy_state["position_size"] > 0:
            if exit_long_condition:
                exit_price = current_close # Exit di harga close saat cross
                entry_price_val = strategy_state["entry_price_custom"]
                pnl = ((exit_price - entry_price_val) / entry_price_val) * 100.0 if entry_price_val else 0.0
                
                log_msg_exit = f"EXIT ORDER (Supertrend Cross) @ {exit_price:.5f}. PnL: {pnl:.2f}%"
                log_info(f"{AnsiColors.BLUE}{AnsiColors.BOLD}{log_msg_exit}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()
                # Notifikasi Termux & Email (sama seperti logic SL/TP exit)
                termux_title_exit = f"EXIT Signal (ST): {pair_name}"
                termux_content_exit = f"ST Cross @ {exit_price:.5f}. PnL: {pnl:.2f}%"
                send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)
                email_subject_exit = f"Trade Closed (Supertrend): {pair_name}"
                email_body_exit = (f"Trade closed for {pair_name} by Supertrend signal.\n\n"
                                   f"Exit Price: {exit_price:.5f}\nEntry: {entry_price_val:.5f}\nPnL: {pnl:.2f}%\n"
                                   f"ST Val: {current_st_value:.5f}, Direction: {current_st_direction}\n"
                                   f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject_exit, email_body_exit, {**crypto_config, 'pair_name': pair_name})
                
                # Reset state posisi
                strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None
                strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False
                strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
                log_info("State posisi direset setelah Supertrend exit.", pair_name=pair_name)

        # Jika TIDAK dalam posisi Long
        elif strategy_state["position_size"] == 0:
            if entry_long_condition:
                entry_price = current_close
                strategy_state["position_size"] = 1
                strategy_state["entry_price_custom"] = entry_price
                strategy_state["emergency_sl_level_custom"] = entry_price * (1 - sl_percentage)
                strategy_state["highest_price_for_trailing"] = current_high # Inisialisasi dengan high candle entry
                strategy_state["trailing_tp_active_custom"] = False
                strategy_state["current_trailing_stop_level"] = None
                
                log_msg_entry = (f"BUY ENTRY (Supertrend) @ {entry_price:.5f}. "
                                 f"ST Val: {current_st_value:.5f}, Direction: {current_st_direction}. "
                                 f"SL: {strategy_state['emergency_sl_level_custom']:.5f}")
                log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg_entry}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()
                # Notifikasi Termux & Email
                termux_title_entry = f"BUY Signal (ST): {pair_name}"
                termux_content_entry = (f"Entry @ {entry_price:.5f}. ST: {current_st_value:.5f}. "
                                     f"SL: {strategy_state['emergency_sl_level_custom']:.5f}")
                send_termux_notification(termux_title_entry, termux_content_entry, global_settings, pair_name_for_log=pair_name)
                email_subject_entry = f"BUY Signal (Supertrend): {pair_name}"
                email_body_entry = (f"New BUY signal for {pair_name} by Supertrend.\n\n"
                                    f"Entry Price: {entry_price:.5f}\n"
                                    f"Supertrend Value: {current_st_value:.5f}\nSupertrend Direction: {'Bullish' if current_st_direction == 1 else 'Bearish'}\n"
                                    f"Stop Loss: {strategy_state['emergency_sl_level_custom']:.5f}\n"
                                    f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject_entry, email_body_entry, {**crypto_config, 'pair_name': pair_name})

            elif exit_long_condition: # Sinyal Jual Potensial (jika tidak ada posisi)
                log_msg_short_alert = (f"SINYAL JUAL POTENSIAL (Supertrend) @ Close {current_close:.5f}. "
                                       f"ST Val: {current_st_value:.5f}, Direction: {current_st_direction}.")
                log_info(f"{AnsiColors.MAGENTA}{log_msg_short_alert}{AnsiColors.ENDC}", pair_name=pair_name)
                # Notifikasi opsional untuk sinyal jual potensial
                termux_title_short_alert = f"SELL Alert (ST): {pair_name}"
                termux_content_short_alert = f"SELL Alert @ {current_close:.5f}. ST: {current_st_value:.5f}"
                send_termux_notification(termux_title_short_alert, termux_content_short_alert, global_settings, pair_name_for_log=pair_name)


    # --- Fixed SL & Trailing TP (Hanya jika dalam posisi Long) ---
    if strategy_state["position_size"] > 0:
        entry_price_val = strategy_state["entry_price_custom"]
        
        # Update harga tertinggi untuk trailing
        if current_high is not None:
            if strategy_state["highest_price_for_trailing"] is None: strategy_state["highest_price_for_trailing"] = current_high
            else: strategy_state["highest_price_for_trailing"] = max(strategy_state["highest_price_for_trailing"], current_high)

        # Aktivasi Trailing TP
        if not strategy_state["trailing_tp_active_custom"] and entry_price_val and entry_price_val > 0:
            highest_price = strategy_state["highest_price_for_trailing"]
            if highest_price:
                profit_perc = ((highest_price - entry_price_val) / entry_price_val)
                if profit_perc >= tp_activation_percentage:
                    strategy_state["trailing_tp_active_custom"] = True
                    log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_perc*100:.2f}%, High: {highest_price:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        # Update Trailing Stop Level
        if strategy_state["trailing_tp_active_custom"]:
            highest_price = strategy_state["highest_price_for_trailing"]
            if highest_price:
                potential_new_stop = highest_price * (1 - tp_trailing_gap_percentage)
                if strategy_state["current_trailing_stop_level"] is None or potential_new_stop > strategy_state["current_trailing_stop_level"]:
                    strategy_state["current_trailing_stop_level"] = potential_new_stop
                    log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        # Tentukan SL yang berlaku
        effective_sl_level = strategy_state["emergency_sl_level_custom"]
        exit_reason = "Stop Loss"
        exit_color = AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"]:
            if effective_sl_level is None or strategy_state["current_trailing_stop_level"] > effective_sl_level:
                effective_sl_level = strategy_state["current_trailing_stop_level"]
                exit_reason = "Trailing TP"
                exit_color = AnsiColors.BLUE

        # Cek jika SL/Trailing TP kena
        if effective_sl_level and current_low and current_low <= effective_sl_level:
            exit_price = effective_sl_level 
            pnl = ((exit_price - entry_price_val) / entry_price_val) * 100.0 if entry_price_val else 0.0
            if exit_reason == "Trailing TP" and pnl < 0: exit_color = AnsiColors.RED; exit_reason = "Trailing SL (Loss)"
            
            log_msg_sltp_exit = f"EXIT ORDER ({exit_reason}) @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg_sltp_exit}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()
            # Notifikasi Termux & Email (sama seperti sebelumnya)
            termux_title_sltp_exit = f"EXIT Signal ({exit_reason}): {pair_name}"
            termux_content_sltp_exit = f"{exit_reason} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_sltp_exit, termux_content_sltp_exit, global_settings, pair_name_for_log=pair_name)
            email_subject_sltp_exit = f"Trade Closed ({exit_reason}): {pair_name}"
            email_body_sltp_exit = (f"Trade closed for {pair_name}.\n\n"
                                  f"Reason: {exit_reason}\nExit Price: {exit_price:.5f}\nEntry: {entry_price_val:.5f}\nPnL: {pnl:.2f}%\n"
                                  f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject_sltp_exit, email_body_sltp_exit, {**crypto_config, 'pair_name': pair_name})

            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
            log_info(f"State posisi direset setelah {exit_reason}.", pair_name=pair_name)
            
        elif strategy_state["position_size"] > 0 : 
            entry_display = strategy_state.get('entry_price_custom', 0)
            sl_display_str = f'{effective_sl_level:.5f} ({exit_reason})' if effective_sl_level else 'N/A'
            high_trail_display = strategy_state.get('highest_price_for_trailing', 0)
            log_debug(f"Posisi Aktif. Entry: {entry_display:.5f}, SL: {sl_display_str}, High: {high_trail_display:.5f}, ST: {current_st_value:.5f} (Dir: {current_st_direction})", pair_name=pair_name)

    return strategy_state

# CHART_INTEGRATION_START
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot):
    if pair_id_to_display not in current_data_manager_snapshot:
        return None

    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]
    candles_full_history = pair_specific_data.get("all_candles_list", [])
    current_strategy_state = pair_specific_data.get("strategy_state", {})
    pair_config = pair_specific_data.get("config", {})

    candles_for_chart_display = candles_full_history[-TARGET_BIG_DATA_CANDLES:]

    ohlc_data_points = []
    supertrend_series_data = []
    
    if not candles_for_chart_display:
        return {"ohlc": [], "supertrend_series": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None}

    # Hitung ulang Supertrend untuk data yang akan ditampilkan di chart
    chart_close_prices = [c.get('close') for c in candles_for_chart_display]
    chart_high_prices = [c.get('high') for c in candles_for_chart_display]
    chart_low_prices = [c.get('low') for c in candles_for_chart_display]
    chart_atr_len = pair_config.get('atr_length', 10)
    chart_st_factor = pair_config.get('factor', 3.0)
    
    chart_st_line, chart_st_direction = calculate_supertrend(
        chart_high_prices, chart_low_prices, chart_close_prices, chart_atr_len, chart_st_factor
    )

    for i, candle in enumerate(candles_for_chart_display):
        if all(k in candle and candle[k] is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = candle['timestamp'].timestamp() * 1000
            ohlc_data_points.append({'x': ts_ms, 'y': [candle['open'], candle['high'], candle['low'], candle['close']]})
            if i < len(chart_st_line) and chart_st_line[i] is not None:
                color = '#26A69A' if chart_st_direction[i] == 1 else ('#EF5350' if chart_st_direction[i] == -1 else '#888888') # Green for up, Red for down, Gray for neutral
                supertrend_series_data.append({'x': ts_ms, 'y': chart_st_line[i], 'color': color}) # Simpan warna per titik
        else:
            log_debug(f"Skipping incomplete candle for chart: {candle.get('timestamp')}", pair_name="SYSTEM_CHART")

    chart_annotations_yaxis = [] # Anotasi SL/Entry (sama seperti sebelumnya)
    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") is not None:
        entry_price_val = current_strategy_state.get("entry_price_custom")
        if ohlc_data_points:
             chart_annotations_yaxis.append({'y': entry_price_val, 'borderColor': '#2698FF', 'strokeDashArray': 4, 'label': {'borderColor': '#2698FF', 'style': {'color': '#fff', 'background': '#2698FF'}, 'text': f'Entry: {entry_price_val:.5f}'}})
        sl_level_val = current_strategy_state.get("emergency_sl_level_custom")
        sl_type_text = "SL"
        if current_strategy_state.get("trailing_tp_active_custom") and current_strategy_state.get("current_trailing_stop_level"):
            current_trailing_sl_val = current_strategy_state.get("current_trailing_stop_level")
            if sl_level_val is None or (current_trailing_sl_val and current_trailing_sl_val > sl_level_val):
                sl_level_val = current_trailing_sl_val; sl_type_text = "Trail.SL"
        if sl_level_val and ohlc_data_points:
            chart_annotations_yaxis.append({'y': sl_level_val, 'borderColor': '#FF4560', 'label': {'borderColor': '#FF4560', 'style': {'color': '#fff', 'background': '#FF4560'}, 'text': f'{sl_type_text}: {sl_level_val:.5f}'}})
    
    last_updated_tv_val = candles_for_chart_display[-1]['timestamp'].timestamp() * 1000 if candles_for_chart_display and candles_for_chart_display[-1].get('timestamp') else None

    return {
        "ohlc": ohlc_data_points,
        "supertrend_series": supertrend_series_data, 
        "st_atr_len_label": chart_atr_len, "st_factor_label": chart_st_factor,
        "annotations_yaxis": chart_annotations_yaxis, "annotations_points": [],
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": last_updated_tv_val,
        "strategy_state_info": {
            "supertrend_value": current_strategy_state.get("last_supertrend_value"),
            "supertrend_direction": current_strategy_state.get("supertrend_direction")
        }
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supertrend Strategy Chart</title> <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style> body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;} #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width:100%; max-width: 1200px; } select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; } #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; } h1 { color: #00bcd4; margin-bottom:10px; font-size:1.5em; } #lastUpdatedLabel { font-size: 0.8em; color: #aaa; margin-left: auto; } #strategyInfoLabel { font-size: 0.8em; color: #FFD700; margin-left: 10px; white-space: pre; }</style>
</head>
<body>
    <h1>Supertrend Strategy Chart</h1>
    <div id="controls">
        <label for="pairSelector">Pilih Pair:</label> <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh</button>
        <span id="strategyInfoLabel">Status: -</span> <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container"><div id="chart"></div></div>
    <script>
        let activeChart, currentSelectedPairId = '', lastKnownDataTimestamp = null, autoRefreshIntervalId = null, isLoadingData = false;
        const initialChartOptions = {
            series: [{ name: 'Candlestick', type: 'candlestick', data: [] }, { name: 'Supertrend', type: 'line', data: [] }],
            chart: { type: 'candlestick', height: 550, background: '#2a2a2a', animations: { enabled: false }, toolbar: { show: true } }, // Animasi disable untuk ST
            theme: { mode: 'dark' }, title: { text: 'Memuat Data Pair...', align: 'left', style: {color: '#e0e0e0'} },
            xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: '#aaa'}, formatter: (v) => v ? v.toFixed(5) : '' } },
            stroke: { width: [1, 2], curve: 'straight' }, // Straight line untuk ST
            markers: { size: 0 }, colors: ['#FEB019', '#888888'], // Default ST color (akan di-override per titik)
            grid: { borderColor: '#444' }, annotations: { yaxis: [], points: [] },
            tooltip: { theme: 'dark', shared: true, intersect: false, 
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    let o, h, l, c, stVal, stDir;
                    const csIdx = w.globals.series.findIndex(s => s.type === 'candlestick');
                    const stIdx = w.globals.series.findIndex(s => s.name.startsWith('Supertrend'));
                    if (csIdx !== -1 && w.globals.seriesCandleO[csIdx]?.[dataPointIndex] !== undefined) {
                        [o,h,l,c] = [w.globals.seriesCandleO[csIdx][dataPointIndex], w.globals.seriesCandleH[csIdx][dataPointIndex], w.globals.seriesCandleL[csIdx][dataPointIndex], w.globals.seriesCandleC[csIdx][dataPointIndex]];
                    }
                    if (stIdx !== -1 && series[stIdx]?.[dataPointIndex] !== undefined && w.config.series[stIdx].data[dataPointIndex]) {
                        stVal = w.config.series[stIdx].data[dataPointIndex].y;
                        // Arah ST tidak mudah didapat dari sini, bisa dikirim terpisah atau logic tooltip diubah
                    }
                    let html = '<div style="padding:5px 10px; background:#333; color:#fff; border:1px solid #555;">';
                    if (o !== undefined) html += ['O','H','L','C'].map((k,i) => `<div>${k}: <span style="font-weight:bold;">${[o,h,l,c][i].toFixed(5)}</span></div>`).join('');
                    if (stVal !== undefined) html += `<div>ST: <span style="font-weight:bold;">${stVal.toFixed(5)}</span></div>`;
                    html += '</div>';
                    return (o !== undefined || stVal !== undefined) ? html : '';
                }
            },
            noData: { text: 'Tidak ada data.', align: 'center', style: {color: '#ccc'} }
        };
        async function fetchAvailablePairs() {
            try {
                const r = await fetch('/api/available_pairs'); if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const pairs = await r.json(); const sel = document.getElementById('pairSelector'); sel.innerHTML = '';
                if (pairs.length > 0) {
                    pairs.forEach(p => { const opt = document.createElement('option'); opt.value = p.id; opt.textContent = p.name; sel.appendChild(opt); });
                    currentSelectedPairId = sel.value || pairs[0].id; loadChartDataForCurrentPair();
                } else { sel.innerHTML = '<option value="">No pairs</option>'; if(activeChart) activeChart.destroy(); activeChart=null; document.getElementById('chart').innerHTML = 'No pairs configured.';}
            } catch (e) { console.error("Err pairs:", e); if(activeChart)activeChart.destroy();activeChart=null; document.getElementById('chart').innerHTML = `Err: ${e.message}`; }
        }
        function handlePairSelectionChange() { currentSelectedPairId = document.getElementById('pairSelector').value; lastKnownDataTimestamp = null; loadChartDataForCurrentPair(); }
        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId || isLoadingData) return;
            isLoadingData = true; document.getElementById('lastUpdatedLabel').textContent = `Loading ${currentSelectedPairId}...`;
            try {
                const r = await fetch(`/api/chart_data/${currentSelectedPairId}`); if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const payload = await r.json();
                if (!payload || !payload.ohlc || payload.ohlc.length === 0) {
                    const noDataOpts = {...initialChartOptions, title: {...initialChartOptions.title, text: `${payload.pair_name || currentSelectedPairId} - No Data`}, series: initialChartOptions.series.map(s => ({...s, data:[]})) };
                    if (!activeChart) { activeChart = new ApexCharts(document.querySelector("#chart"), noDataOpts); activeChart.render(); } 
                    else { activeChart.updateOptions(noDataOpts); }
                    lastKnownDataTimestamp = payload.last_updated_tv || null;
                    document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data (empty) @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "No data";
                    document.getElementById('strategyInfoLabel').textContent = "Status: Data Kosong";
                    isLoadingData = false; return;
                }
                if (payload.last_updated_tv && payload.last_updated_tv === lastKnownDataTimestamp) { console.log("Chart data unchanged."); document.getElementById('lastUpdatedLabel').textContent = `Last @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`; 
                    const si = payload.strategy_state_info || {}; let it = `ST: ${si.supertrend_value?.toFixed(3) || 'N/A'}\nDir: ${si.supertrend_direction || 'N/A'}`; document.getElementById('strategyInfoLabel').textContent = it;
                    isLoadingData = false; return; 
                }
                lastKnownDataTimestamp = payload.last_updated_tv;
                document.getElementById('lastUpdatedLabel').textContent = `Last @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                
                const si = payload.strategy_state_info || {}; let it = `ST: ${si.supertrend_value?.toFixed(3) || 'N/A'}\nDir: ${si.supertrend_direction == 1 ? 'Up' : (si.supertrend_direction == -1 ? 'Down' : 'N/A')}`; document.getElementById('strategyInfoLabel').textContent = it;

                // Untuk Supertrend dengan warna per segmen, kita butuh series terpisah atau custom render.
                // Simplifikasi: Kirim data ST sebagai satu line, warna diatur di backend (sudah dilakukan).
                // ApexCharts tidak native support warna per titik di line series, tapi kita bisa update series.colors jika mau.
                // Atau, yang lebih baik, buat multiple series untuk ST jika warna sangat penting. Untuk sekarang, satu line dengan warna default.
                const stLabel = `Supertrend (${payload.st_atr_len_label}/${payload.st_factor_label})`;
                // Ambil data y dan warna dari supertrend_series
                const stLineData = payload.supertrend_series.map(d => ({x: d.x, y: d.y}));
                // Untuk warna, kita bisa pakai series.colors, tapi itu untuk seluruh series.
                // Atau, bisa modifikasi `stroke.colors` array agar sesuai panjang data ST. Ini lebih kompleks.
                // Cara termudah adalah membuat series ST terpisah untuk UP dan DOWN.

                // Untuk simplifikasi dan sesuai permintaan "jangan ubah yg lain-lain" secara drastis,
                // kita plot ST sebagai satu garis. Warna dari `supertrend_series[0].color` jika ada, atau default.
                let stSeriesColor = initialChartOptions.colors[1]; // Default
                if (payload.supertrend_series && payload.supertrend_series.length > 0 && payload.supertrend_series[payload.supertrend_series.length-1].color) {
                     stSeriesColor = payload.supertrend_series[payload.supertrend_series.length-1].color; // Warna ST dari candle terakhir
                }

                const newOpts = { ...initialChartOptions, title: { ...initialChartOptions.title, text: `${payload.pair_name} - Supertrend` },
                    series: [ { name: 'Candlestick', type: 'candlestick', data: payload.ohlc || [] }, { name: stLabel, type: 'line', data: stLineData } ], // Kirim hanya y untuk ST line
                    annotations: { yaxis: payload.annotations_yaxis || [], points: payload.annotations_points || [] },
                    colors: [initialChartOptions.colors[0], stSeriesColor] // Warna candlestick dan warna ST terakhir
                };
                if (!activeChart) { activeChart = new ApexCharts(document.querySelector("#chart"), newOpts); activeChart.render(); } 
                else { activeChart.updateOptions(newOpts); }
            } catch (e) { console.error("Err chart data:", e); if(activeChart)activeChart.destroy();activeChart=null; document.getElementById('chart').innerHTML = `Err: ${e.message}`; } 
            finally { isLoadingData = false; }
        }
        document.addEventListener('DOMContentLoaded', () => {
            if (!activeChart) { activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions); activeChart.render(); }
            fetchAvailablePairs();
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = setInterval(async () => { if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) await loadChartDataForCurrentPair(); }, 15000);
        });
    </script>
</body></html>
"""
# Endpoint Flask (get_available_pairs, get_chart_data_for_frontend, run_flask_server_thread) sama seperti sebelumnya
# ... (Isi fungsi-fungsi Flask sama persis, pastikan `prepare_chart_data_for_pair` dipanggil dengan benar) ...
@flask_app_instance.route('/')
def serve_index_page(): return render_template_string(HTML_CHART_TEMPLATE)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs():
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs_info = []
    for pair_id, pair_data in data_manager_view.items():
        cfg = pair_data.get("config", {})
        if cfg.get("enabled", True): active_pairs_info.append({"id": pair_id, "name": cfg.get('pair_name', pair_id)})
    return jsonify(active_pairs_info)

@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request):
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager: return jsonify({"error": "Pair not found"}), 404
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))
    if not pair_data_snapshot: return jsonify({"error": "Data empty for pair"}), 200 # Kirim 200 agar frontend handle
    temp_manager = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    if not prepared_data.get("ohlc"): return jsonify({"error": "No OHLC data to display", **prepared_data}), 200
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e: log_error(f"Flask server gagal: {e}", pair_name="SYSTEM_CHART")

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)
    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key valid. Tidak dapat memulai.{AnsiColors.ENDC}"); input("Enter..."); return
    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}"); input("Enter..."); return

    animated_text_display("=========== SUPERTREND STRATEGY START (Multi-Pair) ===========", color=AnsiColors.HEADER)
    # ... (Logika inisialisasi local_crypto_data_manager, fetch data awal, warm-up state, dan loop utama sama strukturnya)
    # (PENTING: Pastikan `run_strategy_logic` yang dipanggil adalah versi Supertrend)
    # (PENTING: Pastikan min_len_for_logic_warmup dan min_len_for_logic_run_live sesuai dengan kebutuhan Supertrend)
    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        animated_text_display(f"\nMenginisialisasi {config['pair_name']}...", color=AnsiColors.MAGENTA)
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_indicators_init = config.get('atr_length', 10) + 50 # ATR + buffer
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        initial_candles, initial_fetch_successful = [], False
        # ... (Loop fetch data awal sama, menggunakan initial_candles_target)
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: log_error(f"BIG DATA: Semua API key habis untuk {config['pair_name']}.", pair_name=config['pair_name']); break
            try:
                log_info(f"BIG DATA: Fetching {initial_candles_target} untuk {config['pair_name']}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key, config['timeframe'], pair_name=config['pair_name'])
                initial_fetch_successful = True
            except APIKeyError: 
                if not api_key_manager.switch_to_next_key(): break
                retries_done_initial +=1
            except Exception as e_init_fetch: log_error(f"BIG DATA: Error fetch {config['pair_name']}: {e_init_fetch}.", pair_name=config['pair_name']); break
        if not initial_candles and not initial_fetch_successful: log_error(f"BIG DATA: Gagal fetch awal {config['pair_name']}.", pair_name=config['pair_name']); continue
        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])
        
        # Warm-up
        if initial_candles:
            min_len_for_warmup = config.get('atr_length', 10) + 2 # Minimal untuk ST dan cross
            if len(initial_candles) >= min_len_for_warmup:
                log_info(f"Warm-up state untuk {config['pair_name']}...", pair_name=config['pair_name'])
                for i in range(min_len_for_warmup -1, len(initial_candles) - 1):
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_warmup: continue
                    temp_state = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state["position_size"] = 0 # No trading during warm-up
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state, global_settings_dict)
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: # Reset jika ada posisi palsu
                        local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0
                        local_crypto_data_manager[pair_id]["strategy_state"]["entry_price_custom"] = None
                log_info(f"Warm-up {config['pair_name']} selesai.", pair_name=config['pair_name'])
        
        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            # ... (Logika notifikasi email Big Data tercapai, sama seperti sebelumnya)
            log_info(f"TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}.", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(f"Data Complete: {config['pair_name']}", f"Data download complete for {TARGET_BIG_DATA_CANDLES} candles for {config['pair_name']}.", {**config, 'pair_name': config['pair_name']})
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"---------- MULAI LIVE ANALYSIS ({config['pair_name']}) ----------", pair_name=config['pair_name'])
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}--- SEMUA PAIR DIINISIALISASI ---{AnsiColors.ENDC}", color=AnsiColors.HEADER)
    
    try: # Main loop
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False
            # ... (Loop utama untuk setiap pair, fetch data, jalankan run_strategy_logic, sleep) ...
            # (Struktur loop sama, hanya pastikan `min_len_for_logic_run_live` disesuaikan untuk Supertrend)
            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']
                # ... (Cooldown logic sama)
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600:
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600); continue
                    else: data_per_pair["data_fetch_failed_consecutively"] = 0
                
                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Interval fetch lebih agresif saat big data
                    required_interval = 60 if config_for_pair.get('timeframe') == "minute" else 3600 
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch); continue
                
                log_info(f"Memproses {pair_name_for_log}...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                # ... (Fetch update logic sama, pastikan limit_fetch_for_update sesuai)
                new_candles_batch, fetch_update_successful = [], False
                limit_fetch = 3
                if data_per_pair["big_data_collection_phase_active"]:
                    needed = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed <= 0: fetch_update_successful = True # Sudah cukup
                    else: limit_fetch = min(needed, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch > 0 or data_per_pair["big_data_collection_phase_active"]: # Hanya fetch jika perlu
                    # ... (Loop retry fetch update sama)
                    max_retries_update = api_key_manager.total_keys() or 1
                    retries_update = 0
                    while retries_update < max_retries_update and not fetch_update_successful:
                        key_update = api_key_manager.get_current_key()
                        if not key_update: break # No more keys globally
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch, config_for_pair['exchange'], key_update, config_for_pair['timeframe'], pair_name_for_log)
                            fetch_update_successful = True; data_per_pair["data_fetch_failed_consecutively"] = 0; any_data_fetched_this_cycle = True
                        except APIKeyError:
                            data_per_pair["data_fetch_failed_consecutively"] += 1
                            if not api_key_manager.switch_to_next_key(): break
                            retries_update +=1
                        except Exception: data_per_pair["data_fetch_failed_consecutively"] +=1; break # Network error, etc.
                # ... (Handle jika semua key gagal untuk pair ini, merge candle, cek Big Data tercapai)
                if data_per_pair.get("data_fetch_failed_consecutively",0) >= (api_key_manager.total_keys() or 1) +1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now()
                if not fetch_update_successful or not new_candles_batch:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval); continue
                
                # Merge
                merged_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added, updated_count = 0,0
                for c_new in new_candles_batch:
                    ts_new = c_new['timestamp']
                    if ts_new not in merged_dict: merged_dict[ts_new] = c_new; newly_added+=1
                    elif merged_dict[ts_new] != c_new : merged_dict[ts_new] = c_new; updated_count+=1
                data_per_pair["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                if newly_added + updated_count > 0: log_info(f"{newly_added+updated_count} candle baru/update untuk {pair_name_for_log}.",pair_name_for_log)

                # Cek Big Data (display)
                if data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                    # ... (Logika notifikasi Big Data tercapai sama)
                    data_per_pair["big_data_collection_phase_active"] = False
                    active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                elif not data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES:
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                
                # Jalankan Logika
                min_len_for_logic_live = config_for_pair.get('atr_length', 10) + 2
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_live:
                    process_now = (newly_added + updated_count > 0 or 
                                   (not data_per_pair["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) )
                    if process_now:
                        log_info(f"Run Supertrend logic for {pair_name_for_log}...", pair_name_for_log)
                        data_per_pair["strategy_state"] = run_strategy_logic(data_per_pair["all_candles_list"], config_for_pair, data_per_pair["strategy_state"], global_settings_dict)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            # Sleep logic (sama)
            sleep_duration = 15
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_duration = 3600
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 30 # Cek lebih sering jika masih big data
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
            if sleep_duration > 0: show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: time.sleep(1)

    except KeyboardInterrupt: log_info("Proses dihentikan.", pair_name="SYSTEM")
    except Exception as e: log_exception("Error loop utama:", pair_name="SYSTEM")
    finally: animated_text_display("STRATEGY STOP", color=AnsiColors.HEADER); input("Enter...")


# --- MENU UTAMA (Struktur sama, judul diubah) ---
def main_menu():
    settings = load_settings()
    flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True); flask_thread.start()
    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Supertrend Strategy Runner =========", color=AnsiColors.HEADER)
        # ... (Tampilan menu utama sama, hanya judul strategi yang berubah)
        pick_title_main = "" # Sama
        active_configs_list = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs_list:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs_list)}) ---\n"
            for i, cfg_item in enumerate(active_configs_list): pick_title_main += f"  {i+1}. {cfg_item.get('symbol','N/A')}-{cfg_item.get('currency','N/A')}\n"
        else: pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"
        # ... (Info API Key, Termux, Chart Server sama)
        pick_title_main += "-----------------------------------------------\n"
        api_s_main = settings.get("api_settings", {})
        pkd_main = api_s_main.get('primary_key', 'BELUM DIATUR'); pkd_main = "..." + pkd_main[-5:] if len(pkd_main) > 10 and pkd_main not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"] else pkd_main
        nrk_main = len([k for k in api_s_main.get('recovery_keys',[]) if k])
        tn_main = "Aktif" if api_s_main.get("enable_termux_notifications", False) else "Nonaktif"
        pick_title_main += f"Target Data Display: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {pkd_main} | Recovery: {nrk_main}\n"
        pick_title_main += f"Notif Termux: {tn_main}\nChart: http://localhost:5001\n"
        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"
        main_menu_options_plain = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        selected_main_text, selected_main_index = None, -1
        try: selected_main_text, selected_main_index = pick(main_menu_options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception: # Fallback
            print(pick_title_main)
            for idx_main, opt_text_main in enumerate(main_menu_options_plain): print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main_val = int(input("Pilih nomor opsi: ").strip()) -1
                if 0 <= choice_main_val < len(main_menu_options_plain): selected_main_index = choice_main_val
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue

        if selected_main_index == 0: settings = load_settings(); start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif selected_main_index == 1: settings = settings_menu(settings)
        elif selected_main_index == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Sampai jumpa!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan.", color=AnsiColors.ORANGE)
    except Exception as e:
        clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL: {e}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL CRITICAL ERROR:", pair_name="SYSTEM_CRITICAL")
        input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
