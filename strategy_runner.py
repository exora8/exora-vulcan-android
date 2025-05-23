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

log_file_name = "trading_log_supertrend.txt" # Nama log disesuaikan
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


SETTINGS_FILE = "settings_supertrend.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 250 
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
            self.keys.extend([k for k in recovery_keys_list if k]) # Hanya tambahkan key yang tidak kosong

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}

        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys:
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None # Indeks di luar jangkauan (semua key sudah dicoba dalam siklus ini)

    def switch_to_next_key(self):
        if not self.keys or len(self.keys) == 1 and self.current_index == 0 : # Jika hanya 1 key dan sudah dipakai, atau tidak ada key
             if self.current_index >= len(self.keys) -1 : # Jika sudah di key terakhir atau tidak ada key lagi
                # log_info(f"{AnsiColors.ORANGE}Sudah mencapai akhir daftar API key atau hanya ada 1 key. Tidak ada key berikutnya untuk diganti.{AnsiColors.ENDC}")
                return None # Tidak ada key berikutnya

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
        else: # self.current_index >= len(self.keys)
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data lebih lanjut dengan key baru di siklus ini.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "KRITIS: SEMUA API Key CryptoCompare Gagal!"
                email_body = (f"Skrip trading telah mencoba semua API key CryptoCompare yang tersedia (primary dan recovery) dan semuanya gagal.\n\n"
                              f"Skrip tidak dapat lagi mengambil data harga dengan key baru sampai siklus API key direset (jika ada mekanisme reset eksternal atau restart skrip).\n"
                              f"Harap segera periksa akun CryptoCompare Anda dan konfigurasi API key di skrip.")
                dummy_email_cfg = { # Sama seperti di atas
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal (APIKeyManager).")
            return None # Tidak ada key valid tersisa untuk diganti

    def reset_key_index_to_primary(self):
        """Mereset index ke key utama (0). Berguna jika ingin memulai siklus key dari awal."""
        # log_info(f"{AnsiColors.CYAN}Indeks API Key direset ke utama (Index 0).{AnsiColors.ENDC}")
        self.current_index = 0

    def has_valid_keys(self):
        return bool(self.keys)

    def total_keys(self):
        return len(self.keys)

    def get_current_key_index(self):
        return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION (Sama seperti sebelumnya) ---
def play_notification_sound():
    try:
        if sys.platform == "win32": import winsound; winsound.Beep(1000, 500)
        else: print('\a', end='', flush=True)
    except Exception as e: log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False): return
    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")
    pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
    if not all([sender_email, sender_password, receiver_email]):
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=pair_name_ctx); return
    msg = MIMEText(body_text); msg['Subject'] = subject; msg['From'] = sender_email; msg['To'] = receiver_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password); smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)

def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    api_settings = global_settings.get("api_settings", {})
    if not api_settings.get("enable_termux_notifications", False): return
    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError: log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired: log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN (Sama seperti sebelumnya, get_default_crypto_config dan _prompt_crypto_config sudah Supertrend) ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "hour", "refresh_interval_seconds": 60,
        "atr_length": 10, "factor": 3.0,
        "emergency_sl_percent": 2.0, "profit_target_percent_activation": 2.0, "trailing_stop_gap_percent": 1.0,
        "ma_length": 50, "stoch_length": 14, "stoch_smooth_k": 3, "stoch_smooth_d": 3,
        "stoch_overbought": 80, "stoch_oversold": 20,
        "left_strength": 50, "right_strength": 150, "enable_secure_fib": True, "secure_fib_check_price": "Close",
        "enable_email_notifications": False,
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }
def load_settings(): # Sama
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
def save_settings(settings): # Sama
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")
def _prompt_crypto_config(current_config): # Sama, sudah Supertrend
    clear_screen_animated()
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)
    enabled_input = input(f"Aktifkan analisa? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))
    new_config["symbol"] = (input(f"Simbol Crypto [{new_config.get('symbol','BTC')}]: ") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"Mata Uang Quote [{new_config.get('currency','USD')}]: ") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"Exchange [{new_config.get('exchange','CCCAGG')}]: ") or new_config.get('exchange','CCCAGG')).strip()
    tf_input = (input(f"Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: ") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    refresh_input_str = input(f"Interval Refresh (detik) [{new_config.get('refresh_interval_seconds',60)}]: ").strip()
    try: new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60))
    except ValueError: new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))
    animated_text_display("\n-- Parameter Supertrend --", color=AnsiColors.HEADER)
    try:
        new_config["atr_length"] = int(input(f"ATR Length [{new_config.get('atr_length',10)}]: ") or new_config.get('atr_length',10))
        new_config["factor"] = float(input(f"Factor [{new_config.get('factor',3.0)}]: ") or new_config.get('factor',3.0))
    except ValueError: print(f"{AnsiColors.RED}Input Supertrend tidak valid.{AnsiColors.ENDC}") # Default akan dipertahankan
    animated_text_display("\n-- Parameter Exits --", color=AnsiColors.HEADER)
    try:
        new_config["emergency_sl_percent"] = float(input(f"Fixed Stop Loss % [{new_config.get('emergency_sl_percent',2.0)}]: ") or new_config.get('emergency_sl_percent',2.0))
        new_config["profit_target_percent_activation"] = float(input(f"Trail Activation Profit % [{new_config.get('profit_target_percent_activation',2.0)}]: ") or new_config.get('profit_target_percent_activation',2.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"Trail Gap % [{new_config.get('trailing_stop_gap_percent',1.0)}]: ") or new_config.get('trailing_stop_gap_percent',1.0))
    except ValueError: print(f"{AnsiColors.RED}Input Exits tidak valid.{AnsiColors.ENDC}")
    # Email (sama)
    animated_text_display("\n-- Notifikasi Email --", color=AnsiColors.HEADER)
    email_enable_input = input(f"Aktifkan Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = (input(f"Email Pengirim [{new_config.get('email_sender_address','')}]: ") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"App Password Pengirim [{new_config.get('email_sender_app_password','')}]: ") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"Email Penerima [{new_config.get('email_receiver_address','')}]: ") or new_config.get('email_receiver_address','')).strip()
    return new_config
def settings_menu(current_settings): # Sama, sudah Supertrend
    # ... (Struktur menu pick sama, _prompt_crypto_config akan panggil versi Supertrend) ...
    # (Isi fungsi ini sama persis dengan versi Supertrend sebelumnya, tidak perlu diulang di sini)
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
        pick_title_settings += "Strategi Aktif: Supertrend Entry\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"
        if not current_settings.get("cryptos"): pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\nPilih tindakan:"
        original_options_structure = [("header", "--- API & Global ---"),("option", "Atur Primary API Key"), ("option", "Kelola Recovery API Keys"),("option", "Atur Email Global Notifikasi Sistem"), ("option", "Notifikasi Termux Realtime"),("header", "--- Crypto Pair ---"),("option", "Tambah Konfigurasi Crypto"), ("option", "Ubah Konfigurasi Crypto"),("option", "Hapus Konfigurasi Crypto"),("header", "-------------------"),("option", "Kembali ke Menu Utama")]
        selectable_options = [text for type, text in original_options_structure if type == "option"]
        selected_option_text, action_choice = None, -1
        try: selected_option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception: print(pick_title_settings); [print(f"  {idx + 1}. {opt}") for idx, opt in enumerate(selectable_options)]; choice = input("Pilih: "); action_choice = int(choice)-1 if choice.isdigit() and 0 <= int(choice)-1 < len(selectable_options) else -1

        try:
            clear_screen_animated()
            if action_choice == 0: api_s["primary_key"] = (input(f"Primary API Key baru [{api_s.get('primary_key','')}]: ") or api_s.get('primary_key','')).strip(); current_settings["api_settings"] = api_s
            elif action_choice == 1: # Kelola Recovery
                while True:
                    clear_screen_animated(); recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"; current_recovery = [k for k in api_s.get('recovery_keys', []) if k]; api_s['recovery_keys'] = current_recovery
                    if not current_recovery: recovery_pick_title += "  (Kosong)\n"
                    else: [recovery_pick_title := recovery_pick_title + f"  {i+1}. {k[:5]}...{k[-3:] if len(k)>8 else k}\n" for i, k in enumerate(current_recovery)]
                    recovery_pick_title += "\nPilih:"
                    rec_opts = ["Tambah", "Hapus", "Kembali"]; _, rec_idx = pick(rec_opts, recovery_pick_title, indicator='=>')
                    clear_screen_animated()
                    if rec_idx == 0: new_r_key = input("Recovery Key baru: ").strip(); current_recovery.append(new_r_key) if new_r_key else None
                    elif rec_idx == 1 and current_recovery: del_opts = [f"{k[:5]}...{k[-3:]}" for k in current_recovery]+["Batal"]; _, idx_del = pick(del_opts,"Hapus key:",indicator='=>'); current_recovery.pop(idx_del) if idx_del < len(current_recovery) else None
                    elif rec_idx == 2: break
                    api_s['recovery_keys'] = current_recovery; save_settings(current_settings); show_spinner(1,"...")
            elif action_choice == 2: # Email Global
                api_s['enable_global_email_notifications_for_key_switch'] = (input(f"Email global aktif? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower() == 'true')
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ") or api_s.get('email_sender_address','')).strip()
                api_s['email_sender_app_password'] = (input(f"App Password Global [{api_s.get('email_sender_app_password','')}]: ") or api_s.get('email_sender_app_password','')).strip()
                api_s['email_receiver_address_admin'] = (input(f"Email Admin [{api_s.get('email_receiver_address_admin','')}]: ") or api_s.get('email_receiver_address_admin','')).strip()
                current_settings["api_settings"] = api_s
            elif action_choice == 3: api_s['enable_termux_notifications'] = (input(f"Notif Termux? (true/false) [{api_s.get('enable_termux_notifications',False)}]: ").lower() == 'true'); current_settings["api_settings"] = api_s
            elif action_choice == 4: current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_choice == 5 and current_settings.get("cryptos"): edit_opts = [f"{c.get('symbol','N/A')}-{c.get('currency','N/A')}" for c in current_settings["cryptos"]]+["Batal"]; _, idx_edit = pick(edit_opts, "Ubah:",indicator='=>'); current_settings["cryptos"][idx_edit] = _prompt_crypto_config(current_settings["cryptos"][idx_edit]) if idx_edit < len(current_settings["cryptos"]) else None
            elif action_choice == 6 and current_settings.get("cryptos"): del_c_opts = [f"{c.get('symbol','N/A')}-{c.get('currency','N/A')}" for c in current_settings["cryptos"]]+["Batal"]; _, idx_del_c = pick(del_c_opts, "Hapus:",indicator='=>'); current_settings["cryptos"].pop(idx_del_c) if idx_del_c < len(current_settings["cryptos"]) else None
            elif action_choice == 7: break
            save_settings(current_settings); show_spinner(1,"Menyimpan...")
        except Exception as e_settings: log_error(f"Error menu: {e_settings}"); show_spinner(1.5, "Error...")
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

        params = {"fsym": symbol, "tsym": currency, "limit": limit_for_this_api_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                key_disp = ("..." + current_api_key_to_use[-3:]) if len(current_api_key_to_use) > 8 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: {key_disp}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)
            response = requests.get(url, params=params, timeout=20)

            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}; 
                try: error_data = response.json()
                except json.JSONDecodeError: pass
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display_err = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: {key_display_err}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}") # Memicu penggantian key

            response.raise_for_status() # Untuk error HTTP lainnya
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'Pesan error tidak tersedia dari API.')
                # Menggunakan list pesan error yang lebih komprehensif dari skrip asli Anda
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit",
                    "please pass an API key", "api_key not found"
                ]
                key_display_err_json = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: {key_display_err_json}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}") # Memicu penggantian key
                else:
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare (lainnya): {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Hentikan fetch untuk pair ini jika error API tidak terkait key

            # Proses data candle (sama seperti sebelumnya)
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break
            raw_candles_from_api = data['Data']['Data']
            if not raw_candles_from_api: break
            batch_candles_list = []
            for item in raw_candles_from_api:
                req_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in req_keys): continue
                batch_candles_list.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list and batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                batch_candles_list.pop()
            if not batch_candles_list and current_to_ts is not None: break
            all_accumulated_candles = batch_candles_list + all_accumulated_candles
            if raw_candles_from_api: current_to_ts = raw_candles_from_api[0]['time']
            else: break
            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
            if len(raw_candles_from_api) < limit_for_this_api_call: break
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.3)

        except APIKeyError: # Tangkap APIKeyError untuk diteruskan ke atas
            raise
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Hentikan fetch untuk batch ini, coba lagi di siklus berikutnya jika diizinkan
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles (batch loop): {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error Fetch Candles (batch loop):", pair_name=pair_name)
            break

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)
    return all_accumulated_candles

# --- LOGIKA STRATEGI (Supertrend, sama seperti sebelumnya) ---
def get_initial_strategy_state(): # Sama
    return {
        "last_supertrend_value": None, "supertrend_direction": 0, "last_atr_value_for_chart": None,
        "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
        "last_ma_value_for_chart": None, "last_stoch_k_value_for_chart": None, "last_stoch_d_value_for_chart": None,
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None, "last_pivot_low_display_info": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None, "active_fib_line_start_index": None,
    }
def calculate_atr(high_prices, low_prices, close_prices, period): # Sama
    if len(close_prices) < period or period <= 0: return [None] * len(close_prices)
    tr_values, atr_values = [None] * len(close_prices), [None] * len(close_prices)
    for i in range(len(close_prices)):
        if high_prices[i] is None or low_prices[i] is None or close_prices[i] is None: continue
        high_low = high_prices[i] - low_prices[i]
        if i > 0 and close_prices[i-1] is not None:
            tr_values[i] = max(high_low, abs(high_prices[i] - close_prices[i-1]), abs(low_prices[i] - close_prices[i-1]))
        elif i == 0 : tr_values[i] = high_low
    first_valid_tr_index = next((i for i, tr in enumerate(tr_values) if tr is not None), -1)
    if first_valid_tr_index == -1 or sum(1 for tr in tr_values[first_valid_tr_index:] if tr is not None) < period: return atr_values
    sum_first_trs, count_first_trs, atr_start_index = 0,0,-1
    temp_idx_walk = first_valid_tr_index
    while count_first_trs < period and temp_idx_walk < len(tr_values):
        if tr_values[temp_idx_walk] is not None: sum_first_trs += tr_values[temp_idx_walk]; count_first_trs +=1
        if count_first_trs == period: atr_start_index = temp_idx_walk; break
        temp_idx_walk +=1
    if atr_start_index != -1 and count_first_trs == period:
        atr_values[atr_start_index] = sum_first_trs / period
        for i in range(atr_start_index + 1, len(close_prices)):
            if tr_values[i] is not None and atr_values[i-1] is not None:
                atr_values[i] = (atr_values[i-1] * (period - 1) + tr_values[i]) / period
    return atr_values
def calculate_supertrend(high_prices, low_prices, close_prices, atr_length, factor): # Sama
    num_candles = len(close_prices)
    if num_candles < atr_length + 1: return ([None] * num_candles, [0] * num_candles)
    atr_values = calculate_atr(high_prices, low_prices, close_prices, atr_length)
    supertrend_values, direction_values = [None] * num_candles, [0] * num_candles
    first_valid_candle_idx = next((i for i in range(atr_length, num_candles) if all(p is not None for p in [close_prices[i], high_prices[i], low_prices[i], atr_values[i]])), -1)
    if first_valid_candle_idx == -1: return (supertrend_values, direction_values)
    mid_init = (high_prices[first_valid_candle_idx] + low_prices[first_valid_candle_idx]) / 2
    if close_prices[first_valid_candle_idx] <= mid_init:
        supertrend_values[first_valid_candle_idx] = mid_init + factor * atr_values[first_valid_candle_idx]
        direction_values[first_valid_candle_idx] = -1
    else:
        supertrend_values[first_valid_candle_idx] = mid_init - factor * atr_values[first_valid_candle_idx]
        direction_values[first_valid_candle_idx] = 1
    for i in range(first_valid_candle_idx + 1, num_candles):
        if any(p is None for p in [close_prices[i], high_prices[i], low_prices[i], atr_values[i], close_prices[i-1], supertrend_values[i-1]]):
            supertrend_values[i] = None; direction_values[i] = direction_values[i-1]; continue
        basic_upper = ((high_prices[i] + low_prices[i]) / 2) + factor * atr_values[i]
        basic_lower = ((high_prices[i] + low_prices[i]) / 2) - factor * atr_values[i]
        prev_st, prev_dir = supertrend_values[i-1], direction_values[i-1]
        curr_st, curr_dir = None, prev_dir
        if prev_dir == 1: curr_st = max(basic_lower, prev_st); curr_dir, curr_st = (-1, basic_upper) if close_prices[i] < curr_st else (curr_dir, curr_st)
        elif prev_dir == -1: curr_st = min(basic_upper, prev_st); curr_dir, curr_st = (1, basic_lower) if close_prices[i] > curr_st else (curr_dir, curr_st)
        else: mid_curr = (high_prices[i] + low_prices[i]) / 2; curr_st, curr_dir = (basic_upper,-1) if close_prices[i]<=mid_curr else (basic_lower,1)
        supertrend_values[i], direction_values[i] = curr_st, curr_dir
    return supertrend_values, direction_values
def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings): # Sama
    # ... (Isi fungsi run_strategy_logic Supertrend sama persis seperti versi Supertrend sebelumnya) ...
    # (Tidak perlu diulang di sini, hanya pastikan panggilannya benar)
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    atr_len = crypto_config.get('atr_length', 10); st_factor = crypto_config.get('factor', 3.0)
    sl_percentage = crypto_config.get('emergency_sl_percent', 2.0) / 100.0
    tp_activation_percentage = crypto_config.get('profit_target_percent_activation', 2.0) / 100.0
    tp_trailing_gap_percentage = crypto_config.get('trailing_stop_gap_percent', 1.0) / 100.0
    min_data_needed = atr_len + 2
    if len(candles_history) < min_data_needed: return strategy_state
    close_prices = [c.get('close') for c in candles_history]; high_prices = [c.get('high') for c in candles_history]; low_prices = [c.get('low') for c in candles_history]
    supertrend_line, supertrend_direction_signal = calculate_supertrend(high_prices, low_prices, close_prices, atr_len, st_factor)
    current_candle_idx = len(candles_history) - 1; prev_candle_idx = current_candle_idx - 1
    current_candle = candles_history[current_candle_idx]
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']): return strategy_state
    current_close = current_candle['close']; current_low = current_candle['low']; current_high = current_candle['high']
    current_st_value = supertrend_line[current_candle_idx]; current_st_direction = supertrend_direction_signal[current_candle_idx]
    strategy_state["last_supertrend_value"] = current_st_value; strategy_state["supertrend_direction"] = current_st_direction
    prev_close, prev_st_value = None, None
    if prev_candle_idx >= 0:
        prev_close = candles_history[prev_candle_idx].get('close')
        prev_st_value = supertrend_line[prev_candle_idx]
    if current_st_value is None or prev_st_value is None or prev_close is None: pass # Skip cross logic
    else:
        crossed_above_st = prev_close <= prev_st_value and current_close > current_st_value
        entry_long_condition = crossed_above_st and current_st_direction == 1
        crossed_below_st = prev_close >= prev_st_value and current_close < current_st_value
        exit_long_condition = crossed_below_st
        if strategy_state["position_size"] > 0:
            if exit_long_condition: # Supertrend exit
                exit_price = current_close; entry_price_val = strategy_state["entry_price_custom"]
                pnl = ((exit_price - entry_price_val) / entry_price_val) * 100.0 if entry_price_val else 0.0
                log_info(f"{AnsiColors.BLUE}EXIT (ST Cross) @ {exit_price:.5f}. PnL: {pnl:.2f}%{AnsiColors.ENDC}", pair_name=pair_name); play_notification_sound()
                # Notif (disingkat)
                send_termux_notification(f"EXIT (ST): {pair_name}", f"ST Cross @ {exit_price:.5f}", global_settings, pair_name)
                send_email_notification(f"Trade Closed (ST): {pair_name}", f"ST Exit @ {exit_price:.5f}, PnL {pnl:.2f}%", {**crypto_config, 'pair_name': pair_name})
                strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None; strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False; strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
        elif strategy_state["position_size"] == 0:
            if entry_long_condition: # Supertrend entry
                entry_price = current_close; strategy_state["position_size"] = 1; strategy_state["entry_price_custom"] = entry_price
                strategy_state["emergency_sl_level_custom"] = entry_price * (1 - sl_percentage); strategy_state["highest_price_for_trailing"] = current_high
                log_info(f"{AnsiColors.GREEN}BUY (ST) @ {entry_price:.5f}. SL: {strategy_state['emergency_sl_level_custom']:.5f}{AnsiColors.ENDC}", pair_name=pair_name); play_notification_sound()
                # Notif (disingkat)
                send_termux_notification(f"BUY (ST): {pair_name}", f"Entry @ {entry_price:.5f}, SL {strategy_state['emergency_sl_level_custom']:.5f}", global_settings, pair_name)
                send_email_notification(f"BUY Signal (ST): {pair_name}", f"Entry @ {entry_price:.5f}, SL {strategy_state['emergency_sl_level_custom']:.5f}", {**crypto_config, 'pair_name': pair_name})
            elif exit_long_condition: log_info(f"{AnsiColors.MAGENTA}SELL Alert (ST) @ {current_close:.5f}{AnsiColors.ENDC}", pair_name=pair_name) # Hanya alert
    # SL/TP logic (sama)
    if strategy_state["position_size"] > 0:
        entry_price_val = strategy_state["entry_price_custom"]
        if current_high: strategy_state["highest_price_for_trailing"] = max(strategy_state.get("highest_price_for_trailing") or current_high, current_high)
        if not strategy_state["trailing_tp_active_custom"] and entry_price_val and strategy_state["highest_price_for_trailing"]:
            if ((strategy_state["highest_price_for_trailing"] - entry_price_val) / entry_price_val) >= tp_activation_percentage:
                strategy_state["trailing_tp_active_custom"] = True; log_info(f"{AnsiColors.BLUE}Trailing TP Aktif.{AnsiColors.ENDC}", pair_name=pair_name)
        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"]:
            potential_stop = strategy_state["highest_price_for_trailing"] * (1 - tp_trailing_gap_percentage)
            strategy_state["current_trailing_stop_level"] = max(strategy_state.get("current_trailing_stop_level") or potential_stop, potential_stop)
        effective_sl = strategy_state["emergency_sl_level_custom"]; exit_reason = "SL"; exit_color = AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"]:
            if effective_sl is None or strategy_state["current_trailing_stop_level"] > effective_sl:
                effective_sl = strategy_state["current_trailing_stop_level"]; exit_reason = "TrailTP"; exit_color = AnsiColors.BLUE
        if effective_sl and current_low and current_low <= effective_sl:
            exit_price = effective_sl; pnl = ((exit_price - entry_price_val) / entry_price_val) * 100.0 if entry_price_val else 0.0
            if exit_reason == "TrailTP" and pnl < 0: exit_color = AnsiColors.RED; exit_reason = "TrailSL (Loss)"
            log_info(f"{exit_color}EXIT ({exit_reason}) @ {exit_price:.5f}. PnL: {pnl:.2f}%{AnsiColors.ENDC}", pair_name=pair_name); play_notification_sound()
            # Notif (disingkat)
            send_termux_notification(f"EXIT ({exit_reason}): {pair_name}", f"{exit_reason} @ {exit_price:.5f}", global_settings, pair_name)
            send_email_notification(f"Trade Closed ({exit_reason}): {pair_name}", f"{exit_reason} @ {exit_price:.5f}, PnL {pnl:.2f}%", {**crypto_config, 'pair_name': pair_name})
            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None; strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False; strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
    return strategy_state

# CHART_INTEGRATION (Sama seperti sebelumnya, prepare_chart_data_for_pair dan HTML template sudah Supertrend)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()
def prepare_chart_data_for_pair(pair_id, snapshot): # Sama
    # ... (Isi fungsi ini sama persis dengan versi Supertrend sebelumnya, tidak perlu diulang di sini) ...
    if pair_id not in snapshot: return None
    pair_data = snapshot[pair_id]; candles_hist = pair_data.get("all_candles_list", []); state = pair_data.get("strategy_state", {}); cfg = pair_data.get("config", {})
    chart_candles = candles_hist[-TARGET_BIG_DATA_CANDLES:]
    ohlc, st_series = [], []
    if not chart_candles: return {"ohlc": [], "supertrend_series": [], "annotations_yaxis": [], "pair_name": cfg.get('pair_name', pair_id), "last_updated_tv": None}
    chart_closes = [c.get('close') for c in chart_candles]; chart_highs = [c.get('high') for c in chart_candles]; chart_lows = [c.get('low') for c in chart_candles]
    chart_atr, chart_factor = cfg.get('atr_length',10), cfg.get('factor',3.0)
    st_line, st_dir = calculate_supertrend(chart_highs, chart_lows, chart_closes, chart_atr, chart_factor)
    for i, c in enumerate(chart_candles):
        if all(k in c and c[k] is not None for k in ['timestamp','open','high','low','close']):
            ts_ms = c['timestamp'].timestamp()*1000
            ohlc.append({'x':ts_ms, 'y':[c['open'],c['high'],c['low'],c['close']]})
            if i < len(st_line) and st_line[i] is not None: st_series.append({'x':ts_ms, 'y':st_line[i], 'color': '#26A69A' if st_dir[i]==1 else ('#EF5350' if st_dir[i]==-1 else '#888')})
    ann_y = [] # Anotasi SL/Entry (sama)
    if state.get("position_size",0)>0 and state.get("entry_price_custom") is not None:
        ep = state.get("entry_price_custom"); ann_y.append({'y':ep, 'borderColor':'#2698FF', 'label':{'text':f'Entry: {ep:.5f}'}})
        sl = state.get("emergency_sl_level_custom"); sl_txt="SL"
        if state.get("trailing_tp_active_custom") and state.get("current_trailing_stop_level"):
            tsl = state.get("current_trailing_stop_level"); sl = tsl if sl is None or (tsl and tsl > sl) else sl; sl_txt="TrailSL"
        if sl: ann_y.append({'y':sl, 'borderColor':'#FF4560', 'label':{'text':f'{sl_txt}: {sl:.5f}'}})
    last_tv = chart_candles[-1]['timestamp'].timestamp()*1000 if chart_candles and chart_candles[-1].get('timestamp') else None
    return {"ohlc":ohlc, "supertrend_series":st_series, "st_atr_len_label":chart_atr, "st_factor_label":chart_factor, "annotations_yaxis":ann_y, "pair_name":cfg.get('pair_name',pair_id), "last_updated_tv":last_tv, "strategy_state_info":{"supertrend_value":state.get("last_supertrend_value"), "supertrend_direction":state.get("supertrend_direction")}}
flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """<!DOCTYPE html><html><head><title>Supertrend Chart</title><script src="https://cdn.jsdelivr.net/npm/apexcharts"></script><style>body{font-family:sans-serif;margin:0;background:#1e1e1e;color:#e0e0e0;display:flex;flex-direction:column;align-items:center;padding:10px}#controls{background:#2a2a2a;padding:10px;border-radius:8px;margin-bottom:10px;display:flex;align-items:center;gap:10px;width:100%;max-width:1200px}select,button{padding:8px 12px;border-radius:5px;border:1px solid #444;background:#333;color:#e0e0e0}#chart-container{width:100%;max-width:1200px;background:#2a2a2a;padding:15px;border-radius:8px}h1{color:#00bcd4}#lastUpdatedLabel{font-size:.8em;color:#aaa;margin-left:auto}#strategyInfoLabel{font-size:.8em;color:#FFD700;margin-left:10px;white-space:pre}</style></head><body><h1>Supertrend Strategy Chart</h1><div id="controls"><label for="pairSelector">Pair:</label><select id="pairSelector" onchange="handlePairSelectionChange()"></select><button onclick="loadChartDataForCurrentPair()">Refresh</button><span id="strategyInfoLabel">Status:-</span><span id="lastUpdatedLabel">Memuat...</span></div><div id="chart-container"><div id="chart"></div></div><script>let activeChart,currentPairId='',lastTs=null,isLoading=false,intervalId=null;const initOpts={series:[{name:'Candlestick',type:'candlestick',data:[]},{name:'Supertrend',type:'line',data:[]}],chart:{type:'candlestick',height:550,background:'#2a2a2a',animations:{enabled:false},toolbar:{show:true}},theme:{mode:'dark'},title:{text:'Memuat...',align:'left'},xaxis:{type:'datetime',labels:{style:{colors:'#aaa'}},tooltip:{enabled:false}},yaxis:{tooltip:{enabled:true},labels:{style:{colors:'#aaa'},formatter:(v)=>v?v.toFixed(5):''}},stroke:{width:[1,2],curve:'straight'},markers:{size:0},colors:['#FEB019','#888'],grid:{borderColor:'#444'},annotations:{yaxis:[]},tooltip:{theme:'dark',shared:true,intersect:false,custom:({series,seriesIndex,dataPointIndex,w})=>{let o,h,l,c,st,html='<div style="padding:5px 10px;background:#333;color:#fff;border:1px solid #555;">';const csIdx=w.globals.series.findIndex(s=>s.type==='candlestick'),stIdx=w.globals.series.findIndex(s=>s.name.startsWith('Supertrend'));if(csIdx!==-1&&w.globals.seriesCandleO[csIdx]?.[dataPointIndex]!==undefined){[o,h,l,c]=[w.globals.seriesCandleO[csIdx][dataPointIndex],w.globals.seriesCandleH[csIdx][dataPointIndex],w.globals.seriesCandleL[csIdx][dataPointIndex],w.globals.seriesCandleC[csIdx][dataPointIndex]];html+=['O','H','L','C'].map((k,i)=>`<div>${k}:<b>${[o,h,l,c][i].toFixed(5)}</b></div>`).join('')}if(stIdx!==-1&&w.config.series[stIdx].data[dataPointIndex]){st=w.config.series[stIdx].data[dataPointIndex].y;html+=`<div>ST:<b>${st.toFixed(5)}</b></div>`}html+='</div>';return(o||st)?html:''}},noData:{text:'No data'}};async function fetchPairs(){try{const r=await fetch('/api/available_pairs');if(!r.ok)throw Error(r.status);const pairs=await r.json(),sel=document.getElementById('pairSelector');sel.innerHTML='';if(pairs.length>0){pairs.forEach(p=>{const o=document.createElement('option');o.value=p.id;o.textContent=p.name;sel.appendChild(o)});currentPairId=sel.value||pairs[0].id;loadChart()}else sel.innerHTML='<option value="">No pairs</option>'}catch(e){console.error(e)}}function handlePairSelectionChange(){currentPairId=document.getElementById('pairSelector').value;lastTs=null;loadChart()}async function loadChart(){if(!currentPairId||isLoading)return;isLoading=true;document.getElementById('lastUpdatedLabel').textContent=`Loading ${currentPairId}...`;try{const r=await fetch(`/api/chart_data/${currentPairId}`);if(!r.ok)throw Error(r.status);const p=await r.json();if(!p||!p.ohlc||p.ohlc.length===0){const noData={...initOpts,title:{...initOpts.title,text:`${p.pair_name||currentPairId} - No Data`},series:initOpts.series.map(s=>({...s,data:[]}))};if(!activeChart)activeChart=new ApexCharts(document.querySelector("#chart"),noData);else activeChart.updateOptions(noData);lastTs=p.last_updated_tv||null;document.getElementById('lastUpdatedLabel').textContent=lastTs?`Data(empty)@${new Date(lastTs).toLocaleTimeString()}`:"No data";document.getElementById('strategyInfoLabel').textContent="Status:Empty";isLoading=false;return}if(p.last_updated_tv&&p.last_updated_tv===lastTs){document.getElementById('lastUpdatedLabel').textContent=`Last@${new Date(lastTs).toLocaleTimeString()}`;const si=p.strategy_state_info||{};document.getElementById('strategyInfoLabel').textContent=`ST:${si.supertrend_value?.toFixed(3)||'N/A'}\\nDir:${si.supertrend_direction==1?'Up':(si.supertrend_direction==-1?'Down':'N/A')}`;isLoading=false;return}lastTs=p.last_updated_tv;document.getElementById('lastUpdatedLabel').textContent=`Last@${new Date(lastTs).toLocaleTimeString()}`;const si=p.strategy_state_info||{};document.getElementById('strategyInfoLabel').textContent=`ST:${si.supertrend_value?.toFixed(3)||'N/A'}\\nDir:${si.supertrend_direction==1?'Up':(si.supertrend_direction==-1?'Down':'N/A')}`;const stLabel=`Supertrend(${p.st_atr_len_label}/${p.st_factor_label})`;const stData=p.supertrend_series.map(d=>({x:d.x,y:d.y}));let stColor=initOpts.colors[1];if(p.supertrend_series?.length>0&&p.supertrend_series[p.supertrend_series.length-1].color)stColor=p.supertrend_series[p.supertrend_series.length-1].color;const newOpts={...initOpts,title:{...initOpts.title,text:`${p.pair_name} - Supertrend`},series:[{name:'Candlestick',type:'candlestick',data:p.ohlc||[]},{name:stLabel,type:'line',data:stData}],annotations:{yaxis:p.annotations_yaxis||[]},colors:[initOpts.colors[0],stColor]};if(!activeChart)activeChart=new ApexCharts(document.querySelector("#chart"),newOpts);else activeChart.updateOptions(newOpts)}catch(e){console.error(e)}finally{isLoading=false}}document.addEventListener('DOMContentLoaded',()=>{if(!activeChart)activeChart=new ApexCharts(document.querySelector("#chart"),initOpts);fetchPairs();if(intervalId)clearInterval(intervalId);intervalId=setInterval(async()=>{if(currentPairId&&document.visibilityState==='visible'&&!isLoading)await loadChart()},15000)});</script></body></html>"""
@flask_app_instance.route('/')
def serve_index_page(): return render_template_string(HTML_CHART_TEMPLATE)
@flask_app_instance.route('/api/available_pairs')
def get_available_pairs(): # Sama
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs_info = [{"id": pair_id, "name": pair_data.get("config", {}).get('pair_name', pair_id)} for pair_id, pair_data in data_manager_view.items() if pair_data.get("config", {}).get("enabled", True)]
    return jsonify(active_pairs_info)
@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request): # Sama
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager: return jsonify({"error": "Pair not found"}), 404
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))
    if not pair_data_snapshot: return jsonify({"error": "Data empty"}), 200
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, {pair_id_from_request: pair_data_snapshot})
    if not prepared_data: return jsonify({"error": "Failed to process"}), 500
    if not prepared_data.get("ohlc"): return jsonify({"error": "No OHLC", **prepared_data}), 200
    return jsonify(prepared_data)
def run_flask_server_thread(): # Sama
    log_info("Flask server mulai di http://localhost:5001", pair_name="SYSTEM_CHART")
    try: logging.getLogger('werkzeug').setLevel(logging.ERROR); flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e: log_error(f"Flask server gagal: {e}", pair_name="SYSTEM_CHART")

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)
    
    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key valid. Tidak dapat memulai.{AnsiColors.ENDC}"); input("Tekan Enter..."); return
    
    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}"); input("Tekan Enter..."); return

    animated_text_display("=========== SUPERTREND STRATEGY START (Multi-Pair) ===========", color=AnsiColors.HEADER)
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()}. Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {}
    # Inisialisasi semua pair
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        animated_text_display(f"\nMenginisialisasi {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA)
        
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0,
            "active_api_key_retries_this_pair": 0 # Untuk menghitung percobaan per key untuk pair ini
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_indicators_init = config.get('atr_length', 10) + 50 # Cukup untuk ATR dan buffer awal
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        initial_candles, initial_fetch_successful = [], False
        
        # Reset API key index ke primary untuk setiap pair baru saat inisialisasi besar
        # Jika tidak, satu pair gagal bisa membuat semua pair berikutnya mulai dari recovery key
        # Namun, jika primary key memang sudah habis global, ini akan tetap gagal dengan cepat.
        # api_key_manager.reset_key_index_to_primary() # Opsional, tergantung preferensi. Saat ini global.

        # Loop untuk mencoba semua API key untuk initial fetch pair ini
        # `api_key_manager.total_keys()` memberikan jumlah total upaya yang mungkin.
        max_initial_retries_for_pair = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        
        for attempt_num in range(max_initial_retries_for_pair):
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key (global) habis sebelum mencoba fetch untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break # Tidak ada key lagi untuk dicoba

            log_info(f"BIG DATA: Mencoba fetch awal {config['pair_name']} (Upaya {attempt_num+1}/{max_initial_retries_for_pair}, Key Index: {api_key_manager.get_current_key_index()})...", pair_name=config['pair_name'])
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key, config['timeframe'], pair_name=config['pair_name'])
                initial_fetch_successful = True
                local_crypto_data_manager[pair_id]["active_api_key_retries_this_pair"] = 0 # Reset jika berhasil
                break # Sukses, keluar dari loop retry untuk pair ini
            except APIKeyError as e_api:
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}: {e_api}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if api_key_manager.switch_to_next_key() is None: # Jika tidak ada key lagi setelah switch
                    log_error(f"BIG DATA: Semua API key telah dicoba dan gagal untuk inisialisasi {config['pair_name']}.", pair_name=config['pair_name'])
                    break # Keluar loop retry jika semua key habis
                # Lanjut ke iterasi berikutnya dengan key baru
            except requests.exceptions.RequestException as e_req:
                log_error(f"BIG DATA: Error jaringan saat fetch awal {config['pair_name']}: {e_req}. Tidak mengganti key, coba lagi nanti.", pair_name=config['pair_name'])
                break # Hentikan upaya untuk pair ini pada error jaringan, coba lagi di siklus utama
            except Exception as e_init_fetch:
                log_error(f"BIG DATA: Error umum saat fetch awal {config['pair_name']}: {e_init_fetch}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Initial Fetch Error:", pair_name=config['pair_name'])
                break # Hentikan untuk pair ini

        if not initial_fetch_successful:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya API key.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Tandai gagal
            local_crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Update waktu agar tidak langsung dicoba
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue # Lanjut ke pair berikutnya

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])
        
        # Warm-up state (sama)
        if initial_candles:
            min_len_for_warmup = config.get('atr_length', 10) + 2 
            if len(initial_candles) >= min_len_for_warmup:
                log_info(f"Warm-up state untuk {config['pair_name']}...", pair_name=config['pair_name'])
                for i in range(min_len_for_warmup -1, len(initial_candles) - 1):
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_warmup: continue
                    temp_state = local_crypto_data_manager[pair_id]["strategy_state"].copy(); temp_state["position_size"] = 0
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state, global_settings_dict)
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0; local_crypto_data_manager[pair_id]["strategy_state"]["entry_price_custom"] = None
                log_info(f"Warm-up {config['pair_name']} selesai.", pair_name=config['pair_name'])
        
        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}.", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(f"Data Complete: {config['pair_name']}", f"Data download complete for {TARGET_BIG_DATA_CANDLES} candles for {config['pair_name']}.", {**config, 'pair_name': config['pair_name']})
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"---------- MULAI LIVE ANALYSIS ({config['pair_name']}) ----------", pair_name=config['pair_name'])
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}--- SEMUA PAIR DIINISIALISASI ---{AnsiColors.ENDC}", color=AnsiColors.HEADER)
    
    # Main Trading Loop
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False
            
            # Penting: Reset index API key manager di awal setiap siklus besar agar setiap pair
            # di siklus berikutnya memulai lagi dari primary key jika sebelumnya gagal.
            # Namun, jika primary key memang sudah habis secara global, ini tidak akan membantu banyak.
            # api_key_manager.reset_key_index_to_primary() # Opsional, bisa menyebabkan primary key cepat habis jika sering error.

            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                # Cooldown jika semua key GAGAL KONSEKUTIF untuk pair ini
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        log_debug(f"Pair {pair_name_for_log} cooldown 1 jam (semua key gagal berturut-turut).", pair_name=pair_name_for_log)
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600); continue
                    else: data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset counter
                
                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    required_interval = 60 if config_for_pair.get('timeframe') == "minute" else 3600 
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch); continue
                
                log_info(f"Memproses {pair_name_for_log}...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                new_candles_batch, fetch_update_successful = [], False
                limit_fetch = 3 # Untuk update live
                if data_per_pair["big_data_collection_phase_active"]:
                    needed = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed <= 0: fetch_update_successful = True # Sudah cukup untuk display
                    else: limit_fetch = min(needed, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch > 0 or (data_per_pair["big_data_collection_phase_active"] and not fetch_update_successful) :
                    # Loop untuk mencoba semua API key untuk update fetch pair ini
                    max_update_retries_for_pair = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    
                    for attempt_update_num in range(max_update_retries_for_pair):
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE: Semua API key (global) habis sebelum mencoba fetch untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            break 

                        log_info(f"UPDATE: Mencoba fetch {pair_name_for_log} (Upaya {attempt_update_num+1}/{max_update_retries_for_pair}, Key Index: {api_key_manager.get_current_key_index()})...", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch, config_for_pair['exchange'], current_api_key_update, config_for_pair['timeframe'], pair_name_for_log)
                            fetch_update_successful = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 # Reset jika berhasil
                            any_data_fetched_this_cycle = True
                            break # Sukses, keluar loop retry
                        except APIKeyError as e_api_update:
                            log_warning(f"UPDATE: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {pair_name_for_log}: {e_api_update}. Mencoba key berikutnya.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] += 1
                            data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() # Catat waktu gagal terakhir
                            if api_key_manager.switch_to_next_key() is None:
                                log_error(f"UPDATE: Semua API key telah dicoba dan gagal untuk update {pair_name_for_log}.", pair_name=pair_name_for_log)
                                break
                        except requests.exceptions.RequestException as e_req_update:
                            log_error(f"UPDATE: Error jaringan saat fetch {pair_name_for_log}: {e_req_update}.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] += 1 # Hitung sebagai gagal juga
                            data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now()
                            break 
                        except Exception as e_gen_update:
                            log_error(f"UPDATE: Error umum saat fetch {pair_name_for_log}: {e_gen_update}.", pair_name=pair_name_for_log)
                            log_exception("Traceback Update Fetch Error:", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] += 1
                            data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now()
                            break
                
                # Merge candles (sama)
                if not fetch_update_successful or not new_candles_batch:
                    if not data_per_pair["big_data_collection_phase_active"]: # Hanya log jika sudah live
                         log_info(f"Tidak ada data candle baru atau fetch gagal untuk {pair_name_for_log}.",pair_name_for_log)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval); with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair); continue
                
                merged_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added, updated_count = 0,0
                for c_new in new_candles_batch:
                    ts_new = c_new['timestamp']
                    if ts_new not in merged_dict: merged_dict[ts_new] = c_new; newly_added+=1
                    elif merged_dict[ts_new] != c_new : merged_dict[ts_new] = c_new; updated_count+=1
                data_per_pair["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                if newly_added + updated_count > 0: log_info(f"{newly_added+updated_count} candle baru/update untuk {pair_name_for_log}.",pair_name_for_log)

                # Cek Big Data (display) & Jalankan Logika (sama)
                if data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                    data_per_pair["big_data_collection_phase_active"] = False; active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                    # Notif email
                    if not data_per_pair["big_data_email_sent"]:
                        send_email_notification(f"Data Display Ready: {pair_name_for_log}", f"{TARGET_BIG_DATA_CANDLES} candles ready for display for {pair_name_for_log}.", {**config_for_pair, 'pair_name': pair_name_for_log})
                        data_per_pair["big_data_email_sent"] = True
                    log_info(f"TARGET DISPLAY tercapai {pair_name_for_log}.", pair_name_for_log)
                elif not data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES:
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                
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
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None and api_key_manager.current_index >= api_key_manager.total_keys() -1 : # Semua key global habis
                log_error("Semua API key GAGAL GLOBAL dan tidak ada data fetch. Menunggu 1 jam.", pair_name="SYSTEM")
                sleep_duration = 3600
                api_key_manager.reset_key_index_to_primary() # Reset agar siklus berikutnya mulai dari primary
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 30 
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
            
            if sleep_duration > 0: show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: time.sleep(1)

    except KeyboardInterrupt: log_info("Proses dihentikan oleh pengguna.", pair_name="SYSTEM")
    except Exception as e: log_exception("Error loop utama:", pair_name="SYSTEM")
    finally: animated_text_display("STRATEGY STOP", color=AnsiColors.HEADER); input("Tekan Enter...");

# --- MENU UTAMA (Sama seperti sebelumnya, judul sudah Supertrend) ---
def main_menu(): # Sama
    settings = load_settings()
    flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True); flask_thread.start()
    while True:
        clear_screen_animated(); animated_text_display("========= Crypto Supertrend Strategy Runner =========", color=AnsiColors.HEADER)
        pick_title_main = ""; active_configs_list = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs_list: pick_title_main += f"--- Crypto Aktif ({len(active_configs_list)}) ---\n"; [pick_title_main := pick_title_main + f"  {i+1}. {cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}\n" for i, cfg in enumerate(active_configs_list)]
        else: pick_title_main += "Tidak ada konfigurasi crypto aktif.\n"
        pick_title_main += "-----------------------------------------------\n"
        api_s_main = settings.get("api_settings", {}); pkd_main = api_s_main.get('primary_key', 'BELUM DIATUR'); pkd_main = ("..." + pkd_main[-5:]) if len(pkd_main) > 10 and pkd_main not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"] else pkd_main
        nrk_main = len([k for k in api_s_main.get('recovery_keys',[]) if k]); tn_main = "Aktif" if api_s_main.get("enable_termux_notifications", False) else "Nonaktif"
        pick_title_main += f"Target Data Display: {TARGET_BIG_DATA_CANDLES} candle\nPrimary API Key: {pkd_main} | Recovery: {nrk_main}\nNotif Termux: {tn_main}\nChart: http://localhost:5001\n-----------------------------------------------\nPilih Opsi:"
        main_menu_options_plain = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]; selected_main_index = -1
        try: _, selected_main_index = pick(main_menu_options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception: print(pick_title_main); [print(f"  {i+1}. {opt}") for i, opt in enumerate(main_menu_options_plain)]; choice = input("Pilih: "); selected_main_index = int(choice)-1 if choice.isdigit() and 0<=int(choice)-1<len(main_menu_options_plain) else -1
        if selected_main_index == 0: settings = load_settings(); start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif selected_main_index == 1: settings = settings_menu(settings)
        elif selected_main_index == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Sampai jumpa!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan.", color=AnsiColors.ORANGE)
    except Exception as e: clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL: {e}{AnsiColors.ENDC}"); log_exception("MAIN LEVEL CRITICAL ERROR:", pair_name="SYSTEM_CRITICAL"); input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
