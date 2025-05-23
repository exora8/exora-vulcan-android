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


SETTINGS_FILE = "settings_supertrend.json"
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
            self.keys.extend([k for k in recovery_keys_list if k]) # Filter out empty strings

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}

        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys:
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None # All keys exhausted or no keys initially

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
            return None # Explicitly return None when all keys are exhausted

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
        "atr_length": 10, "factor": 3.0,
        "emergency_sl_percent": 2.0, "profit_target_percent_activation": 2.0, "trailing_stop_gap_percent": 1.0,
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
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid.{AnsiColors.ENDC}");
    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60))
    except ValueError: new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))

    animated_text_display("\n-- Parameter Supertrend --", color=AnsiColors.HEADER)
    try:
        new_config["atr_length"] = int(input(f"{AnsiColors.BLUE}Supertrend - ATR Length [{new_config.get('atr_length',10)}]: {AnsiColors.ENDC}").strip() or new_config.get('atr_length',10))
        new_config["factor"] = float(input(f"{AnsiColors.BLUE}Supertrend - Factor [{new_config.get('factor',3.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('factor',3.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input Supertrend tidak valid. Default digunakan.{AnsiColors.ENDC}")
        def_cfg = get_default_crypto_config()
        new_config["atr_length"] = new_config.get('atr_length', def_cfg['atr_length'])
        new_config["factor"] = new_config.get('factor', def_cfg['factor'])

    animated_text_display("\n-- Parameter Exits --", color=AnsiColors.HEADER)
    try:
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Fixed Stop Loss % [{new_config.get('emergency_sl_percent',2.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',2.0))
        animated_text_display("\n-- Trailing Take Profit (Persentase) --", color=AnsiColors.CYAN)
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Trail Activation Profit % [{new_config.get('profit_target_percent_activation',2.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',2.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Trail Gap % [{new_config.get('trailing_stop_gap_percent',1.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',1.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input Exits tidak valid. Default digunakan.{AnsiColors.ENDC}")
        def_cfg = get_default_crypto_config()
        for k_exit in ["emergency_sl_percent", "profit_target_percent_activation", "trailing_stop_gap_percent"]:
            new_config[k_exit] = new_config.get(k_exit, def_cfg[k_exit])
    
    animated_text_display("\n-- Notifikasi Email (Gmail) --", color=AnsiColors.HEADER)
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()
    return new_config

def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        pkd = api_s.get('primary_key', 'N/A'); pkd = pkd[:5]+"..."+pkd[-3:] if len(pkd)>8 and pkd not in ["YOUR_PRIMARY_KEY", "N/A"] else pkd
        nrk = len([k for k in api_s.get('recovery_keys', []) if k])
        tns = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        title = f"--- Menu Pengaturan ---\nAPI Key: {pkd} | Recovery: {nrk} | Termux: {tns}\nStrategi: Supertrend Entry\nCrypto Pairs:\n"
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]): title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({cfg.get('timeframe','?')}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'}\n"
        title += "----------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux", "Tambah Crypto Pair", "Ubah Crypto Pair", "Hapus Crypto Pair", "Kembali"]
        _, action_idx = pick(opts, title, indicator='=>')
        clear_screen_animated()
        try:
            if action_idx == 0:
                api_s["primary_key"] = (input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
            elif action_idx == 1: # Kelola Recovery (loop internal)
                while True:
                    clear_screen_animated()
                    animated_text_display("--- Kelola Recovery API Keys ---", color=AnsiColors.HEADER)
                    if not api_s.get('recovery_keys'):
                        api_s['recovery_keys'] = []
                    
                    current_rec_keys_display = "\nKunci Recovery Saat Ini:\n"
                    if api_s['recovery_keys']:
                        for i_rec, rk in enumerate(api_s['recovery_keys']):
                            rk_disp = rk[:5]+"..."+rk[-3:] if len(rk)>8 else rk
                            current_rec_keys_display += f"  {i_rec+1}. {rk_disp}\n"
                    else:
                        current_rec_keys_display += "  (Kosong)\n"
                    current_rec_keys_display += "-----------------------------\n"
                    
                    rec_opts = ["Tambah Recovery Key Baru", "Hapus Recovery Key", "Kembali ke Menu Pengaturan"]
                    if not api_s['recovery_keys']:
                        rec_opts.pop(1) # Hapus opsi "Hapus" jika tidak ada key
                        
                    _, rec_action_idx = pick(rec_opts, current_rec_keys_display + "Pilih tindakan:", indicator='=>')
                    
                    if rec_opts[rec_action_idx] == "Tambah Recovery Key Baru":
                        new_rec_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_rec_key:
                            api_s['recovery_keys'].append(new_rec_key)
                            log_info("Recovery key ditambahkan.", pair_name="SETTINGS")
                        else:
                            log_warning("Tidak ada input, key tidak ditambahkan.", pair_name="SETTINGS")
                        show_spinner(1, "Memproses...")
                    elif rec_opts[rec_action_idx] == "Hapus Recovery Key":
                        if api_s['recovery_keys']:
                            del_rec_key_opts = [f"{i_del+1}. {rk[:5]}...{rk[-3:] if len(rk)>8 else rk}" for i_del, rk in enumerate(api_s['recovery_keys'])] + ["Batal"]
                            _, del_rec_key_choice_idx = pick(del_rec_key_opts, "Pilih key untuk dihapus:", indicator='=>')
                            if del_rec_key_choice_idx < len(api_s['recovery_keys']):
                                removed_key = api_s['recovery_keys'].pop(del_rec_key_choice_idx)
                                log_info(f"Recovery key '{removed_key[:5]}...' dihapus.", pair_name="SETTINGS")
                            show_spinner(1, "Memproses...")
                    elif rec_opts[rec_action_idx] == "Kembali ke Menu Pengaturan":
                        break # Keluar dari sub-menu recovery
                    
                    # Simpan pengaturan setelah setiap perubahan di sub-menu recovery
                    current_settings["api_settings"] = api_s
                    save_settings(current_settings)

            elif action_idx == 2:
                api_s['enable_global_email_notifications_for_key_switch'] = input(f"Aktifkan Notif Sistem via Email? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip() == 'true'
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Admin (Global) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))

            elif action_idx == 3:
                api_s['enable_termux_notifications'] = input(f"Aktifkan Notifikasi Termux (Global)? (true/false) [{api_s.get('enable_termux_notifications',False)}]: ").lower().strip() == 'true'

            elif action_idx == 4: current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_idx == 5:
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada pair untuk diubah.{AnsiColors.ENDC}"); show_spinner(1.5,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')} ({'Aktif' if c.get('enabled',True) else 'Nonaktif'})" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:", indicator='=>')
                if edit_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 6:
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada pair untuk dihapus.{AnsiColors.ENDC}"); show_spinner(1.5,""); continue
                del_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, del_c_idx = pick(del_opts, "Pilih pair untuk dihapus:", indicator='=>')
                if del_c_idx < len(current_settings["cryptos"]): 
                    removed = current_settings["cryptos"].pop(del_c_idx)
                    log_info(f"Pair {removed.get('symbol','?')}-{removed.get('currency','?')} dihapus.", pair_name="SETTINGS")
            elif action_idx == 7: break # Kembali
            current_settings["api_settings"] = api_s # Pastikan api_s disimpan kembali
            save_settings(current_settings)
            if action_idx not in [1,7]: show_spinner(1, "Disimpan...") # Sub-menu recovery akan handle savenya sendiri, kembali tidak save
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1.5, "Terjadi error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = {"minute": "histominute", "hour": "histohour", "day": "histoday"}.get(timeframe, "histohour")
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 20 # Show progress for larger fetches

    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
    
    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts: params["toTs"] = current_to_ts
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429]: # Key-related HTTP errors
                err_msg_detail = ""
                try: err_msg_detail = response.json().get('Message', f"HTTP Error {response.status_code}")
                except: err_msg_detail = f"HTTP Error {response.status_code}, no JSON body."
                log_warning(f"API Key Error (HTTP {response.status_code}): {err_msg_detail} | Key: ...{current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else 'INVALID_KEY'}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {err_msg_detail}")
            response.raise_for_status() # Other HTTP errors
            data = response.json()
            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error')
                key_err_patterns = ["api key is invalid", "apikey_is_missing", "rate limit", "monthly_calls", "tier", "you are not authorized"]
                if any(p.lower() in err_msg.lower() for p in key_err_patterns):
                    log_warning(f"API Key Error (JSON): {err_msg} | Key: ...{current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else 'INVALID_KEY'}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {err_msg}")
                else: log_error(f"API Error: {err_msg}", pair_name=pair_name); break 
            
            raw_candles = data.get('Data', {}).get('Data', [])
            if not raw_candles:
                if len(all_accumulated_candles) > 0 : log_debug(f"Tidak ada candle baru dari API (mungkin akhir histori). Total: {len(all_accumulated_candles)}", pair_name=pair_name)
                else: log_warning(f"Tidak ada data candle sama sekali dari API untuk {pair_name}.", pair_name=pair_name)
                break

            batch = []
            for item in raw_candles:
                if all(k in item and item[k] is not None for k in ['time', 'open', 'high', 'low', 'close', 'volumefrom']):
                    batch.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            
            if current_to_ts and all_accumulated_candles and batch and batch[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']: batch.pop() # Remove overlap
            if not batch and current_to_ts: break # No new data after overlap removal

            all_accumulated_candles = batch + all_accumulated_candles
            if raw_candles: current_to_ts = raw_candles[0]['time']
            else: break

            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
            if len(raw_candles) < limit_call: break # Reached end of available history for this timeframe
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.2) # Small delay for large fetches
        except APIKeyError: raise # Re-raise to be caught by calling function for key switching
        except requests.exceptions.RequestException as e_req: log_error(f"Kesalahan Jaringan: {e_req}", pair_name=pair_name); break # Don't switch key for network error
        except Exception as e_gen: log_exception(f"Error lain dalam fetch_candles: {e_gen}", pair_name=pair_name); break
    
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai')
    return all_accumulated_candles

# --- LOGIKA STRATEGI (Supertrend) ---
def get_initial_strategy_state():
    return {
        "last_supertrend_value": None, "supertrend_direction": 0, "last_atr_value_for_chart": None,
        "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
        "last_ma_value_for_chart": None, "last_stoch_k_value_for_chart": None, "last_stoch_d_value_for_chart": None,
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None, "last_pivot_low_display_info": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None, "active_fib_line_start_index": None,
        "last_notification_type": None, "last_notification_price": None, # Untuk mencegah notif berulang
    }

def calculate_atr(high_prices, low_prices, close_prices, period):
    if len(close_prices) < period or period <= 0: return [None] * len(close_prices)
    tr, atr = [None] * len(close_prices), [None] * len(close_prices)
    for i in range(len(close_prices)):
        if not all(x is not None for x in [high_prices[i], low_prices[i], close_prices[i]]): continue
        hl = high_prices[i] - low_prices[i]
        hpc = abs(high_prices[i] - (close_prices[i-1] if i>0 and close_prices[i-1] is not None else high_prices[i]))
        lpc = abs(low_prices[i] - (close_prices[i-1] if i>0 and close_prices[i-1] is not None else low_prices[i]))
        tr[i] = max(hl, hpc, lpc) if i>0 or (i==0 and close_prices[i-1] is None) else (hl if i==0 else None)

    first_valid_tr_idx = next((i for i, x in enumerate(tr) if x is not None), -1)
    if first_valid_tr_idx == -1 or len([x for x in tr[first_valid_tr_idx:] if x is not None]) < period : return atr
    
    atr_start_idx = -1; sum_tr = 0; count = 0
    for i in range(first_valid_tr_idx, len(tr)):
        if tr[i] is not None: sum_tr += tr[i]; count += 1
        if count == period: atr_start_idx = i; atr[i] = sum_tr / period; break
    if atr_start_idx == -1: return atr
        
    for i in range(atr_start_idx + 1, len(close_prices)):
        if tr[i] is not None and atr[i-1] is not None: atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_supertrend(high_prices, low_prices, close_prices, atr_length, factor):
    n = len(close_prices)
    if n < atr_length + 1: return ([None]*n, [0]*n)
    atr_values = calculate_atr(high_prices, low_prices, close_prices, atr_length) # Renamed local var
    st, direction = [None]*n, [0]*n
    
    first_valid_idx = -1
    for i in range(atr_length, n): # Start where ATR might be valid
        if all(x is not None for x in [close_prices[i], high_prices[i], low_prices[i], atr_values[i]]):
            first_valid_idx = i; break
    if first_valid_idx == -1: return (st, direction)

    mid_init = (high_prices[first_valid_idx] + low_prices[first_valid_idx]) / 2
    st[first_valid_idx] = mid_init - factor * atr_values[first_valid_idx] if close_prices[first_valid_idx] > mid_init else mid_init + factor * atr_values[first_valid_idx]
    direction[first_valid_idx] = 1 if close_prices[first_valid_idx] > st[first_valid_idx] else -1
    if close_prices[first_valid_idx] == st[first_valid_idx]: direction[first_valid_idx] = 1 

    for i in range(first_valid_idx + 1, n):
        if not all(x is not None for x in [close_prices[i], high_prices[i], low_prices[i], atr_values[i], close_prices[i-1], st[i-1]]):
            st[i] = st[i-1]; direction[i] = direction[i-1]; continue

        basic_ub = ((high_prices[i] + low_prices[i]) / 2) + factor * atr_values[i]
        basic_lb = ((high_prices[i] + low_prices[i]) / 2) - factor * atr_values[i]
        
        prev_st, prev_dir = st[i-1], direction[i-1]
        curr_st, curr_dir = prev_st, prev_dir

        if prev_dir == 1: # Uptrend
            curr_st = max(basic_lb, prev_st)
            if close_prices[i] < curr_st: curr_dir = -1; curr_st = basic_ub
        elif prev_dir == -1: # Downtrend
            curr_st = min(basic_ub, prev_st)
            if close_prices[i] > curr_st: curr_dir = 1; curr_st = basic_lb
        else: 
            mid = (high_prices[i] + low_prices[i]) / 2
            curr_st = basic_lb if close_prices[i] > mid else basic_ub
            curr_dir = 1 if close_prices[i] > curr_st else -1
            if close_prices[i] == curr_st : curr_dir = 1

        st[i], direction[i] = curr_st, curr_dir
    return st, direction

def _send_and_log_trade_notification(event_type, price, pnl, pair_name, crypto_config, global_settings):
    subject = f"Supertrend {event_type.upper()} Signal: {pair_name} @ {price:.5f}"
    body = f"Supertrend {event_type.upper()} Signal for {pair_name} at price {price:.5f}.\n"
    if "PnL" in event_type: body += f"Estimated PnL: {pnl:.2f}%\n"
    body += f"Timeframe: {crypto_config.get('timeframe')}\n"
    body += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    log_info(f"{AnsiColors.GREEN if 'BUY' in event_type.upper() else AnsiColors.RED if 'EXIT' in event_type.upper() or 'SELL' in event_type.upper() else AnsiColors.BLUE}{event_type.upper()} @ {price:.5f}. {('PnL: ' + str(pnl) + '%' if pnl is not None else '')}{AnsiColors.ENDC}", pair_name=pair_name)
    play_notification_sound()
    send_email_notification(subject, body, {**crypto_config, 'pair_name': pair_name}) # merge crypto_config for email settings
    send_termux_notification(f"{pair_name} {event_type.upper()}", f"Price: {price:.5f}{(', PnL: ' + str(pnl) + '%' if pnl is not None else '')}", global_settings, pair_name_for_log=pair_name)


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    atr_len, st_factor = crypto_config.get('atr_length', 10), crypto_config.get('factor', 3.0)
    sl_perc = crypto_config.get('emergency_sl_percent',2.0)/100.0
    tp_act_perc = crypto_config.get('profit_target_percent_activation',2.0)/100.0
    tp_trail_perc = crypto_config.get('trailing_stop_gap_percent',1.0)/100.0
    
    min_data_needed_for_st = atr_len + 2
    if len(candles_history) < min_data_needed_for_st: return strategy_state

    closes = [c.get('close') for c in candles_history]
    highs = [c.get('high') for c in candles_history]
    lows = [c.get('low') for c in candles_history]
    
    st_line, st_dir_signal = calculate_supertrend(highs, lows, closes, atr_len, st_factor)
    curr_idx = len(candles_history)-1
    prev_idx = curr_idx - 1

    curr_candle = candles_history[curr_idx]
    if any(curr_candle.get(k) is None for k in ['open','high','low','close','timestamp']): return strategy_state # Invalid current candle
    
    curr_c, curr_l, curr_h = curr_candle['close'], curr_candle['low'], curr_candle['high']
    curr_st_val, curr_st_dir = st_line[curr_idx], st_dir_signal[curr_idx]
    
    strategy_state["last_supertrend_value"], strategy_state["supertrend_direction"] = curr_st_val, curr_st_dir
    # strategy_state["last_atr_value_for_chart"] = calculate_atr(highs,lows,closes,atr_len)[-1] # For display if needed

    # Ensure prev values are valid before using for cross logic
    prev_c, prev_st_val, prev_st_dir = (None, None, None)
    if prev_idx >= 0:
        prev_candle = candles_history[prev_idx]
        if all(prev_candle.get(k) is not None for k in ['close']):
            prev_c = prev_candle['close']
        if st_line[prev_idx] is not None:
            prev_st_val = st_line[prev_idx]
        prev_st_dir = st_dir_signal[prev_idx]

    # Check for valid values before processing signals
    if curr_st_val is None or prev_st_val is None or prev_c is None:
        # log_debug(f"Skipping signal check for {pair_name} due to None values (curr_st: {curr_st_val}, prev_st: {prev_st_val}, prev_c: {prev_c})", pair_name=pair_name)
        return strategy_state 

    # --- TRADE LOGIC ---
    # Only process if the last notification was different or price is significantly different
    def should_notify(event_type, price):
        if strategy_state.get("last_notification_type") == event_type and \
           strategy_state.get("last_notification_price") is not None and \
           abs(price - strategy_state.get("last_notification_price")) / price < 0.001: # 0.1% change
            return False
        strategy_state["last_notification_type"] = event_type
        strategy_state["last_notification_price"] = price
        return True

    # Exit Long Position (check first)
    if strategy_state["position_size"] > 0:
        entry_p_val = strategy_state["entry_price_custom"]
        
        # 1. Trailing Stop Loss / Take Profit
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"]:
            if curr_l <= strategy_state["current_trailing_stop_level"]:
                exit_price_tsl = strategy_state["current_trailing_stop_level"]
                pnl_tsl = ((exit_price_tsl - entry_p_val) / entry_p_val) * 100.0 if entry_p_val else 0.0
                event_label = "Trailing TP Hit" if pnl_tsl >=0 else "Trailing SL Hit (Loss)"
                if should_notify(event_label, exit_price_tsl):
                    _send_and_log_trade_notification(event_label, exit_price_tsl, pnl_tsl, pair_name, crypto_config, global_settings)
                strategy_state = get_initial_strategy_state() # Reset state
                strategy_state["last_supertrend_value"], strategy_state["supertrend_direction"] = curr_st_val, curr_st_dir
                return strategy_state

        # 2. Fixed Emergency Stop Loss
        if strategy_state["emergency_sl_level_custom"] and curr_l <= strategy_state["emergency_sl_level_custom"]:
            exit_price_sl = strategy_state["emergency_sl_level_custom"]
            pnl_sl = ((exit_price_sl - entry_p_val) / entry_p_val) * 100.0 if entry_p_val else 0.0
            if should_notify("Emergency SL Hit", exit_price_sl):
                _send_and_log_trade_notification("Emergency SL Hit", exit_price_sl, pnl_sl, pair_name, crypto_config, global_settings)
            strategy_state = get_initial_strategy_state()
            strategy_state["last_supertrend_value"], strategy_state["supertrend_direction"] = curr_st_val, curr_st_dir
            return strategy_state
        
        # 3. Supertrend Cross Down for Exit
        crossed_below_st = prev_st_dir == 1 and curr_st_dir == -1 # Cross from up to down
        # Or more robust: prev_c >= prev_st_val and curr_c < curr_st_val (already in uptrend implies prev_c > prev_st_val usually)
        # More precise cross check: (prev_c >= prev_st_val and curr_c < curr_st_val)
        if crossed_below_st:
            exit_price_st_cross = curr_c
            pnl_st_cross = ((exit_price_st_cross - entry_p_val) / entry_p_val) * 100.0 if entry_p_val else 0.0
            if should_notify("Exit Long (ST Cross Down)", exit_price_st_cross):
                 _send_and_log_trade_notification("Exit Long (ST Cross Down)", exit_price_st_cross, pnl_st_cross, pair_name, crypto_config, global_settings)
            strategy_state = get_initial_strategy_state()
            strategy_state["last_supertrend_value"], strategy_state["supertrend_direction"] = curr_st_val, curr_st_dir
            return strategy_state

    # Entry Long Position
    elif strategy_state["position_size"] == 0:
        crossed_above_st = prev_st_dir == -1 and curr_st_dir == 1 # Cross from down to up
        # Or more robust: (prev_c <= prev_st_val and curr_c > curr_st_val)
        if crossed_above_st:
            entry_price_new = curr_c
            if should_notify("Enter Long (ST Cross Up)", entry_price_new):
                _send_and_log_trade_notification("Enter Long (ST Cross Up)", entry_price_new, None, pair_name, crypto_config, global_settings)
            strategy_state["position_size"] = 1
            strategy_state["entry_price_custom"] = entry_price_new
            strategy_state["emergency_sl_level_custom"] = entry_price_new * (1 - sl_perc)
            strategy_state["highest_price_for_trailing"] = curr_h # Initialize with current high
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            # No return here, let trailing logic update in the same candle if applicable

    # Update Trailing SL if in position
    if strategy_state["position_size"] > 0:
        entry_p_val = strategy_state["entry_price_custom"]
        if curr_h is not None:
            strategy_state["highest_price_for_trailing"] = max(strategy_state.get("highest_price_for_trailing", curr_h) or curr_h, curr_h)
        
        if not strategy_state["trailing_tp_active_custom"] and entry_p_val and strategy_state["highest_price_for_trailing"]:
            if ((strategy_state["highest_price_for_trailing"] - entry_p_val) / entry_p_val) >= tp_act_perc:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Activated for {pair_name}. Highest price: {strategy_state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
        
        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"]:
            new_potential_tsl = strategy_state["highest_price_for_trailing"] * (1 - tp_trail_perc)
            # Only update if new TSL is higher than current, or if current TSL is None
            if strategy_state["current_trailing_stop_level"] is None or new_potential_tsl > strategy_state["current_trailing_stop_level"]:
                # Also ensure TSL doesn't go below entry
                # strategy_state["current_trailing_stop_level"] = max(new_potential_tsl, entry_p_val) # Option: TSL cannot be below entry
                strategy_state["current_trailing_stop_level"] = new_potential_tsl
                # log_debug(f"Trailing SL for {pair_name} updated to: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)
    
    # Alert for potential short (if not in long position)
    if strategy_state["position_size"] == 0:
        crossed_below_st_alert = prev_st_dir == 1 and curr_st_dir == -1
        if crossed_below_st_alert:
            if should_notify("Potential Short (ST Cross Down)", curr_c):
                _send_and_log_trade_notification("Potential Short (ST Cross Down)", curr_c, None, pair_name, crypto_config, global_settings)

    return strategy_state


# CHART_INTEGRATION_START & Flask Endpoints
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id, snapshot_of_manager): 
    if pair_id not in snapshot_of_manager: return None
    data = snapshot_of_manager[pair_id]
    
    # Use a consistent number of candles for display, e.g., last 200-300
    # TARGET_BIG_DATA_CANDLES might be too large (2500) for charting performance.
    # Let's define a new constant or use a fixed number for charts.
    CHART_DISPLAY_CANDLES = 300 
    hist_full = data.get("all_candles_list", [])
    hist_for_chart = hist_full[-CHART_DISPLAY_CANDLES:] if len(hist_full) > CHART_DISPLAY_CANDLES else hist_full

    cfg = data.get("config", {})
    state = data.get("strategy_state", {})
    ohlc, st_series = [], []
    
    if not hist_for_chart: 
        return {"ohlc":[], "supertrend_series":[], "pair_name": cfg.get('pair_name',pair_id), "last_updated_tv":None, "strategy_state_info":{}}

    # Recalculate ST for the displayed segment of candles for accuracy on chart
    chart_closes = [c.get('close') for c in hist_for_chart]
    chart_highs = [c.get('high') for c in hist_for_chart]
    chart_lows = [c.get('low') for c in hist_for_chart]
    
    chart_st_values, chart_st_directions = calculate_supertrend(
        chart_highs, chart_lows, chart_closes, 
        cfg.get('atr_length',10), cfg.get('factor',3.0)
    )

    for i, c in enumerate(hist_for_chart):
        if all(c.get(k) is not None for k in ['timestamp','open','high','low','close']):
            tsms = c['timestamp'].timestamp()*1000
            ohlc.append({'x':tsms, 'y':[c['open'],c['high'],c['low'],c['close']]})
            if i < len(chart_st_values) and chart_st_values[i] is not None:
                # Color ST line based on direction from recalculation
                st_color = '#00E396' if chart_st_directions[i] == 1 else ('#FF4560' if chart_st_directions[i] == -1 else '#888888')
                st_series.append({'x':tsms, 'y':chart_st_values[i], 'color': st_color}) # Pass color for ApexCharts if needed
        
    annotations_yaxis = []
    if state.get("position_size",0) > 0 and state.get("entry_price_custom") is not None:
        ep = state.get("entry_price_custom")
        annotations_yaxis.append({
            'y': ep, 'borderColor': '#2698FF', 'label': {
                'borderColor': '#2698FF', 'style': {'color': '#fff', 'background': '#2698FF'},
                'text': f'Entry: {ep:.5f}'
            }
        })
        
        # Emergency SL
        sl_val_emergency = state.get("emergency_sl_level_custom")
        if sl_val_emergency:
             annotations_yaxis.append({
                'y': sl_val_emergency, 'borderColor': '#FF4560', 'label': {
                    'borderColor': '#FF4560', 'style': {'color': '#fff', 'background': '#FF4560'},
                    'text': f'Fix SL: {sl_val_emergency:.5f}'
                }
            })

        # Trailing SL (if active and different from emergency SL or emergency SL is not set)
        tsl_val = state.get("current_trailing_stop_level")
        if state.get("trailing_tp_active_custom") and tsl_val:
            # Only show if it's the effective SL
            is_tsl_effective = True
            if sl_val_emergency and tsl_val <= sl_val_emergency: # If emergency SL is tighter, don't show TSL
                is_tsl_effective = False
            
            if is_tsl_effective:
                annotations_yaxis.append({
                    'y': tsl_val, 'borderColor': '#FEB019', 'label': { # Orange for TSL
                        'borderColor': '#FEB019', 'style': {'color': '#fff', 'background': '#FEB019'},
                        'text': f'Trail SL: {tsl_val:.5f}'
                    }
                })
    
    last_ts = hist_for_chart[-1]['timestamp'].timestamp()*1000 if hist_for_chart else None
    
    # Get the latest ST value and direction from the *full dataset* state, not recalculated chart segment
    live_st_val = state.get("last_supertrend_value")
    live_st_dir = state.get("supertrend_direction")

    return {
        "ohlc": ohlc, 
        "supertrend_series": st_series, 
        "annotations_yaxis": annotations_yaxis,
        "pair_name": cfg.get('pair_name', pair_id),
        "last_updated_tv": last_ts,
        "st_atr_len_label": cfg.get('atr_length'), 
        "st_factor_label": cfg.get('factor'),
        "strategy_state_info": {
            "supertrend_value": live_st_val, # Use live state
            "supertrend_direction": live_st_dir # Use live state
        }
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """ <!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>Supertrend Chart</title> <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script> <style>body{font-family:sans-serif;margin:0;background-color:#1e1e1e;color:#e0e0e0;display:flex;flex-direction:column;align-items:center;padding:10px}#controls{background-color:#2a2a2a;padding:10px;border-radius:8px;margin-bottom:10px;display:flex;align-items:center;gap:10px;width:100%;max-width:1200px}select,button{padding:8px 12px;border-radius:5px;border:1px solid #444;background-color:#333;color:#e0e0e0;cursor:pointer}#chart-container{width:100%;max-width:1200px;background-color:#2a2a2a;padding:15px;border-radius:8px}h1{color:#00bcd4;margin-bottom:10px;font-size:1.5em}#lastUpdatedLabel{font-size:.8em;color:#aaa;margin-left:auto}#strategyInfoLabel{font-size:.8em;color:#ffd700;margin-left:10px;white-space:pre}</style> </head> <body> <h1>Supertrend Strategy Chart</h1> <div id="controls"> <label for="pairSelector">Pair:</label> <select id="pairSelector" onchange="handlePairSelectionChange()"></select> <button onclick="loadChartDataForCurrentPair()">Refresh</button> <span id="strategyInfoLabel">Status: -</span> <span id="lastUpdatedLabel">Memuat...</span> </div> <div id="chart-container"><div id="chart"></div></div> <script>let activeChart,currentSelectedPairId="",lastKnownDataTimestamp=null,autoRefreshIntervalId=null,isLoadingData=!1;const initialChartOptions={series:[{name:"Candlestick",type:"candlestick",data:[]},{name:"Supertrend",type:"line",data:[]}],chart:{type:"candlestick",height:550,background:"#2a2a2a",animations:{enabled:!1},toolbar:{show:!0}},theme:{mode:"dark"},title:{text:"Memuat Data Pair...",align:"left",style:{color:"#e0e0e0"}},xaxis:{type:"datetime",labels:{style:{colors:"#aaa"}},tooltip:{enabled:!1}},yaxis:{tooltip:{enabled:!0},labels:{style:{colors:"#aaa"},formatter:v=>v?v.toFixed(5):""}},stroke:{width:[1,2],curve:"straight"},markers:{size:0},colors:["#FEB019","#888888"],grid:{borderColor:"#444"},annotations:{yaxis:[],points:[]},tooltip:{theme:"dark",shared:!0,intersect:!1,custom:function({series:s,seriesIndex:e,dataPointIndex:a,w:t}){let n,i,r,l,o,d;const c=t.globals.series.findIndex(s=>"candlestick"===s.type),h=t.globals.series.findIndex(s=>s.name.startsWith("Supertrend"));c!==-1&&void 0!==t.globals.seriesCandleO[c]?.[a]&&([n,i,r,l]=[t.globals.seriesCandleO[c][a],t.globals.seriesCandleH[c][a],t.globals.seriesCandleL[c][a],t.globals.seriesCandleC[c][a]]),h!==-1&&s[h]?.[a]!==void 0&&t.config.series[h].data[a]&&(o=t.config.series[h].data[a].y);let u='<div style="padding:5px 10px;background:#333;color:#fff;border:1px solid #555;">';return void 0!==n&&(u+=["O","H","L","C"].map((s,e)=>`<div>${s}: <span style="font-weight:bold;">${[n,i,r,l][e].toFixed(5)}</span></div>`).join("")),void 0!==o&&(u+=`<div>ST: <span style="font-weight:bold;">${o.toFixed(5)}</span></div>`),u+="</div>",void 0!==n||void 0!==o?u:""},noData:{text:"Tidak ada data.",align:"center",style:{color:"#ccc"}}};async function fetchAvailablePairs(){try{const s=await fetch("/api/available_pairs");if(!s.ok)throw new Error(`HTTP ${s.status}`);const e=await s.json(),a=document.getElementById("pairSelector");if(a.innerHTML="","e.length>0?(e.forEach(s=>{const e=document.createElement("option");e.value=s.id,e.textContent=s.name,a.appendChild(e)}),currentSelectedPairId=a.value||e[0].id,loadChartDataForCurrentPair()):(a.innerHTML='<option value="">No pairs</option>',activeChart&&(activeChart.destroy(),activeChart=null),document.getElementById("chart").innerHTML="No pairs configured.")}catch(s){console.error("Err pairs:",s),activeChart&&(activeChart.destroy(),activeChart=null),document.getElementById("chart").innerHTML=`Err: ${s.message}`}}function handlePairSelectionChange(){currentSelectedPairId=document.getElementById("pairSelector").value,lastKnownDataTimestamp=null,loadChartDataForCurrentPair()}async function loadChartDataForCurrentPair(){if(currentSelectedPairId&&!isLoadingData){isLoadingData=!0,document.getElementById("lastUpdatedLabel").textContent=`Loading ${currentSelectedPairId}...`;try{const s=await fetch(`/api/chart_data/${currentSelectedPairId}`);if(!s.ok)throw new Error(`HTTP ${s.status}`);const e=await s.json();if(e&&e.ohlc&&0!==e.ohlc.length){if(e.last_updated_tv&&e.last_updated_tv===lastKnownDataTimestamp){console.log("Chart data unchanged."),document.getElementById("lastUpdatedLabel").textContent=`Last @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;const s=e.strategy_state_info||{};document.getElementById("strategyInfoLabel").textContent=`ST: ${s.supertrend_value?.toFixed(3)||"N/A"}\nDir: ${s.supertrend_direction||"N/A"}`,isLoadingData=!1;return}lastKnownDataTimestamp=e.last_updated_tv,document.getElementById("lastUpdatedLabel").textContent=`Last @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;const a=e.strategy_state_info||{};document.getElementById("strategyInfoLabel").textContent=`ST: ${a.supertrend_value?.toFixed(3)||"N/A"}\nDir: ${1==a.supertrend_direction?"Up":-1==a.supertrend_direction?"Down":"N/A"}`;const t=`Supertrend (${e.st_atr_len_label}/${e.st_factor_label})`,n=e.supertrend_series.map(s=>({x:s.x,y:s.y}));let i=initialChartOptions.colors[1];if(e.supertrend_series&&e.supertrend_series.length>0){const s=e.supertrend_series[e.supertrend_series.length-1];s&&s.color&&(i=s.color)}const r={...initialChartOptions,title:{...initialChartOptions.title,text:`${e.pair_name} - Supertrend`},series:[{name:"Candlestick",type:"candlestick",data:e.ohlc||[]},{name:t,type:"line",data:n}],annotations:{yaxis:e.annotations_yaxis||[],points:e.annotations_points||[]},colors:[initialChartOptions.colors[0],i]};activeChart?activeChart.updateOptions(r):(activeChart=new ApexCharts(document.querySelector("#chart"),r),activeChart.render())}else{const s={...initialChartOptions,title:{...initialChartOptions.title,text:`${e.pair_name||currentSelectedPairId} - No Data`},series:initialChartOptions.series.map(s=>({...s,data:[]}))};activeChart?activeChart.updateOptions(s):(activeChart=new ApexCharts(document.querySelector("#chart"),s),activeChart.render()),lastKnownDataTimestamp=e.last_updated_tv||null,document.getElementById("lastUpdatedLabel").textContent=lastKnownDataTimestamp?`Data (empty) @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`:"No data",document.getElementById("strategyInfoLabel").textContent="Status: Data Kosong",isLoadingData=!1}}catch(s){console.error("Err chart data:",s),activeChart&&(activeChart.destroy(),activeChart=null),document.getElementById("chart").innerHTML=`Err: ${s.message}`}finally{isLoadingData=!1}}}document.addEventListener("DOMContentLoaded",()=>{activeChart||(activeChart=new ApexCharts(document.querySelector("#chart"),initialChartOptions),activeChart.render()),fetchAvailablePairs(),autoRefreshIntervalId&&clearInterval(autoRefreshIntervalId),autoRefreshIntervalId=setInterval(async()=>{currentSelectedPairId&&"visible"===document.visibilityState&&!isLoadingData&&await loadChartDataForCurrentPair()},15e3)}); </script> </body></html>"""

@flask_app_instance.route('/')
def serve_index_page(): return render_template_string(HTML_CHART_TEMPLATE)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs_flask(): 
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs_info = []
    for pair_id, pair_data in data_manager_view.items():
        cfg = pair_data.get("config", {})
        if cfg.get("enabled", True): active_pairs_info.append({"id": pair_id, "name": cfg.get('pair_name', pair_id)})
    active_pairs_info.sort(key=lambda x: x['name']) # Sort pairs alphabetically
    return jsonify(active_pairs_info)

@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend_flask(pair_id_from_request): 
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager: return jsonify({"error": "Pair not found"}), 404
        # Deepcopy only the specific pair data to avoid holding lock for too long during prepare_chart_data
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))
    
    if not pair_data_snapshot: return jsonify({"error": "Data empty for pair", "ohlc":[], "supertrend_series":[]}), 200
    
    # Pass a temporary manager containing only the snapshot of the requested pair
    temp_manager_for_preparation = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager_for_preparation)
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    if not prepared_data.get("ohlc"): return jsonify({"error": "No OHLC data to display", **prepared_data}), 200 # Send other info if ohlc is empty
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        # Silence Werkzeug's default logger to avoid duplicate logs if our logger is also capturing them
        flask_log = logging.getLogger('werkzeug')
        flask_log.setLevel(logging.ERROR) 
        # For more control, could add a handler that formats werkzeug logs to match ours
        # flask_log.handlers = [ch] # Example: use our console handler for werkzeug
        
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask: 
        log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")

# CHART_INTEGRATION_END


# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE); input(); return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE); input(); return

    animated_text_display("=========== SUPERTREND STRATEGY START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    key_idx_display = api_key_manager.get_current_key_index()
    key_val_display = api_key_manager.get_current_key()
    key_val_display_short = ("..." + key_val_display[-3:]) if key_val_display and len(key_val_display) > 8 else key_val_display
    log_info(f"Menggunakan API Key Index: {key_idx_display} ({key_val_display_short}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}_{config.get('exchange','CCCAGG')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}" # Used for logging
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']} ({config['timeframe']}){AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0,
            "last_attempt_after_all_keys_failed": None # Untuk cooldown
        }
        # Initial update to shared manager for this pair
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_indicators_init = config.get('atr_length', 10) + 50 # ATR + buffer
        initial_candles_target_for_pair = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        
        initial_candles = []
        initial_fetch_successful = False
        max_initial_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_key_attempts_done = 0
        original_api_key_index_for_initial = api_key_manager.get_current_key_index() # Track if key changed during initial fetch

        while initial_key_attempts_done < max_initial_key_attempts and not initial_fetch_successful:
            current_api_key_init = api_key_manager.get_current_key()
            if not current_api_key_init:
                log_error(f"BIG DATA: Semua API key habis (global) sebelum mencoba fetch untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break 
            
            log_info(f"BIG DATA: Mencoba fetch awal ({initial_candles_target_for_pair} candles) untuk {config['pair_name']} dengan key index {api_key_manager.get_current_key_index()} (Attempt {initial_key_attempts_done + 1}/{max_initial_key_attempts})", pair_name=config['pair_name'])
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target_for_pair, config['exchange'], current_api_key_init, config['timeframe'], pair_name=config['pair_name'])
                initial_fetch_successful = True 
            except APIKeyError:
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): 
                    log_error(f"BIG DATA: Gagal beralih ke key berikutnya, semua key habis untuk {config['pair_name']}.", pair_name=config['pair_name'])
                    break 
            except requests.exceptions.RequestException as e_req_init:
                log_error(f"BIG DATA: Error Jaringan saat fetch awal {config['pair_name']}: {e_req_init}. Tidak ganti key.", pair_name=config['pair_name'])
                break 
            except Exception as e_gen_init:
                log_exception(f"BIG DATA: Error Umum saat fetch awal {config['pair_name']}: {e_gen_init}. Tidak ganti key.", pair_name=config['pair_name'])
                break 
            initial_key_attempts_done += 1

        if not initial_fetch_successful or not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini akan dilewati di loop utama hingga cooldown.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts + 1 
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue 

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])
        
        if initial_candles:
            min_len_for_warmup = config.get('atr_length', 10) + 2 
            if len(initial_candles) >= min_len_for_warmup:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state {config['pair_name']}...", pair_name=config['pair_name'])
                # Warm-up: Iterate through historical data to set initial Supertrend state without generating signals
                temp_state_for_warmup = get_initial_strategy_state() # Fresh state for warm-up
                for i_warmup in range(min_len_for_warmup -1, len(initial_candles) -1): # Process up to the second to last candle
                    historical_slice = initial_candles[:i_warmup+1] 
                    if len(historical_slice) < min_len_for_warmup: continue
                    
                    # Simulate running logic but prevent actual trade signals/notifications during warm-up
                    # Create a temporary config or modify state to suppress notifications if needed
                    temp_state_for_warmup = run_strategy_logic(historical_slice, config, temp_state_for_warmup, {"api_settings": {}}) # Pass dummy global settings

                # After warm-up, assign the final warmed-up state (excluding any simulated position)
                local_crypto_data_manager[pair_id]["strategy_state"] = temp_state_for_warmup
                local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] = 0
                local_crypto_data_manager[pair_id]["strategy_state"]["entry_price_custom"] = None
                local_crypto_data_manager[pair_id]["strategy_state"]["last_notification_type"] = None # Reset notification state
                local_crypto_data_manager[pair_id]["strategy_state"]["last_notification_price"] = None

                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])

        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"]:
                email_cfg_for_big_data = {**config, 'pair_name': config['pair_name']} # ensure pair_name is in context
                send_email_notification(
                    f"Data Downloading Complete: {config['pair_name']}", 
                    f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now ready for live analysis on {config['pair_name']}.", 
                    email_cfg_for_big_data
                )
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({config['pair_name']}) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
    
    try: 
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False # Track if any fetch happened to adjust sleep if all keys fail globally

            # Create a list of pair_ids to iterate over, to allow removing items from local_crypto_data_manager if needed
            current_pair_ids_to_process = list(local_crypto_data_manager.keys())

            for pair_id in current_pair_ids_to_process:
                if pair_id not in local_crypto_data_manager: continue # Pair might have been removed (e.g. due to persistent error)
                
                data_per_pair = local_crypto_data_manager[pair_id]
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    cooldown_duration = 3600 # 1 hour cooldown
                    time_since_last_failure = (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds()
                    if time_since_last_failure < cooldown_duration:
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, cooldown_duration - time_since_last_failure)
                        # log_debug(f"{pair_name_for_log} in cooldown. Remaining: {cooldown_duration - time_since_last_failure:.0f}s", pair_name=pair_name_for_log)
                        continue
                    else: 
                        log_info(f"Cooldown berakhir untuk {pair_name_for_log}. Mencoba lagi.", pair_name=pair_name_for_log)
                        data_per_pair["data_fetch_failed_consecutively"] = 0 
                        data_per_pair["last_attempt_after_all_keys_failed"] = None # Reset cooldown timestamp

                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Fetch more aggressively during big data collection, but respect API limits
                    limit_based_on_tf = {"minute": 15, "hour": 60, "day": 300}.get(config_for_pair.get('timeframe'), 60)
                    required_interval = min(limit_based_on_tf, config_for_pair.get('refresh_interval_seconds', 60))
                else: # Live trading, ensure minimum interval if set
                     required_interval = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, config_for_pair.get('refresh_interval_seconds',60))


                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch)
                    continue
                
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                if data_per_pair["big_data_collection_phase_active"]: animated_text_display(f"\n--- BIG DATA {pair_name_for_log} ({num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}) [{config_for_pair['timeframe']}] ---", color=AnsiColors.BOLD+AnsiColors.MAGENTA, new_line=False)
                else: animated_text_display(f"\n--- LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {num_candles_before_fetch} [{config_for_pair['timeframe']}] ---", color=AnsiColors.BOLD+AnsiColors.CYAN, new_line=False)

                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 
                if data_per_pair["big_data_collection_phase_active"]:
                    needed_for_big_data = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed_for_big_data <=0 : 
                        fetch_update_successful = True # Already have enough, no need to fetch
                        limit_fetch_update = 0 # Set to 0 to skip fetch block
                    else: limit_fetch_update = min(needed_for_big_data, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0 : # Only fetch if needed
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    original_api_key_index_for_this_update_cycle = api_key_manager.get_current_key_index() # Track if key changed

                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE: Semua API key habis (global) untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        
                        # log_debug(f"UPDATE: Mencoba fetch untuk {pair_name_for_log} dengan key idx {api_key_manager.get_current_key_index()} (Attempt {update_key_attempts_done + 1}/{max_update_key_attempts})", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_update, config_for_pair['timeframe'], pair_name=pair_name_for_log)
                            fetch_update_successful = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 
                            any_data_fetched_this_cycle = True # Mark that a fetch attempt was made
                            if api_key_manager.get_current_key_index() != original_api_key_index_for_this_update_cycle:
                                log_info(f"UPDATE: Fetch berhasil dengan key index {api_key_manager.get_current_key_index()} setelah retry untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                        except APIKeyError:
                            log_warning(f"UPDATE: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] +=1
                            if not api_key_manager.switch_to_next_key():
                                log_error(f"UPDATE: Gagal beralih, semua key habis untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        except requests.exceptions.RequestException as e_req_upd:
                            log_error(f"UPDATE: Error Jaringan {pair_name_for_log}: {e_req_upd}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break 
                        except Exception as e_gen_upd:
                            log_exception(f"UPDATE: Error Umum {pair_name_for_log}: {e_gen_upd}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break
                        update_key_attempts_done += 1
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 
                    log_warning(f"Semua API key gagal untuk {pair_name_for_log} di siklus ini. Masuk cooldown.", pair_name=pair_name_for_log)


                if not fetch_update_successful or (limit_fetch_update > 0 and not new_candles_batch): # Check if fetch was intended but failed or returned empty
                    if fetch_update_successful and not new_candles_batch and not data_per_pair["big_data_collection_phase_active"]: # Successfully fetched, but no new data
                        log_info(f" Tidak ada candle baru diterima untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                    elif not fetch_update_successful and limit_fetch_update > 0: # Fetch was intended and failed
                         log_error(f" {AnsiColors.RED}Gagal update {pair_name_for_log}.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    # else: # e.g. fetch_update_successful is True because limit_fetch_update was 0 (big data complete)
                    #    sys.stdout.write(f"{AnsiColors.GREEN} Data Lengkap.{AnsiColors.ENDC}\n")
                    #    sys.stdout.flush()


                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                    with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair) # Save state even on failed fetch (e.g. consecutive failure count)
                    sys.stdout.write("\n"); sys.stdout.flush() # Ensure newline after animated text
                    continue # Next pair

                sys.stdout.write(f"{AnsiColors.GREEN} OK.{AnsiColors.ENDC}\n") # For the animated text if fetch was OK or not needed.
                sys.stdout.flush()

                if new_candles_batch: # Only merge if there's something new
                    merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                    newly_added_count_this_batch, updated_count_this_batch = 0,0
                    for candle in new_candles_batch:
                        ts = candle['timestamp']
                        if ts not in merged_candles_dict: 
                            merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                        elif merged_candles_dict[ts]['close'] != candle['close'] or merged_candles_dict[ts]['high'] != candle['high'] or merged_candles_dict[ts]['low'] != candle['low']: # Check if candle data changed
                            merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                    
                    data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                    if newly_added_count_this_batch > 0 or updated_count_this_batch > 0: 
                        log_info(f"{newly_added_count_this_batch} baru, {updated_count_this_batch} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)

                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if not data_per_pair["big_data_email_sent"]:
                            email_cfg_bd_done = {**config_for_pair, 'pair_name': pair_name_for_log}
                            send_email_notification(f"Data Downloading Complete: {pair_name_for_log}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name_for_log}.", email_cfg_bd_done)
                            data_per_pair["big_data_email_sent"] = True
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                
                # Trim excess candles if not in big data phase and list is too long
                if not data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 100: # Keep a bit more than target for calcs
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 50):]


                min_len_for_logic_run_live = config_for_pair.get('atr_length', 10) + 2
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run_live:
                    # Process logic if new candles arrived OR if it's the first run after big data collection completed
                    process_logic_now = ( (newly_added_count_this_batch + updated_count_this_batch > 0) or
                                         (not data_per_pair["big_data_collection_phase_active"] and 
                                          num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and 
                                          len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) 
                                       )
                    if process_logic_now :
                         # log_debug(f"Running strategy for {pair_name_for_log}...", pair_name=pair_name_for_log)
                         data_per_pair["strategy_state"] = run_strategy_logic(data_per_pair["all_candles_list"], config_for_pair, data_per_pair["strategy_state"], global_settings_dict)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None and api_key_manager.total_keys() > 0: 
                # All keys were exhausted globally during this cycle's attempts
                log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY GLOBAL GAGAL di siklus ini. Skrip akan tidur 1 jam sebelum mencoba lagi.{AnsiColors.ENDC}", pair_name="SYSTEM")
                sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: 
                sleep_duration = min(15, int(min_overall_next_refresh_seconds) if min_overall_next_refresh_seconds != float('inf') else 15)
                sleep_duration = max(1, sleep_duration) # Ensure at least 1s sleep
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
            
            if sleep_duration > 0: 
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s (Next check: ~{datetime.now() + timedelta(seconds=sleep_duration):%H:%M:%S})...")
            else: # Should not happen if logic above is correct, but as a fallback
                time.sleep(1)

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan oleh pengguna.{AnsiColors.ENDC}",color=AnsiColors.ORANGE, new_line=True)
    except Exception as e_main_loop: 
        log_exception(f"{AnsiColors.RED}Error Kritis di loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM_CRITICAL")
        animated_text_display(f"\n{AnsiColors.RED}ERROR KRITIS DALAM LOOP UTAMA: {e_main_loop}{AnsiColors.ENDC}", color=AnsiColors.RED, new_line=True)
    finally: 
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}",color=AnsiColors.HEADER, new_line=True)
        input("Tekan Enter untuk kembali ke menu utama...")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    # Start Flask server in a separate thread
    # Ensure Flask thread is daemon so it exits when main thread exits
    flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
    flask_thread.start()
    
    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Supertrend Strategy Runner =========", color=AnsiColors.HEADER)
        
        pick_title_main = ""
        # Reload settings in case they were changed by another instance or manually (less likely for desktop app)
        # For multi-instance scenarios, IPC or a proper database for settings would be better.
        # For this script, loading here ensures `settings_menu` changes are reflected if user goes back and forth.
        settings = load_settings() 
        
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs: 
            pick_title_main += f"{AnsiColors.CYAN}--- Crypto Aktif ({len(active_cfgs)}) ---{AnsiColors.ENDC}\n"
            for i,c in enumerate(active_cfgs):
                pick_title_main += f"  {AnsiColors.GREEN}{i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')}, {c.get('exchange','CCCAGG')}){AnsiColors.ENDC}\n"
        else: 
            pick_title_main += f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}\n"
        
        api_s_main = settings.get("api_settings", {})
        pk_disp_val = api_s_main.get('primary_key','N/A')
        pk_disp = ("..." + pk_disp_val[-5:]) if pk_disp_val and len(pk_disp_val) > 10 and pk_disp_val not in ["YOUR_PRIMARY_KEY","N/A"] else pk_disp_val
        num_recovery = len([k for k in api_s_main.get('recovery_keys',[]) if k])

        pick_title_main += f"{AnsiColors.BLUE}-----------------------------------------------{AnsiColors.ENDC}\n"
        pick_title_main += f"Primary API Key: {AnsiColors.MAGENTA}{pk_disp}{AnsiColors.ENDC} | Recovery Keys: {AnsiColors.MAGENTA}{num_recovery}{AnsiColors.ENDC}\n"
        pick_title_main += f"Chart Server: {AnsiColors.YELLOW_BG}{AnsiColors.BLUE}http://localhost:5001{AnsiColors.ENDC}\n"
        pick_title_main += f"{AnsiColors.BLUE}-----------------------------------------------{AnsiColors.ENDC}\n"
        pick_title_main += "Pilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        selected_option, main_idx = pick(main_opts, pick_title_main, indicator='=>', options_map_func=lambda x: AnsiColors.BOLD + x + AnsiColors.ENDC)
        
        if main_idx == 0: 
            # Ensure settings are fresh before starting trading
            current_fresh_settings = load_settings() 
            start_trading(current_fresh_settings, shared_crypto_data_manager, shared_data_lock)
        elif main_idx == 1: 
            settings = settings_menu(settings) # settings_menu saves inside itself, and returns the latest
        elif main_idx == 2: 
            log_info("Aplikasi ditutup oleh pengguna.", pair_name="SYSTEM")
            break 
            
    animated_text_display("\nTerima kasih telah menggunakan skrip ini!", color=AnsiColors.MAGENTA)
    show_spinner(0.5, "Exiting")
    print(AnsiColors.ENDC) # Ensure colors are reset

if __name__ == "__main__":
    try: 
        main_menu()
    except KeyboardInterrupt: 
        clear_screen_animated()
        animated_text_display("\nAplikasi dihentikan oleh pengguna (Ctrl+C di menu utama).", color=AnsiColors.ORANGE)
    except Exception as e_global: 
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}ERROR KRITIKAL GLOBAL: {e_global}{AnsiColors.ENDC}")
        log_exception("MAIN_APP_ERROR_UNHANDLED:",pair_name="SYS_CRITICAL")
        input("Tekan Enter untuk keluar...")
    finally: 
        # Ensure all output buffers are flushed before exiting
        sys.stdout.flush()
        sys.stderr.flush()
        print(AnsiColors.ENDC) # Final reset of colors
