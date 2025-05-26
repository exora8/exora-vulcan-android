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


SETTINGS_FILE = "settings_ema_trend.json" # Ganti nama file settings
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500 
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15
EMA_LENGTH_FIXED = 500 # EMA length is fixed at 500

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
        "ema_lookback_period": 10, # BARU: Sesuai PineScript
        # Parameter Supertrend dan SL/TP dihilangkan
        "enable_email_notifications": False, # Untuk notifikasi sistem (mis. Big Data tercapai)
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
                # Pastikan semua keys dari template ada di config yang diload
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
                # Hapus key lama yang tidak relevan lagi (opsional, tapi lebih bersih)
                keys_to_remove = ["atr_length", "factor", "emergency_sl_percent", 
                                  "profit_target_percent_activation", "trailing_stop_gap_percent",
                                  "ma_length", "stoch_length", "stoch_smooth_k", "stoch_smooth_d",
                                  "stoch_overbought", "stoch_oversold", "left_strength", 
                                  "right_strength", "enable_secure_fib", "secure_fib_check_price"]
                for old_key in keys_to_remove:
                    if old_key in crypto_cfg:
                        del crypto_cfg[old_key]
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

    animated_text_display("\n-- Parameter EMA Trend Detector --", color=AnsiColors.HEADER)
    try:
        new_config["ema_lookback_period"] = int(input(f"{AnsiColors.BLUE}EMA Lookback Period [{new_config.get('ema_lookback_period',10)}]: {AnsiColors.ENDC}").strip() or new_config.get('ema_lookback_period',10))
    except ValueError:
        print(f"{AnsiColors.RED}Input EMA Lookback Period tidak valid. Default digunakan.{AnsiColors.ENDC}")
        def_cfg = get_default_crypto_config()
        new_config["ema_lookback_period"] = new_config.get('ema_lookback_period', def_cfg['ema_lookback_period'])
    
    # Input SL/TP dihilangkan
    
    animated_text_display("\n-- Notifikasi Email (Gmail) - Untuk Notif Sistem --", color=AnsiColors.HEADER)
    email_enable_input = input(f"Aktifkan Notifikasi Email Sistem? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
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
        title = f"--- Menu Pengaturan ---\nAPI Key: {pkd} | Recovery: {nrk} | Termux: {tns}\nStrategi: EMA 500 Trend Detector\nCrypto Pairs:\n" # Ganti nama strategi
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]): title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({cfg.get('timeframe','?')}, EMA Lookback: {cfg.get('ema_lookback_period','?')}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'}\n"
        title += "----------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux", "Tambah Crypto Pair", "Ubah Crypto Pair", "Hapus Crypto Pair", "Kembali"]
        _, action_idx = pick(opts, title, indicator='=>')
        clear_screen_animated()
        try:
            if action_idx == 0: # Primary API Key
                new_pk = input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip()
                if new_pk: api_s["primary_key"] = new_pk
                elif not api_s.get('primary_key'): api_s["primary_key"] = "YOUR_PRIMARY_KEY" # Set to placeholder if cleared and was empty

            elif action_idx == 1: # Recovery API Keys
                while True:
                    clear_screen_animated()
                    current_recovery = api_s.get('recovery_keys', [])
                    rec_title = "--- Kelola Recovery API Keys ---\n"
                    if not current_recovery: rec_title += "  (Tidak ada recovery key)\n"
                    else:
                        for i_rec, key_rec in enumerate(current_recovery):
                             rec_title += f"  {i_rec+1}. {key_rec[:5]}...{key_rec[-3:] if len(key_rec)>8 else key_rec}\n"
                    rec_title += "----------------------------\nPilih:"
                    rec_opts = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Menu Pengaturan"]
                    _, rec_action_idx = pick(rec_opts, rec_title, indicator='=>')
                    
                    if rec_action_idx == 0: # Tambah
                        new_rec_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_rec_key:
                            api_s.setdefault('recovery_keys',[]).append(new_rec_key)
                            save_settings(current_settings)
                            animated_text_display("Recovery key ditambahkan.", color=AnsiColors.GREEN)
                        else: animated_text_display("Input kosong, tidak ada key ditambahkan.", color=AnsiColors.ORANGE)
                    elif rec_action_idx == 1: # Hapus
                        if not api_s.get('recovery_keys'):
                            animated_text_display("Tidak ada recovery key untuk dihapus.", color=AnsiColors.ORANGE); time.sleep(1); continue
                        del_rec_opts = [f"{k_rec[:5]}...{k_rec[-3:] if len(k_rec)>8 else k_rec}" for k_rec in api_s['recovery_keys']] + ["Batal"]
                        animated_text_display("Pilih recovery key untuk dihapus:", color=AnsiColors.HEADER,new_line=False)
                        _, del_rec_key_idx = pick(del_rec_opts, "Pilih recovery key untuk dihapus:", indicator='=>')
                        if del_rec_key_idx < len(api_s['recovery_keys']):
                            api_s['recovery_keys'].pop(del_rec_key_idx)
                            save_settings(current_settings)
                            animated_text_display("Recovery key dihapus.", color=AnsiColors.GREEN)
                    elif rec_action_idx == 2: break # Kembali ke menu pengaturan utama
                    show_spinner(0.5, "Memproses...")

            elif action_idx == 2: # Email Global Notif Sistem
                api_s['enable_global_email_notifications_for_key_switch'] = input(f"Aktifkan Email Notif Sistem Global? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip() == 'true'
                api_s['email_sender_address'] = (input(f"Alamat Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Email Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Alamat Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
            
            elif action_idx == 3: # Notifikasi Termux
                api_s['enable_termux_notifications'] = input(f"Aktifkan Notifikasi Termux? (true/false) [{api_s.get('enable_termux_notifications',False)}]: ").lower().strip() == 'true'
            
            elif action_idx == 4: current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_idx == 5:
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk diubah."); show_spinner(1,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:")
                if edit_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 6:
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk dihapus."); show_spinner(1,""); continue
                del_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, del_c_idx = pick(del_opts, "Pilih pair untuk dihapus:")
                if del_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"].pop(del_c_idx)
            elif action_idx == 7: break # Kembali
            current_settings["api_settings"] = api_s 
            save_settings(current_settings)
            if action_idx not in [1,7]: show_spinner(1, "Disimpan...") 
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = {"minute": "histominute", "hour": "histohour", "day": "histoday"}.get(timeframe, "histohour")
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 20 

    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
    
    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts: params["toTs"] = current_to_ts
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429]: 
                err_msg = response.json().get('Message', f"HTTP Error {response.status_code}") if response.content else f"HTTP Error {response.status_code}"
                log_warning(f"API Key Error (HTTP {response.status_code}): {err_msg} | Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {err_msg}")
            response.raise_for_status() 
            data = response.json()
            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error')
                key_err_patterns = ["api key is invalid", "apikey_is_missing", "rate limit", "monthly_calls", "tier"]
                if any(p.lower() in err_msg.lower() for p in key_err_patterns):
                    log_warning(f"API Key Error (JSON): {err_msg} | Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
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
            
            if current_to_ts and all_accumulated_candles and batch and batch[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']: batch.pop() 
            if not batch and current_to_ts: break 

            all_accumulated_candles = batch + all_accumulated_candles
            if raw_candles: current_to_ts = raw_candles[0]['time']
            else: break

            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
            if len(raw_candles) < limit_call: break 
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.2) 
        except APIKeyError: raise 
        except requests.exceptions.RequestException as e_req: log_error(f"Kesalahan Jaringan: {e_req}", pair_name=pair_name); break 
        except Exception as e_gen: log_exception(f"Error lain dalam fetch_candles: {e_gen}", pair_name=pair_name); break
    
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai')
    return all_accumulated_candles

# --- LOGIKA STRATEGI (EMA 500 Trend Detector) ---
def get_initial_strategy_state():
    return {
        "previous_trend_is_uptrend": False,
        "previous_trend_is_downtrend": False,
        "last_ema500_value_for_chart": None, 
        "current_trend_color_for_chart": "gray" 
    }

def calculate_ema(prices, period):
    if not prices or len(prices) < period or period <= 0:
        return [None] * len(prices)
    
    ema_values = [None] * len(prices)
    
    # Find first valid price to start calculation
    first_valid_price_index = -1
    for i, p in enumerate(prices):
        if p is not None:
            first_valid_price_index = i
            break
            
    if first_valid_price_index == -1: # No valid prices at all
        return ema_values

    # Check if enough data points *after* the first valid price for initial SMA
    if (len(prices) - first_valid_price_index) < period:
        return ema_values

    # Calculate initial SMA for the first EMA value
    sma_sum = 0.0
    valid_prices_in_initial_period = 0
    initial_period_end_index = first_valid_price_index + period -1

    for i in range(first_valid_price_index, initial_period_end_index + 1):
        if prices[i] is not None:
            sma_sum += prices[i]
            valid_prices_in_initial_period +=1
        else: # Gap in data within initial period
            # Shift first_valid_price_index and retry SMA calc or return None
            # For simplicity now, if there's a None in the first `period` prices, we can't start.
            # This part can be made more robust to handle initial Nones.
            # For now, let's assume calculate_ema is called with enough leading non-None data.
            # Or, more simply, the first EMA value will be None if any of its components are None.
            return ema_values # Cannot calculate initial SMA reliably

    if valid_prices_in_initial_period < period: # Should be caught by gap check
         return ema_values

    ema_values[initial_period_end_index] = sma_sum / period
    multiplier = 2 / (period + 1)

    # Calculate subsequent EMA values
    for i in range(initial_period_end_index + 1, len(prices)):
        if prices[i] is not None and ema_values[i-1] is not None:
            ema_values[i] = (prices[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        elif prices[i] is not None and ema_values[i-1] is None:
            # Try to re-initialize EMA if previous was None due to a gap
            # This requires looking back `period` non-None prices again.
            # For this implementation, if ema[i-1] is None, ema[i] will also be None
            # unless we add logic for re-initialization.
            # For now, a sustained None in prices will lead to sustained None in EMA.
            pass # ema_values[i] remains None
        # If prices[i] is None, ema_values[i] remains None.
    return ema_values


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings, is_warmup=False):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    lookback_period = crypto_config.get('ema_lookback_period', 10)
    # ema_length is fixed at EMA_LENGTH_FIXED (500)

    min_data_needed = EMA_LENGTH_FIXED + lookback_period
    if len(candles_history) < min_data_needed:
        # log_debug(f"Not enough data for EMA trend ({len(candles_history)}/{min_data_needed}) for {pair_name}", pair_name=pair_name)
        strategy_state["last_ema500_value_for_chart"] = None
        strategy_state["current_trend_color_for_chart"] = "gray" # PineScript: color.new(color.gray, 50)
        return strategy_state

    closes = [c.get('close') for c in candles_history]
    ema500_series = calculate_ema(closes, EMA_LENGTH_FIXED)

    current_idx = len(candles_history) - 1
    ema_value = ema500_series[current_idx]
    
    strategy_state["last_ema500_value_for_chart"] = ema_value

    past_ema_idx = current_idx - lookback_period
    # past_ema_idx should be >= EMA_LENGTH_FIXED - 1 for ema500_series[past_ema_idx] to be potentially valid
    if past_ema_idx < (EMA_LENGTH_FIXED - 1) : 
        strategy_state["current_trend_color_for_chart"] = "gray"
        strategy_state["previous_trend_is_uptrend"] = False # Reset if not enough history for comparison
        strategy_state["previous_trend_is_downtrend"] = False
        return strategy_state

    ema_past_value = ema500_series[past_ema_idx]

    if ema_value is None or ema_past_value is None:
        # log_debug(f"EMA value or past EMA value is None for {pair_name}. Current: {ema_value}, Past: {ema_past_value}", pair_name=pair_name)
        strategy_state["current_trend_color_for_chart"] = "gray"
        strategy_state["previous_trend_is_uptrend"] = False
        strategy_state["previous_trend_is_downtrend"] = False
        return strategy_state

    is_uptrend_now = ema_value > ema_past_value
    is_downtrend_now = ema_value < ema_past_value
    # is_sideways_now handled by the 'else' in color determination

    # Determine color for chart (mirrors PineScript logic)
    if is_uptrend_now:
        strategy_state["current_trend_color_for_chart"] = "green" # PineScript: color.new(color.green, 0)
    elif is_downtrend_now:
        strategy_state["current_trend_color_for_chart"] = "red"   # PineScript: color.new(color.red, 0)
    else: # Sideways or equal
        strategy_state["current_trend_color_for_chart"] = "yellow" # PineScript: color.new(color.yellow, 0)


    if not is_warmup: # Only send notifications if not in warm-up phase
        if is_uptrend_now and not strategy_state.get("previous_trend_is_uptrend", False):
            message = f"EMA500 Trend UP: This is a good time to trade on {pair_name}"
            log_info(f"{AnsiColors.GREEN}{message}{AnsiColors.ENDC}", pair_name=pair_name)
            send_termux_notification(f"UPTREND: {pair_name}", message, global_settings, pair_name_for_log=pair_name)
            # play_notification_sound() # Optional sound

        elif is_downtrend_now and not strategy_state.get("previous_trend_is_downtrend", False):
            message = f"EMA500 Trend DOWN: This is not a good time to trade on {pair_name}"
            log_info(f"{AnsiColors.RED}{message}{AnsiColors.ENDC}", pair_name=pair_name)
            send_termux_notification(f"DOWNTREND: {pair_name}", message, global_settings, pair_name_for_log=pair_name)
            # play_notification_sound() # Optional sound
            
    # Update previous trend state for the next iteration
    strategy_state["previous_trend_is_uptrend"] = is_uptrend_now
    strategy_state["previous_trend_is_downtrend"] = is_downtrend_now
    
    return strategy_state

# CHART_INTEGRATION_START & Flask Endpoints
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id, snapshot):
    if pair_id not in snapshot: return None
    data = snapshot[pair_id]
    hist = data.get("all_candles_list", [])[-TARGET_BIG_DATA_CANDLES:] 
    cfg = data.get("config", {})
    state = data.get("strategy_state", {})
    
    ohlc_data = []
    ema500_series_data = []
    
    pair_display_name = cfg.get('pair_name', pair_id)
    ema_lookback_chart = cfg.get('ema_lookback_period', 10)

    if not hist:
        return {
            "ohlc": [], "ema500_series": [], "ema500_line_color": "gray",
            "annotations_yaxis": [], "pair_name": pair_display_name,
            "last_updated_tv": None, "ema_lookback_label": ema_lookback_chart,
            "strategy_state_info": {"ema500_value": None, "ema_trend_color": "gray"}
        }

    closes_hist = [c.get('close') for c in hist]
    ema500_values_hist = calculate_ema(closes_hist, EMA_LENGTH_FIXED)

    for i, c in enumerate(hist):
        if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = c['timestamp'].timestamp() * 1000
            ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})
            if ema500_values_hist[i] is not None:
                ema500_series_data.append({'x': ts_ms, 'y': ema500_values_hist[i]})
    
    current_ema_color = state.get("current_trend_color_for_chart", "gray")
    last_ema_value = state.get("last_ema500_value_for_chart")

    return {
        "ohlc": ohlc_data,
        "ema500_series": ema500_series_data,
        "ema500_line_color": current_ema_color,
        "annotations_yaxis": [], # SL/Entry annotations removed
        "pair_name": pair_display_name,
        "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
        "ema_lookback_label": ema_lookback_chart, # Keep for info
        "strategy_state_info": {
            "ema500_value": last_ema_value,
            "ema_trend_color": current_ema_color,
            # Add isUptrend/isDowntrend from state for more detailed display if needed
            "is_uptrend": state.get("previous_trend_is_uptrend", False), # Displaying the state of the last processed candle
            "is_downtrend": state.get("previous_trend_is_downtrend", False)
        }
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMA 500 Trend Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width: 100%; max-width: 1200px; }
        select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor: pointer; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; }
        h1 { color: #00bcd4; margin-bottom: 10px; font-size: 1.5em; }
        #lastUpdatedLabel { font-size: .8em; color: #aaa; margin-left: auto; }
        #strategyInfoLabel { font-size: .8em; color: #ffd700; margin-left: 10px; white-space: pre; }
    </style>
</head>
<body>
    <h1>EMA 500 Trend Detector Chart</h1>
    <div id="controls">
        <label for="pairSelector">Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh</button>
        <span id="strategyInfoLabel">Status: -</span>
        <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container"><div id="chart"></div></div>
    <script>
        let activeChart, currentSelectedPairId = "", lastKnownDataTimestamp = null, autoRefreshIntervalId = null, isLoadingData = false;
        const initialChartOptions = {
            series: [
                { name: "Candlestick", type: "candlestick", data: [] },
                { name: "EMA 500", type: "line", data: [] }
            ],
            chart: { type: "candlestick", height: 550, background: "#2a2a2a", animations: { enabled: false }, toolbar: { show: true } },
            theme: { mode: "dark" },
            title: { text: "Memuat Data Pair...", align: "left", style: { color: "#e0e0e0" } },
            xaxis: { type: "datetime", labels: { style: { colors: "#aaa" } }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: "#aaa" }, formatter: v => v ? v.toFixed(5) : "" } },
            stroke: { width: [1, 2], curve: "straight" }, // EMA line thickness
            markers: { size: 0 },
            colors: ["#FEB019", "#888888"], // Default color for candlestick, EMA line
            grid: { borderColor: "#444" },
            annotations: { yaxis: [], points: [] }, // Annotations can be cleared or used for other things if needed
            tooltip: {
                theme: "dark", shared: true, intersect: false,
                custom: function({ series, seriesIndex, dataPointIndex, w }) {
                    let ohlcOpen, ohlcHigh, ohlcLow, ohlcClose, emaValue;
                    const candleSeriesIdx = w.globals.series.findIndex(s => s.type === 'candlestick');
                    const emaSeriesIdx = w.globals.series.findIndex(s => s.name.startsWith("EMA 500"));

                    if (candleSeriesIdx !== -1 && w.globals.seriesCandleO[candleSeriesIdx]?.[dataPointIndex] !== undefined) {
                        [ohlcOpen, ohlcHigh, ohlcLow, ohlcClose] = [
                            w.globals.seriesCandleO[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleH[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleL[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleC[candleSeriesIdx][dataPointIndex]
                        ];
                    }
                    if (emaSeriesIdx !== -1 && series[emaSeriesIdx]?.[dataPointIndex] !== undefined && w.config.series[emaSeriesIdx].data[dataPointIndex]) {
                        emaValue = w.config.series[emaSeriesIdx].data[dataPointIndex].y;
                    }

                    let html = '<div style="padding:5px 10px;background:#333;color:#fff;border:1px solid #555;">';
                    if (ohlcOpen !== undefined) {
                        html += ['O', 'H', 'L', 'C'].map((label, idx) => 
                            `<div>${label}: <span style="font-weight:bold;">${[ohlcOpen, ohlcHigh, ohlcLow, ohlcClose][idx].toFixed(5)}</span></div>`
                        ).join('');
                    }
                    if (emaValue !== undefined) {
                        html += `<div>EMA 500: <span style="font-weight:bold;">${emaValue.toFixed(5)}</span></div>`;
                    }
                    html += '</div>';
                    return (ohlcOpen !== undefined || emaValue !== undefined) ? html : "";
                }
            },
            noData: { text: "Tidak ada data.", align: "center", style: { color: "#ccc" } }
        };

        async function fetchAvailablePairs() {
            try {
                const response = await fetch("/api/available_pairs");
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const pairs = await response.json();
                const selector = document.getElementById("pairSelector");
                selector.innerHTML = ""; 
                if (pairs.length > 0) {
                    pairs.forEach(pair => {
                        const option = document.createElement("option");
                        option.value = pair.id; option.textContent = pair.name;
                        selector.appendChild(option);
                    });
                    currentSelectedPairId = selector.value || pairs[0].id;
                    loadChartDataForCurrentPair();
                } else {
                    selector.innerHTML = '<option value="">No pairs</option>';
                    if (activeChart) { activeChart.destroy(); activeChart = null; }
                    document.getElementById("chart").innerHTML = "No pairs configured.";
                }
            } catch (error) {
                console.error("Error fetching available pairs:", error);
                if (activeChart) { activeChart.destroy(); activeChart = null; }
                document.getElementById("chart").innerHTML = `Error loading pairs: ${error.message}`;
            }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById("pairSelector").value;
            lastKnownDataTimestamp = null; 
            loadChartDataForCurrentPair();
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId || isLoadingData) return;
            isLoadingData = true;
            document.getElementById("lastUpdatedLabel").textContent = `Loading ${currentSelectedPairId}...`;
            try {
                const response = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                if (data && data.ohlc) { // Check if data.ohlc exists
                    if (data.last_updated_tv && data.last_updated_tv === lastKnownDataTimestamp) {
                        console.log("Chart data is unchanged based on timestamp.");
                        document.getElementById("lastUpdatedLabel").textContent = `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                        const sstate = data.strategy_state_info || {};
                         document.getElementById("strategyInfoLabel").textContent = 
                            `EMA500: ${sstate.ema500_value !== null && sstate.ema500_value !== undefined ? sstate.ema500_value.toFixed(3) : "N/A"}\n` +
                            `Trend: ${sstate.ema_trend_color || "N/A"}`;
                        isLoadingData = false;
                        return;
                    }
                    lastKnownDataTimestamp = data.last_updated_tv;
                     document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Data Loaded";
                    
                    const strategyState = data.strategy_state_info || {};
                     document.getElementById("strategyInfoLabel").textContent = 
                        `EMA500: ${strategyState.ema500_value !== null && strategyState.ema500_value !== undefined ? strategyState.ema500_value.toFixed(3) : "N/A"}\n`+
                        `Trend: ${strategyState.ema_trend_color || "N/A"} (${strategyState.is_uptrend ? "Up" : strategyState.is_downtrend ? "Down" : "Side"})`;

                    const emaLineColor = data.ema500_line_color || initialChartOptions.colors[1];

                    const chartOptionsUpdate = {
                        ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${data.pair_name} - EMA 500 (Lookback: ${data.ema_lookback_label})` },
                        series: [
                            { name: "Candlestick", type: "candlestick", data: data.ohlc || [] },
                            { name: "EMA 500", type: "line", data: data.ema500_series || [], color: emaLineColor }
                        ],
                        annotations: { yaxis: data.annotations_yaxis || [] , points: data.annotations_points || []},
                         colors: [initialChartOptions.colors[0], emaLineColor] // Ensure candlestick color is first, then EMA color
                    };
                     if (data.ohlc.length === 0) { // If OHLC is empty, show no data message
                        chartOptionsUpdate.title.text = `${data.pair_name || currentSelectedPairId} - No Data`;
                        chartOptionsUpdate.series = initialChartOptions.series.map(s => ({ ...s, data: [] }));
                        document.getElementById("strategyInfoLabel").textContent = "Status: Data Kosong";
                    }


                    if (activeChart) {
                        activeChart.updateOptions(chartOptionsUpdate);
                    } else {
                        activeChart = new ApexCharts(document.querySelector("#chart"), chartOptionsUpdate);
                        activeChart.render();
                    }
                } else { // data or data.ohlc is missing
                     const noDataOptions = { ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${data.pair_name || currentSelectedPairId} - No Data` },
                        series: initialChartOptions.series.map(s => ({ ...s, data: [] }))
                    };
                    if (activeChart) activeChart.updateOptions(noDataOptions);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), noDataOptions); activeChart.render(); }
                    lastKnownDataTimestamp = data.last_updated_tv || null;
                     document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? `Data (empty) @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "No data";
                    document.getElementById("strategyInfoLabel").textContent = "Status: Data Kosong";
                }
            } catch (error) {
                console.error("Error loading chart data:", error);
                if (activeChart) { activeChart.destroy(); activeChart = null; }
                document.getElementById("chart").innerHTML = `Error loading chart: ${error.message}`;
            } finally {
                isLoadingData = false;
            }
        }

        document.addEventListener("DOMContentLoaded", () => {
            if (!activeChart) { // Initialize with basic options if not already done
                activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions);
                activeChart.render();
            }
            fetchAvailablePairs();
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) {
                    await loadChartDataForCurrentPair();
                }
            }, 15000); // Refresh every 15 seconds
        });
    </script>
</body>
</html>
"""
@flask_app_instance.route('/')
def serve_index_page(): return render_template_string(HTML_CHART_TEMPLATE)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs_flask(): 
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs_info = []
    for pair_id, pair_data in data_manager_view.items():
        cfg = pair_data.get("config", {})
        if cfg.get("enabled", True): active_pairs_info.append({"id": pair_id, "name": cfg.get('pair_name', pair_id)})
    return jsonify(active_pairs_info)

@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend_flask(pair_id_from_request): 
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager: return jsonify({"error": "Pair not found"}), 404
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))
    
    if not pair_data_snapshot: 
        # Return a structure that the frontend expects even for empty data
        default_cfg = get_default_crypto_config()
        pair_name_default = f"{default_cfg['symbol']}-{default_cfg['currency']}" # Fallback name
        return jsonify({
            "ohlc":[], "ema500_series":[], "ema500_line_color": "gray",
            "annotations_yaxis":[], "pair_name": pair_name_default, 
            "last_updated_tv": None, 
            "ema_lookback_label": default_cfg['ema_lookback_period'],
            "strategy_state_info": {"ema500_value":None,"ema_trend_color":"gray"}
        }), 200

    temp_manager = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    # Even if ohlc is empty, return the prepared structure
    # if not prepared_data.get("ohlc"): return jsonify({"error": "No OHLC data to display", **prepared_data}), 200
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask: log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
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

    animated_text_display("=========== EMA 500 TREND DETECTOR START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    key_idx_display = api_key_manager.get_current_key_index()
    key_val_display = api_key_manager.get_current_key()
    key_val_display_short = ("..." + key_val_display[-3:]) if key_val_display and len(key_val_display) > 8 else key_val_display
    log_info(f"Menggunakan API Key Index: {key_idx_display} ({key_val_display_short}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_indicators_init = EMA_LENGTH_FIXED + config.get('ema_lookback_period', 10) + 50 # EMA len + lookback + buffer
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        
        initial_candles = []
        initial_fetch_successful = False
        max_initial_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_key_attempts_done = 0

        while initial_key_attempts_done < max_initial_key_attempts and not initial_fetch_successful:
            current_api_key_init = api_key_manager.get_current_key()
            if not current_api_key_init:
                log_error(f"BIG DATA: Semua API key habis (global) sebelum mencoba fetch untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break 
            
            log_info(f"BIG DATA: Mencoba fetch awal untuk {config['pair_name']} dengan key index {api_key_manager.get_current_key_index()} (Attempt {initial_key_attempts_done + 1}/{max_initial_key_attempts})", pair_name=config['pair_name'])
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key_init, config['timeframe'], pair_name=config['pair_name'])
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
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts +1 
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue 

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])
        
        if initial_candles:
            min_len_for_warmup = EMA_LENGTH_FIXED + config.get('ema_lookback_period', 10) 
            if len(initial_candles) >= min_len_for_warmup:
                log_info(f"Memproses {max(0, len(initial_candles) - (min_len_for_warmup -1) )} candle historis awal untuk inisialisasi state {config['pair_name']}...", pair_name=config['pair_name'])
                # Run strategy logic on historical slices to populate initial state (e.g., previous_trend)
                # No alerts will be sent during warmup due to is_warmup=True
                for i_warmup in range(min_len_for_warmup -1, len(initial_candles) -1): # Process all but the last candle for warmup
                    historical_slice = initial_candles[:i_warmup+1] 
                    if len(historical_slice) < min_len_for_warmup: continue # Ensure enough data for this slice
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict, is_warmup=True
                    )
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Tidak cukup data untuk warm-up ({len(initial_candles)}/{min_len_for_warmup}) untuk {config['pair_name']}", pair_name=config['pair_name'])


        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"] and config.get("enable_email_notifications"):
                send_email_notification(f"Data Downloading Complete: {config['pair_name']}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now analyzing {config['pair_name']}.", {**config, 'pair_name': config['pair_name']})
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({config['pair_name']}) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
    
    try: 
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: 
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600); continue
                    else: data_per_pair["data_fetch_failed_consecutively"] = 0 

                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    required_interval = 60 if config_for_pair.get('timeframe') == "minute" else 300 # Fetch more aggressively during big data, e.g., every 5 min for hour/day
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch); continue
                
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                if data_per_pair["big_data_collection_phase_active"]: animated_text_display(f"\n--- BIG DATA {pair_name_for_log} ({num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD+AnsiColors.MAGENTA)
                else: animated_text_display(f"\n--- LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {num_candles_before_fetch} candles ---", color=AnsiColors.BOLD+AnsiColors.CYAN)

                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 
                if data_per_pair["big_data_collection_phase_active"]:
                    needed_for_big_data = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed_for_big_data <=0 : fetch_update_successful = True 
                    else: limit_fetch_update = min(needed_for_big_data, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0 or (data_per_pair["big_data_collection_phase_active"] and not fetch_update_successful): 
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    original_api_key_index_for_this_update = api_key_manager.get_current_key_index()

                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE: Semua API key habis (global) untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        
                        log_info(f"UPDATE: Mencoba fetch untuk {pair_name_for_log} dengan key index {api_key_manager.get_current_key_index()} (Attempt {update_key_attempts_done + 1}/{max_update_key_attempts})", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_update, config_for_pair['timeframe'], pair_name=pair_name_for_log)
                            fetch_update_successful = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 
                            any_data_fetched_this_cycle = True
                            if api_key_manager.get_current_key_index() != original_api_key_index_for_this_update:
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
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) +1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 

                if not fetch_update_successful or not new_candles_batch:
                    if fetch_update_successful and not new_candles_batch and not data_per_pair["big_data_collection_phase_active"]:
                        log_info(f"Tidak ada data candle baru diterima untuk {pair_name_for_log} (fetch dianggap berhasil tapi batch kosong).", pair_name=pair_name_for_log)
                    elif not fetch_update_successful:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                    with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                    continue

                merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added_count_this_batch, updated_count_this_batch = 0,0
                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                if newly_added_count_this_batch + updated_count_this_batch > 0: log_info(f"{newly_added_count_this_batch + updated_count_this_batch} candle baru/diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)

                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if not data_per_pair["big_data_email_sent"] and config_for_pair.get("enable_email_notifications"):
                            send_email_notification(f"Data Downloading Complete: {pair_name_for_log}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now analyzing {pair_name_for_log}.", {**config_for_pair, 'pair_name': pair_name_for_log})
                            data_per_pair["big_data_email_sent"] = True
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 100: # Trim if significantly over, keep some buffer
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 50):]


                min_len_for_logic_run_live = EMA_LENGTH_FIXED + config_for_pair.get('ema_lookback_period', 10)
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run_live:
                    process_logic_now = (newly_added_count_this_batch + updated_count_this_batch > 0 or
                                         (not data_per_pair["big_data_collection_phase_active"] and 
                                          num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and # e.g. just finished big data
                                          len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) ) 
                    
                    if process_logic_now:
                         log_info(f"Menjalankan logika EMA Trend Detector untuk {pair_name_for_log}...", pair_name=pair_name_for_log)
                         data_per_pair["strategy_state"] = run_strategy_logic(
                             data_per_pair["all_candles_list"], 
                             config_for_pair, 
                             data_per_pair["strategy_state"], 
                             global_settings_dict,
                             is_warmup=False # Live processing
                        )
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 30 
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
            
            if sleep_duration > 0 : show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s ({time.strftime('%H:%M:%S')})...")
            else: time.sleep(1) # Minimal sleep

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan.{AnsiColors.ENDC}",color=AnsiColors.ORANGE)
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EMA TREND DETECTOR STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali ke menu utama...")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    # Periksa apakah flask_thread sudah ada dan berjalan untuk menghindari multiple Flask servers
    # Cara sederhana: cek apakah thread dengan nama spesifik ada di active_count()
    # Ini tidak ideal untuk production, tapi cukup untuk skrip ini.
    is_flask_running = any(t.name == "FlaskServerThread" for t in threading.enumerate())
    
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThread")
        flask_thread.start()
    else:
        log_info("Flask server sudah berjalan di thread lain.", "SYSTEM_CHART")


    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto EMA 500 Trend Detector =========", color=AnsiColors.HEADER) # Ganti judul
        pick_title_main = ""
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs: pick_title_main += f"--- Crypto Aktif ({len(active_cfgs)}) ---\n" + "".join([f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')}, EMA Lookback: {c.get('ema_lookback_period','?')})\n" for i,c in enumerate(active_cfgs)])
        else: pick_title_main += "Tidak ada konfigurasi crypto aktif.\n"
        
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5001\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        _, main_idx = pick(main_opts, pick_title_main, indicator='=>')
        
        if main_idx == 0: 
            settings = load_settings() # Reload settings before starting
            start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif main_idx == 1: settings = settings_menu(settings)
        elif main_idx == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Terima kasih!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan.", color=AnsiColors.ORANGE)
    except Exception as e_global: clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL: {e_global}{AnsiColors.ENDC}"); log_exception("MAIN ERROR:",pair_name="SYS_CRIT"); input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
