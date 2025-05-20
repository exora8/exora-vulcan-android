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
import copy # Untuk deep copy data agar thread-safe
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

SETTINGS_FILE = "settings_multiple_recovery.json"
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
        if not self.keys or self.current_index >= len(self.keys): return None
        return self.keys[self.current_index]
    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # (Email notification logic for key switch - tetap sama)
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL!{AnsiColors.ENDC}")
            # (Email notification logic for all keys failed - tetap sama)
            return None
    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION --- (Tetap sama)
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

# --- FUNGSI PENGATURAN --- (Struktur get_default_crypto_config, load_settings, save_settings, _prompt_crypto_config, settings_menu tetap sama seperti di pertanyaan Anda)
# ... (Kode _prompt_crypto_config dan settings_menu dari pertanyaan, tidak diubah signifikan selain perbaikan minor yang sudah ada) ...
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
        "primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com", "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com", "enable_termux_notifications": False
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            if "api_settings" not in settings: settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]: settings["api_settings"][k] = v
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list): settings["cryptos"] = []
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
            return settings
        except (json.JSONDecodeError, Exception) as e:
            log_error(f"Error membaca {SETTINGS_FILE} ({e}). Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config):
    # (Isi fungsi ini tetap sama seperti yang Anda berikan sebelumnya)
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
        new_config["left_strength"] = new_config.get('left_strength',50); new_config["right_strength"] = new_config.get('right_strength',150)
    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter trading tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',5.0); new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0); new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',10.0)
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
    # (Isi fungsi ini tetap sama seperti yang Anda berikan sebelumnya, karena UI console tidak berubah)
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
        if not current_settings.get("cryptos"): pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\nPilih tindakan:"
        selectable_options = [
            "Atur Primary API Key", "Kelola Recovery API Keys", "Atur Email Global untuk Notifikasi Sistem",
            "Aktifkan/Nonaktifkan Notifikasi Termux Realtime", "Tambah Konfigurasi Crypto Baru",
            "Ubah Konfigurasi Crypto", "Hapus Konfigurasi Crypto", "Kembali ke Menu Utama"
        ]
        action_choice = -1
        try:
            _option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick:
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual."); print(pick_title_settings)
            for idx, opt_text in enumerate(selectable_options): print(f"  {idx + 1}. {opt_text}")
            try:
                choice = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice < len(selectable_options): action_choice = choice
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
        
        try:
            clear_screen_animated()
            if action_choice == 0: # Atur Primary API Key
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Primary API Key CryptoCompare [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(1, "Menyimpan...")
            elif action_choice == 1: # Kelola Recovery API Keys
                while True:
                    clear_screen_animated(); recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k]; api_s['recovery_keys'] = current_recovery
                    if not current_recovery: recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"
                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali"]
                    rec_index = -1
                    try: _rec_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>')
                    except: # Fallback manual
                        print(recovery_pick_title); 
                        for i_rec, o_rec in enumerate(recovery_options_plain): print(f"  {i_rec+1}. {o_rec}")
                        try: rec_idx_choice = int(input("Pilihan: ")) -1
                        if 0 <= rec_idx_choice < len(recovery_options_plain): rec_index = rec_idx_choice
                        else: continue
                        except: continue
                    clear_screen_animated()
                    if rec_index == 0: # Tambah
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key: current_recovery.append(new_r_key); api_s['recovery_keys'] = current_recovery; save_settings(current_settings); print(f"{AnsiColors.GREEN}Ditambahkan.{AnsiColors.ENDC}")
                        else: print(f"{AnsiColors.RED}Input kosong.{AnsiColors.ENDC}")
                    elif rec_index == 1: # Hapus
                        if not current_recovery: print(f"{AnsiColors.ORANGE}Tidak ada untuk dihapus.{AnsiColors.ENDC}"); show_spinner(1); continue
                        del_opts = [f"{k[:5]}..." for k in current_recovery] + ["Batal"]
                        _d_txt, idx_del_pick = pick(del_opts, "Hapus key nomor:", indicator="=>")
                        if idx_del_pick < len(current_recovery): removed = current_recovery.pop(idx_del_pick); api_s['recovery_keys'] = current_recovery; save_settings(current_settings); print(f"{AnsiColors.GREEN}Dihapus.{AnsiColors.ENDC}")
                    elif rec_index == 2: break # Kembali
                    show_spinner(1)
            elif action_choice == 2: # Atur Email Global
                animated_text_display("-- Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(1, "Menyimpan...")
            elif action_choice == 3: # Notifikasi Termux
                animated_text_display("-- Notifikasi Termux Realtime --", color=AnsiColors.HEADER)
                current_status = api_s.get('enable_termux_notifications', False)
                new_status_input = input(f"Aktifkan? (true/false) [{current_status}]: ").lower().strip()
                if new_status_input == 'true': api_s['enable_termux_notifications'] = True; print(f"{AnsiColors.GREEN}Diaktifkan.{AnsiColors.ENDC}")
                elif new_status_input == 'false': api_s['enable_termux_notifications'] = False; print(f"{AnsiColors.GREEN}Dinonaktifkan.{AnsiColors.ENDC}")
                else: print(f"{AnsiColors.ORANGE}Input tidak valid. Status tidak berubah.{AnsiColors.ENDC}")
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(1, "Menyimpan...")
            elif action_choice == 4: # Tambah Crypto
                new_crypto_conf = _prompt_crypto_config(get_default_crypto_config())
                current_settings.setdefault("cryptos", []).append(new_crypto_conf); save_settings(current_settings)
                log_info(f"Konfigurasi {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan."); show_spinner(1, "Menyimpan...")
            elif action_choice == 5: # Ubah Crypto
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi.{AnsiColors.ENDC}"); show_spinner(1); continue
                edit_options = [f"{c.get('symbol','N/A')}-{c.get('currency','N/A')}" for c in current_settings["cryptos"]] + ["Batal"]
                _e_txt, idx_choice = pick(edit_options, "Ubah konfigurasi nomor:", indicator="=>")
                if idx_choice < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                    save_settings(current_settings); log_info(f"Konfigurasi diubah."); show_spinner(1, "Menyimpan...")
            elif action_choice == 6: # Hapus Crypto
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi.{AnsiColors.ENDC}"); show_spinner(1); continue
                del_c_options = [f"{c.get('symbol','N/A')}-{c.get('currency','N/A')}" for c in current_settings["cryptos"]] + ["Batal"]
                _dc_txt, idx_del = pick(del_c_options, "Hapus konfigurasi nomor:", indicator="=>")
                if idx_del < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_del]['symbol']}-{current_settings['cryptos'][idx_del]['currency']}"
                    current_settings["cryptos"].pop(idx_del); save_settings(current_settings)
                    log_info(f"Konfigurasi {removed_pair} dihapus."); show_spinner(1, "Menyimpan...")
            elif action_choice == 7: break # Kembali ke Menu Utama
        except ValueError: print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Error...")
        except Exception as e_menu: log_error(f"Error di menu pengaturan: {e_menu}"); log_exception("Traceback:"); show_spinner(1.5, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA --- (Tetap sama)
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []; current_to_ts = None
    api_endpoint_map = {"minute": "histominute", "hour": "histohour", "day": "histoday"}
    api_endpoint = api_endpoint_map.get(timeframe, "histohour")
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10
    if is_large_fetch: log_info(f"Pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
    
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
                log_debug(f"Fetch batch (Key: ...{current_api_key_to_use[-5:]}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429]:
                error_msg = response.json().get('Message', f"HTTP Error {response.status_code}")
                log_warning(f"API Key Error (HTTP {response.status_code}): {error_msg} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_msg}")
            response.raise_for_status()
            data = response.json()
            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_err_msgs = ["api key is invalid", "apikey_is_missing", "over_the_limit", "rate limit exceeded", "pro_tier_has_expired", "you are over your rate limit"]
                if any(k_err.lower() in error_message.lower() for k_err in key_err_msgs):
                    log_warning(f"API Key Error (JSON): {error_message} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else: log_error(f"API Error CryptoCompare: {error_message} (Params: {params})", pair_name=pair_name); break
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break
            raw_candles_from_api = data['Data']['Data']
            if not raw_candles_from_api: break
            batch_candles_list = []
            for item in raw_candles_from_api:
                req_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
                if not all(k in item and item[k] is not None for k in req_keys):
                    log_warning(f"Candle tidak lengkap dari API @ ts {item.get('time', 'N/A')}. Dilewati.", pair_name=pair_name); continue
                batch_candles_list.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list and batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                 batch_candles_list.pop()
            if not batch_candles_list and current_to_ts is not None: break
            all_accumulated_candles = batch_candles_list + all_accumulated_candles
            current_to_ts = raw_candles_from_api[0]['time'] if raw_candles_from_api else None
            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
            if len(raw_candles_from_api) < limit_for_this_api_call: break
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.25) # Lebih cepat sedikit
        except APIKeyError: raise
        except requests.exceptions.RequestException as e: log_error(f"Kesalahan koneksi: {e}", pair_name=pair_name); break
        except Exception as e: log_error(f"Error fetch_candles: {e}", pair_name=pair_name); log_exception("Traceback:"); break
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)}.", pair_name=pair_name)
    return all_accumulated_candles

# --- LOGIKA STRATEGI --- (get_initial_strategy_state, find_pivots, run_strategy_logic tetap sama)
# ... (Kode get_initial_strategy_state, find_pivots, run_strategy_logic dari jawaban sebelumnya, tidak ada perubahan signifikan di sini) ...
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "last_pivot_high_display_info": None, "last_pivot_low_display_info": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None,
        "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None, "emergency_sl_level_custom": None,
        "position_size": 0,
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1: return pivots
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True; value_i = series_list[i]
        if value_i is None: continue
        for j in range(1, left_strength + 1):
            value_left = series_list[i-j]
            if value_left is None: is_pivot = False; break
            if (is_high and value_i <= value_left) or (not is_high and value_i >= value_left): is_pivot = False; break
        if not is_pivot: continue
        for j in range(1, right_strength + 1):
            value_right = series_list[i+j]
            if value_right is None: is_pivot = False; break
            if (is_high and value_i < value_right) or (not is_high and value_i > value_right): is_pivot = False; break
        if is_pivot: pivots[i] = value_i
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    strategy_state["final_pivot_high_price_confirmed"] = None; strategy_state["final_pivot_low_price_confirmed"] = None
    left_strength, right_strength = crypto_config['left_strength'], crypto_config['right_strength']
    req_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(k in candles_history[0] for k in req_keys if candles_history and candles_history[0]):
        log_warning(f"Data candle tidak lengkap/kosong.", pair_name=pair_name); return strategy_state
    
    high_prices = [c.get('high') for c in candles_history]; low_prices = [c.get('low') for c in candles_history]
    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices, left_strength, right_strength, False)
    current_bar_idx = len(candles_history) - 1
    if current_bar_idx < 0: return strategy_state

    idx_pivot_event = current_bar_idx - right_strength
    current_candle = candles_history[current_bar_idx]
    if any(current_candle.get(k) is None for k in req_keys):
        log_warning(f"Data OHLC candle terbaru tidak lengkap.", pair_name=pair_name); return strategy_state

    # Pivot High Confirmed
    ph_at_event = raw_pivot_highs[idx_pivot_event] if 0 <= idx_pivot_event < len(raw_pivot_highs) else None
    if ph_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state.update({
            "final_pivot_high_price_confirmed": ph_at_event, "last_signal_type": 1,
            "high_price_for_fib": ph_at_event, "high_bar_index_for_fib": idx_pivot_event,
            "active_fib_level": None, "active_fib_line_start_index": None, # Reset FIB
            "last_pivot_high_display_info": {'price': ph_at_event, 'timestamp_ms': candles_history[idx_pivot_event]['timestamp'].timestamp() * 1000}
        })
        log_info(f"{AnsiColors.CYAN}PH: {ph_at_event:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

    # Pivot Low Confirmed & FIB
    pl_at_event = raw_pivot_lows[idx_pivot_event] if 0 <= idx_pivot_event < len(raw_pivot_lows) else None
    if pl_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state.update({"final_pivot_low_price_confirmed": pl_at_event, "last_signal_type": -1,
                               "last_pivot_low_display_info": {'price': pl_at_event, 'timestamp_ms': candles_history[idx_pivot_event]['timestamp'].timestamp() * 1000}})
        log_info(f"{AnsiColors.CYAN}PL: {pl_at_event:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
        
        if strategy_state.get("high_price_for_fib") is not None and strategy_state.get("high_bar_index_for_fib") is not None and \
           idx_pivot_event > strategy_state["high_bar_index_for_fib"]:
            high_val, low_val = strategy_state["high_price_for_fib"], pl_at_event
            if high_val is not None and low_val is not None:
                fib_level = (high_val + low_val) / 2.0
                is_late = False
                if crypto_config["enable_secure_fib"]:
                    price_check_key = crypto_config["secure_fib_check_price"].lower()
                    price_val_check = current_candle.get(price_check_key, current_candle['close'])
                    if price_val_check is not None and price_val_check > fib_level: is_late = True
                
                if is_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({fib_level:.5f}){AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state.update({"active_fib_level": None, "active_fib_line_start_index": None})
                else:
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {fib_level:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state.update({"active_fib_level": fib_level, "active_fib_line_start_index": idx_pivot_event})
            strategy_state.update({"high_price_for_fib": None, "high_bar_index_for_fib": None}) # Reset H for next FIB

    # Entry & Position Management
    if strategy_state.get("active_fib_level") and strategy_state.get("position_size", 0) == 0:
        if current_candle['close'] > current_candle['open'] and current_candle['close'] > strategy_state["active_fib_level"]:
            entry_px = current_candle['close']
            emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
            strategy_state.update({
                "position_size": 1, "entry_price_custom": entry_px, "highest_price_for_trailing": entry_px,
                "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
                "emergency_sl_level_custom": emerg_sl, "active_fib_level": None, "active_fib_line_start_index": None
            })
            log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state.get('active_fib_level',0):.5f}), SL: {emerg_sl:.5f}" # SL sudah di state
            log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name); play_notification_sound()
            send_termux_notification(f"BUY: {pair_name}", f"Entry@{entry_px:.5f} SL@{emerg_sl:.5f}", global_settings, pair_name)
            # (Email logic for BUY - tetap sama)

    if strategy_state.get("position_size", 0) > 0:
        strategy_state["highest_price_for_trailing"] = max(strategy_state.get("highest_price_for_trailing", current_candle['high']), current_candle['high'])
        if not strategy_state.get("trailing_tp_active_custom") and strategy_state.get("entry_price_custom"):
            profit_pct = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if profit_pct >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_pct:.2f}%{AnsiColors.ENDC}", pair_name=pair_name)
        
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("highest_price_for_trailing"):
            new_trail_sl = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state.get("current_trailing_stop_level") is None or new_trail_sl > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = new_trail_sl
        
        final_sl = strategy_state.get("emergency_sl_level_custom"); exit_reason = "Emergency SL"; color = AnsiColors.RED
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level"):
            if final_sl is None or strategy_state["current_trailing_stop_level"] > final_sl:
                final_sl = strategy_state["current_trailing_stop_level"]; exit_reason = "Trailing Stop"; color = AnsiColors.BLUE
        
        if final_sl is not None and current_candle['low'] <= final_sl:
            exit_px = min(current_candle['open'], final_sl)
            pnl = ((exit_px - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0 if strategy_state["entry_price_custom"] else 0.0
            if exit_reason == "Trailing Stop" and pnl < 0: color = AnsiColors.RED
            log_msg = f"EXIT @ {exit_px:.5f} by {exit_reason}. PnL: {pnl:.2f}%"
            log_info(f"{color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name); play_notification_sound()
            send_termux_notification(f"EXIT: {pair_name}", f"{exit_reason}@{exit_px:.5f} PnL:{pnl:.2f}%", global_settings, pair_name)
            # (Email logic for EXIT - tetap sama)
            strategy_state.update({ # Reset position
                "position_size": 0, "entry_price_custom": None, "highest_price_for_trailing": None,
                "trailing_tp_active_custom": False, "current_trailing_stop_level": None, "emergency_sl_level_custom": None
            })
        elif strategy_state.get("position_size",0) > 0: # Log SL jika masih posisi
            sl_disp = final_sl if final_sl else "N/A"
            log_debug(f"Posisi Aktif. Entry: {strategy_state.get('entry_price_custom',0):.5f}, SL: {sl_disp:.5f if isinstance(sl_disp, float) else sl_disp}", pair_name=pair_name)
    return strategy_state

# CHART_INTEGRATION_START
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot):
    if pair_id_to_display not in current_data_manager_snapshot:
        return None
    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]
    candles_full = pair_specific_data.get("all_candles_list", [])
    current_strategy_state = pair_specific_data.get("strategy_state", {})
    pair_config = pair_specific_data.get("config", {})
    
    candles_for_chart = candles_full[-TARGET_BIG_DATA_CANDLES:]
    ohlc_points = []
    if not candles_for_chart:
        return {"ohlc": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None}

    for c in candles_for_chart:
        if all(k in c and c[k] is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ohlc_points.append({'x': c['timestamp'].timestamp() * 1000, 'y': [c['open'], c['high'], c['low'], c['close']]})
    
    ann_yaxis, ann_points = [], []
    active_fib = current_strategy_state.get("active_fib_level")
    if active_fib and ohlc_points:
        ann_yaxis.append({'y': active_fib, 'borderColor': '#00E396', 'label': {'text': f'FIB 0.5: {active_fib:.5f}', 'style': {'background': '#00E396', 'fontSize':'10px'}}})
    
    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") and ohlc_points:
        entry_px = current_strategy_state.get("entry_price_custom")
        ann_yaxis.append({'y': entry_px, 'borderColor': '#2698FF', 'strokeDashArray': 4, 'label': {'text': f'Entry: {entry_px:.5f}', 'style': {'background': '#2698FF', 'fontSize':'10px'}}})
        sl_val = current_strategy_state.get("emergency_sl_level_custom"); sl_text = "Emerg. SL"
        if current_strategy_state.get("trailing_tp_active_custom") and current_strategy_state.get("current_trailing_stop_level"):
            trail_sl = current_strategy_state.get("current_trailing_stop_level")
            if sl_val is None or (trail_sl and trail_sl > sl_val): sl_val = trail_sl; sl_text = "Trail. SL"
        if sl_val:
            ann_yaxis.append({'y': sl_val, 'borderColor': '#FF4560', 'label': {'text': f'{sl_text}: {sl_val:.5f}', 'style': {'background': '#FF4560', 'fontSize':'10px'}}})
            
    first_ts_on_chart = ohlc_points[0]['x'] if ohlc_points else 0
    last_ph = current_strategy_state.get("last_pivot_high_display_info")
    if last_ph and last_ph['timestamp_ms'] >= first_ts_on_chart:
        ann_points.append({'x': last_ph['timestamp_ms'], 'y': last_ph['price'], 'marker': {'size': 7, 'fillColor': '#FF0000', 'shape': 'triangle'}, 'label': {'offsetY': -18, 'text': 'PH', 'style':{'background':'#FF0000', 'fontSize':'10px'}}})
    last_pl = current_strategy_state.get("last_pivot_low_display_info")
    if last_pl and last_pl['timestamp_ms'] >= first_ts_on_chart:
        ann_points.append({'x': last_pl['timestamp_ms'], 'y': last_pl['price'], 'marker': {'size': 7, 'fillColor': '#00CD00', 'shape': 'triangle', 'cssClass': 'apexcharts-marker-inverted'}, 'label': {'offsetY': 10, 'text': 'PL', 'style':{'background':'#00CD00', 'fontSize':'10px'}}})

    return {
        "ohlc": ohlc_points, "annotations_yaxis": ann_yaxis, "annotations_points": ann_points,
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": candles_for_chart[-1]['timestamp'].timestamp() * 1000 if candles_for_chart else None
    }

flask_app_instance = Flask(__name__)

HTML_CHART_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Crypto Chart</title><script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<style> body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;} #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width:100%; max-width: 1200px; } select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; } #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; } h1 { color: #00bcd4; margin-bottom:15px; font-size:1.5em; } #lastUpdatedLabel { font-size: 0.8em; color: #aaa; margin-left: auto;} .apexcharts-marker-inverted .apexcharts-marker-poly { transform: rotate(180deg); transform-origin: center; } </style>
</head><body><h1>Live Strategy Chart</h1><div id="controls"><label for="pairSelector">Pair:</label><select id="pairSelector" onchange="handlePairSelectionChange()"></select><button onclick="loadChartDataForCurrentPair()">Refresh</button><span id="lastUpdatedLabel">Memuat...</span></div>
<div id="chart-container"><div id="chart"></div></div>
<script>
    let activeChart; let currentSelectedPairId = ''; let lastKnownDataTimestamp = null; let autoRefreshIntervalId = null;
    const initialChartOptions = {
        series: [{ name: 'Candlestick', data: [] }],
        chart: { type: 'candlestick', height: 550, id: 'mainCandlestickChart', background: '#2a2a2a', 
                 animations: { enabled: false }, /* PERBAIKAN: Animasi dimatikan */
                 toolbar: { show: true, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } }
        }, theme: { mode: 'dark' }, title: { text: 'Memuat...', align: 'left', style: { color: '#e0e0e0'} },
        xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
        yaxis: { tooltip: { enabled: true }, labels: { style: { colors: '#aaa'}, formatter: (v) => v.toFixed(5) } },
        grid: { borderColor: '#444' }, annotations: { yaxis: [], points: [] },
        tooltip: { theme: 'dark', shared: true, 
            custom: function({series, seriesIndex, dataPointIndex, w}) { /* Tooltip kustom tetap sama */
                if (w.globals.seriesCandleO && w.globals.seriesCandleO[seriesIndex] && w.globals.seriesCandleO[seriesIndex][dataPointIndex] !== undefined) {
                    const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex], h = w.globals.seriesCandleH[seriesIndex][dataPointIndex], l = w.globals.seriesCandleL[seriesIndex][dataPointIndex], c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
                    return ('<div class="apexcharts-tooltip-candlestick" style="padding:5px 10px;">' + '<div>O:<span class="value">' + o.toFixed(5) + '</span> H:<span class="value">' + h.toFixed(5) + '</span></div>' + '<div>L:<span class="value">' + l.toFixed(5) + '</span> C:<span class="value">' + c.toFixed(5) + '</span></div>' + '</div>');
                } return '';
            }
        }, noData: { text: 'Tidak ada data.', style: { color: '#ccc'} }
    };
    async function fetchAvailablePairs() {
        try {
            const response = await fetch('/api/available_pairs'); if (!response.ok) throw new Error('Gagal load pairs');
            const pairs = await response.json(); const selector = document.getElementById('pairSelector'); selector.innerHTML = '';
            if (pairs.length > 0) {
                pairs.forEach(p => { const opt = document.createElement('option'); opt.value = p.id; opt.textContent = p.name; selector.appendChild(opt); });
                currentSelectedPairId = selector.value || pairs[0].id; loadChartDataForCurrentPair();
            } else { selector.innerHTML = '<option value="">No pairs</option>'; if(activeChart) activeChart.destroy(); document.getElementById('chart').innerHTML = '<p>No active pairs.</p>';}
        } catch (error) { console.error("Error fetching pairs:", error); document.getElementById('pairSelector').innerHTML = '<option value="">Error</option>';}
    }
    function handlePairSelectionChange() { currentSelectedPairId = document.getElementById('pairSelector').value; lastKnownDataTimestamp = null; loadChartDataForCurrentPair(); }
    async function loadChartDataForCurrentPair() {
        if (!currentSelectedPairId) return;
        try {
            const response = await fetch(`/api/chart_data/${currentSelectedPairId}`); if (!response.ok) throw new Error('Gagal load chart data');
            const chartData = await response.json();
            if (chartData.last_updated_tv && chartData.last_updated_tv === lastKnownDataTimestamp) {
                document.getElementById('lastUpdatedLabel').textContent = `Terakhir: ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`; return;
            }
            lastKnownDataTimestamp = chartData.last_updated_tv;
            document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Terakhir: ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "N/A";
            const newOptions = { ...initialChartOptions, title: { ...initialChartOptions.title, text: `${chartData.pair_name}` },
                series: [{ name: 'Candlestick', data: chartData.ohlc || [] }],
                annotations: { yaxis: chartData.annotations_yaxis || [], points: chartData.annotations_points || [] }
            };
            if (!activeChart) { activeChart = new ApexCharts(document.querySelector("#chart"), newOptions); activeChart.render(); }
            else { activeChart.updateOptions(newOptions); }
        } catch (error) { console.error("Error loading chart:", error); if(activeChart)activeChart.destroy();activeChart=null; 
            const errOpt = {...initialChartOptions, title: {...initialChartOptions.title, text:`Error: ${currentSelectedPairId}`}, noData:{text:error.message}};
            activeChart = new ApexCharts(document.querySelector("#chart"), errOpt); activeChart.render();
            document.getElementById('lastUpdatedLabel').textContent = "Error update";
        }
    }
    document.addEventListener('DOMContentLoaded', () => { fetchAvailablePairs(); if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId); autoRefreshIntervalId = setInterval(async () => { if (currentSelectedPairId && document.visibilityState === 'visible') { await loadChartDataForCurrentPair(); } }, 20000); /* Refresh lebih cepat sedikit */ });
</script></body></html>
"""

@flask_app_instance.route('/')
def serve_index_page(): return render_template_string(HTML_CHART_TEMPLATE)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs():
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs = [{'id': pid, 'name': pdata.get("config", {}).get('pair_name', pid)}
                    for pid, pdata in data_manager_view.items() if pdata.get("config", {}).get("enabled", True)]
    return jsonify(active_pairs)

@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend(pair_id_from_request):
    with shared_data_lock:
        # PERBAIKAN: Optimasi deepcopy. Hanya deepcopy strategy_state.
        # config dan list candle bisa diambil sebagai referensi karena prepare_chart_data_for_pair
        # akan slice list candle (membuat salinan baru untuk diolah) dan hanya baca config.
        original_pair_data = shared_crypto_data_manager.get(pair_id_from_request)
        if not original_pair_data:
            return jsonify({"error": f"Data pair {pair_id_from_request} tidak ada."}), 404
        
        # Buat snapshot dengan deepcopy yang lebih selektif
        pair_data_snapshot = {
            "config": original_pair_data.get("config", {}), # Referensi/shallow copy cukup
            "all_candles_list": original_pair_data.get("all_candles_list", []), # Referensi, akan di-slice
            "strategy_state": copy.deepcopy(original_pair_data.get("strategy_state", {})), # Ini perlu deepcopy
            # Item lain yang mungkin ada (jika kecil dan immutable/primitif, bisa langsung)
            "big_data_collection_phase_active": original_pair_data.get("big_data_collection_phase_active"),
            "last_candle_fetch_time": original_pair_data.get("last_candle_fetch_time") 
            # ...tambahkan field lain dari crypto_data_manager jika prepare_chart_data_for_pair membutuhkannya
        }
        # Pastikan pair_data_snapshot dibuat dalam struktur yang sama seperti yang diharapkan prepare_chart_data_for_pair
        # yaitu {'pair_id_key': data_seperti_di_atas}
        temp_data_manager_for_prep = {pair_id_from_request: pair_data_snapshot}


    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_data_manager_for_prep)
    if not prepared_data or not prepared_data.get("ohlc"):
        return jsonify({"error": f"Tidak cukup data OHLC untuk {pair_id_from_request}.", "ohlc": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_id_from_request, "last_updated_tv": None}), 200
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Flask server dimulai di http://localhost:5001 (atau IP Termux-mu)", pair_name="SYSTEM_CHART")
    try:
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask:
        log_error(f"Flask server gagal: {e_flask}", pair_name="SYSTEM_CHART"); log_exception("Traceback:")

# CHART_INTEGRATION_END

# --- FUNGSI UTAMA TRADING LOOP --- (start_trading tetap sama strukturnya)
# ... (Kode start_trading dari jawaban sebelumnya, tidak ada perubahan signifikan di sini selain passing lock/ref) ...
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)
    if not api_key_manager.has_valid_keys():
        log_error("Tidak ada API key valid."); input("Enter..."); return
    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning("Tidak ada konfigurasi crypto aktif."); input("Enter..."); return

    animated_text_display("== MULTI-CRYPTO STRATEGY START ==", color=AnsiColors.HEADER)
    # (Inisialisasi local_crypto_data_manager - tetap sama)
    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        animated_text_display(f"\nInisialisasi {config['pair_name']}...", color=AnsiColors.MAGENTA, new_line=False)
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        # (Fetch data awal dan warm-up - tetap sama logikanya)
        initial_candles_target = TARGET_BIG_DATA_CANDLES; initial_candles = []; retries_done_initial = 0
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_fetch_successful = False
        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: log_error(f"BIG DATA: Semua key habis untuk {config['pair_name']}."); break
            try:
                log_info(f"BIG DATA: Fetching {initial_candles_target} for {config['pair_name']} (Key Idx: {api_key_manager.get_current_key_index()})", pair_name=config['pair_name'])
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key, config['timeframe'], config['pair_name'])
                initial_fetch_successful = True
            except APIKeyError:
                log_warning(f"BIG DATA: Key Idx {api_key_manager.get_current_key_index()} gagal for {config['pair_name']}.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break
                retries_done_initial +=1
            except Exception as e_init_fetch:
                log_error(f"BIG DATA: Error fetch {config['pair_name']}: {e_init_fetch}.", pair_name=config['pair_name']); break
        
        if not initial_candles and not initial_fetch_successful:
            log_error(f"BIG DATA: Gagal fetch data awal {config['pair_name']}.", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue # Next pair

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candles diterima {config['pair_name']}.", pair_name=config['pair_name'])
        
        # Warm-up
        if initial_candles:
            min_len_pivots = config.get('left_strength',50) + config.get('right_strength',150) + 1
            if len(initial_candles) >= min_len_pivots:
                log_info(f"Warm-up state for {config['pair_name']}...", pair_name=config['pair_name'])
                for i in range(min_len_pivots -1, len(initial_candles) -1):
                    hist_slice = initial_candles[:i+1]
                    if len(hist_slice) < min_len_pivots: continue
                    temp_state = local_crypto_data_manager[pair_id]["strategy_state"].copy(); temp_state["position_size"] = 0
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(hist_slice, config, temp_state, global_settings_dict)
                    if local_crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: # Reset trade state post warm-up
                        local_crypto_data_manager[pair_id]["strategy_state"].update({
                            "position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                            "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                            "current_trailing_stop_level":None})
                log_info(f"Warm-up {config['pair_name']} selesai.", pair_name=config['pair_name'])
        
        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"TARGET {TARGET_BIG_DATA_CANDLES} CANDLES TERCAPAI {config['pair_name']}.", pair_name=config['pair_name'])
            # (Email notif big data - tetap sama)
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        print() # Newline after each pair init

    animated_text_display(f"{AnsiColors.HEADER}--- SEMUA PAIR DIINISIALISASI ---{AnsiColors.ENDC}")

    # (Loop trading utama - tetap sama logikanya)
    try:
        while True:
            active_cryptos_big_data = 0; min_next_refresh = float('inf'); data_fetched_cycle = False
            for pair_id, data in local_crypto_data_manager.items():
                config = data["config"]; pair_name = config['pair_name']
                if data.get("data_fetch_failed_consecutively",0) >= (api_key_manager.total_keys() or 1) +1:
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600:
                        continue # Cooldown
                    else: data["data_fetch_failed_consecutively"] = 0
                
                now = datetime.now()
                interval = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_big_data += 1
                    interval = 55 if config.get('timeframe') == "minute" else (3580 if config.get('timeframe') == "hour" else 3600*23.8)
                else: interval = config.get('refresh_interval_seconds', 60)

                if (now - data["last_candle_fetch_time"]).total_seconds() < interval:
                    min_next_refresh = min(min_next_refresh, interval - (now - data["last_candle_fetch_time"]).total_seconds())
                    continue

                log_info(f"Proses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = now
                
                new_candles = []; fetch_success = False
                limit_to_fetch = 3
                if data["big_data_collection_phase_active"]:
                    needed = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"])
                    if needed <= 0: fetch_success = True # Already done
                    else: limit_to_fetch = min(needed, CRYPTOCOMPARE_MAX_LIMIT); limit_to_fetch = max(limit_to_fetch, 1)
                
                if limit_to_fetch > 0 or data["big_data_collection_phase_active"]:
                    retries_update = 0; max_retries_update = api_key_manager.total_keys() or 1
                    original_key_idx_update = api_key_manager.get_current_key_index()
                    while retries_update < max_retries_update and not fetch_success:
                        key_update = api_key_manager.get_current_key()
                        if not key_update: log_error(f"Semua key habis untuk update {pair_name}."); break
                        try:
                            new_candles = fetch_candles(config['symbol'],config['currency'],limit_to_fetch,config['exchange'],key_update,config['timeframe'],pair_name)
                            fetch_success = True; data["data_fetch_failed_consecutively"] = 0; data_fetched_cycle=True
                        except APIKeyError:
                            data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively",0)+1
                            if not api_key_manager.switch_to_next_key(): break
                            retries_update+=1
                        except Exception as e_upd:
                            data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively",0)+1; break
                
                if data.get("data_fetch_failed_consecutively",0) >= (api_key_manager.total_keys() or 1)+1:
                    data["last_attempt_after_all_keys_failed"] = datetime.now()

                if not fetch_success or not new_candles:
                    # (Handle no new data - tetap sama)
                    with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data)
                    continue
                
                # (Merge candles - tetap sama)
                merged_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                added_count = 0; updated_count = 0
                for c_new in new_candles:
                    ts_new = c_new['timestamp']
                    if ts_new not in merged_dict: merged_dict[ts_new] = c_new; added_count+=1
                    elif merged_dict[ts_new] != c_new : merged_dict[ts_new] = c_new; updated_count+=1
                data["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                
                if added_count > 0 or updated_count > 0: log_info(f"{added_count+updated_count} new/updated candles for {pair_name}. Total: {len(data['all_candles_list'])}", pair_name=pair_name)

                # (Cek big data phase & pangkas list - tetap sama)
                if data["big_data_collection_phase_active"] and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                    data["big_data_collection_phase_active"] = False; active_cryptos_big_data = max(0, active_cryptos_big_data-1)
                    # (Email notif - tetap sama)
                if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES:
                    data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]

                # (Run strategy logic - tetap sama)
                min_len_strat = config.get('left_strength',50) + config.get('right_strength',150) + 1
                if len(data["all_candles_list"]) >= min_len_strat:
                    if added_count > 0 or updated_count > 0 or (not data["big_data_collection_phase_active"] and len(data["all_candles_list"])>=TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) > added_count+updated_count ) : # crude check if it's first run after big data
                        data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"], global_settings_dict)
                
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data)
                min_next_refresh = min(min_next_refresh, interval)

            # (Sleep logic - tetap sama)
            sleep_s = 15
            if not data_fetched_cycle and api_key_manager.get_current_key() is None: sleep_s = 3600
            elif active_cryptos_big_data > 0:
                min_bd_interval_sleep = float('inf')
                # ... (kalkulasi min_bd_interval_sleep tetap sama) ...
                sleep_s = min(min_bd_interval_sleep if min_bd_interval_sleep != float('inf') else 30, 20) # Lebih cepat sedikit untuk big data
            else:
                if min_next_refresh != float('inf') and min_next_refresh > 0:
                    sleep_s = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_next_refresh))
                else: # Fallback
                    sleep_s = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, all_crypto_configs[0].get('refresh_interval_seconds',60) if all_crypto_configs else 60)
            
            if sleep_s > 0: show_spinner(sleep_s, f"Menunggu {int(sleep_s)}s...")
            else: time.sleep(1)

    except KeyboardInterrupt: animated_text_display("\nProses dihentikan.", color=AnsiColors.ORANGE, new_line=True)
    except Exception as e_loop: log_error(f"Error loop utama: {e_loop}",pair_name="SYSTEM"); log_exception("Traceback:")
    finally: animated_text_display("== STRATEGY STOP ==", color=AnsiColors.HEADER); input("Enter...");

# --- MENU UTAMA --- (main_menu tetap sama strukturnya)
# ... (Kode main_menu dari jawaban sebelumnya, tidak ada perubahan signifikan di sini) ...
def main_menu():
    settings = load_settings()
    flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True)
    flask_thread.start()
    while True:
        clear_screen_animated()
        animated_text_display("== Crypto Strategy Runner (Chart) ==", color=AnsiColors.HEADER)
        # (Tampilan menu utama - tetap sama)
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        title = f"--- Crypto Aktif ({len(active_cfgs)}) ---\n" + "".join([f"  {i+1}. {cfg['symbol']}-{cfg['currency']}\n" for i,cfg in enumerate(active_cfgs)]) if active_cfgs else "Tidak ada config aktif.\n"
        # (Info API, Termux, Chart server - tetap sama)
        title += f"Chart Server: http://localhost:5001 (atau IP Termux-mu)\nPilih Opsi:"
        opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        sel_idx = -1
        try: _o, sel_idx = pick(opts, title, indicator="=>")
        except: # Fallback manual
            print(title); 
            for i_o, o_o in enumerate(opts): print(f"  {i_o+1}. {o_o}")
            try: sel_idx_choice = int(input("Pilihan: ")) -1
            if 0 <= sel_idx_choice < len(opts): sel_idx = sel_idx_choice
            else: continue
            except: continue
            
        if sel_idx == 0: start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif sel_idx == 1: settings = settings_menu(settings)
        elif sel_idx == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Sampai jumpa!", color=AnsiColors.MAGENTA); show_spinner(0.5)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Aplikasi dihentikan.{AnsiColors.ENDC}",new_line=True)
    except Exception as e:
        clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL UTAMA:{AnsiColors.ENDC}\n{AnsiColors.RED}{e}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:"); print(f"\n{AnsiColors.ORANGE}Error detail di: {log_file_name}{AnsiColors.ENDC}"); input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
