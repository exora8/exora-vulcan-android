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
import subprocess
import math
import threading
import copy

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    print("Flask tidak terinstal. Silakan install dengan: pip install Flask")
    sys.exit(1)

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
    LOW_LIQ_BG_CONSOLE = '\033[44m'
    HIGH_LIQ_BG_CONSOLE = '\033[47m\033[30m'
    AGG_DROP_ALERT_BG = '\033[41m\033[97m' # Latar belakang merah, teks putih untuk alert drop

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
logger = logging.getLogger("ExoraAggregator") # Nama logger baru
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

log_file_name = "trading_log_exora_aggregator.txt" # Nama file log baru
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


SETTINGS_FILE = "settings_exora_aggregator.json" # Nama file settings BARU
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 200
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
        if not self.keys: return None
        return self.keys[self.current_index] if self.current_index < len(self.keys) else None

    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                subject = "Exora Aggregator: API Key Switched"
                body = f"Bot beralih ke API key berikutnya (Index {self.current_index}). Key lama mungkin bermasalah."
                send_email_notification(subject, body, self.global_email_settings, pair_name_ctx_override="GLOBAL_SYSTEM")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                subject = "Exora Aggregator: SEMUA API KEY GAGAL"
                body = "Semua API key (primary dan recovery) telah gagal atau habis. Bot tidak dapat mengambil data."
                send_email_notification(subject, body, self.global_email_settings, pair_name_ctx_override="GLOBAL_SYSTEM")
            return None

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

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

def send_email_notification(subject, body_text, settings_for_email, pair_name_ctx_override=None):
    if not settings_for_email.get("enable_email_notifications", False): return
    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")
    
    pair_name_ctx = pair_name_ctx_override or settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))

    if not all([sender_email, sender_password, receiver_email]):
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=pair_name_ctx)
        return
    msg = MIMEText(body_text)
    if "Exora Aggregator" not in subject:
        subject = f"Exora Aggregator: {subject}"

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
    if not api_settings.get("enable_termux_notifications", False): return
    
    if "Exora Aggregator" not in title and pair_name_for_log != "SYSTEM_CHART" :
        title = f"Exora Aggregator: {title}"

    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg, '--priority', 'max', '--sound', '-id', f'exora_agg_{pair_name_for_log.replace("/", "_")}_{str(uuid.uuid4())[:4]}'],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- HELPER TIMEFRAME ---
def timeframe_to_seconds(tf_str):
    tf_lower = tf_str.lower()
    num_part = int("".join(filter(str.isdigit, tf_lower))) if any(char.isdigit() for char in tf_lower) else 1
    if 'm' in tf_lower: return num_part * 60
    if 'h' in tf_lower: return num_part * 3600
    if 'd' in tf_lower: return num_part * 86400
    if 'w' in tf_lower: return num_part * 604800
    return 3600 # Default 1 jam jika tidak dikenali

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config(): # Ini untuk setiap pair individual
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "1m",
        "refresh_interval_seconds": 60,
        "enable_email_notifications": False, # Notifikasi email per pair
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
        # Hapus pengaturan detektor likuiditas dari config individual jika tidak digunakan lagi di sini
    }

def get_default_aggregation_settings():
    return {
        "enabled_aggregation_tracker": False,
        "crypto_ids_for_aggregation": [], # list of "id" from cryptos section
        "aggregation_timeframe": "1h", # e.g., "15m", "1h", "4h"
        "lookback_bars_drop_agg": 5,
        "drop_percentage_threshold_agg": 3.0,
        "alert_cooldown_seconds_agg": 300, # 5 menit cooldown untuk alert yang sama
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
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
                # Hapus key lama yang tidak relevan dengan agregator
                keys_to_remove_old_liq = [
                    "enable_lowliq_detector", "lowliq_pattern_length", "enable_highliq_detector",
                    "highliq_pattern_length", "highliq_lookback_confirmation_bars",
                    "alert_on_liquidity_state_change", "alert_on_high_liquidity_confirmed"
                ]
                for old_key in keys_to_remove_old_liq:
                    if old_key in crypto_cfg: del crypto_cfg[old_key]

            if "aggregation_settings" not in settings:
                settings["aggregation_settings"] = get_default_aggregation_settings()
            else:
                default_agg_template = get_default_aggregation_settings()
                for key, default_value in default_agg_template.items():
                    if key not in settings["aggregation_settings"]:
                        settings["aggregation_settings"][key] = default_value
            return settings
        except Exception as e:
            log_error(f"Error membaca {SETTINGS_FILE}: {e}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), 
                    "cryptos": [get_default_crypto_config()], 
                    "aggregation_settings": get_default_aggregation_settings()}
    return {"api_settings": default_api_settings.copy(), 
            "cryptos": [get_default_crypto_config()],
            "aggregation_settings": get_default_aggregation_settings()}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

def _prompt_type(prompt_text, current_value, target_type, default_value, min_val=None, max_val=None, step=None, valid_options=None):
    while True:
        val_str = input(f"{AnsiColors.BLUE}{prompt_text} [{current_value}]: {AnsiColors.ENDC}").strip()
        if not val_str: return current_value
        try:
            if target_type == bool:
                if val_str.lower() in ['true', 't', 'y', 'yes', '1']: return True
                if val_str.lower() in ['false', 'f', 'n', 'no', '0']: return False
                raise ValueError("Input boolean tidak valid (true/false)")
            
            typed_val = target_type(val_str)
            
            if valid_options and typed_val not in valid_options:
                print(f"{AnsiColors.RED}Pilihan tidak valid. Pilihan yang tersedia: {valid_options}{AnsiColors.ENDC}")
                continue
            if min_val is not None and typed_val < min_val:
                print(f"{AnsiColors.RED}Nilai harus >= {min_val}.{AnsiColors.ENDC}"); continue
            if max_val is not None and typed_val > max_val:
                print(f"{AnsiColors.RED}Nilai harus <= {max_val}.{AnsiColors.ENDC}"); continue
            return typed_val
        except ValueError as e_val:
            print(f"{AnsiColors.RED}Input tidak valid: {e_val}. Harap masukkan tipe {target_type.__name__}.{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): # Konfigurasi untuk pair individual
    clear_screen_animated()
    new_config = current_config.copy()
    def_cfg = get_default_crypto_config()

    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol',def_cfg['symbol'])}-{new_config.get('currency',def_cfg['currency'])}) ---", color=AnsiColors.HEADER)
    new_config["enabled"] = _prompt_type("Aktifkan pair ini?", new_config.get('enabled', def_cfg['enabled']), bool, def_cfg['enabled'])
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar [{new_config.get('symbol',def_cfg['symbol'])}]: {AnsiColors.ENDC}") or new_config.get('symbol',def_cfg['symbol'])).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote [{new_config.get('currency',def_cfg['currency'])}]: {AnsiColors.ENDC}") or new_config.get('currency',def_cfg['currency'])).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange [{new_config.get('exchange',def_cfg['exchange'])}]: {AnsiColors.ENDC}") or new_config.get('exchange',def_cfg['exchange'])).strip()
    
    valid_tf_keys = ["1m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "1d", "3d", "1w"]
    new_config["timeframe"] = _prompt_type(f"Timeframe ({', '.join(valid_tf_keys)})", new_config.get('timeframe',def_cfg['timeframe']), str, def_cfg['timeframe'], valid_options=valid_tf_keys).lower().strip()

    new_config["refresh_interval_seconds"] = _prompt_type("Interval Refresh (detik)", new_config.get('refresh_interval_seconds',def_cfg['refresh_interval_seconds']), int, def_cfg['refresh_interval_seconds'], min_val=MIN_REFRESH_INTERVAL_AFTER_BIG_DATA)
    
    # Pengaturan email untuk pair ini (opsional, jika diperlukan notif spesifik pair)
    # animated_text_display("\n-- Notifikasi Email Pair --", color=AnsiColors.HEADER)
    # new_config["enable_email_notifications"] = _prompt_type("Aktifkan Notifikasi Email untuk pair ini?", new_config.get('enable_email_notifications',def_cfg['enable_email_notifications']), bool, def_cfg['enable_email_notifications'])
    # if new_config["enable_email_notifications"]:
    #     new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim Pair [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    #     new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Pair [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    #     new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima Pair [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()

    return new_config

def _prompt_aggregation_settings(current_agg_settings, all_crypto_configs):
    clear_screen_animated()
    new_settings = current_agg_settings.copy()
    def_agg_cfg = get_default_aggregation_settings()

    animated_text_display("--- Konfigurasi Pelacak Agregasi & Sinyal Penurunan ---", color=AnsiColors.HEADER)
    new_settings["enabled_aggregation_tracker"] = _prompt_type(
        "Aktifkan Pelacak Agregasi?", 
        new_settings.get('enabled_aggregation_tracker', def_agg_cfg['enabled_aggregation_tracker']), 
        bool, def_agg_cfg['enabled_aggregation_tracker']
    )

    if new_settings["enabled_aggregation_tracker"]:
        animated_text_display("\n-- Pilihan Kripto untuk Agregasi --", color=AnsiColors.HEADER)
        if not all_crypto_configs:
            print(f"{AnsiColors.ORANGE}Tidak ada crypto pair yang dikonfigurasi. Tambahkan dulu.{AnsiColors.ENDC}")
            new_settings["crypto_ids_for_aggregation"] = []
        else:
            current_selected_ids = new_settings.get('crypto_ids_for_aggregation', [])
            available_cryptos_options = []
            for cfg in all_crypto_configs:
                if cfg.get("enabled", True): # Hanya tawarkan yang aktif
                    pair_display = f"{cfg['symbol']}-{cfg['currency']} ({cfg['timeframe']})"
                    available_cryptos_options.append({"name": pair_display, "id": cfg['id']})
            
            if not available_cryptos_options:
                print(f"{AnsiColors.ORANGE}Tidak ada crypto pair yang AKTIF untuk dipilih.{AnsiColors.ENDC}")
                new_settings["crypto_ids_for_aggregation"] = []
            else:
                selected_for_agg = []
                print(f"{AnsiColors.CYAN}Pilih crypto untuk dimasukkan ke dalam agregasi (spasi untuk pilih, enter untuk selesai):{AnsiColors.ENDC}")
                
                # Buat daftar opsi untuk pick, tandai yang sudah terpilih
                pick_options = []
                default_selected_indices = []
                for i, opt_data in enumerate(available_cryptos_options):
                    is_currently_selected = opt_data['id'] in current_selected_ids
                    pick_options.append(f"[{'x' if is_currently_selected else ' '}] {opt_data['name']}")
                    if is_currently_selected:
                        default_selected_indices.append(i)
                
                try:
                    # Pick library untuk multiple selection
                    # Ini memerlukan versi pick yang mendukung multi-select atau penyesuaian
                    # Untuk kesederhanaan, kita akan meminta pengguna input Y/N per crypto
                    temp_selected_ids = []
                    for crypto_opt_data in available_cryptos_options:
                        is_selected = crypto_opt_data['id'] in current_selected_ids
                        user_choice = _prompt_type(f"Sertakan {crypto_opt_data['name']} dalam agregasi?", is_selected, bool, is_selected)
                        if user_choice:
                            temp_selected_ids.append(crypto_opt_data['id'])
                    new_settings["crypto_ids_for_aggregation"] = temp_selected_ids

                except Exception as e_pick_multi:
                    log_warning(f"Gagal menggunakan pick untuk multi-seleksi crypto agregasi: {e_pick_multi}. Gunakan pilihan manual.")
                    # Fallback jika pick gagal atau tidak mendukung multi-select
                    # (Logika ini bisa dibuat lebih canggih jika diperlukan)
                    print(f"{AnsiColors.ORANGE}Input ID kripto yang akan diagregasi, pisahkan dengan koma (contoh: id1,id2):{AnsiColors.ENDC}")
                    for i, crypto_opt in enumerate(available_cryptos_options):
                        print(f"  {i+1}. {crypto_opt['name']} (ID: {crypto_opt['id']})")
                    ids_str = input(f"ID terpilih saat ini: {','.join(current_selected_ids)}\nMasukkan ID baru: ").strip()
                    if ids_str:
                        new_settings["crypto_ids_for_aggregation"] = [s.strip() for s in ids_str.split(',')]


        valid_tf_keys_agg = ["1m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
        new_settings["aggregation_timeframe"] = _prompt_type(
            f"Timeframe Bar Agregasi ({', '.join(valid_tf_keys_agg)})",
            new_settings.get('aggregation_timeframe', def_agg_cfg['aggregation_timeframe']),
            str, def_agg_cfg['aggregation_timeframe'], valid_options=valid_tf_keys_agg
        ).lower().strip()

        animated_text_display("\n-- Pengaturan Sinyal Penurunan Agregasi --", color=AnsiColors.HEADER)
        new_settings["lookback_bars_drop_agg"] = _prompt_type(
            "Periode Cek Penurunan Agregasi (bar)",
            new_settings.get('lookback_bars_drop_agg', def_agg_cfg['lookback_bars_drop_agg']),
            int, def_agg_cfg['lookback_bars_drop_agg'], min_val=1
        )
        new_settings["drop_percentage_threshold_agg"] = _prompt_type(
            "Ambang Batas Penurunan Agregasi (%)",
            new_settings.get('drop_percentage_threshold_agg', def_agg_cfg['drop_percentage_threshold_agg']),
            float, def_agg_cfg['drop_percentage_threshold_agg'], min_val=0.1, step=0.1
        )
        new_settings["alert_cooldown_seconds_agg"] = _prompt_type(
            "Cooldown Alert Penurunan Agregasi (detik)",
            new_settings.get('alert_cooldown_seconds_agg', def_agg_cfg['alert_cooldown_seconds_agg']),
            int, def_agg_cfg['alert_cooldown_seconds_agg'], min_val=0
        )
    return new_settings


def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        agg_s = current_settings.get("aggregation_settings", {})
        
        pkd = api_s.get('primary_key', 'N/A'); pkd = pkd[:5]+"..."+pkd[-3:] if len(pkd)>8 and pkd not in ["YOUR_PRIMARY_KEY", "N/A"] else pkd
        nrk = len([k for k in api_s.get('recovery_keys', []) if k])
        tns = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        agg_status_display = "Aktif" if agg_s.get("enabled_aggregation_tracker", False) else "Nonaktif"
        
        title = f"--- Menu Pengaturan Exora Aggregator ---\n"
        title += f"API Key: {pkd} | Recovery: {nrk} | Termux Global: {tns}\n"
        title += f"Pelacak Agregasi: {AnsiColors.BOLD}{agg_status_display}{AnsiColors.ENDC}\n"
        if agg_s.get("enabled_aggregation_tracker", False):
            title += f"  Timeframe Agregasi: {agg_s.get('aggregation_timeframe', 'N/A')}\n"
            title += f"  Drop Check: {agg_s.get('drop_percentage_threshold_agg', 0)}% over {agg_s.get('lookback_bars_drop_agg', 0)} bars\n"
            
            # Tampilkan kripto yang dipilih untuk agregasi
            selected_ids_for_agg_menu = agg_s.get('crypto_ids_for_aggregation', [])
            if selected_ids_for_agg_menu:
                selected_names_for_agg_menu = []
                for crypto_cfg_menu in current_settings.get("cryptos", []):
                    if crypto_cfg_menu.get('id') in selected_ids_for_agg_menu:
                        selected_names_for_agg_menu.append(f"{crypto_cfg_menu.get('symbol')}-{crypto_cfg_menu.get('currency')}")
                title += f"  Kripto Agregasi: {', '.join(selected_names_for_agg_menu) or 'Belum ada'}\n"
            else:
                title += "  Kripto Agregasi: Belum ada yang dipilih\n"


        title += "\n--- Crypto Pairs Individu ---\n"
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]): title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({cfg.get('timeframe','?')}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'} (ID: {cfg.get('id', 'N/A')[:8]}..)\n"
        
        title += "----------------------\nPilih tindakan:"
        opts = [
            "Pengaturan API & Notifikasi Global", 
            "Pengaturan Pelacak Agregasi",
            "Tambah Crypto Pair Individu", 
            "Ubah Crypto Pair Individu", 
            "Hapus Crypto Pair Individu", 
            "Kembali ke Menu Utama"
        ]
        
        options_for_pick = [opt[:70] + ('...' if len(opt) > 70 else '') for opt in opts]
        try:
            _, action_idx = pick(options_for_pick, title, indicator='=>')
        except Exception as e_pick:
            log_warning(f"Pick library error: {e_pick}. Gunakan input angka.")
            print(title)
            for i, opt_disp in enumerate(options_for_pick): print(f"{i}. {opt_disp}")
            try:
                action_idx = int(input("Masukkan nomor pilihan: "))
                if not (0 <= action_idx < len(options_for_pick)): raise ValueError("Diluar range")
            except ValueError:
                print("Input tidak valid."); time.sleep(1); continue
        
        clear_screen_animated()
        try:
            if action_idx == 0: # Pengaturan API & Notifikasi Global
                animated_text_display("--- Pengaturan API & Notifikasi Global ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Primary API Key CryptoCompare [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key','YOUR_PRIMARY_KEY'))
                
                # Kelola Recovery Keys (Sama seperti sebelumnya)
                while True:
                    clear_screen_animated()
                    current_recovery = api_s.get('recovery_keys', [])
                    rec_title = "--- Kelola Recovery API Keys ---\n"
                    if not current_recovery: rec_title += "  (Tidak ada recovery key)\n"
                    else:
                        for i_rec, key_rec in enumerate(current_recovery):
                             rec_title += f"  {i_rec+1}. {key_rec[:5]}...{key_rec[-3:] if len(key_rec)>8 else key_rec}\n"
                    rec_title += "----------------------------\nPilih:"
                    rec_opts = ["Tambah Recovery Key", "Hapus Recovery Key", "Selesai Kelola Recovery Key"]
                    _, rec_action_idx = pick(rec_opts, rec_title, indicator='=>')
                    
                    if rec_action_idx == 0: # Tambah
                        new_rec_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_rec_key: api_s.setdefault('recovery_keys',[]).append(new_rec_key); animated_text_display("Ditambahkan.", color=AnsiColors.GREEN)
                        else: animated_text_display("Kosong, tidak ditambah.", color=AnsiColors.ORANGE)
                    elif rec_action_idx == 1: # Hapus
                        if not api_s.get('recovery_keys'): animated_text_display("Tidak ada untuk dihapus.", color=AnsiColors.ORANGE); time.sleep(1); continue
                        del_rec_opts = [f"{k_rec[:5]}...{k_rec[-3:] if len(k_rec)>8 else k_rec}" for k_rec in api_s['recovery_keys']] + ["Batal"]
                        _, del_rec_key_idx = pick(del_rec_opts, "Pilih recovery key untuk dihapus:", indicator='=>')
                        if del_rec_key_idx < len(api_s['recovery_keys']): api_s['recovery_keys'].pop(del_rec_key_idx); animated_text_display("Dihapus.", color=AnsiColors.GREEN)
                    elif rec_action_idx == 2: break 
                    show_spinner(0.5, "Memproses...")

                api_s['enable_global_email_notifications_for_key_switch'] = _prompt_type("Aktifkan Email Notif Sistem Global (API Key Switch, etc)?", api_s.get('enable_global_email_notifications_for_key_switch',False), bool, False)
                if api_s['enable_global_email_notifications_for_key_switch']:
                    api_s['email_sender_address'] = (input(f"Alamat Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                    api_s['email_sender_app_password'] = (input(f"App Password Email Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                    api_s['email_receiver_address_admin'] = (input(f"Alamat Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                
                api_s['enable_termux_notifications'] = _prompt_type("Aktifkan Notifikasi Termux Global?", api_s.get('enable_termux_notifications',False), bool, False)
                current_settings["api_settings"] = api_s

            elif action_idx == 1: # Pengaturan Pelacak Agregasi
                current_settings["aggregation_settings"] = _prompt_aggregation_settings(
                    current_settings.get("aggregation_settings", get_default_aggregation_settings()),
                    current_settings.get("cryptos", [])
                )
            elif action_idx == 2: # Tambah Crypto Pair Individu
                current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_idx == 3: # Ubah Crypto Pair Individu
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk diubah."); show_spinner(1,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe', '?')}) (ID: {c.get('id','N/A')[:8]}..)" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:")
                if edit_c_idx < len(current_settings["cryptos"]): 
                    current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 4: # Hapus Crypto Pair Individu
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk dihapus."); show_spinner(1,""); continue
                del_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe', '?')}) (ID: {c.get('id','N/A')[:8]}..)" for c in current_settings["cryptos"]] + ["Batal"]
                _, del_c_idx = pick(del_opts, "Pilih pair untuk dihapus:")
                if del_c_idx < len(current_settings["cryptos"]): 
                    # Juga hapus dari daftar agregasi jika ada
                    deleted_id = current_settings["cryptos"][del_c_idx].get("id")
                    current_settings["cryptos"].pop(del_c_idx)
                    if deleted_id and current_settings.get("aggregation_settings", {}).get("crypto_ids_for_aggregation"):
                        if deleted_id in current_settings["aggregation_settings"]["crypto_ids_for_aggregation"]:
                            current_settings["aggregation_settings"]["crypto_ids_for_aggregation"].remove(deleted_id)
                            log_info(f"Pair dengan ID {deleted_id} juga dihapus dari daftar agregasi.", "SETTINGS")
            elif action_idx == 5: break # Kembali
            
            save_settings(current_settings)
            if action_idx != 5 : show_spinner(1, "Disimpan...") # Tidak perlu spinner jika "Kembali"
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA (fetch_candles) ---
# (Sama seperti di skrip Anda, tidak perlu diubah signifikan untuk logika ini)
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe_str="1h", pair_name="N/A"):
    timeframe_details = {"endpoint_segment": "histohour", "aggregate": 1}
    tf_lower = timeframe_str.lower()

    try:
        num_part = int("".join(filter(str.isdigit, tf_lower))) if any(char.isdigit() for char in tf_lower) else 1
        if 'm' in tf_lower:
            timeframe_details["endpoint_segment"] = "histominute"; timeframe_details["aggregate"] = num_part
        elif 'h' in tf_lower:
            timeframe_details["endpoint_segment"] = "histohour"; timeframe_details["aggregate"] = num_part
        elif 'd' in tf_lower:
            timeframe_details["endpoint_segment"] = "histoday"; timeframe_details["aggregate"] = num_part
        elif 'w' in tf_lower:
            timeframe_details["endpoint_segment"] = "histoday"; timeframe_details["aggregate"] = 7 * num_part
        else:
            log_warning(f"Timeframe '{timeframe_str}' tidak dikenali, menggunakan 1 hour.", pair_name=pair_name)
            timeframe_details["endpoint_segment"] = "histohour"; timeframe_details["aggregate"] = 1
    except ValueError:
        log_warning(f"Error parsing timeframe '{timeframe_str}', menggunakan 1 hour.", pair_name=pair_name)
        timeframe_details["endpoint_segment"] = "histohour"; timeframe_details["aggregate"] = 1

    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = timeframe_details["endpoint_segment"]
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    
    is_large_fetch = total_limit_desired > 20
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')

    retries_network_error = 3; current_network_retry = 0

    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if timeframe_details["aggregate"] > 1: params["aggregate"] = timeframe_details["aggregate"]
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts: params["toTs"] = current_to_ts
        
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429, 400]:
                err_msg_detail = response.json().get('Message', f"HTTP Error {response.status_code}") if response.content else f"HTTP Error {response.status_code}, no content"
                key_display_for_log = current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else "KEY_SHORT"
                log_warning(f"API Key/Req Error (HTTP {response.status_code}): {err_msg_detail} | Key ...{key_display_for_log}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {err_msg_detail}")
            response.raise_for_status() 
            data = response.json()

            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error')
                key_err_patterns = ["api key is invalid", "apikey_is_missing", "rate limit", "monthly_calls", "tier", "not valid"]
                if any(p.lower() in err_msg.lower() for p in key_err_patterns):
                    key_display_for_log = current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else "KEY_SHORT"
                    log_warning(f"API Key Error (JSON): {err_msg} | Key ...{key_display_for_log}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {err_msg}")
                else: log_error(f"API Error: {err_msg}", pair_name=pair_name); break 
            
            raw_candles = data.get('Data', {}).get('Data', [])
            if not raw_candles:
                if len(all_accumulated_candles) > 0 : log_debug(f"Tidak ada candle baru. Total: {len(all_accumulated_candles)}", pair_name=pair_name)
                else: log_warning(f"Tidak ada data candle dari API untuk {pair_name}.", pair_name=pair_name)
                break

            batch = [{'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']}
                     for item in raw_candles if all(k in item and item[k] is not None for k in ['time', 'open', 'high', 'low', 'close', 'volumefrom'])]
            
            if current_to_ts and all_accumulated_candles and batch and batch[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']: batch.pop() 
            if not batch and current_to_ts: break 

            all_accumulated_candles = batch + all_accumulated_candles
            if raw_candles: current_to_ts = raw_candles[0]['time']
            else: break

            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
            if len(raw_candles) < limit_call or len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.2)
        
        except APIKeyError: raise
        except requests.exceptions.RequestException as e_req:
            current_network_retry += 1
            if current_network_retry <= retries_network_error:
                log_warning(f"Network Error: {e_req}. Retry {current_network_retry}/{retries_network_error}...", pair_name=pair_name)
                time.sleep(current_network_retry * 2)
                continue
            else: log_error(f"Network Error ({e_req}) after {retries_network_error} retries. Gagal.", pair_name=pair_name); break 
        except Exception as e_gen: log_exception(f"Error lain fetch_candles: {e_gen}", pair_name=pair_name); break
    
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai')
    return all_accumulated_candles

# CHART_INTEGRATION_START & Flask Endpoints
# (Sama seperti di skrip Anda, hanya mengganti nama state jika perlu dan judul)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()
shared_aggregation_data = { # Data baru untuk chart agregasi jika diinginkan
    "aggregated_series_for_chart": [], # Misal: {'timestamp': ..., 'value': ...}
    "last_significant_drop_info_for_chart": None
}

def prepare_chart_data_for_pair(pair_id, snapshot): # Untuk chart individual pair
    if pair_id not in snapshot: return None
    data = snapshot[pair_id]
    chart_display_candles = 500 
    hist = data.get("all_candles_list", [])[-chart_display_candles:] 
    cfg = data.get("config", {})
    # Jika ada state analisis individual, bisa ditambahkan di sini
    
    ohlc_data = []
    pair_display_name = cfg.get('pair_name', pair_id)

    if not hist:
        return { "ohlc": [], "pair_name": pair_display_name, "last_updated_tv": None, "config_info": cfg }

    for c in hist:
        if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = c['timestamp'].timestamp() * 1000
            ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})
    
    return {
        "ohlc": ohlc_data, "pair_name": pair_display_name,
        "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
        "config_info": cfg
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE_AGGREGATOR = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exora Aggregator - Charts</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        .chart-section { width: 100%; max-width: 1200px; margin-bottom: 20px; }
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width: 100%; max-width: 1200px; flex-wrap: wrap; }
        select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor: pointer; }
        .chart-container { width: 100%; background-color: #2a2a2a; padding: 15px; border-radius: 8px; }
        h1, h2 { color: #00bcd4; margin-bottom: 10px; text-align: center; }
        h1 { font-size: 1.8em; } h2 { font-size: 1.3em; }
        #lastUpdatedLabel, #aggStatusLabel { font-size: .8em; color: #aaa; margin-left: auto; }
        #aggStatusLabel { color: #ffd700; }
    </style>
</head>
<body>
    <h1>Exora Aggregator - Monitoring</h1>
    
    <div class="chart-section">
        <h2>Aggregated Index Tracker</h2>
        <div id="controls-agg">
             <button onclick="loadAggregatedChartData()">Refresh Aggregated</button>
             <span id="aggStatusLabel">Agg Status: -</span>
        </div>
        <div class="chart-container"><div id="chart-aggregated"></div></div>
    </div>

    <div class="chart-section">
        <h2>Individual Crypto Pair</h2>
        <div id="controls">
            <label for="pairSelector">Pair:</label>
            <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
            <button onclick="loadChartDataForCurrentPair()">Refresh Pair</button>
            <span id="lastUpdatedLabel">Memuat...</span>
        </div>
        <div class="chart-container"><div id="chart-individual"></div></div>
    </div>

    <script>
        let activeIndividualChart, activeAggregatedChart, currentSelectedPairId = "", lastKnownIndividualDataTimestamp = null;
        let lastKnownAggDataTimestamp = null, autoRefreshIntervalId = null, isLoadingIndividual = false, isLoadingAgg = false;

        const commonChartOptions = {
            chart: { background: "transparent", animations: { enabled: false }, toolbar: { show: true } },
            theme: { mode: "dark" },
            xaxis: { type: "datetime", labels: { style: { colors: "#aaa" } }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: "#aaa" }, formatter: v => v ? v.toFixed(5) : "" } },
            stroke: { width: 1, curve: "straight" }, markers: { size: 0 },
            grid: { borderColor: "#444" },
            tooltip: { theme: "dark", shared: true, intersect: false, y: { formatter: val => val ? val.toFixed(5) : val } },
            noData: { text: "Tidak ada data.", align: "center", style: { color: "#ccc" } }
        };

        const initialIndividualChartOptions = { ...commonChartOptions,
            series: [{ name: "Candlestick", type: "candlestick", data: [] }],
            chart: { ...commonChartOptions.chart, type: "candlestick", height: 450 },
            title: { text: "Memuat Data Pair...", align: "left", style: { color: "#e0e0e0" } },
        };
        const initialAggregatedChartOptions = { ...commonChartOptions,
             series: [{ name: "Aggregated Value", type: "line", data: [] }],
             chart: { ...commonChartOptions.chart, type: "line", height: 350 },
             title: { text: "Aggregated Index Value", align: "left", style: { color: "#e0e0e0" } },
             yaxis: { ...commonChartOptions.yaxis, title: { text: "Avg. Price Index" }, labels:{ formatter: v => v ? v.toFixed(2) : "" } },
             colors: ["#FF4560"], // Red color for the aggregated line
        };

        async function fetchAvailablePairs() {
            try {
                const response = await fetch("/api/available_pairs");
                if (!response.ok) throw new Error(\`HTTP \${response.status}\`);
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
                    if (activeIndividualChart) { activeIndividualChart.destroy(); activeIndividualChart = null; }
                    document.getElementById("chart-individual").innerHTML = "No pairs configured.";
                }
            } catch (error) {
                console.error("Error fetching available pairs:", error);
                if (activeIndividualChart) { activeIndividualChart.destroy(); activeIndividualChart = null; }
                document.getElementById("chart-individual").innerHTML = \`Error loading pairs: \${error.message}\`;
            }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById("pairSelector").value;
            lastKnownIndividualDataTimestamp = null; 
            loadChartDataForCurrentPair();
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId || isLoadingIndividual) return;
            isLoadingIndividual = true;
            document.getElementById("lastUpdatedLabel").textContent = \`Loading \${currentSelectedPairId}...\`;
            try {
                const response = await fetch(\`/api/chart_data/\${currentSelectedPairId}\`);
                if (!response.ok) throw new Error(\`HTTP \${response.status}\`);
                const data = await response.json();

                if (data && data.ohlc) {
                    if (data.last_updated_tv && data.last_updated_tv === lastKnownIndividualDataTimestamp && !data.force_update_chart) {
                        isLoadingIndividual = false; return;
                    }
                    lastKnownIndividualDataTimestamp = data.last_updated_tv;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownIndividualDataTimestamp ? \`Last @ \${new Date(lastKnownIndividualDataTimestamp).toLocaleTimeString()}\` : "Data Loaded";
                    
                    const chartOptionsUpdate = {
                        title: { ...initialIndividualChartOptions.title, text: \`\${data.pair_name} (\${data.config_info.timeframe})\` },
                        series: [{ name: "Candlestick", type: "candlestick", data: data.ohlc || [] }],
                    };
                    
                    if (activeIndividualChart) activeIndividualChart.updateOptions(chartOptionsUpdate);
                    else { activeIndividualChart = new ApexCharts(document.querySelector("#chart-individual"), initialIndividualChartOptions); activeIndividualChart.render(); activeIndividualChart.updateOptions(chartOptionsUpdate); }
                } else {
                    const noDataOptions = { ...initialIndividualChartOptions,
                        title: { ...initialIndividualChartOptions.title, text: \`\${data.pair_name || currentSelectedPairId} - No Data\` },
                        series: initialIndividualChartOptions.series.map(s => ({ ...s, data: [] }))
                    };
                    if (activeIndividualChart) activeIndividualChart.updateOptions(noDataOptions);
                    else { activeIndividualChart = new ApexCharts(document.querySelector("#chart-individual"), noDataOptions); activeIndividualChart.render(); }
                    lastKnownIndividualDataTimestamp = data.last_updated_tv || null;
                    document.getElementById("lastUpdatedLabel").textContent = "No data";
                }
            } catch (error) {
                console.error("Error loading individual chart:", error);
                if (activeIndividualChart) { activeIndividualChart.destroy(); activeIndividualChart = null; }
                document.getElementById("chart-individual").innerHTML = \`Error: \${error.message}\`;
            } finally { isLoadingIndividual = false; }
        }

        async function loadAggregatedChartData() {
            if (isLoadingAgg) return;
            isLoadingAgg = true;
            document.getElementById("aggStatusLabel").textContent = "Loading Aggregated Data...";
            try {
                const response = await fetch("/api/aggregated_chart_data");
                if (!response.ok) throw new Error(\`HTTP \${response.status}\`);
                const data = await response.json();

                if (data && data.aggregated_series_for_chart) {
                    const seriesData = data.aggregated_series_for_chart.map(d => ({ x: d.timestamp, y: d.value }));
                    const newTimestamp = data.aggregated_series_for_chart.length > 0 ? data.aggregated_series_for_chart[data.aggregated_series_for_chart.length - 1].timestamp : null;

                    if (newTimestamp && newTimestamp === lastKnownAggDataTimestamp && !data.force_update_chart) {
                        isLoadingAgg = false; return;
                    }
                    lastKnownAggDataTimestamp = newTimestamp;
                    document.getElementById("aggStatusLabel").textContent = \`Agg. Last @ \${newTimestamp ? new Date(newTimestamp).toLocaleTimeString() : 'N/A'}\`;

                    const aggChartOptionsUpdate = {
                         series: [{ name: "Aggregated Value", data: seriesData }],
                         title: { text: \`Aggregated Index (TF: \${data.aggregation_timeframe || 'N/A'})\`},
                         annotations: {} // Clear previous annotations
                    };

                    if (data.last_significant_drop_info_for_chart) {
                        const dropInfo = data.last_significant_drop_info_for_chart;
                        aggChartOptionsUpdate.annotations = {
                            points: [{
                                x: dropInfo.timestamp,
                                y: dropInfo.value_at_drop,
                                marker: { size: 8, fillColor: '#FFFF00', strokeColor: '#FF0000', radius: 2, cssClass: 'apexcharts-custom- १६' },
                                label: { borderColor: '#FF4560', offsetY: 0, style: { color: '#fff', background: '#FF4560'}, text: \`Drop \${dropInfo.percentage.toFixed(1)}%\`}
                            }]
                        };
                    }

                    if (activeAggregatedChart) activeAggregatedChart.updateOptions(aggChartOptionsUpdate);
                    else { activeAggregatedChart = new ApexCharts(document.querySelector("#chart-aggregated"), initialAggregatedChartOptions); activeAggregatedChart.render(); activeAggregatedChart.updateOptions(aggChartOptionsUpdate); }
                } else {
                    if (activeAggregatedChart) activeAggregatedChart.updateOptions({ series: [{data:[]}], title: {text: "Aggregated Index - No Data"} });
                    else { activeAggregatedChart = new ApexCharts(document.querySelector("#chart-aggregated"), {...initialAggregatedChartOptions, series: [{data:[]}], title: {text: "Aggregated Index - No Data"}}); activeAggregatedChart.render(); }
                    document.getElementById("aggStatusLabel").textContent = "Agg. Status: No Data";
                }

            } catch (error) {
                console.error("Error loading aggregated chart:", error);
                 if (activeAggregatedChart) { activeAggregatedChart.destroy(); activeAggregatedChart = null; }
                document.getElementById("chart-aggregated").innerHTML = \`Error: \${error.message}\`;
            } finally { isLoadingAgg = false; }
        }

        document.addEventListener("DOMContentLoaded", () => {
            // Initialize charts with no data message
            if (!activeIndividualChart) { activeIndividualChart = new ApexCharts(document.querySelector("#chart-individual"), initialIndividualChartOptions); activeIndividualChart.render(); }
            if (!activeAggregatedChart) { activeAggregatedChart = new ApexCharts(document.querySelector("#chart-aggregated"), initialAggregatedChartOptions); activeAggregatedChart.render(); }
            
            fetchAvailablePairs(); // Load individual pairs for selector
            loadAggregatedChartData(); // Load aggregated data on page load

            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = setInterval(async () => {
                if (document.visibilityState === 'visible') {
                    if (currentSelectedPairId && !isLoadingIndividual) await loadChartDataForCurrentPair();
                    if (!isLoadingAgg) await loadAggregatedChartData();
                }
            }, 7000); // Refresh every 7 seconds
         });
    </script>
</body>
</html>
"""
@flask_app_instance.route('/')
def serve_index_page_agg(): return render_template_string(HTML_CHART_TEMPLATE_AGGREGATOR)

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
        default_cfg_chart = get_default_crypto_config()
        pair_name_default = f"{default_cfg_chart['symbol']}-{default_cfg_chart['currency']}"
        return jsonify({"ohlc":[], "pair_name": pair_name_default, "last_updated_tv": None, "config_info": default_cfg_chart }), 200

    temp_manager = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    return jsonify(prepared_data)

@flask_app_instance.route('/api/aggregated_chart_data')
def get_aggregated_chart_data_flask():
    with shared_data_lock:
        # Kirim subset data agregasi untuk chart, misal 200 poin terakhir
        chart_series = shared_aggregation_data.get("aggregated_series_for_chart", [])[-500:]
        # Ubah timestamp ke ms untuk ApexCharts
        chart_series_ms = [{'timestamp': s['timestamp'].timestamp() * 1000, 'value': s['value']} for s in chart_series]
        
        drop_info_for_chart = shared_aggregation_data.get("last_significant_drop_info_for_chart")
        if drop_info_for_chart:
            drop_info_for_chart['timestamp'] = drop_info_for_chart['timestamp'].timestamp() * 1000 # juga ke ms

        settings_from_main_thread = load_settings() # Untuk mendapatkan timeframe agregasi
        agg_tf_display = settings_from_main_thread.get("aggregation_settings", {}).get("aggregation_timeframe", "N/A")

    return jsonify({
        "aggregated_series_for_chart": chart_series_ms,
        "last_significant_drop_info_for_chart": drop_info_for_chart,
        "aggregation_timeframe": agg_tf_display
    })


def run_flask_server_thread():
    port = 5002 # Ganti port jika 5001 sudah dipakai
    log_info(f"Memulai Flask server Exora Aggregator di http://localhost:{port}", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e_flask: log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
# CHART_INTEGRATION_END


# --- FUNGSI UTAMA ANALYSIS LOOP ---
def start_analysis_loop(global_settings_dict, shared_dm_ref, lock_ref, shared_agg_data_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    agg_settings = global_settings_dict.get("aggregation_settings", get_default_aggregation_settings())
    
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key yang valid. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE); input(); return

    all_crypto_configs_individual = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs_individual:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto individual yang aktif.{AnsiColors.ENDC}")

    # --- Inisialisasi untuk Pelacak Agregasi ---
    aggregated_close_series = [] # List of {'timestamp': ..., 'value': ...}
    last_aggregation_calc_time = datetime.min
    last_drop_alert_time_agg = datetime.min
    is_significant_drop_previously_agg = False
    
    # Validasi ID kripto untuk agregasi
    valid_crypto_ids_for_agg = []
    if agg_settings.get("enabled_aggregation_tracker", False):
        log_info("--- PELACAK AGREGRASI AKTIF ---", "AGGREGATOR")
        all_individual_ids = [c.get('id') for c in all_crypto_configs_individual]
        for req_id in agg_settings.get("crypto_ids_for_aggregation", []):
            if req_id in all_individual_ids:
                # Cari nama pair untuk log
                pair_name_for_id = "UnknownID"
                for c_ind in all_crypto_configs_individual:
                    if c_ind.get('id') == req_id:
                        pair_name_for_id = f"{c_ind.get('symbol')}-{c_ind.get('currency')}"
                        break
                valid_crypto_ids_for_agg.append({"id": req_id, "name": pair_name_for_id})
            else:
                log_warning(f"ID Kripto '{req_id}' untuk agregasi tidak ditemukan atau tidak aktif, akan diabaikan.", "AGGREGATOR")
        
        if not valid_crypto_ids_for_agg:
            log_error("Tidak ada kripto valid yang terpilih untuk agregasi. Pelacak agregasi tidak akan berjalan.", "AGGREGATOR")
            agg_settings["enabled_aggregation_tracker"] = False # Nonaktifkan jika tidak ada kripto
        else:
            log_info(f"Kripto yang akan diagregasi: {', '.join([p['name'] for p in valid_crypto_ids_for_agg])}", "AGGREGATOR")
            log_info(f"Timeframe Agregasi: {agg_settings.get('aggregation_timeframe', 'N/A')}", "AGGREGATOR")
            log_info(f"Cek Penurunan: {agg_settings.get('drop_percentage_threshold_agg')}% dalam {agg_settings.get('lookback_bars_drop_agg')} bar agregasi", "AGGREGATOR")

    animated_text_display("=========== EXORA AGGREGATOR (Python) START ===========", color=AnsiColors.HEADER, delay=0.005)
    
    local_crypto_data_manager = {} # Untuk data individual pairs
    for config in all_crypto_configs_individual:
        pair_id = config.get('id', str(uuid.uuid4())) # Gunakan ID yang sudah ada atau buat baru
        config['id'] = pair_id # Pastikan ID tersimpan di config
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')} ({config.get('timeframe','DEF')})"
        
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        
        local_crypto_data_manager[pair_id] = {
            "config": config, 
            "all_candles_list": [], 
            "big_data_collection_phase_active": True, 
            "last_candle_fetch_time": datetime.min, 
            "data_fetch_failed_consecutively": 0,
            "last_attempt_after_all_keys_failed": datetime.min
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        initial_candles_target = TARGET_BIG_DATA_CANDLES
        log_info(f"Target data awal: {initial_candles_target} candles.", pair_name=config['pair_name'])
        
        # Fetch data awal untuk pair individual (Logic sama seperti skrip Anda)
        initial_candles = []
        initial_fetch_successful = False
        max_initial_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_key_attempts_done = 0

        while initial_key_attempts_done < max_initial_key_attempts and not initial_fetch_successful:
            current_api_key_init = api_key_manager.get_current_key()
            if not current_api_key_init: break
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key_init, config['timeframe'], pair_name=config['pair_name'])
                initial_fetch_successful = True
            except APIKeyError:
                if not api_key_manager.switch_to_next_key(): break
            except Exception: break # Handle other errors as needed
            initial_key_attempts_done += 1

        if not initial_fetch_successful or not initial_candles:
            log_error(f"BIG DATA: Gagal ambil data awal {config['pair_name']}. Dilewati sementara.", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts + 1
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
        else:
            local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
            log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima {config['pair_name']}.", pair_name=config['pair_name'])
            if len(initial_candles) >= TARGET_BIG_DATA_CANDLES:
                local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
                log_info(f"{AnsiColors.GREEN}Target data awal tercapai {config['pair_name']}.{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}----------------- INISIALISASI PAIR SELESAI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
    
    try: 
        while True:
            current_loop_master_time = datetime.now()
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            # --- PROSES PAIR INDIVIDUAL (Mirip dengan skrip Anda) ---
            active_cryptos_still_in_big_data_collection = 0
            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) + 1:
                    if (current_loop_master_time - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600:
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600)
                        continue
                    else: data_per_pair["data_fetch_failed_consecutively"] = 0
                
                time_since_last_fetch = (current_loop_master_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection +=1
                    required_interval = 60 if 'm' in config_for_pair.get('timeframe','1h') else 120
                
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch)
                    continue
                
                data_per_pair["last_candle_fetch_time"] = current_loop_master_time
                
                # Fetch update untuk pair individual (Logic sama seperti skrip Anda)
                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 
                if data_per_pair["big_data_collection_phase_active"]:
                    needed = TARGET_BIG_DATA_CANDLES - len(data_per_pair["all_candles_list"])
                    limit_fetch_update = min(needed, CRYPTOCOMPARE_MAX_LIMIT) if needed > 0 else 3
                
                if limit_fetch_update > 0:
                    # ... (logika fetch_candles dengan key switching, sama seperti di skrip Anda) ...
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update: break
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_update, config_for_pair['timeframe'], pair_name=pair_name_for_log)
                            fetch_update_successful = True; any_data_fetched_this_cycle = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0
                        except APIKeyError:
                            data_per_pair["data_fetch_failed_consecutively"] +=1
                            if not api_key_manager.switch_to_next_key(): break
                        except Exception: data_per_pair["data_fetch_failed_consecutively"] +=1; break
                        update_key_attempts_done +=1

                if not fetch_update_successful and limit_fetch_update > 0:
                     log_error(f"Gagal update {pair_name_for_log}.", pair_name=pair_name_for_log)
                
                if data_per_pair.get("data_fetch_failed_consecutively",0) >= (api_key_manager.total_keys() or 1)+1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now()

                if new_candles_batch: # Merge candles
                    merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                    added_count, updated_count = 0,0
                    for candle in new_candles_batch:
                        ts = candle['timestamp']
                        if ts not in merged_candles_dict: merged_candles_dict[ts] = candle; added_count +=1
                        elif any(merged_candles_dict[ts][k] != candle[k] for k in ['open','high','low','close']):
                            merged_candles_dict[ts] = candle; updated_count +=1
                    data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                    if added_count + updated_count > 0: log_info(f"{added_count} baru, {updated_count} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)

                if data_per_pair["big_data_collection_phase_active"] and len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                    data_per_pair["big_data_collection_phase_active"] = False
                    active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection - 1)
                    log_info(f"{AnsiColors.GREEN}Target data tercapai {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 200: # Truncate
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 100):]
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)

            # --- PROSES PELACAK AGREGRASI ---
            if agg_settings.get("enabled_aggregation_tracker", False) and valid_crypto_ids_for_agg:
                agg_timeframe_seconds = timeframe_to_seconds(agg_settings.get("aggregation_timeframe", "1h"))
                
                # Cek apakah sudah waktunya membuat bar agregasi baru
                if (current_loop_master_time - last_aggregation_calc_time).total_seconds() >= agg_timeframe_seconds:
                    last_aggregation_calc_time = current_loop_master_time # Atau timestamp pembulatan timeframe
                    
                    # Bulatkan timestamp ke awal periode timeframe agregasi
                    # Misal agg_tf = 1h, current_time = 10:35 -> agg_ts = 10:00
                    # Misal agg_tf = 15m, current_time = 10:37 -> agg_ts = 10:30
                    current_agg_ts_seconds = int(current_loop_master_time.timestamp() / agg_timeframe_seconds) * agg_timeframe_seconds
                    current_aggregation_bar_timestamp = datetime.fromtimestamp(current_agg_ts_seconds)

                    prices_to_average = []
                    constituent_names_for_alert = [] # Untuk pesan alert

                    with lock_ref: # Akses data pair individual yang sudah diupdate
                        for crypto_info_agg in valid_crypto_ids_for_agg:
                            pair_id_agg = crypto_info_agg["id"]
                            if pair_id_agg in shared_dm_ref and shared_dm_ref[pair_id_agg]["all_candles_list"]:
                                # Ambil candle terakhir dari pair individual ini
                                # Untuk kesederhanaan, kita ambil saja close terakhirnya.
                                # Idealnya, kita akan mencari candle yang cocok dengan `current_aggregation_bar_timestamp`
                                latest_candle_individual = shared_dm_ref[pair_id_agg]["all_candles_list"][-1]
                                prices_to_average.append(latest_candle_individual['close'])
                                constituent_names_for_alert.append(crypto_info_agg["name"].split(" ")[0]) # Ambil BTC/USD dari "BTC/USD (1h)"
                            else:
                                log_warning(f"Tidak ada data candle untuk {crypto_info_agg['name']} saat kalkulasi agregasi.", "AGGREGATOR")
                    
                    if prices_to_average:
                        aggregated_value = sum(prices_to_average) / len(prices_to_average)
                        log_info(f"Nilai Agregasi Baru ({current_aggregation_bar_timestamp.strftime('%H:%M:%S')}): {aggregated_value:.4f} (dari {len(prices_to_average)} kripto)", "AGGREGATOR")
                        
                        # Tambahkan ke series agregasi, pastikan timestamp unik jika ada pembulatan
                        # Cek apakah bar dengan timestamp ini sudah ada (jika loop cepat dan pembulatan sama)
                        add_new_agg_bar = True
                        if aggregated_close_series and aggregated_close_series[-1]['timestamp'] == current_aggregation_bar_timestamp:
                            aggregated_close_series[-1]['value'] = aggregated_value # Update nilai jika timestamp sama
                            add_new_agg_bar = False
                        
                        if add_new_agg_bar:
                            aggregated_close_series.append({'timestamp': current_aggregation_bar_timestamp, 'value': aggregated_value})

                        # Jaga ukuran series agregasi
                        max_agg_series_len = agg_settings.get("lookback_bars_drop_agg", 5) + 50 # Buffer
                        if len(aggregated_close_series) > max_agg_series_len:
                            aggregated_close_series = aggregated_close_series[-max_agg_series_len:]
                        
                        # Update data agregasi untuk chart
                        with lock_ref:
                            shared_agg_data_ref["aggregated_series_for_chart"] = copy.deepcopy(aggregated_close_series)

                        # --- Logika Deteksi Penurunan pada Data Agregasi ---
                        lookback_agg = agg_settings.get("lookback_bars_drop_agg", 5)
                        drop_threshold_pct_agg = agg_settings.get("drop_percentage_threshold_agg", 3.0)

                        if len(aggregated_close_series) > lookback_agg : # Perlu lookback_agg + 1 data point
                            current_agg_close = aggregated_close_series[-1]['value']
                            price_at_lookback_agg = aggregated_close_series[-1 - lookback_agg]['value'] # -1 adalah current, -1-lookback adalah N bar lalu

                            is_significant_drop_now_agg = False
                            percentage_drop_actual = 0
                            if price_at_lookback_agg > 0: # Hindari ZeroDivisionError
                                percentage_drop_actual = ((current_agg_close - price_at_lookback_agg) / price_at_lookback_agg) * 100
                                if current_agg_close < (price_at_lookback_agg * (1 - drop_threshold_pct_agg / 100.0)):
                                    is_significant_drop_now_agg = True
                            
                            # Sinyal non-repainting (hanya alert sekali saat kondisi pertama kali terpenuhi)
                            actual_drop_signal_agg = is_significant_drop_now_agg and not is_significant_drop_previously_agg
                            
                            if actual_drop_signal_agg:
                                time_since_last_alert = (current_loop_master_time - last_drop_alert_time_agg).total_seconds()
                                cooldown_agg = agg_settings.get("alert_cooldown_seconds_agg", 300)
                                if time_since_last_alert >= cooldown_agg:
                                    alert_title = f"ALERT: INDEX AGREGRASI TURUN SIGNIFIKAN!"
                                    alert_msg_console = (
                                        f"{AnsiColors.AGG_DROP_ALERT_BG}"
                                        f"Penurunan {abs(percentage_drop_actual):.2f}% pada Indeks Agregasi! "
                                        f"(dari {price_at_lookback_agg:.4f} ke {current_agg_close:.4f} dalam {lookback_agg} bar agregasi). "
                                        f"Kripto: {', '.join(constituent_names_for_alert)} @ {current_aggregation_bar_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                                        f"{AnsiColors.ENDC}"
                                    )
                                    alert_msg_termux = (
                                        f"Penurunan {abs(percentage_drop_actual):.2f}% pada Indeks Agregasi! "
                                        f"Val: {current_agg_close:.3f} (was {price_at_lookback_agg:.3f}). "
                                        f"Bars: {lookback_agg}. "
                                        f"Kripto: {', '.join(constituent_names_for_alert)}."
                                    )
                                    log_warning(alert_msg_console, "AGGREGATOR_DROP")
                                    send_termux_notification(alert_title, alert_msg_termux, global_settings_dict, pair_name_for_log="AGG_DROP")
                                    play_notification_sound()
                                    last_drop_alert_time_agg = current_loop_master_time
                                    
                                    # Simpan info drop untuk chart
                                    with lock_ref:
                                        shared_agg_data_ref["last_significant_drop_info_for_chart"] = {
                                            "timestamp": current_aggregation_bar_timestamp,
                                            "value_at_drop": current_agg_close,
                                            "percentage": percentage_drop_actual
                                        }
                                else:
                                    log_info(f"Sinyal penurunan terdeteksi TAPI masih dalam cooldown. Drop: {abs(percentage_drop_actual):.2f}%", "AGGREGATOR_DROP")
                            
                            is_significant_drop_previously_agg = is_significant_drop_now_agg
                        else:
                             log_debug(f"Belum cukup data agregasi ({len(aggregated_close_series)}/{lookback_agg + 1}) untuk cek penurunan.", "AGGREGATOR")
                    else:
                        log_warning("Tidak ada harga valid untuk dihitung rata-rata agregasi.", "AGGREGATOR")

            # --- Sleep Logic ---
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 10
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(1, int(min_overall_next_refresh_seconds))
            
            # Jika agregasi aktif, pastikan sleep tidak lebih lama dari interval agregasi berikutnya
            if agg_settings.get("enabled_aggregation_tracker", False):
                time_to_next_agg_calc = agg_timeframe_seconds - (current_loop_master_time - last_aggregation_calc_time).total_seconds()
                if time_to_next_agg_calc > 0 :
                     sleep_duration = min(sleep_duration, time_to_next_agg_calc)

            sleep_duration = max(1, int(sleep_duration)) # Minimal 1 detik sleep
            show_spinner(sleep_duration, f"Menunggu {sleep_duration}s ({time.strftime('%H:%M:%S')})...")

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan.{AnsiColors.ENDC}",color=AnsiColors.ORANGE)
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM_LOOP")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EXORA AGGREGATOR (Python) STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali...")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    is_flask_running = any(t.name == "FlaskServerThreadAggregator" for t in threading.enumerate())
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThreadAggregator")
        flask_thread.start()
        time.sleep(1)
    else:
        log_info("Flask server Exora Aggregator sudah berjalan.", "SYSTEM_CHART")

    while True:
        clear_screen_animated()
        settings = load_settings() # Selalu load settings terbaru saat kembali ke menu
        animated_text_display("========= Exora Aggregator (Price Index & Drop Alert) =========", color=AnsiColors.HEADER)
        
        agg_s_main = settings.get("aggregation_settings", {})
        pick_title_main = f"Status Pelacak Agregasi: {AnsiColors.BOLD}{'Aktif' if agg_s_main.get('enabled_aggregation_tracker') else 'Nonaktif'}{AnsiColors.ENDC}\n"

        active_cfgs_main = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs_main: pick_title_main += f"--- Crypto Individu Aktif ({len(active_cfgs_main)}) ---\n" + "".join([f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')})\n" for i,c in enumerate(active_cfgs_main[:3])]) # Tampilkan maks 3
        else: pick_title_main += "Tidak ada konfigurasi crypto individu aktif.\n"
        if len(active_cfgs_main) > 3: pick_title_main += "  ... (dan lainnya)\n"
        
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5002\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        try:
            _, main_idx = pick(main_opts, pick_title_main, indicator='=>')
        except Exception: # Fallback jika pick gagal
            print(pick_title_main); [print(f"{i}. {o}") for i,o in enumerate(main_opts)]
            try: main_idx = int(input("Pilihan: ")); assert 0 <= main_idx < len(main_opts)
            except: print("Input tidak valid."); time.sleep(1); continue

        if main_idx == 0: 
            start_analysis_loop(settings, shared_crypto_data_manager, shared_data_lock, shared_aggregation_data)
        elif main_idx == 1: settings = settings_menu(settings) # settings_menu akan save sendiri
        elif main_idx == 2: log_info("Aplikasi Exora Aggregator ditutup."); break
    animated_text_display("Terima kasih!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan paksa.", color=AnsiColors.ORANGE)
    except Exception as e_global: 
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}ERROR KRITIKAL GLOBAL:{AnsiColors.ENDC}")
        print(f"{AnsiColors.RED}{str(e_global)}{AnsiColors.ENDC}")
        log_exception("MAIN CRITICAL ERROR:",pair_name="SYS_CRIT")
        input("Tekan Enter untuk keluar...")
    finally: 
        sys.stdout.flush(); sys.stderr.flush()
