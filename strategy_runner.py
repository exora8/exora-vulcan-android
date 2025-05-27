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
import subprocess # Untuk termux-notification
import math # Untuk floor, max, dll.

# Import pandas dan pandas-ta
try:
    import pandas as pd
    import pandas_ta as ta
except ImportError:
    print("Pandas atau pandas-ta tidak terinstal. Silakan install dengan: pip install pandas pandas-ta")
    sys.exit(1)

# CHART_INTEGRATION_START
import threading
import copy
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
logger.setLevel(logging.INFO) # Bisa diubah ke logging.DEBUG untuk lebih detail
if logger.hasHandlers():
    logger.handlers.clear()

log_file_name = "trading_log_exora.txt"
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


SETTINGS_FILE = "settings_exora_bot.json" # Nama file settings baru
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 500 # Sesuaikan dengan kebutuhan minimal indikator Exora (swingLookback bisa besar)
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

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION (Sama seperti sebelumnya) ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a', end='', flush=True) # Beep untuk sistem non-Windows
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
    try:
        # Untuk Termux, `--led-color` dan `--led-on`, `--led-off` bisa ditambahkan jika diinginkan,
        # tapi mungkin tidak semua perangkat mendukungnya dengan baik.
        # subprocess.run(['termux-notification', '--title', title, '--content', content_msg, '--led-color', '00FF00', '--led-on', '500', '--led-off', '500'],
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API sudah terinstal dan `termux-notification` bisa diakses.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)


# --- FUNGSI PENGATURAN ---
def get_default_crypto_config_exora(): # Diubah untuk parameter Exora
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "1", # PineScript menyebut "1M", jadi default ke 1 menit
                          # Di API CryptoCompare, untuk menit gunakan 'histominute' dan timeframe value adalah angka
        "refresh_interval_seconds": 60,

        # Parameter Inti Indikator (dari PineScript grpCore)
        "rsiLen": 20,
        "rsiExtremeOversold": 28,
        "rsiExtremeOverbought": 73,
        # rsiSource default ke 'close'
        "stochK": 41,
        "stochSmoothK": 25,
        "stochD": 3, # Periode D Stochastic (untuk smoothing %K yang sudah dismoothing)
        "stochExtremeOversold": 10,
        "stochExtremeOverbought": 80,

        # Filter S/R & Trend (dari PineScript grpFilters)
        "useSwingFilter": True,
        "swingLookback": 100, # Ini adalah KIRI dan KANAN. Pandas-TA mungkin butuh penyesuaian.
        "avoidResistanceProximity": 0.5,

        # Cooldown Setelah Dump (dari PineScript grpCooldown)
        "useDumpCooldown": True,
        "dumpThresholdPercent": 1.0,
        "cooldownPeriodAfterDump": 500, # Ini dalam jumlah bar

        # Strategi Exit (dari PineScript grpExit)
        "useFixedSL": True,
        "slPercent": 4.0,
        "useStandardTP": False, # Defaultnya false di PineScript
        "standardTpPercent": 10.0,
        "useNewTrailingTP": True, # Defaultnya true di PineScript (Step-based)
        "trailingStepPercent": 3.0,
        "trailingGapPercent": 1.5,

        # Pengaturan email per-pair (tetap ada untuk notif sistem jika perlu)
        "enable_email_notifications": False,
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }

def load_settings(): # Disesuaikan untuk config Exora
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
            
            default_crypto_template = get_default_crypto_config_exora()
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
                # Hapus key lama yang tidak relevan (jika ada dari setting sebelumnya)
                keys_to_remove = ["ema_lookback_period", "atr_length", "factor", "emergency_sl_percent", 
                                  "profit_target_percent_activation", "trailing_stop_gap_percent",
                                  "ma_length", "stoch_length", "stoch_smooth_k", "stoch_smooth_d", # stoch_length, smooth_k/d diganti nama di Exora
                                  "stoch_overbought", "stoch_oversold", "left_strength", 
                                  "right_strength", "enable_secure_fib", "secure_fib_check_price"]
                for old_key in keys_to_remove:
                    if old_key in crypto_cfg:
                        del crypto_cfg[old_key]
            return settings
        except Exception as e:
            log_error(f"Error membaca {SETTINGS_FILE}: {e}. Menggunakan default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config_exora()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config_exora()]}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")

def _prompt_crypto_config_exora(current_config): # Diubah untuk Exora
    clear_screen_animated()
    new_config = current_config.copy()
    default_cfg = get_default_crypto_config_exora() # Untuk nilai default
    
    animated_text_display(f"--- Konfigurasi Crypto Pair Exora ({new_config.get('symbol',default_cfg['symbol'])}-{new_config.get('currency',default_cfg['currency'])}) ---", color=AnsiColors.HEADER)
    
    enabled_input = input(f"Aktifkan pair ini? (true/false) [{new_config.get('enabled',default_cfg['enabled'])}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',default_cfg['enabled']))
    
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar [{new_config.get('symbol',default_cfg['symbol'])}]: {AnsiColors.ENDC}") or new_config.get('symbol',default_cfg['symbol'])).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote [{new_config.get('currency',default_cfg['currency'])}]: {AnsiColors.ENDC}") or new_config.get('currency',default_cfg['currency'])).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange [{new_config.get('exchange',default_cfg['exchange'])}]: {AnsiColors.ENDC}") or new_config.get('exchange',default_cfg['exchange'])).strip()
    
    # Timeframe: PineScript "1M". CryptoCompare API uses 'histominute', 'histohour', 'histoday'.
    # Untuk 'histominute', nilai 'limit' dan 'aggregate' (jika digunakan) menjadi penting.
    # Script ini menggunakan 'limit' untuk jumlah candle, bukan 'aggregate'.
    # Jadi, timeframe "1" akan menjadi 1 menit, "60" menjadi 1 jam (jika API mendukung), "1D" menjadi harian.
    # Mari kita buat input lebih jelas untuk timeframe menit/jam/hari
    tf_options = {"minute": "minute", "hour": "hour", "day": "day"}
    current_tf_api_val = new_config.get('timeframe', default_cfg['timeframe']) # Ini akan berupa "1", "60", "1D" dll.

    # Konversi nilai API ke pilihan yang mudah dipahami pengguna
    current_tf_display = "minute" # Default jika tidak cocok
    if isinstance(current_tf_api_val, str) and current_tf_api_val.lower() == "day":
         current_tf_display = "day"
    elif isinstance(current_tf_api_val, str) and current_tf_api_val.lower() == "hour":
         current_tf_display = "hour"
    elif isinstance(current_tf_api_val, int) or (isinstance(current_tf_api_val, str) and current_tf_api_val.isdigit()):
        val_int = int(current_tf_api_val)
        if val_int == 1: current_tf_display = "minute"
        elif val_int == 60 : current_tf_display = "hour" # Asumsi umum
        # ... bisa ditambahkan konversi lain jika perlu

    tf_input_user = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{current_tf_display}]: {AnsiColors.ENDC}") or current_tf_display).lower().strip()

    if tf_input_user == "minute": new_config["timeframe_api"] = "histominute"; new_config["timeframe_value"] = 1
    elif tf_input_user == "hour": new_config["timeframe_api"] = "histohour"; new_config["timeframe_value"] = 1
    elif tf_input_user == "day": new_config["timeframe_api"] = "histoday"; new_config["timeframe_value"] = 1
    else:
        print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default ({current_tf_display}).{AnsiColors.ENDC}")
        # Set ulang ke nilai lama jika input tidak valid
        if 'timeframe_api' not in new_config or 'timeframe_value' not in new_config : # Jika belum pernah diset
            new_config["timeframe_api"] = "histominute"; new_config["timeframe_value"] = 1 # Default ke 1 menit
    
    # Simpan juga 'timeframe' lama untuk kompatibilitas jika diperlukan (atau hapus jika tidak)
    # Untuk kesederhanaan, kita gunakan 'timeframe_api' dan 'timeframe_value' untuk fetch
    new_config["timeframe_display_name"] = tf_input_user # Untuk ditampilkan di UI/log


    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik) [{new_config.get('refresh_interval_seconds',default_cfg['refresh_interval_seconds'])}]: {AnsiColors.ENDC}").strip()
    try:
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',default_cfg['refresh_interval_seconds']))
    except ValueError: new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',default_cfg['refresh_interval_seconds']))

    animated_text_display("\n-- Parameter Indikator Inti (Exora) --", color=AnsiColors.HEADER)
    def get_int_input(prompt, current_val, default_val):
        try: return int(input(f"{AnsiColors.BLUE}{prompt} [{current_val}]: {AnsiColors.ENDC}").strip() or current_val)
        except ValueError: print(f"{AnsiColors.RED}Input tidak valid. Menggunakan nilai {default_val}.{AnsiColors.ENDC}"); return default_val
    def get_float_input(prompt, current_val, default_val):
        try: return float(input(f"{AnsiColors.BLUE}{prompt} [{current_val}]: {AnsiColors.ENDC}").strip() or current_val)
        except ValueError: print(f"{AnsiColors.RED}Input tidak valid. Menggunakan nilai {default_val}.{AnsiColors.ENDC}"); return default_val
    def get_bool_input(prompt, current_val, default_val):
        val = input(f"{AnsiColors.BLUE}{prompt} (true/false) [{current_val}]: {AnsiColors.ENDC}").strip().lower()
        if val == 'true': return True
        if val == 'false': return False
        return default_val

    new_config["rsiLen"] = get_int_input("Periode RSI", new_config.get("rsiLen", default_cfg["rsiLen"]), default_cfg["rsiLen"])
    new_config["rsiExtremeOversold"] = get_int_input("Level RSI Oversold (Entry)", new_config.get("rsiExtremeOversold", default_cfg["rsiExtremeOversold"]), default_cfg["rsiExtremeOversold"])
    new_config["rsiExtremeOverbought"] = get_int_input("Level RSI Overbought (Exit)", new_config.get("rsiExtremeOverbought", default_cfg["rsiExtremeOverbought"]), default_cfg["rsiExtremeOverbought"])
    new_config["stochK"] = get_int_input("Periode %K Stochastic", new_config.get("stochK", default_cfg["stochK"]), default_cfg["stochK"])
    new_config["stochSmoothK"] = get_int_input("Smoothing %K Stochastic", new_config.get("stochSmoothK", default_cfg["stochSmoothK"]), default_cfg["stochSmoothK"])
    new_config["stochD"] = get_int_input("Periode %D Stochastic", new_config.get("stochD", default_cfg["stochD"]), default_cfg["stochD"])
    new_config["stochExtremeOversold"] = get_int_input("Level Stoch SANGAT Oversold (Entry)", new_config.get("stochExtremeOversold", default_cfg["stochExtremeOversold"]), default_cfg["stochExtremeOversold"])
    new_config["stochExtremeOverbought"] = get_int_input("Level Stoch SANGAT Overbought (Exit)", new_config.get("stochExtremeOverbought", default_cfg["stochExtremeOverbought"]), default_cfg["stochExtremeOverbought"])

    animated_text_display("\n-- Filter Trend & S/R (Exora) --", color=AnsiColors.HEADER)
    new_config["useSwingFilter"] = get_bool_input("Gunakan Filter Swing High/Low?", new_config.get("useSwingFilter", default_cfg["useSwingFilter"]), default_cfg["useSwingFilter"])
    new_config["swingLookback"] = get_int_input("Periode Lookback Swing (Kiri & Kanan)", new_config.get("swingLookback", default_cfg["swingLookback"]), default_cfg["swingLookback"])
    new_config["avoidResistanceProximity"] = get_float_input("Jarak Aman % dari Swing High", new_config.get("avoidResistanceProximity", default_cfg["avoidResistanceProximity"]), default_cfg["avoidResistanceProximity"])

    animated_text_display("\n-- Cooldown Setelah Dump (Exora) --", color=AnsiColors.HEADER)
    new_config["useDumpCooldown"] = get_bool_input("Gunakan Cooldown Setelah Dump?", new_config.get("useDumpCooldown", default_cfg["useDumpCooldown"]), default_cfg["useDumpCooldown"])
    new_config["dumpThresholdPercent"] = get_float_input("Min. Penurunan Candle utk Dump (%)", new_config.get("dumpThresholdPercent", default_cfg["dumpThresholdPercent"]), default_cfg["dumpThresholdPercent"])
    new_config["cooldownPeriodAfterDump"] = get_int_input("Periode Cooldown Setelah Dump (bars)", new_config.get("cooldownPeriodAfterDump", default_cfg["cooldownPeriodAfterDump"]), default_cfg["cooldownPeriodAfterDump"])

    animated_text_display("\n-- Strategi Exit (Exora) --", color=AnsiColors.HEADER)
    new_config["useFixedSL"] = get_bool_input("Gunakan Stop Loss Tetap Awal?", new_config.get("useFixedSL", default_cfg["useFixedSL"]), default_cfg["useFixedSL"])
    new_config["slPercent"] = get_float_input("Stop Loss Awal (%)", new_config.get("slPercent", default_cfg["slPercent"]), default_cfg["slPercent"])
    
    new_config["useStandardTP"] = get_bool_input("Gunakan Take Profit Tetap Standar?", new_config.get("useStandardTP", default_cfg["useStandardTP"]), default_cfg["useStandardTP"])
    new_config["standardTpPercent"] = get_float_input("Take Profit Tetap (%)", new_config.get("standardTpPercent", default_cfg["standardTpPercent"]), default_cfg["standardTpPercent"])
    
    new_config["useNewTrailingTP"] = get_bool_input("Gunakan Trailing TP (Step-based)?", new_config.get("useNewTrailingTP", default_cfg["useNewTrailingTP"]), default_cfg["useNewTrailingTP"])
    new_config["trailingStepPercent"] = get_float_input("Trailing Profit Step (%)", new_config.get("trailingStepPercent", default_cfg["trailingStepPercent"]), default_cfg["trailingStepPercent"])
    new_config["trailingGapPercent"] = get_float_input("Trailing Gap dari Step (%)", new_config.get("trailingGapPercent", default_cfg["trailingGapPercent"]), default_cfg["trailingGapPercent"])
    
    animated_text_display("\n-- Notifikasi Email Sistem (Per Pair) --", color=AnsiColors.HEADER)
    new_config["enable_email_notifications"] = get_bool_input("Aktifkan Notifikasi Email Sistem (untuk pair ini)?", new_config.get("enable_email_notifications", default_cfg["enable_email_notifications"]), default_cfg["enable_email_notifications"])
    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()
    
    return new_config

def settings_menu(current_settings): # Disesuaikan untuk Exora
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        pkd = api_s.get('primary_key', 'N/A'); pkd = pkd[:5]+"..."+pkd[-3:] if len(pkd)>8 and pkd not in ["YOUR_PRIMARY_KEY", "N/A"] else pkd
        nrk = len([k for k in api_s.get('recovery_keys', []) if k])
        tns = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        title = f"--- Menu Pengaturan (Exora Bot) ---\nAPI Key: {pkd} | Recovery: {nrk} | Termux: {tns}\nCrypto Pairs:\n"
        
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]):
                 tf_display = cfg.get('timeframe_display_name', '?m')
                 title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({tf_display}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'}\n"
        title += "----------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux", "Tambah Crypto Pair (Exora)", "Ubah Crypto Pair (Exora)", "Hapus Crypto Pair", "Kembali"]
        _, action_idx = pick(opts, title, indicator='=>')
        clear_screen_animated()
        try:
            if action_idx == 0: # Primary API Key (Sama)
                new_pk = input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip()
                if new_pk: api_s["primary_key"] = new_pk
                elif not api_s.get('primary_key'): api_s["primary_key"] = "YOUR_PRIMARY_KEY"

            elif action_idx == 1: # Recovery API Keys (Sama)
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
                    elif rec_action_idx == 2: break # Kembali
                    show_spinner(0.5, "Memproses...")

            elif action_idx == 2: # Email Global Notif Sistem (Sama)
                api_s['enable_global_email_notifications_for_key_switch'] = input(f"Aktifkan Email Notif Sistem Global? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip() == 'true'
                api_s['email_sender_address'] = (input(f"Alamat Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Email Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Alamat Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
            
            elif action_idx == 3: # Notifikasi Termux (Sama)
                api_s['enable_termux_notifications'] = input(f"Aktifkan Notifikasi Termux? (true/false) [{api_s.get('enable_termux_notifications',False)}]: ").lower().strip() == 'true'
            
            elif action_idx == 4: # Tambah Crypto Pair (Exora)
                current_settings.setdefault("cryptos", []).append(_prompt_crypto_config_exora(get_default_crypto_config_exora()))
            elif action_idx == 5: # Ubah Crypto Pair (Exora)
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk diubah."); show_spinner(1,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:")
                if edit_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config_exora(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 6: # Hapus Crypto Pair (Sama)
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
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe_api_endpoint="histohour", timeframe_value_for_api=1, pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    
    all_accumulated_candles = []
    current_to_ts = None
    # timeframe_api_endpoint sudah diberikan (e.g., "histominute", "histohour", "histoday")
    # timeframe_value_for_api adalah 'aggregate' atau jumlah unit untuk 'histominute'
    url = f"https://min-api.cryptocompare.com/data/v2/{timeframe_api_endpoint}"
    is_large_fetch = total_limit_desired > 20 

    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
    
    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts: params["toTs"] = current_to_ts
        
        # Untuk histominute, 'aggregate' adalah jumlah menit per candle. 
        # Jika timeframe_value_for_api=1 dan endpoint='histominute', ini berarti candle 1 menit.
        # Jika endpoint='histohour' atau 'histoday', 'aggregate' biasanya 1.
        if timeframe_api_endpoint == "histominute" and timeframe_value_for_api > 1:
            params["aggregate"] = timeframe_value_for_api
        elif timeframe_api_endpoint in ["histohour", "histoday"] and timeframe_value_for_api > 1 :
             params["aggregate"] = timeframe_value_for_api # Jika ingin candle multi-jam/hari

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
                else: log_warning(f"Tidak ada data candle sama sekali dari API untuk {pair_name} dengan params: {params}", pair_name=pair_name)
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

# --- LOGIKA STRATEGI (EXORA BOT) ---
def get_initial_strategy_state_exora():
    return {
        # Status Posisi & SL/TP
        "in_position": False,
        "entryPriceForTrail": None,
        "highestHighSinceEntry": None,
        "highestNumStepsAchieved": 0,
        "currentTrailingStopLevel": None, # Level SL/Trailing aktif
        "initialStopForCurrentTrade": None,

        # Indikator & Kondisi Entry PineScript
        "hasEnteredOversoldZone": False,
        "rsiHasExitedOversoldZone": False,
        "stochHasExitedOversoldZone": False,

        # Cooldown
        "isCooldownActive": False,
        "cooldownBarsRemaining": 0,

        # Swing Filter State
        "lastValidSwingHigh": None,
        "lastValidSwingLow": None,

        # Data chart (minimal)
        "last_close_for_chart": None,
        "active_sl_tp_for_chart": None, # Akan berisi SL atau Trailing SL
        "entry_price_for_chart": None
    }

def run_strategy_logic_exora(candles_history, crypto_config, strategy_state, global_settings, is_warmup=False):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    cfg = crypto_config # Alias untuk kemudahan

    # Minimal data yang dibutuhkan (perkiraan kasar, swingLookback bisa dominan)
    min_data_needed = max(cfg['rsiLen'], cfg['stochK'] + cfg['stochSmoothK'] + cfg['stochD'], cfg['swingLookback'] * 2 + 5, cfg['cooldownPeriodAfterDump']) + 50 # Buffer tambahan
    
    if len(candles_history) < min_data_needed:
        # log_debug(f"EXORA: Not enough data ({len(candles_history)}/{min_data_needed}) for {pair_name}", pair_name=pair_name)
        return strategy_state # Kembalikan state apa adanya

    # Buat DataFrame pandas untuk kemudahan kalkulasi indikator
    df = pd.DataFrame(candles_history)
    df.set_index('timestamp', inplace=True)

    # === PERHITUNGAN INDIKATOR ===
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=cfg['rsiLen'])
    
    # Stochastic: pandas_ta.stoch(high, low, close, k, d, smooth_k)
    # PineScript: k_val = ta.sma(ta.stoch(close, high, low, stochK), stochSmoothK)
    #             d_val = ta.sma(k_val, stochD) -> %D ini tidak dipakai di kondisi PineScript, hanya %K (k_val)
    # pandas-ta stoch() dengan smooth_k akan menghasilkan %K yang sudah di-smooth oleh SMA(smooth_k).
    # Ini setara dengan `k_val` di PineScript.
    stoch_df = ta.stoch(df['high'], df['low'], df['close'], 
                       k=cfg['stochK'], 
                       d=cfg['stochD'], # Periode D untuk pandas-ta (smoothing dari %K yg sudah di-smooth)
                       smooth_k=cfg['stochSmoothK']) # Smooth_k di pandas-ta adalah smoothing untuk %K mentah
    
    if stoch_df is not None and not stoch_df.empty:
         # Kolom STOCHk_{k}_{smooth_k}_{d} adalah %K yang sudah di-smooth oleh smooth_k
         # Kolom STOCHd_{k}_{smooth_k}_{d} adalah %D (SMA dari %K yg sudah di-smooth)
        k_col_name = f"STOCHk_{cfg['stochK']}_{cfg['stochSmoothK']}_{cfg['stochD']}"
        # d_col_name = f"STOCHd_{cfg['stochK']}_{cfg['stochSmoothK']}_{cfg['stochD']}" # Tidak dipakai di kondisi Exora

        if k_col_name in stoch_df.columns:
            df['k_val'] = stoch_df[k_col_name]
        else: # Fallback jika nama kolom tidak persis, cari yang paling mirip
            for col in stoch_df.columns:
                if col.startswith("STOCHk"):
                    df['k_val'] = stoch_df[col]
                    log_debug(f"EXORA: Menggunakan kolom Stoch %K: {col}",pair_name=pair_name)
                    break
        # if d_col_name in stoch_df.columns: df['d_val'] = stoch_df[d_col_name]
    else:
        df['k_val'] = pd.Series([None] * len(df)) # Atau float('nan')
        # df['d_val'] = pd.Series([None] * len(df))


    # Swing High/Low: ta.pivothigh dan ta.pivotlow di PineScript
    # pandas_ta.pivotlow(high, n=lookback) - n adalah total window, PineScript (left, right)
    # Untuk mencocokkan, kita perlu cara mendapatkan pivot yang dikonfirmasi.
    # PineScript: lastSwingHighPrice = ta.pivothigh(high, swingLookback, swingLookback) -> Cek jika *bar saat ini* adalah pivot.
    # Lalu menyimpan `lastValidSwingHigh`. Ini lebih mudah di Python.
    # Kita akan iterasi mundur untuk mencari pivot terakhir.
    # Untuk penyederhanaan saat ini, kita akan gunakan pandas_ta.pivotlow/high dengan asumsi n adalah lookback ke kiri.
    # Implementasi pivot PineScript yang persis butuh logika custom.
    # pandas_ta.pivot.left (int): left LKB. Default: 5
    # pandas_ta.pivot.right (int): right LKB. Default: 5
    # Untuk match `swingLookback` PineScript (yang simetris):
    if cfg['useSwingFilter']:
        # pandas-ta pivot akan mengembalikan harga pivot jika ada, NaN jika tidak.
        # Kita perlu menyimpan nilai pivot terakhir yang valid.
        pivots_high = ta.pivot(df['high'], left=cfg['swingLookback'], right=cfg['swingLookback'], take="high")
        pivots_low = ta.pivot(df['low'], left=cfg['swingLookback'], right=cfg['swingLookback'], take="low")

        if pivots_high is not None and not pivots_high.empty:
            last_valid_high_from_series = pivots_high[pivots_high.notna()].iloc[-1] if pivots_high.notna().any() else None
            if last_valid_high_from_series is not None:
                strategy_state['lastValidSwingHigh'] = last_valid_high_from_series

        if pivots_low is not None and not pivots_low.empty:
            last_valid_low_from_series = pivots_low[pivots_low.notna()].iloc[-1] if pivots_low.notna().any() else None
            if last_valid_low_from_series is not None:
                strategy_state['lastValidSwingLow'] = last_valid_low_from_series

    # Ambil nilai terbaru
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not df['rsi'].empty else None
    current_k_val = df['k_val'].iloc[-1] if 'k_val' in df.columns and not df['k_val'].empty else None
    # current_d_val = df['d_val'].iloc[-1] if 'd_val' in df.columns and not df['d_val'].empty else None

    current_open = df['open'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_close = df['close'].iloc[-1]
    
    # Untuk crossover, kita butuh nilai sebelumnya
    prev_rsi = df['rsi'].iloc[-2] if 'rsi' in df.columns and len(df['rsi']) >= 2 else None
    prev_k_val = df['k_val'].iloc[-2] if 'k_val' in df.columns and len(df['k_val']) >= 2 else None

    strategy_state["last_close_for_chart"] = current_close


    # === LOGIKA COOLDOWN ===
    if strategy_state['isCooldownActive']:
        strategy_state['cooldownBarsRemaining'] -= 1
        if strategy_state['cooldownBarsRemaining'] <= 0:
            strategy_state['isCooldownActive'] = False
            log_info(f"EXORA: Cooldown berakhir untuk {pair_name}", pair_name=pair_name)

    is_bearish_candle = current_close < current_open
    candle_body_percent_drop = ((current_open - current_close) / current_open * 100) if is_bearish_candle and current_open > 0 else 0.0
    is_dump_candle_now = is_bearish_candle and candle_body_percent_drop >= cfg['dumpThresholdPercent']

    if cfg['useDumpCooldown'] and is_dump_candle_now and not strategy_state['isCooldownActive']: # Hanya trigger jika belum aktif
        strategy_state['isCooldownActive'] = True
        strategy_state['cooldownBarsRemaining'] = cfg['cooldownPeriodAfterDump']
        log_info(f"EXORA: DUMP Candle terdeteksi ({candle_body_percent_drop:.2f}% drop). Cooldown diaktifkan untuk {cfg['cooldownPeriodAfterDump']} bar di {pair_name}", pair_name=pair_name)
        if not is_warmup:
            send_termux_notification(f"DUMP DETECT: {pair_name}", f"Penurunan {candle_body_percent_drop:.2f}%, cooldown {cfg['cooldownPeriodAfterDump']} bar.", global_settings, pair_name_for_log=pair_name)


    # === KONDISI STRATEGI ===
    # (Mirip dengan PineScript `condBuyCoreNew` dan filter)
    if current_rsi is None or current_k_val is None:
        # log_debug(f"EXORA: RSI atau Stoch K bernilai None, skip logic untuk {pair_name}", pair_name=pair_name)
        return strategy_state # Tidak bisa lanjut jika indikator utama None

    rsi_is_currently_oversold = current_rsi < cfg['rsiExtremeOversold']
    stoch_is_currently_oversold = current_k_val < cfg['stochExtremeOversold']

    # Update status masuk/keluar zona oversold
    if rsi_is_currently_oversold and stoch_is_currently_oversold:
        strategy_state['hasEnteredOversoldZone'] = True
        strategy_state['rsiHasExitedOversoldZone'] = False
        strategy_state['stochHasExitedOversoldZone'] = False
    
    if strategy_state['hasEnteredOversoldZone']:
        # Cek Crossover RSI (keluar dari oversold)
        if prev_rsi is not None and prev_rsi < cfg['rsiExtremeOversold'] and current_rsi > cfg['rsiExtremeOversold']:
            strategy_state['rsiHasExitedOversoldZone'] = True
        # Cek Crossover Stochastic %K (keluar dari oversold)
        if prev_k_val is not None and prev_k_val < cfg['stochExtremeOversold'] and current_k_val > cfg['stochExtremeOversold']:
            strategy_state['stochHasExitedOversoldZone'] = True
        
        # Jika masih di dalam zona, reset flag exit (sesuai PineScript)
        if rsi_is_currently_oversold:
            strategy_state['rsiHasExitedOversoldZone'] = False
        if stoch_is_currently_oversold:
            strategy_state['stochHasExitedOversoldZone'] = False
            
    cond_buy_core_new = (strategy_state['hasEnteredOversoldZone'] and
                         strategy_state['rsiHasExitedOversoldZone'] and
                         strategy_state['stochHasExitedOversoldZone'])

    # Filter Resistance
    resistance_filter_ok = True
    if cfg['useSwingFilter'] and strategy_state['lastValidSwingHigh'] is not None:
        # Pine: close < lastValidSwingHigh * (1 - avoidResistanceProximity / 100)
        threshold_price = strategy_state['lastValidSwingHigh'] * (1 - cfg['avoidResistanceProximity'] / 100)
        if current_close >= threshold_price: # Jika harga saat ini DI ATAS atau SAMA DENGAN (terlalu dekat)
            resistance_filter_ok = False
            log_debug(f"EXORA: Resistance filter aktif. Close {current_close} vs SwingHighThreshold {threshold_price} (SwingHigh: {strategy_state['lastValidSwingHigh']})", pair_name=pair_name)

    # Kondisi Beli Final
    buy_condition_filtered = (cond_buy_core_new and
                              resistance_filter_ok and
                              not strategy_state['in_position'] and # Belum ada posisi
                              (not cfg['useDumpCooldown'] or not strategy_state['isCooldownActive']))


    # === MANAJEMEN POSISI & EXIT ===
    # Jika tidak dalam posisi, cek kondisi entry
    if not strategy_state['in_position']:
        if buy_condition_filtered:
            if not is_warmup:
                strategy_state['in_position'] = True
                strategy_state['entryPriceForTrail'] = current_close # Asumsi entry di close candle sinyal
                strategy_state['highestHighSinceEntry'] = current_high
                strategy_state['highestNumStepsAchieved'] = 0
                strategy_state['initialStopForCurrentTrade'] = None
                strategy_state['currentTrailingStopLevel'] = None
                
                strategy_state['entry_price_for_chart'] = strategy_state['entryPriceForTrail']

                if cfg['useFixedSL']:
                    sl_price = strategy_state['entryPriceForTrail'] * (1 - cfg['slPercent'] / 100)
                    strategy_state['initialStopForCurrentTrade'] = sl_price
                    strategy_state['currentTrailingStopLevel'] = sl_price
                    strategy_state['active_sl_tp_for_chart'] = sl_price
                
                entry_msg = f"ENTRY SIGNAL: {pair_name} @ {strategy_state['entryPriceForTrail']:.5f}."
                if strategy_state['currentTrailingStopLevel']:
                    entry_msg += f" Initial SL: {strategy_state['currentTrailingStopLevel']:.5f}"

                log_info(f"{AnsiColors.GREEN}{entry_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"EXORA ENTRY: {pair_name}", f"BUY @ {strategy_state['entryPriceForTrail']:.5f}. SL: {strategy_state.get('currentTrailingStopLevel', 'N/A'):.5f}", global_settings, pair_name_for_log=pair_name)
                play_notification_sound()

            # Reset flag kondisi entry PineScript
            strategy_state['hasEnteredOversoldZone'] = False
            strategy_state['rsiHasExitedOversoldZone'] = False
            strategy_state['stochHasExitedOversoldZone'] = False
    
    # Jika dalam posisi, cek kondisi exit
    elif strategy_state['in_position']:
        entry_price = strategy_state['entryPriceForTrail']
        if entry_price is None: # Seharusnya tidak terjadi jika in_position=True
            log_error(f"EXORA: In position tapi entryPriceForTrail None untuk {pair_name}. Resetting position.", pair_name=pair_name)
            strategy_state.update(get_initial_strategy_state_exora()) # Reset state
            return strategy_state

        strategy_state['highestHighSinceEntry'] = max(strategy_state.get('highestHighSinceEntry', current_high), current_high)
        
        exit_reason = None
        exit_price_for_notif = current_close # Default, bisa di-override oleh SL/TP

        # --- Logika Trailing TP (Step-based) PineScript ---
        if cfg['useNewTrailingTP'] and entry_price > 0 and strategy_state['highestHighSinceEntry'] is not None:
            current_profit_percent = (strategy_state['highestHighSinceEntry'] - entry_price) / entry_price * 100.0
            
            num_steps_achieved = 0
            if cfg['trailingStepPercent'] > 0: # Hindari division by zero
                num_steps_achieved = math.floor(max(0, current_profit_percent) / cfg['trailingStepPercent'])

            if num_steps_achieved > strategy_state['highestNumStepsAchieved'] and num_steps_achieved >= 1:
                strategy_state['highestNumStepsAchieved'] = num_steps_achieved
                
                profit_checkpoint_percent = float(strategy_state['highestNumStepsAchieved']) * cfg['trailingStepPercent']
                locked_profit_percent = max(0.0, profit_checkpoint_percent - cfg['trailingGapPercent'])
                new_calculated_trail_level = entry_price * (1 + locked_profit_percent / 100.0)

                if strategy_state['currentTrailingStopLevel'] is None:
                    strategy_state['currentTrailingStopLevel'] = new_calculated_trail_level
                else:
                    strategy_state['currentTrailingStopLevel'] = max(strategy_state['currentTrailingStopLevel'], new_calculated_trail_level)
                
                # Pastikan trail stop juga menghormati initial SL jika ada (sebagai batas bawah)
                if strategy_state['initialStopForCurrentTrade'] is not None:
                    strategy_state['currentTrailingStopLevel'] = max(strategy_state['currentTrailingStopLevel'], strategy_state['initialStopForCurrentTrade'])
                
                log_info(f"EXORA: Trailing Stop {pair_name} NAIK ke {strategy_state['currentTrailingStopLevel']:.5f} (Profit Step {num_steps_achieved} tercapai)", pair_name=pair_name)
                strategy_state['active_sl_tp_for_chart'] = strategy_state['currentTrailingStopLevel']


        # Tentukan Stop Loss aktual yang akan dicek
        actual_stop_price_for_check = None
        if cfg['useNewTrailingTP'] and strategy_state['currentTrailingStopLevel'] is not None:
            actual_stop_price_for_check = strategy_state['currentTrailingStopLevel']
        elif cfg['useFixedSL'] and strategy_state['initialStopForCurrentTrade'] is not None: # Trailing OFF, Fixed SL ON
             actual_stop_price_for_check = strategy_state['initialStopForCurrentTrade']
        
        strategy_state['active_sl_tp_for_chart'] = actual_stop_price_for_check # Update untuk chart


        # Cek Hit Stop Loss
        if actual_stop_price_for_check is not None and current_low <= actual_stop_price_for_check:
            exit_reason = f"Stop Loss/Trailing Stop terpicu"
            exit_price_for_notif = actual_stop_price_for_check # Harga SL yang terpicu
            # (Simulasi: keluar di level SL, bukan di close candle)

        # Cek Standard Take Profit (HANYA jika Trailing TP BARU dinonaktifkan)
        if not cfg['useNewTrailingTP'] and cfg['useStandardTP'] and entry_price > 0 and not exit_reason:
            tp_price = entry_price * (1 + cfg['standardTpPercent'] / 100)
            if current_high >= tp_price:
                exit_reason = f"Take Profit Standar terpicu"
                exit_price_for_notif = tp_price # Harga TP
                # (Simulasi: keluar di level TP)

        # Cek Exit berdasarkan Kondisi Puncak Ekstrem Indikator (PineScript `sellConditionExtremeHigh`)
        cond_sell_core = current_rsi > cfg['rsiExtremeOverbought'] and current_k_val > cfg['stochExtremeOverbought']
        if cond_sell_core and not exit_reason:
            exit_reason = "Sinyal Exit Puncak Ekstrem Indikator"
            # Keluar di harga close candle ini

        # Jika ada alasan untuk exit
        if exit_reason:
            if not is_warmup:
                profit_loss_percent = ((exit_price_for_notif - entry_price) / entry_price) * 100 if entry_price else 0
                
                close_msg = f"CLOSE SIGNAL: {pair_name} karena {exit_reason} @ ~{exit_price_for_notif:.5f}. " \
                            f"Entry: {entry_price:.5f}. P/L: {profit_loss_percent:.2f}%"
                
                log_color = AnsiColors.RED if profit_loss_percent < 0 else AnsiColors.GREEN
                log_info(f"{log_color}{close_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"EXORA CLOSE: {pair_name}", f"SELL: {exit_reason} @ ~{exit_price_for_notif:.5f}. P/L: {profit_loss_percent:.2f}%", global_settings, pair_name_for_log=pair_name)
                play_notification_sound()

            # Reset semua state posisi
            strategy_state['in_position'] = False
            strategy_state['entryPriceForTrail'] = None
            strategy_state['highestHighSinceEntry'] = None
            strategy_state['highestNumStepsAchieved'] = 0
            strategy_state['currentTrailingStopLevel'] = None
            strategy_state['initialStopForCurrentTrade'] = None
            strategy_state['active_sl_tp_for_chart'] = None # Hapus dari chart
            strategy_state['entry_price_for_chart'] = None

    return strategy_state

# CHART_INTEGRATION_START & Flask Endpoints
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair_exora(pair_id, snapshot): # Disesuaikan untuk Exora
    if pair_id not in snapshot: return None
    data = snapshot[pair_id]
    hist = data.get("all_candles_list", [])[-TARGET_BIG_DATA_CANDLES:] 
    cfg = data.get("config", {})
    state = data.get("strategy_state", {}) # State Exora
    
    ohlc_data = []
    
    pair_display_name = cfg.get('pair_name', pair_id)
    
    # Anotasi untuk SL/TP/Entry
    annotations_yaxis = []
    annotations_points = [] # Untuk menandai entry dengan point

    if state.get("in_position"):
        if state.get("entry_price_for_chart") is not None:
            annotations_yaxis.append({
                "y": state["entry_price_for_chart"], "borderColor": "#007bff", "label": {
                    "borderColor": "#007bff", "style": {"color": "#fff", "background": "#007bff"},
                    "text": f"Entry: {state['entry_price_for_chart']:.4f}"
                }
            })
             # Cari timestamp entry jika memungkinkan (untuk point annotation)
            entry_candle_ts = None
            if hist: # Cari candle yang paling dekat dengan entry (asumsi entry di close)
                # Ini hanya perkiraan, timestamp entry aktual tidak disimpan secara eksplisit per candle
                # Untuk kesederhanaan, kita bisa menandai candle terakhir jika dalam posisi
                # Atau, kita bisa menyimpan timestamp entry di state saat entry terjadi
                # Untuk sekarang, kita skip point annotation untuk entry agar tidak rumit
                pass


        if state.get("active_sl_tp_for_chart") is not None:
            sl_color = "#FF0000" # Merah untuk SL
            label_text = f"SL/Trail: {state['active_sl_tp_for_chart']:.4f}"
            # Jika SL di atas entry (mis. trailing profit), ubah warna jadi hijau
            if state.get("entry_price_for_chart") and state["active_sl_tp_for_chart"] > state["entry_price_for_chart"]:
                sl_color = "#00FF00" # Hijau untuk trailing profit

            annotations_yaxis.append({
                "y": state["active_sl_tp_for_chart"], "borderColor": sl_color, "label": {
                    "borderColor": sl_color, "style": {"color": "#fff", "background": sl_color},
                    "text": label_text
                }
            })

    if not hist:
        return {
            "ohlc": [], "pair_name": pair_display_name,
            "last_updated_tv": None, 
            "strategy_state_info": state, # Kirim semua state untuk debugging jika perlu
            "annotations_yaxis": annotations_yaxis,
            "annotations_points": annotations_points,
            "timeframe_display": cfg.get("timeframe_display_name", "N/A")
        }

    for i, c in enumerate(hist):
        if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = c['timestamp'].timestamp() * 1000
            ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})
    
    return {
        "ohlc": ohlc_data,
        "pair_name": pair_display_name,
        "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
        "strategy_state_info": state,
        "annotations_yaxis": annotations_yaxis,
        "annotations_points": annotations_points,
        "timeframe_display": cfg.get("timeframe_display_name", "N/A")
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE_EXORA = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exora Bot Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width: 100%; max-width: 1200px; }
        select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor: pointer; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; }
        h1 { color: #00bcd4; margin-bottom: 10px; font-size: 1.5em; }
        #lastUpdatedLabel { font-size: .8em; color: #aaa; margin-left: auto; }
        #strategyInfoLabel { font-size: .8em; color: #ffd700; margin-left: 10px; white-space: pre; max-height:100px; overflow-y:auto; border:1px solid #444; padding:5px; }
    </style>
</head>
<body>
    <h1>Exora Bot Chart</h1>
    <div id="controls">
        <label for="pairSelector">Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh</button>
        <div id="strategyInfoLabel">Status: -</div>
        <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container"><div id="chart"></div></div>
    <script>
        let activeChart, currentSelectedPairId = "", lastKnownDataTimestamp = null, autoRefreshIntervalId = null, isLoadingData = false;
        const initialChartOptions = {
            series: [{ name: "Candlestick", type: "candlestick", data: [] }],
            chart: { type: "candlestick", height: 550, background: "#2a2a2a", animations: { enabled: true, dynamicAnimation: { enabled:true, speed:350 } }, toolbar: { show: true } },
            theme: { mode: "dark" },
            title: { text: "Memuat Data Pair...", align: "left", style: { color: "#e0e0e0" } },
            xaxis: { type: "datetime", labels: { style: { colors: "#aaa" } }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: "#aaa" }, formatter: v => v ? v.toFixed(5) : "" } },
            stroke: { width: [1], curve: "straight" },
            markers: { size: 0 }, colors: ["#FEB019"], grid: { borderColor: "#444" },
            annotations: { yaxis: [], points: [] },
            tooltip: { theme: "dark", shared: true, intersect: false,
                custom: function({ series, seriesIndex, dataPointIndex, w }) {
                    let ohlcOpen, ohlcHigh, ohlcLow, ohlcClose;
                    const candleSeriesIdx = w.globals.series.findIndex(s => s.type === 'candlestick');
                    if (candleSeriesIdx !== -1 && w.globals.seriesCandleO[candleSeriesIdx]?.[dataPointIndex] !== undefined) {
                        [ohlcOpen, ohlcHigh, ohlcLow, ohlcClose] = [
                            w.globals.seriesCandleO[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleH[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleL[candleSeriesIdx][dataPointIndex],
                            w.globals.seriesCandleC[candleSeriesIdx][dataPointIndex]
                        ];
                    }
                    let html = '<div style="padding:5px 10px;background:#333;color:#fff;border:1px solid #555;">';
                    if (ohlcOpen !== undefined) {
                        html += ['O', 'H', 'L', 'C'].map((label, idx) => 
                            `<div>${label}: <span style="font-weight:bold;">${[ohlcOpen, ohlcHigh, ohlcLow, ohlcClose][idx].toFixed(5)}</span></div>`
                        ).join('');
                    }
                    html += '</div>';
                    return (ohlcOpen !== undefined) ? html : "";
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
            } catch (error) { console.error("Error fetching available pairs:", error); if (activeChart) { activeChart.destroy(); activeChart = null; } document.getElementById("chart").innerHTML = `Error loading pairs: ${error.message}`; }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById("pairSelector").value;
            lastKnownDataTimestamp = null; 
            loadChartDataForCurrentPair();
        }

        function formatStrategyState(state) {
            if (!state) return "State: N/A";
            let info = `Posisi: ${state.in_position ? 'Aktif' : 'Tidak Aktif'}\n`;
            if (state.in_position) {
                info += `Entry: ${state.entryPriceForTrail ? state.entryPriceForTrail.toFixed(4) : 'N/A'}\n`;
                info += `Trailing SL: ${state.currentTrailingStopLevel ? state.currentTrailingStopLevel.toFixed(4) : 'N/A'}\n`;
                info += `Highest High: ${state.highestHighSinceEntry ? state.highestHighSinceEntry.toFixed(4) : 'N/A'}\n`;
            }
            info += `Cooldown: ${state.isCooldownActive ? state.cooldownBarsRemaining + ' bar lagi' : 'Tidak Aktif'}\n`;
            info += `Swing H: ${state.lastValidSwingHigh ? state.lastValidSwingHigh.toFixed(4) : 'N/A'}\n`;
            info += `Swing L: ${state.lastValidSwingLow ? state.lastValidSwingLow.toFixed(4) : 'N/A'}\n`;
            return info;
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId || isLoadingData) return;
            isLoadingData = true;
            document.getElementById("lastUpdatedLabel").textContent = `Loading ${currentSelectedPairId}...`;
            try {
                const response = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                if (data && data.ohlc) {
                    if (data.last_updated_tv && data.last_updated_tv === lastKnownDataTimestamp && JSON.stringify(data.annotations_yaxis) === JSON.stringify(activeChart ? activeChart.w.config.annotations.yaxis : [])) {
                        console.log("Chart data is unchanged.");
                        document.getElementById("lastUpdatedLabel").textContent = `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                        isLoadingData = false;
                        return;
                    }
                    lastKnownDataTimestamp = data.last_updated_tv;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()} (${data.timeframe_display})` : "Data Loaded";
                    document.getElementById("strategyInfoLabel").textContent = formatStrategyState(data.strategy_state_info);

                    const chartOptionsUpdate = {
                        ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${data.pair_name} (${data.timeframe_display})` },
                        series: [{ name: "Candlestick", type: "candlestick", data: data.ohlc || [] }],
                        annotations: { yaxis: data.annotations_yaxis || [], points: data.annotations_points || [] }
                    };
                    if (data.ohlc.length === 0) {
                        chartOptionsUpdate.title.text = `${data.pair_name || currentSelectedPairId} - No Data`;
                        chartOptionsUpdate.series = initialChartOptions.series.map(s => ({ ...s, data: [] }));
                         document.getElementById("strategyInfoLabel").textContent = "Status: Data Kosong";
                    }

                    if (activeChart) activeChart.updateOptions(chartOptionsUpdate);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), chartOptionsUpdate); activeChart.render(); }
                } else {
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
            } finally { isLoadingData = false; }
        }

        document.addEventListener("DOMContentLoaded", () => {
            if (!activeChart) { activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions); activeChart.render(); }
            fetchAvailablePairs();
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) {
                    await loadChartDataForCurrentPair();
                }
            }, 10000); // Refresh every 10 seconds
        });
    </script>
</body>
</html>
"""

@flask_app_instance.route('/')
def serve_index_page_exora(): return render_template_string(HTML_CHART_TEMPLATE_EXORA)

@flask_app_instance.route('/api/available_pairs')
def get_available_pairs_flask_exora(): 
    with shared_data_lock: data_manager_view = shared_crypto_data_manager.copy()
    active_pairs_info = []
    for pair_id, pair_data in data_manager_view.items():
        cfg = pair_data.get("config", {})
        if cfg.get("enabled", True): active_pairs_info.append({"id": pair_id, "name": cfg.get('pair_name', pair_id)})
    return jsonify(active_pairs_info)

@flask_app_instance.route('/api/chart_data/<pair_id_from_request>')
def get_chart_data_for_frontend_flask_exora(pair_id_from_request): 
    with shared_data_lock:
        if pair_id_from_request not in shared_crypto_data_manager: return jsonify({"error": "Pair not found"}), 404
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {}))
    
    if not pair_data_snapshot: 
        default_cfg = get_default_crypto_config_exora()
        pair_name_default = f"{default_cfg['symbol']}-{default_cfg['currency']}"
        return jsonify({
            "ohlc":[], "pair_name": pair_name_default, "last_updated_tv": None, 
            "strategy_state_info": get_initial_strategy_state_exora(), # Kirim state default
            "annotations_yaxis": [], "annotations_points": [],
            "timeframe_display": default_cfg.get("timeframe_display_name", "N/A")
        }), 200

    # Gunakan fungsi prepare yang sesuai untuk Exora
    prepared_data = prepare_chart_data_for_pair_exora(pair_id_from_request, {pair_id_from_request: pair_data_snapshot})
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server (Exora) di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask: log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
# CHART_INTEGRATION_END

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading_exora(global_settings_dict, shared_dm_ref, lock_ref): # Diubah untuk Exora
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key yang valid. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE); input(); return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE); input(); return

    animated_text_display("=========== EXORA BOT START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    key_idx_display = api_key_manager.get_current_key_index()
    key_val_display = api_key_manager.get_current_key()
    key_val_display_short = ("..." + key_val_display[-3:]) if key_val_display and len(key_val_display) > 8 else key_val_display
    log_info(f"Menggunakan API Key Index: {key_idx_display} ({key_val_display_short}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        # Gunakan timeframe_api dan timeframe_value untuk ID pair agar unik jika ada pair sama tapi beda agregasi menit
        tf_api_id_part = config.get('timeframe_api','histominute') + "_" + str(config.get('timeframe_value',1))
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{tf_api_id_part}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}"
        
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']} ({config.get('timeframe_display_name','?')}){AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], 
            "strategy_state": get_initial_strategy_state_exora(), # State khusus Exora
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        # Perkiraan data minimal untuk Exora
        min_len_for_indicators_init = max(
            config.get('rsiLen', 20),
            config.get('stochK', 41) + config.get('stochSmoothK', 25) + config.get('stochD',3), # Estimasi kasar
            config.get('swingLookback', 100) * 2, # Kiri dan kanan
            config.get('cooldownPeriodAfterDump', 500)
        ) + 100 # Buffer aman
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        log_info(f"EXORA INIT: Target candle awal untuk {config['pair_name']}: {initial_candles_target} (Min logic: {min_len_for_indicators_init - 100})", pair_name=config['pair_name'])

        initial_candles = []
        initial_fetch_successful = False
        max_initial_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_key_attempts_done = 0

        while initial_key_attempts_done < max_initial_key_attempts and not initial_fetch_successful:
            current_api_key_init = api_key_manager.get_current_key()
            if not current_api_key_init:
                log_error(f"BIG DATA: Semua API key habis (global) sebelum fetch {config['pair_name']}.", pair_name=config['pair_name']); break 
            
            log_info(f"BIG DATA: Mencoba fetch awal {config['pair_name']} dengan key idx {api_key_manager.get_current_key_index()} (Attempt {initial_key_attempts_done + 1}/{max_initial_key_attempts})", pair_name=config['pair_name'])
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key_init, 
                                                timeframe_api_endpoint=config.get('timeframe_api', 'histominute'), 
                                                timeframe_value_for_api=config.get('timeframe_value', 1),
                                                pair_name=config['pair_name'])
                initial_fetch_successful = True 
            except APIKeyError:
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): 
                    log_error(f"BIG DATA: Gagal beralih key, semua habis untuk {config['pair_name']}.", pair_name=config['pair_name']); break 
            except requests.exceptions.RequestException as e_req_init:
                log_error(f"BIG DATA: Error Jaringan fetch awal {config['pair_name']}: {e_req_init}. Tidak ganti key.", pair_name=config['pair_name']); break 
            except Exception as e_gen_init:
                log_exception(f"BIG DATA: Error Umum fetch awal {config['pair_name']}: {e_gen_init}. Tidak ganti key.", pair_name=config['pair_name']); break 
            initial_key_attempts_done += 1

        if not initial_fetch_successful or not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal ambil data awal {config['pair_name']}. Dilewati sementara.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts +1 
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Stop trying for this pair for a while
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue 

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])
        
        # Warm-up Strategy State (opsional, tapi bagus untuk `lastValidSwingHigh/Low` dll)
        if initial_candles:
            min_len_for_warmup = min_len_for_indicators_init - 50 # Sedikit lebih kecil dari target, karena kita proses slice
            if len(initial_candles) >= min_len_for_warmup:
                log_info(f"EXORA WARMUP: Memproses {len(initial_candles) - (min_len_for_warmup -1) if len(initial_candles) > min_len_for_warmup -1 else 0} candle historis untuk {config['pair_name']}...", pair_name=config['pair_name'])
                # Loop dari data yang cukup untuk indikator pertama, hingga candle kedua terakhir
                for i_warmup in range(min_len_for_warmup -1, len(initial_candles) -1): 
                    historical_slice = initial_candles[:i_warmup+1] 
                    if len(historical_slice) < (min_len_for_indicators_init - 100): continue # Pastikan cukup data bahkan untuk slice warmup
                    
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic_exora(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict, is_warmup=True
                    )
                log_info(f"{AnsiColors.CYAN}EXORA WARMUP: Inisialisasi state untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"EXORA WARMUP: Tidak cukup data untuk warm-up ({len(initial_candles)}/{min_len_for_warmup}) untuk {config['pair_name']}", pair_name=config['pair_name'])


        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES: # TARGET_BIG_DATA_CANDLES mungkin lebih kecil dari initial_candles_target
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not local_crypto_data_manager[pair_id]["big_data_email_sent"] and config.get("enable_email_notifications"):
                send_email_notification(f"Exora Data Complete: {config['pair_name']}", f"Pengumpulan data Exora ({TARGET_BIG_DATA_CANDLES} candle) selesai untuk {config['pair_name']}.", {**config, 'pair_name': config['pair_name']})
                local_crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS EXORA ({config['pair_name']}) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
    
    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR EXORA DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
    
    try: 
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf')
            any_data_fetched_this_cycle = False

            for pair_id, data_per_pair in local_crypto_data_manager.items():
                config_for_pair = data_per_pair["config"]
                pair_name_for_log = config_for_pair['pair_name']

                # Cooldown jika semua API key gagal
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) + 1 : 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Cooldown 1 jam
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600)
                        continue
                    else: # Reset counter setelah cooldown
                        data_per_pair["data_fetch_failed_consecutively"] = 0 

                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Fetch lebih agresif saat big data (misal tiap menit untuk timeframe menit, atau tiap 5 menit untuk jam/hari)
                    if config_for_pair.get('timeframe_api') == "histominute" and config_for_pair.get('timeframe_value') == 1:
                        required_interval = 60 
                    else:
                        required_interval = 300 
                
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch)
                    continue
                
                log_info(f"EXORA: Memproses {pair_name_for_log} ({config_for_pair.get('timeframe_display_name','?')}, Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                if data_per_pair["big_data_collection_phase_active"]: animated_text_display(f"\n--- BIG DATA EXORA {pair_name_for_log} ({num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD+AnsiColors.MAGENTA)
                else: animated_text_display(f"\n--- LIVE EXORA {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {num_candles_before_fetch} candles ---", color=AnsiColors.BOLD+AnsiColors.CYAN)

                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 # Default untuk update live (ambil beberapa candle terakhir)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    needed_for_big_data = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed_for_big_data <=0 : # Sudah cukup untuk target awal
                        fetch_update_successful = True # Tidak perlu fetch banyak
                        limit_fetch_update = 3 # Hanya update candle terakhir
                    else: 
                        limit_fetch_update = min(needed_for_big_data, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0 or (data_per_pair["big_data_collection_phase_active"] and not fetch_update_successful): 
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    original_api_key_index_for_this_update = api_key_manager.get_current_key_index()

                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE EXORA: Semua API key habis (global) untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        
                        log_info(f"UPDATE EXORA: Mencoba fetch {pair_name_for_log} dgn key idx {api_key_manager.get_current_key_index()} (Attempt {update_key_attempts_done + 1}/{max_update_key_attempts})", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_update, 
                                                              timeframe_api_endpoint=config_for_pair.get('timeframe_api', 'histominute'), 
                                                              timeframe_value_for_api=config_for_pair.get('timeframe_value', 1),
                                                              pair_name=pair_name_for_log)
                            fetch_update_successful = True
                            data_per_pair["data_fetch_failed_consecutively"] = 0 
                            any_data_fetched_this_cycle = True
                            if api_key_manager.get_current_key_index() != original_api_key_index_for_this_update:
                                log_info(f"UPDATE EXORA: Fetch berhasil dgn key idx {api_key_manager.get_current_key_index()} stlh retry {pair_name_for_log}.", pair_name=pair_name_for_log)
                        except APIKeyError:
                            log_warning(f"UPDATE EXORA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                            data_per_pair["data_fetch_failed_consecutively"] +=1
                            if not api_key_manager.switch_to_next_key():
                                log_error(f"UPDATE EXORA: Gagal beralih, semua key habis untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        except requests.exceptions.RequestException as e_req_upd:
                            log_error(f"UPDATE EXORA: Error Jaringan {pair_name_for_log}: {e_req_upd}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break 
                        except Exception as e_gen_upd:
                            log_exception(f"UPDATE EXORA: Error Umum {pair_name_for_log}: {e_gen_upd}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break
                        update_key_attempts_done += 1
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) +1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 

                if not fetch_update_successful or not new_candles_batch:
                    if fetch_update_successful and not new_candles_batch and not data_per_pair["big_data_collection_phase_active"]:
                        log_info(f"EXORA: Tidak ada data candle baru diterima untuk {pair_name_for_log} (fetch berhasil tapi batch kosong).", pair_name=pair_name_for_log)
                    elif not fetch_update_successful:
                         log_error(f"{AnsiColors.RED}EXORA: Gagal mengambil update {pair_name_for_log} stlh semua upaya.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                    with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                    continue

                # Merge candle baru dengan yang lama
                merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                newly_added_count_this_batch, updated_count_this_batch = 0,0
                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: 
                        merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Cek jika ada update di candle yang sama
                        merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                
                data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                if newly_added_count_this_batch + updated_count_this_batch > 0: 
                    log_info(f"EXORA: {newly_added_count_this_batch} baru, {updated_count_this_batch} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)

                # Update status big data collection
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}EXORA: TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if not data_per_pair["big_data_email_sent"] and config_for_pair.get("enable_email_notifications"):
                            send_email_notification(f"Exora Data Complete: {pair_name_for_log}", f"Pengumpulan data Exora ({TARGET_BIG_DATA_CANDLES} candle) selesai untuk {pair_name_for_log}.", {**config_for_pair, 'pair_name': pair_name_for_log})
                            data_per_pair["big_data_email_sent"] = True
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS EXORA ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 200: # Trim jika jauh melebihi target
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 100):]


                # Jalankan logika strategi Exora
                min_len_for_logic_run_live_exora = max(
                    config_for_pair.get('rsiLen', 20),
                    config_for_pair.get('stochK', 41) + config_for_pair.get('stochSmoothK', 25) + config_for_pair.get('stochD',3),
                    config_for_pair.get('swingLookback', 100) * 2, # Kiri dan kanan
                    config_for_pair.get('cooldownPeriodAfterDump', 500)
                ) + 2 # Buffer minimal untuk prev_value
                
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run_live_exora:
                    # Proses logika jika ada candle baru/update, atau baru selesai big data
                    process_logic_now = (newly_added_count_this_batch + updated_count_this_batch > 0 or
                                         (not data_per_pair["big_data_collection_phase_active"] and 
                                          num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and 
                                          len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) ) 
                    
                    if process_logic_now:
                         log_info(f"EXORA: Menjalankan logika strategi untuk {pair_name_for_log}...", pair_name=pair_name_for_log)
                         data_per_pair["strategy_state"] = run_strategy_logic_exora(
                             data_per_pair["all_candles_list"], 
                             config_for_pair, 
                             data_per_pair["strategy_state"], 
                             global_settings_dict,
                             is_warmup=False # Ini adalah pemrosesan live
                        )
                else:
                    log_debug(f"EXORA: Data belum cukup untuk menjalankan logika live ({len(data_per_pair['all_candles_list'])}/{min_len_for_logic_run_live_exora}) untuk {pair_name_for_log}", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            # Tentukan durasi sleep
            sleep_duration = 15 # Default
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: 
                sleep_duration = 3600 # Jika semua key habis, tunggu lama
            elif active_cryptos_still_in_big_data_collection > 0: 
                sleep_duration = 30 # Jika masih ada yg kumpulkan big data, sleep lebih pendek
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
            
            if sleep_duration > 0 : show_spinner(sleep_duration, f"EXORA: Menunggu {int(sleep_duration)}s ({time.strftime('%H:%M:%S')})...")
            else: time.sleep(1) # Minimal sleep

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses Exora dihentikan.{AnsiColors.ENDC}",color=AnsiColors.ORANGE)
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama Exora: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM_EXORA")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EXORA BOT STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali ke menu utama...")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings() # Load settings Exora
    is_flask_running = any(t.name == "FlaskServerThreadExora" for t in threading.enumerate())
    
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThreadExora")
        flask_thread.start()
    else:
        log_info("Flask server Exora sudah berjalan.", "SYSTEM_CHART")

    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Exora Bot V6 =========", color=AnsiColors.HEADER)
        pick_title_main = ""
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs: 
            pick_title_main += f"--- Crypto Aktif ({len(active_cfgs)}) ---\n"
            for i,c in enumerate(active_cfgs):
                tf_disp = c.get('timeframe_display_name', '?')
                pick_title_main += f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({tf_disp})\n"
        else: pick_title_main += "Tidak ada konfigurasi crypto aktif.\n"
        
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5001\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Exora Bot", "Pengaturan Exora Bot", "Keluar"]
        _, main_idx = pick(main_opts, pick_title_main, indicator='=>')
        
        if main_idx == 0: 
            settings = load_settings() # Reload settings sebelum mulai
            start_trading_exora(settings, shared_crypto_data_manager, shared_data_lock)
        elif main_idx == 1: settings = settings_menu(settings) # settings_menu sudah diadaptasi
        elif main_idx == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Terima kasih!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan.", color=AnsiColors.ORANGE)
    except Exception as e_global: clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL: {e_global}{AnsiColors.ENDC}"); log_exception("MAIN ERROR EXORA:",pair_name="SYS_CRIT_EXORA"); input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
