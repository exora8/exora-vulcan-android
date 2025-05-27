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
import math # Untuk math.floor, math.max, dll.

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
logger.setLevel(logging.INFO) # Ubah ke logging.DEBUG untuk lebih detail jika perlu
if logger.hasHandlers():
    logger.handlers.clear()

log_file_name = "trading_log_exora_v6.txt" # Nama file log baru
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


SETTINGS_FILE = "settings_exora_v6.json" # Nama file settings BARU
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 500 # Exora bot mungkin butuh lebih sedikit untuk start, misal 500, tapi swing lookback bisa besar
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15
# EMA_LENGTH_FIXED = 500 # Ini dari strategi lama, akan dihapus dari global jika tidak dipakai lagi

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
        return self.keys[self.current_index] if self.current_index < len(self.keys) else None

    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # Notifikasi email untuk switch key (opsional, sudah ada di skrip Anda)
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                # ... (kode notifikasi email untuk switch key sudah ada) ...
                pass
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            # Notifikasi email untuk semua key gagal (opsional, sudah ada di skrip Anda)
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                 # ... (kode notifikasi email untuk semua key gagal sudah ada) ...
                pass
            return None

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION (Sedikit Modifikasi untuk detail notifikasi trading) ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a', end='', flush=True) # Beep standar
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
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg, '--priority', 'high', '--sound'], # Tambah prioritas & sound
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN (Dirombak untuk Parameter Exora V6) ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "1m", # Default ke 1M sesuai nama PineScript
        "refresh_interval_seconds": 60,

        # Parameter Inti Indikator (grpCore)
        "rsi_len": 20,
        "rsi_extreme_oversold": 28,
        "rsi_extreme_overbought": 73,
        # "rsi_source": "close", # Selalu close di Python ini

        "stoch_k": 41,
        "stoch_smooth_k": 25,
        "stoch_d": 3,
        "stoch_extreme_oversold": 10,
        "stoch_extreme_overbought": 80,

        # Filter S/R & Trend (grpFilters)
        "use_swing_filter": True,
        "swing_lookback": 100, # Periode lookback KIRI saja untuk Python live
        "avoid_resistance_proximity_percent": 0.5,

        # Cooldown Setelah Dump (grpCooldown)
        "use_dump_cooldown": True,
        "dump_threshold_percent": 1.0,
        "cooldown_period_after_dump_bars": 500,

        # Strategi Exit (grpExit)
        "use_fixed_sl": True,
        "sl_percent": 4.0,

        "use_standard_tp": False,
        "standard_tp_percent": 10.0,

        "use_new_trailing_tp": True, # Step-based Trailing TP
        "trailing_step_percent": 3.0,
        "trailing_gap_percent": 1.5,
        
        "enable_email_notifications": False, # Untuk notifikasi sistem (mis. Big Data) & trading
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
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
                # Hapus key lama dari strategi EMA Trend jika ada
                keys_to_remove_old_strat = ["ema_lookback_period"] 
                for old_key in keys_to_remove_old_strat:
                    if old_key in crypto_cfg: del crypto_cfg[old_key]
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

def _prompt_type(prompt_text, current_value, target_type, default_value, min_val=None, max_val=None, step=None):
    while True:
        val_str = input(f"{AnsiColors.BLUE}{prompt_text} [{current_value}]: {AnsiColors.ENDC}").strip()
        if not val_str: return current_value
        try:
            if target_type == bool:
                if val_str.lower() == 'true': return True
                if val_str.lower() == 'false': return False
                raise ValueError("Input boolean tidak valid (true/false)")
            
            typed_val = target_type(val_str)
            
            if min_val is not None and typed_val < min_val:
                print(f"{AnsiColors.RED}Nilai harus >= {min_val}.{AnsiColors.ENDC}")
                continue
            if max_val is not None and typed_val > max_val:
                print(f"{AnsiColors.RED}Nilai harus <= {max_val}.{AnsiColors.ENDC}")
                continue
            # Step tidak di-enforce di sini, tapi bisa ditambahkan validasinya
            return typed_val
        except ValueError as e_val:
            print(f"{AnsiColors.RED}Input tidak valid: {e_val}. Harap masukkan tipe {target_type.__name__}.{AnsiColors.ENDC}")
            # Fallback to default if conversion fails severely or use current_value as fallback
            # return default_value # or current_value

def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy()
    def_cfg = get_default_crypto_config() # Untuk nilai default

    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol',def_cfg['symbol'])}-{new_config.get('currency',def_cfg['currency'])}) ---", color=AnsiColors.HEADER)
    new_config["enabled"] = _prompt_type("Aktifkan pair ini?", new_config.get('enabled', def_cfg['enabled']), bool, def_cfg['enabled'])
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar [{new_config.get('symbol',def_cfg['symbol'])}]: {AnsiColors.ENDC}") or new_config.get('symbol',def_cfg['symbol'])).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote [{new_config.get('currency',def_cfg['currency'])}]: {AnsiColors.ENDC}") or new_config.get('currency',def_cfg['currency'])).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange [{new_config.get('exchange',def_cfg['exchange'])}]: {AnsiColors.ENDC}") or new_config.get('exchange',def_cfg['exchange'])).strip()
    
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d) [{new_config.get('timeframe',def_cfg['timeframe'])}]: {AnsiColors.ENDC}") or new_config.get('timeframe',def_cfg['timeframe'])).lower().strip()
    # Validasi timeframe lebih ketat
    valid_timeframes_map = {"1m": "minute", "5m": "minute", "15m": "minute", "30m":"minute", 
                            "1h": "hour", "4h":"hour", "1d": "day"} # Map ke format CryptoCompare jika perlu
    # Untuk setting, simpan saja input pengguna jika valid
    valid_tf_keys = ["1m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "1d", "3d", "1w"] # Lebih banyak opsi
    if tf_input in valid_tf_keys: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default/sebelumnya.{AnsiColors.ENDC}");

    new_config["refresh_interval_seconds"] = _prompt_type("Interval Refresh (detik)", new_config.get('refresh_interval_seconds',def_cfg['refresh_interval_seconds']), int, def_cfg['refresh_interval_seconds'], min_val=MIN_REFRESH_INTERVAL_AFTER_BIG_DATA)

    animated_text_display("\n-- Indikator Inti --", color=AnsiColors.HEADER)
    new_config["rsi_len"] = _prompt_type("Periode RSI", new_config.get('rsi_len', def_cfg['rsi_len']), int, def_cfg['rsi_len'], min_val=1)
    new_config["rsi_extreme_oversold"] = _prompt_type("Level RSI Oversold (Entry)", new_config.get('rsi_extreme_oversold', def_cfg['rsi_extreme_oversold']), int, def_cfg['rsi_extreme_oversold'], min_val=1, max_val=99)
    new_config["rsi_extreme_overbought"] = _prompt_type("Level RSI Overbought (Exit)", new_config.get('rsi_extreme_overbought', def_cfg['rsi_extreme_overbought']), int, def_cfg['rsi_extreme_overbought'], min_val=1, max_val=99)
    new_config["stoch_k"] = _prompt_type("Periode %K Stochastic", new_config.get('stoch_k', def_cfg['stoch_k']), int, def_cfg['stoch_k'], min_val=1)
    new_config["stoch_smooth_k"] = _prompt_type("Smoothing %K Stochastic", new_config.get('stoch_smooth_k', def_cfg['stoch_smooth_k']), int, def_cfg['stoch_smooth_k'], min_val=1)
    new_config["stoch_d"] = _prompt_type("Periode %D Stochastic", new_config.get('stoch_d', def_cfg['stoch_d']), int, def_cfg['stoch_d'], min_val=1)
    new_config["stoch_extreme_oversold"] = _prompt_type("Level Stoch SANGAT Oversold (Entry)", new_config.get('stoch_extreme_oversold', def_cfg['stoch_extreme_oversold']), int, def_cfg['stoch_extreme_oversold'], min_val=1, max_val=99)
    new_config["stoch_extreme_overbought"] = _prompt_type("Level Stoch SANGAT Overbought (Exit)", new_config.get('stoch_extreme_overbought', def_cfg['stoch_extreme_overbought']), int, def_cfg['stoch_extreme_overbought'], min_val=1, max_val=99)

    animated_text_display("\n-- Filter Trend & S/R --", color=AnsiColors.HEADER)
    new_config["use_swing_filter"] = _prompt_type("Gunakan Filter Swing High/Low?", new_config.get('use_swing_filter', def_cfg['use_swing_filter']), bool, def_cfg['use_swing_filter'])
    new_config["swing_lookback"] = _prompt_type("Periode Lookback Swing (Kiri & Kanan di Pine, Kiri saja di Python)", new_config.get('swing_lookback', def_cfg['swing_lookback']), int, def_cfg['swing_lookback'], min_val=2)
    new_config["avoid_resistance_proximity_percent"] = _prompt_type("Jarak Aman % dari Swing High (Hindari Beli)", new_config.get('avoid_resistance_proximity_percent', def_cfg['avoid_resistance_proximity_percent']), float, def_cfg['avoid_resistance_proximity_percent'], min_val=0.0, step=0.1)

    animated_text_display("\n-- Cooldown Setelah Dump --", color=AnsiColors.HEADER)
    new_config["use_dump_cooldown"] = _prompt_type("Gunakan Cooldown Setelah Dump?", new_config.get('use_dump_cooldown', def_cfg['use_dump_cooldown']), bool, def_cfg['use_dump_cooldown'])
    new_config["dump_threshold_percent"] = _prompt_type("Min. Penurunan Candle utk Dump (%)", new_config.get('dump_threshold_percent', def_cfg['dump_threshold_percent']), float, def_cfg['dump_threshold_percent'], min_val=0.1)
    new_config["cooldown_period_after_dump_bars"] = _prompt_type("Periode Cooldown Setelah Dump (bars)", new_config.get('cooldown_period_after_dump_bars', def_cfg['cooldown_period_after_dump_bars']), int, def_cfg['cooldown_period_after_dump_bars'], min_val=1)

    animated_text_display("\n-- Strategi Exit --", color=AnsiColors.HEADER)
    new_config["use_fixed_sl"] = _prompt_type("Gunakan Stop Loss Tetap Awal?", new_config.get('use_fixed_sl', def_cfg['use_fixed_sl']), bool, def_cfg['use_fixed_sl'])
    new_config["sl_percent"] = _prompt_type("Stop Loss Awal (%)", new_config.get('sl_percent', def_cfg['sl_percent']), float, def_cfg['sl_percent'], min_val=0.1)
    
    new_config["use_standard_tp"] = _prompt_type("Gunakan Take Profit Tetap Standar? (Jika Trailing TP Step OFF)", new_config.get('use_standard_tp', def_cfg['use_standard_tp']), bool, def_cfg['use_standard_tp'])
    new_config["standard_tp_percent"] = _prompt_type("Take Profit Tetap (%)", new_config.get('standard_tp_percent', def_cfg['standard_tp_percent']), float, def_cfg['standard_tp_percent'], min_val=0.1)
    
    new_config["use_new_trailing_tp"] = _prompt_type("Gunakan Trailing TP (Step-based)?", new_config.get('use_new_trailing_tp', def_cfg['use_new_trailing_tp']), bool, def_cfg['use_new_trailing_tp'])
    new_config["trailing_step_percent"] = _prompt_type("Trailing Profit Step (%)", new_config.get('trailing_step_percent', def_cfg['trailing_step_percent']), float, def_cfg['trailing_step_percent'], min_val=0.1, step=0.1)
    new_config["trailing_gap_percent"] = _prompt_type("Trailing Gap dari Step (%)", new_config.get('trailing_gap_percent', def_cfg['trailing_gap_percent']), float, def_cfg['trailing_gap_percent'], min_val=0.0, step=0.1)
    if new_config["trailing_gap_percent"] >= new_config["trailing_step_percent"] and new_config["trailing_step_percent"] > 0:
        print(f"{AnsiColors.ORANGE}Peringatan: Trailing Gap ({new_config['trailing_gap_percent']}%) >= Trailing Step ({new_config['trailing_step_percent']}%) bisa berarti tidak ada profit terkunci per step.{AnsiColors.ENDC}")


    animated_text_display("\n-- Notifikasi Email (Gmail) - Untuk Sinyal Trading & Sistem --", color=AnsiColors.HEADER)
    new_config["enable_email_notifications"] = _prompt_type("Aktifkan Notifikasi Email untuk pair ini?", new_config.get('enable_email_notifications',def_cfg['enable_email_notifications']), bool, def_cfg['enable_email_notifications'])
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
        title = f"--- Menu Pengaturan ---\nAPI Key: {pkd} | Recovery: {nrk} | Termux: {tns}\nStrategi: Exora Bot V6 Spot\nCrypto Pairs:\n"
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]): title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({cfg.get('timeframe','?')}, RSI:{cfg.get('rsi_len','?')}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'}\n"
        title += "----------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux", "Tambah Crypto Pair", "Ubah Crypto Pair", "Hapus Crypto Pair", "Kembali"]
        try:
            # selected_option, action_idx = pick(opts, title, indicator='=>', options_map_func=lambda opt_str: opt_str[:50]) # Potong jika terlalu panjang
            options_for_pick = [opt[:70] + ('...' if len(opt) > 70 else '') for opt in opts] # Batasi panjang opsi
            _, action_idx = pick(options_for_pick, title, indicator='=>')

        except Exception as e_pick: # Fallback jika pick gagal karena layar terlalu kecil
            log_warning(f"Pick library error: {e_pick}. Gunakan input angka.")
            print(title)
            for i, opt_disp in enumerate(options_for_pick): print(f"{i}. {opt_disp}")
            try:
                action_idx = int(input("Masukkan nomor pilihan: "))
                if not (0 <= action_idx < len(options_for_pick)): raise ValueError("Diluar range")
            except ValueError:
                print("Input tidak valid."); time.sleep(1); continue
        
        clear_screen_animated()
        # ... (Sisa dari menu settings seperti API key, recovery, email global, termux notif tetap sama)
        try:
            if action_idx == 0: # Primary API Key
                new_pk = input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip()
                if new_pk: api_s["primary_key"] = new_pk
                elif not api_s.get('primary_key'): api_s["primary_key"] = "YOUR_PRIMARY_KEY" 

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
                            # save_settings(current_settings) # Save moved to end of main settings loop
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
                            # save_settings(current_settings) # Save moved
                            animated_text_display("Recovery key dihapus.", color=AnsiColors.GREEN)
                    elif rec_action_idx == 2: break 
                    show_spinner(0.5, "Memproses...")

            elif action_idx == 2: # Email Global Notif Sistem
                api_s['enable_global_email_notifications_for_key_switch'] = _prompt_type("Aktifkan Email Notif Sistem Global?", api_s.get('enable_global_email_notifications_for_key_switch',False), bool, False)
                api_s['email_sender_address'] = (input(f"Alamat Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Email Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Alamat Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
            
            elif action_idx == 3: # Notifikasi Termux
                 api_s['enable_termux_notifications'] = _prompt_type("Aktifkan Notifikasi Termux Global?", api_s.get('enable_termux_notifications',False), bool, False)
            
            elif action_idx == 4: current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_idx == 5: # Ubah
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk diubah."); show_spinner(1,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:")
                if edit_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 6: # Hapus
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk dihapus."); show_spinner(1,""); continue
                del_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')}" for c in current_settings["cryptos"]] + ["Batal"]
                _, del_c_idx = pick(del_opts, "Pilih pair untuk dihapus:")
                if del_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"].pop(del_c_idx)
            elif action_idx == 7: break # Kembali
            
            current_settings["api_settings"] = api_s 
            save_settings(current_settings) # Simpan semua perubahan di akhir iterasi menu
            if action_idx not in [1,7]: show_spinner(1, "Disimpan...") # kecuali saat di sub-menu recovery atau exit
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA (Sama seperti sebelumnya, hanya modifikasi timeframe mapping) ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe_str="1h", pair_name="N/A"):
    # timeframe_str is user input like "1m", "1h", "1d"
    # Map to CryptoCompare's histometa endpoint requirements
    
    timeframe_details = {"endpoint_segment": "histohour", "aggregate": 1} # Default to 1 hour
    
    tf_lower = timeframe_str.lower()
    if 'm' in tf_lower: # minute
        timeframe_details["endpoint_segment"] = "histominute"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('m',''))
        except: timeframe_details["aggregate"] = 1
    elif 'h' in tf_lower: # hour
        timeframe_details["endpoint_segment"] = "histohour"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('h',''))
        except: timeframe_details["aggregate"] = 1
    elif 'd' in tf_lower: # day
        timeframe_details["endpoint_segment"] = "histoday"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('d',''))
        except: timeframe_details["aggregate"] = 1
    elif 'w' in tf_lower: # week
        timeframe_details["endpoint_segment"] = "histoday" # Weekly is 7 days aggregate
        timeframe_details["aggregate"] = 7 * (int(tf_lower.replace('w','')) if tf_lower.replace('w','').isdigit() else 1)


    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []
    current_to_ts = None
    # api_endpoint = {"minute": "histominute", "hour": "histohour", "day": "histoday"}.get(timeframe, "histohour") # Old logic
    api_endpoint = timeframe_details["endpoint_segment"]
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    
    is_large_fetch = total_limit_desired > 20
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')

    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if timeframe_details["aggregate"] > 1 and api_endpoint != "histominute": # Aggregate only for hour/day if > 1
            params["aggregate"] = timeframe_details["aggregate"]
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


# --- FUNGSI INDIKATOR ---
def calculate_sma(data, period):
    if not data or len(data) < period: return [None] * len(data)
    sma_values = [None] * (period - 1)
    current_sum = sum(data[i] for i in range(period) if data[i] is not None)
    valid_points = sum(1 for i in range(period) if data[i] is not None)

    if valid_points < period : # Handle jika ada None di periode awal
         for i in range(period-1, len(data)):
            current_sum = 0
            valid_points_window = 0
            can_calculate = True
            for j in range(i - period + 1, i + 1):
                if data[j] is None:
                    can_calculate = False
                    break
                current_sum += data[j]
                valid_points_window +=1
            if can_calculate and valid_points_window == period:
                sma_values.append(current_sum / period)
            else:
                sma_values.append(None)
         return sma_values


    sma_values.append(current_sum / period)
    for i in range(period, len(data)):
        if data[i] is not None and data[i-period] is not None and sma_values[-1] is not None:
            current_sum = current_sum - data[i-period] + data[i]
            sma_values.append(current_sum / period)
        else: # Ada None, coba hitung ulang dari awal untuk window ini
            sub_slice = data[i-period+1 : i+1]
            if None not in sub_slice and len(sub_slice) == period:
                sma_values.append(sum(sub_slice) / period)
            else:
                sma_values.append(None)
    return sma_values

def calculate_rsi(prices, period):
    if not prices or len(prices) < period + 1: return [None] * len(prices)
    
    rsi_values = [None] * period 
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period if gains[:period] else 0
    avg_loss = sum(losses[:period]) / period if losses[:period] else 0
    
    if avg_loss == 0: rs = 100 if avg_gain > 0 else 50 # Avoid division by zero; if no losses, RSI is 100 or 50 if no gains either
    else: rs = avg_gain / avg_loss
    
    rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0: rs_i = 100 if avg_gain > 0 else 50
        else: rs_i = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs_i)))
        
    # Pad RSI values list to match length of prices list
    return [None]*(len(prices)-len(rsi_values)) + rsi_values


def calculate_stochastic(candles_history, k_period, smooth_k_period, d_period):
    closes = [c['close'] for c in candles_history]
    highs = [c['high'] for c in candles_history]
    lows = [c['low'] for c in candles_history]
    
    if len(closes) < k_period:
        return [None] * len(closes), [None] * len(closes)

    stoch_k_raw = [None] * (k_period - 1)
    for i in range(k_period - 1, len(closes)):
        period_highs = highs[i - k_period + 1 : i + 1]
        period_lows = lows[i - k_period + 1 : i + 1]
        
        highest_high = max(ph for ph in period_highs if ph is not None) if any(ph is not None for ph in period_highs) else None
        lowest_low = min(pl for pl in period_lows if pl is not None) if any(pl is not None for pl in period_lows) else None
        current_close = closes[i]

        if current_close is not None and highest_high is not None and lowest_low is not None and highest_high != lowest_low:
            k_val = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            stoch_k_raw.append(k_val)
        elif highest_high == lowest_low and current_close is not None and lowest_low is not None: # Avoid division by zero
            stoch_k_raw.append(50) # Or 0 or 100 depending on preference when range is zero
        else:
            stoch_k_raw.append(None)
            
    stoch_k_smoothed = calculate_sma(stoch_k_raw, smooth_k_period)
    stoch_d_smoothed = calculate_sma(stoch_k_smoothed, d_period)
    
    return stoch_k_smoothed, stoch_d_smoothed

def find_latest_confirmed_pivot(candles_history, lookback_period, is_high_pivot=True):
    """
    PineScript's pivothigh(source, leftbars, rightbars) confirms a pivot at index `i`
    if source[i] is greater than `leftbars` on left and `rightbars` on right.
    This means a pivot is confirmed `rightbars` *after* it forms.
    For live Python, `rightbars` are in the future.
    So, we search backwards for the most recent pivot that *could have been confirmed by now*.
    The `lookback_period` from Pine is used for both left and right.
    """
    if not candles_history or len(candles_history) < 2 * lookback_period + 1:
        return None, None # Price, Timestamp

    latest_pivot_price = None
    latest_pivot_timestamp = None

    # Iterate backwards from `len - 1 - lookback` because the pivot at this index
    # would have `lookback` bars to its right available for confirmation.
    # Minimum index to check is `lookback` itself (to have `lookback` bars to its left).
    for i in range(len(candles_history) - 1 - lookback_period, lookback_period - 1, -1):
        candle_to_check = candles_history[i]
        current_val = candle_to_check['high'] if is_high_pivot else candle_to_check['low']
        if current_val is None: continue

        is_pivot = True
        # Check left side
        for j in range(1, lookback_period + 1):
            left_candle_val = candles_history[i-j]['high'] if is_high_pivot else candles_history[i-j]['low']
            if left_candle_val is None or (is_high_pivot and left_candle_val >= current_val) or \
               (not is_high_pivot and left_candle_val <= current_val):
                is_pivot = False; break
        if not is_pivot: continue

        # Check right side
        for j in range(1, lookback_period + 1):
            right_candle_val = candles_history[i+j]['high'] if is_high_pivot else candles_history[i+j]['low']
            if right_candle_val is None or (is_high_pivot and right_candle_val > current_val) or \
               (not is_high_pivot and right_candle_val < current_val): # Pine typically uses > for high, < for low on right
                is_pivot = False; break
        
        if is_pivot:
            latest_pivot_price = current_val
            latest_pivot_timestamp = candle_to_check['timestamp']
            break # Found the most recent one
            
    return latest_pivot_price, latest_pivot_timestamp


# --- LOGIKA STRATEGI (Exora Bot V6) ---
def get_initial_strategy_state():
    return {
        # Indikator state (current values are calculated on the fly)
        "last_rsi_value": None,
        "last_stoch_k_value": None,
        "last_stoch_d_value": None, # Though not directly used in Pine buy/sell conditions

        # Swing S/R Filter state
        "last_valid_swing_high_price": None,
        "last_valid_swing_high_time": None,
        "last_valid_swing_low_price": None, # Not used by this PineScript logic but good to have
        "last_valid_swing_low_time": None,

        # Cooldown state
        "is_cooldown_active": False,
        "cooldown_bars_remaining": 0,

        # Entry condition state (mirrors Pine's `var bool` flags)
        "has_entered_oversold_zone": False,
        "rsi_has_exited_oversold_zone": False,
        "stoch_has_exited_oversold_zone": False,

        # Position state
        "in_position": False,
        "entry_price": None, # Actual entry price after simulated fill
        "position_entry_timestamp": None, # Timestamp of entry bar
        "last_trade_id": None, # To link exit to entry for strategy.exit

        # Trailing TP & SL state
        "entry_price_for_trail": None, # avg_price in Pine, used as base for SL/TP %
        "highest_high_since_entry": None,
        "highest_num_steps_achieved": 0,
        "current_trailing_stop_level": None,
        "initial_stop_for_current_trade": None,
        
        # Charting helper
        "active_sl_tp_for_chart": None,
        "swing_high_for_chart": None,
        "swing_low_for_chart": None,
        "buy_signal_on_chart": False, # To plot shapes
        "sell_signal_on_chart": False,
        "dump_trigger_on_chart": False,
        "cooldown_active_on_chart": False
    }

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings, is_warmup=False):
    pair_name = crypto_config['pair_name']
    cfg = crypto_config # Shorthand

    # Ensure candles_history is not empty
    if not candles_history:
        log_debug("candles_history is empty, skipping strategy logic.", pair_name=pair_name)
        return strategy_state
        
    current_candle = candles_history[-1]
    current_close = current_candle['close']
    current_open = current_candle['open']
    current_high = current_candle['high']
    current_low = current_candle['low']
    current_time = current_candle['timestamp']

    # Reset chart signals for current bar
    strategy_state["buy_signal_on_chart"] = False
    strategy_state["sell_signal_on_chart"] = False
    strategy_state["dump_trigger_on_chart"] = False
    strategy_state["cooldown_active_on_chart"] = False


    # 1. CALCULATE INDICATORS
    # RSI
    rsi_values = calculate_rsi([c['close'] for c in candles_history], cfg['rsi_len'])
    rsi = rsi_values[-1] if rsi_values and len(rsi_values) > 0 else None
    strategy_state["last_rsi_value"] = rsi
    
    # Stochastic
    stoch_k_series, stoch_d_series = calculate_stochastic(candles_history, cfg['stoch_k'], cfg['stoch_smooth_k'], cfg['stoch_d'])
    stoch_k = stoch_k_series[-1] if stoch_k_series and len(stoch_k_series) > 0 else None
    stoch_d = stoch_d_series[-1] if stoch_d_series and len(stoch_d_series) > 0 else None # Not used in buy/sell, but calculated
    strategy_state["last_stoch_k_value"] = stoch_k
    strategy_state["last_stoch_d_value"] = stoch_d

    # Swing High/Low Filter
    if cfg['use_swing_filter']:
        # For 'lastValidSwingHigh', find the latest one confirmed up to previous bar
        # Pine's pivothigh(len,len) means the pivot is `len` bars ago, confirmed now.
        # Our find_latest_confirmed_pivot searches for pivots that could be confirmed by now.
        # This means the pivot itself is `len` bars in the past from the perspective of its right `len` confirmation bars.
        l_sw_h_p, l_sw_h_t = find_latest_confirmed_pivot(candles_history, cfg['swing_lookback'], is_high_pivot=True)
        l_sw_l_p, l_sw_l_t = find_latest_confirmed_pivot(candles_history, cfg['swing_lookback'], is_high_pivot=False)
        
        if l_sw_h_p is not None: # Update if a new one is found or it's the first time
            strategy_state['last_valid_swing_high_price'] = l_sw_h_p
            strategy_state['last_valid_swing_high_time'] = l_sw_h_t
        if l_sw_l_p is not None:
            strategy_state['last_valid_swing_low_price'] = l_sw_l_p
            strategy_state['last_valid_swing_low_time'] = l_sw_l_t
        
        strategy_state["swing_high_for_chart"] = strategy_state['last_valid_swing_high_price']
        strategy_state["swing_low_for_chart"] = strategy_state['last_valid_swing_low_price']


    # 2. COOLDOWN LOGIC
    # This needs to be processed *before* entry conditions.
    # Cooldown decrement happens on *every bar*, regardless of dump detection on current bar.
    if strategy_state['is_cooldown_active']:
        strategy_state['cooldown_bars_remaining'] -= 1
        if strategy_state['cooldown_bars_remaining'] <= 0:
            strategy_state['is_cooldown_active'] = False
            log_info(f"Cooldown period ended for {pair_name}.", pair_name=pair_name)

    is_bearish_candle = current_close < current_open
    candle_body_percent_drop = 0.0
    if is_bearish_candle and current_open > 0: # Avoid division by zero
        candle_body_percent_drop = (current_open - current_close) / current_open * 100.0
    
    is_dump_candle_now = is_bearish_candle and candle_body_percent_drop >= cfg['dump_threshold_percent']

    if cfg['use_dump_cooldown'] and is_dump_candle_now and not strategy_state['is_cooldown_active']: # Activate cooldown only if not already active to prevent reset
        strategy_state['is_cooldown_active'] = True
        strategy_state['cooldown_bars_remaining'] = cfg['cooldown_period_after_dump_bars']
        strategy_state["dump_trigger_on_chart"] = True # For plotting
        log_warning(f"DUMP DETECTED ({candle_body_percent_drop:.2f}%) on {pair_name}. Cooldown for {cfg['cooldown_period_after_dump_bars']} bars activated.", pair_name=pair_name)

    strategy_state["cooldown_active_on_chart"] = strategy_state['is_cooldown_active']


    # 3. STRATEGY CONDITIONS (Entry & Extreme Exit)
    if rsi is None or stoch_k is None:
        log_debug(f"RSI ({rsi}) or Stoch K ({stoch_k}) is None. Skipping signal logic for {pair_name}.", pair_name=pair_name)
        return strategy_state

    # Entry Logic (Buy when exiting oversold after being in oversold)
    rsi_is_currently_oversold = rsi < cfg['rsi_extreme_oversold']
    stoch_is_currently_oversold = stoch_k < cfg['stoch_extreme_oversold']

    # Get previous RSI and Stoch K if available for crossover detection
    prev_rsi = None
    prev_stoch_k = None
    if len(candles_history) > 1 and len(rsi_values) > 1 and len(stoch_k_series) > 1:
        prev_rsi = rsi_values[-2]
        prev_stoch_k = stoch_k_series[-2]

    if rsi_is_currently_oversold and stoch_is_currently_oversold:
        strategy_state['has_entered_oversold_zone'] = True
        strategy_state['rsi_has_exited_oversold_zone'] = False # Reset exit flags
        strategy_state['stoch_has_exited_oversold_zone'] = False
    
    if strategy_state['has_entered_oversold_zone']:
        # Check for RSI crossover upwards
        if prev_rsi is not None and prev_rsi < cfg['rsi_extreme_oversold'] and rsi >= cfg['rsi_extreme_oversold']:
            strategy_state['rsi_has_exited_oversold_zone'] = True
        # Check for Stoch K crossover upwards
        if prev_stoch_k is not None and prev_stoch_k < cfg['stoch_extreme_oversold'] and stoch_k >= cfg['stoch_extreme_oversold']:
            strategy_state['stoch_has_exited_oversold_zone'] = True
        
        # If currently still oversold, it means it hasn't truly exited yet
        if rsi_is_currently_oversold:
            strategy_state['rsi_has_exited_oversold_zone'] = False
        if stoch_is_currently_oversold:
            strategy_state['stoch_has_exited_oversold_zone'] = False

    cond_buy_core_new = (strategy_state['has_entered_oversold_zone'] and
                         strategy_state['rsi_has_exited_oversold_zone'] and
                         strategy_state['stoch_has_exited_oversold_zone'])

    # Resistance Filter
    resistance_filter_ok = True
    if cfg['use_swing_filter']:
        last_sw_high = strategy_state['last_valid_swing_high_price']
        if last_sw_high is not None:
            avoid_price_level = last_sw_high * (1 - cfg['avoid_resistance_proximity_percent'] / 100.0)
            if current_close >= avoid_price_level: # Price is too close or above resistance
                resistance_filter_ok = False
                log_debug(f"Resistance filter: BUY BLOCKED. Close {current_close} near SwingH {last_sw_high} (threshold {avoid_price_level})", pair_name=pair_name)
        # If no swing high found, filter is OK by default (or could be strict and block) - Pine: na(lastValidSwingHigh) is OK
    
    # Cooldown Filter
    cooldown_filter_ok = not (cfg['use_dump_cooldown'] and strategy_state['is_cooldown_active'])
    if not cooldown_filter_ok:
         log_debug(f"Cooldown filter: BUY BLOCKED. Cooldown active for {strategy_state['cooldown_bars_remaining']} more bars.", pair_name=pair_name)


    # Final Buy Condition
    buy_condition_filtered = (cond_buy_core_new and
                              resistance_filter_ok and
                              cooldown_filter_ok and
                              not strategy_state['in_position']) # Not already in a position

    # Extreme Overbought Sell Condition (for exiting a current position)
    cond_sell_core_extreme = (rsi > cfg['rsi_extreme_overbought'] and
                              stoch_k > cfg['stoch_extreme_overbought'])
    
    # --- POSITION MANAGEMENT ---
    trade_closed_this_bar = False

    # Check Exit conditions if in position
    if strategy_state['in_position']:
        entry_p = strategy_state['entry_price_for_trail']
        
        # Update highest high since entry
        if strategy_state['highest_high_since_entry'] is None:
            strategy_state['highest_high_since_entry'] = current_high # Initialize on first bar in position
        else:
            strategy_state['highest_high_since_entry'] = max(strategy_state['highest_high_since_entry'], current_high)

        # Step-based Trailing TP Logic
        if cfg['use_new_trailing_tp'] and entry_p is not None and entry_p > 0 and strategy_state['highest_high_since_entry'] is not None:
            current_profit_percent = (strategy_state['highest_high_since_entry'] - entry_p) / entry_p * 100.0
            
            num_steps_achieved = 0
            if cfg['trailing_step_percent'] > 0: # Avoid division by zero
                num_steps_achieved = math.floor(max(0, current_profit_percent) / cfg['trailing_step_percent'])

            if num_steps_achieved > strategy_state['highest_num_steps_achieved'] and num_steps_achieved >= 1:
                old_trail_level = strategy_state['current_trailing_stop_level']
                strategy_state['highest_num_steps_achieved'] = num_steps_achieved
                
                profit_checkpoint_percent = float(strategy_state['highest_num_steps_achieved']) * cfg['trailing_step_percent']
                locked_profit_percent = max(0.0, profit_checkpoint_percent - cfg['trailing_gap_percent'])
                new_calculated_trail_level = entry_p * (1 + locked_profit_percent / 100.0)

                if strategy_state['current_trailing_stop_level'] is None:
                    strategy_state['current_trailing_stop_level'] = new_calculated_trail_level
                else:
                    strategy_state['current_trailing_stop_level'] = max(strategy_state['current_trailing_stop_level'], new_calculated_trail_level)
                
                # Ensure it also respects the initialStopForCurrentTrade
                if strategy_state['initial_stop_for_current_trade'] is not None:
                    strategy_state['current_trailing_stop_level'] = max(strategy_state['current_trailing_stop_level'], strategy_state['initial_stop_for_current_trade'])

                if strategy_state['current_trailing_stop_level'] != old_trail_level and not is_warmup:
                    msg = f"TRAILING STOP MOVED UP for {pair_name} to {strategy_state['current_trailing_stop_level']:.5f} ({locked_profit_percent:.2f}% locked profit of entry)."
                    log_info(f"{AnsiColors.MAGENTA}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
                    send_termux_notification(f"Trail Up {pair_name}", msg, global_settings, pair_name_for_log=pair_name)
                    if cfg.get("enable_email_notifications"):
                         send_email_notification(f"ExoraV6 Trail UP: {pair_name}", msg, cfg, pair_name_ctx_override=pair_name)


        # Determine Actual Stop/Limit for exit check
        actual_stop_price_for_exit = None
        if cfg['use_new_trailing_tp'] and strategy_state['current_trailing_stop_level'] is not None:
            actual_stop_price_for_exit = strategy_state['current_trailing_stop_level']
        elif cfg['use_fixed_sl'] and strategy_state['initial_stop_for_current_trade'] is not None: # New Trailing TP OFF, but fixed SL ON
            actual_stop_price_for_exit = strategy_state['initial_stop_for_current_trade']
        
        actual_take_profit_price_for_exit = None
        if not cfg['use_new_trailing_tp'] and cfg['use_standard_tp'] and entry_p is not None:
            actual_take_profit_price_for_exit = entry_p * (1 + cfg['standard_tp_percent'] / 100.0)

        strategy_state["active_sl_tp_for_chart"] = actual_stop_price_for_exit # Plot active SL/Trail

        # Check for SL hit
        if actual_stop_price_for_exit is not None and current_low <= actual_stop_price_for_exit:
            exit_price = min(current_open, actual_stop_price_for_exit) # Simulate execution
            profit_percent = (exit_price - strategy_state['entry_price']) / strategy_state['entry_price'] * 100 if strategy_state['entry_price'] else 0
            msg = f"STOP LOSS HIT for {pair_name} at ~{exit_price:.5f}. Entry: {strategy_state['entry_price']:.5f}. Profit: {profit_percent:.2f}%"
            if not is_warmup:
                log_info(f"{AnsiColors.RED}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"SL HIT {pair_name}", msg, global_settings, pair_name_for_log=pair_name)
                if cfg.get("enable_email_notifications"):
                    send_email_notification(f"ExoraV6 SL HIT: {pair_name}", msg, cfg, pair_name_ctx_override=pair_name)
            
            # Reset state for new trade
            strategy_state['in_position'] = False; trade_closed_this_bar = True

        # Check for Standard TP hit (only if SL not hit and standard TP active)
        elif not trade_closed_this_bar and actual_take_profit_price_for_exit is not None and current_high >= actual_take_profit_price_for_exit:
            exit_price = max(current_open, actual_take_profit_price_for_exit) # Simulate execution
            profit_percent = (exit_price - strategy_state['entry_price']) / strategy_state['entry_price'] * 100 if strategy_state['entry_price'] else 0
            msg = f"STANDARD TAKE PROFIT HIT for {pair_name} at ~{exit_price:.5f}. Entry: {strategy_state['entry_price']:.5f}. Profit: {profit_percent:.2f}%"
            if not is_warmup:
                log_info(f"{AnsiColors.GREEN}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"TP HIT {pair_name}", msg, global_settings, pair_name_for_log=pair_name)
                if cfg.get("enable_email_notifications"):
                    send_email_notification(f"ExoraV6 TP HIT: {pair_name}", msg, cfg, pair_name_ctx_override=pair_name)

            strategy_state['in_position'] = False; trade_closed_this_bar = True

        # Check for Extreme Overbought Exit (only if SL/TP not hit)
        elif not trade_closed_this_bar and cond_sell_core_extreme:
            exit_price = current_close # Exit at current close for this signal
            profit_percent = (exit_price - strategy_state['entry_price']) / strategy_state['entry_price'] * 100 if strategy_state['entry_price'] else 0
            msg = f"EXTREME OVERBOUGHT EXIT for {pair_name} at ~{exit_price:.5f}. Entry: {strategy_state['entry_price']:.5f}. Profit: {profit_percent:.2f}%"
            if not is_warmup:
                log_info(f"{AnsiColors.ORANGE}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"Exit OB {pair_name}", msg, global_settings, pair_name_for_log=pair_name)
                if cfg.get("enable_email_notifications"):
                    send_email_notification(f"ExoraV6 Exit Overbought: {pair_name}", msg, cfg, pair_name_ctx_override=pair_name)
                strategy_state["sell_signal_on_chart"] = True


            strategy_state['in_position'] = False; trade_closed_this_bar = True
        
        if trade_closed_this_bar: # Common reset logic if any exit condition met
            strategy_state['entry_price'] = None
            strategy_state['position_entry_timestamp'] = None
            strategy_state['entry_price_for_trail'] = None
            strategy_state['highest_high_since_entry'] = None
            strategy_state['highest_num_steps_achieved'] = 0
            strategy_state['current_trailing_stop_level'] = None
            strategy_state['initial_stop_for_current_trade'] = None
            strategy_state['active_sl_tp_for_chart'] = None
            # Do NOT reset has_entered_oversold_zone here, let it be reset by entry logic only


    # Execute Buy if conditions met and not in position
    if buy_condition_filtered and not strategy_state['in_position'] and not trade_closed_this_bar: # Ensure no exit happened on same bar before new entry
        entry_price_simulated = current_close # Enter at close of signal bar
        strategy_state['in_position'] = True
        strategy_state['entry_price'] = entry_price_simulated
        strategy_state['position_entry_timestamp'] = current_time
        strategy_state['last_trade_id'] = str(uuid.uuid4()) # For potential future use

        # Initialize SL/TP vars for the new trade (mimicking Pine's next bar setup, but done immediately here)
        strategy_state['entry_price_for_trail'] = entry_price_simulated
        strategy_state['highest_high_since_entry'] = current_high # Initial high is current bar's high
        strategy_state['highest_num_steps_achieved'] = 0
        
        if cfg['use_fixed_sl']:
            strategy_state['initial_stop_for_current_trade'] = entry_price_simulated * (1 - cfg['sl_percent'] / 100.0)
            strategy_state['current_trailing_stop_level'] = strategy_state['initial_stop_for_current_trade']
        else:
            strategy_state['initial_stop_for_current_trade'] = None
            strategy_state['current_trailing_stop_level'] = None # No SL unless trailing step hit
        
        strategy_state["active_sl_tp_for_chart"] = strategy_state['current_trailing_stop_level']


        if not is_warmup:
            msg = f"ENTRY LONG SIGNAL for {pair_name} at ~{entry_price_simulated:.5f}. SL: {strategy_state['current_trailing_stop_level']:.5f if strategy_state['current_trailing_stop_level'] else 'N/A'}"
            log_info(f"{AnsiColors.GREEN}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
            send_termux_notification(f"ENTRY {pair_name}", msg, global_settings, pair_name_for_log=pair_name)
            if cfg.get("enable_email_notifications"):
                 send_email_notification(f"ExoraV6 ENTRY: {pair_name}", msg, cfg, pair_name_ctx_override=pair_name)
            strategy_state["buy_signal_on_chart"] = True

        # Reset oversold zone flags as per PineScript
        strategy_state['has_entered_oversold_zone'] = False
        strategy_state['rsi_has_exited_oversold_zone'] = False
        strategy_state['stoch_has_exited_oversold_zone'] = False

    return strategy_state


# CHART_INTEGRATION_START & Flask Endpoints (Modified for Exora V6 data)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id, snapshot):
    if pair_id not in snapshot: return None
    data = snapshot[pair_id]
    hist = data.get("all_candles_list", [])[-TARGET_BIG_DATA_CANDLES:] 
    cfg = data.get("config", {})
    state = data.get("strategy_state", {})
    
    ohlc_data = []
    # Indikator series (opsional, bisa ditambahkan jika diinginkan, tapi akan ramai)
    # rsi_series_data = [] 
    # stoch_k_series_data = []
    active_sl_tp_line_data = []
    swing_high_points = []
    swing_low_points = []
    
    pair_display_name = cfg.get('pair_name', pair_id)

    if not hist: # Return empty structure if no history
        return {
            "ohlc": [], "active_sl_tp_line": [], "swing_high_points": [], "swing_low_points": [],
            "pair_name": pair_display_name, "last_updated_tv": None,
            "strategy_state_info": state, # Pass full state for potential display
            "config_info": cfg, # Pass config for display
            "annotations_points": [] # For buy/sell signals
        }

    # Calculate indicators for chart if needed, or use stored values from state if sufficient
    # For simplicity, we'll mainly use values stored in state by run_strategy_logic
    # Full recalculation for chart can be heavy.
    
    annotations_points_chart = []

    for i, c in enumerate(hist):
        if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = c['timestamp'].timestamp() * 1000
            ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})

            # Check if this is the *current* bar being processed by strategy_state
            # This is a simplification; ideally, chart should show historical state too.
            is_current_eval_bar = (i == len(hist) - 1)

            if is_current_eval_bar:
                if state.get("active_sl_tp_for_chart") is not None:
                     # For line, we need at least two points or extend it visually
                    if len(ohlc_data) > 1: # Need previous point to draw line segment
                         active_sl_tp_line_data.append({'x': ohlc_data[-2]['x'], 'y': state.get("active_sl_tp_for_chart")}) # Start from prev bar
                    active_sl_tp_line_data.append({'x': ts_ms, 'y': state.get("active_sl_tp_for_chart")})
                
                if cfg.get("use_swing_filter"):
                    if state.get("swing_high_for_chart") and state.get("last_valid_swing_high_time"):
                        # Plot a point if the swing high is recent enough to be in view
                        # For simplicity, plot if it's the one from state for the current bar
                        swing_high_points.append({
                            'x': state.get("last_valid_swing_high_time").timestamp() * 1000,
                            'y': state.get("swing_high_for_chart"),
                            'marker': {'size': 6, 'fillColor': '#FF0000', 'strokeColor': '#FF0000'},
                            'label': {'borderColor': '#FF0000', 'offsetY': -15, 'style': {'color': '#fff', 'background': '#FF0000'}, 'text': 'SH'}
                        })
                    if state.get("swing_low_for_chart") and state.get("last_valid_swing_low_time"):
                         swing_low_points.append({
                            'x': state.get("last_valid_swing_low_time").timestamp() * 1000,
                            'y': state.get("swing_low_for_chart"),
                            'marker': {'size': 6, 'fillColor': '#00FF00', 'strokeColor': '#00FF00'},
                            'label': {'borderColor': '#00FF00', 'offsetY': 15, 'style': {'color': '#fff', 'background': '#00FF00'}, 'text': 'SLow'}
                        })
                
                # Buy/Sell/Dump Signals as point annotations
                if state.get("buy_signal_on_chart"):
                    annotations_points_chart.append({
                        'x': ts_ms, 'y': c['low'], 'marker': {'size': 8, 'fillColor': '#26E7A5', 'strokeColor': '#26E7A5', 'shape': 'triangle'},
                        'label': {'borderColor': '#26E7A5', 'offsetY': 10, 'style': {'color': '#fff', 'background': '#26E7A5'}, 'text': 'BUY'}
                    })
                if state.get("sell_signal_on_chart"): # Extreme OB exit
                    annotations_points_chart.append({
                        'x': ts_ms, 'y': c['high'], 'marker': {'size': 8, 'fillColor': '#FF4560', 'strokeColor': '#FF4560', 'shape': 'triangle-down'},
                        'label': {'borderColor': '#FF4560', 'offsetY': -10, 'style': {'color': '#fff', 'background': '#FF4560'}, 'text': 'SELL-OB'}
                    })
                if state.get("dump_trigger_on_chart"):
                     annotations_points_chart.append({
                        'x': ts_ms, 'y': c['high'], 'marker': {'size': 6, 'fillColor': '#FFA500', 'strokeColor': '#FFA500', 'shape': 'square'},
                        'label': {'borderColor': '#FFA500', 'offsetY': -10, 'style': {'color': '#fff', 'background': '#FFA500'}, 'text': 'DUMP'}
                    })


    return {
        "ohlc": ohlc_data,
        "active_sl_tp_line": active_sl_tp_line_data,
        "swing_high_points": swing_high_points, # These will be point annotations
        "swing_low_points": swing_low_points,   # These will be point annotations
        "annotations_points": annotations_points_chart, # For Buy/Sell signals
        "pair_name": pair_display_name,
        "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
        "strategy_state_info": state, # Pass state for display
        "config_info": cfg,
        "cooldown_active_bg": state.get("cooldown_active_on_chart", False) # For background color
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exora V6 Spot Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width: 100%; max-width: 1200px; flex-wrap: wrap; }
        select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor: pointer; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; }
        h1 { color: #00bcd4; margin-bottom: 10px; font-size: 1.5em; }
        #lastUpdatedLabel { font-size: .8em; color: #aaa; margin-left: auto; }
        #strategyInfoLabel { font-size: .8em; color: #ffd700; margin-left: 10px; white-space: pre-wrap; max-width: 300px; }
        .cooldown-active { background-color: rgba(128, 128, 128, 0.3) !important; } /* Grayish transparent */
    </style>
</head>
<body>
    <h1>Exora V6 Spot Strategy Chart</h1>
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
                { name: "Active SL/Trail", type: "line", data: [], color: "#FFA500" } // Orange for SL/Trail line
            ],
            chart: { type: "candlestick", height: 550, background: "#2a2a2a", animations: { enabled: false }, toolbar: { show: true } },
            theme: { mode: "dark" },
            title: { text: "Memuat Data Pair...", align: "left", style: { color: "#e0e0e0" } },
            xaxis: { type: "datetime", labels: { style: { colors: "#aaa" } }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: "#aaa" }, formatter: v => v ? v.toFixed(5) : "" } },
            stroke: { width: [1, 2], curve: "straight" }, 
            markers: { size: 0 }, // Default no markers, will be added by annotations
            colors: ["#FEB019", "#FFA500"], 
            grid: { borderColor: "#444" },
            annotations: { yaxis: [], points: [] },
            tooltip: { theme: "dark", shared: true, intersect: false, y: { formatter: val => val ? val.toFixed(5) : val } },
            noData: { text: "Tidak ada data.", align: "center", style: { color: "#ccc" } }
        };

        async function fetchAvailablePairs() { /* ... (sama seperti sebelumnya) ... */ 
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


        function handlePairSelectionChange() { /* ... (sama seperti sebelumnya) ... */
            currentSelectedPairId = document.getElementById("pairSelector").value;
            lastKnownDataTimestamp = null; 
            loadChartDataForCurrentPair();
         }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId || isLoadingData) return;
            isLoadingData = true;
            document.getElementById("lastUpdatedLabel").textContent = `Loading ${currentSelectedPairId}...`;
            const chartContainer = document.getElementById("chart-container");
            try {
                const response = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                if (data && data.ohlc) {
                    if (data.last_updated_tv && data.last_updated_tv === lastKnownDataTimestamp && !data.force_update_chart) { // Add force_update_chart concept if needed
                        console.log("Chart data is unchanged based on timestamp.");
                        document.getElementById("lastUpdatedLabel").textContent = `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                        isLoadingData = false;
                        // Update cooldown bg color even if data is same
                        if (data.cooldown_active_bg) chartContainer.classList.add("cooldown-active");
                        else chartContainer.classList.remove("cooldown-active");
                        return;
                    }
                    lastKnownDataTimestamp = data.last_updated_tv;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Data Loaded";
                    
                    const stateInfo = data.strategy_state_info || {};
                    const configInfo = data.config_info || {};
                    let infoText = \`In Position: \${stateInfo.in_position ? 'YES' : 'NO'}\\n\`;
                    if(stateInfo.in_position) {
                        infoText += \`Entry: \${stateInfo.entry_price ? stateInfo.entry_price.toFixed(5) : 'N/A'}\\n\`;
                        infoText += \`SL/Trail: \${stateInfo.active_sl_tp_for_chart ? stateInfo.active_sl_tp_for_chart.toFixed(5) : 'N/A'}\\n\`;
                        infoText += \`Highest High: \${stateInfo.highest_high_since_entry ? stateInfo.highest_high_since_entry.toFixed(5) : 'N/A'}\\n\`;
                    }
                    infoText += \`RSI(\${configInfo.rsi_len}): \${stateInfo.last_rsi_value ? stateInfo.last_rsi_value.toFixed(2) : 'N/A'}\\n\`;
                    infoText += \`StochK(\${configInfo.stoch_k},\${configInfo.stoch_smooth_k}): \${stateInfo.last_stoch_k_value ? stateInfo.last_stoch_k_value.toFixed(2) : 'N/A'}\\n\`;
                    infoText += \`Cooldown: \${stateInfo.is_cooldown_active ? stateInfo.cooldown_bars_remaining + ' bars' : 'OFF'}\`;
                    document.getElementById("strategyInfoLabel").textContent = infoText;

                    // Combine all point annotations
                    let all_point_annotations = [
                        ...(data.swing_high_points || []), 
                        ...(data.swing_low_points || []),
                        ...(data.annotations_points || []) // Buy/Sell signals
                    ];
                    
                    const chartOptionsUpdate = {
                        title: { ...initialChartOptions.title, text: \`\${data.pair_name} - Exora V6 Spot (\${configInfo.timeframe})\` },
                        series: [
                            { name: "Candlestick", type: "candlestick", data: data.ohlc || [] },
                            { name: "Active SL/Trail", type: "line", data: data.active_sl_tp_line || [], color: "#FFA500" }
                        ],
                        annotations: { yaxis: [], points: all_point_annotations }, // SL/TP levels as y-axis lines can be added if needed
                        colors: ["#FEB019", "#FFA500"]
                    };
                    
                    if (data.cooldown_active_bg) chartContainer.classList.add("cooldown-active");
                    else chartContainer.classList.remove("cooldown-active");


                    if (activeChart) activeChart.updateOptions(chartOptionsUpdate);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), chartOptionsUpdate); activeChart.render(); }
                } else {
                     const noDataOptions = { ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: \`\${data.pair_name || currentSelectedPairId} - No Data\` },
                        series: initialChartOptions.series.map(s => ({ ...s, data: [] }))
                    };
                    if (activeChart) activeChart.updateOptions(noDataOptions);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), noDataOptions); activeChart.render(); }
                    lastKnownDataTimestamp = data.last_updated_tv || null;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? \`Data (empty) @ \${new Date(lastKnownDataTimestamp).toLocaleTimeString()}\` : "No data";
                    document.getElementById("strategyInfoLabel").textContent = "Status: Data Kosong";
                    chartContainer.classList.remove("cooldown-active");
                }
            } catch (error) {
                console.error("Error loading chart data:", error);
                if (activeChart) { activeChart.destroy(); activeChart = null; }
                document.getElementById("chart").innerHTML = `Error loading chart: ${error.message}`;
                 chartContainer.classList.remove("cooldown-active");
            } finally {
                isLoadingData = false;
            }
        }

        document.addEventListener("DOMContentLoaded", () => { /* ... (sama seperti sebelumnya, tapi autoRefresh lebih sering) ... */
            if (!activeChart) { 
                activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions);
                activeChart.render();
            }
            fetchAvailablePairs();
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) {
                    await loadChartDataForCurrentPair();
                }
            }, 5000); // Refresh every 5 seconds for faster updates of signals
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
        # Deepcopy is important here to avoid race conditions if the main loop modifies it
        pair_data_snapshot = copy.deepcopy(shared_crypto_data_manager.get(pair_id_from_request, {})) 
    
    if not pair_data_snapshot: 
        default_cfg = get_default_crypto_config()
        pair_name_default = f"{default_cfg['symbol']}-{default_cfg['currency']}"
        return jsonify({
            "ohlc":[], "active_sl_tp_line":[], "swing_high_points": [], "swing_low_points": [],
            "pair_name": pair_name_default, "last_updated_tv": None, 
            "strategy_state_info": get_initial_strategy_state(), "config_info": default_cfg,
            "annotations_points": []
        }), 200

    temp_manager = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask: log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
# CHART_INTEGRATION_END


# --- FUNGSI UTAMA TRADING LOOP (Dirombak untuk Exora V6) ---
def start_trading(global_settings_dict, shared_dm_ref, lock_ref):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key yang valid. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE); input(); return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE); input(); return

    animated_text_display("=========== EXORA V6 BOT (Python) START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    # ... (key info logging) ...

    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')} ({config.get('timeframe','DEF')})" # Lebih deskriptif
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        # Tentukan data minimal untuk warmup berdasarkan parameter
        min_len_for_indicators_init = max(config.get('rsi_len', 20) + 1, 
                                          config.get('stoch_k', 41) + config.get('stoch_smooth_k', 25) + config.get('stoch_d', 3),
                                          2 * config.get('swing_lookback', 100) + 1 
                                          ) + 50 # Buffer
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        log_info(f"Target data awal: {initial_candles_target} candles. Min untuk indikator: {min_len_for_indicators_init - 50}", pair_name=config['pair_name'])
        
        # ... (Logika fetch data awal dan warmup, mirip dengan skrip Anda)
        # Bagian warmup perlu memanggil run_strategy_logic secara iteratif
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
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']}. Pair ini akan dilewati.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts +1
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        if initial_candles:
            min_len_for_warmup_run = max(config.get('rsi_len', 20) + 1, # Minimal untuk RSI
                                         config.get('stoch_k', 41) + config.get('stoch_smooth_k', 25) + config.get('stoch_d', 3) -1, # Minimal untuk Stoch
                                         (2 * config.get('swing_lookback',100) +1 if config.get('use_swing_filter') else 1) # Minimal untuk Swing
                                        )

            if len(initial_candles) >= min_len_for_warmup_run:
                log_info(f"Warm-up: Memproses {len(initial_candles) - min_len_for_warmup_run +1} candle historis untuk {config['pair_name']}...", pair_name=config['pair_name'])
                for i_warmup in range(min_len_for_warmup_run -1, len(initial_candles)): # Proses semua, termasuk candle terakhir untuk state akhir
                    historical_slice = initial_candles[:i_warmup+1]
                    if len(historical_slice) < min_len_for_warmup_run: continue
                    
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy() # Gunakan state sebelumnya sebagai basis
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict, is_warmup=True
                    )
                log_info(f"{AnsiColors.CYAN}Warm-up state untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Tidak cukup data untuk warm-up ({len(initial_candles)}/{min_len_for_warmup_run}) untuk {config['pair_name']}", pair_name=config['pair_name'])


        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            # ... (Email notif Big Data jika perlu) ...
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
                    # Fetch lebih agresif saat big data (misal, per 1 menit jika timeframe < 1h, atau 5 menit jika >= 1h)
                    tf_is_minute = 'm' in config_for_pair.get('timeframe','1h')
                    required_interval = 60 if tf_is_minute else 300 
                
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch); continue
                
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                if data_per_pair["big_data_collection_phase_active"]: animated_text_display(f"\n--- BIG DATA {pair_name_for_log} ({num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD+AnsiColors.MAGENTA)
                else: animated_text_display(f"\n--- LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {num_candles_before_fetch} candles ---", color=AnsiColors.BOLD+AnsiColors.CYAN)

                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 # Untuk live, fetch beberapa candle terakhir untuk update
                if data_per_pair["big_data_collection_phase_active"]:
                    needed_for_big_data = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed_for_big_data <=0 : 
                        fetch_update_successful = True # Sudah cukup, tak perlu fetch lagi untuk big data
                        limit_fetch_update = 3 # Tetap fetch sedikit untuk update bar terakhir
                    else: 
                        limit_fetch_update = min(needed_for_big_data, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0: # Hanya fetch jika ada yang dibutuhkan atau untuk update live
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    original_api_key_index_for_this_update = api_key_manager.get_current_key_index()

                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        # ... (logika fetch data mirip dengan skrip Anda, menggunakan api_key_manager) ...
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE: Semua API key habis (global) untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        
                        log_info(f"UPDATE: Mencoba fetch {limit_fetch_update} candle untuk {pair_name_for_log} dengan key index {api_key_manager.get_current_key_index()} (Attempt {update_key_attempts_done + 1}/{max_update_key_attempts})", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_update, config_for_pair['timeframe'], pair_name=pair_name_for_log)
                            fetch_update_successful = True # Fetch attempt was made
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
                else: # limit_fetch_update was 0 (big data complete and no live update needed conceptually, but we still process logic)
                    fetch_update_successful = True # No fetch needed, so "successful" in that sense
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) +1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 

                # if not fetch_update_successful: # Ini berarti fetch gagal (bukan karena limit_fetch_update=0)
                if not fetch_update_successful and limit_fetch_update > 0 : # Hanya log error jika memang ada upaya fetch
                     log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                     min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                     with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair) # Simpan state kegagalan
                     continue
                
                # Merge candles if new ones were fetched
                if new_candles_batch:
                    merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                    newly_added_count_this_batch, updated_count_this_batch = 0,0
                    for candle in new_candles_batch:
                        ts = candle['timestamp']
                        if ts not in merged_candles_dict: merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                        elif merged_candles_dict[ts]['close'] != candle['close'] or merged_candles_dict[ts]['high'] != candle['high'] : # Check if candle actually changed
                            merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                    data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                    if newly_added_count_this_batch + updated_count_this_batch > 0: 
                        log_info(f"{newly_added_count_this_batch} candle baru, {updated_count_this_batch} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                
                # Update Big Data phase status
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        # ... (Email notif Big Data) ...
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 200: # Trim if significantly over
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 100):]


                # Run strategy logic
                # Tentukan apakah perlu menjalankan logika (misal, jika ada candle baru, atau baru selesai big data)
                # Untuk Exora, karena stateful (cooldown, in_position), jalankan setiap siklus setelah data cukup.
                min_len_for_logic_run_live = max(config_for_pair.get('rsi_len', 20) + 1,
                                                 config_for_pair.get('stoch_k', 41) + config_for_pair.get('stoch_smooth_k', 25) + config_for_pair.get('stoch_d', 3) -1,
                                                 (2 * config_for_pair.get('swing_lookback',100) +1 if config_for_pair.get('use_swing_filter') else 1)
                                                )

                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run_live:
                    log_debug(f"Menjalankan logika Exora V6 untuk {pair_name_for_log}...", pair_name=pair_name_for_log)
                    data_per_pair["strategy_state"] = run_strategy_logic(
                         data_per_pair["all_candles_list"], 
                         config_for_pair, 
                         data_per_pair["strategy_state"], 
                         global_settings_dict,
                         is_warmup=False # Live processing
                    )
                else:
                    log_debug(f"Belum cukup data ({len(data_per_pair['all_candles_list'])}/{min_len_for_logic_run_live}) untuk menjalankan logika {pair_name_for_log}", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            # Sleep logic (mirip dengan skrip Anda)
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 10 # Lebih cepat saat big data
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(1, int(min_overall_next_refresh_seconds)) # Minimal 1 detik
            
            if sleep_duration > 0 : show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s ({time.strftime('%H:%M:%S')})...")
            else: time.sleep(0.1) # Minimal sleep jika interval sangat kecil

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan.{AnsiColors.ENDC}",color=AnsiColors.ORANGE)
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EXORA V6 BOT (Python) STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali ke menu utama...")


# --- MENU UTAMA (Dirombak untuk Exora V6) ---
def main_menu():
    settings = load_settings()
    is_flask_running = any(t.name == "FlaskServerThread" for t in threading.enumerate())
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThread")
        flask_thread.start()
    else:
        log_info("Flask server sudah berjalan di thread lain.", "SYSTEM_CHART")

    while True:
        clear_screen_animated()
        animated_text_display("========= Exora V6 Spot Bot (Python) =========", color=AnsiColors.HEADER)
        pick_title_main = ""
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs: pick_title_main += f"--- Crypto Aktif ({len(active_cfgs)}) ---\n" + "".join([f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')})\n" for i,c in enumerate(active_cfgs)]) # Deskripsi singkat
        else: pick_title_main += "Tidak ada konfigurasi crypto aktif.\n"
        
        # ... (info API Key) ...
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5001\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        try:
            # selected_option, main_idx = pick(main_opts, pick_title_main, indicator='=>', options_map_func=lambda opt_str: opt_str[:50])
            options_for_pick_main = [opt[:70] + ('...' if len(opt) > 70 else '') for opt in main_opts]
            _, main_idx = pick(options_for_pick_main, pick_title_main, indicator='=>')
        except Exception as e_pick_main: # Fallback
            log_warning(f"Pick library error: {e_pick_main}. Gunakan input angka.")
            print(pick_title_main)
            for i, opt_disp_main in enumerate(options_for_pick_main): print(f"{i}. {opt_disp_main}")
            try:
                main_idx = int(input("Masukkan nomor pilihan: "))
                if not (0 <= main_idx < len(options_for_pick_main)): raise ValueError("Diluar range")
            except ValueError:
                print("Input tidak valid."); time.sleep(1); continue

        if main_idx == 0: 
            settings = load_settings() 
            start_trading(settings, shared_crypto_data_manager, shared_data_lock)
        elif main_idx == 1: settings = settings_menu(settings)
        elif main_idx == 2: log_info("Aplikasi ditutup."); break
    animated_text_display("Terima kasih!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan.", color=AnsiColors.ORANGE)
    except Exception as e_global: clear_screen_animated(); print(f"{AnsiColors.RED}ERROR KRITIKAL: {e_global}{AnsiColors.ENDC}"); log_exception("MAIN ERROR:",pair_name="SYS_CRIT"); input("Enter untuk keluar...")
    finally: sys.stdout.flush(); sys.stderr.flush()
