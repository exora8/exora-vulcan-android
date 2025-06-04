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
import math

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
    LOW_LIQ_BG_CONSOLE = '\033[44m' # Biru untuk low liq
    HIGH_LIQ_BG_CONSOLE = '\033[47m\033[30m' # Putih BG, Hitam Teks untuk high liq (mirip chart)

# --- ANIMATION HELPER FUNCTIONS --- (Sama seperti sebelumnya)
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
logger = logging.getLogger("ExoraVulcanFutures") # Nama logger baru
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

log_file_name = "trading_log_exora_vulcan_futures.txt" # Nama file log baru
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


SETTINGS_FILE = "settings_exora_vulcan_futures.json" # Nama file settings BARU
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 200 # Kebutuhan data awal untuk analisa likuiditas mungkin tidak sebesar strategi trading kompleks
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15


# --- FUNGSI CLEAR SCREEN --- (Sama)
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER --- (Sama)
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
                subject = "Exora Vulcan Futures: API Key Switched"
                body = f"Bot beralih ke API key berikutnya (Index {self.current_index}). Key lama mungkin bermasalah."
                send_email_notification(subject, body, self.global_email_settings, pair_name_ctx_override="GLOBAL_SYSTEM")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                subject = "Exora Vulcan Futures: SEMUA API KEY GAGAL"
                body = "Semua API key (primary dan recovery) telah gagal atau habis. Bot tidak dapat mengambil data."
                send_email_notification(subject, body, self.global_email_settings, pair_name_ctx_override="GLOBAL_SYSTEM")
            return None

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION --- (Sama, hanya subjek email disesuaikan)
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
    # Tambahkan prefix "Exora Vulcan Futures" ke subjek jika belum ada
    if "Exora Vulcan Futures" not in subject:
        subject = f"Exora Vulcan Futures: {subject}"

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
    
    # Tambahkan prefix "Exora Vulcan" ke judul jika belum ada
    if "Exora Vulcan" not in title and pair_name_for_log != "SYSTEM_CHART" : # Hindari prefix ganda atau untuk notif sistem chart
        title = f"Exora Vulcan: {title}"

    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg, '--priority', 'max', '--sound', '-id', f'exora_vulcan_{pair_name_for_log.replace("/", "_")}_{str(uuid.uuid4())[:4]}'],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10) # ID unik untuk notifikasi
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired:
        log_warning(f"{AnsiColors.ORANGE}Timeout saat mengirim notifikasi Termux untuk '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)


# --- FUNGSI PENGATURAN (Dirombak untuk Parameter Deteksi Likuiditas) ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True,
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG",
        "timeframe": "1m",
        "refresh_interval_seconds": 60,

        # Pengaturan Detektor Likuiditas (grp_liq_detect)
        "enable_lowliq_detector": True,
        "lowliq_pattern_length": 4, # Sesuai PineScript: lowliq_len
        "enable_highliq_detector": True,
        "highliq_pattern_length": 5, # Sesuai PineScript: highliq_len
        "highliq_lookback_confirmation_bars": 3, # Sesuai PineScript

        # Pengaturan Alert
        "alert_on_liquidity_state_change": True,
        "alert_on_high_liquidity_confirmed": False, # Default false agar tidak terlalu banyak notif
        
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
                for key, default_value in default_crypto_template.items():
                    if key not in crypto_cfg: crypto_cfg[key] = default_value
                
                # Hapus key lama dari strategi Exora V6 jika ada
                keys_to_remove_old_strat = [
                    "rsi_len", "rsi_extreme_oversold", "rsi_extreme_overbought",
                    "stoch_k", "stoch_smooth_k", "stoch_d", "stoch_extreme_oversold", "stoch_extreme_overbought",
                    "use_swing_filter", "swing_lookback", "avoid_resistance_proximity_percent",
                    "use_dump_cooldown", "dump_threshold_percent", "cooldown_period_after_dump_bars",
                    "use_fixed_sl", "sl_percent", "use_standard_tp", "standard_tp_percent",
                    "use_new_trailing_tp", "trailing_step_percent", "trailing_gap_percent"
                ]
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
            return typed_val
        except ValueError as e_val:
            print(f"{AnsiColors.RED}Input tidak valid: {e_val}. Harap masukkan tipe {target_type.__name__}.{AnsiColors.ENDC}")


def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy()
    def_cfg = get_default_crypto_config()

    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol',def_cfg['symbol'])}-{new_config.get('currency',def_cfg['currency'])}) ---", color=AnsiColors.HEADER)
    new_config["enabled"] = _prompt_type("Aktifkan pair ini?", new_config.get('enabled', def_cfg['enabled']), bool, def_cfg['enabled'])
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar [{new_config.get('symbol',def_cfg['symbol'])}]: {AnsiColors.ENDC}") or new_config.get('symbol',def_cfg['symbol'])).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote [{new_config.get('currency',def_cfg['currency'])}]: {AnsiColors.ENDC}") or new_config.get('currency',def_cfg['currency'])).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange [{new_config.get('exchange',def_cfg['exchange'])}]: {AnsiColors.ENDC}") or new_config.get('exchange',def_cfg['exchange'])).strip()
    
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d) [{new_config.get('timeframe',def_cfg['timeframe'])}]: {AnsiColors.ENDC}") or new_config.get('timeframe',def_cfg['timeframe'])).lower().strip()
    valid_tf_keys = ["1m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "1d", "3d", "1w"]
    if tf_input in valid_tf_keys: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default/sebelumnya.{AnsiColors.ENDC}");

    new_config["refresh_interval_seconds"] = _prompt_type("Interval Refresh (detik)", new_config.get('refresh_interval_seconds',def_cfg['refresh_interval_seconds']), int, def_cfg['refresh_interval_seconds'], min_val=MIN_REFRESH_INTERVAL_AFTER_BIG_DATA)

    animated_text_display("\n-- Pengaturan Detektor Likuiditas --", color=AnsiColors.HEADER)
    new_config["enable_lowliq_detector"] = _prompt_type("Aktifkan Deteksi Low Liquidity?", new_config.get('enable_lowliq_detector', def_cfg['enable_lowliq_detector']), bool, def_cfg['enable_lowliq_detector'])
    new_config["lowliq_pattern_length"] = _prompt_type("Panjang Pola Low Liquidity (candles)", new_config.get('lowliq_pattern_length', def_cfg['lowliq_pattern_length']), int, def_cfg['lowliq_pattern_length'], min_val=2)
    new_config["enable_highliq_detector"] = _prompt_type("Aktifkan Deteksi High Liquidity?", new_config.get('enable_highliq_detector', def_cfg['enable_highliq_detector']), bool, def_cfg['enable_highliq_detector'])
    new_config["highliq_pattern_length"] = _prompt_type("Panjang Pola High Liquidity (candles)", new_config.get('highliq_pattern_length', def_cfg['highliq_pattern_length']), int, def_cfg['highliq_pattern_length'], min_val=2)
    new_config["highliq_lookback_confirmation_bars"] = _prompt_type("Lookback Konfirmasi High Liquidity (bars)", new_config.get('highliq_lookback_confirmation_bars', def_cfg['highliq_lookback_confirmation_bars']), int, def_cfg['highliq_lookback_confirmation_bars'], min_val=1)

    animated_text_display("\n-- Pengaturan Notifikasi Likuiditas --", color=AnsiColors.HEADER)
    new_config["alert_on_liquidity_state_change"] = _prompt_type("Notifikasi saat State Likuiditas Berubah?", new_config.get('alert_on_liquidity_state_change', def_cfg['alert_on_liquidity_state_change']), bool, def_cfg['alert_on_liquidity_state_change'])
    new_config["alert_on_high_liquidity_confirmed"] = _prompt_type("Notifikasi saat High Liquidity Terkonfirmasi?", new_config.get('alert_on_high_liquidity_confirmed', def_cfg['alert_on_high_liquidity_confirmed']), bool, def_cfg['alert_on_high_liquidity_confirmed'])

    animated_text_display("\n-- Notifikasi Email (Gmail) - Untuk Alert & Sistem --", color=AnsiColors.HEADER)
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
        title = f"--- Menu Pengaturan Exora Vulcan Futures ---\nAPI Key: {pkd} | Recovery: {nrk} | Termux: {tns}\nStrategi: Deteksi Perubahan Likuiditas\nCrypto Pairs:\n"
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg in enumerate(current_settings["cryptos"]): title += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} ({cfg.get('timeframe','?')}) - {'Aktif' if cfg.get('enabled',True) else 'Nonaktif'}\n"
        title += "----------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux", "Tambah Crypto Pair", "Ubah Crypto Pair", "Hapus Crypto Pair", "Kembali"]
        
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
            if action_idx == 0: # Primary API Key
                new_pk = input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip()
                if new_pk: api_s["primary_key"] = new_pk
                elif not api_s.get('primary_key'): api_s["primary_key"] = "YOUR_PRIMARY_KEY" 
            elif action_idx == 1: # Recovery API Keys (Logic sama)
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
                            animated_text_display("Recovery key dihapus.", color=AnsiColors.GREEN)
                    elif rec_action_idx == 2: break 
                    show_spinner(0.5, "Memproses...")
            elif action_idx == 2: # Email Global Notif Sistem (Logic sama)
                api_s['enable_global_email_notifications_for_key_switch'] = _prompt_type("Aktifkan Email Notif Sistem Global?", api_s.get('enable_global_email_notifications_for_key_switch',False), bool, False)
                api_s['email_sender_address'] = (input(f"Alamat Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Email Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Alamat Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
            elif action_idx == 3: # Notifikasi Termux (Logic sama)
                 api_s['enable_termux_notifications'] = _prompt_type("Aktifkan Notifikasi Termux Global?", api_s.get('enable_termux_notifications',False), bool, False)
            elif action_idx == 4: current_settings.setdefault("cryptos", []).append(_prompt_crypto_config(get_default_crypto_config()))
            elif action_idx == 5: # Ubah
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk diubah."); show_spinner(1,""); continue
                edit_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe', '?')})" for c in current_settings["cryptos"]] + ["Batal"]
                _, edit_c_idx = pick(edit_opts, "Pilih pair untuk diubah:")
                if edit_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"][edit_c_idx] = _prompt_crypto_config(current_settings["cryptos"][edit_c_idx])
            elif action_idx == 6: # Hapus
                if not current_settings.get("cryptos"): print("Tidak ada pair untuk dihapus."); show_spinner(1,""); continue
                del_opts = [f"{c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe', '?')})" for c in current_settings["cryptos"]] + ["Batal"]
                _, del_c_idx = pick(del_opts, "Pilih pair untuk dihapus:")
                if del_c_idx < len(current_settings["cryptos"]): current_settings["cryptos"].pop(del_c_idx)
            elif action_idx == 7: break # Kembali
            
            current_settings["api_settings"] = api_s 
            save_settings(current_settings)
            if action_idx not in [1,7]: show_spinner(1, "Disimpan...")
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA --- (Sama, hanya timeframe mapping lebih robust)
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe_str="1h", pair_name="N/A"):
    timeframe_details = {"endpoint_segment": "histohour", "aggregate": 1}
    tf_lower = timeframe_str.lower()

    try:
        num_part = int("".join(filter(str.isdigit, tf_lower))) if any(char.isdigit() for char in tf_lower) else 1
        if 'm' in tf_lower:
            timeframe_details["endpoint_segment"] = "histominute"
            timeframe_details["aggregate"] = num_part
        elif 'h' in tf_lower:
            timeframe_details["endpoint_segment"] = "histohour"
            timeframe_details["aggregate"] = num_part
        elif 'd' in tf_lower:
            timeframe_details["endpoint_segment"] = "histoday"
            timeframe_details["aggregate"] = num_part
        elif 'w' in tf_lower:
            timeframe_details["endpoint_segment"] = "histoday"
            timeframe_details["aggregate"] = 7 * num_part
        else: # Default or unknown, try histohour
            log_warning(f"Timeframe '{timeframe_str}' tidak dikenali, menggunakan 1 hour.", pair_name=pair_name)
            timeframe_details["endpoint_segment"] = "histohour"
            timeframe_details["aggregate"] = 1
    except ValueError:
        log_warning(f"Error parsing timeframe '{timeframe_str}', menggunakan 1 hour.", pair_name=pair_name)
        timeframe_details["endpoint_segment"] = "histohour"
        timeframe_details["aggregate"] = 1


    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = timeframe_details["endpoint_segment"]
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    
    is_large_fetch = total_limit_desired > 20
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')

    retries_network_error = 3
    current_network_retry = 0

    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        if timeframe_details["aggregate"] > 1: # Apply aggregate if not 1
            params["aggregate"] = timeframe_details["aggregate"]
        
        # Aggregate Togethertots = false is usually better for exact candle counts when limit is applied
        # However, for simplicity and standard CryptoCompare behavior, we'll omit it unless issues arise.
        # params["agggregate"] = timeframe_details["aggregate"]
        # params["aggregatePredictableTimePeriods"] = True # New parameter for more consistent time periods

        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts: params["toTs"] = current_to_ts
        
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429, 400]: # 400 can also be related to key or request limits
                err_msg_detail = ""
                try: err_msg_detail = response.json().get('Message', f"HTTP Error {response.status_code}")
                except: err_msg_detail = f"HTTP Error {response.status_code}, no JSON body."
                
                # Log API key part for debugging
                key_display_for_log = current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else "KEY_SHORT"
                
                log_warning(f"API Key/Request Error (HTTP {response.status_code}): {err_msg_detail} | Key ends: ...{key_display_for_log}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {err_msg_detail}")
            response.raise_for_status() 
            data = response.json()

            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error')
                key_err_patterns = ["api key is invalid", "apikey_is_missing", "rate limit", "monthly_calls", "tier", "not valid"]
                if any(p.lower() in err_msg.lower() for p in key_err_patterns):
                    key_display_for_log = current_api_key_to_use[-5:] if current_api_key_to_use and len(current_api_key_to_use) > 5 else "KEY_SHORT"
                    log_warning(f"API Key Error (JSON): {err_msg} | Key ends: ...{key_display_for_log}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {err_msg}")
                else: log_error(f"API Error: {err_msg}", pair_name=pair_name); break 
            
            raw_candles = data.get('Data', {}).get('Data', [])
            if not raw_candles:
                if len(all_accumulated_candles) > 0 : log_debug(f"Tidak ada candle baru dari API (mungkin akhir histori atau toTs terlalu baru). Total: {len(all_accumulated_candles)}", pair_name=pair_name)
                else: log_warning(f"Tidak ada data candle sama sekali dari API untuk {pair_name}.", pair_name=pair_name)
                break

            batch = []
            for item in raw_candles:
                if all(k in item and item[k] is not None for k in ['time', 'open', 'high', 'low', 'close', 'volumefrom']):
                    batch.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            
            if current_to_ts and all_accumulated_candles and batch and batch[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']: batch.pop() 
            if not batch and current_to_ts: break 

            all_accumulated_candles = batch + all_accumulated_candles
            if raw_candles: current_to_ts = raw_candles[0]['time'] # Fetch before this timestamp next
            else: break

            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')
            if len(raw_candles) < limit_call: break # API returned fewer than requested, assume end of data for this segment
            if len(all_accumulated_candles) >= total_limit_desired: break
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.2) # Small delay for large fetches
        
        except APIKeyError: raise # Re-raise to be caught by caller for key switching
        except requests.exceptions.RequestException as e_req:
            current_network_retry += 1
            if current_network_retry <= retries_network_error:
                log_warning(f"Kesalahan Jaringan: {e_req}. Retry {current_network_retry}/{retries_network_error} dalam beberapa detik...", pair_name=pair_name)
                time.sleep(current_network_retry * 2) # Exponential backoff-like delay
                continue # Retry the while loop iteration
            else:
                log_error(f"Kesalahan Jaringan ({e_req}) setelah {retries_network_error} retries. Gagal fetch.", pair_name=pair_name); break 
        except Exception as e_gen: log_exception(f"Error lain dalam fetch_candles: {e_gen}", pair_name=pair_name); break
    
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai')
    return all_accumulated_candles


# --- LOGIKA DETEKSI LIKUIDITAS (BARU) ---
def get_initial_liquidity_analysis_state():
    return {
        "current_liquidity_state": 0,  # 0: Undetermined, 1: Low, 2: High
        "previous_liquidity_state": 0, # Untuk deteksi perubahan
        "high_liq_consecutive_bars": 0,
        "is_high_liquidity_confirmed_by_lookback": False,
        "last_alerted_state_change_to": None, # Menyimpan state tujuan dari alert terakhir
        "last_alerted_high_liq_confirmed_at_ts": None, # Timestamp konfirmasi high liq terakhir
        # Untuk chart
        "liquidity_bg_color_for_chart": "na" # 'na', 'low', 'high'
    }

def is_bullish_py(candle):
    return candle and candle['close'] > candle['open']

def is_bearish_py(candle):
    return candle and candle['close'] < candle['open']

def check_low_liquidity_pattern_py(candles_history, length):
    if not candles_history or len(candles_history) < length:
        return False

    relevant_candles = candles_history[-length:] # Ambil `length` candle terakhir
    
    # Pola 1: Bullish, Bearish, Bullish, ... (candle terakhir adalah index length-1 dari relevant_candles)
    # Pine: i=0 (candle saat ini/terbaru), i=len-1 (candle terlama)
    # Python relevant_candles: index 0 (terlama), index length-1 (terbaru)
    # Kita periksa dari candle terlama ke terbaru dalam slice
    p1_match = True
    for i in range(length):
        candle_idx_in_slice = i # Iterasi dari terlama ke terbaru
        # Pine (i % 2 == 0 is bullish) -> candle terlama (i=len-1) genap jika len ganjil, ganjil jika len genap
        # Pine: (i % 2 == 0 and not is_bullish(i)) or (i % 2 != 0 and not is_bearish(i))
        # Maksudnya: Jika index ke-i (dari terbaru) adalah genap, harus bullish. Jika ganjil, harus bearish.
        # Di Python, kita iterate dari terlama (idx 0 di slice).
        # Pola Bullish, Bearish, Bullish, Bearish ...
        # Slice[0] (terlama) = Bullish, Slice[1] = Bearish, Slice[2] = Bullish
        if (i % 2 == 0 and not is_bullish_py(relevant_candles[i])) or \
           (i % 2 != 0 and not is_bearish_py(relevant_candles[i])):
            p1_match = False
            break
    if p1_match: return True

    # Pola 2: Bearish, Bullish, Bearish, ...
    # Slice[0] (terlama) = Bearish, Slice[1] = Bullish, Slice[2] = Bearish
    p2_match = True
    for i in range(length):
        if (i % 2 == 0 and not is_bearish_py(relevant_candles[i])) or \
           (i % 2 != 0 and not is_bullish_py(relevant_candles[i])):
            p2_match = False
            break
    if p2_match: return True
            
    return False


def check_high_liquidity_pattern_py(candles_history, length):
    if not candles_history or len(candles_history) < length:
        return False
    
    relevant_candles = candles_history[-length:]
    
    all_bullish = True
    for candle in relevant_candles:
        if not is_bullish_py(candle):
            all_bullish = False
            break
    if all_bullish: return True
        
    all_bearish = True
    for candle in relevant_candles:
        if not is_bearish_py(candle):
            all_bearish = False
            break
    if all_bearish: return True
            
    return False

def analyze_liquidity_and_alert(candles_history, crypto_config, current_analysis_state, global_settings, is_warmup=False):
    pair_name = crypto_config['pair_name']
    cfg = crypto_config 
    
    if not candles_history or len(candles_history) < max(cfg.get('lowliq_pattern_length',2), cfg.get('highliq_pattern_length',2)):
        log_debug(f"Tidak cukup data candle untuk analisis likuiditas {pair_name}.", pair_name=pair_name)
        # Pastikan state chart direset jika tidak ada data
        current_analysis_state["liquidity_bg_color_for_chart"] = "na"
        return current_analysis_state

    current_candle_time = candles_history[-1]['timestamp']
    
    # Simpan state sebelumnya untuk deteksi perubahan
    current_analysis_state["previous_liquidity_state"] = current_analysis_state["current_liquidity_state"]
    
    # Trigger deteksi
    low_liq_trigger = False
    if cfg.get('enable_lowliq_detector', True) and cfg.get('lowliq_pattern_length', 0) > 1:
        low_liq_trigger = check_low_liquidity_pattern_py(candles_history, cfg['lowliq_pattern_length'])

    high_liq_trigger = False
    if cfg.get('enable_highliq_detector', True) and cfg.get('highliq_pattern_length', 0) > 1:
        high_liq_trigger = check_high_liquidity_pattern_py(candles_history, cfg['highliq_pattern_length'])

    # State Management (mengikuti logika PineScript)
    current_state = current_analysis_state["current_liquidity_state"]
    
    if current_state == 0: # Undetermined
        if high_liq_trigger:
            current_analysis_state["current_liquidity_state"] = 2 # High
        elif low_liq_trigger:
            current_analysis_state["current_liquidity_state"] = 1 # Low
    elif current_state == 1: # Currently Low
        if high_liq_trigger:
            current_analysis_state["current_liquidity_state"] = 2 # Switch to High
        # Tidak ada transisi Low -> Low atau Low -> Undetermined di PineScript ini, state tetap Low jika tidak ada trigger High
    elif current_state == 2: # Currently High
        if low_liq_trigger:
            current_analysis_state["current_liquidity_state"] = 1 # Switch to Low
        # Tidak ada transisi High -> High atau High -> Undetermined, state tetap High jika tidak ada trigger Low

    # Update High Liquidity Consecutive Bars & Confirmation
    if current_analysis_state["current_liquidity_state"] == 2:
        current_analysis_state["high_liq_consecutive_bars"] += 1
    else:
        current_analysis_state["high_liq_consecutive_bars"] = 0
        current_analysis_state["is_high_liquidity_confirmed_by_lookback"] = False # Reset konfirmasi jika tidak lagi high

    newly_confirmed_high_liq = False
    if cfg.get('enable_highliq_detector', True) and current_analysis_state["current_liquidity_state"] == 2 and \
       current_analysis_state["high_liq_consecutive_bars"] >= cfg.get('highliq_lookback_confirmation_bars', 1):
        if not current_analysis_state["is_high_liquidity_confirmed_by_lookback"]: # Baru terkonfirmasi
            newly_confirmed_high_liq = True
        current_analysis_state["is_high_liquidity_confirmed_by_lookback"] = True
    else: # Jika kondisi konfirmasi tidak terpenuhi (mis. state bukan 2, atau bars < lookback)
        current_analysis_state["is_high_liquidity_confirmed_by_lookback"] = False


    # --- ALERTING ---
    if not is_warmup:
        new_state = current_analysis_state["current_liquidity_state"]
        prev_state = current_analysis_state["previous_liquidity_state"]
        
        state_map = {0: "Undetermined", 1: "Low Liquidity", 2: "High Liquidity"}
        alert_title_prefix = f"Liq {pair_name}"

        # 1. Alert on State Change
        if cfg.get("alert_on_liquidity_state_change", True) and new_state != prev_state:
            # Hindari alert duplikat jika state bolak-balik dengan cepat sebelum alert sempat di-reset
            # if current_analysis_state.get("last_alerted_state_change_to") != new_state: # Ini mungkin terlalu ketat
            
            msg = f"STATE CHANGE: {state_map[prev_state]} -> {state_map[new_state]} on {pair_name} at {current_candle_time.strftime('%H:%M:%S')}."
            log_color = AnsiColors.ORANGE
            if new_state == 1: log_color = AnsiColors.LOW_LIQ_BG_CONSOLE # Biru
            elif new_state == 2: log_color = AnsiColors.HIGH_LIQ_BG_CONSOLE # Putih

            log_info(f"{log_color}{msg}{AnsiColors.ENDC}", pair_name=pair_name)
            
            termux_title = f"{alert_title_prefix}: {state_map[prev_state]} -> {state_map[new_state]}"
            termux_content = f"Pair: {pair_name}. New state: {state_map[new_state]} at {current_candle_time.strftime('%H:%M:%S')}."
            send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)
            
            if crypto_config.get("enable_email_notifications", False):
                email_subject = f"Liquidity State Change: {pair_name} to {state_map[new_state]}"
                send_email_notification(email_subject, msg, crypto_config, pair_name_ctx_override=pair_name)
            
            current_analysis_state["last_alerted_state_change_to"] = new_state # Catat alert terakhir
            play_notification_sound()


        # 2. Alert on High Liquidity Confirmed
        if cfg.get("alert_on_high_liquidity_confirmed", False) and newly_confirmed_high_liq:
            # Hanya alert jika timestamp konfirmasi terakhir berbeda (menghindari spam jika state tetap confirmed)
            if current_analysis_state.get("last_alerted_high_liq_confirmed_at_ts") != current_candle_time:
                msg = f"CONFIRMED HIGH LIQUIDITY for {pair_name} ({current_analysis_state['high_liq_consecutive_bars']} bars) at {current_candle_time.strftime('%H:%M:%S')}."
                log_info(f"{AnsiColors.HIGH_LIQ_BG_CONSOLE}{AnsiColors.BOLD}{msg}{AnsiColors.ENDC}", pair_name=pair_name)

                termux_title = f"{alert_title_prefix}: High Liq Confirmed"
                termux_content = f"{pair_name} entered Confirmed High Liquidity ({current_analysis_state['high_liq_consecutive_bars']} bars) at {current_candle_time.strftime('%H:%M:%S')}."
                send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)

                if crypto_config.get("enable_email_notifications", False):
                    email_subject = f"Confirmed High Liquidity: {pair_name}"
                    send_email_notification(email_subject, msg, crypto_config, pair_name_ctx_override=pair_name)
                
                current_analysis_state["last_alerted_high_liq_confirmed_at_ts"] = current_candle_time
                play_notification_sound()

    # Update chart state
    if current_analysis_state["current_liquidity_state"] == 1 and cfg.get('enable_lowliq_detector', True):
        current_analysis_state["liquidity_bg_color_for_chart"] = "low"
    elif current_analysis_state["current_liquidity_state"] == 2 and cfg.get('enable_highliq_detector', True):
        current_analysis_state["liquidity_bg_color_for_chart"] = "high"
    else:
        current_analysis_state["liquidity_bg_color_for_chart"] = "na"
        
    return current_analysis_state


# CHART_INTEGRATION_START & Flask Endpoints (Modified for Liquidity Analysis Data)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()

def prepare_chart_data_for_pair(pair_id, snapshot):
    if pair_id not in snapshot: return None
    data = snapshot[pair_id]
    # Untuk chart, tampilkan lebih banyak histori jika ada, misal 500, tapi kalkulasi utama hanya butuh TARGET_BIG_DATA_CANDLES
    chart_display_candles = 500 
    hist = data.get("all_candles_list", [])[-chart_display_candles:] 
    cfg = data.get("config", {})
    state = data.get("liquidity_analysis_state", {}) # Ganti nama state
    
    ohlc_data = []
    pair_display_name = cfg.get('pair_name', pair_id)

    default_state_for_chart = get_initial_liquidity_analysis_state()

    if not hist:
        return {
            "ohlc": [], 
            "pair_name": pair_display_name, "last_updated_tv": None,
            "liquidity_analysis_info": state or default_state_for_chart, # Kirim state analisis
            "config_info": cfg,
            "current_liquidity_state_for_bg": (state or default_state_for_chart).get("liquidity_bg_color_for_chart", "na")
        }

    for i, c in enumerate(hist):
        if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
            ts_ms = c['timestamp'].timestamp() * 1000
            ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})
    
    return {
        "ohlc": ohlc_data,
        "pair_name": pair_display_name,
        "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
        "liquidity_analysis_info": state or default_state_for_chart,
        "config_info": cfg,
        "current_liquidity_state_for_bg": (state or default_state_for_chart).get("liquidity_bg_color_for_chart", "na")
    }

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exora Vulcan Futures - Liquidity Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; width: 100%; max-width: 1200px; flex-wrap: wrap; }
        select, button { padding: 8px 12px; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor: pointer; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; transition: background-color 0.5s ease; }
        h1 { color: #00bcd4; margin-bottom: 10px; font-size: 1.5em; }
        #lastUpdatedLabel { font-size: .8em; color: #aaa; margin-left: auto; }
        #liquidityInfoLabel { font-size: .9em; color: #ffd700; margin-left: 10px; white-space: pre-wrap; max-width: 400px; }
        .liq-bg-low { background-color: rgba(38, 34, 171, 0.68) !important; /* #2622ab with alpha */ }
        .liq-bg-high { background-color: rgba(255, 255, 255, 0.20) !important; /* #ffffff80 with alpha, but for chart bg, make it less opaque */ }
        .liq-bg-na { background-color: #2a2a2a !important; /* Default chart bg */ }
    </style>
</head>
<body>
    <h1>Exora Vulcan Futures - Liquidity Analysis</h1>
    <div id="controls">
        <label for="pairSelector">Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh</button>
        <span id="liquidityInfoLabel">Status: -</span>
        <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container"><div id="chart"></div></div>
    <script>
        let activeChart, currentSelectedPairId = "", lastKnownDataTimestamp = null, autoRefreshIntervalId = null, isLoadingData = false;
        const initialChartOptions = {
            series: [{ name: "Candlestick", type: "candlestick", data: [] }],
            chart: { type: "candlestick", height: 550, background: "transparent", animations: { enabled: false }, toolbar: { show: true } }, // Transparent for container bg
            theme: { mode: "dark" },
            title: { text: "Memuat Data Pair...", align: "left", style: { color: "#e0e0e0" } },
            xaxis: { type: "datetime", labels: { style: { colors: "#aaa" } }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: "#aaa" }, formatter: v => v ? v.toFixed(5) : "" } },
            stroke: { width: 1, curve: "straight" }, 
            markers: { size: 0 },
            colors: ["#FEB019"], 
            grid: { borderColor: "#444" },
            tooltip: { theme: "dark", shared: true, intersect: false, y: { formatter: val => val ? val.toFixed(5) : val } },
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
                    document.getElementById("chart-container").className = 'liq-bg-na';
                }
            } catch (error) {
                console.error("Error fetching available pairs:", error);
                if (activeChart) { activeChart.destroy(); activeChart = null; }
                document.getElementById("chart").innerHTML = `Error loading pairs: ${error.message}`;
                document.getElementById("chart-container").className = 'liq-bg-na';
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
            const chartContainer = document.getElementById("chart-container");
            try {
                const response = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                if (data && data.ohlc) {
                    if (data.last_updated_tv && data.last_updated_tv === lastKnownDataTimestamp && !data.force_update_chart) {
                        console.log("Chart data is unchanged based on timestamp.");
                        document.getElementById("lastUpdatedLabel").textContent = `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                        // Update background and info even if ohlc is same
                        chartContainer.className = 'liq-bg-' + (data.current_liquidity_state_for_bg || 'na');
                        updateLiquidityInfoLabel(data.liquidity_analysis_info, data.config_info);
                        isLoadingData = false;
                        return;
                    }
                    lastKnownDataTimestamp = data.last_updated_tv;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? `Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Data Loaded";
                    
                    updateLiquidityInfoLabel(data.liquidity_analysis_info, data.config_info);
                    chartContainer.className = 'liq-bg-' + (data.current_liquidity_state_for_bg || 'na');
                    
                    const chartOptionsUpdate = {
                        title: { ...initialChartOptions.title, text: \`\${data.pair_name} - Exora Vulcan (\${data.config_info.timeframe})\` },
                        series: [{ name: "Candlestick", type: "candlestick", data: data.ohlc || [] }],
                    };
                    
                    if (activeChart) activeChart.updateOptions(chartOptionsUpdate);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions); activeChart.render(); activeChart.updateOptions(chartOptionsUpdate); } // Render then update for title etc.
                } else {
                     const noDataOptions = { ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: \`\${data.pair_name || currentSelectedPairId} - No Data\` },
                        series: initialChartOptions.series.map(s => ({ ...s, data: [] }))
                    };
                    if (activeChart) activeChart.updateOptions(noDataOptions);
                    else { activeChart = new ApexCharts(document.querySelector("#chart"), noDataOptions); activeChart.render(); }
                    lastKnownDataTimestamp = data.last_updated_tv || null;
                    document.getElementById("lastUpdatedLabel").textContent = lastKnownDataTimestamp ? \`Data (empty) @ \${new Date(lastKnownDataTimestamp).toLocaleTimeString()}\` : "No data";
                    document.getElementById("liquidityInfoLabel").textContent = "Status: Data Kosong";
                    chartContainer.className = 'liq-bg-na';
                }
            } catch (error) {
                console.error("Error loading chart data:", error);
                if (activeChart) { activeChart.destroy(); activeChart = null; }
                document.getElementById("chart").innerHTML = `Error loading chart: ${error.message}`;
                chartContainer.className = 'liq-bg-na';
            } finally {
                isLoadingData = false;
            }
        }
        
        function updateLiquidityInfoLabel(state, config) {
            const stateMap = {0: "Undetermined", 1: "Low Liquidity", 2: "High Liquidity"};
            let infoText = \`Current State: \${stateMap[state.current_liquidity_state] || 'N/A'}\\n\`;
            if (state.current_liquidity_state === 2) {
                infoText += \`High Liq Bars: \${state.high_liq_consecutive_bars} / \${config.highliq_lookback_confirmation_bars}\\n\`;
                infoText += \`Confirmed High: \${state.is_high_liquidity_confirmed_by_lookback ? 'YES' : 'NO'}\`;
            }
            document.getElementById("liquidityInfoLabel").textContent = infoText;
        }

        document.addEventListener("DOMContentLoaded", () => {
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
            }, 5000); // Refresh every 5 seconds
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
        default_cfg_chart = get_default_crypto_config()
        pair_name_default = f"{default_cfg_chart['symbol']}-{default_cfg_chart['currency']}"
        return jsonify({
            "ohlc":[], 
            "pair_name": pair_name_default, "last_updated_tv": None, 
            "liquidity_analysis_info": get_initial_liquidity_analysis_state(), 
            "config_info": default_cfg_chart,
            "current_liquidity_state_for_bg": "na"
        }), 200

    temp_manager = {pair_id_from_request: pair_data_snapshot}
    prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
    
    if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
    return jsonify(prepared_data)

def run_flask_server_thread():
    log_info("Memulai Flask server Exora Vulcan Futures di http://localhost:5001", pair_name="SYSTEM_CHART")
    try:
        logging.getLogger('werkzeug').setLevel(logging.ERROR) 
        flask_app_instance.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e_flask: log_error(f"Flask server gagal dijalankan: {e_flask}", pair_name="SYSTEM_CHART")
# CHART_INTEGRATION_END


# --- FUNGSI UTAMA ANALYSIS LOOP (Dirombak untuk Deteksi Likuiditas) ---
def start_analysis_loop(global_settings_dict, shared_dm_ref, lock_ref):
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

    animated_text_display("=========== EXORA VULCAN FUTURES (Python) START (Multi-Pair) ===========", color=AnsiColors.HEADER, delay=0.005)
    
    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')} ({config.get('timeframe','DEF')})"
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        
        local_crypto_data_manager[pair_id] = {
            "config": config, 
            "all_candles_list": [], 
            "liquidity_analysis_state": get_initial_liquidity_analysis_state(), # Ganti nama state
            "big_data_collection_phase_active": True, 
            "last_candle_fetch_time": datetime.min, 
            "data_fetch_failed_consecutively": 0,
            "last_attempt_after_all_keys_failed": datetime.min
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_analysis_init = max(config.get('lowliq_pattern_length', 2), 
                                        config.get('highliq_pattern_length', 2),
                                        config.get('highliq_lookback_confirmation_bars',1) # Technically not needed for first pattern, but good buffer
                                       ) + 5 # Buffer kecil
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_analysis_init)
        log_info(f"Target data awal: {initial_candles_target} candles. Min untuk analisis: {min_len_for_analysis_init - 5}", pair_name=config['pair_name'])
        
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
                    break # Keluar dari while loop fetch awal untuk pair ini
            except requests.exceptions.RequestException as e_req_init:
                log_error(f"BIG DATA: Error Jaringan saat fetch awal {config['pair_name']}: {e_req_init}. Tidak ganti key.", pair_name=config['pair_name'])
                break 
            except Exception as e_gen_init:
                log_exception(f"BIG DATA: Error Umum saat fetch awal {config['pair_name']}: {e_gen_init}. Tidak ganti key.", pair_name=config['pair_name'])
                break
            initial_key_attempts_done += 1

        if not initial_fetch_successful or not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']}. Pair ini akan dilewati sementara.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id]["data_fetch_failed_consecutively"] = max_initial_key_attempts + 1 # Tandai gagal parah
            local_crypto_data_manager[pair_id]["last_attempt_after_all_keys_failed"] = datetime.now()
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Anggap selesai (gagal)
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
            continue

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        # Warm-up: Jalankan analisis pada data historis untuk set state awal, TANPA alert
        if initial_candles:
            min_len_for_warmup_run = max(config.get('lowliq_pattern_length', 2), config.get('highliq_pattern_length', 2)) # Hanya butuh panjang pola
            if len(initial_candles) >= min_len_for_warmup_run:
                log_info(f"Warm-up: Memproses {len(initial_candles) - min_len_for_warmup_run +1} candle historis untuk {config['pair_name']}...", pair_name=config['pair_name'])
                for i_warmup in range(min_len_for_warmup_run -1, len(initial_candles)):
                    historical_slice = initial_candles[:i_warmup+1]
                    if len(historical_slice) < min_len_for_warmup_run: continue
                    
                    temp_state_for_warmup = local_crypto_data_manager[pair_id]["liquidity_analysis_state"].copy()
                    local_crypto_data_manager[pair_id]["liquidity_analysis_state"] = analyze_liquidity_and_alert(
                        historical_slice, config, temp_state_for_warmup, global_settings_dict, is_warmup=True
                    )
                log_info(f"{AnsiColors.CYAN}Warm-up state untuk {config['pair_name']} selesai. State: {local_crypto_data_manager[pair_id]['liquidity_analysis_state']['current_liquidity_state']}{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Tidak cukup data untuk warm-up ({len(initial_candles)}/{min_len_for_warmup_run}) untuk {config['pair_name']}", pair_name=config['pair_name'])

        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
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

                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) + 1: 
                    if (datetime.now() - data_per_pair.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Coba lagi setelah 1 jam jika semua key gagal
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600)
                        continue
                    else: # Reset counter setelah 1 jam agar mencoba lagi
                        data_per_pair["data_fetch_failed_consecutively"] = 0 
                        log_info(f"Mencoba lagi fetch data untuk {pair_name_for_log} setelah periode tunggu karena semua key gagal.", pair_name=pair_name_for_log)

                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data_per_pair["last_candle_fetch_time"]).total_seconds()
                required_interval = config_for_pair.get('refresh_interval_seconds', 60)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    tf_is_minute = 'm' in config_for_pair.get('timeframe','1h')
                    required_interval = 60 if tf_is_minute else 120 # Fetch lebih sering saat big data (1 atau 2 menit)
                
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch)
                    continue
                
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                if data_per_pair["big_data_collection_phase_active"]: animated_text_display(f"\n--- BIG DATA {pair_name_for_log} ({num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD+AnsiColors.MAGENTA)
                else: animated_text_display(f"\n--- LIVE {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S')}) | {num_candles_before_fetch} candles ---", color=AnsiColors.BOLD+AnsiColors.CYAN)

                new_candles_batch = []
                fetch_update_successful = False
                limit_fetch_update = 3 # Untuk live, fetch beberapa candle terakhir
                if data_per_pair["big_data_collection_phase_active"]:
                    needed_for_big_data = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed_for_big_data <=0 : 
                        fetch_update_successful = True 
                        limit_fetch_update = 3
                    else: 
                        limit_fetch_update = min(needed_for_big_data, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0:
                    max_update_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    update_key_attempts_done = 0
                    original_api_key_index_for_this_update = api_key_manager.get_current_key_index()

                    while update_key_attempts_done < max_update_key_attempts and not fetch_update_successful:
                        current_api_key_update = api_key_manager.get_current_key()
                        if not current_api_key_update:
                            log_error(f"UPDATE: Semua API key habis (global) untuk {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        
                        log_info(f"UPDATE: Mencoba fetch {limit_fetch_update} candle untuk {pair_name_for_log} dengan key index {api_key_manager.get_current_key_index()} (Attempt {update_key_attempts_done + 1}/{max_update_key_attempts})", pair_name=pair_name_for_log)
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
                else:
                    fetch_update_successful = True
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) +1:
                    data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 

                if not fetch_update_successful and limit_fetch_update > 0 :
                     log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name_for_log} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                     min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                     with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
                     continue
                
                if new_candles_batch:
                    merged_candles_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                    newly_added_count_this_batch, updated_count_this_batch = 0,0
                    for candle in new_candles_batch:
                        ts = candle['timestamp']
                        if ts not in merged_candles_dict: merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                        elif merged_candles_dict[ts]['close'] != candle['close'] or merged_candles_dict[ts]['high'] != candle['high'] or merged_candles_dict[ts]['low'] != candle['low'] or merged_candles_dict[ts]['open'] != candle['open'] : 
                            merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                    data_per_pair["all_candles_list"] = sorted(list(merged_candles_dict.values()), key=lambda c_sort: c_sort['timestamp'])
                    if newly_added_count_this_batch + updated_count_this_batch > 0: 
                        log_info(f"{newly_added_count_this_batch} candle baru, {updated_count_this_batch} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 200:
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 100):]

                min_len_for_logic_run_live = max(config_for_pair.get('lowliq_pattern_length', 2), config_for_pair.get('highliq_pattern_length', 2))
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_run_live:
                    log_debug(f"Menjalankan analisis likuiditas untuk {pair_name_for_log}...", pair_name=pair_name_for_log)
                    data_per_pair["liquidity_analysis_state"] = analyze_liquidity_and_alert(
                         data_per_pair["all_candles_list"], 
                         config_for_pair, 
                         data_per_pair["liquidity_analysis_state"], 
                         global_settings_dict,
                         is_warmup=False
                    )
                else:
                    log_debug(f"Belum cukup data ({len(data_per_pair['all_candles_list'])}/{min_len_for_logic_run_live}) untuk analisis {pair_name_for_log}", pair_name=pair_name_for_log)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair)
            
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: sleep_duration = 10
            elif min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                sleep_duration = max(1, int(min_overall_next_refresh_seconds))
            
            if sleep_duration > 0 : show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s ({time.strftime('%H:%M:%S')})...")
            else: time.sleep(0.1)

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan.{AnsiColors.ENDC}",color=AnsiColors.ORANGE)
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM_ANALYSIS_LOOP")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EXORA VULCAN FUTURES (Python) STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali ke menu utama...")


# --- MENU UTAMA (Dirombak untuk Exora Vulcan Futures) ---
def main_menu():
    settings = load_settings()
    # Start Flask server in a separate thread if not already running
    is_flask_running = any(t.name == "FlaskServerThreadVulcan" for t in threading.enumerate()) # Unique name
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThreadVulcan")
        flask_thread.start()
        time.sleep(1) # Give flask a moment to start
    else:
        log_info("Flask server Exora Vulcan Futures sudah berjalan di thread lain.", "SYSTEM_CHART")

    while True:
        clear_screen_animated()
        animated_text_display("========= Exora Vulcan Futures (Deteksi Likuiditas) =========", color=AnsiColors.HEADER)
        pick_title_main = ""
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        if active_cfgs: pick_title_main += f"--- Crypto Aktif ({len(active_cfgs)}) ---\n" + "".join([f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')})\n" for i,c in enumerate(active_cfgs)])
        else: pick_title_main += "Tidak ada konfigurasi crypto aktif.\n"
        
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5001\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        options_for_pick_main = [opt[:70] + ('...' if len(opt) > 70 else '') for opt in main_opts]
        try:
            _, main_idx = pick(options_for_pick_main, pick_title_main, indicator='=>')
        except Exception as e_pick_main:
            log_warning(f"Pick library error: {e_pick_main}. Gunakan input angka.")
            print(pick_title_main)
            for i, opt_disp_main in enumerate(options_for_pick_main): print(f"{i}. {opt_disp_main}")
            try:
                main_idx = int(input("Masukkan nomor pilihan: "))
                if not (0 <= main_idx < len(options_for_pick_main)): raise ValueError("Diluar range")
            except ValueError:
                print("Input tidak valid."); time.sleep(1); continue

        if main_idx == 0: 
            settings = load_settings() # Reload settings before starting
            start_analysis_loop(settings, shared_crypto_data_manager, shared_data_lock)
        elif main_idx == 1: settings = settings_menu(settings)
        elif main_idx == 2: log_info("Aplikasi Exora Vulcan Futures ditutup."); break
    animated_text_display("Terima kasih!", color=AnsiColors.MAGENTA); show_spinner(0.5, "Exiting")

if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display("Aplikasi dihentikan paksa.", color=AnsiColors.ORANGE)
    except Exception as e_global: 
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}ERROR KRITIKAL TIDAK TERDUGA:{AnsiColors.ENDC}")
        print(f"{AnsiColors.RED}{str(e_global)}{AnsiColors.ENDC}")
        log_exception("MAIN CRITICAL ERROR:",pair_name="SYS_CRIT")
        input("Tekan Enter untuk keluar...")
    finally: 
        sys.stdout.flush()
        sys.stderr.flush()
