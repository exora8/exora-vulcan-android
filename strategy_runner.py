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
    # sys.exit(1) # Komentari agar bisa jalan tanpa flask untuk fokus ke alert
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

log_file_name = "trading_log_exora_v6_liquidity.txt" # Nama file log baru
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


SETTINGS_FILE = "settings_exora_v6_liquidity.json" # Nama file settings BARU
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 500 
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
                try:
                    send_email_notification(
                        f"Peringatan API Key Switch Global",
                        f"Bot telah beralih ke API key berikutnya (Index {self.current_index}, Key: ...{self.keys[self.current_index][-5:] if self.keys[self.current_index] else 'N/A'}) karena key sebelumnya mungkin bermasalah.",
                        { # Temporary settings dict for this specific email
                            "enable_email_notifications": True,
                            "email_sender_address": self.global_email_settings.get("email_sender_address"),
                            "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                            "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin")
                        },
                        pair_name_ctx_override="SYSTEM_API_KEY"
                    )
                except Exception as e_mail_sw: log_error(f"Gagal kirim email notif global key switch: {e_mail_sw}")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                try:
                    send_email_notification(
                        f"KRITIS: Semua API Key Gagal Global",
                        f"Bot kehabisan semua API key yang valid. Perlu tindakan segera untuk memperbarui API key.",
                         { # Temporary settings dict
                            "enable_email_notifications": True,
                            "email_sender_address": self.global_email_settings.get("email_sender_address"),
                            "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                            "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin")
                        },
                        pair_name_ctx_override="SYSTEM_API_KEY_CRITICAL"
                    )
                except Exception as e_mail_fail: log_error(f"Gagal kirim email notif global semua key gagal: {e_mail_fail}")
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
        # Pastikan title dan content tidak terlalu panjang dan aman untuk shell
        safe_title = title[:100].replace("'", "")
        safe_content = content_msg[:400].replace("'", "") # Batasi panjang konten
        subprocess.run(['termux-notification', '--title', safe_title, '--content', safe_content, '--priority', 'high', '--sound'], 
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{safe_title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
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
        "timeframe": "1m",
        "refresh_interval_seconds": 60,

        # Exora V6 Params
        "rsi_len": 20, "rsi_extreme_oversold": 28, "rsi_extreme_overbought": 73,
        "stoch_k": 41, "stoch_smooth_k": 25, "stoch_d": 3,
        "stoch_extreme_oversold": 10, "stoch_extreme_overbought": 80,
        "use_swing_filter": True, "swing_lookback": 100, 
        "avoid_resistance_proximity_percent": 0.5,
        "use_dump_cooldown": True, "dump_threshold_percent": 1.0, 
        "cooldown_period_after_dump_bars": 500,
        "use_fixed_sl": True, "sl_percent": 4.0,
        "use_standard_tp": False, "standard_tp_percent": 10.0,
        "use_new_trailing_tp": True, "trailing_step_percent": 3.0, "trailing_gap_percent": 1.5,
        
        # >>> BARU: Parameter Deteksi Likuiditas (dari Pine Script) <<<
        "enable_liquidity_detection_py": True,
        "enable_lowliq_py": True, 
        "lowliq_len_py": 4,
        "enable_highliq_py": True,
        "highliq_len_py": 5,
        # Warna tidak dipakai untuk alert, tapi bisa disimpan untuk chart masa depan
        # "lowliq_color_py": "#2622ab", 
        # "highliq_color_py": "#ffffff",

        "enable_email_notifications": False, 
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }

def load_settings():
    default_api_settings = {
        "primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com", "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com",
        "enable_termux_notifications": True # Aktifkan by default untuk termux
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
                if val_str.lower() in ['true', 'yes', 'y', '1']: return True
                if val_str.lower() in ['false', 'no', 'n', '0']: return False
                raise ValueError("Input boolean tidak valid (true/false)")
            
            typed_val = target_type(val_str)
            
            if min_val is not None and typed_val < min_val:
                print(f"{AnsiColors.RED}Nilai harus >= {min_val}.{AnsiColors.ENDC}"); continue
            if max_val is not None and typed_val > max_val:
                print(f"{AnsiColors.RED}Nilai harus <= {max_val}.{AnsiColors.ENDC}"); continue
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
    
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (1m, 5m, 1h, 1d, dll.) [{new_config.get('timeframe',def_cfg['timeframe'])}]: {AnsiColors.ENDC}") or new_config.get('timeframe',def_cfg['timeframe'])).lower().strip()
    valid_tf_keys = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "1d", "3d", "1w"] 
    if tf_input in valid_tf_keys: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default/sebelumnya.{AnsiColors.ENDC}");

    new_config["refresh_interval_seconds"] = _prompt_type("Interval Refresh (detik)", new_config.get('refresh_interval_seconds',def_cfg['refresh_interval_seconds']), int, def_cfg['refresh_interval_seconds'], min_val=MIN_REFRESH_INTERVAL_AFTER_BIG_DATA)

    animated_text_display("\n-- Parameter Exora V6 --", color=AnsiColors.HEADER) # Diringkas
    # ... (Prompt untuk parameter Exora V6 lainnya bisa ditambahkan jika ingin diubah semua)
    # Untuk singkatnya, saya akan skip prompt detail Exora dan langsung ke liquidity
    # Jika ingin mengubah parameter Exora, un-comment bagian di bawah dan sesuaikan
    # new_config["rsi_len"] = _prompt_type("Periode RSI", new_config.get('rsi_len', def_cfg['rsi_len']), int, def_cfg['rsi_len'], min_val=1)
    # ... (dan seterusnya untuk semua parameter Exora V6) ...

    animated_text_display("\n-- Deteksi Likuiditas (Python) --", color=AnsiColors.HEADER)
    new_config["enable_liquidity_detection_py"] = _prompt_type("Aktifkan Deteksi Likuiditas (Alert)?", new_config.get('enable_liquidity_detection_py', def_cfg['enable_liquidity_detection_py']), bool, def_cfg['enable_liquidity_detection_py'])
    new_config["enable_lowliq_py"] = _prompt_type("  Deteksi Low Liquidity?", new_config.get('enable_lowliq_py', def_cfg['enable_lowliq_py']), bool, def_cfg['enable_lowliq_py'])
    new_config["lowliq_len_py"] = _prompt_type("    Panjang Pola Low Liquidity", new_config.get('lowliq_len_py', def_cfg['lowliq_len_py']), int, def_cfg['lowliq_len_py'], min_val=2)
    new_config["enable_highliq_py"] = _prompt_type("  Deteksi High Liquidity?", new_config.get('enable_highliq_py', def_cfg['enable_highliq_py']), bool, def_cfg['enable_highliq_py'])
    new_config["highliq_len_py"] = _prompt_type("    Panjang Pola High Liquidity", new_config.get('highliq_len_py', def_cfg['highliq_len_py']), int, def_cfg['highliq_len_py'], min_val=2)


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
        
        title = f"--- Menu Pengaturan ---\nAPI Key Utama: {pkd} | Recovery Keys: {nrk}\nNotif Termux Global: {tns}\nStrategi: Exora Bot V6 Spot + Alert Likuiditas\n\n--- Crypto Pairs Dikonfigurasi ---\n"
        if not current_settings.get("cryptos"): title += "  (Kosong)\n"
        else:
            for i, cfg_item in enumerate(current_settings["cryptos"]):
                liq_detect_status = "Aktif" if cfg_item.get('enable_liquidity_detection_py', False) else "Nonaktif"
                title += f"  {i+1}. {cfg_item.get('symbol','?')}-{cfg_item.get('currency','?')} ({cfg_item.get('timeframe','?')}) - {'Aktif' if cfg_item.get('enabled',True) else 'Nonaktif'} [Deteksi Liq: {liq_detect_status}]\n"
        
        title += "------------------------------------\nPilih tindakan:"
        opts = ["Primary API Key", "Recovery API Keys", "Email Global Notif Sistem", "Notifikasi Termux Global", 
                "Tambah Crypto Pair", "Ubah Crypto Pair", "Hapus Crypto Pair", "Kembali"]
        
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
            except ValueError: print("Input tidak valid."); time.sleep(1); continue
        
        clear_screen_animated()
        try:
            if action_idx == 0: 
                new_pk = input(f"Primary API Key [{api_s.get('primary_key','')}]: ").strip()
                if new_pk: api_s["primary_key"] = new_pk
                elif not api_s.get('primary_key'): api_s["primary_key"] = "YOUR_PRIMARY_KEY" 
            elif action_idx == 1: 
                while True:
                    clear_screen_animated(); current_recovery = api_s.get('recovery_keys', [])
                    rec_title = "--- Kelola Recovery API Keys ---\n" + ("".join([f"  {i_r+1}. {k_r[:5]}...{k_r[-3:] if len(k_r)>8 else k_r}\n" for i_r,k_r in enumerate(current_recovery)]) if current_recovery else "  (Tidak ada recovery key)\n") + "----------------------------\nPilih:"
                    rec_opts = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali"]
                    _, rec_action_idx = pick(rec_opts, rec_title, indicator='=>')
                    if rec_action_idx == 0: 
                        new_r_key = input("Recovery API Key baru: ").strip()
                        if new_r_key: api_s.setdefault('recovery_keys',[]).append(new_r_key); animated_text_display("Ditambahkan.", color=AnsiColors.GREEN)
                        else: animated_text_display("Input kosong.", color=AnsiColors.ORANGE)
                    elif rec_action_idx == 1 and api_s.get('recovery_keys'):
                        del_r_opts = [f"{k_rd[:5]}...{k_rd[-3:] if len(k_rd)>8 else k_rd}" for k_rd in api_s['recovery_keys']] + ["Batal"]
                        _, del_r_key_idx = pick(del_r_opts, "Pilih key untuk dihapus:", indicator='=>')
                        if del_r_key_idx < len(api_s['recovery_keys']): api_s['recovery_keys'].pop(del_r_key_idx); animated_text_display("Dihapus.", color=AnsiColors.GREEN)
                    elif rec_action_idx == 2: break 
                    show_spinner(0.5, "Memproses...")
            elif action_idx == 2: 
                api_s['enable_global_email_notifications_for_key_switch'] = _prompt_type("Aktifkan Email Notif Sistem Global?", api_s.get('enable_global_email_notifications_for_key_switch',False), bool, False)
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Admin Global [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
            elif action_idx == 3:
                 api_s['enable_termux_notifications'] = _prompt_type("Aktifkan Notifikasi Termux Global?", api_s.get('enable_termux_notifications',True), bool, True) # Default True
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
            elif action_idx == 7: break 
            
            current_settings["api_settings"] = api_s 
            save_settings(current_settings) 
            if action_idx not in [1,7]: show_spinner(1, "Disimpan...") 
        except Exception as e_menu: log_error(f"Error menu: {e_menu}"); show_spinner(1, "Error...")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe_str="1h", pair_name="N/A"):
    timeframe_details = {"endpoint_segment": "histohour", "aggregate": 1}
    tf_lower = timeframe_str.lower()
    if 'm' in tf_lower: 
        timeframe_details["endpoint_segment"] = "histominute"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('m',''))
        except: timeframe_details["aggregate"] = 1
    elif 'h' in tf_lower: 
        timeframe_details["endpoint_segment"] = "histohour"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('h',''))
        except: timeframe_details["aggregate"] = 1
    elif 'd' in tf_lower: 
        timeframe_details["endpoint_segment"] = "histoday"
        try: timeframe_details["aggregate"] = int(tf_lower.replace('d',''))
        except: timeframe_details["aggregate"] = 1
    elif 'w' in tf_lower: 
        timeframe_details["endpoint_segment"] = "histoday" 
        timeframe_details["aggregate"] = 7 * (int(tf_lower.replace('w','')) if tf_lower.replace('w','').isdigit() else 1)

    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = timeframe_details["endpoint_segment"]
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    
    is_large_fetch = total_limit_desired > 20
    if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')

    while len(all_accumulated_candles) < total_limit_desired:
        limit_call = min(total_limit_desired - len(all_accumulated_candles), CRYPTOCOMPARE_MAX_LIMIT)
        if limit_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_call, "api_key": current_api_key_to_use}
        # Aggregate untuk histominute harus selalu 1 dari sisi API, kita handle pembentukan timeframe dari candle 1 menit jika > 1m
        if api_endpoint != "histominute" and timeframe_details["aggregate"] > 1 :
             params["aggregate"] = timeframe_details["aggregate"]
        # Untuk histominute, Crytocompare tidak support aggregate > 1. Jika kita mau timeframe misal 5m, kita fetch 1m dan aggregate manual.
        # Namun, untuk skrip ini, kita asumsikan API bisa handle aggregate di histohour/day, dan histominute kita ambil sesuai yg di-request (misal jika minta 5m, maka aggregate=5).
        # Jika API tidak support, data yg kembali adalah 1 menit, maka perlu penyesuaian manual pembentukan candle timeframe custom.
        # Untuk saat ini, jika endpoint histominute dan aggregate > 1, kita tetap pass aggregate. Jika API tidak support, dia akan return 1m candles.
        # Dan jika `aggregate` > 1 untuk `histominute`, ini akan tetap dikirim ke API.
        if timeframe_details["aggregate"] > 1: # Ini untuk semua endpoint
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
                key_err_patterns = ["api key is invalid", "apikey_is_missing", "rate limit", "monthly_calls", "tier", "no data for"]
                if any(p.lower() in err_msg.lower() for p in key_err_patterns):
                    if "no data for" in err_msg.lower() and "toTs" in params: # Kemungkinan sudah di ujung histori
                        log_debug(f"Tidak ada data sebelum toTs {params['toTs']} untuk {pair_name}. Mungkin akhir histori.", pair_name=pair_name)
                        break 
                    log_warning(f"API Error (JSON): {err_msg} | Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                    if not ("no data for" in err_msg.lower()): # Jangan raise APIKeyError jika hanya no data
                        raise APIKeyError(f"JSON Error: {err_msg}")
                    else: break # Berhenti jika "no data for"
                else: log_error(f"API Error Non-Key: {err_msg}", pair_name=pair_name); break 
            
            raw_candles = data.get('Data', {}).get('Data', [])
            if not raw_candles:
                if len(all_accumulated_candles) > 0 : log_debug(f"Tidak ada candle baru dari API (mungkin akhir histori atau 'no data for fsym'). Total: {len(all_accumulated_candles)}", pair_name=pair_name)
                else: log_warning(f"Tidak ada data candle sama sekali dari API untuk {pair_name}.", pair_name=pair_name)
                break

            batch = []
            for item in raw_candles:
                if all(k in item and item[k] is not None for k in ['time', 'open', 'high', 'low', 'close', 'volumefrom']):
                    batch.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volumefrom']})
            
            if current_to_ts and all_accumulated_candles and batch and batch[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']: batch.pop() 
            if not batch and current_to_ts: break 

            all_accumulated_candles = batch + all_accumulated_candles
            if raw_candles: current_to_ts = raw_candles[0]['time'] -1 # Kurangi 1 detik untuk toTs agar tidak duplikat candle terakhir
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


# --- FUNGSI INDIKATOR (RSI, STOCH, SMA, PIVOT) ---
# (Fungsi calculate_sma, calculate_rsi, calculate_stochastic, find_latest_confirmed_pivot tetap sama)
def calculate_sma(data, period):
    if not data or len(data) < period: return [None] * len(data)
    sma_values = [None] * (period - 1)
    current_sum = sum(data[i] for i in range(period) if data[i] is not None)
    valid_points = sum(1 for i in range(period) if data[i] is not None)

    if valid_points < period : 
         for i in range(period-1, len(data)):
            current_sum = 0; valid_points_window = 0; can_calculate = True
            for j in range(i - period + 1, i + 1):
                if data[j] is None: can_calculate = False; break
                current_sum += data[j]; valid_points_window +=1
            if can_calculate and valid_points_window == period: sma_values.append(current_sum / period)
            else: sma_values.append(None)
         return sma_values
    sma_values.append(current_sum / period)
    for i in range(period, len(data)):
        if data[i] is not None and data[i-period] is not None and sma_values[-1] is not None:
            current_sum = current_sum - data[i-period] + data[i]
            sma_values.append(current_sum / period)
        else: 
            sub_slice = data[i-period+1 : i+1]
            if None not in sub_slice and len(sub_slice) == period: sma_values.append(sum(sub_slice) / period)
            else: sma_values.append(None)
    return sma_values

def calculate_rsi(prices, period):
    if not prices or len(prices) < period + 1: return [None] * len(prices)
    rsi_values = [None] * period 
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]; losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period if gains[:period] else 0
    avg_loss = sum(losses[:period]) / period if losses[:period] else 0
    rs = 100 if avg_loss == 0 else (avg_gain / avg_loss if avg_gain > 0 else 0) # Handle avg_gain=0 juga
    if avg_loss == 0 and avg_gain == 0: rs = 1 # results in RSI 50
    elif avg_loss == 0: rs = 100 # Max RSI if no loss
    else: rs = avg_gain / avg_loss
    rsi_values.append(100 - (100 / (1 + rs)))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0 and avg_gain == 0: rs_i = 1
        elif avg_loss == 0: rs_i = 100
        else: rs_i = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs_i)))
    return [None]*(len(prices)-len(rsi_values)) + rsi_values

def calculate_stochastic(candles_history, k_period, smooth_k_period, d_period):
    closes = [c['close'] for c in candles_history]; highs = [c['high'] for c in candles_history]; lows = [c['low'] for c in candles_history]
    if len(closes) < k_period: return [None] * len(closes), [None] * len(closes)
    stoch_k_raw = [None] * (k_period - 1)
    for i in range(k_period - 1, len(closes)):
        period_highs = highs[i - k_period + 1 : i + 1]; period_lows = lows[i - k_period + 1 : i + 1]
        highest_high = max(ph for ph in period_highs if ph is not None) if any(ph is not None for ph in period_highs) else None
        lowest_low = min(pl for pl in period_lows if pl is not None) if any(pl is not None for pl in period_lows) else None
        current_close = closes[i]
        if current_close is not None and highest_high is not None and lowest_low is not None and highest_high != lowest_low:
            stoch_k_raw.append(((current_close - lowest_low) / (highest_high - lowest_low)) * 100)
        elif highest_high == lowest_low and current_close is not None and lowest_low is not None: stoch_k_raw.append(50)
        else: stoch_k_raw.append(None)
    stoch_k_smoothed = calculate_sma(stoch_k_raw, smooth_k_period)
    stoch_d_smoothed = calculate_sma(stoch_k_smoothed, d_period)
    return stoch_k_smoothed, stoch_d_smoothed

def find_latest_confirmed_pivot(candles_history, lookback_period, is_high_pivot=True):
    if not candles_history or len(candles_history) < 2 * lookback_period + 1: return None, None
    latest_pivot_price = None; latest_pivot_timestamp = None
    for i in range(len(candles_history) - 1 - lookback_period, lookback_period - 1, -1):
        candle_to_check = candles_history[i]
        current_val = candle_to_check['high'] if is_high_pivot else candle_to_check['low']
        if current_val is None: continue
        is_pivot = True
        for j in range(1, lookback_period + 1): # Check left
            left_val = candles_history[i-j]['high'] if is_high_pivot else candles_history[i-j]['low']
            if left_val is None or (is_high_pivot and left_val >= current_val) or \
               (not is_high_pivot and left_val <= current_val): is_pivot = False; break
        if not is_pivot: continue
        for j in range(1, lookback_period + 1): # Check right
            right_val = candles_history[i+j]['high'] if is_high_pivot else candles_history[i+j]['low']
            if right_val is None or (is_high_pivot and right_val > current_val) or \
               (not is_high_pivot and right_val < current_val): is_pivot = False; break
        if is_pivot: latest_pivot_price = current_val; latest_pivot_timestamp = candle_to_check['timestamp']; break 
    return latest_pivot_price, latest_pivot_timestamp

# --- BARU: FUNGSI DETEKSI LIKUIDITAS (Python) ---
def is_candle_bullish_py(candle):
    return candle and candle['close'] is not None and candle['open'] is not None and candle['close'] > candle['open']

def is_candle_bearish_py(candle):
    return candle and candle['close'] is not None and candle['open'] is not None and candle['close'] < candle['open']

def detect_low_liquidity_pattern_py(candles_history, length):
    if not candles_history or len(candles_history) < length:
        return False
    
    p1 = True  # Starts Bullish: B, S, B, S...
    p2 = True  # Starts Bearish: S, B, S, B...

    # Loop from the most recent candle in the pattern (index 0) to the oldest (index length-1)
    for i in range(length):
        candle_index_in_history = -1 - i # Accesses candles_history[-1], then [-2], ..., [-length]
        candle = candles_history[candle_index_in_history]

        if not candle or candle['open'] is None or candle['close'] is None: # Invalid candle data
            p1, p2 = False, False
            break

        # Pattern 1: B, S, B, S...
        # i=0 (current) expects Bullish, i=1 (prev) expects Bearish
        if i % 2 == 0: # Expect Bullish for p1
            if not is_candle_bullish_py(candle): p1 = False
        else: # Expect Bearish for p1
            if not is_candle_bearish_py(candle): p1 = False

        # Pattern 2: S, B, S, B...
        # i=0 (current) expects Bearish, i=1 (prev) expects Bullish
        if i % 2 == 0: # Expect Bearish for p2
            if not is_candle_bearish_py(candle): p2 = False
        else: # Expect Bullish for p2
            if not is_candle_bullish_py(candle): p2 = False
            
        if not p1 and not p2:
            break
            
    return p1 or p2

def detect_high_liquidity_pattern_py(candles_history, length):
    if not candles_history or len(candles_history) < length:
        return False

    all_bullish = True
    all_bearish = True

    for i in range(length):
        candle_index_in_history = -1 - i
        candle = candles_history[candle_index_in_history]

        if not candle or candle['open'] is None or candle['close'] is None:
            all_bullish, all_bearish = False, False
            break
            
        if not is_candle_bullish_py(candle):
            all_bullish = False
        if not is_candle_bearish_py(candle):
            all_bearish = False
        
        if not all_bullish and not all_bearish:
            break
            
    return all_bullish or all_bearish


# --- LOGIKA STRATEGI (Exora Bot V6 + Analisis Likuiditas) ---
def get_initial_strategy_state():
    return {
        "last_rsi_value": None, "last_stoch_k_value": None, "last_stoch_d_value": None,
        "last_valid_swing_high_price": None, "last_valid_swing_high_time": None,
        "last_valid_swing_low_price": None, "last_valid_swing_low_time": None,
        "is_cooldown_active": False, "cooldown_bars_remaining": 0,
        "has_entered_oversold_zone": False, "rsi_has_exited_oversold_zone": False, "stoch_has_exited_oversold_zone": False,
        "in_position": False, "entry_price": None, "position_entry_timestamp": None, "last_trade_id": None,
        "entry_price_for_trail": None, "highest_high_since_entry": None,
        "highest_num_steps_achieved": 0, "current_trailing_stop_level": None, "initial_stop_for_current_trade": None,
        
        "active_sl_tp_for_chart": None, "swing_high_for_chart": None, "swing_low_for_chart": None,
        "buy_signal_on_chart": False, "sell_signal_on_chart": False, "dump_trigger_on_chart": False, "cooldown_active_on_chart": False,

        # >>> BARU: State untuk Likuiditas <<<
        "current_liquidity_state_py": 0, # 0: Undetermined, 1: Low, 2: High
        "previous_liquidity_state_py": 0, 
        "last_liquidity_alert_msg": "", # Untuk menghindari alert duplikat per bar
    }

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings, is_warmup=False):
    pair_name = crypto_config['pair_name']
    cfg = crypto_config 

    if not candles_history:
        log_debug("candles_history kosong, skip logika strategi.", pair_name=pair_name)
        return strategy_state
        
    current_candle = candles_history[-1]
    current_close = current_candle['close']
    current_open = current_candle['open']
    current_high = current_candle['high']
    current_low = current_candle['low']
    current_time = current_candle['timestamp']

    strategy_state["buy_signal_on_chart"] = False; strategy_state["sell_signal_on_chart"] = False
    strategy_state["dump_trigger_on_chart"] = False; strategy_state["cooldown_active_on_chart"] = False

    # 1. CALCULATE INDICATORS (Exora V6)
    rsi_values = calculate_rsi([c['close'] for c in candles_history], cfg['rsi_len'])
    rsi = rsi_values[-1] if rsi_values and len(rsi_values) > 0 else None
    strategy_state["last_rsi_value"] = rsi
    
    stoch_k_series, stoch_d_series = calculate_stochastic(candles_history, cfg['stoch_k'], cfg['stoch_smooth_k'], cfg['stoch_d'])
    stoch_k = stoch_k_series[-1] if stoch_k_series and len(stoch_k_series) > 0 else None
    strategy_state["last_stoch_k_value"] = stoch_k; strategy_state["last_stoch_d_value"] = stoch_d_series[-1] if stoch_d_series and len(stoch_d_series) > 0 else None

    if cfg['use_swing_filter']:
        l_sw_h_p, l_sw_h_t = find_latest_confirmed_pivot(candles_history, cfg['swing_lookback'], is_high_pivot=True)
        l_sw_l_p, l_sw_l_t = find_latest_confirmed_pivot(candles_history, cfg['swing_lookback'], is_high_pivot=False)
        if l_sw_h_p is not None: strategy_state['last_valid_swing_high_price'] = l_sw_h_p; strategy_state['last_valid_swing_high_time'] = l_sw_h_t
        if l_sw_l_p is not None: strategy_state['last_valid_swing_low_price'] = l_sw_l_p; strategy_state['last_valid_swing_low_time'] = l_sw_l_t
        strategy_state["swing_high_for_chart"] = strategy_state['last_valid_swing_high_price']
        strategy_state["swing_low_for_chart"] = strategy_state['last_valid_swing_low_price']

    # --- BARU: ANALISIS LIKUIDITAS ---
    if cfg.get("enable_liquidity_detection_py", False) and not is_warmup: # Hanya alert di live, tidak saat warmup
        low_liq_trigger_py = False
        if cfg.get("enable_lowliq_py", False) and cfg.get("lowliq_len_py", 0) > 1:
            low_liq_trigger_py = detect_low_liquidity_pattern_py(candles_history, cfg["lowliq_len_py"])

        high_liq_trigger_py = False
        if cfg.get("enable_highliq_py", False) and cfg.get("highliq_len_py", 0) > 1:
            high_liq_trigger_py = detect_high_liquidity_pattern_py(candles_history, cfg["highliq_len_py"])

        current_state_from_logic = strategy_state["current_liquidity_state_py"] # Ambil state saat ini sebelum diubah

        if current_state_from_logic == 0: # Undetermined
            if high_liq_trigger_py:
                strategy_state["current_liquidity_state_py"] = 2 # High
            elif low_liq_trigger_py:
                strategy_state["current_liquidity_state_py"] = 1 # Low
        elif current_state_from_logic == 1: # Currently Low
            if high_liq_trigger_py:
                strategy_state["current_liquidity_state_py"] = 2 # Switch to High
        elif current_state_from_logic == 2: # Currently High
            if low_liq_trigger_py:
                strategy_state["current_liquidity_state_py"] = 1 # Switch to Low
        
        # Cek perubahan state untuk alert
        if strategy_state["current_liquidity_state_py"] != strategy_state["previous_liquidity_state_py"]:
            liquidity_state_map = {0: "Undetermined", 1: "Low Liquidity", 2: "High Liquidity"}
            prev_state_name = liquidity_state_map[strategy_state["previous_liquidity_state_py"]]
            current_state_name = liquidity_state_map[strategy_state["current_liquidity_state_py"]]
            
            alert_title = f"Liquidity Shift: {pair_name}"
            alert_msg_body = f"{pair_name}: Liquidity state changed from {AnsiColors.ORANGE}{prev_state_name}{AnsiColors.ENDC} to {AnsiColors.GREEN if strategy_state['current_liquidity_state_py'] == 2 else AnsiColors.MAGENTA if strategy_state['current_liquidity_state_py'] == 1 else AnsiColors.CYAN}{current_state_name}{AnsiColors.ENDC}."
            
            # Hindari alert duplikat jika run_strategy_logic dipanggil beberapa kali dengan state sama untuk bar yg sama
            unique_alert_identifier = f"{pair_name}_{prev_state_name}_to_{current_state_name}_{current_time}" # Tambah timestamp untuk keunikan per bar
            if strategy_state.get("last_liquidity_alert_id") != unique_alert_identifier:
                log_info(f"{AnsiColors.YELLOW_BG}{AnsiColors.BLUE}ALERT:{AnsiColors.ENDC} {alert_msg_body}", pair_name=pair_name)
                send_termux_notification(alert_title, f"From {prev_state_name} to {current_state_name}", global_settings, pair_name_for_log=pair_name)
                if cfg.get("enable_email_notifications"):
                    send_email_notification(f"ALERT Liquidity Shift: {pair_name}", f"Liquidity state changed from {prev_state_name} to {current_state_name} for {pair_name} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}.", cfg, pair_name_ctx_override=pair_name)
                strategy_state["last_liquidity_alert_id"] = unique_alert_identifier
            
            strategy_state["previous_liquidity_state_py"] = strategy_state["current_liquidity_state_py"]
    elif is_warmup and cfg.get("enable_liquidity_detection_py", False) : # Saat warmup, update state saja tanpa alert
        low_liq_trigger_py = False
        if cfg.get("enable_lowliq_py", False) and cfg.get("lowliq_len_py", 0) > 1:
            low_liq_trigger_py = detect_low_liquidity_pattern_py(candles_history, cfg["lowliq_len_py"])
        high_liq_trigger_py = False
        if cfg.get("enable_highliq_py", False) and cfg.get("highliq_len_py", 0) > 1:
            high_liq_trigger_py = detect_high_liquidity_pattern_py(candles_history, cfg["highliq_len_py"])
        
        current_state_from_logic = strategy_state["current_liquidity_state_py"]
        if current_state_from_logic == 0: 
            if high_liq_trigger_py: strategy_state["current_liquidity_state_py"] = 2
            elif low_liq_trigger_py: strategy_state["current_liquidity_state_py"] = 1
        elif current_state_from_logic == 1: 
            if high_liq_trigger_py: strategy_state["current_liquidity_state_py"] = 2
        elif current_state_from_logic == 2: 
            if low_liq_trigger_py: strategy_state["current_liquidity_state_py"] = 1
        strategy_state["previous_liquidity_state_py"] = strategy_state["current_liquidity_state_py"]


    # 2. COOLDOWN LOGIC (Exora V6)
    if strategy_state['is_cooldown_active']:
        strategy_state['cooldown_bars_remaining'] -= 1
        if strategy_state['cooldown_bars_remaining'] <= 0:
            strategy_state['is_cooldown_active'] = False
            log_info(f"Cooldown period ended for {pair_name}.", pair_name=pair_name)
    is_bearish_candle = current_close < current_open
    candle_body_percent_drop = ((current_open - current_close) / current_open * 100.0) if is_bearish_candle and current_open > 0 else 0.0
    is_dump_candle_now = is_bearish_candle and candle_body_percent_drop >= cfg['dump_threshold_percent']
    if cfg['use_dump_cooldown'] and is_dump_candle_now and not strategy_state['is_cooldown_active']:
        strategy_state['is_cooldown_active'] = True; strategy_state['cooldown_bars_remaining'] = cfg['cooldown_period_after_dump_bars']
        strategy_state["dump_trigger_on_chart"] = True
        log_warning(f"DUMP DETECTED ({candle_body_percent_drop:.2f}%) on {pair_name}. Cooldown for {cfg['cooldown_period_after_dump_bars']} bars activated.", pair_name=pair_name)
    strategy_state["cooldown_active_on_chart"] = strategy_state['is_cooldown_active']

    # 3. STRATEGY CONDITIONS (Entry & Extreme Exit - Exora V6)
    if rsi is None or stoch_k is None:
        log_debug(f"RSI ({rsi}) or Stoch K ({stoch_k}) is None. Skipping signal logic for {pair_name}.", pair_name=pair_name)
        return strategy_state

    rsi_is_currently_oversold = rsi < cfg['rsi_extreme_oversold']; stoch_is_currently_oversold = stoch_k < cfg['stoch_extreme_oversold']
    prev_rsi = rsi_values[-2] if len(candles_history) > 1 and len(rsi_values) > 1 else None
    prev_stoch_k = stoch_k_series[-2] if len(candles_history) > 1 and len(stoch_k_series) > 1 else None

    if rsi_is_currently_oversold and stoch_is_currently_oversold:
        strategy_state['has_entered_oversold_zone'] = True
        strategy_state['rsi_has_exited_oversold_zone'] = False; strategy_state['stoch_has_exited_oversold_zone'] = False
    
    if strategy_state['has_entered_oversold_zone']:
        if prev_rsi is not None and prev_rsi < cfg['rsi_extreme_oversold'] and rsi >= cfg['rsi_extreme_oversold']: strategy_state['rsi_has_exited_oversold_zone'] = True
        if prev_stoch_k is not None and prev_stoch_k < cfg['stoch_extreme_oversold'] and stoch_k >= cfg['stoch_extreme_oversold']: strategy_state['stoch_has_exited_oversold_zone'] = True
        if rsi_is_currently_oversold: strategy_state['rsi_has_exited_oversold_zone'] = False
        if stoch_is_currently_oversold: strategy_state['stoch_has_exited_oversold_zone'] = False

    cond_buy_core_new = (strategy_state['has_entered_oversold_zone'] and strategy_state['rsi_has_exited_oversold_zone'] and strategy_state['stoch_has_exited_oversold_zone'])
    
    resistance_filter_ok = True
    if cfg['use_swing_filter']:
        last_sw_high = strategy_state['last_valid_swing_high_price']
        if last_sw_high is not None and current_close >= (last_sw_high * (1 - cfg['avoid_resistance_proximity_percent'] / 100.0)):
            resistance_filter_ok = False; log_debug(f"Resistance filter: BUY BLOCKED near SwingH {last_sw_high}", pair_name=pair_name)
    
    cooldown_filter_ok = not (cfg['use_dump_cooldown'] and strategy_state['is_cooldown_active'])
    if not cooldown_filter_ok: log_debug(f"Cooldown filter: BUY BLOCKED. Cooldown active.", pair_name=pair_name)

    buy_condition_filtered = (cond_buy_core_new and resistance_filter_ok and cooldown_filter_ok and not strategy_state['in_position'])
    cond_sell_core_extreme = (rsi > cfg['rsi_extreme_overbought'] and stoch_k > cfg['stoch_extreme_overbought'])
    
    # POSITION MANAGEMENT (Exora V6)
    trade_closed_this_bar = False
    if strategy_state['in_position']:
        entry_p_trail = strategy_state['entry_price_for_trail']
        if strategy_state['highest_high_since_entry'] is None: strategy_state['highest_high_since_entry'] = current_high
        else: strategy_state['highest_high_since_entry'] = max(strategy_state['highest_high_since_entry'], current_high)

        if cfg['use_new_trailing_tp'] and entry_p_trail is not None and entry_p_trail > 0 and strategy_state['highest_high_since_entry'] is not None:
            current_profit_p = (strategy_state['highest_high_since_entry'] - entry_p_trail) / entry_p_trail * 100.0
            num_steps = math.floor(max(0, current_profit_p) / cfg['trailing_step_percent']) if cfg['trailing_step_percent'] > 0 else 0
            if num_steps > strategy_state['highest_num_steps_achieved'] and num_steps >= 1:
                old_trail_lvl = strategy_state['current_trailing_stop_level']
                strategy_state['highest_num_steps_achieved'] = num_steps
                profit_chkpt_p = float(num_steps) * cfg['trailing_step_percent']
                locked_profit_p = max(0.0, profit_chkpt_p - cfg['trailing_gap_percent'])
                new_trail_lvl = entry_p_trail * (1 + locked_profit_p / 100.0)
                if strategy_state['current_trailing_stop_level'] is None: strategy_state['current_trailing_stop_level'] = new_trail_lvl
                else: strategy_state['current_trailing_stop_level'] = max(strategy_state['current_trailing_stop_level'], new_trail_lvl)
                if strategy_state['initial_stop_for_current_trade'] is not None: strategy_state['current_trailing_stop_level'] = max(strategy_state['current_trailing_stop_level'], strategy_state['initial_stop_for_current_trade'])
                if strategy_state['current_trailing_stop_level'] != old_trail_lvl and not is_warmup:
                    trail_msg = f"TRAILING STOP MOVED UP for {pair_name} to {strategy_state['current_trailing_stop_level']:.5f} ({locked_profit_p:.2f}% locked)."
                    log_info(f"{AnsiColors.MAGENTA}{trail_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                    send_termux_notification(f"Trail Up {pair_name}", trail_msg, global_settings, pair_name_for_log=pair_name)
                    if cfg.get("enable_email_notifications"): send_email_notification(f"ExoraV6 Trail UP: {pair_name}", trail_msg, cfg, pair_name_ctx_override=pair_name)

        actual_sl = None
        if cfg['use_new_trailing_tp'] and strategy_state['current_trailing_stop_level'] is not None: actual_sl = strategy_state['current_trailing_stop_level']
        elif cfg['use_fixed_sl'] and strategy_state['initial_stop_for_current_trade'] is not None: actual_sl = strategy_state['initial_stop_for_current_trade']
        
        actual_tp = None
        if not cfg['use_new_trailing_tp'] and cfg['use_standard_tp'] and entry_p_trail is not None:
            actual_tp = entry_p_trail * (1 + cfg['standard_tp_percent'] / 100.0)
        strategy_state["active_sl_tp_for_chart"] = actual_sl

        exit_reason = None
        exit_price_sim = None
        if actual_sl is not None and current_low <= actual_sl:
            exit_price_sim = min(current_open, actual_sl); exit_reason = "STOP LOSS"
        elif not trade_closed_this_bar and actual_tp is not None and current_high >= actual_tp:
            exit_price_sim = max(current_open, actual_tp); exit_reason = "STANDARD TP"
        elif not trade_closed_this_bar and cond_sell_core_extreme:
            exit_price_sim = current_close; exit_reason = "EXTREME OVERBOUGHT"
            if not is_warmup: strategy_state["sell_signal_on_chart"] = True
        
        if exit_reason and exit_price_sim is not None:
            profit_p = (exit_price_sim - strategy_state['entry_price']) / strategy_state['entry_price'] * 100 if strategy_state['entry_price'] else 0
            exit_msg_log = f"{exit_reason} HIT for {pair_name} at ~{exit_price_sim:.5f}. Entry: {strategy_state['entry_price']:.5f}. Profit: {profit_p:.2f}%"
            if not is_warmup:
                color_map = {"STOP LOSS": AnsiColors.RED, "STANDARD TP": AnsiColors.GREEN, "EXTREME OVERBOUGHT": AnsiColors.ORANGE}
                log_info(f"{color_map.get(exit_reason, AnsiColors.ORANGE)}{exit_msg_log}{AnsiColors.ENDC}", pair_name=pair_name)
                send_termux_notification(f"{exit_reason} {pair_name}", exit_msg_log, global_settings, pair_name_for_log=pair_name)
                if cfg.get("enable_email_notifications"): send_email_notification(f"ExoraV6 {exit_reason}: {pair_name}", exit_msg_log, cfg, pair_name_ctx_override=pair_name)
            strategy_state['in_position'] = False; trade_closed_this_bar = True
            # Reset state
            strategy_state['entry_price'] = None; strategy_state['position_entry_timestamp'] = None
            strategy_state['entry_price_for_trail'] = None; strategy_state['highest_high_since_entry'] = None
            strategy_state['highest_num_steps_achieved'] = 0; strategy_state['current_trailing_stop_level'] = None
            strategy_state['initial_stop_for_current_trade'] = None; strategy_state['active_sl_tp_for_chart'] = None

    if buy_condition_filtered and not strategy_state['in_position'] and not trade_closed_this_bar:
        entry_p_sim = current_close
        strategy_state['in_position'] = True; strategy_state['entry_price'] = entry_p_sim
        strategy_state['position_entry_timestamp'] = current_time; strategy_state['last_trade_id'] = str(uuid.uuid4())
        strategy_state['entry_price_for_trail'] = entry_p_sim; strategy_state['highest_high_since_entry'] = current_high
        strategy_state['highest_num_steps_achieved'] = 0
        if cfg['use_fixed_sl']:
            strategy_state['initial_stop_for_current_trade'] = entry_p_sim * (1 - cfg['sl_percent'] / 100.0)
            strategy_state['current_trailing_stop_level'] = strategy_state['initial_stop_for_current_trade']
        else: strategy_state['initial_stop_for_current_trade'] = None; strategy_state['current_trailing_stop_level'] = None
        strategy_state["active_sl_tp_for_chart"] = strategy_state['current_trailing_stop_level']
        if not is_warmup:
            entry_msg = f"ENTRY LONG SIGNAL for {pair_name} at ~{entry_p_sim:.5f}. SL: {strategy_state['current_trailing_stop_level']:.5f if strategy_state['current_trailing_stop_level'] else 'N/A'}"
            log_info(f"{AnsiColors.GREEN}{entry_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            send_termux_notification(f"ENTRY {pair_name}", entry_msg, global_settings, pair_name_for_log=pair_name)
            if cfg.get("enable_email_notifications"): send_email_notification(f"ExoraV6 ENTRY: {pair_name}", entry_msg, cfg, pair_name_ctx_override=pair_name)
            strategy_state["buy_signal_on_chart"] = True
        strategy_state['has_entered_oversold_zone'] = False; strategy_state['rsi_has_exited_oversold_zone'] = False; strategy_state['stoch_has_exited_oversold_zone'] = False

    return strategy_state


# CHART_INTEGRATION_START & Flask Endpoints (Tidak diubah signifikan, hanya memastikan tidak crash)
shared_crypto_data_manager = {}
shared_data_lock = threading.Lock()
flask_app_instance = None
try:
    flask_app_instance = Flask(__name__)
except Exception: # Jika Flask tidak ada
    pass 

HTML_CHART_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exora V6 Spot Chart (Liquidity)</title><script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<style>body{font-family:sans-serif;margin:0;background-color:#1e1e1e;color:#e0e0e0;display:flex;flex-direction:column;align-items:center;padding:10px}#controls{background-color:#2a2a2a;padding:10px;border-radius:8px;margin-bottom:10px;display:flex;align-items:center;gap:10px;width:100%;max-width:1200px;flex-wrap:wrap}select,button{padding:8px 12px;border-radius:5px;border:1px solid #444;background-color:#333;color:#e0e0e0;cursor:pointer}#chart-container{width:100%;max-width:1200px;background-color:#2a2a2a;padding:15px;border-radius:8px}h1{color:#00bcd4;margin-bottom:10px;font-size:1.5em}#lastUpdatedLabel{font-size:.8em;color:#aaa;margin-left:auto}#strategyInfoLabel{font-size:.8em;color:#ffd700;margin-left:10px;white-space:pre-wrap;max-width:300px;max-height:100px;overflow-y:auto}#liquidityStateLabel{font-size:.8em;color:#87CEEB;margin-left:10px;}.cooldown-active{background-color:rgba(128,128,128,.3)!important}</style></head>
<body><h1>Exora V6 Spot Strategy Chart (Liquidity Alerts)</h1><div id="controls"><label for="pairSelector">Pair:</label><select id="pairSelector" onchange="handlePairSelectionChange()"></select><button onclick="loadChartDataForCurrentPair()">Refresh</button><span id="strategyInfoLabel">Status: -</span><span id="liquidityStateLabel">Liq: -</span><span id="lastUpdatedLabel">Memuat...</span></div><div id="chart-container"><div id="chart"></div></div>
<script>
let activeChart,currentSelectedPairId="",lastKnownDataTimestamp=null,autoRefreshIntervalId=null,isLoadingData=!1;
const initialChartOptions={series:[{name:"Candlestick",type:"candlestick",data:[]},{name:"Active SL/Trail",type:"line",data:[],color:"#FFA500"}],chart:{type:"candlestick",height:550,background:"#2a2a2a",animations:{enabled:!1},toolbar:{show:!0}},theme:{mode:"dark"},title:{text:"Memuat Data Pair...",align:"left",style:{color:"#e0e0e0"}},xaxis:{type:"datetime",labels:{style:{colors:"#aaa"}},tooltip:{enabled:!1}},yaxis:{tooltip:{enabled:!0},labels:{style:{colors:"#aaa"},formatter:v=>v?v.toFixed(5):""}},stroke:{width:[1,2],curve:"straight"},markers:{size:0},colors:["#FEB019","#FFA500"],grid:{borderColor:"#444"},annotations:{yaxis:[],points:[]},tooltip:{theme:"dark",shared:!0,intersect:!1,y:{formatter:val=>val?val.toFixed(5):val}},noData:{text:"Tidak ada data.",align:"center",style:{color:"#ccc"}}};
async function fetchAvailablePairs(){try{const e=await fetch("/api/available_pairs");if(!e.ok)throw new Error(`HTTP ${e.status}`);const t=await e.json(),a=document.getElementById("pairSelector");if(a.innerHTML="",t.length>0)t.forEach(e=>{const t=document.createElement("option");t.value=e.id,t.textContent=e.name,a.appendChild(t)}),currentSelectedPairId=a.value||t[0].id,loadChartDataForCurrentPair();else{a.innerHTML='<option value="">No pairs</option>';if(activeChart){activeChart.destroy();activeChart=null}document.getElementById("chart").innerHTML="No pairs configured."}}catch(e){console.error("Error fetching available pairs:",e);if(activeChart){activeChart.destroy();activeChart=null}document.getElementById("chart").innerHTML=`Error loading pairs: ${e.message}`}}
function handlePairSelectionChange(){currentSelectedPairId=document.getElementById("pairSelector").value,lastKnownDataTimestamp=null,loadChartDataForCurrentPair()}
async function loadChartDataForCurrentPair(){if(!currentSelectedPairId||isLoadingData)return;isLoadingData=!0,document.getElementById("lastUpdatedLabel").textContent=`Loading ${currentSelectedPairId}...`;const e=document.getElementById("chart-container");try{const t=await fetch(`/api/chart_data/${currentSelectedPairId}`);if(!t.ok)throw new Error(`HTTP ${t.status}`);const a=await t.json();if(a&&a.ohlc){if(a.last_updated_tv&&a.last_updated_tv===lastKnownDataTimestamp&&!a.force_update_chart){console.log("Chart data is unchanged.");document.getElementById("lastUpdatedLabel").textContent=`Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;isLoadingData=!1;a.cooldown_active_bg?e.classList.add("cooldown-active"):e.classList.remove("cooldown-active");const n={0:"Undetermined",1:"Low Liquidity",2:"High Liquidity"};document.getElementById("liquidityStateLabel").textContent=`Liq: ${n[a.strategy_state_info.current_liquidity_state_py]||"N/A"}`;return}lastKnownDataTimestamp=a.last_updated_tv,document.getElementById("lastUpdatedLabel").textContent=lastKnownDataTimestamp?`Last @ ${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`:"Data Loaded";const o=a.strategy_state_info||{},l=a.config_info||{};let i=\`In Pos: \${o.in_position?"YES":"NO"}\\n\`;o.in_position&&(i+=\`Entry: \${o.entry_price?o.entry_price.toFixed(5):"N/A"}\\n\`,i+=\`SL/Trail: \${o.active_sl_tp_for_chart?o.active_sl_tp_for_chart.toFixed(5):"N/A"}\\n\`);i+=\`RSI(\${l.rsi_len}): \${o.last_rsi_value?o.last_rsi_value.toFixed(2):"N/A"}\\n\`,i+=\`StochK(\${l.stoch_k}): \${o.last_stoch_k_value?o.last_stoch_k_value.toFixed(2):"N/A"}\\n\`,i+=\`Cooldown: \${o.is_cooldown_active?o.cooldown_bars_remaining+" bars":"OFF"}\`,document.getElementById("strategyInfoLabel").textContent=i;const s={0:"Undetermined",1:"Low Liq",2:"High Liq"};document.getElementById("liquidityStateLabel").textContent=\`Liq: \${s[o.current_liquidity_state_py]||"N/A"}\`;let d=[...a.swing_high_points||[],...a.swing_low_points||[],...a.annotations_points||[]];const r={title:{...initialChartOptions.title,text:\`\${a.pair_name} - Exora V6 (\${l.timeframe})\`},series:[{name:"Candlestick",type:"candlestick",data:a.ohlc||[]},{name:"Active SL/Trail",type:"line",data:a.active_sl_tp_line||[],color:"#FFA500"}],annotations:{yaxis:[],points:d},colors:["#FEB019","#FFA500"]};a.cooldown_active_bg?e.classList.add("cooldown-active"):e.classList.remove("cooldown-active");activeChart?activeChart.updateOptions(r):(activeChart=new ApexCharts(document.querySelector("#chart"),r),activeChart.render())}else{const t={...initialChartOptions,title:{...initialChartOptions.title,text:\`\${a.pair_name||currentSelectedPairId} - No Data\`},series:initialChartOptions.series.map(e=>({...e,data:[]}))};activeChart?activeChart.updateOptions(t):(activeChart=new ApexCharts(document.querySelector("#chart"),t),activeChart.render());lastKnownDataTimestamp=a.last_updated_tv||null,document.getElementById("lastUpdatedLabel").textContent=lastKnownDataTimestamp?\`Data (empty) @ \${new Date(lastKnownDataTimestamp).toLocaleTimeString()}\`:"No data",document.getElementById("strategyInfoLabel").textContent="Status: Data Kosong",document.getElementById("liquidityStateLabel").textContent="Liq: N/A",e.classList.remove("cooldown-active")}}catch(t){console.error("Error loading chart data:",t);if(activeChart){activeChart.destroy();activeChart=null}document.getElementById("chart").innerHTML=`Error loading chart: ${t.message}`,e.classList.remove("cooldown-active")}finally{isLoadingData=!1}}
document.addEventListener("DOMContentLoaded",()=>{if(!activeChart){activeChart=new ApexCharts(document.querySelector("#chart"),initialChartOptions);activeChart.render()}fetchAvailablePairs();if(autoRefreshIntervalId)clearInterval(autoRefreshIntervalId);autoRefreshIntervalId=setInterval(async()=>{if(currentSelectedPairId&&"visible"===document.visibilityState&&!isLoadingData)await loadChartDataForCurrentPair()},5e3)});
</script></body></html>
""" # Minified HTML
if flask_app_instance:
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
            default_cfg = get_default_crypto_config()
            pair_name_default = f"{default_cfg['symbol']}-{default_cfg['currency']}"
            return jsonify({
                "ohlc":[], "active_sl_tp_line":[], "swing_high_points": [], "swing_low_points": [],
                "pair_name": pair_name_default, "last_updated_tv": None, 
                "strategy_state_info": get_initial_strategy_state(), "config_info": default_cfg,
                "annotations_points": []
            }), 200

        # CHART_INTEGRATION_START (prepare_chart_data_for_pair definition)
        # (Konten fungsi prepare_chart_data_for_pair tetap sama, tidak perlu diubah untuk alert likuiditas)
        # Hanya pastikan state likuiditas terkirim ke chart jika ingin ditampilkan.
        # Snapshot sudah mengandung strategy_state yang kini memiliki info likuiditas.
        temp_manager = {pair_id_from_request: pair_data_snapshot}

        # (Definisi prepare_chart_data_for_pair yang sudah ada di skrip Anda)
        # Ini adalah versi ringkas untuk placeholder, gunakan versi lengkap dari skrip Anda
        def prepare_chart_data_for_pair(pair_id, snapshot):
            if pair_id not in snapshot: return None
            data = snapshot[pair_id]
            hist = data.get("all_candles_list", [])[-TARGET_BIG_DATA_CANDLES:] 
            cfg = data.get("config", {})
            state = data.get("strategy_state", {}) # Ini sudah mengandung state likuiditas
            
            ohlc_data = []
            active_sl_tp_line_data = []
            swing_high_points = []
            swing_low_points = []
            annotations_points_chart = []
            pair_display_name = cfg.get('pair_name', pair_id)

            if not hist:
                return { "ohlc": [], "active_sl_tp_line": [], "swing_high_points": [], "swing_low_points": [],
                         "annotations_points": [], "pair_name": pair_display_name, "last_updated_tv": None,
                         "strategy_state_info": state, "config_info": cfg, "cooldown_active_bg": False }

            for i, c in enumerate(hist):
                if all(c.get(k) is not None for k in ['timestamp', 'open', 'high', 'low', 'close']):
                    ts_ms = c['timestamp'].timestamp() * 1000
                    ohlc_data.append({'x': ts_ms, 'y': [c['open'], c['high'], c['low'], c['close']]})
                    is_current_eval_bar = (i == len(hist) - 1)
                    if is_current_eval_bar:
                        if state.get("active_sl_tp_for_chart") is not None:
                            if len(ohlc_data) > 1: active_sl_tp_line_data.append({'x': ohlc_data[-2]['x'], 'y': state.get("active_sl_tp_for_chart")})
                            active_sl_tp_line_data.append({'x': ts_ms, 'y': state.get("active_sl_tp_for_chart")})
                        if cfg.get("use_swing_filter"):
                            if state.get("swing_high_for_chart") and state.get("last_valid_swing_high_time"):
                                swing_high_points.append({'x': state.get("last_valid_swing_high_time").timestamp() * 1000, 'y': state.get("swing_high_for_chart"), 'marker': {'size': 6, 'fillColor': '#FF0000'}, 'label': {'text': 'SH'}})
                            if state.get("swing_low_for_chart") and state.get("last_valid_swing_low_time"):
                                swing_low_points.append({'x': state.get("last_valid_swing_low_time").timestamp() * 1000, 'y': state.get("swing_low_for_chart"), 'marker': {'size': 6, 'fillColor': '#00FF00'}, 'label': {'text': 'SLow'}})
                        if state.get("buy_signal_on_chart"): annotations_points_chart.append({'x': ts_ms, 'y': c['low'], 'marker': {'size': 8, 'fillColor': '#26E7A5', 'shape': 'triangle'}, 'label': {'text': 'BUY'}})
                        if state.get("sell_signal_on_chart"): annotations_points_chart.append({'x': ts_ms, 'y': c['high'], 'marker': {'size': 8, 'fillColor': '#FF4560', 'shape': 'triangle-down'}, 'label': {'text': 'SELL-OB'}})
                        if state.get("dump_trigger_on_chart"): annotations_points_chart.append({'x': ts_ms, 'y': c['high'], 'marker': {'size': 6, 'fillColor': '#FFA500', 'shape': 'square'}, 'label': {'text': 'DUMP'}})
            return { "ohlc": ohlc_data, "active_sl_tp_line": active_sl_tp_line_data, "swing_high_points": swing_high_points,
                     "swing_low_points": swing_low_points, "annotations_points": annotations_points_chart,
                     "pair_name": pair_display_name, "last_updated_tv": hist[-1]['timestamp'].timestamp() * 1000 if hist else None,
                     "strategy_state_info": state, "config_info": cfg, "cooldown_active_bg": state.get("cooldown_active_on_chart", False) }
        # CHART_INTEGRATION_END (prepare_chart_data_for_pair definition)
        
        prepared_data = prepare_chart_data_for_pair(pair_id_from_request, temp_manager)
        if not prepared_data: return jsonify({"error": "Failed to process chart data"}), 500
        return jsonify(prepared_data)

def run_flask_server_thread():
    if not flask_app_instance: log_warning("Flask tidak terinstal, server chart tidak akan berjalan.", "SYSTEM_CHART"); return
    log_info("Memulai Flask server di http://localhost:5001 (atau http://<ip_termux>:5001)", pair_name="SYSTEM_CHART")
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
        log_error(f"{AnsiColors.RED}Tidak ada API key yang valid. Tidak dapat memulai.{AnsiColors.ENDC}"); input("Enter..."); return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif.{AnsiColors.ENDC}"); input("Enter..."); return

    animated_text_display("=========== EXORA V6 BOT (Python) + LIQUIDITY ALERTS START ===========", color=AnsiColors.HEADER, delay=0.005)
    
    local_crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')} ({config.get('timeframe','DEF')})"
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC}...", color=AnsiColors.MAGENTA, delay=0.01)
        local_crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }
        with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id])
        
        min_len_for_indicators_init = max(
            config.get('rsi_len', 20) + 1, 
            config.get('stoch_k', 41) + config.get('stoch_smooth_k', 25) + config.get('stoch_d', 3),
            (2 * config.get('swing_lookback', 100) + 1) if config.get('use_swing_filter') else 1,
            config.get('lowliq_len_py', 4), # Min untuk liquidity
            config.get('highliq_len_py', 5) # Min untuk liquidity
        ) + 50 # Buffer
        initial_candles_target = max(TARGET_BIG_DATA_CANDLES, min_len_for_indicators_init)
        log_info(f"Target data awal: {initial_candles_target} candles. Min untuk indikator/liq: {min_len_for_indicators_init - 50}", pair_name=config['pair_name'])
        
        initial_candles = []; initial_fetch_successful = False
        max_initial_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        initial_key_attempts_done = 0

        while initial_key_attempts_done < max_initial_key_attempts and not initial_fetch_successful:
            current_api_key_init = api_key_manager.get_current_key()
            if not current_api_key_init: log_error(f"BIG DATA: Semua API key habis (global) untuk {config['pair_name']}.", pair_name=config['pair_name']); break
            log_info(f"BIG DATA: Mencoba fetch awal {config['pair_name']} (Key Idx {api_key_manager.get_current_key_index()}, Att {initial_key_attempts_done + 1}/{max_initial_key_attempts})", pair_name=config['pair_name'])
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], initial_candles_target, config['exchange'], current_api_key_init, config['timeframe'], pair_name=config['pair_name'])
                initial_fetch_successful = True
            except APIKeyError:
                log_warning(f"BIG DATA: API Key (Idx {api_key_manager.get_current_key_index()}) gagal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): log_error(f"BIG DATA: Gagal beralih, semua key habis untuk {config['pair_name']}.", pair_name=config['pair_name']); break
            except requests.exceptions.RequestException as e_req_init: log_error(f"BIG DATA: Error Jaringan {config['pair_name']}: {e_req_init}.", pair_name=config['pair_name']); break
            except Exception as e_gen_init: log_exception(f"BIG DATA: Error Umum {config['pair_name']}: {e_gen_init}.", pair_name=config['pair_name']); break
            initial_key_attempts_done += 1

        if not initial_fetch_successful or not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal ambil data awal {config['pair_name']}. Dilewati.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            local_crypto_data_manager[pair_id].update({"data_fetch_failed_consecutively": max_initial_key_attempts +1, "last_attempt_after_all_keys_failed": datetime.now(), "big_data_collection_phase_active": False})
            with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(local_crypto_data_manager[pair_id]); continue

        local_crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {config['pair_name']}.", pair_name=config['pair_name'])

        if initial_candles:
            min_len_for_warmup_run = max(
                config.get('rsi_len', 20) + 1, config.get('stoch_k', 41) + config.get('stoch_smooth_k', 25) + config.get('stoch_d', 3) -1,
                (2 * config.get('swing_lookback',100) +1 if config.get('use_swing_filter') else 1),
                config.get('lowliq_len_py', 4), config.get('highliq_len_py', 5)
            )
            if len(initial_candles) >= min_len_for_warmup_run:
                log_info(f"Warm-up: Memproses {len(initial_candles) - min_len_for_warmup_run +1} candle historis untuk {config['pair_name']}...", pair_name=config['pair_name'])
                for i_warmup in range(min_len_for_warmup_run -1, len(initial_candles)):
                    historical_slice = initial_candles[:i_warmup+1]
                    if len(historical_slice) < min_len_for_warmup_run: continue
                    temp_state_warmup = local_crypto_data_manager[pair_id]["strategy_state"].copy()
                    local_crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_warmup, global_settings_dict, is_warmup=True)
                log_info(f"{AnsiColors.CYAN}Warm-up state untuk {config['pair_name']} selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else: log_warning(f"Tidak cukup data ({len(initial_candles)}/{min_len_for_warmup_run}) untuk warm-up {config['pair_name']}", pair_name=config['pair_name'])

        if len(local_crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            local_crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI {config['pair_name']}!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            # Notif email global untuk big data tercapai
            if not data_per_pair.get("big_data_email_sent", False) and api_settings.get("enable_global_email_notifications_for_key_switch", False):
                send_email_notification(
                    f"BIG DATA Tercapai: {pair_name_for_log}",
                    f"Target {TARGET_BIG_DATA_CANDLES} candle telah tercapai untuk {pair_name_for_log}. Bot kini beroperasi dalam mode live.",
                    { # Temp settings dict for global email
                        "enable_email_notifications": True,
                        "email_sender_address": api_settings.get("email_sender_address"),
                        "email_sender_app_password": api_settings.get("email_sender_app_password"),
                        "email_receiver_address": api_settings.get("email_receiver_address_admin")
                    },
                    pair_name_ctx_override=f"SYSTEM_BIG_DATA_{pair_name_for_log.split(' ')[0]}" # E.g. SYSTEM_BIG_DATA_BTC-USD
                )
                data_per_pair["big_data_email_sent"] = True # Tandai sudah kirim
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
                    tf_is_minute = 'm' in config_for_pair.get('timeframe','1h')
                    required_interval = 60 if tf_is_minute else 300 # Fetch lebih agresif saat big data
                
                if time_since_last_fetch < required_interval:
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval - time_since_last_fetch); continue
                
                log_info(f"Memproses {pair_name_for_log} (Interval: {required_interval}s)...", pair_name=pair_name_for_log)
                data_per_pair["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data_per_pair["all_candles_list"])
                
                header_color = AnsiColors.BOLD + (AnsiColors.MAGENTA if data_per_pair["big_data_collection_phase_active"] else AnsiColors.CYAN)
                header_text = f"--- {'BIG DATA' if data_per_pair['big_data_collection_phase_active'] else 'LIVE'} {pair_name_for_log} ({current_loop_time.strftime('%H:%M:%S') if not data_per_pair['big_data_collection_phase_active'] else f'{num_candles_before_fetch}/{TARGET_BIG_DATA_CANDLES}'}) | {num_candles_before_fetch} candles ---"
                animated_text_display(f"\n{header_text}", color=header_color)

                new_candles_batch = []; fetch_update_successful = False
                limit_fetch_update = 3 
                if data_per_pair["big_data_collection_phase_active"]:
                    needed = TARGET_BIG_DATA_CANDLES - num_candles_before_fetch
                    if needed <=0 : fetch_update_successful = True; limit_fetch_update = 3 
                    else: limit_fetch_update = min(needed, CRYPTOCOMPARE_MAX_LIMIT)
                
                if limit_fetch_update > 0: 
                    max_upd_key_attempts = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                    upd_key_attempts_done = 0; orig_api_key_idx_upd = api_key_manager.get_current_key_index()
                    while upd_key_attempts_done < max_upd_key_attempts and not fetch_update_successful:
                        current_api_key_upd = api_key_manager.get_current_key()
                        if not current_api_key_upd: log_error(f"UPDATE: Semua API key habis (global) {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        log_info(f"UPDATE: Fetch {limit_fetch_update} candle {pair_name_for_log} (Key Idx {api_key_manager.get_current_key_index()}, Att {upd_key_attempts_done + 1}/{max_upd_key_attempts})", pair_name=pair_name_for_log)
                        try:
                            new_candles_batch = fetch_candles(config_for_pair['symbol'], config_for_pair['currency'], limit_fetch_update, config_for_pair['exchange'], current_api_key_upd, config_for_pair['timeframe'], pair_name=pair_name_for_log)
                            fetch_update_successful = True; data_per_pair["data_fetch_failed_consecutively"] = 0; any_data_fetched_this_cycle = True
                            if api_key_manager.get_current_key_index() != orig_api_key_idx_upd: log_info(f"UPDATE: Fetch berhasil dgn key idx {api_key_manager.get_current_key_index()} setelah retry {pair_name_for_log}.", pair_name=pair_name_for_log)
                        except APIKeyError:
                            log_warning(f"UPDATE: API Key (Idx {api_key_manager.get_current_key_index()}) gagal {pair_name_for_log}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1
                            if not api_key_manager.switch_to_next_key(): log_error(f"UPDATE: Gagal beralih, semua key habis {pair_name_for_log}.", pair_name=pair_name_for_log); break
                        except requests.exceptions.RequestException as e_r_u: log_error(f"UPDATE: Error Jaringan {pair_name_for_log}: {e_r_u}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break 
                        except Exception as e_g_u: log_exception(f"UPDATE: Error Umum {pair_name_for_log}: {e_g_u}.", pair_name=pair_name_for_log); data_per_pair["data_fetch_failed_consecutively"] +=1; break
                        upd_key_attempts_done += 1
                else: fetch_update_successful = True
                
                if data_per_pair.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() or 1) +1: data_per_pair["last_attempt_after_all_keys_failed"] = datetime.now() 
                if not fetch_update_successful and limit_fetch_update > 0 :
                     log_error(f"{AnsiColors.RED}Gagal ambil update {pair_name_for_log} setelah semua upaya.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                     min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval)
                     with lock_ref: shared_dm_ref[pair_id] = copy.deepcopy(data_per_pair); continue
                
                if new_candles_batch:
                    merged_dict = {c['timestamp']: c for c in data_per_pair["all_candles_list"]}
                    added_c, updated_c = 0,0
                    for candle_n in new_candles_batch:
                        ts_n = candle_n['timestamp']
                        if ts_n not in merged_dict: merged_dict[ts_n] = candle_n; added_c +=1
                        elif merged_dict[ts_n]['close']!=candle_n['close'] or merged_dict[ts_n]['high']!=candle_n['high'] : merged_dict[ts_n]=candle_n;updated_c+=1
                    data_per_pair["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c_s: c_s['timestamp'])
                    if added_c + updated_c > 0: log_info(f"{added_c} baru, {updated_c} diupdate untuk {pair_name_for_log}.", pair_name=pair_name_for_log)
                
                if data_per_pair["big_data_collection_phase_active"]:
                    if len(data_per_pair["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        data_per_pair["big_data_collection_phase_active"] = False
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1) 
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI {pair_name_for_log}!{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                        if not data_per_pair.get("big_data_email_sent", False) and api_settings.get("enable_global_email_notifications_for_key_switch", False):
                             send_email_notification(f"BIG DATA Tercapai: {pair_name_for_log}", f"Target {TARGET_BIG_DATA_CANDLES} tercapai untuk {pair_name_for_log}. Mode live.", 
                                                     {"enable_email_notifications": True, **api_settings}, 
                                                     pair_name_ctx_override=f"SYSTEM_BD_{pair_name_for_log.split(' ')[0]}")
                             data_per_pair["big_data_email_sent"] = True
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({pair_name_for_log}) ----------{AnsiColors.ENDC}", pair_name=pair_name_for_log)
                elif len(data_per_pair["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 200: 
                    data_per_pair["all_candles_list"] = data_per_pair["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 100):]

                min_len_for_logic_live = max(
                    config_for_pair.get('rsi_len', 20) + 1, config_for_pair.get('stoch_k', 41) + config_for_pair.get('stoch_smooth_k', 25) + config_for_pair.get('stoch_d', 3) -1,
                    (2 * config_for_pair.get('swing_lookback',100) +1 if config_for_pair.get('use_swing_filter') else 1),
                    config_for_pair.get('lowliq_len_py', 4), config_for_pair.get('highliq_len_py', 5)
                )
                if len(data_per_pair["all_candles_list"]) >= min_len_for_logic_live:
                    log_debug(f"Menjalankan logika Exora V6 + Liq untuk {pair_name_for_log}...", pair_name=pair_name_for_log)
                    data_per_pair["strategy_state"] = run_strategy_logic(data_per_pair["all_candles_list"], config_for_pair, data_per_pair["strategy_state"], global_settings_dict, is_warmup=False)
                else: log_debug(f"Belum cukup data ({len(data_per_pair['all_candles_list'])}/{min_len_for_logic_live}) untuk logika {pair_name_for_log}", pair_name=pair_name_for_log)
                
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
    except Exception as e_main_loop: log_exception(f"{AnsiColors.RED}Error loop utama: {e_main_loop}{AnsiColors.ENDC}", pair_name="SYSTEM")
    finally: animated_text_display(f"{AnsiColors.HEADER}=========== EXORA V6 BOT + LIQUIDITY ALERTS STOP ==========={AnsiColors.ENDC}",color=AnsiColors.HEADER); input("Tekan Enter untuk kembali...")


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    is_flask_running = any(t.name == "FlaskServerThread" for t in threading.enumerate())
    if not is_flask_running and flask_app_instance: # Cek jika flask_app_instance ada
        flask_thread = threading.Thread(target=run_flask_server_thread, daemon=True, name="FlaskServerThread")
        flask_thread.start()
    elif flask_app_instance:
        log_info("Flask server sudah berjalan.", "SYSTEM_CHART")

    while True:
        clear_screen_animated()
        animated_text_display("========= Exora V6 Spot Bot (Python) + Liquidity Alerts =========", color=AnsiColors.HEADER)
        active_cfgs = [c for c in settings.get("cryptos",[]) if c.get("enabled",True)]
        pick_title_main = f"--- Crypto Aktif ({len(active_cfgs)}) ---\n" + ("".join([f"  {i+1}. {c.get('symbol','?')}-{c.get('currency','?')} ({c.get('timeframe','?')})\n" for i,c in enumerate(active_cfgs)]) if active_cfgs else "Tidak ada konfigurasi crypto aktif.\n")
        
        api_s_main = settings.get("api_settings", {})
        pk_disp = api_s_main.get('primary_key','N/A'); pk_disp = ("..."+pk_disp[-5:]) if len(pk_disp)>10 and pk_disp not in ["YOUR_PRIMARY_KEY", "N/A"] else pk_disp
        pick_title_main += f"-----------------------------------------------\nPrimary API Key: {pk_disp}\nChart Server: http://localhost:5001 (jika Flask aktif)\n-----------------------------------------------\nPilih Opsi:"
        
        main_opts = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        try:
            options_for_pick_main = [opt[:70] + ('...' if len(opt) > 70 else '') for opt in main_opts]
            _, main_idx = pick(options_for_pick_main, pick_title_main, indicator='=>')
        except Exception as e_pick_main: 
            log_warning(f"Pick library error: {e_pick_main}. Gunakan input angka.")
            print(pick_title_main)
            for i, opt_disp_main in enumerate(options_for_pick_main): print(f"{i}. {opt_disp_main}")
            try:
                main_idx = int(input("Masukkan nomor pilihan: "))
                if not (0 <= main_idx < len(options_for_pick_main)): raise ValueError("Diluar range")
            except ValueError: print("Input tidak valid."); time.sleep(1); continue

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
