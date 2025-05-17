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
try:
    from pick import pick # Pastikan library pick terinstal: pip install pick
except ImportError:
    print("Error: Library 'pick' tidak ditemukan. Silakan install dengan 'pip install pick'")
    sys.exit(1)
try:
    import getpass # Untuk input password
except ImportError:
    pass


# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m' # Ungu Cerah
    BLUE = '\033[94m'   # Biru
    GREEN = '\033[92m'  # Hijau
    ORANGE = '\033[93m' # Oranye/Kuning (Warning)
    RED = '\033[91m'    # Merah (Error)
    ENDC = '\033[0m'    # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[96m'   # Cyan
    MAGENTA = '\033[35m'# Magenta
    YELLOW_BG = '\033[43m' # Background Kuning
    WHITE = '\033[97m'  # Putih Cerah
    GREY = '\033[90m'   # Abu-abu

# --- HELPER FUNCTION FOR TERMINAL WIDTH ---
def get_terminal_width(default_width=80):
    """Mendapatkan lebar terminal saat ini atau default jika tidak dapat ditentukan."""
    try:
        # sys.stdout.isatty() mengecek apakah stdout adalah TTY (terminal)
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return os.get_terminal_size().columns
        return default_width
    except (OSError, AttributeError):
        # OSError jika get_terminal_size gagal, AttributeError jika isatty tidak ada
        return default_width

# --- ANIMATION HELPER FUNCTIONS ---

def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True, bold=False, center=False, width=None):
    """Menampilkan teks dengan efek ketik per karakter, opsi bold, dan center."""
    if width is None:
        width = get_terminal_width()

    if center:
        # Hitung padding berdasarkan panjang teks non-ANSI
        actual_text_length = len(text) # Ini sederhana, untuk teks berwarna, perlu dihitung tanpa ANSI
        try:
            # Untuk centering yang lebih akurat dengan ANSI codes, kita perlu menghitung panjang tanpa ANSI
            # Namun, untuk kesederhanaan, kita akan mengabaikannya di sini,
            # atau bisa menggunakan library eksternal jika sangat penting.
            # Untuk sekarang, kita asumsikan 'text' adalah teks visualnya.
            # Ini mungkin tidak sempurna jika 'text' sudah mengandung ANSI codes.
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            actual_text_length = len(ansi_escape.sub('', text))
        except ImportError:
            pass # Gunakan len(text) jika re tidak tersedia/gagal

        padding = (width - actual_text_length) // 2
        padding = max(0, padding) 
        sys.stdout.write(" " * padding)
        sys.stdout.flush()

    for char_index, char in enumerate(text):
        output_char = AnsiColors.BOLD if bold else ''
        output_char += color if color else ''
        output_char += char
        output_char += AnsiColors.ENDC 
        
        sys.stdout.write(output_char)
        sys.stdout.flush()
        time.sleep(delay)

    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing...", color=AnsiColors.MAGENTA):
    """Menampilkan animasi spinner sederhana untuk durasi tertentu."""
    spinner_chars = ['◢', '◣', '◤', '◥'] 
    start_time = time.time()
    idx = 0
    term_width = get_terminal_width()

    while (time.time() - start_time) < duration_seconds:
        display_message = message[:term_width - 7] 
        prefix = color + AnsiColors.BOLD
        suffix = AnsiColors.ENDC
        sys.stdout.write(f"\r{prefix}{display_message} {spinner_chars[idx % len(spinner_chars)]} {suffix}")
        sys.stdout.flush()
        time.sleep(0.15) 
        idx += 1
    sys.stdout.write(f"\r{' ' * (term_width -1)}\r") 
    sys.stdout.write(AnsiColors.ENDC) 
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='❚', print_end="\r"):
    """Membuat dan menampilkan progress bar sederhana, disesuaikan untuk Termux."""
    term_width_val = get_terminal_width()
    # Dynamic length adjustment
    # Perhitungan panjang non-bar perlu lebih hati-hati jika prefix/suffix berwarna
    # Untuk sementara, kita sederhanakan
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    len_prefix_no_ansi = len(ansi_escape.sub('', prefix))
    len_suffix_no_ansi = len(ansi_escape.sub('', suffix))

    non_bar_len = len_prefix_no_ansi + len_suffix_no_ansi + 12 # Ruang untuk persentase, spasi, dan buffer
    
    # Pastikan length positif dan tidak terlalu besar
    effective_length = max(10, min(length, term_width_val - non_bar_len))
    if effective_length < 0 : effective_length = 10 # Fallback jika perhitungan salah

    percent_val = (100 * (iteration / float(total))) if total > 0 else 0
    percent_str = ("{0:." + str(decimals) + "f}").format(percent_val)
    
    filled_length = int(effective_length * iteration // total) if total > 0 else 0
    bar_chars = fill * filled_length + '░' * (effective_length - filled_length) 

    bar_color = AnsiColors.GREEN
    if percent_val < 30: bar_color = AnsiColors.ORANGE
    elif percent_val < 70: bar_color = AnsiColors.YELLOW_BG + AnsiColors.BLUE
    if iteration == total and total > 0 : bar_color = AnsiColors.GREEN + AnsiColors.BOLD

    progress_line = f'{prefix} {bar_color}|{bar_chars}| {percent_str}%{AnsiColors.ENDC} {suffix}'
    
    sys.stdout.write(f"\r{progress_line:{term_width_val}}") # Pad to full width
    sys.stdout.flush()

    if iteration == total and total > 0:
        sys.stdout.write('\n')
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
console_formatter_template = (
    f"{AnsiColors.GREY}%(asctime)s{AnsiColors.ENDC} - "
    f"%(log_color)s{AnsiColors.BOLD}%(levelname)-8s{AnsiColors.ENDC} - "
    f"{AnsiColors.MAGENTA}[%(pair_name)s]{AnsiColors.ENDC} - "
    f"%(log_color)s%(message_content)s{AnsiColors.ENDC}"
)

class ColoredFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: AnsiColors.GREY,
        logging.INFO: AnsiColors.CYAN,
        logging.WARNING: AnsiColors.ORANGE,
        logging.ERROR: AnsiColors.RED,
        logging.CRITICAL: AnsiColors.BOLD + AnsiColors.RED,
    }
    def format(self, record):
        record.log_color = self.LOG_COLORS.get(record.levelno, AnsiColors.WHITE)
        level_emoji_map = {
            logging.INFO: "ℹ️ ", logging.WARNING: "⚠️ ",
            logging.ERROR: "❌ ", logging.CRITICAL: "🔥 "
        }
        original_message = record.getMessage()
        record.message_content = original_message
        if not original_message.startswith('\033['):
             record.message_content = level_emoji_map.get(record.levelno, "") + original_message
        formatter = logging.Formatter(console_formatter_template, datefmt='%H:%M:%S')
        return formatter.format(record)

ch.setFormatter(ColoredFormatter())
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

def clear_screen_animated():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY":
            self.keys.append(primary_key)
        if recovery_keys_list: 
            self.keys.extend([k for k in recovery_keys_list if isinstance(k, str) and k.strip()])
        self.current_index = 0
        self.global_email_settings = global_settings_for_email if global_settings_for_email is not None else {}
        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) dikonfigurasi.")

    def get_current_key(self):
        if not self.keys or self.current_index >= len(self.keys):
            return None
        return self.keys[self.current_index]

    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key = self.keys[self.current_index]
            new_key_display = new_key[:5] + "..." + new_key[-3:] if len(new_key) > 8 else new_key
            log_info(f"{AnsiColors.ORANGE}🔑 Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti"
                email_body = (f"Skrip trading otomatis mengganti API key CryptoCompare.\n"
                              f"Key sebelumnya mungkin limit/tidak valid.\n"
                              f"Sekarang menggunakan API key index: {self.current_index}\n"
                              f"Key: ...{new_key_display[-8:]}\n"
                              f"Periksa status API key Anda di CryptoCompare.")
                cfg_for_email = {
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(cfg_for_email.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                     send_email_notification(email_subject, email_body, cfg_for_email, pair_name_override="SYSTEM_KEY_SWITCH")
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi ganti API key.")
            return new_key
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}🆘 SEMUA API KEY HABIS/GAGAL! Tidak dapat ambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "KRITIS: SEMUA API Key CryptoCompare Gagal!"
                email_body = (f"Skrip trading telah mencoba semua API key CryptoCompare dan semuanya gagal.\n"
                              f"Skrip tidak dapat lagi mengambil data harga.\n"
                              f"Segera periksa akun CryptoCompare Anda dan konfigurasi API key.")
                cfg_for_email_critical = {
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(cfg_for_email_critical.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, cfg_for_email_critical, pair_name_override="SYSTEM_KEY_FAIL")
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS API key habis.")
            return None

    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP & EMAIL ---
def play_notification_sound(count=1, interval=0.15, freq=1000, dur=200):
    try:
        for _ in range(count):
            if sys.platform == "win32":
                import winsound
                winsound.Beep(freq, dur) 
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
            if count > 1: time.sleep(interval)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email, pair_name_override=None):
    if not settings_for_email.get("enable_email_notifications", False): return

    sender = settings_for_email.get("email_sender_address")
    password = settings_for_email.get("email_sender_app_password")
    receiver = settings_for_email.get("email_receiver_address")
    
    ctx_pair_name = pair_name_override if pair_name_override else settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))

    if not all([sender, password, receiver]):
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=ctx_pair_name)
        return

    msg = MIMEText(body_text)
    msg['Subject'] = f"📢 {subject}"
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        log_info(f"{AnsiColors.GREEN}📧 Notifikasi email '{subject[:30]}...' berhasil dikirim ke {receiver}{AnsiColors.ENDC}", pair_name=ctx_pair_name)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=ctx_pair_name)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {"id": str(uuid.uuid4()),"enabled": True,"symbol": "BTC","currency": "USD","exchange": "CCCAGG","timeframe": "hour","refresh_interval_seconds": 60,"left_strength": 50,"right_strength": 150,"profit_target_percent_activation": 5.0,"trailing_stop_gap_percent": 5.0,"emergency_sl_percent": 10.0,"enable_secure_fib": True,"secure_fib_check_price": "Close","enable_email_notifications": False,"email_sender_address": "","email_sender_app_password": "","email_receiver_address": ""}

def load_settings():
    defaults = {"primary_key": "YOUR_PRIMARY_KEY","recovery_keys": [],"enable_global_email_notifications_for_key_switch": False,"email_sender_address": "","email_sender_app_password": "","email_receiver_address_admin": ""}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            if "api_settings" not in settings: settings["api_settings"] = defaults.copy()
            else: settings["api_settings"] = {**defaults, **settings["api_settings"]} 
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list): settings["cryptos"] = []
            for cfg in settings["cryptos"]: 
                cfg.setdefault("id", str(uuid.uuid4()))
                cfg.setdefault("enabled", True)
            return settings
        except json.JSONDecodeError:
            log_error(f"Error baca {SETTINGS_FILE}. Membuat file default baru.")
            return {"api_settings": defaults.copy(), "cryptos": []}
        except Exception as e:
            log_error(f"Error load_settings: {e}. Menggunakan default.")
            return {"api_settings": defaults.copy(), "cryptos": [get_default_crypto_config()]} 
    else: 
        log_info(f"File {SETTINGS_FILE} tidak ditemukan. Membuat dengan default.")
        new_settings = {"api_settings": defaults.copy(), "cryptos": []}
        save_settings(new_settings) 
        return new_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.GREEN}💾 Pengaturan berhasil disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}")

def termux_input(prompt_text, current_value="", color=AnsiColors.CYAN, highlight_color=AnsiColors.WHITE, is_password=False, prefix_icon="▸"):
    prompt_prefix = f"{color}{AnsiColors.BOLD}{prefix_icon} {prompt_text}{AnsiColors.ENDC}"
    default_suffix = f" [{AnsiColors.ORANGE}{current_value}{AnsiColors.ENDC}]" if current_value or isinstance(current_value, (bool, int, float)) else ""
    input_indicator = f": {highlight_color}"
    full_prompt = f"{prompt_prefix}{default_suffix}{input_indicator}"
    user_input = ""
    if is_password:
        try: 
            if 'getpass' in sys.modules: user_input = getpass.getpass(prompt=full_prompt)
            else: raise ImportError 
        except (ImportError, RuntimeError, EOFError): 
            print(full_prompt, end=""); sys.stdout.flush()
            user_input = sys.stdin.readline().strip()
    else:
        print(full_prompt, end=""); sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
    sys.stdout.write(AnsiColors.ENDC) 
    return user_input or str(current_value) 

def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_cfg = current_config.copy() 
    sym = new_cfg.get('symbol','BARU'); cur = new_cfg.get('currency','BARU')
    animated_text_display(f"--- ⚙️ Konfigurasi Crypto ({sym}-{cur}) ---", color=AnsiColors.HEADER, bold=True, center=True)
    print(AnsiColors.GREY + "Tekan Enter untuk menggunakan nilai default dalam []" + AnsiColors.ENDC)
    print(AnsiColors.GREY + ("-" * get_terminal_width(60)) + AnsiColors.ENDC)

    en_raw = termux_input("Aktifkan pair ini?", str(new_cfg.get('enabled',True)).lower()).lower()
    new_cfg["enabled"] = True if en_raw == 'true' else (False if en_raw == 'false' else new_cfg.get('enabled',True))
    print()
    new_cfg["symbol"] = termux_input("Simbol Crypto Dasar (BTC)", new_cfg.get('symbol','BTC')).upper()
    new_cfg["currency"] = termux_input("Simbol Mata Uang Quote (USDT)", new_cfg.get('currency','USD')).upper()
    new_cfg["exchange"] = termux_input("Exchange (Binance, CCCAGG)", new_cfg.get('exchange','CCCAGG'))
    print()
    tf_raw = termux_input("Timeframe (minute/hour/day)", new_cfg.get('timeframe','hour')).lower()
    if tf_raw in ['minute', 'hour', 'day']: new_cfg["timeframe"] = tf_raw
    else: log_warning(f"Timeframe tidak valid, menggunakan '{new_cfg.get('timeframe','hour')}'")
    ref_raw = termux_input(f"Interval Refresh (detik, min {MIN_REFRESH_INTERVAL_AFTER_BIG_DATA})", str(new_cfg.get('refresh_interval_seconds',60)))
    try: new_cfg["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(ref_raw))
    except ValueError: log_warning(f"Input refresh interval tidak valid, menggunakan default.") 
    print()
    animated_text_display("-- Parameter Pivot --", color=AnsiColors.HEADER, bold=True, delay=0.005)
    try:
        new_cfg["left_strength"] = int(termux_input("Left Strength", str(new_cfg.get('left_strength',50))))
        new_cfg["right_strength"] = int(termux_input("Right Strength", str(new_cfg.get('right_strength',150))))
    except ValueError: log_warning(f"Input strength tidak valid, menggunakan default.")
    print()
    animated_text_display("-- Parameter Trading --", color=AnsiColors.HEADER, bold=True, delay=0.005)
    try:
        new_cfg["profit_target_percent_activation"] = float(termux_input("Profit % Aktivasi Trailing TP", str(new_cfg.get('profit_target_percent_activation',5.0))))
        new_cfg["trailing_stop_gap_percent"] = float(termux_input("Gap Trailing TP %", str(new_cfg.get('trailing_stop_gap_percent',5.0))))
        new_cfg["emergency_sl_percent"] = float(termux_input("Emergency SL %", str(new_cfg.get('emergency_sl_percent',10.0)), color=AnsiColors.ORANGE))
    except ValueError: log_warning(f"Input parameter trading tidak valid, menggunakan default.")
    print()
    animated_text_display("-- Fitur Secure FIB --", color=AnsiColors.HEADER, bold=True, delay=0.005)
    sf_en_raw = termux_input("Aktifkan Secure FIB?", str(new_cfg.get('enable_secure_fib',True)).lower()).lower()
    new_cfg["enable_secure_fib"] = True if sf_en_raw == 'true' else (False if sf_en_raw == 'false' else new_cfg.get('enable_secure_fib',True))
    sf_price_raw = termux_input("Harga Cek Secure FIB (Close/High)", new_cfg.get('secure_fib_check_price','Close')).capitalize()
    if sf_price_raw in ["Close", "High"]: new_cfg["secure_fib_check_price"] = sf_price_raw
    else: log_warning(f"Pilihan harga Secure FIB tidak valid, menggunakan '{new_cfg.get('secure_fib_check_price','Close')}'")
    print()
    animated_text_display("-- Notifikasi Email (Pair Ini) --", color=AnsiColors.HEADER, bold=True, delay=0.005)
    print(f"{AnsiColors.GREY}Kosongkan jika ingin pakai pengaturan email global (jika aktif).{AnsiColors.ENDC}")
    email_en_raw = termux_input("Aktifkan Email (pair ini)?", str(new_cfg.get('enable_email_notifications',False)).lower()).lower()
    new_cfg["enable_email_notifications"] = True if email_en_raw == 'true' else (False if email_en_raw == 'false' else new_cfg.get('enable_email_notifications',False))
    new_cfg["email_sender_address"] = termux_input("Email Pengirim (Gmail)", new_cfg.get('email_sender_address',''))
    new_cfg["email_sender_app_password"] = termux_input("App Password Pengirim", new_cfg.get('email_sender_app_password',''), is_password=True)
    new_cfg["email_receiver_address"] = termux_input("Email Penerima", new_cfg.get('email_receiver_address',''))
    print()
    return new_cfg

def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        pk_val = api_s.get('primary_key', ''); pk_disp = "BELUM DIATUR"
        if pk_val and pk_val not in ["YOUR_PRIMARY_KEY", "YOUR_API_KEY_HERE"]: pk_disp = pk_val[:5] + "..." + pk_val[-3:] if len(pk_val) > 8 else pk_val
        rec_keys = [k for k in api_s.get('recovery_keys', []) if isinstance(k, str) and k.strip()]; num_rec_keys = len(rec_keys)

        term_width = get_terminal_width(60)
        title = [AnsiColors.HEADER + AnsiColors.BOLD + "╔" + "═" * (term_width - 2) + "╗" + AnsiColors.ENDC,
                 AnsiColors.HEADER + AnsiColors.BOLD + "║" + "⚙️ PENGATURAN UTAMA".center(term_width - 2) + "║" + AnsiColors.ENDC,
                 AnsiColors.HEADER + AnsiColors.BOLD + "╠" + "═" * (term_width - 2) + "╣" + AnsiColors.ENDC]
        pk_color = AnsiColors.GREEN if pk_disp != "BELUM DIATUR" else AnsiColors.ORANGE
        title.append(AnsiColors.CYAN + f"  🔑 Primary API Key : {pk_color}{pk_disp}{AnsiColors.ENDC}")
        rk_color = AnsiColors.GREEN if num_rec_keys > 0 else AnsiColors.ORANGE
        title.append(AnsiColors.CYAN + f"  🔄 Recovery Keys   : {rk_color}{num_rec_keys} tersimpan{AnsiColors.ENDC}")
        g_email_active = api_s.get('enable_global_email_notifications_for_key_switch', False) and api_s.get('email_sender_address') and api_s.get('email_receiver_address_admin')
        g_email_stat = "Aktif" if g_email_active else "Nonaktif"; g_email_color = AnsiColors.GREEN if g_email_active else AnsiColors.ORANGE
        title.append(AnsiColors.CYAN + f"  📧 Email Global Sys: {g_email_color}{g_email_stat}{AnsiColors.ENDC}")
        title.append(AnsiColors.GREY + "  " + "-" * (term_width - 6) + AnsiColors.ENDC)
        title_pick = "\n".join(title) + "\n"
        title_pick += AnsiColors.MAGENTA + AnsiColors.BOLD + " Daftar Konfigurasi Crypto:" + AnsiColors.ENDC + "\n"
        if not current_settings["cryptos"]: title_pick += AnsiColors.ORANGE + "  (Belum ada konfigurasi crypto)\n" + AnsiColors.ENDC
        for i, cfg in enumerate(current_settings["cryptos"]):
            stat_col = AnsiColors.GREEN if cfg.get('enabled', True) else AnsiColors.RED
            stat_txt = "Aktif" if cfg.get('enabled', True) else "Nonaktif"
            title_pick += f"  {AnsiColors.WHITE}{i+1}. {AnsiColors.CYAN}{cfg['symbol']}-{cfg['currency']}{AnsiColors.ENDC} {AnsiColors.GREY}({cfg['timeframe']}){AnsiColors.ENDC} - {stat_col}{stat_txt}{AnsiColors.ENDC}\n"
        title_pick += AnsiColors.GREY + "  " + "-" * (term_width - 6) + AnsiColors.ENDC + "\n"
        title_pick += AnsiColors.MAGENTA + AnsiColors.BOLD + " Pilih tindakan:" + AnsiColors.ENDC
        options = ["🔑 Atur Primary API Key", "🔄 Kelola Recovery API Keys", "📧 Atur Email Global Notifikasi Sistem",
                   "➕ Tambah Konfigurasi Crypto", "✏️ Ubah Konfigurasi Crypto", "🗑️ Hapus Konfigurasi Crypto",
                   "🚪 Kembali ke Menu Utama"]
        try: opt_txt, idx = pick(options, title_pick, indicator=f'{AnsiColors.GREEN}▶{AnsiColors.ENDC}', default_index=0)
        except (KeyboardInterrupt, Exception) as e_pick: 
            log_warning(f"Pemilihan dibatalkan/error di menu pengaturan: {e_pick}")
            if isinstance(e_pick, KeyboardInterrupt): return current_settings 
            show_spinner(1.5, "Error pick, kembali...", color=AnsiColors.RED); return current_settings
        clear_screen_animated()
        try:
            if idx == 0: 
                animated_text_display("--- 🔑 Atur Primary API Key ---", color=AnsiColors.HEADER, bold=True, center=True)
                new_pk = termux_input("Masukkan Primary API Key CryptoCompare baru", api_s.get('primary_key','YOUR_PRIMARY_KEY'), is_password=True).strip()
                api_s["primary_key"] = new_pk if new_pk else "YOUR_PRIMARY_KEY"; save_settings(current_settings)
            elif idx == 1: 
                while True: 
                    clear_screen_animated(); api_s_rec = current_settings.get("api_settings", {}) 
                    current_rec_keys = [k for k in api_s_rec.get('recovery_keys', []) if isinstance(k, str) and k.strip()]
                    rec_title_lines = [AnsiColors.HEADER + AnsiColors.BOLD + "╔" + "═" * (term_width - 2) + "╗" + AnsiColors.ENDC,
                                       AnsiColors.HEADER + AnsiColors.BOLD + "║" + "🔄 KELOLA RECOVERY API KEYS".center(term_width - 2) + "║" + AnsiColors.ENDC,
                                       AnsiColors.HEADER + AnsiColors.BOLD + "╠" + "═" * (term_width - 2) + "╣" + AnsiColors.ENDC]
                    if not current_rec_keys: rec_title_lines.append(AnsiColors.ORANGE + "  (Tidak ada recovery key tersimpan)" + AnsiColors.ENDC)
                    else:
                        for i, rk in enumerate(current_rec_keys):
                            rk_disp = rk[:5] + "..." + rk[-3:] if len(rk) > 8 else rk
                            rec_title_lines.append(f"  {AnsiColors.WHITE}{i+1}. {AnsiColors.CYAN}{rk_disp}{AnsiColors.ENDC}")
                    rec_title_lines.append(AnsiColors.GREY + "  " + "-" * (term_width - 6) + AnsiColors.ENDC)
                    rec_title_lines.append(AnsiColors.MAGENTA + AnsiColors.BOLD + " Pilih tindakan:" + AnsiColors.ENDC)
                    rec_pick_title = "\n".join(rec_title_lines) + "\n"; rec_options = ["➕ Tambah Recovery Key", "🗑️ Hapus Recovery Key", "🚪 Kembali"]
                    try: rec_opt_text, rec_idx = pick(rec_options, rec_pick_title, indicator=f'{AnsiColors.GREEN}▶{AnsiColors.ENDC}')
                    except (KeyboardInterrupt, Exception) as e_rec_pick:
                        if isinstance(e_rec_pick, KeyboardInterrupt): break 
                        log_warning(f"Error pick di menu recovery key: {e_rec_pick}"); show_spinner(1, "Error, kembali..."); break
                    clear_screen_animated()
                    if rec_idx == 0: 
                        animated_text_display("-- ➕ Tambah Recovery Key --", color=AnsiColors.HEADER, bold=True, center=True)
                        new_rk_val = termux_input("Masukkan Recovery API Key baru", is_password=True).strip()
                        if new_rk_val and new_rk_val not in current_rec_keys:
                            current_rec_keys.append(new_rk_val); api_s_rec['recovery_keys'] = current_rec_keys
                            current_settings['api_settings']['recovery_keys'] = current_rec_keys; save_settings(current_settings)
                            log_info("Recovery key ditambahkan.")
                        elif not new_rk_val: log_warning("Input tidak boleh kosong.")
                        else: log_warning("Recovery key tersebut sudah ada.")
                    elif rec_idx == 1: 
                        animated_text_display("-- 🗑️ Hapus Recovery Key --", color=AnsiColors.HEADER, bold=True, center=True)
                        if not current_rec_keys: log_warning("Tidak ada recovery key untuk dihapus.")
                        else:
                            del_opts_rk = [f"{i+1}. {rk[:5]}...{rk[-3:] if len(rk)>8 else rk}" for i, rk in enumerate(current_rec_keys)] + ["Batal"]
                            del_title_rk = AnsiColors.MAGENTA + AnsiColors.BOLD + "Pilih key yang akan dihapus:" + AnsiColors.ENDC
                            try:
                                _, del_idx_rk = pick(del_opts_rk, del_title_rk, indicator=f'{AnsiColors.RED}▶{AnsiColors.ENDC}')
                                if del_idx_rk < len(current_rec_keys): 
                                    removed_rk = current_rec_keys.pop(del_idx_rk); api_s_rec['recovery_keys'] = current_rec_keys
                                    current_settings['api_settings']['recovery_keys'] = current_rec_keys; save_settings(current_settings)
                                    log_info(f"Recovery key '{removed_rk[:5]}...' dihapus.")
                                else: log_info("Penghapusan dibatalkan.")
                            except (KeyboardInterrupt, Exception) as e_del_rk_pick: 
                                if isinstance(e_del_rk_pick, KeyboardInterrupt): log_info("Penghapusan dibatalkan.")
                                else: log_warning(f"Error pick hapus recovery key: {e_del_rk_pick}")
                                show_spinner(1, "Kembali...")
                    elif rec_idx == 2: break 
                    if rec_idx in [0,1]: show_spinner(1.5, "Memproses...", color=AnsiColors.BLUE)
            elif idx == 2: 
                animated_text_display("-- 📧 Pengaturan Email Global --", color=AnsiColors.HEADER, bold=True, center=True)
                print(AnsiColors.GREY + "Email untuk notifikasi sistem (ganti API key, dll)." + AnsiColors.ENDC); print(AnsiColors.GREY + ("-" * term_width) + AnsiColors.ENDC)
                en_g_email_raw = termux_input("Aktifkan email global?", str(api_s.get('enable_global_email_notifications_for_key_switch',False)).lower()).lower()
                api_s['enable_global_email_notifications_for_key_switch'] = True if en_g_email_raw == 'true' else (False if en_g_email_raw == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = termux_input("Email Pengirim Global (Gmail)", api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = termux_input("App Password Pengirim Global", api_s.get('email_sender_app_password',''), is_password=True)
                api_s['email_receiver_address_admin'] = termux_input("Email Penerima Notifikasi Sistem (Admin)", api_s.get('email_receiver_address_admin','')); save_settings(current_settings)
            elif idx == 3: 
                new_crypto = _prompt_crypto_config(get_default_crypto_config()) 
                current_settings["cryptos"].append(new_crypto); save_settings(current_settings)
                log_info(f"Konfigurasi {new_crypto['symbol']}-{new_crypto['currency']} ditambahkan.")
            elif idx == 4: 
                if not current_settings["cryptos"]: log_warning("Tidak ada konfigurasi untuk diubah.")
                else:
                    animated_text_display("-- ✏️ Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER, bold=True, center=True)
                    crypto_opts_edit = [f"{i+1}. {c['symbol']}-{c['currency']}" for i, c in enumerate(current_settings["cryptos"])] + ["Batal"]
                    edit_title = AnsiColors.MAGENTA + AnsiColors.BOLD + "Pilih crypto yang akan diubah:" + AnsiColors.ENDC
                    try:
                        _, idx_edit = pick(crypto_opts_edit, edit_title, indicator=f'{AnsiColors.GREEN}▶{AnsiColors.ENDC}')
                        if idx_edit < len(current_settings["cryptos"]):
                            current_settings["cryptos"][idx_edit] = _prompt_crypto_config(current_settings["cryptos"][idx_edit]); save_settings(current_settings)
                            log_info(f"Konfigurasi {current_settings['cryptos'][idx_edit]['symbol']}-{current_settings['cryptos'][idx_edit]['currency']} diubah.")
                        else: log_info("Perubahan dibatalkan.")
                    except (KeyboardInterrupt, Exception) as e_edit_c_pick: 
                        if isinstance(e_edit_c_pick, KeyboardInterrupt): log_info("Perubahan dibatalkan.")
                        else: log_warning(f"Error pick ubah crypto: {e_edit_c_pick}")
                        show_spinner(1, "Kembali...")
            elif idx == 5: 
                if not current_settings["cryptos"]: log_warning("Tidak ada konfigurasi untuk dihapus.")
                else:
                    animated_text_display("-- 🗑️ Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER, bold=True, center=True)
                    crypto_opts_del = [f"{i+1}. {c['symbol']}-{c['currency']}" for i, c in enumerate(current_settings["cryptos"])] + ["Batal"]
                    del_title = AnsiColors.MAGENTA + AnsiColors.BOLD + "Pilih crypto yang akan dihapus:" + AnsiColors.ENDC
                    try:
                        _, idx_del = pick(crypto_opts_del, del_title, indicator=f'{AnsiColors.RED}▶{AnsiColors.ENDC}')
                        if idx_del < len(current_settings["cryptos"]):
                            removed = current_settings["cryptos"].pop(idx_del); save_settings(current_settings)
                            log_info(f"Konfigurasi {removed['symbol']}-{removed['currency']} dihapus.")
                        else: log_info("Penghapusan dibatalkan.")
                    except (KeyboardInterrupt, Exception) as e_del_c_pick: 
                        if isinstance(e_del_c_pick, KeyboardInterrupt): log_info("Penghapusan dibatalkan.")
                        else: log_warning(f"Error pick hapus crypto: {e_del_c_pick}")
                        show_spinner(1, "Kembali...")
            elif idx == 6: break 
            if idx < 6 : show_spinner(1.5, "Menyimpan & Kembali...", color=AnsiColors.GREEN if idx not in [0,1,2] else AnsiColors.BLUE)
        except ValueError: log_error("Input angka tidak valid."); show_spinner(1.5, "Error, kembali...", color=AnsiColors.RED)
        except Exception as e_menu_action:
            log_error(f"Error di menu pengaturan: {e_menu_action}"); log_exception("Traceback Pengaturan:")
            show_spinner(1.5, "Error besar, kembali...", color=AnsiColors.RED)
    return current_settings

# --- FUNGSI PENGAMBILAN DATA (MODIFIED) ---
# (Fungsi fetch_candles, run_strategy_logic, start_trading tetap sama seperti versi sebelumnya yang sudah benar)
# ... [SALIN SEMUA FUNGSI fetch_candles, get_initial_strategy_state, find_pivots, run_strategy_logic, start_trading DARI JAWABAN SEBELUMNYA] ...
# PASTIKAN SEMUA FUNGSI INI DISALIN PERSIS SEPERTI YANG ADA DI JAWABAN SEBELUMNYA YANG SUDAH BENAR SECARA LOGIKA
# (Saya akan menyingkatnya di sini agar tidak terlalu panjang, tetapi di skrip Anda, mereka harus lengkap)

def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia.")
    all_candles, to_ts = [], None
    endpoint = {"minute": "histominute", "hour": "histohour", "day": "histoday"}.get(timeframe, "histohour")
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    is_large = total_limit_desired > 20 
    if is_large and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
        log_info(f"Pengambilan data besar: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
        simple_progress_bar(0, total_limit_desired, prefix=f'{AnsiColors.BOLD}{pair_name}{AnsiColors.ENDC} Data:', suffix='Candles', length=20)
    loops = 0
    while len(all_candles) < total_limit_desired:
        needed = total_limit_desired - len(all_candles)
        limit = min(needed + 1 if to_ts and needed > 1 else needed, CRYPTOCOMPARE_MAX_LIMIT)
        if limit <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if to_ts: params["toTs"] = to_ts
        key_short = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
        try:
            if is_large and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: log_debug(f"Batch (Key: ...{key_short}, Limit: {limit})", pair_name=pair_name)
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code in [401, 403, 429]: 
                err_data = resp.json() if resp.content else {}; err_msg = err_data.get('Message', f"HTTP {resp.status_code}")
                log_warning(f"API Key Error (HTTP {resp.status_code}): {err_msg} Key: ...{key_short}", pair_name=pair_name); raise APIKeyError(f"HTTP {resp.status_code}: {err_msg}")
            resp.raise_for_status(); data = resp.json()
            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error'); key_errs = ["api key is invalid", "apikey_is_missing", "apikey_invalid", "monthly_calls", "rate limit", "pro_tier"]
                if any(k_err.lower() in err_msg.lower() for k_err in key_errs):
                    log_warning(f"API Key Error (JSON): {err_msg} Key: ...{key_short}", pair_name=pair_name); raise APIKeyError(f"JSON Error: {err_msg}")
                else: log_error(f"API Error CryptoCompare: {err_msg} (Params: {params})", pair_name=pair_name); break 
            if not data.get('Data') or not data['Data'].get('Data'): log_info(f"Tidak ada data candle dari API atau format salah. Diambil: {len(all_candles)}.", pair_name=pair_name); break 
            raw_api_candles = data['Data']['Data']
            if not raw_api_candles: log_info(f"API mengembalikan list candle kosong. Diambil: {len(all_candles)}.", pair_name=pair_name); break
            batch = []
            for item in raw_api_candles:
                if not all(k in item for k in ['time', 'open', 'high', 'low', 'close']): log_warning(f"Data candle tidak lengkap dari API: {item}. Dilewati.", pair_name=pair_name); continue
                batch.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item['open'], 'high': item['high'],'low': item['low'], 'close': item['close'], 'volume': item.get('volumefrom')})
            if to_ts and all_candles and batch and batch[-1]['timestamp'] == all_candles[0]['timestamp']:
                if is_large: log_debug(f"Hapus overlap: {batch[-1]['timestamp']}", pair_name=pair_name); batch.pop() 
            if not batch and to_ts : 
                if is_large: log_info("Batch kosong setelah overlap removal. Akhir data.", pair_name=pair_name); break
            all_candles = batch + all_candles
            if raw_api_candles: to_ts = raw_api_candles[0]['time'] 
            else: break 
            loops +=1
            if is_large and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (loops % 2 == 0 or len(all_candles) >= total_limit_desired): 
                simple_progress_bar(len(all_candles), total_limit_desired, prefix=f'{AnsiColors.BOLD}{pair_name}{AnsiColors.ENDC} Data:', suffix='Candles', length=20)
            if len(raw_api_candles) < limit: 
                if is_large: log_info(f"API mengembalikan < limit ({len(raw_api_candles)} vs {limit}). Akhir histori.", pair_name=pair_name); break 
            if len(all_candles) >= total_limit_desired: break 
            if len(all_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large: time.sleep(0.25) 
        except APIKeyError: raise 
        except requests.exceptions.RequestException as e: log_error(f"Kesalahan koneksi saat ambil batch: {e}", pair_name=pair_name); break 
        except Exception as e: log_error(f"Error tak terduga fetch_candles: {e}", pair_name=pair_name); log_exception("Traceback Fetch Candles:", pair_name=pair_name); break 
    if len(all_candles) > total_limit_desired: all_candles = all_candles[-total_limit_desired:]
    if is_large and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_candles), total_limit_desired, prefix=f'{AnsiColors.BOLD}{pair_name}{AnsiColors.ENDC} Data:', suffix='Selesai', length=20)
    if is_large or total_limit_desired <=5 : log_info(f"Pengambilan data selesai. Total {AnsiColors.GREEN}{len(all_candles)}{AnsiColors.ENDC} (target: {total_limit_desired}).", pair_name=pair_name)
    return all_candles

def get_initial_strategy_state():
    return {"last_signal_type": 0,"final_pivot_high_price_confirmed": None,"final_pivot_low_price_confirmed": None,"high_price_for_fib": None,"high_bar_index_for_fib": None,"active_fib_level": None,"active_fib_line_start_index": None,"entry_price_custom": None,"highest_price_for_trailing": None,"trailing_tp_active_custom": False,"current_trailing_stop_level": None,"emergency_sl_level_custom": None,"position_size": 0}

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list); 
    if len(series_list) < left_strength + right_strength + 1: return pivots 
    for i in range(left_strength, len(series_list) - right_strength):
        val_i = series_list[i]; 
        if val_i is None: continue; is_pivot = True
        for j in range(1, left_strength + 1):
            val_ij = series_list[i-j]
            if val_ij is None : is_pivot = False; break 
            if (is_high and val_i <= val_ij) or (not is_high and val_i >= val_ij): is_pivot = False; break
        if not is_pivot: continue
        for j in range(1, right_strength + 1):
            val_ij = series_list[i+j]
            if val_ij is None: is_pivot = False; break
            if (is_high and val_i < val_ij) or (not is_high and val_i > val_ij): is_pivot = False; break
        if is_pivot: pivots[i] = val_i
    return pivots

def run_strategy_logic(candles, cfg, state): 
    pair = f"{cfg['symbol']}-{cfg['currency']}"; state["final_pivot_high_price_confirmed"] = None; state["final_pivot_low_price_confirmed"] = None
    req_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles or not (candles[0] and all(k in candles[0] for k in req_keys)): log_warning(f"Data candle kosong/tidak lengkap.", pair_name=pair); return state
    highs = [c.get('high') for c in candles]; lows = [c.get('low') for c in candles]
    piv_highs = find_pivots(highs, cfg['left_strength'], cfg['right_strength'], True); piv_lows = find_pivots(lows,  cfg['left_strength'], cfg['right_strength'], False)
    curr_idx = len(candles) - 1; 
    if curr_idx < 0: return state
    idx_piv_event_h = curr_idx - cfg['right_strength']; idx_piv_event_l = curr_idx - cfg['right_strength']
    raw_piv_h_price = piv_highs[idx_piv_event_h] if 0 <= idx_piv_event_h < len(piv_highs) else None
    raw_piv_l_price = piv_lows[idx_piv_event_l] if 0 <= idx_piv_event_l < len(piv_lows) else None
    if raw_piv_h_price is not None and state["last_signal_type"] != 1:
        state["final_pivot_high_price_confirmed"] = raw_piv_h_price; state["last_signal_type"] = 1 
        if 0 <= idx_piv_event_h < len(candles) and candles[idx_piv_event_h] and candles[idx_piv_event_h].get('timestamp'):
            ts = candles[idx_piv_event_h]['timestamp']; log_info(f"{AnsiColors.CYAN}📈 PIVOT HIGH: {state['final_pivot_high_price_confirmed']:.5f} @ {ts.strftime('%H:%M')}{AnsiColors.ENDC}", pair_name=pair)
        else: log_warning(f"Indeks pivot high ({idx_piv_event_h}) di luar jangkauan/data invalid.", pair_name=pair)
    if raw_piv_l_price is not None and state["last_signal_type"] != -1:
        state["final_pivot_low_price_confirmed"] = raw_piv_l_price; state["last_signal_type"] = -1 
        if 0 <= idx_piv_event_l < len(candles) and candles[idx_piv_event_l] and candles[idx_piv_event_l].get('timestamp'):
            ts = candles[idx_piv_event_l]['timestamp']; log_info(f"{AnsiColors.CYAN}📉 PIVOT LOW:  {state['final_pivot_low_price_confirmed']:.5f} @ {ts.strftime('%H:%M')}{AnsiColors.ENDC}", pair_name=pair)
        else: log_warning(f"Indeks pivot low ({idx_piv_event_l}) di luar jangkauan/data invalid.", pair_name=pair)
    curr_c = candles[curr_idx] 
    if not curr_c or any(curr_c.get(k) is None for k in ['open', 'high', 'low', 'close']):
        ts_str = curr_c.get('timestamp', datetime.now()).strftime('%H:%M'); log_warning(f"Data OHLC tidak lengkap candle terbaru @ {ts_str}. Skip.", pair_name=pair); return state
    if state["final_pivot_high_price_confirmed"] is not None:
        state["high_price_for_fib"] = state["final_pivot_high_price_confirmed"]; state["high_bar_index_for_fib"] = idx_piv_event_h
        if state["active_fib_level"] is not None: log_debug("Reset active FIB (new High).", pair_name=pair); state.update({"active_fib_level": None, "active_fib_line_start_index": None})
    if state["final_pivot_low_price_confirmed"] is not None and state["high_price_for_fib"] is not None and state["high_bar_index_for_fib"] is not None and idx_piv_event_l > state["high_bar_index_for_fib"]: 
        h_fib, l_fib = state["high_price_for_fib"], state["final_pivot_low_price_confirmed"]; calc_fib = (h_fib + l_fib) / 2.0; is_late = False
        if cfg["enable_secure_fib"]: price_check = curr_c.get(cfg["secure_fib_check_price"].lower(), curr_c['close']); 
            if price_check is not None and price_check > calc_fib: is_late = True
        if is_late: price_disp = f"{price_check:.5f}" if price_check is not None else "N/A"; log_info(f"{AnsiColors.ORANGE}⏳ FIB Terlambat ({calc_fib:.5f}), Cek({cfg['secure_fib_check_price']}:{price_disp}) > FIB.{AnsiColors.ENDC}", pair_name=pair); state.update({"active_fib_level": None, "active_fib_line_start_index": None})
        else: log_info(f"{AnsiColors.CYAN}✨ FIB 0.5 Aktif: {calc_fib:.5f}{AnsiColors.ENDC} (H:{h_fib:.2f},L:{l_fib:.2f})", pair_name=pair); state.update({"active_fib_level": calc_fib, "active_fib_line_start_index": idx_piv_event_l})
        state.update({"high_price_for_fib": None, "high_bar_index_for_fib": None}) 
    if state["active_fib_level"] is not None and state["position_size"] == 0 and curr_c['close'] > curr_c['open'] and curr_c['close'] > state["active_fib_level"]:
        entry_px = curr_c['close']; emerg_sl = entry_px * (1 - cfg["emergency_sl_percent"] / 100.0)
        state.update({"position_size": 1, "entry_price_custom": entry_px, "highest_price_for_trailing": entry_px, "trailing_tp_active_custom": False, "current_trailing_stop_level": None, "emergency_sl_level_custom": emerg_sl, "active_fib_level": None, "active_fib_line_start_index": None})
        log_msg = f"🚀 BUY ENTRY @ {entry_px:.5f} (FIB {state.get('active_fib_level', calc_fib):.5f} dilewati). SL: {emerg_sl:.5f}"; log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair); play_notification_sound(count=2, freq=1200, dur=150)
        ts_email = curr_c['timestamp'].strftime('%Y-%m-%d %H:%M:%S'); send_email_notification(f"BUY Signal: {pair}", f"New BUY: {pair} @ {entry_px:.5f}\nFIB Lvl: {state.get('active_fib_level', calc_fib):.5f}\nEmerg SL: {emerg_sl:.5f}\nTime: {ts_email}", {**cfg, 'pair_name': pair})
    if state["position_size"] > 0:
        highest_trailing = state.get("highest_price_for_trailing", curr_c.get('high', 0)); current_high = curr_c.get('high', highest_trailing) 
        state["highest_price_for_trailing"] = max(highest_trailing, current_high); entry_p = state.get("entry_price_custom")
        if not state["trailing_tp_active_custom"] and entry_p is not None and entry_p != 0:
            profit_pct = ((state["highest_price_for_trailing"] - entry_p) / entry_p) * 100.0
            if profit_pct >= cfg["profit_target_percent_activation"]: state["trailing_tp_active_custom"] = True; log_info(f"{AnsiColors.BLUE}🛡️ Trailing TP Aktif. Profit: {profit_pct:.2f}%, High: {state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}", pair_name=pair)
        if state["trailing_tp_active_custom"] and state["highest_price_for_trailing"] is not None:
            new_stop = state["highest_price_for_trailing"] * (1 - (cfg["trailing_stop_gap_percent"] / 100.0))
            if state.get("current_trailing_stop_level") is None or new_stop > state["current_trailing_stop_level"]: state["current_trailing_stop_level"] = new_stop; log_debug(f"Trailing SL update: {new_stop:.5f}", pair_name=pair)
        stop_exit = state.get("emergency_sl_level_custom"); exit_reason, exit_clr = "Emergency SL", AnsiColors.RED
        if state["trailing_tp_active_custom"] and state.get("current_trailing_stop_level") is not None:
            if stop_exit is None or state["current_trailing_stop_level"] > stop_exit: stop_exit = state["current_trailing_stop_level"]; exit_reason, exit_clr = "Trailing Stop", AnsiColors.BLUE
        curr_low = curr_c.get('low')
        if stop_exit is not None and curr_low is not None and curr_low <= stop_exit:
            exit_px = min(curr_c.get('open', stop_exit), stop_exit); pnl = 0.0
            if entry_p is not None and entry_p != 0: pnl = ((exit_px - entry_p) / entry_p) * 100.0
            if exit_reason == "Trailing Stop" and pnl < 0: exit_clr = AnsiColors.RED
            pnl_clr = AnsiColors.GREEN if pnl >=0 else AnsiColors.RED; log_msg = f"🛑 EXIT @ {exit_px:.5f} by {exit_reason}. PnL: {pnl_clr}{pnl:.2f}%{AnsiColors.ENDC}"; log_info(f"{exit_clr}{AnsiColors.BOLD}{log_msg}", pair_name=pair); play_notification_sound(freq=800, dur=300)
            ts_email_exit = curr_c['timestamp'].strftime('%Y-%m-%d %H:%M:%S'); send_email_notification(f"Trade Closed: {pair} ({exit_reason})", f"Trade Closed: {pair}\nExit: {exit_px:.5f} by {exit_reason}\nEntry: {entry_p if entry_p else 0:.5f}\nPnL: {pnl:.2f}%\nTime: {ts_email_exit}", {**cfg, 'pair_name': pair})
            state.update(get_initial_strategy_state()); state["last_signal_type"] = 0 
    if state["position_size"] > 0:
        sl_plot = state.get("emergency_sl_level_custom"); sl_type = "Emergency SL"
        if state.get("trailing_tp_active_custom") and state.get("current_trailing_stop_level") is not None:
            if sl_plot is None or state.get("current_trailing_stop_level") > sl_plot: sl_plot = state.get("current_trailing_stop_level"); sl_type = "Trailing SL"
        entry_disp = state.get('entry_price_custom', 0); sl_disp = f'{sl_plot:.5f} ({sl_type})' if sl_plot is not None else 'N/A'
        curr_close_pnl = curr_c.get('close', 0); curr_pnl_val = 0.0; entry_p_log = state.get("entry_price_custom")
        if entry_p_log and curr_close_pnl != 0 and entry_p_log !=0: curr_pnl_val = ((curr_close_pnl - entry_p_log) / entry_p_log) * 100.0
        pnl_color_log = AnsiColors.GREEN if curr_pnl_val >=0 else AnsiColors.ORANGE; log_debug(f"Posisi Aktif. Entry: {entry_disp:.5f}, SL: {sl_disp}, PnL: {pnl_color_log}{curr_pnl_val:.2f}%{AnsiColors.ENDC}", pair_name=pair)
    return state

def start_trading(global_settings):
    clear_screen_animated(); api_cfg = global_settings.get("api_settings", {})
    key_manager = APIKeyManager(api_cfg.get("primary_key"), api_cfg.get("recovery_keys", []), api_cfg)
    if not key_manager.has_valid_keys(): log_error(f"Tidak ada API key valid. Tidak dapat memulai."); animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE, center=True, bold=True); input(); return
    active_cfgs = [c for c in global_settings.get("cryptos", []) if c.get("enabled", True)]
    if not active_cfgs: log_warning(f"Tidak ada konfigurasi crypto aktif."); animated_text_display("Tekan Enter...", color=AnsiColors.ORANGE, center=True, bold=True); input(); return
    term_w = get_terminal_width(60); header_footer = AnsiColors.HEADER + AnsiColors.BOLD + "═" * term_w + AnsiColors.ENDC
    print(header_footer); animated_text_display("🚀 MULTI-CRYPTO STRATEGY START 🚀", color=AnsiColors.HEADER, bold=True, center=True, delay=0.002); print(header_footer)
    key_val = key_manager.get_current_key(); key_disp = "N/A"
    if key_val: key_disp = key_val[:5] + "..." + key_val[-3:] if len(key_val) > 8 else key_val
    log_info(f"API Key Idx: {AnsiColors.BOLD}{key_manager.get_current_key_index()}{AnsiColors.ENDC} ({key_disp}). Total: {AnsiColors.BOLD}{key_manager.total_keys()}{AnsiColors.ENDC}", pair_name="SYSTEM")
    data_mgr = {} 
    for cfg_item in active_cfgs:
        p_id = f"{cfg_item['symbol']}-{cfg_item['currency']}_{cfg_item['timeframe']}"; cfg_item['pair_name'] = f"{cfg_item['symbol']}-{cfg_item['currency']}"
        animated_text_display(f"\n✨ Init: {AnsiColors.BOLD}{cfg_item['pair_name']}{AnsiColors.ENDC} | {cfg_item['exchange']} | TF: {cfg_item['timeframe']}", color=AnsiColors.MAGENTA, delay=0.003)
        data_mgr[p_id] = {"config": cfg_item, "all_candles_list": [], "strategy_state": get_initial_strategy_state(), "big_data_collection_phase_active": True, "big_data_email_sent": False, "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0, "last_attempt_after_all_keys_failed": datetime.min}
        target_init = TARGET_BIG_DATA_CANDLES; initial_data, retries_init, fetched_ok = [], 0, False; max_retries_init_val = key_manager.total_keys() if key_manager.total_keys() > 0 else 1
        while retries_init < max_retries_init_val and not fetched_ok:
            curr_key = key_manager.get_current_key()
            if not curr_key: log_error(f"BIG DATA: Semua API key habis untuk {cfg_item['pair_name']}.", pair_name=cfg_item['pair_name']); break
            try:
                log_info(f"BIG DATA: Ambil {target_init} candle (Key Idx: {key_manager.get_current_key_index()})...", pair_name=cfg_item['pair_name'])
                initial_data = fetch_candles(cfg_item['symbol'], cfg_item['currency'], target_init, cfg_item['exchange'], curr_key, cfg_item['timeframe'], pair_name=cfg_item['pair_name']); fetched_ok = True
            except APIKeyError: log_warning(f"BIG DATA: API Key gagal {cfg_item['pair_name']}. Coba berikutnya.", pair_name=cfg_item['pair_name']); 
                if not key_manager.switch_to_next_key(): break; retries_init +=1 
            except requests.exceptions.RequestException as e_req: log_error(f"BIG DATA: Error jaringan {cfg_item['pair_name']}: {e_req}. Tidak ganti key.", pair_name=cfg_item['pair_name']); break 
        if not initial_data: log_error(f"BIG DATA: Gagal ambil data awal {cfg_item['pair_name']}. Pair mungkin tidak diproses.", pair_name=cfg_item['pair_name']); data_mgr[p_id].update({"big_data_collection_phase_active": False, "last_candle_fetch_time": datetime.now()}); continue 
        data_mgr[p_id]["all_candles_list"] = initial_data; log_info(f"BIG DATA: {AnsiColors.GREEN}{len(initial_data)}{AnsiColors.ENDC} candle awal diterima.", pair_name=cfg_item['pair_name'])
        if initial_data:
            min_len_piv = cfg_item['left_strength'] + cfg_item['right_strength'] + 1
            if len(initial_data) >= min_len_piv:
                log_info(f"Warm-up state dengan {max(0, len(initial_data) - 1)} candle historis...", pair_name=cfg_item['pair_name']); total_warmup = max(0, len(initial_data) - 1 - (min_len_piv -1))
                if total_warmup > 5: simple_progress_bar(0, total_warmup, prefix=f"{cfg_item['pair_name']} Warm-up:", suffix="State", length=15)
                for i in range(min_len_piv -1, len(initial_data) - 1): 
                    hist_slice = initial_data[:i+1]; 
                    if len(hist_slice) < min_len_piv: continue 
                    temp_state = data_mgr[p_id]["strategy_state"].copy(); temp_state["position_size"] = 0 
                    data_mgr[p_id]["strategy_state"] = run_strategy_logic(hist_slice, cfg_item, temp_state)
                    if data_mgr[p_id]["strategy_state"]["position_size"] > 0: data_mgr[p_id]["strategy_state"].update(get_initial_strategy_state()); data_mgr[p_id]["strategy_state"]["last_signal_type"] = 0 # Reset last signal juga
                    if total_warmup > 5: simple_progress_bar(i - (min_len_piv -1) + 1, total_warmup, prefix=f"{cfg_item['pair_name']} Warm-up:", suffix="State", length=15)
                log_info(f"Warm-up state selesai.", pair_name=cfg_item['pair_name'])
            else: log_warning(f"Data awal ({len(initial_data)}) tidak cukup warm-up pivot (min: {min_len_piv}).", pair_name=cfg_item['pair_name'])
        if len(data_mgr[p_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            data_mgr[p_id]["big_data_collection_phase_active"] = False; log_info(f"{AnsiColors.GREEN}🎯 TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI!{AnsiColors.ENDC}", pair_name=cfg_item['pair_name'])
            if not data_mgr[p_id]["big_data_email_sent"]: send_email_notification(f"Data Complete: {cfg_item['pair_name']}", f"Download {TARGET_BIG_DATA_CANDLES} candles selesai! Trading aktif.", {**cfg_item, 'pair_name': cfg_item['pair_name']}); data_mgr[p_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}--- ▶️ MULAI LIVE ANALYSIS ({len(data_mgr[p_id]['all_candles_list'])} candles) ---{AnsiColors.ENDC}", pair_name=cfg_item['pair_name'])
        print(AnsiColors.GREY + "-" * (term_w // 2) + AnsiColors.ENDC)
    try:
        while True:
            active_big_data_collect = 0; min_next_refresh = float('inf'); data_fetched_cycle = False 
            print("\n" + AnsiColors.BLUE + AnsiColors.BOLD + "─" * term_w + AnsiColors.ENDC)
            cycle_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); animated_text_display(f"🔄 New Cycle @ {cycle_time}", color=AnsiColors.BLUE, bold=True, center=True, delay=0.001, new_line=False); print(" " + AnsiColors.BLUE + AnsiColors.BOLD + "─" * term_w + AnsiColors.ENDC + "\n")
            for p_id, p_data in data_mgr.items():
                cfg = p_data["config"]; p_name = cfg['pair_name'] 
                if p_data.get("data_fetch_failed_consecutively", 0) >= (key_manager.total_keys() or 1) + 1 : 
                    time_since_fail = (datetime.now() - p_data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds()
                    if time_since_fail < 3600: min_next_refresh = min(min_next_refresh, 3600 - time_since_fail); continue 
                    else: p_data["data_fetch_failed_consecutively"] = 0; log_info(f"Cooldown 1 jam {p_name} selesai. Coba fetch.", pair_name=p_name)
                time_now = datetime.now(); secs_since_last_fetch = (time_now - p_data["last_candle_fetch_time"]).total_seconds(); interval_needed = 0
                if p_data["big_data_collection_phase_active"]: active_big_data_collect += 1; tf_intervals = {"minute": 55, "day": 3600 * 23.8, "hour": 3580}; interval_needed = tf_intervals.get(cfg['timeframe'], 3580)
                else: interval_needed = cfg['refresh_interval_seconds']
                if secs_since_last_fetch < interval_needed: min_next_refresh = min(min_next_refresh, interval_needed - secs_since_last_fetch); continue 
                log_info(f"🔎 Proses {AnsiColors.BOLD}{p_name}{AnsiColors.ENDC}...", pair_name=p_name); p_data["last_candle_fetch_time"] = time_now; candles_before = len(p_data["all_candles_list"])
                title_phase = f"--- ⏳ BIG DATA ({candles_before}/{TARGET_BIG_DATA_CANDLES}) ---" if p_data["big_data_collection_phase_active"] else f"--- 📊 ANALISA ({candles_before} candles) ---"
                animated_text_display(title_phase, color=AnsiColors.MAGENTA if p_data["big_data_collection_phase_active"] else AnsiColors.CYAN, bold=True, delay=0.001, center=True)
                new_batch, update_ok = [], False; max_retries_update = key_manager.total_keys() or 1; retries_update_done = 0
                while retries_update_done < max_retries_update and not update_ok:
                    key_for_attempt = key_manager.get_current_key()
                    if not key_for_attempt: log_error(f"Semua API key habis global untuk update {p_name}.", pair_name=p_name); break 
                    fetch_limit = 3 
                    if p_data["big_data_collection_phase_active"]: needed_big = TARGET_BIG_DATA_CANDLES - candles_before; fetch_limit = min(needed_big, CRYPTOCOMPARE_MAX_LIMIT); fetch_limit = max(fetch_limit, 1); 
                        if needed_big <=0 : update_ok = True; break
                    log_info(f"Ambil {fetch_limit} candle (Key Idx: {key_manager.get_current_key_index()})...", pair_name=p_name)
                    try: new_batch = fetch_candles(cfg['symbol'], cfg['currency'], fetch_limit, cfg['exchange'], key_for_attempt, cfg['timeframe'], pair_name=p_name); update_ok = True; p_data["data_fetch_failed_consecutively"] = 0; data_fetched_cycle = True 
                    except APIKeyError: log_warning(f"API Key (Idx: {key_manager.get_current_key_index()}) gagal update {p_name}. Coba berikutnya.", pair_name=p_name); p_data["data_fetch_failed_consecutively"] += 1; 
                        if not key_manager.switch_to_next_key(): log_error(f"Tidak ada API key global lagi setelah gagal {p_name}.", pair_name=p_name); break; retries_update_done += 1 
                    except requests.exceptions.RequestException as e_req_upd: log_error(f"Error jaringan update {p_name}: {e_req_upd}. Tidak ganti key.", pair_name=p_name); p_data["data_fetch_failed_consecutively"] += 1; break 
                if p_data.get("data_fetch_failed_consecutively", 0) >= (key_manager.total_keys() or 1) +1 : p_data["last_attempt_after_all_keys_failed"] = datetime.now(); log_warning(f"Semua API key gagal untuk {p_name}. Masuk cooldown.", pair_name=p_name)
                if not update_ok or not new_batch:
                    if update_ok and not new_batch and not p_data["big_data_collection_phase_active"]: log_warning(f"Tidak ada candle baru {p_name} (fetch ok).", pair_name=p_name)
                    elif not update_ok: log_error(f"Gagal update {p_name} setelah semua upaya.", pair_name=p_name)
                    min_next_refresh = min(min_next_refresh, interval_needed); print(AnsiColors.GREY + "-" * (term_w // 2) + AnsiColors.ENDC); continue 
                merged_dict = {c['timestamp']: c for c in p_data["all_candles_list"]}; added_count, updated_count = 0,0
                for c_new in new_batch:
                    ts_new = c_new['timestamp']
                    if ts_new not in merged_dict: merged_dict[ts_new] = c_new; added_count +=1
                    elif merged_dict[ts_new] != c_new : merged_dict[ts_new] = c_new; updated_count +=1
                p_data["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c_sort: c_sort['timestamp']); new_or_upd_total = added_count + updated_count
                if new_or_upd_total > 0: log_info(f"{AnsiColors.GREEN}{new_or_upd_total}{AnsiColors.ENDC} candle baru/update. Total: {len(p_data['all_candles_list'])}.", pair_name=p_name)
                elif new_batch: log_info("Tidak ada candle dg timestamp baru/update konten. Data identik.", pair_name=p_name)
                if p_data["big_data_collection_phase_active"] and len(p_data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                    log_info(f"{AnsiColors.GREEN}🎯 TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI {p_name}!{AnsiColors.ENDC}", pair_name=p_name)
                    if len(p_data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: p_data["all_candles_list"] = p_data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                    if not p_data["big_data_email_sent"]: send_email_notification(f"Data Complete: {p_name}", f"Download {TARGET_BIG_DATA_CANDLES} candles selesai! Trading aktif.", {**cfg, 'pair_name': p_name}); p_data["big_data_email_sent"] = True
                    p_data["big_data_collection_phase_active"] = False; active_big_data_collect -=1; log_info(f"{AnsiColors.HEADER}--- ▶️ MULAI LIVE ANALYSIS ({len(p_data['all_candles_list'])} candles) ---{AnsiColors.ENDC}", pair_name=p_name)
                elif not p_data["big_data_collection_phase_active"] and len(p_data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: p_data["all_candles_list"] = p_data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                min_len_piv_run = cfg['left_strength'] + cfg['right_strength'] + 1
                if len(p_data["all_candles_list"]) >= min_len_piv_run:
                    run_logic = (new_or_upd_total > 0 or (not p_data["big_data_collection_phase_active"] and candles_before < TARGET_BIG_DATA_CANDLES and len(p_data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or (p_data["big_data_collection_phase_active"]))
                    if run_logic: log_info(f"Jalankan strategi {len(p_data['all_candles_list'])} candle...", pair_name=p_name); p_data["strategy_state"] = run_strategy_logic(p_data["all_candles_list"], cfg, p_data["strategy_state"])
                    elif not p_data["big_data_collection_phase_active"]: 
                         last_c_ts = p_data["all_candles_list"][-1]['timestamp'].strftime('%H:%M:%S') if p_data["all_candles_list"] and p_data["all_candles_list"][-1] and p_data["all_candles_list"][-1].get('timestamp') else "N/A"
                         log_info(f"Tidak ada candle baru proses {p_name}. Data terakhir @ {last_c_ts}.", pair_name=p_name)
                else: log_info(f"Data ({len(p_data['all_candles_list'])}) {p_name} belum cukup (min: {min_len_piv_run}).", pair_name=p_name)
                min_next_refresh = min(min_next_refresh, interval_needed); print(AnsiColors.GREY + "-" * (term_w // 2) + AnsiColors.ENDC) 
            sleep_val = 15 
            if not data_fetched_cycle and key_manager.get_current_key() is None: log_error("Semua API key gagal global & tidak ada data di-fetch. Tunggu 1 jam.", pair_name="SYSTEM"); sleep_val = 3600 
            elif active_big_data_collect > 0:
                min_big_data_int = float('inf')
                for pid_sleep, pdata_sleep in data_mgr.items():
                    if pdata_sleep["big_data_collection_phase_active"]: cfg_sleep = pdata_sleep["config"]; tf_intervals_sleep = {"minute": 55, "day": 3600 * 23.8, "hour": 3580}; min_big_data_int = min(min_big_data_int, tf_intervals_sleep.get(cfg_sleep['timeframe'], 3580))
                sleep_val = min(min_big_data_int if min_big_data_int != float('inf') else 30, 30); log_debug(f"{active_big_data_collect} pair masih kumpulkan BIG DATA. Sleep {sleep_val}s.", pair_name="SYSTEM")
            else: 
                if min_next_refresh != float('inf') and min_next_refresh > 0: sleep_val = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_next_refresh)); log_debug(f"Semua pair live. Tidur ~{sleep_val}s.", pair_name="SYSTEM")
                else: sleep_val = MIN_REFRESH_INTERVAL_AFTER_BIG_DATA; log_debug(f"Default sleep {sleep_val}s (fallback).", pair_name="SYSTEM")
            if sleep_val > 0: show_spinner(sleep_val, f"💤 Menunggu {int(sleep_val)}s...", color=AnsiColors.BLUE)
    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}🛑 Proses dihentikan pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, bold=True, delay=0.01, center=True)
    except Exception as e_loop: log_error(f"Error tak terduga loop utama: {e_loop}", pair_name="SYSTEM"); log_exception("Traceback Error Loop Utama:", pair_name="SYSTEM")
    finally: 
        print("\n" + header_footer); animated_text_display("🏁 STRATEGY STOPPED 🏁", color=AnsiColors.HEADER, bold=True, center=True, delay=0.002); print(header_footer)
        animated_text_display("Tekan Enter untuk kembali...", color=AnsiColors.ORANGE, bold=True, delay=0.01, center=True); input()

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    while True:
        clear_screen_animated()
        term_w_main = get_terminal_width(60)
        title_main = [AnsiColors.HEADER + AnsiColors.BOLD + "╔" + "═" * (term_w_main - 2) + "╗" + AnsiColors.ENDC,
                      AnsiColors.HEADER + AnsiColors.BOLD + "║" + "🤖 Crypto Strategy Runner 🤖".center(term_w_main - 2) + "║" + AnsiColors.ENDC,
                      AnsiColors.HEADER + AnsiColors.BOLD + "╠" + "═" * (term_w_main - 2) + "╣" + AnsiColors.ENDC]
        active_cfgs_main = [c for c in settings.get("cryptos", []) if c.get("enabled", True)]
        if active_cfgs_main:
            title_main.append(AnsiColors.GREEN + AnsiColors.BOLD + f"  Crypto Aktif ({len(active_cfgs_main)}):" + AnsiColors.ENDC)
            for i, c_main in enumerate(active_cfgs_main[:3]): title_main.append(f"    {AnsiColors.CYAN}{i+1}. {c_main['symbol']}-{c_main['currency']} {AnsiColors.GREY}({c_main['timeframe']}, {c_main['exchange']}){AnsiColors.ENDC}")
            if len(active_cfgs_main) > 3: title_main.append(f"    {AnsiColors.GREY}...dan {len(active_cfgs_main)-3} lainnya.{AnsiColors.ENDC}")
        else: title_main.append(AnsiColors.ORANGE + "  Tidak ada konfigurasi crypto aktif." + AnsiColors.ENDC)
        title_main.append(AnsiColors.GREY + "  " + "-" * (term_w_main - 6) + AnsiColors.ENDC)
        api_s_main = settings.get("api_settings", {}); pk_val_main = api_s_main.get('primary_key', ''); pk_disp_main, pk_clr_main = ("BELUM DIATUR", AnsiColors.ORANGE)
        if pk_val_main and pk_val_main not in ["YOUR_PRIMARY_KEY", "YOUR_API_KEY_HERE"]: pk_disp_main = pk_val_main[:5] + "..." + pk_val_main[-3:] if len(pk_val_main) > 8 else pk_val_main; pk_clr_main = AnsiColors.GREEN
        num_rec_keys_main = len([k for k in api_s_main.get('recovery_keys', []) if isinstance(k, str) and k.strip()]); rk_clr_main = AnsiColors.GREEN if num_rec_keys_main > 0 else AnsiColors.ORANGE
        title_main.append(f"  {AnsiColors.BLUE}Target Data: {AnsiColors.WHITE}{TARGET_BIG_DATA_CANDLES} c/pair{AnsiColors.ENDC}")
        title_main.append(f"  {AnsiColors.BLUE}Primary Key: {pk_clr_main}{pk_disp_main}{AnsiColors.ENDC} | {AnsiColors.BLUE}Recovery: {rk_clr_main}{num_rec_keys_main}{AnsiColors.ENDC}")
        title_main.append(AnsiColors.GREY + "  " + "-" * (term_w_main - 6) + AnsiColors.ENDC)
        title_main.append(AnsiColors.MAGENTA + AnsiColors.BOLD + " Pilih Opsi:" + AnsiColors.ENDC)
        pick_title = "\n".join(title_main) + "\n"; options_main = ["🚀 Mulai Analisa Realtime", "⚙️  Pengaturan", "🚪 Keluar"]
        try:
            opt_txt_main, idx_main = pick(options_main, pick_title, indicator=f'{AnsiColors.GREEN}▶{AnsiColors.ENDC}')
            if idx_main == 0: start_trading(settings)
            elif idx_main == 1: settings = settings_menu(settings) 
            elif idx_main == 2: 
                log_info("Aplikasi ditutup.", pair_name="SYSTEM"); clear_screen_animated()
                animated_text_display("👋 Terima kasih! Sampai jumpa! 👋", color=AnsiColors.MAGENTA, bold=True, center=True, delay=0.01)
                show_spinner(0.7, "Exiting...", color=AnsiColors.BLUE); break
        except (KeyboardInterrupt, Exception) as e_main_menu: 
            log_warning(f"Operasi menu dibatalkan/error: {e_main_menu}", pair_name="SYSTEM")
            if isinstance(e_main_menu, KeyboardInterrupt): clear_screen_animated(); animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, bold=True, center=True); break
            show_spinner(1.5, "Error menu, coba lagi...", color=AnsiColors.RED)

if __name__ == "__main__":
    try:
        clear_screen_animated()
        term_w_banner = get_terminal_width(60) # <--- PERBAIKAN DI SINI
        banner_art = [
            "  ██████╗ ██████╗ ██╗   ██╗██████╗ ████████╗",
            "  ██╔══██╗██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝",
            "  ██████╔╝██████╔╝ ╚████╔╝ ██████╔╝   ██║   ",
            "  ██╔═══╝ ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ",
            "  ██║     ██║  ██║   ██║   ██║        ██║   ",
            "  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝   ",
            "  CRYPTOCURRENCY TRADING BOT (Enhanced UI)",
        ]
        print(AnsiColors.CYAN + AnsiColors.BOLD) 
        for line_art in banner_art[:-1]: 
             print(line_art.center(term_w_banner)) 
             time.sleep(0.03) 
        print(AnsiColors.ENDC)
        animated_text_display(banner_art[-1], color=AnsiColors.MAGENTA, bold=True, delay=0.005, center=True, width=term_w_banner)
        print("\n" * 1)
        show_spinner(0.8, "Memuat Pengaturan...", color=AnsiColors.GREEN)
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}{AnsiColors.BOLD}🚨 Aplikasi dihentikan paksa. Bye! 🚨{AnsiColors.ENDC}", color=AnsiColors.ORANGE, bold=True, delay=0.01, center=True)
    except Exception as e_fatal:
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}🔥🔥 Terjadi error fatal: {e_fatal} 🔥🔥{AnsiColors.ENDC}")
        import traceback
        traceback.print_exc()
        try:
            if logger and logger.handlers: 
                logger.critical("MAIN LEVEL EXCEPTION (FATAL):", exc_info=True)
        except NameError: pass 
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, bold=True, delay=0.01, center=True)
        input()
