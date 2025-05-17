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

# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m' # Warning / Late FIB
    RED = '\033[91m'    # Error / SL
    ENDC = '\033[0m'    # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[96m'
    MAGENTA = '\033[35m'
    YELLOW_BG = '\033[43m'
    LIGHT_GREEN = '\033[92m' # Alternatif untuk Green
    LIGHT_BLUE = '\033[94m'  # Alternatif untuk Blue

# --- ANIMATION & UI HELPER FUNCTIONS ---
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True, bold=False):
    """Menampilkan teks dengan efek ketik per karakter."""
    if not isinstance(text, str): # Safety check
        text = str(text)
    
    full_text = ""
    if bold:
        full_text += AnsiColors.BOLD
    if color:
        full_text += color
    
    for char in text:
        sys.stdout.write(full_text + char + AnsiColors.ENDC if color or bold else char)
        sys.stdout.flush()
        time.sleep(delay)
    
    # Reset setelah loop jika warna atau bold digunakan, tapi tidak di setiap char
    if color or bold:
        sys.stdout.write(AnsiColors.ENDC)
        sys.stdout.flush()

    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing..."):
    """Menampilkan animasi spinner sederhana untuk durasi tertentu."""
    spinner_chars = ['◐', '◓', '◑', '◒'] # Sedikit lebih menarik
    start_time = time.time()
    idx = 0
    terminal_width = os.get_terminal_size().columns if os.isatty() else 80
    
    sys.stdout.write(AnsiColors.MAGENTA)
    while (time.time() - start_time) < duration_seconds:
        max_msg_len = terminal_width - 5 # Ruang untuk spinner dan spasi
        display_message = message[:max_msg_len]
        
        sys.stdout.write(f"\r{AnsiColors.MAGENTA}{display_message} {spinner_chars[idx % len(spinner_chars)]} {AnsiColors.ENDC}")
        sys.stdout.flush()
        time.sleep(0.15) # Sedikit lebih lambat untuk animasi spinner
        idx += 1
    
    # Clear spinner line dengan spasi sebanyak lebar terminal
    sys.stdout.write(f"\r{' ' * terminal_width}\r") 
    sys.stdout.write(AnsiColors.ENDC)
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█', print_end="\r"):
    """Membuat dan menampilkan progress bar sederhana, disesuaikan untuk Termux."""
    # Sesuaikan panjang progress bar agar tidak terlalu lebar di Termux
    terminal_width = os.get_terminal_size().columns if os.isatty() else 80
    # Anggap prefix dan suffix pendek, sisakan ruang untuk persentase
    # Max length untuk bar itu sendiri, misal terminal_width - 20 (untuk prefix, suffix, persen)
    effective_bar_length = min(length, terminal_width - (len(prefix) + len(suffix) + 15))
    if effective_bar_length < 10: effective_bar_length = 10 # Minimum bar length

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(effective_bar_length * iteration // total)
    bar = fill * filled_length + '-' * (effective_bar_length - filled_length)
    
    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    
    # Potong seluruh baris jika masih melebihi lebar terminal
    sys.stdout.write(progress_line[:terminal_width])
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def clear_screen_animated():
    """Membersihkan layar terminal."""
    # Animasi singkat sebelum clear bisa ditambahkan di sini jika diinginkan
    # show_spinner(0.1, "Clearing screen...") 
    os.system('cls' if os.name == 'nt' else 'clear')

def display_welcome_banner():
    """Menampilkan banner selamat datang yang menarik."""
    clear_screen_animated()
    banner_lines = [
        "=================================================",
        " CRYPTO STRATEGY RUNNER ",
        " Multi-Pair with Key Recovery ",
        "=================================================",
        " Developed for Termux & CLI Users ",
        ""
    ]
    colors = [AnsiColors.HEADER, AnsiColors.BOLD + AnsiColors.GREEN, AnsiColors.BOLD + AnsiColors.CYAN, AnsiColors.HEADER, AnsiColors.MAGENTA]
    
    for i, line in enumerate(banner_lines):
        color_idx = i % len(colors)
        delay = 0.005 if "=" in line else 0.01
        # Pusatkan teks jika memungkinkan
        line_width = len(line)
        term_width = os.get_terminal_size().columns if os.isatty() else 80
        padding = (term_width - line_width) // 2
        display_line = " " * padding + line if padding > 0 else line
        animated_text_display(display_line, delay=delay, color=colors[color_idx], bold=False if "=" in line else True)
    time.sleep(0.5)

# --- INPUT HELPER FUNCTIONS (NEW) ---
def prompt_string(prompt_text, current_value="", default_value="", color=AnsiColors.BLUE, animated=True, delay=0.01, new_line_after_prompt=False):
    """Meminta input string dari pengguna dengan animasi dan nilai default."""
    display_current = current_value if current_value is not None else default_value
    full_prompt = f"{prompt_text} [{display_current}]: "
    
    if animated:
        animated_text_display(full_prompt, color=color, delay=delay, new_line=False)
        user_input = input() # Input akan muncul setelah teks animasi
        sys.stdout.write(AnsiColors.ENDC) # Pastikan warna reset
        if new_line_after_prompt or user_input: print() # Baris baru
    else:
        user_input = input(f"{color}{full_prompt}{AnsiColors.ENDC}")

    return user_input.strip() if user_input else (current_value if current_value is not None else default_value)

def prompt_int(prompt_text, current_value, default_value, color=AnsiColors.BLUE, animated=True, delay=0.01, min_val=None, max_val=None):
    """Meminta input integer dari pengguna dengan validasi."""
    prompt_suffix = f" [{current_value}]: "
    if min_val is not None or max_val is not None:
        range_info = []
        if min_val is not None: range_info.append(f"min {min_val}")
        if max_val is not None: range_info.append(f"max {max_val}")
        prompt_suffix = f" ({', '.join(range_info)})" + prompt_suffix
    
    full_prompt = prompt_text + prompt_suffix

    while True:
        if animated:
            animated_text_display(full_prompt, color=color, delay=delay, new_line=False)
            val_str = input()
            sys.stdout.write(AnsiColors.ENDC)
            print() 
        else:
            val_str = input(f"{color}{full_prompt}{AnsiColors.ENDC}")

        if not val_str.strip(): # Pengguna menekan Enter (gunakan nilai saat ini)
            return current_value
        try:
            value = int(val_str)
            if min_val is not None and value < min_val:
                animated_text_display(f"Nilai harus >= {min_val}. Coba lagi.", color=AnsiColors.RED, new_line=True)
                continue
            if max_val is not None and value > max_val:
                animated_text_display(f"Nilai harus <= {max_val}. Coba lagi.", color=AnsiColors.RED, new_line=True)
                continue
            return value
        except ValueError:
            animated_text_display(f"Input tidak valid. Harap masukkan angka bulat. (Nilai saat ini: {current_value})", color=AnsiColors.RED, new_line=True)

def prompt_float(prompt_text, current_value, default_value, color=AnsiColors.BLUE, animated=True, delay=0.01, min_val=None, max_val=None):
    """Meminta input float dari pengguna dengan validasi."""
    prompt_suffix = f" [{current_value}]: "
    if min_val is not None or max_val is not None:
        range_info = []
        if min_val is not None: range_info.append(f"min {min_val:.2f}")
        if max_val is not None: range_info.append(f"max {max_val:.2f}")
        prompt_suffix = f" ({', '.join(range_info)})" + prompt_suffix
    
    full_prompt = prompt_text + prompt_suffix

    while True:
        if animated:
            animated_text_display(full_prompt, color=color, delay=delay, new_line=False)
            val_str = input()
            sys.stdout.write(AnsiColors.ENDC)
            print()
        else:
            val_str = input(f"{color}{full_prompt}{AnsiColors.ENDC}")
        
        if not val_str.strip():
            return current_value
        try:
            value = float(val_str)
            if min_val is not None and value < min_val:
                animated_text_display(f"Nilai harus >= {min_val:.2f}. Coba lagi.", color=AnsiColors.RED, new_line=True)
                continue
            if max_val is not None and value > max_val:
                animated_text_display(f"Nilai harus <= {max_val:.2f}. Coba lagi.", color=AnsiColors.RED, new_line=True)
                continue
            return value
        except ValueError:
            animated_text_display(f"Input tidak valid. Harap masukkan angka desimal (misal 5.0). (Nilai saat ini: {current_value})", color=AnsiColors.RED, new_line=True)

def prompt_bool(prompt_text, current_value, default_value=False, color=AnsiColors.BLUE, animated=True, delay=0.01):
    """Meminta input boolean (true/false) dari pengguna."""
    current_display = 'true' if current_value else 'false'
    full_prompt = f"{prompt_text} (true/false) [{current_display}]: "
    while True:
        if animated:
            animated_text_display(full_prompt, color=color, delay=delay, new_line=False)
            val_str = input().lower()
            sys.stdout.write(AnsiColors.ENDC)
            print()
        else:
            val_str = input(f"{color}{full_prompt}{AnsiColors.ENDC}").lower()

        if not val_str.strip():
            return current_value
        if val_str in ['true', 't', 'y', 'yes', '1']:
            return True
        elif val_str in ['false', 'f', 'n', 'no', '0']:
            return False
        else:
            animated_text_display(f"Input tidak valid. Harap masukkan 'true' atau 'false'. (Nilai saat ini: {current_display})", color=AnsiColors.RED, new_line=True)

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Default ke INFO, bisa diubah jika perlu lebih detail
# logger.setLevel(logging.DEBUG) # Uncomment untuk debugging lebih detail

if logger.hasHandlers():
    logger.handlers.clear()

# File Handler (menyimpan log ke file)
fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# Stream Handler (menampilkan log ke konsol)
ch = logging.StreamHandler()
# Template format konsol yang lebih berwarna dan jelas
console_formatter_template = (
    f"{AnsiColors.LIGHT_BLUE}%(asctime)s{AnsiColors.ENDC} - "
    f"%(log_color)s{AnsiColors.BOLD}%(levelname)-8s{AnsiColors.ENDC} - " # Levelname rata kiri, bold
    f"{AnsiColors.CYAN}[%(pair_name)s]{AnsiColors.ENDC} - %(message)s"
)

class ConsoleFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: AnsiColors.MAGENTA,
        logging.INFO: AnsiColors.GREEN,
        logging.WARNING: AnsiColors.ORANGE,
        logging.ERROR: AnsiColors.RED,
        logging.CRITICAL: AnsiColors.BOLD + AnsiColors.RED,
    }
    def format(self, record):
        record.log_color = self.LOG_COLORS.get(record.levelno, AnsiColors.ENDC)
        # Untuk pesan multi-baris, format indentasi dengan benar
        msg = record.getMessage()
        if "\n" in msg:
            parts = msg.split('\n', 1)
            record.message = parts[0] # Baris pertama seperti biasa
            # Baris berikutnya di-indent
            if len(parts) > 1 and parts[1]:
                 indented_rest = "\n".join([f"    {line}" for line in parts[1].splitlines()])
                 record.message += f"\n{indented_rest}"
        
        # Selesaikan pemformatan dengan template string
        formatter = logging.Formatter(console_formatter_template, datefmt='%H:%M:%S')
        return formatter.format(record)

ch.setFormatter(ConsoleFormatter())
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
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name}) # Untuk debug
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})


SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999 
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15


# --- API KEY MANAGER --- (Tidak ada perubahan signifikan, hanya log disesuaikan)
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY":
            self.keys.append(primary_key)
        if recovery_keys_list:
            self.keys.extend([k for k in recovery_keys_list if k and isinstance(k, str) and k.strip()]) # Pastikan key valid

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}
        
        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys:
            log_error("Tidak ada API key yang tersedia di APIKeyManager.")
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None

    def switch_to_next_key(self):
        if not self.keys: return None

        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display})")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
                email_body = (f"Skrip trading telah secara otomatis mengganti API key CryptoCompare.\n\n"
                              f"API Key sebelumnya mungkin telah mencapai limit atau tidak valid.\n"
                              f"Sekarang menggunakan API key dengan index: {self.current_index}\n"
                              f"Key: ...{new_key_display[-8:]} (bagian akhir ditampilkan untuk identifikasi)\n\n"
                              f"Harap periksa status API key Anda di CryptoCompare.")
                dummy_email_cfg = {
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.values()): # Cek apakah semua nilai penting ada
                     send_email_notification(email_subject, email_body, dummy_email_cfg, pair_name_override="SYSTEM_KEY_SWITCH")
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi pergantian API key.", pair_name="SYSTEM_KEY_SWITCH")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "KRITIS: SEMUA API Key CryptoCompare Gagal!"
                email_body = (f"Skrip trading telah mencoba semua API key CryptoCompare yang tersedia (primary dan recovery) dan semuanya gagal.\n\n"
                              f"Skrip tidak dapat lagi mengambil data harga.\n"
                              f"Harap segera periksa akun CryptoCompare Anda dan konfigurasi API key di skrip.")
                dummy_email_cfg = { # Seperti di atas
                    "enable_email_notifications": True,
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.values()):
                    send_email_notification(email_subject, email_body, dummy_email_cfg, pair_name_override="SYSTEM_KEY_FAIL")
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal.", pair_name="SYSTEM_KEY_FAIL")
            return None

    def has_valid_keys(self):
        return bool(self.keys)
    
    def total_keys(self):
        return len(self.keys)
    
    def get_current_key_index(self):
        return self.current_index

# --- FUNGSI BEEP & EMAIL ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500) 
        else:
            print('\a', end='', flush=True) 
            time.sleep(0.1) # Jeda kecil jika \a perlu di-flush dengan benar
            print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email, pair_name_override=None):
    # Tentukan pair_name untuk logging
    if pair_name_override:
        pair_name_ctx = pair_name_override
    else:
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))


    if not settings_for_email.get("enable_email_notifications", False):
        log_debug(f"Notifikasi email dinonaktifkan untuk konteks ini.", pair_name=pair_name_ctx)
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

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
        log_info(f"Notifikasi email berhasil dikirim ke {receiver_email}", pair_name=pair_name_ctx)
    except Exception as e:
        log_error(f"Gagal mengirim email notifikasi: {e}", pair_name=pair_name_ctx)

# --- FUNGSI PENGATURAN ---
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
        "primary_key": "YOUR_PRIMARY_KEY",
        "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "", # Kosongkan default agar pengguna mengisi
        "email_sender_app_password": "",
        "email_receiver_address_admin": ""
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # Pastikan api_settings ada dan memiliki semua kunci default
                if "api_settings" not in settings or not isinstance(settings.get("api_settings"), dict):
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v
                
                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = []
                
                for crypto_cfg in settings["cryptos"]:
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True 
                    # Pastikan nilai numerik adalah tipe yang benar
                    for key, default_type_val in get_default_crypto_config().items():
                        if isinstance(default_type_val, (int, float)) and key in crypto_cfg:
                            try:
                                if isinstance(default_type_val, int):
                                    crypto_cfg[key] = int(crypto_cfg[key])
                                elif isinstance(default_type_val, float):
                                    crypto_cfg[key] = float(crypto_cfg[key])
                            except (ValueError, TypeError):
                                log_warning(f"Nilai korup '{crypto_cfg[key]}' untuk '{key}' di config crypto, reset ke default.")
                                crypto_cfg[key] = get_default_crypto_config()[key]

                return settings
        except json.JSONDecodeError:
            log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default dan membuat file baru jika memungkinkan.")
        except Exception as e:
            log_error(f"Error tidak terduga saat memuat pengaturan: {e}. Menggunakan default.")
    
    # Jika file tidak ada atau ada error, kembalikan struktur default
    # dan coba simpan agar file tercipta (jika error bukan karena permission)
    default_structure = {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    try:
        save_settings(default_structure) # Coba buat file settings default
        log_info(f"File {SETTINGS_FILE} tidak ditemukan atau korup. File default baru telah dibuat.")
    except Exception as e:
        log_error(f"Tidak dapat membuat file pengaturan default {SETTINGS_FILE}: {e}")
    return default_structure


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        log_info(f"Pengaturan disimpan ke {SETTINGS_FILE}")
    except Exception as e:
        log_error(f"Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}")
        # Mungkin lempar exception lagi jika ini kritis
        # raise


# --- REVISED _prompt_crypto_config using new helper functions ---
def _prompt_crypto_config(current_config):
    clear_screen_animated()
    new_config = current_config.copy() # Bekerja dengan salinan
    
    pair_display = f"{new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}"
    animated_text_display(f"--- Konfigurasi Crypto Pair ({pair_display}) ---", color=AnsiColors.HEADER, bold=True)
    
    new_config["enabled"] = prompt_bool("Aktifkan analisa untuk pair ini?", new_config.get('enabled',True), color=AnsiColors.GREEN)

    new_config["symbol"] = prompt_string("Simbol Crypto Dasar (misal BTC)", new_config.get('symbol','BTC'), "BTC", color=AnsiColors.LIGHT_BLUE).upper()
    new_config["currency"] = prompt_string("Simbol Mata Uang Quote (misal USDT)", new_config.get('currency','USD'), "USD", color=AnsiColors.LIGHT_BLUE).upper()
    new_config["exchange"] = prompt_string("Exchange (misal Binance, CCCAGG)", new_config.get('exchange','CCCAGG'), "CCCAGG", color=AnsiColors.LIGHT_BLUE)
    
    tf_options = ["minute", "hour", "day"]
    tf_current = new_config.get('timeframe','hour')
    tf_prompt = f"Timeframe ({'/'.join(tf_options)}) [{tf_current}]: "
    while True:
        animated_text_display(tf_prompt, color=AnsiColors.LIGHT_BLUE, delay=0.01, new_line=False)
        tf_input = input().lower()
        sys.stdout.write(AnsiColors.ENDC); print()
        if not tf_input: new_config["timeframe"] = tf_current; break
        if tf_input in tf_options: new_config["timeframe"] = tf_input; break
        else: animated_text_display(f"Timeframe tidak valid. Pilih dari: {', '.join(tf_options)}.", color=AnsiColors.RED)

    new_config["refresh_interval_seconds"] = prompt_int(
        f"Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle)", 
        new_config.get('refresh_interval_seconds',60), 60, 
        color=AnsiColors.LIGHT_BLUE, min_val=MIN_REFRESH_INTERVAL_AFTER_BIG_DATA
    )

    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, bold=True, delay=0.01)
    new_config["left_strength"] = prompt_int("Left Strength Pivot", new_config.get('left_strength',50), 50, color=AnsiColors.LIGHT_BLUE, min_val=1)
    new_config["right_strength"] = prompt_int("Right Strength Pivot", new_config.get('right_strength',150), 150, color=AnsiColors.LIGHT_BLUE, min_val=1)

    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, bold=True, delay=0.01)
    new_config["profit_target_percent_activation"] = prompt_float("Profit % Aktivasi Trailing TP", new_config.get('profit_target_percent_activation',5.0), 5.0, color=AnsiColors.LIGHT_BLUE, min_val=0.1)
    new_config["trailing_stop_gap_percent"] = prompt_float("Gap Trailing TP %", new_config.get('trailing_stop_gap_percent',5.0), 5.0, color=AnsiColors.LIGHT_BLUE, min_val=0.1)
    new_config["emergency_sl_percent"] = prompt_float("Emergency SL %", new_config.get('emergency_sl_percent',10.0), 10.0, color=AnsiColors.RED, min_val=0.1)
    
    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, bold=True, delay=0.01)
    new_config["enable_secure_fib"] = prompt_bool("Aktifkan Secure FIB?", new_config.get('enable_secure_fib',True), True, color=AnsiColors.GREEN)
    
    secure_fib_price_options = ["Close", "High"]
    sf_price_current = new_config.get('secure_fib_check_price','Close')
    sf_price_prompt = f"Harga Cek Secure FIB ({'/'.join(secure_fib_price_options)}) [{sf_price_current}]: "
    while True:
        animated_text_display(sf_price_prompt, color=AnsiColors.LIGHT_BLUE, delay=0.01, new_line=False)
        sf_price_input = input().capitalize()
        sys.stdout.write(AnsiColors.ENDC); print()
        if not sf_price_input: new_config["secure_fib_check_price"] = sf_price_current; break
        if sf_price_input in secure_fib_price_options: new_config["secure_fib_check_price"] = sf_price_input; break
        else: animated_text_display(f"Pilihan harga Secure FIB tidak valid. Pilih dari: {', '.join(secure_fib_price_options)}.", color=AnsiColors.RED)

    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, bold=True, delay=0.01)
    animated_text_display(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global (jika notif global aktif).{AnsiColors.ENDC}", delay=0.01)
    new_config["enable_email_notifications"] = prompt_bool("Aktifkan Notifikasi Email Pair Ini?", new_config.get('enable_email_notifications',False), False, color=AnsiColors.GREEN)
    new_config["email_sender_address"] = prompt_string("Email Pengirim (Gmail)", new_config.get('email_sender_address',''), "", color=AnsiColors.LIGHT_BLUE)
    new_config["email_sender_app_password"] = prompt_string("App Password Email Pengirim", new_config.get('email_sender_app_password',''), "", color=AnsiColors.LIGHT_BLUE) # Sebaiknya tidak ditampilkan nilai lama untuk password
    new_config["email_receiver_address"] = prompt_string("Email Penerima", new_config.get('email_receiver_address',''), "", color=AnsiColors.LIGHT_BLUE)
    
    return new_config


def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]: 
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        
        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len([k for k in recovery_keys if k and k.strip()]) # Hitung yang valid

        # Judul untuk 'pick'
        pick_title_settings = f"{AnsiColors.HEADER}--- Menu Pengaturan Utama ---{AnsiColors.ENDC}\n"
        pick_title_settings += f"{AnsiColors.CYAN}Primary API Key: {AnsiColors.BOLD}{primary_key_display}{AnsiColors.ENDC}\n"
        pick_title_settings += f"{AnsiColors.CYAN}Recovery API Keys: {AnsiColors.BOLD}{num_recovery_keys} tersimpan{AnsiColors.ENDC}\n"
        pick_title_settings += f"{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}\n"
        pick_title_settings += f"{AnsiColors.BOLD}Daftar Konfigurasi Crypto:{AnsiColors.ENDC}\n"
        
        # Konversi ke format yang lebih ramah 'pick' (plain text untuk title)
        plain_title_for_pick = "--- Menu Pengaturan Utama ---\n"
        plain_title_for_pick += f"Primary API Key: {primary_key_display}\n"
        plain_title_for_pick += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        plain_title_for_pick += "------------------------------------\n"
        plain_title_for_pick += "Daftar Konfigurasi Crypto:\n"

        if not current_settings["cryptos"]:
            pick_title_settings += f"  {AnsiColors.ORANGE}(Belum ada konfigurasi crypto){AnsiColors.ENDC}\n"
            plain_title_for_pick += "  (Belum ada konfigurasi crypto)\n"
        for i, crypto_conf in enumerate(current_settings["cryptos"]):
            status_color = AnsiColors.GREEN if crypto_conf.get('enabled', True) else AnsiColors.RED
            status_text = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
            display_text = f"  {i+1}. {AnsiColors.BOLD}{crypto_conf['symbol']}-{crypto_conf['currency']}{AnsiColors.ENDC} ({crypto_conf['timeframe']}) - {status_color}{status_text}{AnsiColors.ENDC}\n"
            pick_title_settings += display_text
            plain_title_for_pick += f"  {i+1}. {crypto_conf['symbol']}-{crypto_conf['currency']} ({crypto_conf['timeframe']}) - {status_text}\n"
        
        pick_title_settings += f"{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}\n"
        pick_title_settings += f"{AnsiColors.BOLD}Pilih tindakan:{AnsiColors.ENDC}"
        
        plain_title_for_pick += "------------------------------------\n"
        plain_title_for_pick += "Pilih tindakan:"


        # Opsi untuk pick harus berupa list string biasa
        options_for_pick = [
            "Atur Primary API Key",
            "Kelola Recovery API Keys",
            "Atur Email Global Notifikasi Sistem",
            "Tambah Konfigurasi Crypto Baru",
            "Ubah Konfigurasi Crypto",
            "Hapus Konfigurasi Crypto",
            "Kembali ke Menu Utama"
        ]
        
        # Menampilkan judul berwarna secara manual SEBELUM memanggil pick
        print(pick_title_settings) # Judul berwarna ditampilkan di sini
        
        # 'pick' dipanggil dengan judul plain text agar tidak ada masalah rendering ANSI
        option_text, index = pick(options_for_pick, plain_title_for_pick, indicator=f'{AnsiColors.GREEN}=> {AnsiColors.ENDC}', default_index=0)
        action_choice = index # index langsung dari pick sudah benar

        try:
            # clear_screen_animated() # Tidak perlu clear di sini, karena prompt akan dimulai setelahnya
            if action_choice == 0: 
                clear_screen_animated()
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER, bold=True)
                current_pk = api_s.get('primary_key','')
                api_s["primary_key"] = prompt_string("Masukkan Primary API Key CryptoCompare baru", current_pk, "YOUR_PRIMARY_KEY", color=AnsiColors.LIGHT_GREEN)
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 1: 
                manage_recovery_keys_menu(current_settings) # Buat fungsi terpisah untuk ini
            elif action_choice == 2: 
                clear_screen_animated()
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER, bold=True)
                api_s['enable_global_email_notifications_for_key_switch'] = prompt_bool(
                    "Aktifkan notifikasi email global (API Key switch, dll)?", 
                    api_s.get('enable_global_email_notifications_for_key_switch',False), False, color=AnsiColors.GREEN
                )
                api_s['email_sender_address'] = prompt_string("Email Pengirim Global (Gmail)", api_s.get('email_sender_address',''),"", color=AnsiColors.LIGHT_BLUE)
                api_s['email_sender_app_password'] = prompt_string("App Password Pengirim Global", "", "",color=AnsiColors.LIGHT_BLUE) # Selalu minta baru untuk password
                api_s['email_receiver_address_admin'] = prompt_string("Email Penerima Notifikasi Sistem (Admin)", api_s.get('email_receiver_address_admin',''),"", color=AnsiColors.LIGHT_BLUE)
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 3: 
                new_crypto_conf = get_default_crypto_config() # Mulai dengan default
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) 
                current_settings["cryptos"].append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Konfigurasi ditambahkan...")
            elif action_choice == 4: 
                if not current_settings["cryptos"]: 
                    animated_text_display("Tidak ada konfigurasi untuk diubah.", color=AnsiColors.ORANGE)
                    show_spinner(1.5, "Kembali..."); 
                    continue
                
                clear_screen_animated()
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER, bold=True)
                
                crypto_options = [f"{idx+1}. {c['symbol']}-{c['currency']} ({c['timeframe']}) {'(Aktif)' if c.get('enabled') else '(Nonaktif)'}" for idx, c in enumerate(current_settings["cryptos"])]
                crypto_options.append("Batal")
                
                # Judul untuk pick di sini
                change_pick_title = f"{AnsiColors.BOLD}Pilih konfigurasi yang akan diubah:{AnsiColors.ENDC}\n" + \
                                    "\n".join(crypto_options) + \
                                    f"\n{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}"
                
                # 'pick' dipanggil dengan judul plain text
                plain_change_title = "Pilih konfigurasi yang akan diubah:\n" + "\n".join(crypto_options)
                
                print(change_pick_title) # Tampilkan judul berwarna
                selected_option_text, idx_choice = pick(crypto_options, plain_change_title, indicator=f'{AnsiColors.GREEN}=> {AnsiColors.ENDC}')

                if selected_option_text == "Batal": continue

                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                    show_spinner(1, "Konfigurasi diubah...")
                else: 
                    animated_text_display("Pilihan tidak valid.", color=AnsiColors.RED)
                    show_spinner(1, "Kembali...")
            elif action_choice == 5: 
                if not current_settings["cryptos"]: 
                    animated_text_display("Tidak ada konfigurasi untuk dihapus.", color=AnsiColors.ORANGE)
                    show_spinner(1.5, "Kembali..."); 
                    continue
                
                clear_screen_animated()
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER, bold=True)
                
                crypto_options = [f"{idx+1}. {c['symbol']}-{c['currency']} ({c['timeframe']})" for idx, c in enumerate(current_settings["cryptos"])]
                crypto_options.append("Batal")

                delete_pick_title = f"{AnsiColors.BOLD}Pilih konfigurasi yang akan dihapus:{AnsiColors.ENDC}\n" + \
                                     "\n".join(crypto_options) + \
                                     f"\n{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}"
                plain_delete_title = "Pilih konfigurasi yang akan dihapus:\n" + "\n".join(crypto_options)

                print(delete_pick_title)
                selected_option_text, idx_choice = pick(crypto_options, plain_delete_title, indicator=f'{AnsiColors.GREEN}=> {AnsiColors.ENDC}')

                if selected_option_text == "Batal": continue
                
                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                    # Konfirmasi penghapusan
                    confirm_delete = prompt_bool(f"Yakin ingin menghapus {AnsiColors.BOLD}{removed_pair}{AnsiColors.ENDC}?", False, color=AnsiColors.RED)
                    if confirm_delete:
                        current_settings["cryptos"].pop(idx_choice)
                        save_settings(current_settings)
                        log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                        show_spinner(1, "Konfigurasi dihapus...")
                    else:
                        animated_text_display("Penghapusan dibatalkan.", color=AnsiColors.ORANGE)
                        show_spinner(1, "Dibatalkan...")
                else: 
                    animated_text_display("Pilihan tidak valid.", color=AnsiColors.RED)
                    show_spinner(1, "Kembali...")
            elif action_choice == 6: # Kembali ke Menu Utama
                break
        except ValueError as ve: # Khusus untuk error konversi tipe yang mungkin lolos
            log_error(f"Input tidak valid di menu pengaturan: {ve}")
            animated_text_display(f"Error: Input angka tidak sesuai. Silakan coba lagi.", color=AnsiColors.RED)
            show_spinner(2, "Error input, kembali...")
        except Exception as e:
            log_error(f"Terjadi kesalahan tak terduga di menu pengaturan: {e}")
            log_exception("Traceback error menu pengaturan:")
            animated_text_display(f"Terjadi error: {e}. Periksa log untuk detail.", color=AnsiColors.RED)
            show_spinner(2, "Error, kembali...")
    return current_settings


def manage_recovery_keys_menu(current_settings):
    api_s = current_settings.get("api_settings", {})
    
    while True:
        clear_screen_animated()
        current_recovery = [k for k in api_s.get('recovery_keys', []) if k and k.strip()] # Filter key kosong/spasi
        api_s['recovery_keys'] = current_recovery # Update list di api_s

        title_lines = [
            f"{AnsiColors.HEADER}--- Kelola Recovery API Keys ---{AnsiColors.ENDC}",
            f"{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}"
        ]
        plain_title_for_pick = "--- Kelola Recovery API Keys ---\n"

        if not current_recovery:
            title_lines.append(f"  {AnsiColors.ORANGE}(Tidak ada recovery key tersimpan){AnsiColors.ENDC}")
            plain_title_for_pick += "  (Tidak ada recovery key tersimpan)\n"
        else:
            for i, r_key in enumerate(current_recovery):
                r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                title_lines.append(f"  {i+1}. {AnsiColors.CYAN}{r_key_display}{AnsiColors.ENDC}")
                plain_title_for_pick += f"  {i+1}. {r_key_display}\n"
        
        title_lines.append(f"{AnsiColors.MAGENTA}------------------------------------{AnsiColors.ENDC}")
        title_lines.append(f"{AnsiColors.BOLD}Pilih tindakan:{AnsiColors.ENDC}")
        plain_title_for_pick += "------------------------------------\nPilih tindakan:"

        # Tampilkan judul berwarna
        for line in title_lines: print(line)

        recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan"]
        
        # pick() dengan judul plain
        rec_option_text, rec_index = pick(recovery_options_plain, plain_title_for_pick, indicator=f'{AnsiColors.GREEN}=> {AnsiColors.ENDC}', default_index=0)
        
        clear_screen_animated() # Clear setelah pilihan dibuat
        if rec_index == 0: # Tambah
            animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER, bold=True)
            new_r_key = prompt_string("Masukkan Recovery API Key baru", color=AnsiColors.LIGHT_GREEN).strip()
            if new_r_key:
                if new_r_key not in current_recovery:
                    current_recovery.append(new_r_key)
                    api_s['recovery_keys'] = current_recovery
                    save_settings(current_settings)
                    animated_text_display("Recovery key ditambahkan.", color=AnsiColors.GREEN)
                else:
                    animated_text_display("Recovery key tersebut sudah ada.", color=AnsiColors.ORANGE)
            else:
                animated_text_display("Input tidak boleh kosong.", color=AnsiColors.RED)
            show_spinner(1.5, "Memproses...")
        elif rec_index == 1: # Hapus
            animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER, bold=True)
            if not current_recovery:
                animated_text_display("Tidak ada recovery key untuk dihapus.", color=AnsiColors.ORANGE)
                show_spinner(1.5, "Kembali...")
                continue

            key_options_to_delete = [f"{idx+1}. { (key[:5] + '...' + key[-3:] if len(key) > 8 else key) }" for idx, key in enumerate(current_recovery)]
            key_options_to_delete.append("Batal")

            del_pick_title_lines = [f"{AnsiColors.BOLD}Pilih recovery key yang akan dihapus:{AnsiColors.ENDC}"] + key_options_to_delete
            plain_del_pick_title = "Pilih recovery key yang akan dihapus:\n" + "\n".join(key_options_to_delete)
            
            for line in del_pick_title_lines: print(line)
            
            selected_del_text, idx_del_choice = pick(key_options_to_delete, plain_del_pick_title, indicator=f'{AnsiColors.RED}=> {AnsiColors.ENDC}')

            if selected_del_text == "Batal": continue

            if 0 <= idx_del_choice < len(current_recovery):
                removed_key_display = current_recovery[idx_del_choice][:5] + "..."
                confirm = prompt_bool(f"Yakin ingin menghapus key '{removed_key_display}'?", False, color=AnsiColors.RED)
                if confirm:
                    current_recovery.pop(idx_del_choice)
                    api_s['recovery_keys'] = current_recovery
                    save_settings(current_settings)
                    animated_text_display(f"Recovery key '{removed_key_display}' dihapus.", color=AnsiColors.GREEN)
                else:
                    animated_text_display("Penghapusan dibatalkan.", color=AnsiColors.ORANGE)
            else:
                animated_text_display("Pilihan tidak valid.", color=AnsiColors.RED)
            show_spinner(1.5, "Memproses...")
        elif rec_index == 2: # Kembali
            break
    current_settings["api_settings"] = api_s # Pastikan perubahan di api_s disimpan kembali
    # save_settings(current_settings) # Sebaiknya save dilakukan di setiap aksi tambah/hapus


# --- FUNGSI PENGAMBILAN DATA (MODIFIED) ---
# (Tidak ada perubahan signifikan pada logika fetch, hanya logging disesuaikan dengan formatter baru)
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
            simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')


    fetch_loop_count = 0 
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        if current_to_ts is not None and candles_still_needed > 1 :
             limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)
        
        if limit_for_this_api_call <= 0: break 

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts
        
        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                 log_debug(f"Fetching batch (Key: ...{current_api_key_to_use[-5:]}, Limit: {limit_for_this_api_call})", pair_name=pair_name)
            
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code in [401, 403, 429]: 
                error_data = response.json() if response.content else {}
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                log_warning(f"API Key Error (HTTP {response.status_code}): {error_message}. Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() 
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active"
                ]
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"API Key Error (JSON): {error_message}. Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    log_error(f"API Error CryptoCompare: {error_message} (Params: {params})", pair_name=pair_name)
                    break 
            
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break 
            
            raw_candles_from_api = data['Data']['Data']
            
            if not raw_candles_from_api: 
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom') 
                }
                batch_candles_list.append(candle)

            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop() 
            
            if not batch_candles_list and current_to_ts is not None : 
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles 
            
            if raw_candles_from_api: 
                current_to_ts = raw_candles_from_api[0]['time'] 
            else: 
                break
            
            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): 
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles')

            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break 

            if len(all_accumulated_candles) >= total_limit_desired: break 

            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3) 

        except APIKeyError: 
            raise 
        except requests.exceptions.RequestException as e:
            log_error(f"Kesalahan koneksi/permintaan saat mengambil batch: {e}", pair_name=pair_name)
            break 
        except Exception as e:
            log_error(f"Error tak terduga dalam fetch_candles: {e}", pair_name=pair_name)
            log_exception("Traceback Error fetch_candles:", pair_name=pair_name) 
            break 

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Selesai')
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)
    
    return all_accumulated_candles

# --- LOGIKA STRATEGI --- (Tidak ada perubahan signifikan)
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None, 
        "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None, "emergency_sl_level_custom": None,
        "position_size": 0, 
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1: return pivots

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        current_val = series_list[i]
        if current_val is None: continue # Skip jika data tengah None

        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break
            if is_high:
                if current_val <= series_list[i-j]: is_pivot = False; break
            else: 
                if current_val >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue

        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break
            if is_high: 
                if current_val < series_list[i+j]: is_pivot = False; break 
            else: 
                if current_val > series_list[i+j]: is_pivot = False; break 
        
        if is_pivot:
            pivots[i] = series_list[i]
            
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"Data candle kosong atau kunci OHLC tidak lengkap.", pair_name=pair_name)
        return strategy_state

    high_prices = [c.get('high') for c in candles_history] # Gunakan .get() untuk keamanan
    low_prices = [c.get('low') for c in candles_history]

    # Cek apakah ada None di high_prices atau low_prices yang akan mengganggu find_pivots
    if any(p is None for p in high_prices) or any(p is None for p in low_prices):
        log_warning("Data harga (high/low) mengandung nilai None. Analisa pivot mungkin tidak akurat.", pair_name=pair_name)
        # Bisa return strategy_state di sini jika ingin skip jika ada None

    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    current_bar_index_in_list = len(candles_history) - 1 
    if current_bar_index_in_list < 0 : return strategy_state 

    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength
    
    raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high] if 0 <= idx_pivot_event_high < len(raw_pivot_highs) else None
    raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low] if 0 <= idx_pivot_event_low < len(raw_pivot_lows) else None

    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["last_signal_type"] = 1 
        pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
        log_info(f"PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}", pair_name=pair_name)

    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1 
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        log_info(f"PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}", pair_name=pair_name)
    
    current_candle = candles_history[current_bar_index_in_list] 
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=pair_name)
        return strategy_state

    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high 
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low 

            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0
                
                is_fib_late = False
                if crypto_config["enable_secure_fib"]:
                    price_key_to_check = crypto_config.get("secure_fib_check_price", "Close").lower()
                    price_val_current_candle = current_candle.get(price_key_to_check, current_candle.get('close'))
                    if price_val_current_candle is not None and price_val_current_candle > calculated_fib_level:
                        is_fib_late = True
                
                if is_fib_late:
                    log_info(f"FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config.get('secure_fib_check_price', 'Close')}: {price_val_current_candle:.5f}) > FIB.", pair_name=pair_name)
                    strategy_state["active_fib_level"] = None 
                    strategy_state["active_fib_line_start_index"] = None
                else:
                    log_info(f"FIB 0.5 Aktif: {calculated_fib_level:.5f} (H: {strategy_state['high_price_for_fib']:.5f}, L: {current_low_price:.5f})", pair_name=pair_name)
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = current_low_bar_index

                strategy_state["high_price_for_fib"] = None
                strategy_state["high_bar_index_for_fib"] = None
    
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: 
                strategy_state["position_size"] = 1 
                entry_px = current_candle['close'] 
                strategy_state["entry_price_custom"] = entry_px
                strategy_state["highest_price_for_trailing"] = entry_px 
                strategy_state["trailing_tp_active_custom"] = False 
                strategy_state["current_trailing_stop_level"] = None

                emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl
                
                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                log_info(log_msg, pair_name=pair_name) # Warna sudah di handle logger
                play_notification_sound()
                
                email_subject = f"BUY Signal: {pair_name}"
                email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\n"
                              f"Entry Price: {entry_px:.5f}\nFIB Level: {strategy_state['active_fib_level']:.5f}\n"
                              f"Emergency SL: {emerg_sl:.5f}\nTime: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})


            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    if strategy_state["position_size"] > 0: 
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle['high'])
        if current_high_for_trailing is None: current_high_for_trailing = current_candle['high'] 
        strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0 if strategy_state["entry_price_custom"] != 0 else 0
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state['highest_price_for_trailing']:.5f}", pair_name=pair_name)

        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"] is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
        
        if final_stop_for_exit is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price = min(current_candle['open'], final_stop_for_exit) 
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            log_level_pnl = logging.INFO if pnl >= 0 else logging.WARNING # Untuk warna log
            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            logger.log(log_level_pnl, log_msg, extra={'pair_name': pair_name}) # Gunakan logger.log untuk level dinamis
            play_notification_sound()

            email_subject = f"Trade Closed: {pair_name} ({exit_comment})"
            email_body = (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\nReason: {exit_comment}\n"
                          f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\nPnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': pair_name})


            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None; strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False; strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
    
    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        stop_type_info = "Emergency SL"
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state.get("current_trailing_stop_level")
                stop_type_info = "Trailing SL"
        
        entry_price_display = strategy_state.get('entry_price_custom', 0)
        sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)

    return strategy_state

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings 
    )

    if not api_key_manager.has_valid_keys():
        log_error("Tidak ada API key (primary/recovery) yang valid. Tidak dapat memulai.")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning("Tidak ada konfigurasi crypto yang aktif untuk dijalankan.")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return
    
    animated_text_display("===== MULTI-CRYPTO STRATEGY START (Key Recovery Enabled) =====", color=AnsiColors.HEADER, bold=True, delay=0.005)
    current_key_display = api_key_manager.get_current_key()
    if current_key_display and len(current_key_display) > 8: current_key_display = current_key_display[:5] + "..." + current_key_display[-3:]
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {} 
    for config in all_crypto_configs:
        pair_id = f"{config['symbol']}-{config['currency']}_{config['timeframe']}"
        # Pastikan 'pair_name' ada di config untuk logging yang konsisten
        config['pair_name'] = f"{config['symbol']}-{config['currency']}" 
        
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config['exchange']} | TF: {config['timeframe']}", color=AnsiColors.MAGENTA, delay=0.01, bold=False) # Bold sudah di handle AnsiColors

        crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }

        initial_candles_target = TARGET_BIG_DATA_CANDLES 
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key habis saat mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break 
            
            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target, 
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True
            except APIKeyError:
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break 
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e:
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                break 

        if not initial_candles:
            log_error(f"BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses.", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now()
            continue 
        
        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])

        if initial_candles:
            min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): 
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue 

                    temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 
                    
                    crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_for_warmup)
                    
                    if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        crypto_data_manager[pair_id]["strategy_state"] = {
                            **crypto_data_manager[pair_id]["strategy_state"], 
                            **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                               "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                               "current_trailing_stop_level":None}
                        }
                log_info(f"Inisialisasi state (warm-up) dengan data awal selesai.", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else:
            log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])
        
        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI setelah pengambilan awal!", pair_name=config['pair_name'])
            if not crypto_data_manager[pair_id]["big_data_email_sent"]:
                 send_email_notification(f"Data Downloading Complete: {config['pair_name']}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", {**config, 'pair_name': config['pair_name']})
                 crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) ----------", pair_name=config['pair_name'])

        animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", delay=0.005, new_line=True, bold=False)

    # --- Main Trading Loop ---
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0 
            min_overall_next_refresh_seconds = float('inf') 
            any_data_fetched_this_cycle = False 

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]
                pair_name = config['pair_name'] 
                
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: 
                        log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name)
                        continue 
                    else:
                        data["data_fetch_failed_consecutively"] = 0 
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()
                
                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    if config['timeframe'] == "minute": required_interval_for_this_pair = 55 
                    elif config['timeframe'] == "day": required_interval_for_this_pair = 3600 * 23.8 
                    else: required_interval_for_this_pair = 3580 
                else: 
                    required_interval_for_this_pair = config['refresh_interval_seconds']
                
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue 

                log_info(f"Memproses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time 
                num_candles_before_fetch = len(data["all_candles_list"])
                
                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) --- {pair_name}", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005, bold=False)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles --- {pair_name}", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005, bold=False)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                key_index_at_start_of_pair_fetch_attempt = api_key_manager.get_current_key_index()
                max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_for_this_pair_update = 0

                while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt:
                        log_error(f"Semua API key habis global saat mencoba update untuk {pair_name}.", pair_name=pair_name)
                        break 

                    limit_fetch = 3 
                    if data["big_data_collection_phase_active"]:
                        limit_fetch = min(TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"]), CRYPTOCOMPARE_MAX_LIMIT)
                        limit_fetch = max(limit_fetch, 3) 
                        if limit_fetch <=0 : 
                             fetch_update_successful_for_this_pair = True 
                             break

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch, 
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0 
                        any_data_fetched_this_cycle = True 
                    
                    except APIKeyError:
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        
                        if not api_key_manager.switch_to_next_key(): 
                            log_error(f"Tidak ada lagi API key tersedia global setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break 
                        retries_done_for_this_pair_update += 1 

                    except requests.exceptions.RequestException as e:
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak ganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break 
                
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data["last_attempt_after_all_keys_failed"] = datetime.now() 
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data["big_data_collection_phase_active"]:
                        log_warning(f"Tidak ada data candle baru diterima untuk {pair_name} (fetch berhasil).", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"Gagal mengambil update untuk {pair_name} setelah semua upaya siklus ini.", pair_name=pair_name)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    continue 

                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : 
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1
                
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                elif new_candles_batch : 
                     log_info("Tidak ada candle dengan timestamp baru atau update konten.", pair_name=pair_name)
                
                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                            data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data["big_data_email_sent"]:
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", {**config, 'pair_name': pair_name})
                            data["big_data_email_sent"] = True
                        
                        data["big_data_collection_phase_active"] = False 
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection - 1)
                        log_info(f"---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------", pair_name=pair_name)
                else: 
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                
                min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         (data["big_data_collection_phase_active"]))

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"])
                    elif not data["big_data_collection_phase_active"]: 
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: 
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)

            sleep_duration = 15 
            
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: 
                log_error("Semua API key gagal global & tidak ada data di-fetch. Menunggu 1 jam.", pair_name="SYSTEM")
                sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0:
                min_big_data_interval = float('inf')
                for pid, pdata in crypto_data_manager.items():
                    if pdata["big_data_collection_phase_active"]:
                        pconfig = pdata["config"]
                        interval = 55 if pconfig['timeframe'] == "minute" else (3600 * 23.8 if pconfig['timeframe'] == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval)
                
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) # Cap at 30s during big data
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: 
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s.", pair_name="SYSTEM")
                else: 
                    default_refresh = MIN_REFRESH_INTERVAL_AFTER_BIG_DATA
                    sleep_duration = default_refresh
                    log_debug(f"Default sleep {sleep_duration}s (fallback).", pair_name="SYSTEM")
            
            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu siklus berikutnya ({int(sleep_duration)}s)...")
            
    except KeyboardInterrupt: 
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", delay=0.01)
    except Exception as e:
        log_error(f"Error tak terduga di loop trading utama: {e}", pair_name="SYSTEM")
        log_exception("Traceback Error loop utama:", pair_name="SYSTEM")
    finally: 
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", delay=0.005, bold=True)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        # try catch input for specific environments
        try:
            input()
        except RuntimeError: # For environments where stdin might be problematic after pick
            log_warning("Tidak dapat membaca input untuk kembali ke menu. Silakan restart jika perlu.", pair_name="SYSTEM")
            time.sleep(3) # Beri waktu untuk membaca pesan


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()

    while True:
        display_welcome_banner() # Tampilkan banner setiap kali kembali ke menu utama
        
        # Judul berwarna ditampilkan sebelum pick
        title_lines_main = [
            f"{AnsiColors.HEADER}========= Crypto Strategy Runner (Multi + Key Recovery) =========" + AnsiColors.ENDC,
        ]
        plain_title_for_pick = "Crypto Strategy Runner\n" # Judul plain untuk pick
        
        active_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs:
            title_lines_main.append(f"{AnsiColors.GREEN}--- Crypto Aktif ({len(active_configs)}) ---" + AnsiColors.ENDC)
            plain_title_for_pick += f"--- Crypto Aktif ({len(active_configs)}) ---\n"
            for i, cfg in enumerate(active_configs): 
                line = f"  {i+1}. {AnsiColors.CYAN}{cfg['symbol']}-{cfg['currency']}{AnsiColors.ENDC} (TF: {cfg['timeframe']}, Exch: {cfg['exchange']})"
                title_lines_main.append(line)
                plain_title_for_pick += f"  {i+1}. {cfg['symbol']}-{cfg['currency']} (TF: {cfg['timeframe']}, Exch: {cfg['exchange']})\n"
        else: 
            title_lines_main.append(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif.{AnsiColors.ENDC}")
            plain_title_for_pick += "Tidak ada konfigurasi crypto yang aktif.\n"
        
        api_s = settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]: 
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len([k for k in api_s.get('recovery_keys',[]) if k and k.strip()])


        title_lines_main.extend([
            f"{AnsiColors.MAGENTA}-----------------------------------------------{AnsiColors.ENDC}",
            f"{AnsiColors.LIGHT_BLUE}Target Data per Pair: {AnsiColors.BOLD}{TARGET_BIG_DATA_CANDLES}{AnsiColors.ENDC} candle",
            f"{AnsiColors.LIGHT_BLUE}Primary API Key: {AnsiColors.BOLD}{primary_key_display}{AnsiColors.ENDC} | Recovery Keys: {AnsiColors.BOLD}{num_recovery_keys}{AnsiColors.ENDC}",
            f"{AnsiColors.MAGENTA}-----------------------------------------------{AnsiColors.ENDC}",
            f"{AnsiColors.BOLD}Pilih Opsi:{AnsiColors.ENDC}"
        ])
        plain_title_for_pick += "-----------------------------------------------\n"
        plain_title_for_pick += f"Target Data: {TARGET_BIG_DATA_CANDLES} | Pri.Key: {primary_key_display} | Rec.Keys: {num_recovery_keys}\n"
        plain_title_for_pick += "-----------------------------------------------\nPilih Opsi:"

        for line in title_lines_main: print(line) # Tampilkan judul berwarna

        options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]
        
        try:
            # pick dipanggil dengan judul plain
            option_text, index = pick(options_plain, plain_title_for_pick, indicator=f'{AnsiColors.GREEN}=> {AnsiColors.ENDC}', default_index=0)
            
            if index == 0: 
                start_trading(settings)
            elif index == 1: 
                settings = settings_menu(settings) # settings_menu akan mengembalikan settings yang mungkin diubah
                save_settings(settings) # Simpan perubahan setelah kembali dari settings_menu
            elif index == 2: 
                log_info("Aplikasi ditutup.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA, bold=True)
                show_spinner(0.8, "Exiting...")
                break
        except Exception as e: 
            log_warning(f"Operasi menu dibatalkan atau error: {e}", pair_name="SYSTEM")
            if isinstance(e, KeyboardInterrupt): 
                log_info("Aplikasi dihentikan oleh pengguna dari menu utama.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display("Aplikasi dihentikan. Bye!", color=AnsiColors.ORANGE, bold=True)
                break
            show_spinner(1.5, "Error menu, coba lagi...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display("Aplikasi dihentikan paksa. Bye!", color=AnsiColors.ORANGE, bold=True, delay=0.01)
    except Exception as e:
        clear_screen_animated()
        # Cetak error ke konsol sebelum logger mungkin sempat menanganinya jika error di awal sekali
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        logger.exception("MAIN LEVEL EXCEPTION:") # Logger akan mencatat traceback
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        try:
            input()
        except Exception: # Jaga-jaga jika input pun error
            pass
    finally:
        # Pastikan kursor terlihat lagi jika disembunyikan oleh 'pick' atau library lain
        sys.stdout.write("\033[?25h") 
        sys.stdout.flush()
