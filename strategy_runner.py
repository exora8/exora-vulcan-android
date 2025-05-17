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
from pick import pick # Pastikan library ini terinstal: pip install pick

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

# --- ANIMATION HELPER FUNCTIONS ---
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True):
    """Menampilkan teks dengan efek ketik per karakter."""
    for char in text:
        sys.stdout.write(color + char + AnsiColors.ENDC if color else char)
        sys.stdout.flush()
        time.sleep(delay)
    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing..."):
    """Menampilkan animasi spinner sederhana untuk durasi tertentu."""
    spinner_chars = ['-', '\\', '|', '/'] # PERBAIKAN: Escape backslash
    start_time = time.time()
    idx = 0
    sys.stdout.write(AnsiColors.MAGENTA) # Warna untuk spinner
    while (time.time() - start_time) < duration_seconds:
        # Pastikan pesan tidak melebihi lebar terminal yang wajar sebelum spinner
        terminal_columns = 80 # Default jika tidak di tty
        try:
            terminal_columns = os.get_terminal_size().columns if os.isatty() else 80
        except OSError: # Bisa terjadi di lingkungan tertentu seperti saat output di-pipe
            pass
        display_message = message[:terminal_columns - 5]
        
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") # Clear spinner line
    sys.stdout.write(AnsiColors.ENDC) # Reset warna
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """Membuat dan menampilkan progress bar sederhana."""
    if total == 0: # Hindari DivisionByZero
        percent_str = "0.0" if decimals > 0 else "0"
        percent = ("{0:." + str(decimals) + "f}").format(0.0)

    else:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    filled_length = int(length * iteration // total) if total > 0 else 0
    bar = fill * filled_length + '-' * (length - filled_length)

    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    
    terminal_columns = 80 # Default jika tidak di tty
    try:
        terminal_columns = os.get_terminal_size().columns if os.isatty() else 80
    except OSError:
        pass
    
    sys.stdout.write(progress_line[:terminal_columns]) # Potong jika terlalu panjang
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n') # Baris baru setelah selesai
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Default INFO, bisa diubah ke DEBUG jika perlu
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler (untuk menyimpan log ke file)
try:
    fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
except Exception as e:
    print(f"{AnsiColors.RED}Tidak dapat menginisialisasi file logger: {e}{AnsiColors.ENDC}")

# Stream Handler (untuk output ke konsol)
ch = logging.StreamHandler(sys.stdout) # Eksplisit ke sys.stdout
console_formatter_template = '%(asctime)s - {bold}%(levelname)s{endc} - {cyan}[%(pair_name)s]{endc} - %(message)s'
ch.setFormatter(logging.Formatter(
    console_formatter_template.format(bold=AnsiColors.BOLD, endc=AnsiColors.ENDC, cyan=AnsiColors.CYAN)
))
logger.addHandler(ch)

class AddPairNameFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'pair_name'):
            record.pair_name = 'SYSTEM' # Default pair_name jika tidak diset
        return True
logger.addFilter(AddPairNameFilter())

# Fungsi helper logging
def log_info(message, pair_name="SYSTEM"): logger.info(message, extra={'pair_name': pair_name})
def log_warning(message, pair_name="SYSTEM"): logger.warning(message, extra={'pair_name': pair_name})
def log_error(message, pair_name="SYSTEM"): logger.error(message, extra={'pair_name': pair_name})
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name})
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})


SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999 
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 # Detik

# --- FUNGSI CLEAR SCREEN (DIMODIFIKASI dengan animasi kecil) ---
def clear_screen_animated():
    """Membersihkan layar terminal dengan sedikit animasi."""
    # Optional: Tampilkan pesan singkat sebelum clear
    # animated_text_display("Clearing...", delay=0.01, color=AnsiColors.MAGENTA, new_line=False) # Komentar jika terlalu banyak animasi
    # show_spinner(0.2, "Clearing screen") # Komentar jika terlalu banyak animasi
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None): # PERBAIKAN: init -> __init__
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
            # log_error("Tidak ada API key yang tersedia di APIKeyManager.") # Bisa terlalu verbose jika dipanggil sering
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None

    def switch_to_next_key(self):
        if not self.keys or self.current_index >= len(self.keys) -1 : # Jika tidak ada keys atau sudah di key terakhir
            if self.keys and self.current_index >= len(self.keys) -1 and len(self.keys) > 0: # Jika sudah mencoba semua keys
                 log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
                 # Handle notifikasi email di sini untuk semua key gagal sekali saja
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
                    if all(dummy_email_cfg.values()): # Cek apakah semua field email ada (string kosong jadi False)
                        send_email_notification(email_subject, email_body, dummy_email_cfg)
                    else:
                        log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal.")
            self.current_index = len(self.keys) # Set index di luar jangkauan agar get_current_key return None
            return None


        self.current_index += 1
        # Kondisi di atas sudah menangani jika current_index akan melebihi. Jadi di sini pasti valid.
        new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
        log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
        
        if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
            email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
            email_body = (f"Skrip trading telah secara otomatis mengganti API key CryptoCompare.\n\n"
                          f"API Key sebelumnya mungkin telah mencapai limit atau tidak valid.\n"
                          f"Sekarang menggunakan API key dengan index: {self.current_index}\n"
                          f"Key: ...{new_key_display[-8:]} (bagian akhir ditampilkan untuk identifikasi)\n\n" # Seharusnya new_key_display saja cukup
                          f"Harap periksa status API key Anda di CryptoCompare.")
            dummy_email_cfg = {
                "enable_email_notifications": True,
                "email_sender_address": self.global_email_settings.get("email_sender_address"),
                "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
            }
            if all(dummy_email_cfg.values()): # Cek apakah semua field email ada
                 send_email_notification(email_subject, email_body, dummy_email_cfg)
            else:
                log_warning("Konfigurasi email global tidak lengkap untuk notifikasi pergantian API key.")
        return self.keys[self.current_index]
        # Logika SEMUA API KEY HABIS dipindahkan ke awal fungsi agar hanya log sekali.

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
            import winsound # Import hanya saat dibutuhkan
            winsound.Beep(1000, 500) 
        else:
            # Bell character, standar.
            sys.stdout.write('\a')
            sys.stdout.flush()
            # time.sleep(0.2) # Jeda kecil jika perlu, biasanya tidak perlu untuk \a
            # sys.stdout.write('\a') # Satu kali \a biasanya cukup
            # sys.stdout.flush()
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    # Pastikan semua string config email tidak kosong
    if not all([sender_email, sender_password, receiver_email, 
                isinstance(sender_email, str) and sender_email.strip(),
                isinstance(sender_password, str) and sender_password.strip(),
                isinstance(receiver_email, str) and receiver_email.strip()
                ]):
        pair_name_ctx = settings_for_email.get('symbol', 'GLOBAL_EMAIL') 
        log_warning(f"Konfigurasi email tidak lengkap atau invalid. Notifikasi email dilewati.", pair_name=pair_name_ctx)
        return

    msg = MIMEText(body_text)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as smtp_server: # Tambah timeout
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        pair_name_ctx = settings_for_email.get('symbol', 'GLOBAL_EMAIL')
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e:
        pair_name_ctx = settings_for_email.get('symbol', 'GLOBAL_EMAIL')
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)

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
        "primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "", "email_sender_app_password": "",
        "email_receiver_address_admin": ""
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: # Tambah encoding
                settings = json.load(f)

            # Pastikan api_settings ada dan memiliki semua kunci default
            if "api_settings" not in settings or not isinstance(settings["api_settings"], dict) :
                settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]:
                        settings["api_settings"][k] = v
            
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                settings["cryptos"] = []
            
            # Pastikan setiap crypto config memiliki ID dan status enabled
            for crypto_cfg in settings["cryptos"]:
                if not isinstance(crypto_cfg, dict): continue # Lewati jika item bukan dict
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True 
            
            # Filter keluar item yang bukan dictionary dari list cryptos
            settings["cryptos"] = [cfg for cfg in settings["cryptos"] if isinstance(cfg, dict)]
            return settings
        except json.JSONDecodeError:
            log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan struktur default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
        except Exception as e:
            log_error(f"Error tak terduga saat load_settings: {e}. Menggunakan struktur default.")
            return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    else: # File tidak ada, buat default
        log_info(f"{SETTINGS_FILE} tidak ditemukan. Membuat file dengan pengaturan default.")
        default_settings_to_save = {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
        save_settings(default_settings_to_save) 
        return default_settings_to_save


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: # Tambah encoding
            json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal menyimpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")


def _prompt_crypto_config(current_config):
    # clear_screen_animated() # Mungkin terlalu sering clear
    new_config = current_config.copy()
    title_pair = f"{new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}"
    animated_text_display(f"--- Konfigurasi Crypto Pair ({title_pair}) ---", color=AnsiColors.HEADER)

    try: # Tambahkan try-except untuk input numerik
        enabled_input = input(f"Aktifkan pair ini? (true/false) [{new_config.get('enabled',True)}]: ").strip().lower()
        new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

        new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}").strip() or new_config.get('symbol','BTC')).upper()
        new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}").strip() or new_config.get('currency','USD')).upper()
        new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}").strip() or new_config.get('exchange','CCCAGG'))

        tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}").strip() or new_config.get('timeframe','hour')).lower()
        if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
        else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}"); # Biarkan default

        refresh_input = input(f"{AnsiColors.BLUE}Interval Refresh (detik, min {MIN_REFRESH_INTERVAL_AFTER_BIG_DATA}) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input or new_config.get('refresh_interval_seconds',60)))

        animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))

        animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))

        animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
        enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").strip().lower()
        new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))
        
        secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
        if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
        else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid. Menggunakan default: {new_config.get('secure_fib_check_price','Close')}{AnsiColors.ENDC}");

        animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
        print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global dari API Settings (jika notif global aktif).{AnsiColors.ENDC}")
        email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").strip().lower()
        new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
        new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}").strip() or new_config.get('email_sender_address',''))
        new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}").strip() or new_config.get('email_sender_app_password','')) # Password tidak ditampilkan defaultnya
        new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}").strip() or new_config.get('email_receiver_address',''))
    
    except ValueError:
        print(f"{AnsiColors.RED}Input numerik tidak valid! Perubahan mungkin tidak tersimpan dengan benar untuk bagian numerik.{AnsiColors.ENDC}")
        # Kembalikan config asli jika ada error input fatal
        return current_config 
    except Exception as e:
        print(f"{AnsiColors.RED}Terjadi error saat input konfigurasi: {e}{AnsiColors.ENDC}")
        return current_config

    return new_config

def settings_menu(current_settings):
    while True:
        clear_screen_animated() 
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if isinstance(primary_key_display, str) and len(primary_key_display) > 10: 
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]

        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len([k for k in recovery_keys if k and isinstance(k,str)]) # Hitung key yang valid saja

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"

        if not current_settings.get("cryptos"): # Cek .get()
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                if not isinstance(crypto_conf, dict): continue # Lewati jika bukan dict
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                symbol = crypto_conf.get('symbol', 'N/A')
                currency = crypto_conf.get('currency', 'N/A')
                timeframe = crypto_conf.get('timeframe', 'N/A')
                pick_title_settings += f"  {i+1}. {symbol}-{currency} ({timeframe}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

        # Opsi untuk 'pick' library
        options_for_pick = [
            "Atur Primary API Key",
            "Kelola Recovery API Keys",
            "Atur Email Global untuk Notifikasi Sistem",
            "Tambah Konfigurasi Crypto Baru",
            "Ubah Konfigurasi Crypto",
            "Hapus Konfigurasi Crypto",
            "Kembali ke Menu Utama"
        ]
        
        # Struktur header/option tidak digunakan langsung oleh pick, tapi berguna untuk pemetaan manual jika diperlukan
        # original_options_structure = [
        #     ("header", "--- Pengaturan API & Global ---"),
        #     ("option", "Atur Primary API Key"),
        #     ("option", "Kelola Recovery API Keys"),
        #     ("option", "Atur Email Global untuk Notifikasi Sistem (API Key Switch, dll)"),
        #     ("header", "--- Pengaturan Crypto Pair ---"),
        #     ("option", "Tambah Konfigurasi Crypto Baru"),
        #     ("option", "Ubah Konfigurasi Crypto"),
        #     ("option", "Hapus Konfigurasi Crypto"),
        #     ("header", "-----------------------------"),
        #     ("option", "Kembali ke Menu Utama")
        # ]
        # selectable_options = [text for type, text in original_options_structure if type == "option"]
        # raw_option_texts = selectable_options[:] # Tidak perlu lagi jika options_for_pick langsung dipakai

        try:
            option_text, index = pick(options_for_pick, pick_title_settings, indicator='=>', default_index=0)
            # action_choice = raw_option_texts.index(option_text) # Langsung gunakan index dari pick
            action_choice = index 
        except Exception as e: # Termasuk KeyboardInterrupt dari pick
            log_warning(f"Pemilihan menu dibatalkan atau error: {e}")
            if isinstance(e, KeyboardInterrupt): break # Keluar dari settings_menu
            show_spinner(1, "Error pick, kembali...")
            continue


        # clear_screen_animated() # Pindah clear ke dalam masing-masing case jika perlu, atau di awal loop
        try:
            if action_choice == 0: # Atur Primary API Key
                clear_screen_animated()
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                current_pk = api_s.get('primary_key','')
                new_pk = input(f"Masukkan Primary API Key CryptoCompare baru [{current_pk[:5]+'...' if current_pk else ''}]: ").strip()
                api_s["primary_key"] = new_pk or current_pk # Jika input kosong, pertahankan yang lama
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 1: # Kelola Recovery API Keys
                while True: # Loop untuk menu recovery key
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k and isinstance(k, str)] # Filter key valid
                    
                    if not current_recovery:
                        recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"
                    
                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan"]
                    
                    try:
                        rec_option_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e: # Termasuk KeyboardInterrupt
                         log_warning(f"Pemilihan menu recovery dibatalkan: {e}")
                         if isinstance(e, KeyboardInterrupt): break # Keluar dari menu recovery
                         show_spinner(1, "Error pick, kembali...")
                         continue # Kembali ke awal loop menu recovery
                    
                    # clear_screen_animated() # Clear setelah pick
                    if rec_index == 0: # Tambah Recovery Key
                        clear_screen_animated()
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else:
                            print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1.5, "Kembali...")
                    
                    elif rec_index == 1: # Hapus Recovery Key
                        clear_screen_animated()
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
                        else:
                            # Tampilkan dulu list untuk dipilih
                            del_options = [f"{i+1}. { (key[:5]+'...' if len(key)>8 else key) }" for i, key in enumerate(current_recovery)]
                            del_options.append("Batal")
                            try:
                                _, del_idx = pick(del_options, "Pilih key untuk dihapus:", indicator="=>")
                                if del_idx < len(current_recovery): # Bukan opsi "Batal"
                                    removed = current_recovery.pop(del_idx)
                                    api_s['recovery_keys'] = current_recovery
                                    save_settings(current_settings)
                                    print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                                else: # Opsi "Batal"
                                     print(f"{AnsiColors.ORANGE}Penghapusan dibatalkan.{AnsiColors.ENDC}")
                            except Exception as e:
                                print(f"{AnsiColors.RED}Error saat memilih key untuk dihapus: {e}{AnsiColors.ENDC}")
                        show_spinner(1.5, "Kembali...")
                    
                    elif rec_index == 2: # Kembali ke Pengaturan Utama
                        break # Keluar dari loop menu recovery key
            
            elif action_choice == 2: # Atur Email Global
                clear_screen_animated()
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").strip().lower()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address','')
                api_s['email_sender_app_password'] = input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password','')
                api_s['email_receiver_address_admin'] = input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin','')
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 3: # Tambah Konfigurasi Crypto Baru
                clear_screen_animated()
                new_crypto_conf = get_default_crypto_config() # Dapat ID baru
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) 
                if "cryptos" not in current_settings or not isinstance(current_settings["cryptos"], list):
                    current_settings["cryptos"] = [] # Inisialisasi jika belum ada/salah format
                current_settings["cryptos"].append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")

            elif action_choice == 4: # Ubah Konfigurasi Crypto
                clear_screen_animated()
                if not current_settings.get("cryptos"): 
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                else:
                    animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                    edit_options = [f"{i+1}. {cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for i, cfg in enumerate(current_settings["cryptos"])]
                    edit_options.append("Batal")
                    try:
                        _, idx_choice = pick(edit_options, "Pilih konfigurasi untuk diubah:", indicator="=>")
                        if idx_choice < len(current_settings["cryptos"]): # Bukan "Batal"
                            current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                            save_settings(current_settings)
                            log_info(f"Konfigurasi untuk ID {current_settings['cryptos'][idx_choice]['id']} diubah.")
                        else:
                             print(f"{AnsiColors.ORANGE}Perubahan dibatalkan.{AnsiColors.ENDC}")
                    except Exception as e:
                         print(f"{AnsiColors.RED}Error saat memilih konfigurasi: {e}{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali..."); 
            
            elif action_choice == 5: # Hapus Konfigurasi Crypto
                clear_screen_animated()
                if not current_settings.get("cryptos"): 
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                else:
                    animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                    del_options_crypto = [f"{i+1}. {cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for i, cfg in enumerate(current_settings["cryptos"])]
                    del_options_crypto.append("Batal")
                    try:
                        _, idx_choice = pick(del_options_crypto, "Pilih konfigurasi untuk dihapus:", indicator="=>")
                        if idx_choice < len(current_settings["cryptos"]): # Bukan "Batal"
                            removed_pair = f"{current_settings['cryptos'][idx_choice].get('symbol','N/A')}-{current_settings['cryptos'][idx_choice].get('currency','N/A')}"
                            current_settings["cryptos"].pop(idx_choice)
                            save_settings(current_settings)
                            log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                        else:
                             print(f"{AnsiColors.ORANGE}Penghapusan dibatalkan.{AnsiColors.ENDC}")
                    except Exception as e:
                        print(f"{AnsiColors.RED}Error saat memilih konfigurasi: {e}{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali..."); 

            elif action_choice == 6: # Kembali ke Menu Utama
                break # Keluar dari loop settings_menu
        
        except ValueError: # Ini untuk error konversi int/float yang mungkin tidak tertangkap di _prompt_crypto_config
            print(f"{AnsiColors.RED}Input angka tidak valid di suatu tempat.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e:
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback error menu pengaturan:", "SYSTEM")
            show_spinner(1.5, "Error, kembali...")
            
    return current_settings

# --- FUNGSI PENGAMBILAN DATA (MODIFIED) ---
# Catatan: Urutan candle dari API CryptoCompare adalah [tertua, ..., terbaru] per batch.
# Fungsi ini mengembalikan batch, dan pengurutan global + deduplikasi terjadi di main loop.
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None # Untuk pagination, timestamp candle tertua dari batch sebelumnya
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 20 # Anggap besar jika > 20 candle (misal 1x MAX_LIMIT)

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : # Hanya progress bar jika multi-batch besar
            simple_progress_bar(0, total_limit_desired, prefix=f'{AnsiColors.CYAN}{pair_name}{AnsiColors.GREEN} Data:', suffix='Candles', length=35) # Kurangi panjang untuk potrait

    fetch_loop_count = 0
    max_loops_for_safety = (total_limit_desired // CRYPTOCOMPARE_MAX_LIMIT) + 5 # Batas loop untuk mencegah infinite

    while len(all_accumulated_candles) < total_limit_desired and fetch_loop_count < max_loops_for_safety :
        fetch_loop_count += 1
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # Untuk request tunggal atau terakhir, minta sejumlah yang dibutuhkan.
        # Untuk request berulang, minta CRYPTOCOMPARE_MAX_LIMIT.
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        # CryptoCompare: `toTs` is exclusive. Data returned is *before* this timestamp.
        # Jika `current_to_ts` ada (dari candle tertua batch sebelumnya), kita ingin data *sebelum* itu.
        # `limit` akan mengambil `limit` candle *sebelum* `toTs`.
        # Tidak perlu `+1` di limit untuk overlap jika `toTs` dihandle dengan benar.
        # Overlap akan dihandle oleh `merged_candles_dict` di main loop.

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
                 key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                 log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name=pair_name)
            
            response = requests.get(url, params=params, timeout=20) 
            
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}
                try: error_data = response.json()
                except requests.exceptions.JSONDecodeError: pass # Biarkan error_message default
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
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
                    key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break 
            
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break 
            
            raw_candles_from_api = data['Data']['Data'] # Ini adalah [tertua_batch, ..., terbaru_batch]
            
            if not raw_candles_from_api: 
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                # Pastikan nilai OHLC ada, jika tidak ada (sangat jarang untuk historis), lewati candle tsb
                if any(item.get(k) is None for k in ['open', 'high', 'low', 'close', 'time']):
                    log_warning(f"Candle dengan timestamp {item.get('time')} tidak memiliki data OHLC lengkap. Dilewati.", pair_name=pair_name)
                    continue
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom', 0) # Default ke 0 jika tidak ada
                }
                batch_candles_list.append(candle)
            
            # Tidak perlu overlap removal di sini, serahkan ke dict merge di main loop.
            # Itu lebih robust.

            if not batch_candles_list : 
                if is_large_fetch: log_info("Batch menjadi kosong (mungkin semua candle invalid). Akhir data?", pair_name=pair_name)
                break

            # Tambahkan batch baru ke depan list akumulasi. List akumulasi akan jadi [BatchN, BatchN-1, ...]
            # di mana BatchN adalah [tertua_N ... terbaru_N]
            all_accumulated_candles = batch_candles_list + all_accumulated_candles 
            
            if raw_candles_from_api: 
                # `toTs` untuk request berikutnya adalah timestamp candle tertua di batch ini
                current_to_ts = raw_candles_from_api[0]['time'] 
            else: 
                break
            
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{AnsiColors.CYAN}{pair_name}{AnsiColors.GREEN} Data:', suffix='Candles', length=35)

            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break 

            if len(all_accumulated_candles) >= total_limit_desired: break 

            if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                # log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.2) # Jeda kecil, CryptoCompare punya rate limit (kurangi dari 0.3)

        except APIKeyError: 
            raise 
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break 
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error fetch_candles:", pair_name=pair_name) 
            break 
    
    if fetch_loop_count >= max_loops_for_safety:
        log_warning(f"Fetch_candles mencapai max loops ({max_loops_for_safety}) untuk {pair_name}. Mungkin ada masalah logika.", pair_name=pair_name)

    # Pangkas jika terambil lebih banyak (harusnya tidak signifikan jika logika benar)
    # Pengurutan dan pemangkasan final ada di main loop, jadi ini tidak terlalu krusial di sini.
    # if len(all_accumulated_candles) > total_limit_desired:
    #    all_accumulated_candles = all_accumulated_candles[-total_limit_desired:] # Ini salah jika urutan [BatchN, BatchN-1]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{AnsiColors.CYAN}{pair_name}{AnsiColors.GREEN} Data:', suffix='Candles Complete', length=35)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} diterima (target: {total_limit_desired}).", pair_name=pair_name)

    # `all_accumulated_candles` saat ini adalah list of lists (atau gabungan list) yang mungkin belum terurut global.
    # Pengurutan global dilakukan di `start_trading` loop.
    return all_accumulated_candles

# --- LOGIKA STRATEGI ---
# Membutuhkan candles_history TERURUT dari LAMA ke BARU
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 0 = netral, 1 = pivot high, -1 = pivot low
        "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, 
        "active_fib_level": None, "active_fib_line_start_index": None, 
        "entry_price_custom": None, "highest_price_for_trailing": None, 
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None, 
        "emergency_sl_level_custom": None, "position_size": 0, 
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list) 
    if len(series_list) < left_strength + right_strength + 1:
        return pivots 

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        # Cek bar di kiri
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None or series_list[i] is None: is_pivot = False; break
            if is_high: 
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: 
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue 

        # Cek bar di kanan
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None or series_list[i] is None: is_pivot = False; break
            if is_high: 
                if series_list[i] < series_list[i+j]: is_pivot = False; break 
            else: 
                if series_list[i] > series_list[i+j]: is_pivot = False; break 
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
    # Memastikan candles_history terurut [LAMA ... BARU]
    # Pengecekan data candle, pastikan tidak kosong dan format benar
    if not candles_history:
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state
    # Cek candle terakhir sebagai sampel format (atau candle pertama jika semua harus valid)
    # Kode asli memeriksa candles_history[0], tapi karena data [LAMA...BARU], candles_history[-1] (terbaru) lebih relevan untuk `current_candle`.
    # Namun, find_pivots butuh semua. Jadi, cek sampel di awal cukup.
    sample_candle_for_check = candles_history[0] if candles_history else {} # Ambil sampel
    if not all(key in sample_candle_for_check for key in required_keys):
        log_warning(f"{AnsiColors.ORANGE}Kunci OHLC tidak lengkap di sampel data candle.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state
    
    # Pastikan semua harga adalah float atau bisa dikonversi ke float
    try:
        high_prices = [float(c['high']) for c in candles_history]
        low_prices = [float(c['low']) for c in candles_history]
    except (TypeError, ValueError) as e:
        log_warning(f"{AnsiColors.ORANGE}Data harga high/low tidak valid (bukan angka): {e}{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state


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
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1 
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    current_candle = candles_history[current_bar_index_in_list]
    # Pastikan data OHLC di current_candle valid (float)
    try:
        current_candle_open = float(current_candle.get('open'))
        current_candle_high = float(current_candle.get('high'))
        current_candle_low = float(current_candle.get('low'))
        current_candle_close = float(current_candle.get('close'))
    except (TypeError, ValueError) as e:
        log_warning(f"Data OHLC tidak valid untuk candle terbaru @ {current_candle.get('timestamp')}: {e}. Skip evaluasi.", pair_name=pair_name)
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
                    price_key_for_secure_fib = crypto_config["secure_fib_check_price"].lower()
                    price_val_current_candle_for_check = current_candle_close # Default ke close
                    if price_key_for_secure_fib == 'high':
                        price_val_current_candle_for_check = current_candle_high
                    
                    if price_val_current_candle_for_check > calculated_fib_level: # Harga sudah di atas FIB
                        is_fib_late = True
                
                if is_fib_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle_for_check:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state["active_fib_level"] = None 
                    strategy_state["active_fib_line_start_index"] = None
                else:
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=pair_name)
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = current_low_bar_index # Index bar pivot low saat FIB terbentuk

                strategy_state["high_price_for_fib"] = None # Reset setelah digunakan
                strategy_state["high_bar_index_for_fib"] = None

    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        is_bullish_candle = current_candle_close > current_candle_open
        is_closed_above_fib = current_candle_close > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: 
                strategy_state["position_size"] = 1 
                entry_px = current_candle_close 
                strategy_state["entry_price_custom"] = entry_px
                strategy_state["highest_price_for_trailing"] = entry_px 
                strategy_state["trailing_tp_active_custom"] = False 
                strategy_state["current_trailing_stop_level"] = None

                emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl
                
                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()
                
                email_subject = f"BUY Signal: {pair_name}"
                email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\n"
                              f"Entry Price: {entry_px:.5f}\n"
                              f"FIB Level: {strategy_state['active_fib_level']:.5f}\n"
                              f"Emergency SL: {emerg_sl:.5f}\n"
                              f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, crypto_config)

            strategy_state["active_fib_level"] = None # Nonaktifkan FIB setelah entry
            strategy_state["active_fib_line_start_index"] = None
    
    # --- Logika Manajemen Posisi ---
    if strategy_state["position_size"] > 0: 
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle_high)
        if current_high_for_trailing is None: current_high_for_trailing = current_candle_high 
        strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle_high)

        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: profit_percent = 0.0 # Hindari ZeroDivisionError
            else: profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"] is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE 

        if final_stop_for_exit is not None and current_candle_low <= final_stop_for_exit:
            exit_price = min(current_candle_open, final_stop_for_exit) 
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if exit_comment == "Trailing Stop" and pnl < 0:
                exit_color = AnsiColors.RED

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

            email_subject = f"Trade Closed: {pair_name} ({exit_comment})"
            email_body = (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\n"
                          f"Reason: {exit_comment}\n"
                          f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\n"
                          f"PnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, crypto_config)

            strategy_state.update({ # Reset state posisi dengan lebih ringkas
                "position_size": 0, "entry_price_custom": None, "highest_price_for_trailing": None,
                "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
                "emergency_sl_level_custom": None
            })
    
    # Logging status jika masih ada posisi aktif
    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        stop_type_info = "Emergency SL"
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state.get("current_trailing_stop_level")
                stop_type_info = "Trailing SL"

        entry_price_display = strategy_state.get('entry_price_custom', 0.0) # Default ke float
        sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)

    return strategy_state

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict):
    clear_screen_animated() 
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager( # Menggunakan __init__ yang sudah diperbaiki
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings 
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if isinstance(cfg, dict) and cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_val = api_key_manager.get_current_key()
    current_key_display_str = "N/A"
    if current_key_val:
        current_key_display_str = current_key_val[:5] + "..." + current_key_val[-3:] if len(current_key_val) > 8 else current_key_val
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display_str}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {} 
    for config in all_crypto_configs:
        pair_id = f"{config['symbol']}-{config['currency']}_{config['timeframe']}"
        config['pair_name'] = f"{config['symbol']}-{config['currency']}" 

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config['exchange']} | TF: {config['timeframe']}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0,
            "last_attempt_after_all_keys_failed": datetime.min # Untuk cooldown per pair
        }

        initial_candles_target = TARGET_BIG_DATA_CANDLES 
        initial_candles_raw = [] # Data mentah dari fetch_candles
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0; initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break 
            
            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles_raw = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target, 
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Anggap sukses jika tidak ada exception APIKeyError
            except APIKeyError:
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break 
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e:
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                break 
            except Exception as e: # Tangkap error lain yang mungkin dari fetch_candles
                log_error(f"BIG DATA: Error tak terduga saat fetch_candles untuk {config['pair_name']}: {e}", pair_name=config['pair_name'])
                log_exception("Traceback error fetch_candles awal:", config['pair_name'])
                break


        if not initial_candles_raw and not initial_fetch_successful : # Jika gagal total
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Agar tidak langsung dicoba lagi
            continue 

        # Urutkan dan deduplikasi data awal
        if initial_candles_raw:
            merged_initial_dict = {c['timestamp']: c for c in initial_candles_raw} # Deduplikasi dari fetch_candles jika ada
            sorted_initial_candles = sorted(list(merged_initial_dict.values()), key=lambda c: c['timestamp'])
            crypto_data_manager[pair_id]["all_candles_list"] = sorted_initial_candles[-TARGET_BIG_DATA_CANDLES:] # Ambil sejumlah target dari yang terbaru
            log_info(f"BIG DATA: {len(crypto_data_manager[pair_id]['all_candles_list'])} candle awal diterima dan diurutkan.", pair_name=config['pair_name'])

            # Warm-up strategy state
            min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
            if len(sorted_initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(sorted_initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
                for i in range(min_len_for_pivots -1, len(sorted_initial_candles) - 1): 
                    historical_slice = sorted_initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue 
                    temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 
                    crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_for_warmup)
                    if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        crypto_data_manager[pair_id]["strategy_state"].update({
                            "position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                            "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                            "current_trailing_stop_level":None
                        })
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(sorted_initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else: # Jika initial_candles_raw kosong (mungkin API tidak ada data sama sekali)
            log_warning(f"Tidak ada data awal diterima untuk {config['pair_name']}. Warm-up dilewati.", pair_name=config['pair_name'])


        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not crypto_data_manager[pair_id]["big_data_email_sent"]:
                 send_email_notification(f"Data Downloading Complete: {config['pair_name']}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", config)
                 crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])

    animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
    
    # --- Main Trading Loop ---
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf') 
            any_data_fetched_this_cycle = False 

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]
                pair_name = config['pair_name'] 
                
                # Cooldown per pair jika semua key gagal untuk pair tsb
                # Total keys + 1 karena data_fetch_failed_consecutively bisa lebih besar dari total_keys saat semua sudah dicoba.
                # Jika total_keys = 0 (tidak ada key sama sekali), maka (0)+1 = 1.
                # Jika total_keys = 1, maka (1)+1 = 2.
                # data_fetch_failed_consecutively direset jika ada fetch sukses.
                # Jika api_key_manager.total_keys() == 0, kondisi ini tidak akan pernah benar jika gagal sekali.
                # Mungkin lebih baik: data.get("data_fetch_failed_consecutively", 0) > api_key_manager.total_keys()
                # Atau jika total_keys() == 0, jangan pernah masuk cooldown permanen, tapi APIKeyManager akan mencegah fetch.
                # Logika cooldown: Jika semua key yang *ada* sudah dicoba dan gagal untuk pair ini.
                
                # Jika tidak ada API key valid sama sekali dari awal, loop ini tidak akan berguna banyak.
                # APIKeyManager.get_current_key() akan selalu None. fetch_candles akan raise APIKeyError.
                # switch_to_next_key() akan return None.
                # Jadi data_fetch_failed_consecutively akan bertambah terus.

                # Jika tidak ada API key yang valid, data_fetch_failed_consecutively akan terus bertambah
                # dan akan masuk cooldown.
                # Kondisi cooldown:
                max_fails_before_cooldown = api_key_manager.total_keys()
                if max_fails_before_cooldown == 0: max_fails_before_cooldown = 1 # Jika tidak ada key, gagal sekali langsung cooldown (meski tidak akan fetch)
                
                if data.get("data_fetch_failed_consecutively", 0) >= max_fails_before_cooldown and api_key_manager.total_keys() > 0 : 
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: 
                        # log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name) # Bisa terlalu verbose
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600) # Pertimbangkan cooldown dalam sleep global
                        continue 
                    else:
                        data["data_fetch_failed_consecutively"] = 0 
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()
                
                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    if config['timeframe'] == "minute": required_interval_for_this_pair = 50 # Kurangi sedikit lagi
                    elif config['timeframe'] == "day": required_interval_for_this_pair = 3600 * 23.5 # Kurangi lagi
                    else: required_interval_for_this_pair = 3550 # Kurangi lagi (kurang dari 1 jam)
                else: 
                    required_interval_for_this_pair = config['refresh_interval_seconds']
                
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue 

                log_info(f"Memproses {pair_name} (interval {required_interval_for_this_pair}s)...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time 
                num_candles_before_fetch = len(data["all_candles_list"])
                
                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                new_candles_batch_raw = []
                fetch_update_successful_for_this_pair = False
                
                # max_retries_for_this_pair_update sudah memperhitungkan total_keys().
                # Jika total_keys() == 0, loop ini tidak akan berjalan, current_api_key_for_attempt akan None.
                # fetch_candles akan langsung raise APIKeyError jika key None.
                
                retries_done_for_this_pair_update = 0
                # Loop retry HANYA jika ada key. Jika tidak, fetch_candles akan gagal langsung.
                
                # Jika tidak ada API key sama sekali, APIKeyManager.get_current_key() akan None
                # dan fetch_candles akan raise APIKeyError. Ini akan ditangkap.
                # switch_to_next_key() tidak akan melakukan apa-apa.
                # data_fetch_failed_consecutively akan bertambah.

                while retries_done_for_this_pair_update < (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) and \
                      not fetch_update_successful_for_this_pair:
                    
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    # Jika tidak ada key sama sekali, current_api_key_for_attempt akan None.
                    # fetch_candles akan throw APIKeyError "API Key tidak tersedia". Ini akan ditangkap.
                    # Lalu switch_to_next_key tidak akan menemukan key baru.
                    # retries_done_for_this_pair_update akan bertambah. Loop akan berhenti.
                    # Ini sudah benar.

                    limit_fetch = 3 
                    if data["big_data_collection_phase_active"]:
                        needed = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"])
                        limit_fetch = min(needed, CRYPTOCOMPARE_MAX_LIMIT)
                        limit_fetch = max(limit_fetch, 1 if needed > 0 else 0) # Ambil minimal 1 jika masih butuh, atau 0 jika sudah cukup
                        if limit_fetch <=0 : 
                             fetch_update_successful_for_this_pair = True ; break
                    
                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch_raw = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch, 
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0 
                        any_data_fetched_this_cycle = True 
                    
                    except APIKeyError:
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        if not api_key_manager.switch_to_next_key(): 
                            log_error(f"Tidak ada lagi API key tersedia (global) setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break # Keluar dari loop retry pair ini
                        # Jika switch_to_next_key berhasil, loop retry akan lanjut dengan key baru
                    except requests.exceptions.RequestException as e:
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break # Keluar loop retry untuk pair ini di siklus ini
                    except Exception as e:
                        log_error(f"Error tak terduga saat fetch_candles untuk update {pair_name}: {e}", pair_name=pair_name)
                        log_exception("Traceback error fetch_candles update:", pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break
                    finally: # Selalu increment retry counter dalam loop ini
                        retries_done_for_this_pair_update += 1
                
                if data.get("data_fetch_failed_consecutively", 0) >= max_fails_before_cooldown and api_key_manager.total_keys() > 0:
                    data["last_attempt_after_all_keys_failed"] = datetime.now() 
                    log_warning(f"Semua API key ({api_key_manager.total_keys()}) telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)

                if not fetch_update_successful_for_this_pair or not new_candles_batch_raw:
                    if fetch_update_successful_for_this_pair and not new_candles_batch_raw and not data["big_data_collection_phase_active"]:
                        log_info(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {pair_name} (fetch dianggap berhasil, mungkin tidak ada data baru dari API).{AnsiColors.ENDC}", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    continue 

                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0; updated_count_this_batch = 0 

                for candle in new_candles_batch_raw: # new_candles_batch_raw adalah list dari fetch_candles
                    if not isinstance(candle, dict) or 'timestamp' not in candle: continue # Skip invalid candle
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle; newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : 
                        merged_candles_dict[ts] = candle; updated_count_this_batch +=1
                
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = (len(all_candles_list_temp) - len(data["all_candles_list"])) + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                elif new_candles_batch_raw : 
                     log_info("Tidak ada candle dengan timestamp baru atau update konten. Data terakhir mungkin identik.", pair_name=pair_name)
                
                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!{AnsiColors.ENDC}", pair_name=pair_name)
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data["big_data_email_sent"]:
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", config)
                            data["big_data_email_sent"] = True
                        
                        data["big_data_collection_phase_active"] = False 
                        # active_cryptos_still_in_big_data_collection dikurangi di luar loop pair, di awal loop utama
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------{AnsiColors.ENDC}", pair_name=pair_name)
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
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: 
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
            # Akhir loop 'for pair_id, data in crypto_data_manager.items():'

            # --- Tentukan Durasi Sleep Global ---
            # Hitung ulang active_cryptos_still_in_big_data_collection untuk sleep logic
            current_active_big_data_pairs = sum(1 for d in crypto_data_manager.values() if d["big_data_collection_phase_active"])
            sleep_duration = 15 # Default

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None and api_key_manager.total_keys() > 0: 
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam.", pair_name="SYSTEM")
                sleep_duration = 3600 
            elif current_active_big_data_pairs > 0:
                min_big_data_interval = float('inf')
                for pid_sleep, pdata_sleep in crypto_data_manager.items():
                    if pdata_sleep["big_data_collection_phase_active"]:
                        pconfig_sleep = pdata_sleep["config"]
                        interval_sleep = 50 
                        if pconfig_sleep['timeframe'] == "day": interval_sleep = 3600*23.5
                        elif pconfig_sleep['timeframe'] == "hour": interval_sleep = 3550
                        min_big_data_interval = min(min_big_data_interval, interval_sleep)
                
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) # Max 30s saat big data
                log_debug(f"Masih ada {current_active_big_data_pairs} pair dalam BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua pair live
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0:
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                else: # Fallback jika min_overall_next_refresh_seconds tidak valid
                    sleep_duration = global_settings_dict.get("api_settings",{}).get("refresh_interval_seconds", 60) 
                log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM")
            
            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu ~{int(sleep_duration)}s...")
        # Akhir loop 'while True:'
    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error trading loop:", pair_name="SYSTEM")
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()

    while True:
        clear_screen_animated() 
        animated_text_display("========= Crypto Strategy Runner (Multi-Pair) =========", color=AnsiColors.HEADER, delay=0.005)

        pick_title_main = "" 
        active_configs = [cfg for cfg in settings.get("cryptos", []) if isinstance(cfg,dict) and cfg.get("enabled", True)]
        if active_configs:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs)}) ---\n"
            for i, cfg in enumerate(active_configs): 
                pick_title_main += f"  {i+1}. {cfg.get('symbol','?')}-{cfg.get('currency','?')} (TF: {cfg.get('timeframe','?')}, Exch: {cfg.get('exchange','?')})\n"
        else: 
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s = settings.get("api_settings", {})
        primary_key_val = api_s.get('primary_key', '')
        primary_key_display = 'BELUM DIATUR'
        if primary_key_val and isinstance(primary_key_val, str):
             primary_key_display = primary_key_val[:5] + "..." + primary_key_val[-3:] if len(primary_key_val) > 10 else primary_key_val
        
        num_recovery_keys = len([k for k in api_s.get('recovery_keys',[]) if k and isinstance(k,str)])

        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display} | Recovery Keys: {num_recovery_keys}\n"
        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"

        options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]

        try:
            option_text, index = pick(options_plain, pick_title_main, indicator=f'{AnsiColors.GREEN}=>{AnsiColors.ENDC}', default_index=0)
            
            if index == 0: 
                start_trading(settings)
            elif index == 1: 
                settings = settings_menu(settings) # settings_menu akan mengembalikan settings yang mungkin diubah
                # settings di-pass by reference (untuk dict), jadi perubahan di settings_menu akan tercermin.
                # Namun, jika settings_menu mengembalikan struktur baru, ini sudah benar.
            elif index == 2: 
                log_info("Aplikasi ditutup.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
                show_spinner(0.5, "Exiting")
                break # Keluar dari loop main_menu
        except Exception as e: 
            log_warning(f"Operasi menu dibatalkan atau error: {e}", pair_name="SYSTEM")
            if isinstance(e, KeyboardInterrupt): 
                log_info("Aplikasi dihentikan oleh pengguna dari menu utama.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE)
                break # Keluar dari loop main_menu
            show_spinner(1, "Error menu, coba lagi...")

if __name__ == "__main__": # PERBAIKAN: name -> __name__
    try:
        # Inisialisasi awal yang mungkin penting
        if not os.path.exists(SETTINGS_FILE):
            log_info(f"File {SETTINGS_FILE} tidak ditemukan, akan dibuat saat menu pengaturan pertama kali diakses atau saat load_settings dipanggil.")
            # load_settings() akan otomatis membuat file jika tidak ada.
        
        main_menu()

    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"\n{AnsiColors.ORANGE}Aplikasi dihentikan paksa. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}Terjadi error fatal tak terduga di level utama aplikasi:{AnsiColors.ENDC}")
        print(f"{AnsiColors.RED}{e}{AnsiColors.ENDC}")
        # logger bisa jadi belum terinisialisasi jika error terjadi sangat awal
        if logger.hasHandlers():
            logger.error(f"Error fatal di level utama: {e}", pair_name="SYSTEM_CRITICAL")
            logger.exception("MAIN LEVEL EXCEPTION:")
        else: # Fallback print jika logger tidak ada
            import traceback
            traceback.print_exc()
            
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        log_info("Skrip Selesai.", pair_name="SYSTEM_SHUTDOWN")
