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

# --- ANSI COLOR CODES --- (Sama seperti sebelumnya, ini aman digunakan di luar 'pick')
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
    YELLOW_BG = '\033[43m' # Contoh tambahan jika diperlukan

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
    spinner_chars = ['-', '\\', '|', '/']
    start_time = time.time()
    idx = 0
    sys.stdout.write(AnsiColors.MAGENTA) # Warna untuk spinner
    while (time.time() - start_time) < duration_seconds:
        # Pastikan pesan tidak melebihi lebar terminal yang wajar sebelum spinner
        display_message = message[:os.get_terminal_size().columns - 5] if os.isatty() else message
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") # Clear spinner line
    sys.stdout.write(AnsiColors.ENDC) # Reset warna
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """Membuat dan menampilkan progress bar sederhana."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # Pastikan output tidak melebihi lebar terminal
    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    max_len = os.get_terminal_size().columns if os.isatty() else 80
    sys.stdout.write(progress_line[:max_len]) # Potong jika terlalu panjang
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
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
console_formatter_template = '%(asctime)s - {bold}%(levelname)s{endc} - {cyan}[%(pair_name)s]{endc} - %(message)s'
ch.setFormatter(logging.Formatter(
    console_formatter_template.format(bold=AnsiColors.BOLD, endc=AnsiColors.ENDC, cyan=AnsiColors.CYAN)
))
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
CRYPTOCOMPARE_MAX_LIMIT = 1999 # Sesuai dokumentasi CryptoCompare, limit per request bisa sampai 2000
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# --- FUNGSI CLEAR SCREEN (DIMODIFIKASI dengan animasi kecil) ---
def clear_screen_animated():
    """Membersihkan layar terminal dengan sedikit animasi."""
    # Optional: Tampilkan pesan singkat sebelum clear
    # animated_text_display("Clearing...", delay=0.01, color=AnsiColors.MAGENTA, new_line=False)
    # show_spinner(0.2, "Clearing screen")
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
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
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
                if all(dummy_email_cfg.values()):
                     send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi pergantian API key.")
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
                if all(dummy_email_cfg.values()):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal.")
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
            winsound.Beep(1000, 500) # Frekuensi 1kHz, durasi 500ms
        else:
            # Bell character, standar. Mungkin perlu \a dua kali untuk Termux/Linux
            print('\a', end='', flush=True) 
            time.sleep(0.2) # Jeda kecil jika perlu
            print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('symbol', 'GLOBAL_EMAIL') # Dapatkan pair_name dari config jika ada
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
        "primary_key": "YOUR_PRIMARY_KEY",
        "recovery_keys": [],
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com",
        "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com"
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                settings = json.load(f)
                # Pastikan api_settings ada dan memiliki semua kunci default
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v
                
                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = []
                # Pastikan setiap crypto config memiliki ID dan status enabled
                for crypto_cfg in settings["cryptos"]:
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True # Default ke True jika tidak ada
                return settings
            except json.JSONDecodeError:
                log_error("Error membaca settings.json. Menggunakan default.")
                # Return a structure that includes api_settings and cryptos
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config):
    clear_screen_animated() # Ganti dengan versi animasi
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)
    
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG'))
    
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid.{AnsiColors.ENDC}"); new_config["timeframe"] = new_config.get('timeframe','hour')
    
    refresh_input = input(f"{AnsiColors.BLUE}Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}")
    new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input or new_config.get('refresh_interval_seconds',60)))

    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}") or new_config.get('left_strength',50))
    new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}") or new_config.get('right_strength',150))

    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}") or new_config.get('profit_target_percent_activation',5.0))
    new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}") or new_config.get('trailing_stop_gap_percent',5.0))
    new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}") or new_config.get('emergency_sl_percent',10.0))
    
    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))
    secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}") or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid.{AnsiColors.ENDC}"); new_config["secure_fib_check_price"] = new_config.get('secure_fib_check_price','Close')

    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global dari API Settings (jika notif global aktif).{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')
    new_config["email_sender_app_password"] = input(f"{AnsiColors.BLUE}App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')
    new_config["email_receiver_address"] = input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')
    
    return new_config

def settings_menu(current_settings):
    while True:
        clear_screen_animated() # Ganti dengan versi animasi
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10: primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        
        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len(recovery_keys)

        # Judul untuk 'pick' dibuat tanpa kode ANSI
        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"
        
        if not current_settings["cryptos"]:
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        for i, crypto_conf in enumerate(current_settings["cryptos"]):
            status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
            pick_title_settings += f"  {i+1}. {crypto_conf['symbol']}-{crypto_conf['currency']} ({crypto_conf['timeframe']}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

        original_options_structure = [
            ("header", "--- Pengaturan API & Global ---"),
            ("option", "Atur Primary API Key"),
            ("option", "Kelola Recovery API Keys"),
            ("option", "Atur Email Global untuk Notifikasi Sistem (API Key Switch, dll)"),
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"),
            ("option", "Ubah Konfigurasi Crypto"),
            ("option", "Hapus Konfigurasi Crypto"),
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")
        ]
        
        selectable_options = [text for type, text in original_options_structure if type == "option"]
        raw_option_texts = selectable_options[:] 

        option_text, index = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        action_choice = raw_option_texts.index(option_text)

        try:
            clear_screen_animated() # Ganti dengan versi animasi
            if action_choice == 0: 
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ") or api_s.get('primary_key','')
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
            elif action_choice == 1: 
                while True:
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = api_s.get('recovery_keys', [])
                    if not current_recovery:
                        recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"
                    
                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]
                    rec_option_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    clear_screen_animated()

                    if rec_index == 0: 
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else:
                            print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 1: 
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
                            show_spinner(1, "Kembali...")
                            continue
                        try:
                            idx_del = int(input("Nomor recovery key yang akan dihapus: ")) - 1
                            if 0 <= idx_del < len(current_recovery):
                                removed = current_recovery.pop(idx_del)
                                api_s['recovery_keys'] = current_recovery
                                save_settings(current_settings)
                                print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                            else:
                                print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                        except ValueError:
                            print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 2: 
                        break
            elif action_choice == 2: 
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global (API Key switch, dll)? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ") or api_s.get('email_sender_address','')
                api_s['email_sender_app_password'] = input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ") or api_s.get('email_sender_app_password','')
                api_s['email_receiver_address_admin'] = input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ") or api_s.get('email_receiver_address_admin','')
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 3: 
                new_crypto_conf = get_default_crypto_config()
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) 
                current_settings["cryptos"].append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 4: 
                if not current_settings["cryptos"]: 
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); 
                    continue
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                # Menampilkan daftar untuk dipilih sebelum input angka
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                     print(f"  {i+1}. {crypto_conf['symbol']}-{crypto_conf['currency']}")
                idx_choice = int(input("Nomor konfigurasi crypto yang akan diubah: ")) - 1

                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 5: 
                if not current_settings["cryptos"]: 
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); 
                    continue
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                 # Menampilkan daftar untuk dipilih sebelum input angka
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                     print(f"  {i+1}. {crypto_conf['symbol']}-{crypto_conf['currency']}")
                idx_choice = int(input("Nomor konfigurasi crypto yang akan dihapus: ")) - 1

                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                    current_settings["cryptos"].pop(idx_choice)
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 6: 
                break
        except ValueError:
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e:
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            show_spinner(1.5, "Error, kembali...")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA (MODIFIED) ---
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
    is_large_fetch = total_limit_desired > 10 # Misal, request >10 candle dianggap "besar" untuk progress bar
    
    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)
        # Hanya tampilkan progress bar jika memang mengambil banyak data secara berulang
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT :
            simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)


    fetch_loop_count = 0 # Untuk mengontrol seberapa sering progress bar diupdate jika ada banyak loop kecil
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # Untuk request tunggal atau terakhir, minta sejumlah yang dibutuhkan.
        # Untuk request berulang, minta CRYPTOCOMPARE_MAX_LIMIT.
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        # Koreksi: jika kita sudah punya `current_to_ts` (artinya ini request kedua dst dalam loop),
        # dan kita masih butuh > 1 candle, maka limitnya bisa jadi `candles_still_needed + 1` (untuk overlap)
        # atau `CRYPTOCOMPARE_MAX_LIMIT` (jika butuh banyak).
        if current_to_ts is not None and candles_still_needed > 1 : # Ini adalah request lanjutan
             limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)
        
        if limit_for_this_api_call <= 0: break # Sudah cukup

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts
        
        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Hanya log debug jika memang multi-batch besar
                 log_debug(f"Fetching batch (Key: ...{current_api_key_to_use[-5:]}, Limit: {limit_for_this_api_call})", pair_name=pair_name)
            
            response = requests.get(url, params=params, timeout=20) # Timeout 20 detik
            
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = response.json() if response.content else {}
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() # Untuk error HTTP lainnya
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active" # Ditambahkan
                ]
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    # Error lain dari API, mungkin pair tidak ada di exchange tsb
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Hentikan loop untuk pair ini jika ada error data non-key
            
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break # Tidak ada data lagi atau data kosong
            
            raw_candles_from_api = data['Data']['Data']
            
            if not raw_candles_from_api: # Jika list kosong setelah cek di atas
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            # Konversi data mentah ke format yang kita inginkan
            batch_candles_list = []
            for item in raw_candles_from_api:
                # Pastikan ada nilai default jika open/high/low/close tidak ada (meski jarang untuk data historis)
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom') # atau 'volumeto' tergantung kebutuhan
                }
                batch_candles_list.append(candle)

            # Hapus candle tumpang tindih jika ini bukan pengambilan pertama
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                # Candle terakhir dari batch baru (paling tua) sama dengan candle pertama dari yang sudah ada (paling baru)
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop() # Hapus candle duplikat dari batch baru
            
            if not batch_candles_list and current_to_ts is not None : # Jika setelah hapus overlap, batch jadi kosong
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break


            all_accumulated_candles = batch_candles_list + all_accumulated_candles # Tambah di depan
            
            if raw_candles_from_api: # Pastikan tidak kosong sebelum akses index 0
                current_to_ts = raw_candles_from_api[0]['time'] # Timestamp candle tertua di batch ini, untuk request berikutnya
            else: # Seharusnya tidak terjadi jika cek di atas sudah benar
                break
            
            # Update progress bar jika ini adalah fetch besar dan multi-batch
            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): # Update tiap 2 iterasi atau jika selesai
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

            if len(raw_candles_from_api) < limit_for_this_api_call:
                # Jika API mengembalikan lebih sedikit dari yang diminta (dan bukan karena limit max 2000),
                # berarti kita sudah mencapai akhir dari data historis yang tersedia.
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break # Keluar dari loop while

            if len(all_accumulated_candles) >= total_limit_desired: break # Target tercapai

            # Jeda kecil antar request jika mengambil banyak data berulang kali
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3) # Jeda kecil, CryptoCompare punya rate limit

        except APIKeyError: # Dilempar dari dalam jika ada error terkait API key
            raise # Biarkan APIKeyManager yang menangani penggantian key
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Hentikan pengambilan untuk pair ini pada error jaringan
        except Exception as e:
            # Tangkap error lain yang mungkin terjadi (misal parsing JSON, dll)
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error:", pair_name=pair_name) # Log traceback juga
            break # Hentikan untuk pair ini

    # Pastikan jumlah candle tidak melebihi yang diinginkan (jika terambil lebih karena pembulatan limit)
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        # Pastikan progress bar selesai 100% jika belum
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
             simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)
    
    return all_accumulated_candles
# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 0 = netral, 1 = pivot high, -1 = pivot low
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, # Harga high dari pivot high terakhir, untuk kalkulasi FIB
        "high_bar_index_for_fib": None, # Index bar dari pivot high terakhir
        "active_fib_level": None, # Harga level FIB 0.5 yang aktif
        "active_fib_line_start_index": None, # Index bar saat FIB 0.5 terbentuk
        "entry_price_custom": None, # Harga entry custom
        "highest_price_for_trailing": None, # Harga tertinggi setelah entry untuk trailing TP
        "trailing_tp_active_custom": False, # Status trailing TP custom
        "current_trailing_stop_level": None, # Level stop loss dari trailing TP
        "emergency_sl_level_custom": None, # Level stop loss darurat custom
        "position_size": 0, # Ukuran posisi (misal 1 untuk long, -1 untuk short, 0 untuk flat)
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list) # List untuk menyimpan harga pivot, None jika bukan pivot
    if len(series_list) < left_strength + right_strength + 1:
        return pivots # Data tidak cukup

    # Iterasi dari index 'left_strength' hingga 'len(series) - right_strength - 1'
    # Ini memastikan ada cukup bar di kiri dan kanan untuk perbandingan
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        # Cek bar di kiri
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None or series_list[i] is None: is_pivot = False; break
            if is_high: # Mencari Pivot High
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # Mencari Pivot Low
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue # Lanjut ke bar berikutnya jika bukan pivot dari sisi kiri

        # Cek bar di kanan
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None or series_list[i] is None: is_pivot = False; break
            if is_high: # Mencari Pivot High
                # Untuk pivot high, harga di i harus LEBIH BESAR dari harga di i+j
                # Jika harga di i SAMA DENGAN atau LEBIH KECIL dari harga di i+j, itu BUKAN pivot high yang ketat.
                # Beberapa definisi memperbolehkan sama dengan, tapi untuk yang lebih jelas, kita pakai < (atau <= jika high sama diizinkan)
                if series_list[i] < series_list[i+j]: is_pivot = False; break 
                # Pertimbangkan: series_list[i] <= series_list[i+j] jika high yang sama diizinkan sebagai bagian dari plateau
            else: # Mencari Pivot Low
                # Untuk pivot low, harga di i harus LEBIH KECIL dari harga di i+j
                if series_list[i] > series_list[i+j]: is_pivot = False; break 
                # Pertimbangkan: series_list[i] >= series_list[i+j] jika low yang sama diizinkan
        
        if is_pivot:
            pivots[i] = series_list[i] # Simpan harga pivot di index yang sesuai
            
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    
    # Reset confirmed pivots for this run, they will be re-evaluated
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state

    high_prices = [c['high'] for c in candles_history]
    low_prices = [c['low'] for c in candles_history]

    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    # Pivot dianggap 'confirmed' atau 'event' terjadi pada bar saat ini (current_bar_index)
    # JIKA bar tersebut adalah `right_strength` bar SETELAH bar pivot aktual.
    # Jadi, jika pivot terjadi di index `p`, maka event konfirmasinya ada di `p + right_strength`.
    # Kita bekerja mundur dari bar saat ini. Bar saat ini adalah `len(candles_history) - 1`.
    # Maka, bar pivot aktual yang mungkin terkonfirmasi adalah `(len(candles_history) - 1) - right_strength`.
    current_bar_index_in_list = len(candles_history) - 1 
    if current_bar_index_in_list < 0 : return strategy_state # Seharusnya tidak terjadi jika ada candles_history

    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength
    
    # Ambil harga pivot mentah pada index di mana pivot tersebut akan dikonfirmasi
    # (yaitu, `right_strength` bar setelah pivot aktual terjadi)
    raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high] if 0 <= idx_pivot_event_high < len(raw_pivot_highs) else None
    raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low] if 0 <= idx_pivot_event_low < len(raw_pivot_lows) else None

    # --- Logika Pivot High Terkonfirmasi ---
    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["last_signal_type"] = 1 # Tandai bahwa sinyal terakhir adalah Pivot High
        pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    # --- Logika Pivot Low Terkonfirmasi ---
    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1 # Tandai bahwa sinyal terakhir adalah Pivot Low
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)
    
    # Ambil data candle saat ini (bar terakhir dalam history)
    current_candle = candles_history[current_bar_index_in_list] 
    # Pastikan data OHLC ada untuk candle terbaru
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp')}. Skip evaluasi.", pair_name=pair_name)
        return strategy_state

    # --- Logika Fibonacci dan Entry ---
    # Jika ada Pivot High baru terkonfirmasi, simpan harganya untuk kalkulasi FIB nanti
    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high # Index bar pivot high
        # Jika ada FIB aktif sebelumnya, reset karena ada pivot high baru
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    # Jika ada Pivot Low baru terkonfirmasi DAN kita sudah punya harga High untuk FIB
    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low # Index bar pivot low

            # Pastikan Pivot Low terjadi SETELAH Pivot High yang disimpan
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                # Hitung level FIB 0.5
                calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0
                
                is_fib_late = False
                if crypto_config["enable_secure_fib"]:
                    # Cek apakah harga saat ini (Close atau High candle terakhir) sudah di atas FIB
                    price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                    if price_val_current_candle is not None and price_val_current_candle > calculated_fib_level:
                        is_fib_late = True
                
                if is_fib_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state["active_fib_level"] = None # Jangan aktifkan FIB jika terlambat
                    strategy_state["active_fib_line_start_index"] = None
                else:
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=pair_name)
                    strategy_state["active_fib_level"] = calculated_fib_level
                    strategy_state["active_fib_line_start_index"] = current_low_bar_index

                # Reset harga high untuk FIB agar tidak digunakan lagi sampai ada pivot high baru
                strategy_state["high_price_for_fib"] = None
                strategy_state["high_bar_index_for_fib"] = None
    
    # Cek kondisi entry jika FIB 0.5 aktif
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        # Entry jika candle saat ini bullish (close > open) DAN close di atas level FIB 0.5
        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: # Hanya entry jika belum ada posisi
                strategy_state["position_size"] = 1 # Masuk posisi long
                entry_px = current_candle['close'] # Harga entry adalah harga close candle pemicu
                strategy_state["entry_price_custom"] = entry_px
                strategy_state["highest_price_for_trailing"] = entry_px # Inisialisasi harga tertinggi
                strategy_state["trailing_tp_active_custom"] = False # Trailing TP belum aktif
                strategy_state["current_trailing_stop_level"] = None

                # Set Emergency SL
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

            # Setelah entry atau jika sudah ada posisi, nonaktifkan FIB
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    # --- Logika Manajemen Posisi (Trailing TP & Emergency SL) ---
    if strategy_state["position_size"] > 0: # Jika sedang dalam posisi long
        # Update harga tertinggi yang pernah dicapai sejak entry
        # Gunakan strategy_state.get untuk default jika kunci belum ada (meski seharusnya sudah diinisialisasi)
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle['high'])
        if current_high_for_trailing is None: current_high_for_trailing = current_candle['high'] # Safety net
        strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        # Cek aktivasi Trailing TP
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0 if strategy_state["entry_price_custom"] != 0 else 0
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        # Jika Trailing TP aktif, update level stop loss-nya
        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"] is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            # Hanya update jika stop baru lebih tinggi dari stop lama (atau jika stop lama belum ada)
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        # Tentukan level stop yang akan digunakan untuk exit (antara Emergency SL atau Trailing SL)
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            # Jika Trailing SL aktif dan lebih tinggi dari Emergency SL, gunakan Trailing SL
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE # Default biru untuk trailing stop
        
        # Cek apakah harga low saat ini menyentuh level stop
        if final_stop_for_exit is not None and current_candle['low'] <= final_stop_for_exit:
            # Harga exit diasumsikan terjadi pada level stop atau open candle (mana yang lebih buruk untuk kita, yaitu lebih rendah)
            exit_price = min(current_candle['open'], final_stop_for_exit) 
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            # Jika Trailing Stop kena tapi PnL negatif, warnai merah
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

            # Reset state posisi
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
    
    # Logging status jika masih ada posisi aktif
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
    clear_screen_animated() # Ganti dengan versi animasi
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # global_settings_for_email
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return
    
    animated_text_display("================ MULTI-CRYPTO STRATEGY START (Key Recovery Enabled) ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display = api_key_manager.get_current_key()
    if current_key_display and len(current_key_display) > 8: current_key_display = current_key_display[:5] + "..." + current_key_display[-3:]
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {} # Tempat menyimpan data dan state per pair
    for config in all_crypto_configs:
        # Buat ID unik untuk setiap pair berdasarkan simbol, mata uang, dan timeframe
        pair_id = f"{config['symbol']}-{config['currency']}_{config['timeframe']}"
        config['pair_name'] = f"{config['symbol']}-{config['currency']}" # Tambahkan ke config untuk logging mudah
        
        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config['exchange']} | TF: {config['timeframe']}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, # Awalnya selalu kumpulkan big data
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, # Waktu terakhir pengambilan candle
            "data_fetch_failed_consecutively": 0 # Counter kegagalan fetch berturut-turut untuk pair ini
        }

        # --- Ambil Data Awal (Warm-up) ---
        # Tidak lagi CRYPTOCOMPARE_MAX_LIMIT, tapi TARGET_BIG_DATA_CANDLES untuk warm-up yang lebih baik
        # Namun, fetch_candles akan memecahnya menjadi batch CRYPTOCOMPARE_MAX_LIMIT
        initial_candles_target = TARGET_BIG_DATA_CANDLES 
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key:
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break # Tidak ada key lagi secara global
            
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
                if not api_key_manager.switch_to_next_key(): break # Tidak ada key lagi untuk dicoba
                retries_done_initial +=1 # Hitung upaya ini
            except requests.exceptions.RequestException as e:
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                # Tergantung kebijakan, bisa break atau tidak. Saat ini break.
                break # Keluar dari loop jika ada error jaringan pada pengambilan awal pair ini

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Tandai gagal agar tidak mencoba lagi di state ini
            # Mungkin set last_candle_fetch_time ke waktu sekarang agar tidak langsung dicoba lagi
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now()
            continue # Lanjut ke konfigurasi crypto berikutnya
        
        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])

        # Warm-up strategy state dengan data historis (kecuali candle terakhir)
        if initial_candles:
            # Hanya jalankan warm-up jika data cukup untuk pivot
            min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
                # Iterasi melalui data historis untuk "memanaskan" state, jangan proses candle terakhir.
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # -1 agar candle terakhir tidak memicu sinyal prematur
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue # Lewati jika slice terlalu pendek

                    # Salin state saat ini, tapi pastikan tidak ada posisi aktif selama warm-up
                    temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Pastikan tidak ada posisi saat warm-up
                    
                    crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_for_warmup)
                    
                    # Setelah setiap iterasi warm-up, pastikan state posisi direset
                    # Ini penting agar sinyal dari data historis tidak terbawa sebagai posisi aktif
                    if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: 
                        crypto_data_manager[pair_id]["strategy_state"] = {
                            **crypto_data_manager[pair_id]["strategy_state"], 
                            **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                               "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                               "current_trailing_stop_level":None}
                        }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else:
            log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])
        
        # Cek apakah TARGET_BIG_DATA_CANDLES sudah tercapai setelah initial fetch
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
            min_overall_next_refresh_seconds = float('inf') # Untuk menentukan sleep global
            any_data_fetched_this_cycle = False # Untuk cek apakah ada fetch sukses di siklus ini

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]
                pair_name = config['pair_name'] # Sudah ada di config
                
                # Jika pair ini gagal fetch berkali-kali (semua key sudah dicoba), tunggu sejam sebelum coba lagi
                # Hitungan kegagalan adalah per pair, bukan global.
                # +1 karena index key dari 0, jadi total_keys() adalah jumlah percobaan maksimum sebelum semua key habis.
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Tunggu 1 jam
                        log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name)
                        continue # Lewati pair ini untuk sementara
                    else:
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter untuk mencoba lagi
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)


                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()
                
                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Interval fetch saat pengumpulan big data (lebih sering)
                    if config['timeframe'] == "minute": required_interval_for_this_pair = 55 # sedikit kurang dari 1 menit
                    elif config['timeframe'] == "day": required_interval_for_this_pair = 3600 * 23.8 # sedikit kurang dari 1 hari
                    else: required_interval_for_this_pair = 3580 # sedikit kurang dari 1 jam
                else: # Fase live analysis
                    required_interval_for_this_pair = config['refresh_interval_seconds']
                
                # Cek apakah sudah waktunya fetch untuk pair ini
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    # Hitung sisa waktu hingga refresh berikutnya untuk pair ini dan update min_overall_next_refresh_seconds
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Belum waktunya, lanjut ke pair berikutnya

                log_info(f"Memproses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time # Update waktu fetch SEBELUM mencoba
                num_candles_before_fetch = len(data["all_candles_list"])
                
                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                # Simpan index key saat ini SEBELUM mencoba fetch untuk pair ini.
                # Ini penting untuk logika penggantian key yang benar jika terjadi APIKeyError.
                key_index_at_start_of_pair_fetch_attempt = api_key_manager.get_current_key_index()
                max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_for_this_pair_update = 0

                while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt:
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {pair_name}.", pair_name=pair_name)
                        # Jika semua key habis global, tidak ada gunanya mencoba lagi untuk pair ini di siklus ini
                        break 

                    limit_fetch = 3 # Ambil 3 candle terbaru untuk update (current, previous, pre-previous)
                    # Jika dalam fase big data, kita mungkin ingin mengambil lebih banyak per batch untuk mempercepat
                    if data["big_data_collection_phase_active"]:
                        # Ambil sisa atau CRYPTOCOMPARE_MAX_LIMIT, mana yang lebih kecil
                        limit_fetch = min(TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"]), CRYPTOCOMPARE_MAX_LIMIT)
                        limit_fetch = max(limit_fetch, 3) # Pastikan minimal 3 jika masih butuh banyak
                        if limit_fetch <=0 : # Sudah terkumpul semua
                             fetch_update_successful_for_this_pair = True # Anggap sukses karena tidak ada yg perlu di-fetch
                             break


                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch, 
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter kegagalan untuk pair ini
                        any_data_fetched_this_cycle = True # Tandai bahwa ada fetch sukses di siklus global ini
                    
                    except APIKeyError:
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        
                        # Coba ganti key. Jika tidak ada key lagi, break dari loop retry pair ini.
                        if not api_key_manager.switch_to_next_key(): 
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break 
                        retries_done_for_this_pair_update += 1 # Hitung sebagai upaya retry untuk pair ini

                    except requests.exceptions.RequestException as e:
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        # Untuk error jaringan, biasanya tidak langsung ganti key, tapi tunggu dan coba lagi nanti.
                        # Jadi, break dari loop retry untuk pair ini di siklus ini.
                        break 
                
                # Jika setelah semua retry, pair ini masih gagal fetch (karena semua key dicoba dan gagal untuk pair ini)
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data["last_attempt_after_all_keys_failed"] = datetime.now() # Catat waktu untuk cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)


                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data["big_data_collection_phase_active"]:
                        # Ini bisa terjadi jika API tidak punya data baru sama sekali (misal pasar tutup atau pair tidak likuid)
                        # atau jika limit_fetch di kalkulasi jadi 0 di fase big data
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {pair_name} meskipun fetch (dianggap) berhasil.{AnsiColors.ENDC}", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name)
                    # Lanjut ke pair berikutnya jika fetch gagal atau tidak ada data baru
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    continue 

                # Gabungkan candle baru dengan yang lama, hindari duplikasi dan urutkan
                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 # Untuk candle yang ada tapi kontennya berubah (misal close price update)

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Timestamp sama, tapi konten candle berubah
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1
                
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                elif new_candles_batch : 
                     log_info("Tidak ada candle dengan timestamp baru atau update konten. Data terakhir mungkin identik.", pair_name=pair_name)
                
                # Cek lagi apakah big data collection selesai setelah batch ini
                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!{AnsiColors.ENDC}", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                            data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] # Pangkas ke target
                        
                        if not data["big_data_email_sent"]:
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", config)
                            data["big_data_email_sent"] = True
                        
                        data["big_data_collection_phase_active"] = False 
                        active_cryptos_still_in_big_data_collection -=1 # Kurangi counter
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------{AnsiColors.ENDC}", pair_name=pair_name)
                else: # Jika sudah di fase live analysis, pastikan tidak melebihi target (jaga ukuran memori)
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                
                # Jalankan logika strategi jika ada data baru/update ATAU jika baru selesai big data
                min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    # Selalu proses jika baru saja selesai big data, atau jika ada candle baru/update
                    # atau jika masih dalam fase big data (untuk warm-up berkelanjutan)
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         (data["big_data_collection_phase_active"]))

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"])
                    elif not data["big_data_collection_phase_active"]: # Fase live, tapi tidak ada data baru
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: 
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
                
                # Update min_overall_next_refresh_seconds dengan interval pair ini jika sudah diproses
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)

            # --- Tentukan Durasi Sleep Global ---
            sleep_duration = 15 # Default jika tidak ada info lain
            
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: 
                # Jika TIDAK ADA data berhasil di-fetch SAMA SEKALI di siklus ini DAN semua API key global sudah habis
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600 # Tunggu lama jika semua key mati total
            elif active_cryptos_still_in_big_data_collection > 0:
                # Jika masih ada pair yang mengumpulkan big data, sleep lebih pendek
                # Ambil interval terpendek dari pair yang masih dalam big data collection
                min_big_data_interval = float('inf')
                for pid, pdata in crypto_data_manager.items():
                    if pdata["big_data_collection_phase_active"]:
                        pconfig = pdata["config"]
                        interval = 55 if pconfig['timeframe'] == "minute" else (3600 * 23.8 if pconfig['timeframe'] == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval)
                
                # Sleep sesuai interval terpendek big data, atau default 30s jika tidak ada (seharusnya ada)
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30)
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else: # Semua pair sudah dalam fase live analysis
                if min_overall_next_refresh_seconds != float('inf'):
                    # Sleep hingga waktu refresh terdekat berikutnya, minimal MIN_REFRESH_INTERVAL_AFTER_BIG_DATA
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya (min interval dari semua pair).", pair_name="SYSTEM")
                else: # Seharusnya tidak terjadi jika ada pair aktif
                    default_refresh = global_settings_dict.get("api_settings",{}).get("refresh_interval_seconds", 60) # Fallback
                    sleep_duration = default_refresh
                    log_debug(f"Default sleep {sleep_duration}s (fallback).", pair_name="SYSTEM")
            
            # Gunakan show_spinner untuk visualisasi sleep
            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            
    except KeyboardInterrupt: 
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error:", pair_name="SYSTEM")
    finally: 
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()

    while True:
        clear_screen_animated() # Ganti dengan versi animasi
        
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery) =========", color=AnsiColors.HEADER, delay=0.005)
        
        pick_title_main = "" # Judul untuk pick tetap plain
        active_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs)}) ---\n"
            for i, cfg in enumerate(active_configs): 
                pick_title_main += f"  {i+1}. {cfg['symbol']}-{cfg['currency']} (TF: {cfg['timeframe']}, Exch: {cfg['exchange']})\n"
        else: 
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"
        
        api_s = settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10: primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len(api_s.get('recovery_keys',[]))

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
            option_text, index = pick(options_plain, pick_title_main, indicator='=>', default_index=0)
            
            if index == 0: 
                start_trading(settings)
            elif index == 1: 
                settings = settings_menu(settings) 
            elif index == 2: 
                log_info("Aplikasi ditutup.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
                show_spinner(0.5, "Exiting")
                break
        except Exception as e: 
            log_warning(f"Operasi menu dibatalkan atau error: {e}", pair_name="SYSTEM")
            if isinstance(e, KeyboardInterrupt): 
                log_info("Aplikasi dihentikan oleh pengguna dari menu utama.", pair_name="SYSTEM")
                clear_screen_animated()
                animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE)
                break
            show_spinner(1, "Error menu, coba lagi...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        logger.exception("MAIN LEVEL EXCEPTION:")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
