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

# --- ANSI COLOR CODES --- (Sama seperti sebelumnya)
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

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass

# --- KONFIGURASI LOGGING --- (Sama, pastikan filter ada)
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


SETTINGS_FILE = "settings_multiple_recovery.json" # Nama file baru
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY":
            self.keys.append(primary_key)
        if recovery_keys_list:
            self.keys.extend([k for k in recovery_keys_list if k]) # Tambahkan recovery keys yang valid

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}
        
        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")
            # Tidak raise error di sini, biarkan fetch_candles gagal jika dipanggil tanpa key

    def get_current_key(self):
        if not self.keys:
            log_error("Tidak ada API key yang tersedia di APIKeyManager.")
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None # Semua key sudah habis

    def switch_to_next_key(self):
        if not self.keys: return None

        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # Kirim email notifikasi jika ada key baru dan email diaktifkan (secara global)
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
                email_body = (f"Skrip trading telah secara otomatis mengganti API key CryptoCompare.\n\n"
                              f"API Key sebelumnya mungkin telah mencapai limit atau tidak valid.\n"
                              f"Sekarang menggunakan API key dengan index: {self.current_index}\n"
                              f"Key: ...{new_key_display[-8:]} (bagian akhir ditampilkan untuk identifikasi)\n\n"
                              f"Harap periksa status API key Anda di CryptoCompare.")
                
                # Gunakan fungsi send_email_notification yang ada, tapi butuh 'crypto_settings' palsu
                # atau modifikasi send_email_notification untuk menerima settings global saja.
                # Untuk sementara, kita buat dummy config untuk email global
                dummy_email_cfg = {
                    "enable_email_notifications": True, # Force enable for this specific notification
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address")) # Prioritaskan admin
                }
                if all(dummy_email_cfg.values()): # Hanya kirim jika semua detail email global ada
                     send_email_notification(email_subject, email_body, dummy_email_cfg) # Kirim ke admin/penerima utama
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

# --- FUNGSI BEEP & EMAIL --- (Sama seperti sebelumnya)
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a', end='', flush=True)
            time.sleep(0.2)
            print('\a', end='', flush=True)
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email): # settings_for_email bisa crypto_settings atau dummy global
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('symbol', 'GLOBAL_EMAIL') # Konteks
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
def get_default_crypto_config(): # Sama
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
        "enable_global_email_notifications_for_key_switch": False, # Notifikasi jika key switch
        "email_sender_address": "pengirim.global@gmail.com", # Email global untuk notif admin
        "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com" # Email admin untuk notif key
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                settings = json.load(f)
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else: # Pastikan semua sub-key ada di api_settings
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v

                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = []
                for crypto_cfg in settings["cryptos"]:
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
                return settings
            except json.JSONDecodeError:
                log_error("Error membaca settings.json. Menggunakan default.")
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings): # Sama
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): # Sama
    new_config = current_config.copy()
    print(f"\n{AnsiColors.HEADER}--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---{AnsiColors.ENDC}")
    
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    new_config["symbol"] = (input(f"Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: ") or new_config.get('symbol','BTC')).upper()
    new_config["currency"] = (input(f"Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: ") or new_config.get('currency','USD')).upper()
    new_config["exchange"] = (input(f"Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: ") or new_config.get('exchange','CCCAGG'))
    
    tf_input = (input(f"Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: ") or new_config.get('timeframe','hour')).lower()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print("Timeframe tidak valid."); new_config["timeframe"] = new_config.get('timeframe','hour')
    
    refresh_input = input(f"Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: ")
    new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input or new_config.get('refresh_interval_seconds',60)))

    print(f"\n{AnsiColors.HEADER}-- Parameter Pivot --{AnsiColors.ENDC}")
    new_config["left_strength"] = int(input(f"Left Strength [{new_config.get('left_strength',50)}]: ") or new_config.get('left_strength',50))
    new_config["right_strength"] = int(input(f"Right Strength [{new_config.get('right_strength',150)}]: ") or new_config.get('right_strength',150))

    print(f"\n{AnsiColors.HEADER}-- Parameter Trading --{AnsiColors.ENDC}")
    new_config["profit_target_percent_activation"] = float(input(f"Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: ") or new_config.get('profit_target_percent_activation',5.0))
    new_config["trailing_stop_gap_percent"] = float(input(f"Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: ") or new_config.get('trailing_stop_gap_percent',5.0))
    new_config["emergency_sl_percent"] = float(input(f"Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: ") or new_config.get('emergency_sl_percent',10.0))
    
    print(f"\n{AnsiColors.HEADER}-- Fitur Secure FIB --{AnsiColors.ENDC}")
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))
    secure_fib_price_input = (input(f"Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: ") or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print("Pilihan harga Secure FIB tidak valid."); new_config["secure_fib_check_price"] = new_config.get('secure_fib_check_price','Close')

    print(f"\n{AnsiColors.HEADER}-- Notifikasi Email (Gmail) untuk Pair Ini --{AnsiColors.ENDC}")
    print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global dari API Settings (jika notif global aktif).{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = input(f"Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: ") or new_config.get('email_sender_address','')
    new_config["email_sender_app_password"] = input(f"App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: ") or new_config.get('email_sender_app_password','')
    new_config["email_receiver_address"] = input(f"Email Penerima [{new_config.get('email_receiver_address','')}]: ") or new_config.get('email_receiver_address','')
    
    return new_config

def settings_menu(current_settings):
    while True:
        api_s = current_settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10: primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        
        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len(recovery_keys)

        print(f"\n{AnsiColors.HEADER}--- Menu Pengaturan Utama ---{AnsiColors.ENDC}")
        print(f"Primary API Key: {AnsiColors.CYAN}{primary_key_display}{AnsiColors.ENDC}")
        print(f"Recovery API Keys: {AnsiColors.CYAN}{num_recovery_keys} tersimpan{AnsiColors.ENDC}")
        print("------------------------------------")
        print("Daftar Konfigurasi Crypto:")
        # ... (sama seperti sebelumnya untuk list crypto) ...
        if not current_settings["cryptos"]:
            print(f"  {AnsiColors.ORANGE}(Belum ada konfigurasi crypto){AnsiColors.ENDC}")
        for i, crypto_conf in enumerate(current_settings["cryptos"]):
            status = f"{AnsiColors.GREEN}Aktif{AnsiColors.ENDC}" if crypto_conf.get('enabled', True) else f"{AnsiColors.RED}Nonaktif{AnsiColors.ENDC}"
            print(f"  {i+1}. {AnsiColors.BOLD}{crypto_conf['symbol']}-{crypto_conf['currency']}{AnsiColors.ENDC} ({crypto_conf['timeframe']}) - {status}")
        print("------------------------------------")
        print(f"{AnsiColors.HEADER}--- Pengaturan API & Global ---{AnsiColors.ENDC}")
        print("1. Atur Primary API Key")
        print("2. Kelola Recovery API Keys")
        print("3. Atur Email Global untuk Notifikasi Sistem (API Key Switch, dll)")
        print(f"{AnsiColors.HEADER}--- Pengaturan Crypto Pair ---{AnsiColors.ENDC}")
        print("4. Tambah Konfigurasi Crypto Baru")
        print("5. Ubah Konfigurasi Crypto")
        print("6. Hapus Konfigurasi Crypto")
        print(f"{AnsiColors.HEADER}-----------------------------{AnsiColors.ENDC}")
        print("x. Kembali ke Menu Utama")
        choice = input("Pilihan Anda: ").lower()

        try:
            if choice == '1':
                api_s["primary_key"] = input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ") or api_s.get('primary_key','')
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
            elif choice == '2':
                while True:
                    print(f"\n{AnsiColors.HEADER}-- Kelola Recovery API Keys --{AnsiColors.ENDC}")
                    current_recovery = api_s.get('recovery_keys', [])
                    if not current_recovery:
                        print(f"  {AnsiColors.ORANGE}(Tidak ada recovery key tersimpan){AnsiColors.ENDC}")
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            print(f"  {i+1}. {r_key_display}")
                    print("\n  a. Tambah Recovery Key")
                    print("  b. Hapus Recovery Key")
                    print("  c. Kembali ke Pengaturan Utama")
                    sub_choice = input("Pilihan Recovery Key: ").lower()
                    if sub_choice == 'a':
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else:
                            print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                    elif sub_choice == 'b':
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
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
                    elif sub_choice == 'c':
                        break
                    else:
                        print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
            elif choice == '3':
                print(f"\n{AnsiColors.HEADER}-- Pengaturan Email Global Notifikasi Sistem --{AnsiColors.ENDC}")
                enable_g_email = input(f"Aktifkan notifikasi email global (API Key switch, dll)? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ") or api_s.get('email_sender_address','')
                api_s['email_sender_app_password'] = input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ") or api_s.get('email_sender_app_password','')
                api_s['email_receiver_address_admin'] = input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ") or api_s.get('email_receiver_address_admin','')
                current_settings["api_settings"] = api_s
                save_settings(current_settings)

            elif choice == '4': # Tambah Crypto
                new_crypto_conf = get_default_crypto_config()
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf)
                current_settings["cryptos"].append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
            elif choice == '5': # Ubah Crypto
                if not current_settings["cryptos"]: print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}"); continue
                idx_choice = int(input("Nomor konfigurasi crypto yang akan diubah: ")) - 1
                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
            elif choice == '6': # Hapus Crypto
                if not current_settings["cryptos"]: print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}"); continue
                idx_choice = int(input("Nomor konfigurasi crypto yang akan dihapus: ")) - 1
                if 0 <= idx_choice < len(current_settings["cryptos"]):
                    removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                    current_settings["cryptos"].pop(idx_choice)
                    save_settings(current_settings)
                    log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
            elif choice == 'x':
                break
            else:
                print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
        except ValueError:
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
        except Exception as e:
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA (MODIFIED) ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.") # Penting untuk memicu pergantian

    all_accumulated_candles = []
    current_to_ts = None
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10
    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)

    while len(all_accumulated_candles) < total_limit_desired:
        # ... (logika limit_for_this_api_call dan params sama seperti sebelumnya, tapi gunakan current_api_key_to_use)
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        if current_to_ts is not None and candles_still_needed > 1 :
             limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)
        if limit_for_this_api_call <= 0: break

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use # <--- Gunakan key yang dioper
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts
        
        try:
            if is_large_fetch: log_debug(f"Fetching batch (Key: ...{current_api_key_to_use[-5:]})", pair_name=pair_name)
            response = requests.get(url, params=params, timeout=20) # Timeout sedikit lebih lama
            
            # Cek DULU status code sebelum raise_for_status, untuk error API Key yang lebih spesifik
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = response.json() if response.content else {}
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() # Untuk error lain (5xx, 404, dll)
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                # Cek apakah error message mengindikasikan masalah API Key
                # Ini bisa disesuaikan dengan pesan error aktual dari CryptoCompare
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active" 
                ] # Tambahkan pesan error lain yang relevan
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{current_api_key_to_use[-5:]}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else: # Error API lain, bukan soal key
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break 
            
            # ... (sisa logika parsing data sama seperti sebelumnya) ...
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break 
            raw_candles_from_api = data['Data']['Data']
            if not raw_candles_from_api: 
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break
            batch_candles_list = []
            for item in raw_candles_from_api:
                candle = {'timestamp': datetime.fromtimestamp(item['time']), 'open': item.get('open'), 'high': item.get('high'), 'low': item.get('low'), 'close': item.get('close'), 'volume': item.get('volumefrom') }
                batch_candles_list.append(candle)
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop() 
            if not batch_candles_list and current_to_ts is not None :
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal.", pair_name=pair_name)
                break
            all_accumulated_candles = batch_candles_list + all_accumulated_candles 
            if raw_candles_from_api: current_to_ts = raw_candles_from_api[0]['time'] 
            else: break
            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit. Akhir histori tercapai.", pair_name=pair_name)
                break 
            if len(all_accumulated_candles) >= total_limit_desired: break 
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3)

        except APIKeyError: # Re-raise APIKeyError agar ditangkap oleh start_trading
            raise
        except requests.exceptions.RequestException as e: # Error koneksi, timeout, dll. (BUKAN API Key)
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            # Tidak raise APIKeyError, ini masalah jaringan atau server API, bukan key-nya.
            break 
        except Exception as e:
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error:", pair_name=pair_name)
            break 

    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch:
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)
    return all_accumulated_candles


# --- LOGIKA STRATEGI --- (get_initial_strategy_state, find_pivots, run_strategy_logic sama)
def get_initial_strategy_state(): # Sama
    return {
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None,
        "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False, "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None, "position_size": 0,
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True): # Sama
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1: return pivots
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None or series_list[i] is None: is_pivot = False; break
            if is_high:
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: 
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None or series_list[i] is None: is_pivot = False; break
            if is_high:
                if series_list[i] < series_list[i+j]: is_pivot = False; break 
            else: 
                if series_list[i] > series_list[i+j]: is_pivot = False; break 
        if is_pivot: pivots[i] = series_list[i] 
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state): # Sama
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None
    left_strength, right_strength = crypto_config['left_strength'], crypto_config['right_strength']
    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong/kurang kunci di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state
    high_prices, low_prices = [c['high'] for c in candles_history], [c['low'] for c in candles_history]
    raw_pivot_highs, raw_pivot_lows = find_pivots(high_prices, left_strength, right_strength, True), find_pivots(low_prices,  left_strength, right_strength, False)
    current_bar_index_in_list = len(candles_history) - 1 
    if current_bar_index_in_list < 0 : return strategy_state
    idx_pivot_event_high, idx_pivot_event_low = current_bar_index_in_list - right_strength, current_bar_index_in_list - right_strength
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
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp')}. Skip evaluasi.", pair_name=pair_name)
        return strategy_state

    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"], strategy_state["high_bar_index_for_fib"] = strategy_state["final_pivot_high_price_confirmed"], idx_pivot_event_high
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"], strategy_state["active_fib_line_start_index"] = None, None
    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price, current_low_bar_index = strategy_state["final_pivot_low_price_confirmed"], idx_pivot_event_low
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0
                is_fib_late = False
                if crypto_config["enable_secure_fib"]:
                    price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                    if price_val_current_candle is not None and price_val_current_candle > calculated_fib_level: is_fib_late = True
                if is_fib_late:
                    log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                    strategy_state["active_fib_level"], strategy_state["active_fib_line_start_index"] = None, None
                else:
                    log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=pair_name)
                    strategy_state["active_fib_level"], strategy_state["active_fib_line_start_index"] = calculated_fib_level, current_low_bar_index
                strategy_state["high_price_for_fib"], strategy_state["high_bar_index_for_fib"] = None, None
    
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None:
        is_bullish_candle, is_closed_above_fib = current_candle['close'] > current_candle['open'], current_candle['close'] > strategy_state["active_fib_level"]
        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: 
                strategy_state["position_size"], entry_px = 1, current_candle['close']
                strategy_state["entry_price_custom"], strategy_state["highest_price_for_trailing"] = entry_px, entry_px
                strategy_state["trailing_tp_active_custom"], strategy_state["current_trailing_stop_level"] = False, None
                emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl
                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()
                email_subject, email_body = f"BUY Signal: {pair_name}", (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\nEntry: {entry_px:.5f}\nFIB: {strategy_state['active_fib_level']:.5f}\nEmergSL: {emerg_sl:.5f}\nTime: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, crypto_config)
            strategy_state["active_fib_level"], strategy_state["active_fib_line_start_index"] = None, None

    if strategy_state["position_size"] > 0:
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle['high'])
        if current_high_for_trailing is None: current_high_for_trailing = current_candle['high'] 
        strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0 if strategy_state["entry_price_custom"] != 0 else 0
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state['highest_price_for_trailing']:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
        if strategy_state["trailing_tp_active_custom"] and strategy_state["highest_price_for_trailing"] is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)
        final_stop_for_exit, exit_comment, exit_color = strategy_state["emergency_sl_level_custom"], "Emergency SL", AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit, exit_comment, exit_color = strategy_state["current_trailing_stop_level"], "Trailing Stop", AnsiColors.BLUE 
        if final_stop_for_exit is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price = min(current_candle['open'], final_stop_for_exit) 
            pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"] * 100.0) if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0 else 0.0
            if exit_comment == "Trailing Stop" and pnl < 0: exit_color = AnsiColors.RED
            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()
            email_subject, email_body = f"Trade Closed: {pair_name} ({exit_comment})", (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\nExit: {exit_price:.5f}\nReason: {exit_comment}\nEntry: {strategy_state.get('entry_price_custom', 0):.5f}\nPnL: {pnl:.2f}%\nTime: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, crypto_config)
            strategy_state["position_size"], strategy_state["entry_price_custom"], strategy_state["highest_price_for_trailing"], strategy_state["trailing_tp_active_custom"], strategy_state["current_trailing_stop_level"], strategy_state["emergency_sl_level_custom"] = 0, None, None, False, None, None
    
    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state.get("current_trailing_stop_level") > plot_stop_level: plot_stop_level = strategy_state.get("current_trailing_stop_level")
        entry_price_display, sl_display_str = strategy_state.get('entry_price_custom', 0), f'{plot_stop_level:.5f}' if plot_stop_level is not None else 'N/A'
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=pair_name)
    return strategy_state


# --- FUNGSI UTAMA TRADING LOOP (MODIFIED) ---
def start_trading(global_settings_dict):
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Kirim semua api_settings untuk email global
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        return

    log_info(f"{AnsiColors.HEADER}================ MULTI-CRYPTO STRATEGY START (Key Recovery Enabled) ================{AnsiColors.ENDC}")
    current_key_display = api_key_manager.get_current_key()
    if current_key_display and len(current_key_display) > 8: current_key_display = current_key_display[:5] + "..." + current_key_display[-3:]
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config['symbol']}-{config['currency']}_{config['timeframe']}"
        config['pair_name'] = f"{config['symbol']}-{config['currency']}"
        # ... (log info konfigurasi pair sama seperti sebelumnya) ...
        log_info(f"Menginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config['exchange']} | TF: {config['timeframe']}", pair_name=config['pair_name'])

        crypto_data_manager[pair_id] = {
            "config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0
        }

        # Initial BIG DATA Fetch with retry logic for API keys
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() # Coba semua key jika perlu
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: # Semua key habis
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break 
            
            try:
                log_info(f"BIG DATA: Mengambil data awal ({CRYPTOCOMPARE_MAX_LIMIT} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], CRYPTOCOMPARE_MAX_LIMIT, 
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Jika tidak ada exception, sukses
            except APIKeyError:
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): # Jika switch_to_next_key mengembalikan None (semua habis)
                    break # Keluar dari loop retry jika tidak ada key lagi
                retries_done_initial += 1
            except requests.exceptions.RequestException as e: # Error jaringan, bukan key
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                # Mungkin perlu break atau skip pair ini jika error jaringan persisten
                break # Untuk saat ini, break jika ada error jaringan saat initial fetch

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            continue
        
        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])
        # ... (Warm-up logic sama seperti sebelumnya) ...
        if initial_candles:
            log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
            min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
            temp_strategy_state_before_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
            for i in range(min_len_for_pivots -1, len(initial_candles) - 1): 
                historical_slice = initial_candles[:i+1]
                if len(historical_slice) < min_len_for_pivots: continue
                temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                temp_state_for_warmup["position_size"] = 0 
                crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_for_warmup)
                if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0:
                    crypto_data_manager[pair_id]["strategy_state"] = {**crypto_data_manager[pair_id]["strategy_state"], **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, "highest_price_for_trailing":None, "trailing_tp_active_custom":False, "current_trailing_stop_level":None}}
            log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
        else: log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])
        log_info(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", pair_name=config['pair_name'])

    # Main trading loop
    try:
        while True:
            active_cryptos_in_big_data = 0
            min_next_refresh_interval = float('inf')
            any_data_fetched_this_cycle = False

            for pair_id, data in crypto_data_manager.items():
                if data.get("data_fetch_failed_consecutively", 0) >= api_key_manager.total_keys() + 1 and api_key_manager.total_keys() > 0 : # Jika semua key dicoba dan masih gagal
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # Coba lagi setelah 1 jam
                        # log_debug(f"Skipping {data['config']['pair_name']}, semua key gagal, menunggu 1 jam.", pair_name=data['config']['pair_name'])
                        continue # Skip pair ini untuk sementara
                    else:
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter untuk coba lagi

                config = data["config"]
                pair_name = config['pair_name']
                current_loop_time = datetime.now()
                time_since_last_fetch = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()
                required_interval = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_in_big_data += 1
                    if config['timeframe'] == "minute": required_interval = 55 
                    elif config['timeframe'] == "day": required_interval = 3600 * 23.8
                    else: required_interval = 3580 
                    min_next_refresh_interval = min(min_next_refresh_interval, required_interval)
                else:
                    required_interval = config['refresh_interval_seconds']
                    min_next_refresh_interval = min(min_next_refresh_interval, required_interval)

                if time_since_last_fetch < required_interval: continue

                log_info(f"Memproses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time
                num_candles_before_fetch = len(data["all_candles_list"])
                
                if data["big_data_collection_phase_active"]: log_info(f"\n{AnsiColors.BOLD}--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---{AnsiColors.ENDC}", pair_name=pair_name)
                else: log_info(f"\n{AnsiColors.BOLD}--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---{AnsiColors.ENDC}", pair_name=pair_name)

                # Fetch new candles with retry for API keys
                new_candles_batch = []
                fetch_update_successful = False
                # Retries for update fetch are implicitly handled by the global API key manager cycling.
                # We attempt with the current key from the manager. If it fails, the manager cycles,
                # and on the *next main loop's processing of this pair*, it will try the new key.
                # OR, we implement a small retry loop here too for immediate effect:
                
                max_retries_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_update = 0
                
                original_key_index_before_fetch = api_key_manager.get_current_key_index()

                while retries_done_update < max_retries_update and not fetch_update_successful:
                    current_api_key = api_key_manager.get_current_key()
                    if not current_api_key:
                        log_error(f"Semua API key habis saat mencoba mengambil update untuk {pair_name}.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = (data.get("data_fetch_failed_consecutively", 0) +1) if api_key_manager.get_current_key_index() == original_key_index_before_fetch else 1 # Hanya increment jika key belum ganti global
                        if data["data_fetch_failed_consecutively"] >= api_key_manager.total_keys() +1: data["last_attempt_after_all_keys_failed"] = datetime.now()
                        break # Keluar dari retry loop untuk pair ini

                    log_info(f"Mengambil 3 candle terbaru (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], 3, 
                            config['exchange'], current_api_key, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful = True
                        data["data_fetch_failed_consecutively"] = 0 # Reset counter on success
                        any_data_fetched_this_cycle = True
                    except APIKeyError:
                        log_warning(f"API Key gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = (data.get("data_fetch_failed_consecutively", 0) +1) if api_key_manager.get_current_key_index() == original_key_index_before_fetch else 1
                        if not api_key_manager.switch_to_next_key(): break # Semua key habis
                        # Update original_key_index_before_fetch as manager has switched globally
                        original_key_index_before_fetch = api_key_manager.get_current_key_index() 
                        retries_done_update += 1
                    except requests.exceptions.RequestException as e:
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] += 1 # Anggap sebagai kegagalan fetch
                        if data["data_fetch_failed_consecutively"] >= api_key_manager.total_keys() +1: data["last_attempt_after_all_keys_failed"] = datetime.now()
                        break # Keluar dari retry loop, coba lagi di siklus berikutnya
                
                if not fetch_update_successful or not new_candles_batch:
                    if fetch_update_successful and not new_candles_batch: # Sukses fetch tapi data kosong
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima meskipun fetch berhasil.{AnsiColors.ENDC}", pair_name=pair_name)
                    # Jika tidak, error sudah di log sebelumnya (gagal fetch setelah retry)
                    continue # Lanjut ke crypto berikutnya dalam loop utama

                # ... (Merging candles and strategy logic sama seperti sebelumnya) ...
                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                for candle in new_candles_batch: merged_candles_dict[candle['timestamp']] = candle 
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                num_candles_after_fetch, num_newly_added_or_updated = len(all_candles_list_temp), len(all_candles_list_temp) - num_candles_before_fetch
                if data["all_candles_list"] and all_candles_list_temp and data["all_candles_list"][-1]['timestamp'] == all_candles_list_temp[-1]['timestamp'] and data["all_candles_list"][-1] != all_candles_list_temp[-1]:
                     log_info(f"Candle terakhir @{all_candles_list_temp[-1]['timestamp'].strftime('%H:%M')} diupdate.", pair_name=pair_name)
                     if num_newly_added_or_updated <=0 : num_newly_added_or_updated = 1 
                data["all_candles_list"] = all_candles_list_temp
                if num_newly_added_or_updated > 0 : log_info(f"{num_newly_added_or_updated if num_candles_after_fetch > num_candles_before_fetch else 'Beberapa'} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                else: log_info("Tidak ada candle dengan timestamp baru. Candle terakhir mungkin diupdate.", pair_name=pair_name)

                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI!{AnsiColors.ENDC}", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        if not data["big_data_email_sent"]:
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", config)
                            data["big_data_email_sent"] = True
                        data["big_data_collection_phase_active"] = False 
                        active_cryptos_in_big_data -=1
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({TARGET_BIG_DATA_CANDLES} candles) ----------{AnsiColors.ENDC}", pair_name=pair_name)
                else: 
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                
                min_len_for_pivots = config['left_strength'] + config['right_strength'] + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    if num_newly_added_or_updated > 0 or data["big_data_collection_phase_active"]: 
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"])
                    elif not data["big_data_collection_phase_active"]: 
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: log_info(f"Data ({len(data['all_candles_list'])}) belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
            
            # Sleep logic (sama seperti sebelumnya, tapi dengan pengecekan jika ada data yang berhasil di-fetch)
            sleep_duration = 15 
            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: # Semua key habis dan tidak ada data fetch
                log_error("Semua API key gagal dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
                sleep_duration = 3600
            elif active_cryptos_in_big_data > 0:
                sleep_duration = min(30, min_next_refresh_interval if min_next_refresh_interval != float('inf') else 30) 
                log_debug(f"Masih ada {active_cryptos_in_big_data} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            else:
                sleep_duration = min_next_refresh_interval if min_next_refresh_interval != float('inf') else global_settings_dict.get("api_settings",{}).get("refresh_interval_seconds", 60) # Fallback ke setting global jika ada
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, sleep_duration)
                log_debug(f"Semua pair live. Interval refresh terpendek: {min_next_refresh_interval if min_next_refresh_interval != float('inf') else 'N/A'}s. Sleep {sleep_duration}s.", pair_name="SYSTEM")
            
            if not any(data["big_data_collection_phase_active"] for data in crypto_data_manager.values()) and \
               all( (datetime.now() - data["last_candle_fetch_time"]).total_seconds() < data["config"]["refresh_interval_seconds"] for data in crypto_data_manager.values() if data["last_candle_fetch_time"] != datetime.min):
                time_to_next_potential_refresh = float('inf')
                for pair_id, data in crypto_data_manager.items():
                    if not data["big_data_collection_phase_active"] and data["last_candle_fetch_time"] != datetime.min:
                        time_to_next_potential_refresh = min(time_to_next_potential_refresh, data["config"]["refresh_interval_seconds"] - (datetime.now() - data["last_candle_fetch_time"]).total_seconds())
                sleep_duration = max(1, int(time_to_next_potential_refresh)) if time_to_next_potential_refresh != float('inf') else sleep_duration
                log_debug(f"Semua live & data baru. Tidur sampai refresh berikutnya ~{sleep_duration}s", pair_name="SYSTEM")

            time.sleep(sleep_duration)
            
    except KeyboardInterrupt: log_info(f"\n{AnsiColors.ORANGE}Proses trading dihentikan.{AnsiColors.ENDC}", pair_name="SYSTEM")
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error:", pair_name="SYSTEM")
    finally: log_info(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", pair_name="SYSTEM")


# --- MENU UTAMA ---
def main_menu(): # Sama
    settings = load_settings()
    global all_candles_list 
    all_candles_list = [] 

    while True:
        print(f"\n{AnsiColors.HEADER}========= Crypto Strategy Runner (Multi + Key Recovery) ========={AnsiColors.ENDC}")
        active_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs:
            print(f"{AnsiColors.CYAN}--- Crypto Aktif ({len(active_configs)}) ---{AnsiColors.ENDC}")
            for i, cfg in enumerate(active_configs): print(f"  {i+1}. {AnsiColors.BOLD}{cfg['symbol']}-{cfg['currency']}{AnsiColors.ENDC} (TF: {cfg['timeframe']}, Exch: {cfg['exchange']})")
        else: print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif.{AnsiColors.ENDC}")
        
        api_s = settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if len(primary_key_display) > 10: primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len(api_s.get('recovery_keys',[]))

        print("-----------------------------------------------")
        print(f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle")
        print(f"Primary API Key: {AnsiColors.CYAN}{primary_key_display}{AnsiColors.ENDC} | Recovery Keys: {AnsiColors.CYAN}{num_recovery_keys}{AnsiColors.ENDC}")
        print("-----------------------------------------------")
        print(f"1. {AnsiColors.GREEN}Mulai Analisa Realtime Semua Pair Aktif{AnsiColors.ENDC}")
        print(f"2. {AnsiColors.ORANGE}Pengaturan{AnsiColors.ENDC}")
        print(f"3. {AnsiColors.RED}Keluar{AnsiColors.ENDC}")
        choice = input("Pilihan Anda: ")

        if choice == '1': start_trading(settings)
        elif choice == '2': settings = settings_menu(settings)
        elif choice == '3': log_info("Aplikasi ditutup.", pair_name="SYSTEM"); break
        else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")

if __name__ == "__main__":
    # Filter sudah ditambahkan di awal
    all_candles_list = [] 
    main_menu()
