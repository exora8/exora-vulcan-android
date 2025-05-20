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

# CHART_INTEGRATION_START
import threading
import copy 
try:
    from flask import Flask, jsonify, render_template_string, request # Tambah request
except ImportError:
    print("Flask tidak terinstal. Silakan install dengan: pip install Flask")
    sys.exit(1)
# CHART_INTEGRATION_END

# --- ANSI COLOR CODES ---
class AnsiColors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'; ORANGE = '\033[93m'
    RED = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
    CYAN = '\033[96m'; MAGENTA = '\033[35m'; YELLOW_BG = '\033[43m'

# --- ANIMATION HELPER FUNCTIONS ---
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True):
    for char in text:
        sys.stdout.write(color + char + AnsiColors.ENDC if color else char)
        sys.stdout.flush(); time.sleep(delay)
    if new_line: print()

def show_spinner(duration_seconds, message="Processing..."):
    spinner_chars = ['-', '\\', '|', '/']; start_time = time.time(); idx = 0
    sys.stdout.write(AnsiColors.MAGENTA)
    term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try: term_width = os.get_terminal_size().columns
        except OSError: pass
    while (time.time() - start_time) < duration_seconds:
        display_message = message[:term_width - 5]
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush(); time.sleep(0.1); idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r")
    sys.stdout.write(AnsiColors.ENDC); sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length); term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try: term_width = os.get_terminal_size().columns
        except OSError: pass
    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    sys.stdout.write(progress_line[:term_width]); sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n'); sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception): pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger(); logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
log_file_name = "trading_log.txt"
fh = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pair_name)s - %(message)s')
fh.setFormatter(file_formatter); logger.addHandler(fh)
ch = logging.StreamHandler()
console_formatter_template = '%(asctime)s - {bold}%(levelname)s{endc} - {cyan}[%(pair_name)s]{endc} - %(message)s'
console_formatter = logging.Formatter(console_formatter_template.format(bold=AnsiColors.BOLD, endc=AnsiColors.ENDC, cyan=AnsiColors.CYAN))
ch.setFormatter(console_formatter); logger.addHandler(ch)
class AddPairNameFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'pair_name'): record.pair_name = 'SYSTEM'
        return True
logger.addFilter(AddPairNameFilter())
def log_info(message, pair_name="SYSTEM"): logger.info(message, extra={'pair_name': pair_name})
def log_warning(message, pair_name="SYSTEM"): logger.warning(message, extra={'pair_name': pair_name})
def log_error(message, pair_name="SYSTEM"): logger.error(message, extra={'pair_name': pair_name})
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name})
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})

SETTINGS_FILE = "settings_multiple_recovery.json"; CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500; MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15
MAX_CANDLES_FOR_CHART_DISPLAY = 400 # OPTIMIZATION: Max candles to send to chart

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY": self.keys.append(primary_key)
        if recovery_keys_list: self.keys.extend([k for k in recovery_keys_list if k])
        self.current_index = 0; self.global_email_settings = global_settings_for_email or {}
        if not self.keys: log_warning("Tidak ada API key yang valid dikonfigurasi.")
    def get_current_key(self):
        if not self.keys: return None
        return self.keys[self.current_index] if self.current_index < len(self.keys) else None
    def switch_to_next_key(self):
        if not self.keys: return None; self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_disp = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_disp}){AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
                email_body = (f"Skrip trading otomatis mengganti API key CryptoCompare.\n\nKey sebelumnya mungkin limit/tidak valid.\n"
                              f"Sekarang menggunakan API key index: {self.current_index}\nKey: ...{new_key_disp[-8:] if len(new_key_disp) > 8 else new_key_disp}\n\nPeriksa status API key Anda.")
                dummy_email_cfg = {"enable_email_notifications": True, "email_sender_address": self.global_email_settings.get("email_sender_address"), "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"), "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))}
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]): send_email_notification(email_subject, email_body, dummy_email_cfg)
                else: log_warning("Konfigurasi email global tidak lengkap untuk notif ganti API key (APIKeyManager).")
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY HABIS/GAGAL! Tidak dapat ambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "KRITIS: SEMUA API Key CryptoCompare Gagal!"
                email_body = (f"Skrip trading telah mencoba semua API key CryptoCompare dan semuanya gagal.\n\nSkrip tidak dapat lagi mengambil data harga.\nSegera periksa akun CryptoCompare dan konfigurasi API key.")
                dummy_email_cfg = {"enable_email_notifications": True, "email_sender_address": self.global_email_settings.get("email_sender_address"), "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"), "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))}
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]): send_email_notification(email_subject, email_body, dummy_email_cfg)
                else: log_warning("Konfigurasi email global tidak lengkap untuk notif KRITIS semua API key gagal (APIKeyManager).")
            return None
    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION ---
def play_notification_sound():
    try:
        if sys.platform == "win32": import winsound; winsound.Beep(1000, 500)
        else: print('\a', end='', flush=True)
    except Exception as e: log_warning(f"Tidak bisa mainkan suara notifikasi: {e}")
def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False): return
    sender_email = settings_for_email.get("email_sender_address"); sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address"); pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
    if not all([sender_email, sender_password, receiver_email]):
        log_warning(f"Konfigurasi email tidak lengkap. Notif email dilewati.", pair_name=pair_name_ctx); return
    msg = MIMEText(body_text); msg['Subject'] = subject; msg['From'] = sender_email; msg['To'] = receiver_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        log_info(f"{AnsiColors.CYAN}Notif email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal kirim email notif: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    api_settings = global_settings.get("api_settings", {})
    if not api_settings.get("enable_termux_notifications", False): return
    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        log_info(f"{AnsiColors.CYAN}Notif Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError: log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan...", pair_name=pair_name_for_log)
    except subprocess.TimeoutExpired: log_warning(f"{AnsiColors.ORANGE}Timeout kirim notif Termux '{title}'.{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal kirim notif Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {"id": str(uuid.uuid4()), "enabled": True, "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG", "timeframe": "hour", "refresh_interval_seconds": 60, "left_strength": 50, "right_strength": 150, "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0, "emergency_sl_percent": 10.0, "enable_secure_fib": True, "secure_fib_check_price": "Close", "enable_email_notifications": False, "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""}
def load_settings():
    default_api_settings = {"primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [], "enable_global_email_notifications_for_key_switch": False, "email_sender_address": "pengirim.global@gmail.com", "email_sender_app_password": "xxxx xxxx xxxx xxxx", "email_receiver_address_admin": "admin.penerima@example.com", "enable_termux_notifications": False}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
            if "api_settings" not in settings: settings["api_settings"] = default_api_settings.copy()
            else:
                for k, v in default_api_settings.items():
                    if k not in settings["api_settings"]: settings["api_settings"][k] = v
            if "cryptos" not in settings or not isinstance(settings["cryptos"], list): settings["cryptos"] = []
            for crypto_cfg in settings["cryptos"]:
                if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True
            return settings
        except json.JSONDecodeError: log_error(f"Error baca {SETTINGS_FILE}. Pakai default."); return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
        except Exception as e: log_error(f"Error load_settings: {e}. Pakai default."); return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
        log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal simpan pengaturan ke {SETTINGS_FILE}: {e}{AnsiColors.ENDC}")
def _prompt_crypto_config(current_config):
    clear_screen_animated(); new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)
    enabled_input = input(f"Aktifkan? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Mata Uang Quote (misal USDT) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance/CCCAGG) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Pakai default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}");
    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try: new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60))
    except ValueError: print(f"{AnsiColors.RED}Input interval refresh tidak valid. Pakai default."); new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))
    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))
    except ValueError: print(f"{AnsiColors.RED}Input strength tidak valid. Pakai default."); new_config["left_strength"] = new_config.get('left_strength',50); new_config["right_strength"] = new_config.get('right_strength',150)
    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))
    except ValueError: print(f"{AnsiColors.RED}Input param trading tidak valid. Pakai default."); new_config["profit_target_percent_activation"]=new_config.get('profit_target_percent_activation',5.0); new_config["trailing_stop_gap_percent"]=new_config.get('trailing_stop_gap_percent',5.0); new_config["emergency_sl_percent"]=new_config.get('emergency_sl_percent',10.0)
    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower().strip()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))
    secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid. Pakai default.");
    animated_text_display("\n-- Notifikasi Email (Gmail) Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    print(f"{AnsiColors.ORANGE}Kosongkan jika pakai email global (jika aktif).{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notif Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))
    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()
    return new_config
def settings_menu(current_settings):
    while True:
        clear_screen_animated(); api_s = current_settings.get("api_settings", {})
        primary_key_disp = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_disp and len(primary_key_disp) > 10 and primary_key_disp not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]: primary_key_disp = primary_key_disp[:5] + "..." + primary_key_disp[-3:]
        num_recovery_keys = len([k for k in api_s.get('recovery_keys', []) if k]); termux_notif_s = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        pick_title_s = f"--- Menu Pengaturan Utama ---\nPrimary API Key: {primary_key_disp}\nRecovery API Keys: {num_recovery_keys} tersimpan\nNotifikasi Termux: {termux_notif_s}\n------------------------------------\nDaftar Konfigurasi Crypto:\n"
        if not current_settings.get("cryptos"): pick_title_s += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_s += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_s += "------------------------------------\nPilih tindakan:"
        selectable_opts = ["Atur Primary API Key", "Kelola Recovery API Keys", "Atur Email Global Notif Sistem", "Aktifkan/Nonaktifkan Notif Termux", "Tambah Konfigurasi Crypto", "Ubah Konfigurasi Crypto", "Hapus Konfigurasi Crypto", "Kembali ke Menu Utama"]
        selected_opt_text = None; action_choice_val = -1
        try: selected_opt_text, action_choice_val = pick(selectable_opts, pick_title_s, indicator='=>', default_index=0)
        except Exception as e_pick_s:
            log_error(f"Error 'pick': {e_pick_s}. Input manual."); print(pick_title_s)
            for idx, opt_text in enumerate(selectable_opts): print(f"  {idx + 1}. {opt_text}")
            try:
                choice_input_s = input("Pilih nomor opsi: ").strip()
                if not choice_input_s: continue
                choice_s = int(choice_input_s) -1
                if 0 <= choice_s < len(selectable_opts): action_choice_val = choice_s; selected_opt_text = selectable_opts[choice_s] 
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
            except ValueError: print(f"{AnsiColors.RED}Input harus angka.{AnsiColors.ENDC}"); show_spinner(1.5, "Kembali..."); continue
        if selected_opt_text is not None and action_choice_val < 0: pass 
        elif action_choice_val < 0 and selected_opt_text is None: continue
        try:
            clear_screen_animated()
            if action_choice_val == 0: 
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Primary API Key CryptoCompare [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(1, "Menyimpan...")
            elif action_choice_val == 1: 
                while True:
                    clear_screen_animated(); recovery_pick_title_rec = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery_rec = [k for k in api_s.get('recovery_keys', []) if k]; api_s['recovery_keys'] = current_recovery_rec
                    if not current_recovery_rec: recovery_pick_title_rec += "  (Tidak ada recovery key)\n"
                    else:
                        for i, r_key in enumerate(current_recovery_rec): recovery_pick_title_rec += f"  {i+1}. {r_key[:5]}...{r_key[-3:] if len(r_key)>8 else r_key}\n"
                    recovery_pick_title_rec += "\nPilih tindakan:"; recovery_opts_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali"]
                    rec_selected_text_rec = None; rec_index_rec = -1
                    try: rec_selected_text_rec, rec_index_rec = pick(recovery_opts_plain, recovery_pick_title_rec, indicator='=>', default_index=0)
                    except Exception as e_pick_rec_menu:
                        log_error(f"Error 'pick' menu recovery: {e_pick_rec_menu}. Input manual."); print(recovery_pick_title_rec)
                        for idx_rec, opt_text_rec in enumerate(recovery_opts_plain): print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice_input_rec = input("Pilih nomor opsi: ").strip()
                            if not rec_choice_input_rec: continue
                            rec_choice_val_rec = int(rec_choice_input_rec) -1
                            if 0 <= rec_choice_val_rec < len(recovery_opts_plain): rec_index_rec = rec_choice_val_rec
                            else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                        except ValueError: print(f"{AnsiColors.RED}Input harus angka.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                    clear_screen_animated()
                    if rec_index_rec == 0: 
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key_rec = input("Recovery API Key baru: ").strip()
                        if new_r_key_rec: current_recovery_rec.append(new_r_key_rec); api_s['recovery_keys'] = current_recovery_rec; save_settings(current_settings); print(f"{AnsiColors.GREEN}Recovery key ditambah.{AnsiColors.ENDC}")
                        else: print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index_rec == 1: 
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery_rec: print(f"{AnsiColors.ORANGE}Tidak ada recovery key.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                        try:
                            del_opts_rec = [f"{r_key_del[:5]}...{r_key_del[-3:]}" if len(r_key_del)>8 else r_key_del for r_key_del in current_recovery_rec]; del_opts_rec.append("Batal")
                            _del_text_rec, idx_del_pick_rec = pick(del_opts_rec, "Pilih recovery key yang akan dihapus:", indicator='=>')
                            if idx_del_pick_rec == len(del_opts_rec) -1 : show_spinner(0.5, "Dibatalkan..."); continue
                            if 0 <= idx_del_pick_rec < len(current_recovery_rec):
                                removed_rec = current_recovery_rec.pop(idx_del_pick_rec); api_s['recovery_keys'] = current_recovery_rec; save_settings(current_settings); print(f"{AnsiColors.GREEN}Recovery key '{removed_rec[:5]}...' dihapus.{AnsiColors.ENDC}")
                            else: print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}")
                        except Exception as e_pick_del_rec: log_debug(f"Pick hapus recovery key dibatalkan/error: {e_pick_del_rec}"); print(f"{AnsiColors.ORANGE}Penghapusan dibatalkan.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index_rec == 2: break
            elif action_choice_val == 2: 
                animated_text_display("-- Email Global Notif Sistem --", color=AnsiColors.HEADER)
                enable_g_email_s = input(f"Aktifkan notif email global? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email_s == 'true' else (False if enable_g_email_s == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Notif Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(1, "Menyimpan...")
            elif action_choice_val == 3: 
                animated_text_display("-- Notif Termux Realtime --", color=AnsiColors.HEADER)
                current_status_s = api_s.get('enable_termux_notifications', False)
                new_status_input_s = input(f"Aktifkan Notif Termux? (true/false) [{current_status_s}]: ").lower().strip()
                if new_status_input_s == 'true': api_s['enable_termux_notifications'] = True; print(f"{AnsiColors.GREEN}Notif Termux diaktifkan.{AnsiColors.ENDC}\n{AnsiColors.ORANGE}Pastikan Termux:API terinstal & `pkg install termux-api` dijalankan.{AnsiColors.ENDC}")
                elif new_status_input_s == 'false': api_s['enable_termux_notifications'] = False; print(f"{AnsiColors.GREEN}Notif Termux dinonaktifkan.{AnsiColors.ENDC}")
                else: print(f"{AnsiColors.ORANGE}Input tidak valid. Status tidak berubah: {current_status_s}.{AnsiColors.ENDC}")
                current_settings["api_settings"] = api_s; save_settings(current_settings); show_spinner(2, "Menyimpan...")
            elif action_choice_val == 4: 
                new_crypto_conf_s = get_default_crypto_config(); new_crypto_conf_s = _prompt_crypto_config(new_crypto_conf_s)
                current_settings.setdefault("cryptos", []).append(new_crypto_conf_s); save_settings(current_settings)
                log_info(f"Konfigurasi {new_crypto_conf_s['symbol']}-{new_crypto_conf_s['currency']} ditambah."); show_spinner(1, "Menyimpan...")
            elif action_choice_val == 5: 
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                edit_opts_s = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]; edit_opts_s.append("Batal")
                _edit_text_s, idx_choice_pick_s = pick(edit_opts_s, "Pilih konfigurasi crypto yang akan diubah:", indicator='=>')
                if idx_choice_pick_s == len(edit_opts_s) -1 : show_spinner(0.5, "Dibatalkan..."); continue
                if 0 <= idx_choice_pick_s < len(current_settings["cryptos"]):
                    current_settings["cryptos"][idx_choice_pick_s] = _prompt_crypto_config(current_settings["cryptos"][idx_choice_pick_s]); save_settings(current_settings)
                    log_info(f"Konfigurasi {current_settings['cryptos'][idx_choice_pick_s]['symbol']}-{current_settings['cryptos'][idx_choice_pick_s]['currency']} diubah.")
                else: print(f"{AnsiColors.RED}Pilihan ubah tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice_val == 6: 
                if not current_settings.get("cryptos"): print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}"); show_spinner(1, "Kembali..."); continue
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                del_crypto_opts_s = [f"{cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')}" for cfg in current_settings["cryptos"]]; del_crypto_opts_s.append("Batal")
                _del_c_text_s, idx_del_c_pick_s = pick(del_crypto_opts_s, "Pilih konfigurasi crypto yang akan dihapus:", indicator='=>')
                if idx_del_c_pick_s == len(del_crypto_opts_s) - 1: show_spinner(0.5, "Dibatalkan..."); continue
                if 0 <= idx_del_c_pick_s < len(current_settings["cryptos"]):
                    removed_pair_s = f"{current_settings['cryptos'][idx_del_c_pick_s]['symbol']}-{current_settings['cryptos'][idx_del_c_pick_s]['currency']}"
                    current_settings["cryptos"].pop(idx_del_c_pick_s); save_settings(current_settings); log_info(f"Konfigurasi {removed_pair_s} dihapus.")
                else: print(f"{AnsiColors.RED}Pilihan hapus tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice_val == 7: break
        except ValueError: print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}"); show_spinner(1.5, "Error...")
        except Exception as e_s_menu: log_error(f"Error di menu pengaturan: {e_s_menu}"); log_exception("Traceback Error Settings Menu:"); show_spinner(1.5, "Error...")
    return current_settings

# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use: log_error(f"Tidak ada API key untuk fetch_candles.", pair_name); raise APIKeyError("API Key tidak tersedia.")
    all_accumulated_candles = []; current_to_ts = None; api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"
    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"; is_large_fetch = total_limit_desired > 10
    if is_large_fetch: log_info(f"Pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name)
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles); limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        if limit_for_this_api_call <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit_for_this_api_call, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts
        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
                key_disp = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetch batch (Key: {key_disp}, Limit: {limit_for_this_api_call}, toTs: {current_to_ts})", pair_name)
            response = requests.get(url, params=params, timeout=20)
            if response.status_code in [401, 403, 429]:
                err_data = {}; err_msg = f"HTTP Error {response.status_code}"
                try: err_data = response.json(); err_msg = err_data.get('Message', err_msg)
                except json.JSONDecodeError: pass
                key_disp_err = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {err_msg}{AnsiColors.ENDC} Key: {key_disp_err}", pair_name); raise APIKeyError(f"HTTP {response.status_code}: {err_msg}")
            response.raise_for_status(); data = response.json()
            if data.get('Response') == 'Error':
                err_msg_json = data.get('Message', 'N/A'); key_err_msgs = ["api key is invalid", "apikey_is_missing", "apikey_invalid", "your_monthly_calls_are_over_the_limit", "rate limit exceeded", "your_pro_tier_has_expired_or_is_not_active", "you are over your rate limit", "please pass an API key", "api_key not found"]
                key_disp_json_err = ("..." + current_api_key_to_use[-5:]) if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in err_msg_json.lower() for keyword in key_err_msgs):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {err_msg_json}{AnsiColors.ENDC} Key: {key_disp_json_err}", pair_name); raise APIKeyError(f"JSON Error: {err_msg_json}")
                else: log_error(f"{AnsiColors.RED}API Error CryptoCompare: {err_msg_json}{AnsiColors.ENDC} (Params: {params})", pair_name); break
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle/format tidak sesuai. Total: {len(all_accumulated_candles)}.", pair_name); break
            raw_candles_api = data['Data']['Data']
            if not raw_candles_api:
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total: {len(all_accumulated_candles)}.", pair_name); break
            batch_list = []; req_ohlcv_keys = ['time', 'open', 'high', 'low', 'close', 'volumefrom']
            for item in raw_candles_api:
                if not all(k in item and item[k] is not None for k in req_ohlcv_keys): log_warning(f"Candle tidak lengkap dari API @ ts {item.get('time', 'N/A')}. Dilewati.", pair_name); continue
                batch_list.append({'timestamp': datetime.fromtimestamp(item['time']), 'open': item.get('open'), 'high': item.get('high'), 'low': item.get('low'), 'close': item.get('close'), 'volume': item.get('volumefrom')})
            if current_to_ts is not None and all_accumulated_candles and batch_list and batch_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                 if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_list[-1]['timestamp']}", pair_name); batch_list.pop()
            if not batch_list and current_to_ts is not None :
                if is_large_fetch: log_info("Batch kosong setelah overlap removal. Akhir data/hanya 1 overlap.", pair_name); break
            all_accumulated_candles = batch_list + all_accumulated_candles
            if raw_candles_api: current_to_ts = raw_candles_api[0]['time']
            else: break
            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)
            if len(raw_candles_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit. Akhir histori.", pair_name); break
            if len(all_accumulated_candles) >= total_limit_desired: break
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_list)} baru. Total: {len(all_accumulated_candles)}. Delay...", pair_name); time.sleep(0.3)
        except APIKeyError: raise
        except requests.exceptions.RequestException as e: log_error(f"{AnsiColors.RED}Kesalahan koneksi fetch batch: {e}{AnsiColors.ENDC}", pair_name); break
        except Exception as e: log_error(f"{AnsiColors.RED}Error tak terduga fetch_candles: {e}{AnsiColors.ENDC}", pair_name); log_exception("Traceback Error Fetch Candles:", pair_name); break
    if len(all_accumulated_candles) > total_limit_desired: all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]
    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)}.", pair_name)
    return all_accumulated_candles

# --- LOGIKA STRATEGI ---
def get_initial_strategy_state(): return {"last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None, "last_pivot_high_display_info": None, "last_pivot_low_display_info": None, "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None, "active_fib_line_start_index": None, "entry_price_custom": None, "highest_price_for_trailing": None, "trailing_tp_active_custom": False, "current_trailing_stop_level": None, "emergency_sl_level_custom": None, "position_size": 0}
def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1: return pivots
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True; val_i = series_list[i]
        if val_i is None: continue
        for j in range(1, left_strength + 1):
            val_prev = series_list[i-j]
            if val_prev is None: is_pivot = False; break
            if (is_high and val_i <= val_prev) or (not is_high and val_i >= val_prev): is_pivot = False; break
        if not is_pivot: continue
        for j in range(1, right_strength + 1):
            val_next = series_list[i+j]
            if val_next is None: is_pivot = False; break
            if (is_high and val_i < val_next) or (not is_high and val_i > val_next): is_pivot = False; break # Strict inequality for right side on Highs
        if is_pivot: pivots[i] = val_i
    return pivots
def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"; strategy_state["final_pivot_high_price_confirmed"] = None; strategy_state["final_pivot_low_price_confirmed"] = None
    left_s = crypto_config['left_strength']; right_s = crypto_config['right_strength']; req_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in req_keys if candles_history and candles_history[0]): log_warning(f"{AnsiColors.ORANGE}Data candle kosong/lengkap di run_strategy_logic.{AnsiColors.ENDC}", pair_name); return strategy_state
    high_p = [c.get('high') for c in candles_history]; low_p = [c.get('low') for c in candles_history]
    raw_ph = find_pivots(high_p, left_s, right_s, True); raw_pl = find_pivots(low_p,  left_s, right_s, False)
    curr_bar_idx = len(candles_history) - 1; 
    if curr_bar_idx < 0 : return strategy_state
    idx_pivot_event_h = curr_bar_idx - right_s; idx_pivot_event_l = curr_bar_idx - right_s
    raw_ph_event = raw_ph[idx_pivot_event_h] if 0 <= idx_pivot_event_h < len(raw_ph) else None
    raw_pl_event = raw_pl[idx_pivot_event_l] if 0 <= idx_pivot_event_l < len(raw_pl) else None
    if raw_ph_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_ph_event; strategy_state["last_signal_type"] = 1
        pivot_ts = candles_history[idx_pivot_event_h]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_ts.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name)
        strategy_state["last_pivot_high_display_info"] = {'price': strategy_state['final_pivot_high_price_confirmed'], 'timestamp_ms': pivot_ts.timestamp() * 1000}
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]; strategy_state["high_bar_index_for_fib"] = idx_pivot_event_h
        if strategy_state["active_fib_level"] is not None: log_debug("Reset active FIB (new High).", pair_name); strategy_state["active_fib_level"] = None; strategy_state["active_fib_line_start_index"] = None
    if raw_pl_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pl_event; strategy_state["last_signal_type"] = -1
        pivot_ts = candles_history[idx_pivot_event_l]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_ts.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name)
        strategy_state["last_pivot_low_display_info"] = {'price': strategy_state['final_pivot_low_price_confirmed'], 'timestamp_ms': pivot_ts.timestamp() * 1000}
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None and idx_pivot_event_l > strategy_state["high_bar_index_for_fib"]:
            curr_low_fib = strategy_state["final_pivot_low_price_confirmed"]
            if strategy_state["high_price_for_fib"] is None or curr_low_fib is None: log_warning("Harga FIB tidak valid (None).", pair_name)
            else:
                calc_fib_lvl = (strategy_state["high_price_for_fib"] + curr_low_fib) / 2.0; fib_late = False
                curr_candle_fib_check = candles_history[curr_bar_idx]
                if crypto_config["enable_secure_fib"]:
                    price_key_sec_fib = crypto_config["secure_fib_check_price"].lower(); price_val_curr_c = curr_candle_fib_check.get(price_key_sec_fib)
                    if price_val_curr_c is None: log_warning(f"Harga '{price_key_sec_fib}' tidak ada di candle Secure FIB. Pakai close.", pair_name); price_val_curr_c = curr_candle_fib_check.get('close')
                    if price_val_curr_c is not None and calc_fib_lvl is not None and price_val_curr_c > calc_fib_lvl: fib_late = True
                if fib_late: log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calc_fib_lvl:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_curr_c:.5f}) > FIB.{AnsiColors.ENDC}", pair_name); strategy_state["active_fib_level"] = None; strategy_state["active_fib_line_start_index"] = None
                elif calc_fib_lvl is not None : log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calc_fib_lvl:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.5f}, L: {curr_low_fib:.5f})", pair_name); strategy_state["active_fib_level"] = calc_fib_lvl; strategy_state["active_fib_line_start_index"] = idx_pivot_event_l
            strategy_state["high_price_for_fib"] = None; strategy_state["high_bar_index_for_fib"] = None
    curr_candle = candles_history[curr_bar_idx]
    if any(curr_candle.get(k) is None for k in req_keys): log_warning(f"Data OHLC tidak lengkap candle terbaru @ {curr_candle.get('timestamp', 'N/A')}. Skip trading.", pair_name); return strategy_state
    if strategy_state["active_fib_level"] is not None and strategy_state["active_fib_line_start_index"] is not None and strategy_state["position_size"] == 0:
        is_bullish_c = curr_candle['close'] > curr_candle['open']; is_closed_above_f = curr_candle['close'] > strategy_state["active_fib_level"]
        if is_bullish_c and is_closed_above_f:
            strategy_state["position_size"] = 1; entry_px_val = curr_candle['close']; strategy_state["entry_price_custom"] = entry_px_val; strategy_state["highest_price_for_trailing"] = entry_px_val
            strategy_state["trailing_tp_active_custom"] = False; strategy_state["current_trailing_stop_level"] = None; emerg_sl_val = entry_px_val * (1 - crypto_config["emergency_sl_percent"] / 100.0); strategy_state["emergency_sl_level_custom"] = emerg_sl_val
            log_msg_buy = f"BUY ENTRY @ {entry_px_val:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl_val:.5f}"
            log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg_buy}{AnsiColors.ENDC}", pair_name); play_notification_sound()
            termux_title_buy = f"BUY Signal: {pair_name}"; termux_content_buy = f"Entry @ {entry_px_val:.5f}. SL: {emerg_sl_val:.5f}"; send_termux_notification(termux_title_buy, termux_content_buy, global_settings, pair_name_for_log=pair_name)
            email_subject_buy = f"BUY Signal: {pair_name}"; email_body_buy = (f"New BUY signal {pair_name} on {crypto_config['exchange']}.\n\nEntry Price: {entry_px_val:.5f}\nFIB Level: {strategy_state['active_fib_level']:.5f}\nEmergency SL: {emerg_sl_val:.5f}\nTime: {curr_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"); send_email_notification(email_subject_buy, email_body_buy, {**crypto_config, 'pair_name': pair_name})
            strategy_state["active_fib_level"] = None; strategy_state["active_fib_line_start_index"] = None
    if strategy_state["position_size"] > 0:
        curr_high_trail_upd = strategy_state.get("highest_price_for_trailing", curr_candle.get('high'))
        if curr_high_trail_upd is None or curr_candle.get('high') is None: log_warning("Harga tertinggi/high candle tidak valid untuk trailing.", pair_name)
        else: strategy_state["highest_price_for_trailing"] = max(curr_high_trail_upd , curr_candle['high'])
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            profit_pct = 0.0
            if strategy_state["entry_price_custom"] == 0: pass
            elif strategy_state.get("highest_price_for_trailing") is None: log_warning("highest_price_for_trailing None saat kalkulasi profit.", pair_name)
            else: profit_pct = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if profit_pct >= crypto_config["profit_target_percent_activation"]: strategy_state["trailing_tp_active_custom"] = True; log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_pct:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name)
        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            pot_new_stop_px = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or pot_new_stop_px > strategy_state["current_trailing_stop_level"]: strategy_state["current_trailing_stop_level"] = pot_new_stop_px; log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name)
        final_stop_exit = strategy_state["emergency_sl_level_custom"]; exit_comm = "Emergency SL"; exit_col = AnsiColors.RED
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_exit : final_stop_exit = strategy_state["current_trailing_stop_level"]; exit_comm = "Trailing Stop"; exit_col = AnsiColors.BLUE
        if final_stop_exit is not None and curr_candle.get('low') is not None and curr_candle['low'] <= final_stop_exit:
            exit_px_open_c = curr_candle.get('open'); exit_px_val = final_stop_exit
            if exit_px_open_c is None: log_warning("Harga open candle tidak ada untuk exit. Pakai SL.", pair_name)
            else: exit_px_val = min(exit_px_open_c, final_stop_exit)
            pnl_val = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0: pnl_val = ((exit_px_val - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if exit_comm == "Trailing Stop" and pnl_val < 0: exit_col = AnsiColors.RED
            log_msg_exit = f"EXIT ORDER @ {exit_px_val:.5f} by {exit_comm}. PnL: {pnl_val:.2f}%"
            log_info(f"{exit_col}{AnsiColors.BOLD}{log_msg_exit}{AnsiColors.ENDC}", pair_name); play_notification_sound()
            termux_title_exit = f"EXIT Signal: {pair_name}"; termux_content_exit = f"{exit_comm} @ {exit_px_val:.5f}. PnL: {pnl_val:.2f}%"; send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)
            email_subject_exit = f"Trade Closed: {pair_name} ({exit_comm})"; email_body_exit = (f"Trade closed {pair_name} on {crypto_config['exchange']}.\n\nExit Price: {exit_px_val:.5f}\nReason: {exit_comm}\nEntry: {strategy_state.get('entry_price_custom', 0):.5f}\nPnL: {pnl_val:.2f}%\nTime: {curr_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"); send_email_notification(email_subject_exit, email_body_exit, {**crypto_config, 'pair_name': pair_name})
            strategy_state["position_size"] = 0; strategy_state["entry_price_custom"] = None; strategy_state["highest_price_for_trailing"] = None; strategy_state["trailing_tp_active_custom"] = False; strategy_state["current_trailing_stop_level"] = None; strategy_state["emergency_sl_level_custom"] = None
        elif strategy_state["position_size"] > 0 : 
            plot_stop_lvl = strategy_state.get("emergency_sl_level_custom"); stop_type_info_disp = "Emergency SL"
            if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
                curr_trail_sl = strategy_state.get("current_trailing_stop_level")
                if plot_stop_lvl is None or (curr_trail_sl is not None and curr_trail_sl > plot_stop_lvl): plot_stop_lvl = curr_trail_sl; stop_type_info_disp = "Trailing SL"
            entry_px_disp = strategy_state.get('entry_price_custom', 0); sl_disp_str = f'{plot_stop_lvl:.5f} ({stop_type_info_disp})' if plot_stop_lvl is not None else 'N/A'
            log_debug(f"Posisi Aktif. Entry: {entry_px_disp:.5f}, SL: {sl_disp_str}", pair_name)
    return strategy_state

# CHART_INTEGRATION_START
shared_crypto_data_manager = {}; shared_data_lock = threading.Lock()
def prepare_chart_data_for_pair(pair_id_to_display, current_data_manager_snapshot, max_candles_to_send=MAX_CANDLES_FOR_CHART_DISPLAY):
    if pair_id_to_display not in current_data_manager_snapshot: log_warning(f"Data pair {pair_id_to_display} tidak di snapshot chart.", "SYSTEM_CHART"); return None
    pair_specific_data = current_data_manager_snapshot[pair_id_to_display]; candles_full_history = pair_specific_data.get("all_candles_list", [])
    current_strategy_state = pair_specific_data.get("strategy_state", {}); pair_config = pair_specific_data.get("config", {})
    candles_for_chart_display = candles_full_history[-max_candles_to_send:]
    ohlc_data_points = []
    if not candles_for_chart_display:
        log_warning(f"Tidak ada candle di `candles_for_chart_display` ({max_candles_to_send}) untuk {pair_id_to_display}.", "SYSTEM_CHART")
        return {"ohlc": [], "annotations_yaxis": [], "annotations_points": [], "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": None, "total_candles_on_server": len(candles_full_history)}
    for candle in candles_for_chart_display:
        req_candle_keys = ['timestamp', 'open', 'high', 'low', 'close']
        if all(k in candle and candle[k] is not None for k in req_candle_keys): ohlc_data_points.append({'x': candle['timestamp'].timestamp() * 1000, 'y': [candle['open'], candle['high'], candle['low'], candle['close']]})
        else: log_debug(f"Skip candle tidak lengkap untuk chart: {candle.get('timestamp')}", "SYSTEM_CHART")
    chart_annotations_yaxis = []; chart_annotations_points = []
    active_fib_val = current_strategy_state.get("active_fib_level")
    if active_fib_val and current_strategy_state.get("active_fib_line_start_index") is not None and ohlc_data_points: chart_annotations_yaxis.append({'y': active_fib_val, 'borderColor': '#00E396', 'label': {'borderColor': '#00E396', 'style': {'color': '#fff', 'background': '#00E396', 'fontSize':'10px', 'padding':{'left':'3px','right':'3px','top':'1px','bottom':'1px'}}, 'text': f'FIB 0.5: {active_fib_val:.5f}'}})
    if current_strategy_state.get("position_size", 0) > 0 and current_strategy_state.get("entry_price_custom") is not None and ohlc_data_points:
        entry_price_val = current_strategy_state.get("entry_price_custom"); chart_annotations_yaxis.append({'y': entry_price_val, 'borderColor': '#2698FF', 'strokeDashArray': 4, 'label': {'borderColor': '#2698FF', 'style': {'color': '#fff', 'background': '#2698FF', 'fontSize':'10px', 'padding':{'left':'3px','right':'3px','top':'1px','bottom':'1px'}}, 'text': f'Entry: {entry_price_val:.5f}'}})
        sl_level_val = current_strategy_state.get("emergency_sl_level_custom"); sl_type_text = "Emerg. SL"
        if current_strategy_state.get("trailing_tp_active_custom") and current_strategy_state.get("current_trailing_stop_level") is not None:
            current_trailing_sl_val = current_strategy_state.get("current_trailing_stop_level")
            if sl_level_val is None or (current_trailing_sl_val is not None and current_trailing_sl_val > sl_level_val): sl_level_val = current_trailing_sl_val; sl_type_text = "Trail. SL"
        if sl_level_val: chart_annotations_yaxis.append({'y': sl_level_val, 'borderColor': '#FF4560', 'label': {'borderColor': '#FF4560', 'style': {'color': '#fff', 'background': '#FF4560', 'fontSize':'10px', 'padding':{'left':'3px','right':'3px','top':'1px','bottom':'1px'}}, 'text': f'{sl_type_text}: {sl_level_val:.5f}'}})
    first_candle_ts_ms_on_chart = ohlc_data_points[0]['x'] if ohlc_data_points else 0
    last_ph_info = current_strategy_state.get("last_pivot_high_display_info")
    if last_ph_info and last_ph_info['timestamp_ms'] >= first_candle_ts_ms_on_chart: chart_annotations_points.append({'x': last_ph_info['timestamp_ms'], 'y': last_ph_info['price'], 'marker': {'size': 7, 'fillColor': '#FF0000', 'strokeColor': '#FF0000', 'shape': 'triangle', 'radius':0}, 'label': {'borderColor': '#FF0000','offsetY': -18, 'style': {'color': '#fff', 'background': '#FF0000', 'fontSize':'10px'}, 'text': 'PH'}})
    last_pl_info = current_strategy_state.get("last_pivot_low_display_info")
    if last_pl_info and last_pl_info['timestamp_ms'] >= first_candle_ts_ms_on_chart: chart_annotations_points.append({'x': last_pl_info['timestamp_ms'], 'y': last_pl_info['price'], 'marker': {'size': 7, 'fillColor': '#00CD00', 'strokeColor': '#00CD00', 'shape': 'triangle', 'radius':0, 'cssClass': 'apexcharts-marker-inverted'}, 'label': {'borderColor': '#00CD00','offsetY': 10, 'style': {'color': '#fff', 'background': '#00CD00', 'fontSize':'10px'}, 'text': 'PL'}})
    return {"ohlc": ohlc_data_points, "annotations_yaxis": chart_annotations_yaxis, "annotations_points": chart_annotations_points, "pair_name": pair_config.get('pair_name', pair_id_to_display), "last_updated_tv": candles_for_chart_display[-1]['timestamp'].timestamp() * 1000 if candles_for_chart_display else None, "total_candles_on_server": len(candles_full_history), "candles_displayed": len(ohlc_data_points)}

flask_app_instance = Flask(__name__)
HTML_CHART_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Live Crypto Chart</title><script src="https://cdn.jsdelivr.net/npm/apexcharts"></script><style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;background-color:#1e1e1e;color:#e0e0e0;display:flex;flex-direction:column;align-items:center;padding:10px}#controls{background-color:#2a2a2a;padding:10px;border-radius:8px;margin-bottom:15px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;box-shadow:0 2px 5px rgba(0,0,0,.2);width:100%;max-width:1200px}#controls label{font-size:.9em}select,button{padding:8px 12px;font-size:.9em;border-radius:5px;border:1px solid #444;background-color:#333;color:#e0e0e0;cursor:pointer}button:hover{background-color:#444}#chart-container{width:100%;max-width:1200px;background-color:#2a2a2a;padding:15px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.2)}h1{text-align:center;color:#00bcd4;margin-top:0;margin-bottom:15px;font-size:1.5em}#lastUpdatedLabel,#chartInfoLabel{font-size:.8em;color:#aaa;margin-left:auto}.apexcharts-tooltip-candlestick{background:#333!important;color:#fff!important;border:1px solid #555!important}.apexcharts-tooltip-candlestick .value{font-weight:700}.apexcharts-marker-inverted .apexcharts-marker-poly{transform:rotate(180deg);transform-origin:center}</style></head><body><h1>Live Strategy Chart</h1><div id="controls"><label for="pairSelector">Pilih Pair:</label><select id="pairSelector" onchange="handlePairSelectionChange()"></select><button onclick="loadChartDataForCurrentPair(true)">Refresh Manual</button><span id="chartInfoLabel"></span><span id="lastUpdatedLabel">Memuat...</span></div><div id="chart-container"><div id="chart"></div></div><script>let activeChart,currentSelectedPairId="",lastKnownDataTimestamp=null,autoRefreshIntervalId=null;const MAX_CANDLES_FOR_DISPLAY_JS=""" + str(MAX_CANDLES_FOR_CHART_DISPLAY) + """;const initialChartOptions={series:[{name:"Candlestick",data:[]}],chart:{type:"candlestick",height:550,id:"mainCandlestickChart",background:"#2a2a2a",animations:{enabled:!0,easing:"linear",speed:300,dynamicAnimation:{enabled:!0,speed:300}},toolbar:{show:!0,tools:{download:!0,selection:!0,zoom:!0,zoomin:!0,zoomout:!0,pan:!0,reset:!0}}},theme:{mode:"dark"},title:{text:"Memuat Data Pair...",align:"left",style:{color:"#e0e0e0",fontSize:"16px"}},xaxis:{type:"datetime",labels:{style:{colors:"#aaa"}},tooltip:{enabled:!0}},yaxis:{tooltip:{enabled:!0},labels:{style:{colors:"#aaa"},formatter:function(e){return e?e.toFixed(5):""}}},grid:{borderColor:"#444"},annotations:{yaxis:[],points:[]},tooltip:{theme:"dark",shared:!0,custom:function({series:e,seriesIndex:t,dataPointIndex:a,w:o}){if(o.globals.seriesCandleO&&o.globals.seriesCandleO[t]&&void 0!==o.globals.seriesCandleO[t][a]){const n=o.globals.seriesCandleO[t][a],r=o.globals.seriesCandleH[t][a],l=o.globals.seriesCandleL[t][a],s=o.globals.seriesCandleC[t][a],i=o.globals.seriesX[t][a],c=(new Date(i)).toLocaleString();return'<div class="apexcharts-tooltip-candlestick" style="padding:5px 10px;"><div><strong>'+c+"</strong></div><div>O: <span class=\\"value\\">"+n.toFixed(5)+"</span></div><div>H: <span class=\\"value\\">"+r.toFixed(5)+"</span></div><div>L: <span class=\\"value\\">"+l.toFixed(5)+"</span></div><div>C: <span class=\\"value\\">"+s.toFixed(5)+"</span></div></div>"}return""}},noData:{text:"Tidak ada data untuk ditampilkan.",align:"center",verticalAlign:"middle",style:{color:"#ccc",fontSize:"14px"}}};async function fetchAvailablePairs(){try{const e=await fetch("/api/available_pairs");if(!e.ok)throw new Error(`Gagal memuat daftar pair: ${e.status}`);const t=await e.json(),a=document.getElementById("pairSelector");if(a.innerHTML=" depictions (e.g. pivot highs/lows, fib levels) are correctly aligned with the chart data after this optimization.
*   **Pengujian di Termux**: Tetap penting untuk menguji di lingkungan target (Termux) karena keterbatasan sumber daya di sana bisa lebih signifikan.

Semoga ini memberikan peningkatan performa yang signifikan, Bro!
