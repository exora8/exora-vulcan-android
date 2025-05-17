# Impor dan kelas AnsiColors sama seperti sebelumnya
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
    from pick import pick, Picker 
except ImportError:
    print("Error: Library 'pick' tidak ditemukan. Silakan install dengan 'pip install pick'")
    sys.exit(1)
try:
    import getpass 
except ImportError:
    pass

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
    WHITE = '\033[97m'  
    GREY = '\033[90m'

# ... DEFINISIKAN SEMUA FUNGSI LAINNYA DI SINI ...
# (animated_text_display, show_spinner, simple_progress_bar, logging, APIKeyManager,
#  play_notification_sound, send_email_notification, get_default_crypto_config,
#  load_settings, save_settings, termux_input, _prompt_crypto_config,
#  fetch_candles, get_initial_strategy_state, find_pivots, run_strategy_logic, start_trading)
# Pastikan semua fungsi dari jawaban sebelumnya ada di sini.
# Contoh beberapa fungsi yang relevan (lengkapnya ada di jawaban sebelumnya):

def clear_screen_animated():
    os.system('cls' if os.name == 'nt' else 'clear')

def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True, bold=False, center=False, width=None):
    if width is None:
        try:
            if sys.stdout.isatty(): width = os.get_terminal_size().columns
            else: width = 80
        except OSError: width = 80 
    
    # Hitung panjang teks tanpa ANSI untuk centering yang benar
    plain_text_for_centering = text
    for code in vars(AnsiColors).values(): # Hapus semua kode ANSI dari AnsiColors
        if isinstance(code, str) and code.startswith('\033['):
            plain_text_for_centering = plain_text_for_centering.replace(code, '')
    
    if center:
        padding = max(0, (width - len(plain_text_for_centering)) // 2) 
        sys.stdout.write(" " * padding); sys.stdout.flush()

    # Proses pewarnaan per karakter
    # Ini bisa disederhanakan jika seluruh string memiliki satu warna
    # Jika text sudah mengandung ANSI, color dan bold parameter mungkin tidak diperlukan lagi
    # Untuk sekarang, asumsikan text adalah plain atau sudah pre-formatted
    
    current_style = ""
    if color: current_style += color
    if bold: current_style += AnsiColors.BOLD

    for char_index, char in enumerate(text):
        # Jika text sudah memiliki ANSI, kita tidak ingin menimpanya per karakter
        # kecuali jika 'color' dan 'bold' memang dimaksudkan untuk override.
        # Untuk kasus ini, jika text sudah berwarna, kita biarkan saja.
        if text.startswith('\033['): # Deteksi sederhana jika teks sudah ada ANSI
             sys.stdout.write(char)
        else:
            sys.stdout.write(current_style + char + AnsiColors.ENDC if current_style else char)

        sys.stdout.flush()
        time.sleep(delay)
    
    if not text.startswith('\033['): # Jika kita yang menambahkan style, pastikan ENDC di akhir
        sys.stdout.write(AnsiColors.ENDC)
        sys.stdout.flush()

    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing...", color=AnsiColors.MAGENTA):
    spinner_chars = ['◢', '◣', '◤', '◥'] 
    start_time = time.time(); idx = 0; term_width = 80
    try:
        if sys.stdout.isatty(): term_width = os.get_terminal_size().columns
    except OSError: pass
    while (time.time() - start_time) < duration_seconds:
        display_message = message[:term_width - 7] 
        sys.stdout.write(f"\r{color}{AnsiColors.BOLD}{display_message} {spinner_chars[idx % len(spinner_chars)]} {AnsiColors.ENDC}")
        sys.stdout.flush(); time.sleep(0.15); idx += 1
    sys.stdout.write(f"\r{' ' * (term_width -1)}\r{AnsiColors.ENDC}"); sys.stdout.flush()

# Salin fungsi load_settings, save_settings, _prompt_crypto_config, start_trading, dll. dari jawaban sebelumnya ke sini

def load_settings(): # Contoh stub, gunakan versi lengkap dari jawaban sebelumnya
    defaults = {"primary_key": "YOUR_PRIMARY_KEY","recovery_keys": [],"enable_global_email_notifications_for_key_switch": False,"email_sender_address": "","email_sender_app_password": "","email_receiver_address_admin": ""}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings_data = json.load(f)
            if "api_settings" not in settings_data: settings_data["api_settings"] = defaults.copy()
            else: settings_data["api_settings"] = {**defaults, **settings_data["api_settings"]}
            if "cryptos" not in settings_data or not isinstance(settings_data["cryptos"], list): settings_data["cryptos"] = []
            for cfg in settings_data["cryptos"]:
                cfg.setdefault("id", str(uuid.uuid4())); cfg.setdefault("enabled", True)
            return settings_data
        except (json.JSONDecodeError, Exception) as e: # Tangkap error lebih umum
            log_error(f"Error baca/proses {SETTINGS_FILE}: {e}. Membuat/menggunakan default."); return {"api_settings": defaults.copy(), "cryptos": []}
    else:
        log_info(f"File {SETTINGS_FILE} tidak ditemukan. Membuat default.")
        new_settings_data = {"api_settings": defaults.copy(), "cryptos": []}
        # save_settings(new_settings_data) # Panggil save_settings yang benar
        return new_settings_data

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    while True:
        clear_screen_animated()
        term_w_main_menu = 60
        try:
            if sys.stdout.isatty():
                term_w_main_menu = os.get_terminal_size().columns
        except OSError:
            pass
        
        # --- Cetak Judul Menu Secara Manual DENGAN Warna ---
        title_lines_main = [
            f"{AnsiColors.HEADER}{AnsiColors.BOLD}╔{'═' * (term_w_main_menu - 2)}╗{AnsiColors.ENDC}",
            f"{AnsiColors.HEADER}{AnsiColors.BOLD}║{'🤖 Crypto Strategy Runner 🤖'.center(term_w_main_menu - 2)}║{AnsiColors.ENDC}",
            f"{AnsiColors.HEADER}{AnsiColors.BOLD}╠{'═' * (term_w_main_menu - 2)}╣{AnsiColors.ENDC}"
        ]
        
        active_cfgs_main = [c for c in settings.get("cryptos", []) if c.get("enabled", True)]
        if active_cfgs_main:
            title_lines_main.append(f"{AnsiColors.GREEN}{AnsiColors.BOLD}  Crypto Aktif ({len(active_cfgs_main)}):{AnsiColors.ENDC}")
            for i, c_main in enumerate(active_cfgs_main[:3]): 
                title_lines_main.append(f"    {AnsiColors.CYAN}{i+1}. {c_main['symbol']}-{c_main['currency']} {AnsiColors.GREY}({c_main['timeframe']}, {c_main['exchange']}){AnsiColors.ENDC}")
            if len(active_cfgs_main) > 3: title_lines_main.append(f"    {AnsiColors.GREY}...dan {len(active_cfgs_main)-3} lainnya.{AnsiColors.ENDC}")
        else: title_lines_main.append(f"{AnsiColors.ORANGE}  Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}")
        
        title_lines_main.append(f"{AnsiColors.GREY}  {'-' * (term_w_main_menu - 6)}{AnsiColors.ENDC}")
        
        api_s_main = settings.get("api_settings", {})
        pk_val_main = api_s_main.get('primary_key', '')
        pk_disp_main, pk_clr_main = ("BELUM DIATUR", AnsiColors.ORANGE)
        if pk_val_main and pk_val_main not in ["YOUR_PRIMARY_KEY", "YOUR_API_KEY_HERE"]:
            pk_disp_main = pk_val_main[:5] + "..." + pk_val_main[-3:] if len(pk_val_main) > 8 else pk_val_main
            pk_clr_main = AnsiColors.GREEN
        
        num_rec_keys_main = len([k for k in api_s_main.get('recovery_keys', []) if isinstance(k, str) and k.strip()])
        rk_clr_main = AnsiColors.GREEN if num_rec_keys_main > 0 else AnsiColors.ORANGE

        title_lines_main.append(f"  {AnsiColors.BLUE}Target Data: {AnsiColors.WHITE}{TARGET_BIG_DATA_CANDLES} c/pair{AnsiColors.ENDC}")
        title_lines_main.append(f"  {AnsiColors.BLUE}Primary Key: {pk_clr_main}{pk_disp_main}{AnsiColors.ENDC} | {AnsiColors.BLUE}Recovery: {rk_clr_main}{num_rec_keys_main}{AnsiColors.ENDC}")
        title_lines_main.append(f"{AnsiColors.GREY}  {'-' * (term_w_main_menu - 6)}{AnsiColors.ENDC}")
        
        for line in title_lines_main:
            print(line)
        # --- Akhir Cetak Judul Manual ---

        # Judul untuk 'pick' sekarang sangat sederhana dan TANPA WARNA jika perlu
        pick_title_for_library = " Pilih Opsi:" # Benar-benar plain
        # atau jika masih mau coba sedikit warna:
        # pick_title_for_library = f"{AnsiColors.MAGENTA}{AnsiColors.BOLD} Pilih Opsi:{AnsiColors.ENDC}" 
        # Tapi jika `^[[` muncul, gunakan versi plain di atas.

        # Opsi untuk 'pick' HARUS bersih dari kode ANSI
        options_main_plain = [
            "🚀 Mulai Analisa Realtime", 
            "⚙️  Pengaturan",
            "🚪 Keluar"
        ]
        
        indicator_colored = f'{AnsiColors.GREEN}▶{AnsiColors.ENDC} ' # Tambah spasi agar tidak dempet

        try:
            # Menggunakan pick() langsung
            chosen_option_text, index_main = pick(
                options_main_plain, 
                pick_title_for_library,  # Ini adalah judul yang dilihat oleh library pick
                indicator=indicator_colored, 
                default_index=0
            )

            # Aksi berdasarkan index
            if index_main == 0: 
                settings_copy = json.loads(json.dumps(settings)) # Deep copy
                start_trading(settings_copy) # Pastikan start_trading sudah didefinisikan
            elif index_main == 1: 
                settings = settings_menu(settings) # Pastikan settings_menu sudah didefinisikan
            elif index_main == 2: 
                log_info("Aplikasi ditutup.", pair_name="SYSTEM")
                clear_screen_animated()
                # Gunakan animated_text_display yang sudah dimodifikasi untuk menangani string dengan ANSI
                animated_text_display(f"{AnsiColors.MAGENTA}{AnsiColors.BOLD}👋 Terima kasih! Sampai jumpa! 👋{AnsiColors.ENDC}", color="", bold=False, center=True, delay=0.01, width=term_w_main_menu)
                show_spinner(0.7, "Exiting...", color=AnsiColors.BLUE); break
        except (KeyboardInterrupt, AttributeError) as e_main_menu: # AttributeError jika pick dibatalkan (misal Ctrl+C) -> None, index
            log_warning(f"Operasi menu utama dibatalkan/error: {e_main_menu}", pair_name="SYSTEM")
            if isinstance(e_main_menu, KeyboardInterrupt) or chosen_option_text is None : # Jika pick dibatalkan
                clear_screen_animated()
                animated_text_display(f"{AnsiColors.ORANGE}{AnsiColors.BOLD}Aplikasi dihentikan. Bye!{AnsiColors.ENDC}", color="", bold=False, center=True, width=term_w_main_menu); break
            show_spinner(1.5, "Error menu, coba lagi...", color=AnsiColors.RED)


def settings_menu(current_settings):
    # Implementasi settings_menu dengan prinsip yang sama:
    # Cetak header berwarna manual, opsi untuk pick harus plain.
    # ... (Implementasi lengkap settings_menu dari jawaban sebelumnya,
    #      dengan modifikasi serupa untuk memisahkan print manual dan string untuk pick) ...
    
    # Ini adalah versi SANGAT singkat, Anda perlu mengadaptasi versi lengkapnya
    while True:
        clear_screen_animated()
        term_w_settings = 60
        try:
            if sys.stdout.isatty(): term_w_settings = os.get_terminal_size().columns
        except OSError: pass

        # --- Cetak Header Settings Manual ---
        header_settings_lines = [
             f"{AnsiColors.HEADER}{AnsiColors.BOLD}╔{'═' * (term_w_settings - 2)}╗{AnsiColors.ENDC}",
             f"{AnsiColors.HEADER}{AnsiColors.BOLD}║{'⚙️ PENGATURAN'.center(term_w_settings - 2)}║{AnsiColors.ENDC}",
             # ... (Tambahkan info API key, crypto list, dll. dengan warna di sini) ...
        ]
        # Contoh satu baris info:
        api_s = current_settings.get("api_settings", {})
        pk_val = api_s.get('primary_key', 'BELUM DIATUR')
        pk_disp = pk_val[:5] + "..." + pk_val[-3:] if len(pk_val) > 10 else pk_val
        header_settings_lines.append(f"  {AnsiColors.CYAN}Primary Key: {AnsiColors.GREEN if pk_val != 'BELUM DIATUR' else AnsiColors.ORANGE}{pk_disp}{AnsiColors.ENDC}")

        for line in header_settings_lines: print(line)
        # --- Akhir Cetak Header Settings Manual ---

        pick_title_settings_plain = " Pilih Opsi Pengaturan:" # Plain title untuk pick
        options_settings_plain = [
            "Atur Primary API Key", 
            "Kelola Recovery API Keys",
            # ... (opsi lainnya tanpa ANSI) ...
            "Kembali ke Menu Utama"
        ]
        indicator_settings_colored = f'{AnsiColors.GREEN}▶{AnsiColors.ENDC} '

        try:
            opt_text_settings, idx_settings = pick(
                options_settings_plain,
                pick_title_settings_plain,
                indicator=indicator_settings_colored
            )

            # Lakukan aksi berdasarkan idx_settings
            # clear_screen_animated() dipanggil SETELAH pick, sebelum aksi
            
            # Contoh aksi:
            if opt_text_settings == "Kembali ke Menu Utama": # Lebih aman cek berdasarkan teks jika index bisa berubah
                return current_settings # Kembali ke main_menu

            # ... (logika untuk opsi lain, panggil _prompt_crypto_config, save_settings, dll.) ...
            # Pastikan semua input ke fungsi lain juga bersih jika perlu

            if opt_text_settings: # Jika ada aksi yang dilakukan (bukan pembatalan)
                 show_spinner(1, "Memproses...", color=AnsiColors.BLUE)


        except (KeyboardInterrupt, AttributeError) as e_settings_menu:
            log_warning(f"Operasi menu pengaturan dibatalkan/error: {e_settings_menu}", pair_name="SYSTEM")
            if isinstance(e_settings_menu, KeyboardInterrupt) or opt_text_settings is None:
                return current_settings # Kembali ke main_menu jika dibatalkan
            show_spinner(1.5, "Error, kembali...", color=AnsiColors.RED)
            # Bisa juga 'continue' untuk tetap di settings_menu

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        clear_screen_animated()
        term_w_banner_main = 60
        try:
            if sys.stdout.isatty(): 
                term_w_banner_main = os.get_terminal_size().columns
        except OSError:
            pass

        banner_art_lines = [ # Teks banner tanpa ANSI internal
            "  ██████╗ ██████╗ ██╗   ██╗██████╗ ████████╗",
            "  ██╔══██╗██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝",
            "  ██████╔╝██████╔╝ ╚████╔╝ ██████╔╝   ██║   ",
            "  ██╔═══╝ ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ",
            "  ██║     ██║  ██║   ██║   ██║        ██║   ",
            "  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝   "
        ]
        # Cetak banner dengan warna tapi teksnya sendiri tanpa ANSI untuk centering
        for line_art in banner_art_lines:
            print(f"{AnsiColors.CYAN}{AnsiColors.BOLD}{line_art.center(term_w_banner_main)}{AnsiColors.ENDC}")
            time.sleep(0.03) 
        
        # Baris terakhir banner dengan animasi, pastikan string sudah dibungkus warna
        last_banner_line_colored = f"{AnsiColors.MAGENTA}{AnsiColors.BOLD}CRYPTOCURRENCY TRADING BOT (Enhanced UI){AnsiColors.ENDC}"
        animated_text_display(last_banner_line_colored, color="", bold=False, delay=0.005, center=True, width=term_w_banner_main)

        print("\n" * 1)
        show_spinner(0.8, "Memuat Pengaturan...", color=AnsiColors.GREEN)
        
        # Pastikan semua fungsi yang dibutuhkan oleh main_menu dan settings_menu sudah didefinisikan di atas.
        # Jika belum, Anda akan mendapatkan NameError.
        # Contoh: pastikan load_settings, save_settings, _prompt_crypto_config, start_trading, dll. ada.
        
        main_menu()

    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}{AnsiColors.BOLD}🚨 Aplikasi dihentikan paksa. Bye! 🚨{AnsiColors.ENDC}", color="", bold=False, delay=0.01, center=True)
    except Exception as e_fatal:
        clear_screen_animated()
        print(f"{AnsiColors.RED}{AnsiColors.BOLD}🔥🔥 Terjadi error fatal: {e_fatal} 🔥🔥{AnsiColors.ENDC}")
        import traceback
        traceback.print_exc()
        try:
            # Cek jika logger sudah terdefinisi sebelum digunakan
            if 'logger' in globals() and logger and hasattr(logger, 'handlers') and logger.handlers: 
                logger.critical("MAIN LEVEL EXCEPTION (FATAL):", exc_info=True)
        except NameError: 
            pass 
        animated_text_display(f"{AnsiColors.RED}{AnsiColors.BOLD}Tekan Enter untuk keluar...{AnsiColors.ENDC}", color="", bold=False, delay=0.01, center=True)
        input()
