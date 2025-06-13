import json
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style
import math
import termios # Untuk kontrol terminal di Unix-like (Linux, Termux, macOS)
import tty     # Untuk kontrol terminal di Unix-like
import time    # Pastikan ini sudah diimpor!

# --- KONFIGURASI FILE ---
SETTINGS_FILE = 'settings.json' # Untuk ambil fee_pct
TRADES_FILE = 'trades.json'

# --- KONFIGURASI PAGINASI ---
TRADES_PER_PAGE = 3 # Jumlah trade per halaman, sesuai contohmu

# --- INISIALISASI COLORAMA ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored_direct(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
    """Prints colored text directly to stdout, bypassing the display buffer."""
    print(bright + color + text + Style.RESET_ALL, end=end)

def clear_screen():
    """Membersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_single_char_input():
    """Mendapatkan satu karakter input dari keyboard tanpa perlu Enter (Unix-like).
    Lebih robust dalam mendeteksi tombol panah.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd) # Set raw mode

        # Baca karakter pertama
        ch = sys.stdin.read(1)

        # Jika karakter pertama adalah ESC (\x1b)
        if ch == '\x1b':
            # Coba baca karakter kedua (biasanya '[' atau 'O' untuk tombol panah)
            try:
                second_char = sys.stdin.read(1)
                if second_char == '[':
                    # Ini mungkin escape sequence ANSI, coba baca karakter ketiga
                    third_char = sys.stdin.read(1)
                    # Cek apakah itu karakter untuk tombol panah
                    if third_char == 'A': return '\x1b[A' # Panah Atas
                    if third_char == 'B': return '\x1b[B' # Panah Bawah
                    if third_char == 'C': return '\x1b[C' # Panah Kanan
                    if third_char == 'D': return '\x1b[D' # Panah Kiri
                    # Jika bukan tombol panah, kembalikan urutan lengkap yang didapat
                    return '\x1b[' + third_char
                elif second_char == 'O': # Beberapa terminal lama mengirim 'O'
                    third_char = sys.stdin.read(1)
                    if third_char == 'A': return '\x1bOA' # Panah Atas
                    if third_char == 'B': return '\x1bOB' # Panah Bawah
                    # Dst... (jika diperlukan untuk panah kanan/kiri)
                    return '\x1bO' + third_char
                else:
                    # Ini adalah ESC diikuti karakter lain yang bukan '[' atau 'O'
                    return ch + second_char # Kembalikan urutan yang didapat
            except IOError:
                # Tidak ada karakter kedua yang segera tersedia, mungkin hanya tombol ESC
                return ch # Kembalikan hanya ESC
        else:
            # Bukan ESC, jadi karakter biasa
            return ch

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings) # Kembalikan pengaturan terminal

def load_settings():
    """Memuat pengaturan, hanya untuk mendapatkan fee_pct untuk perhitungan P/L."""
    default_fee_pct = 0.1 # Default jika settings.json tidak ada atau fee_pct tidak ditemukan
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                return settings.get("fee_pct", default_fee_pct)
        except json.JSONDecodeError:
            print_colored_direct(f"Warning: Error membaca '{SETTINGS_FILE}'. Menggunakan fee default.", Fore.YELLOW)
            return default_fee_pct
        except Exception as e:
            print_colored_direct(f"Warning: Kesalahan tidak terduga saat membaca '{SETTINGS_FILE}': {e}. Menggunakan fee default.", Fore.YELLOW)
            return default_fee_pct
    return default_fee_pct

def load_trades():
    """Memuat data trade dari trades.json."""
    if not os.path.exists(TRADES_FILE):
        print_colored_direct(f"\nError: File '{TRADES_FILE}' tidak ditemukan di lokasi yang sama. Pastikan nama file dan lokasi sudah benar.", Fore.RED, Style.BRIGHT)
        return []

    try:
        with open(TRADES_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip() # Baca semua konten dan hapus whitespace di awal/akhir
            
            if not content:
                print_colored_direct(f"Warning: File '{TRADES_FILE}' kosong atau hanya berisi whitespace. Tidak ada trade yang ditemukan.", Fore.YELLOW, Style.BRIGHT)
                return []
            
            # Coba parse konten sebagai JSON
            trades = json.loads(content)
            
            # Pastikan hasil parsing adalah list (array JSON)
            if not isinstance(trades, list):
                print_colored_direct(f"Error: Konten di '{TRADES_FILE}' bukan array JSON yang valid. Konten awal: '{content[:100]}...'", Fore.RED, Style.BRIGHT)
                return []
                
            return trades
    except json.JSONDecodeError as e:
        print_colored_direct(f"Error: Format JSON di '{TRADES_FILE}' tidak valid. Detail: {e}", Fore.RED, Style.BRIGHT)
        print_colored_direct("Pastikan file tidak kosong, tidak terpotong, dan memiliki struktur JSON yang benar (misal: dimulai dengan '[' dan diakhiri dengan ']').", Fore.YELLOW)
        return []
    except Exception as e:
        print_colored_direct(f"Error: Terjadi kesalahan tidak terduga saat membaca '{TRADES_FILE}': {e}", Fore.RED, Style.BRIGHT)
        return []

def calculate_pnl(entry_price, current_price, trade_type):
    """Menghitung P/L dalam persentase."""
    if trade_type == 'LONG':
        return ((current_price - entry_price) / entry_price) * 100
    elif trade_type == 'SHORT':
        return ((entry_price - current_price) / entry_price) * 100 # FIX: Corrected PnL calculation for SHORT
    return 0

def display_trade_block(trade, fee_pct):
    """Memformat satu blok trade untuk ditampilkan."""
    lines = []
    def add_to_block(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
        formatted_text = bright + color + text + Style.RESET_ALL
        if not lines or end == '\n':
            lines.append(formatted_text)
        else:
            lines[-1] += formatted_text

    add_to_block(f"--- Trade ID: {trade.get('id', 'N/A')} ---", Fore.MAGENTA, Style.BRIGHT)
    
    instrument_id = trade.get('instrumentId', 'N/A')
    trade_type = trade.get('type', 'N/A')
    status = trade.get('status', 'N/A')
    
    type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED if trade_type == 'SHORT' else Fore.WHITE
    status_color = Fore.YELLOW if status == 'OPEN' else Fore.WHITE

    add_to_block(f"  Pair       : {instrument_id}", Fore.WHITE)
    add_to_block(f"  Tipe       : {trade_type}", type_color)
    add_to_block(f"  Status     : {status}", status_color)

    entry_time = trade.get('entryTimestamp', 'N/A')
    entry_price = trade.get('entryPrice', 0.0)
    entry_reason = trade.get('entryReason', 'Tidak ada alasan.')
    
    if entry_time != 'N/A':
        entry_time_fmt = datetime.fromisoformat(entry_time.replace('Z', '')).strftime('%Y-%m-%d %H:%M:%S')
    else:
        entry_time_fmt = 'N/A'

    add_to_block(f"  Entry Time : {entry_time_fmt}", Fore.WHITE)
    add_to_block(f"  Entry Price: {entry_price:.4f}", Fore.WHITE)
    add_to_block(f"  Entry Reason: {entry_reason}", Fore.CYAN)

    if status == 'CLOSED':
        exit_time = trade.get('exitTimestamp', 'N/A')
        exit_price = trade.get('exitPrice', 0.0)
        pl_percent = trade.get('pl_percent', 0.0)
        run_up_percent = trade.get('run_up_percent', 0.0)
        max_drawdown_percent = trade.get('max_drawdown_percent', 0.0)
        
        if exit_time != 'N/A':
            exit_time_fmt = datetime.fromisoformat(exit_time.replace('Z', '')).strftime('%Y-%m-%d %H:%M:%S')
        else:
            exit_time_fmt = 'N/A'

        pl_color = Fore.GREEN if pl_percent > fee_pct else Fore.RED if pl_percent < -abs(fee_pct) else Fore.YELLOW
        
        add_to_block(f"  Exit Time  : {exit_time_fmt}", Fore.WHITE)
        add_to_block(f"  Exit Price : {exit_price:.4f}", Fore.WHITE)
        add_to_block(f"  P/L (%)    : {pl_percent:.2f}%", pl_color, Style.BRIGHT)
        add_to_block(f"  Max Profit : {run_up_percent:.2f}%", Fore.YELLOW)
        add_to_block(f"  Max Drawdown: {max_drawdown_percent:.2f}%", Fore.YELLOW)

        current_tp_checkpoint_level = trade.get('current_tp_checkpoint_level', 0.0)
        trailing_stop_price = trade.get('trailing_stop_price')
        if current_tp_checkpoint_level > 0.0 and trailing_stop_price is not None:
            add_to_block(f"  TP Checkpoint: Aktif @ {current_tp_checkpoint_level:.2f}% PnL ({trailing_stop_price:.4f})", Fore.MAGENTA)
        elif current_tp_checkpoint_level == 0.0 and trailing_stop_price is not None: 
            add_to_block(f"  TP Checkpoint: {current_tp_checkpoint_level:.2f}% (Price: {trailing_stop_price:.4f})", Fore.MAGENTA)

        entry_snapshot = trade.get('entry_snapshot')
        if entry_snapshot: 
            add_to_block("  --- Belajar dari Snapshot (Entry) ---", Fore.CYAN)
            add_to_block(f"    Bias Trend        : {entry_snapshot.get('bias', 'N/A')}", Fore.CYAN)
            
            prev_close = entry_snapshot.get('prev_candle_close')
            ema9_prev = entry_snapshot.get('ema9_prev')
            curr_close = entry_snapshot.get('current_candle_close')
            ema9_curr = entry_snapshot.get('ema9_current')

            add_to_block(f"    Prev Close vs EMA9: {prev_close:.4f}" if prev_close is not None else "N/A", end=' vs ')
            add_to_block(f"{ema9_prev:.4f}" if ema9_prev is not None else "N/A", Fore.CYAN)
            
            add_to_block(f"    Curr Close vs EMA9: {curr_close:.4f}" if curr_close is not None else "N/A", end=' vs ')
            add_to_block(f"{ema9_curr:.4f}" if ema9_curr is not None else "N/A", Fore.CYAN)
            
            pre_solidity = [f"{s:.2f}" for s in entry_snapshot.get('pre_entry_candle_solidity', [])]
            pre_direction = entry_snapshot.get('pre_entry_candle_direction', [])
            add_to_block(f"    3 Prev Solidity   : {pre_solidity}", Fore.CYAN)
            add_to_block(f"    3 Prev Direction  : {pre_direction}", Fore.CYAN)
    
    add_to_block("="*60, Fore.CYAN)
    return "\n".join(lines)


def run_interactive_viewer():
    """Menjalankan viewer interaktif."""
    fee_pct = load_settings()
    trades = load_trades()

    if not trades: 
        return 

    # Karena user mau "dari lama ke terbaru" untuk panah bawah,
    # dan kita tampilkan yang terbaru di atas, maka kita akan reverse trades
    # dan paginasi dari trade paling baru.
    trades_display_order = list(reversed(trades))
    total_trades = len(trades_display_order)
    total_pages = math.ceil(total_trades / TRADES_PER_PAGE)
    current_page = 0 # Mulai dari halaman pertama (trade paling baru)

    while True:
        clear_screen()
        print_colored_direct("\n--- Riwayat Trade Detail Interaktif ---", Fore.CYAN, Style.BRIGHT)
        print_colored_direct(f"(Halaman {current_page + 1}/{total_pages} | Trade per halaman: {TRADES_PER_PAGE})", Fore.YELLOW)
        print_colored_direct(f"(Fee per trade untuk perhitungan P/L: {fee_pct:.2f}%)", Fore.YELLOW)
        print_colored_direct("="*60, Fore.CYAN)

        start_idx = current_page * TRADES_PER_PAGE
        end_idx = min(start_idx + TRADES_PER_PAGE, total_trades)
        
        # Ambil trade untuk halaman ini
        trades_on_page = trades_display_order[start_idx:end_idx]

        if not trades_on_page:
            print_colored_direct("Tidak ada trade untuk ditampilkan di halaman ini.", Fore.YELLOW)
        else:
            for trade in trades_on_page:
                print_colored_direct(display_trade_block(trade, fee_pct))
        
        print_colored_direct("\n" + "="*60, Fore.CYAN)
        print_colored_direct("Navigasi: [↑] Terbaru | [↓] Lama | [Q] Keluar", Fore.GREEN, Style.BRIGHT)
        
        user_input = get_single_char_input().lower()

        if user_input == 'q':
            break
        elif user_input == '\x1b[a': # Tombol panah atas (sebelumnya)
            if current_page > 0:
                current_page -= 1
            else:
                print_colored_direct("Sudah di halaman trade paling terbaru.", Fore.YELLOW)
                time.sleep(1) # Tunda sebentar agar pesan terbaca
        elif user_input == '\x1b[b': # Tombol panah bawah (selanjutnya)
            if current_page < total_pages - 1:
                current_page += 1
            else:
                print_colored_direct("Sudah di halaman trade paling lama.", Fore.YELLOW)
                time.sleep(1) # Tunda sebentar
        else:
            print_colored_direct(f"Input tidak valid '{repr(user_input)}'. Gunakan tombol panah atas/bawah atau 'Q'.", Fore.RED)
            time.sleep(1)

    clear_screen() # Bersihkan layar saat keluar
    print_colored_direct("Viewer ditutup.", Fore.CYAN, Style.BRIGHT)

# --- EKSEKUSI SCRIPT ---
if __name__ == "__main__":
    if os.name != 'posix':
        print_colored_direct("Peringatan: Fungsi navigasi interaktif mungkin tidak bekerja optimal di sistem operasi ini (bukan Unix-like).", Fore.YELLOW, Style.BRIGHT)
        print_colored_direct("Dibutuhkan modul 'termios' dan 'tty' yang hanya tersedia di lingkungan Unix-like (Linux, macOS, Termux).", Fore.YELLOW)
        print_colored_direct("Tekan 'Q' untuk keluar jika tampilan tidak interaktif.", Fore.YELLOW)
    
    try:
        run_interactive_viewer()
    except Exception as e:
        clear_screen()
        print_colored_direct(f"Terjadi error tak terduga: {e}", Fore.RED, Style.BRIGHT)
        print_colored_direct("Pastikan terminal mendukung operasi mentah (raw mode) dan modul 'termios'/'tty' berfungsi.", Fore.RED)
        print_colored_direct("Jika Anda tidak menggunakan Termux/Linux/macOS, fitur ini mungkin tidak didukung. Coba gunakan Q untuk keluar.", Fore.YELLOW)
        sys.exit(1)
