import json
import os
from datetime import datetime
from colorama import init, Fore, Style
import subprocess # Tambahkan ini untuk menjalankan perintah eksternal

# --- KONFIGURASI FILE ---
SETTINGS_FILE = 'settings.json' # Untuk ambil fee_pct
TRADES_FILE = 'trades.json'

# --- INISIALISASI COLORAMA ---
init(autoreset=True)

# --- FUNGSI UTILITAS & TAMPILAN ---
def print_colored_direct(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
    """Prints colored text directly to stdout, bypassing the display buffer."""
    print(bright + color + text + Style.RESET_ALL, end=end)

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
        return ((entry_price - entry_price) / entry_price) * 100 # Ini kayaknya salah, harusnya (entry-current)
    return 0

def display_trades_detail():
    """Menampilkan semua data trade secara detail menggunakan pager (less/more)."""
    fee_pct = load_settings()
    trades = load_trades()

    if not trades: 
        # Pesan error/warning sudah ditangani di load_trades, jadi langsung return
        return 

    # Collect all lines to be displayed in the pager
    display_lines = []

    def add_line(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
        # This function will add formatted strings to our list
        formatted_text = bright + color + text + Style.RESET_ALL
        if not display_lines or end == '\n':
            display_lines.append(formatted_text)
        else:
            display_lines[-1] += formatted_text

    add_line("\n--- Riwayat Trade Detail ---", Fore.CYAN, Style.BRIGHT)
    add_line(f"(Fee per trade untuk perhitungan P/L: {fee_pct:.2f}%)", Fore.YELLOW)
    add_line("="*60, Fore.CYAN)

    # Membalik urutan agar yang terbaru tampil di atas
    for trade in reversed(trades):
        add_line(f"\n--- Trade ID: {trade.get('id', 'N/A')} ---", Fore.MAGENTA, Style.BRIGHT)
        
        # Informasi Dasar
        instrument_id = trade.get('instrumentId', 'N/A')
        trade_type = trade.get('type', 'N/A')
        status = trade.get('status', 'N/A')
        
        type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED if trade_type == 'SHORT' else Fore.WHITE
        status_color = Fore.YELLOW if status == 'OPEN' else Fore.WHITE

        add_line(f"  Pair       : {instrument_id}", Fore.WHITE)
        add_line(f"  Tipe       : {trade_type}", type_color)
        add_line(f"  Status     : {status}", status_color)

        # Informasi Entry
        entry_time = trade.get('entryTimestamp', 'N/A')
        entry_price = trade.get('entryPrice', 0.0)
        entry_reason = trade.get('entryReason', 'Tidak ada alasan.')
        
        if entry_time != 'N/A':
            entry_time_fmt = datetime.fromisoformat(entry_time.replace('Z', '')).strftime('%Y-%m-%d %H:%M:%S')
        else:
            entry_time_fmt = 'N/A'

        add_line(f"  Entry Time : {entry_time_fmt}", Fore.WHITE)
        add_line(f"  Entry Price: {entry_price:.4f}", Fore.WHITE)
        add_line(f"  Entry Reason: {entry_reason}", Fore.CYAN)

        # Informasi Exit (jika CLOSED)
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
            
            add_line(f"  Exit Time  : {exit_time_fmt}", Fore.WHITE)
            add_line(f"  Exit Price : {exit_price:.4f}", Fore.WHITE)
            add_line(f"  P/L (%)    : {pl_percent:.2f}%", pl_color, Style.BRIGHT)
            add_line(f"  Max Profit : {run_up_percent:.2f}%", Fore.YELLOW)
            add_line(f"  Max Drawdown: {max_drawdown_percent:.2f}%", Fore.YELLOW)

            # Informasi Trailing TP (jika aktif)
            current_tp_checkpoint_level = trade.get('current_tp_checkpoint_level', 0.0)
            trailing_stop_price = trade.get('trailing_stop_price')
            if current_tp_checkpoint_level > 0.0 and trailing_stop_price is not None:
                add_line(f"  TP Checkpoint: Aktif @ {current_tp_checkpoint_level:.2f}% PnL ({trailing_stop_price:.4f})", Fore.MAGENTA)
            elif current_tp_checkpoint_level == 0.0 and trailing_stop_price is not None: # Ini bisa terjadi jika sudah diaktivasi tapi belum melewati checkpoint pertama
                add_line(f"  TP Checkpoint: {current_tp_checkpoint_level:.2f}% (Price: {trailing_stop_price:.4f})", Fore.MAGENTA)


            # Detail Snapshot Pembelajaran (jika ada dan trade rugi)
            entry_snapshot = trade.get('entry_snapshot')
            if entry_snapshot: # Hanya tampilkan jika ada entry_snapshot
                add_line("  --- Belajar dari Snapshot (Entry) ---", Fore.CYAN)
                add_line(f"    Bias Trend        : {entry_snapshot.get('bias', 'N/A')}", Fore.CYAN)
                
                # Menambahkan pengecekan None untuk menghindari error formatting
                prev_close = entry_snapshot.get('prev_candle_close')
                ema9_prev = entry_snapshot.get('ema9_prev')
                curr_close = entry_snapshot.get('current_candle_close')
                ema9_curr = entry_snapshot.get('ema9_current')

                add_line(f"    Prev Close vs EMA9: {prev_close:.4f}" if prev_close is not None else "N/A", end=' vs ')
                add_line(f"{ema9_prev:.4f}" if ema9_prev is not None else "N/A", Fore.CYAN)
                
                add_line(f"    Curr Close vs EMA9: {curr_close:.4f}" if curr_close is not None else "N/A", end=' vs ')
                add_line(f"{ema9_curr:.4f}" if ema9_curr is not None else "N/A", Fore.CYAN)
                
                # Menampilkan soliditas dan arah 3 candle sebelumnya
                pre_solidity = [f"{s:.2f}" for s in entry_snapshot.get('pre_entry_candle_solidity', [])]
                pre_direction = entry_snapshot.get('pre_entry_candle_direction', [])
                add_line(f"    3 Prev Solidity   : {pre_solidity}", Fore.CYAN)
                add_line(f"    3 Prev Direction  : {pre_direction}", Fore.CYAN)
        
        add_line("="*60, Fore.CYAN)

    # Convert all collected lines into a single string
    full_output_string = "\n".join(display_lines)

    # Try to pipe the output to a pager (less or more)
    pager_command = None
    try:
        # Check if 'less' is available and executable
        subprocess.run(['which', 'less'], check=True, capture_output=True) 
        pager_command = ['less', '-R', '-F', '-S'] 
        # -R: Raw control characters (for colors)
        # -F: Quit if entire file fits on first screen
        # -S: Chop long lines (don't wrap)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            # Fallback to 'more' if 'less' is not found
            subprocess.run(['which', 'more'], check=True, capture_output=True)
            pager_command = ['more']
            print_colored_direct("\n'less' command not found, using 'more' (less features but scrollable).", Fore.YELLOW)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print_colored_direct("\nNeither 'less' nor 'more' commands found. Displaying all output directly (may not be scrollable).", Fore.YELLOW)
            print_colored_direct(full_output_string)
            return

    # If a pager command is found, run it and pipe the output
    try:
        pager_process = subprocess.Popen(pager_command, stdin=subprocess.PIPE, text=True, encoding='utf-8')
        pager_process.communicate(full_output_string)
    except Exception as e:
        print_colored_direct(f"\nError using pager '{' '.join(pager_command)}': {e}. Displaying all output directly.", Fore.RED)
        print_colored_direct(full_output_string)

# --- EKSEKUSI SCRIPT ---
if __name__ == "__main__":
    display_trades_detail()
