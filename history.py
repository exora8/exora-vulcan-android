import json
import os
import sys
import time
from datetime import datetime
from colorama import init, Fore, Style
import math

# --- MODUL UNTUK INPUT INTERAKTIF (HANYA UNTUK UNIX-LIKE) ---
try:
    import termios
    import tty
    import select
    IS_UNIX = True
except ImportError:
    IS_UNIX = False
    print("Peringatan: Modul 'termios', 'tty', atau 'select' tidak ditemukan. Mode interaktif tidak akan berfungsi.")
    print("Script ini dirancang untuk lingkungan Unix-like (Linux, macOS, Termux).")


# --- KONFIGURASI ---
TRADES_FILE = 'trades.json'
TRADES_PER_PAGE = 3
REFRESH_INTERVAL_SECONDS = 1.0 # Refresh rate in seconds if no key is pressed

# --- INISIALISASI ---
init(autoreset=True)

# --- FUNGSI TAMPILAN & UTILITAS ---
def print_colored(text, color=Fore.WHITE, bright=Style.NORMAL, end='\n'):
    """Prints colored text directly to stdout."""
    sys.stdout.write(bright + color + text + Style.RESET_ALL + end)
    sys.stdout.flush()

def clear_screen():
    """Membersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

# MODIFIED: More robust get_key_input
def get_key_input(timeout=1.0):
    """
    Gets a single key press from the user with a timeout (non-blocking).
    More robustly captures multi-byte sequences like arrow keys.
    """
    if not IS_UNIX:
        time.sleep(timeout)
        return None
        
    # Check if there's data to be read from stdin within the timeout
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    
    if rlist:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            # Read the first character
            ch = sys.stdin.read(1)
            # If it's an escape sequence, read subsequent characters if available
            if ch == '\x1b':
                # Loop to read the rest of the sequence without blocking
                while select.select([sys.stdin], [], [], 0)[0]:
                    ch += sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None # Timeout occurred

def load_trades():
    """Loads and returns trades from trades.json."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content) 
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def format_trade_block(trade):
    """Formats a single trade object into a readable multi-line string."""
    lines = []
    
    def add_line(text):
        lines.append(text)

    # Header
    trade_id = trade.get('id', 'N/A')
    add_line(Style.BRIGHT + Fore.MAGENTA + f"--- Trade ID: {trade_id} ---" + Style.RESET_ALL)
    
    # Basic Info
    inst_id = trade.get('instrumentId', 'N/A')
    trade_type = trade.get('type', 'N/A')
    status = trade.get('status', 'N/A')
    type_color = Fore.GREEN if trade_type == 'LONG' else Fore.RED
    status_color = Fore.YELLOW if status == 'OPEN' else Fore.WHITE
    add_line(f"  Pair       : {inst_id}")
    add_line(f"  Tipe       : {type_color}{trade_type}{Style.RESET_ALL}")
    add_line(f"  Status     : {status_color}{status}{Style.RESET_ALL}")
    
    # Entry Info
    entry_time_str = trade.get('entryTimestamp', 'N/A')
    if entry_time_str != 'N/A':
        entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '')).strftime('%Y-%m-%d %H:%M:%S')
    else:
        entry_time = 'N/A'
    add_line(f"  Entry Time : {entry_time}")
    add_line(f"  Entry Price: {trade.get('entryPrice', 0.0):.4f}")

    # Exit Info (if applicable)
    if status == 'CLOSED':
        exit_time_str = trade.get('exitTimestamp', 'N/A')
        if exit_time_str != 'N/A':
            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '')).strftime('%Y-%m-%d %H:%M:%S')
        else:
            exit_time = 'N/A'
        
        pl_percent = trade.get('pl_percent', 0.0)
        pl_color = Fore.GREEN if pl_percent > 0 else Fore.RED
        add_line(f"  Exit Time  : {exit_time}")
        add_line(f"  Exit Price : {trade.get('exitPrice', 0.0):.4f}")
        add_line(f"  P/L (%)    : {pl_color}{Style.BRIGHT}{pl_percent:.2f}%{Style.RESET_ALL}")
        add_line(f"  Max Profit : {Fore.YELLOW}{trade.get('run_up_percent', 0.0):.2f}%{Style.RESET_ALL}")
        add_line(f"  Max Drawdown: {Fore.YELLOW}{trade.get('max_drawdown_percent', 0.0):.2f}%{Style.RESET_ALL}")

    add_line(Style.BRIGHT + Fore.CYAN + "="*60 + Style.RESET_ALL)
    return "\n".join(lines)

# --- MAIN VIEWER LOOP ---
def run_viewer():
    """Main loop to run the real-time trade viewer."""
    if not IS_UNIX:
        sys.exit(1) # Exit if not on a compatible system

    current_page = 0
    running = True

    while running:
        try:
            # 1. Load Data
            all_trades = load_trades()
            trades_display_order = list(reversed(all_trades)) # Newest first
            
            total_trades = len(trades_display_order)
            total_pages = math.ceil(total_trades / TRADES_PER_PAGE) if total_trades > 0 else 1
            
            # 2. Adjust current page if it's out of bounds (e.g., trades were deleted)
            if current_page >= total_pages:
                current_page = max(0, total_pages - 1)
                
            # 3. Clear Screen and Display Header
            clear_screen()
            print_colored("--- Vulcan AI Real-time Trade Viewer ---", Fore.CYAN, Style.BRIGHT)
            print_colored(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')} | Total Trades: {total_trades}", Fore.YELLOW)
            print_colored(f"Halaman {current_page + 1} / {total_pages}", Fore.YELLOW)
            print_colored("="*60, Fore.CYAN)
            
            # 4. Slice and Display Trades for the Current Page
            start_idx = current_page * TRADES_PER_PAGE
            end_idx = start_idx + TRADES_PER_PAGE
            trades_for_page = trades_display_order[start_idx:end_idx]
            
            if not trades_for_page:
                print_colored("\nTidak ada trade untuk ditampilkan.", Fore.WHITE)
            else:
                for trade in trades_for_page:
                    print(format_trade_block(trade))
            
            # 5. Display Navigation Footer
            print_colored("\nNavigasi: [↑] Halaman Sebelumnya | [↓] Halaman Berikutnya | [Q] Keluar", Fore.GREEN, Style.BRIGHT)
            
            # 6. Wait for Input or Timeout
            key = get_key_input(timeout=REFRESH_INTERVAL_SECONDS)
            
            if key:
                if key.lower() == 'q':
                    running = False
                elif key == '\x1b[A': # Standard ANSI Up Arrow
                    if current_page > 0:
                        current_page -= 1
                elif key == '\x1b[B': # Standard ANSI Down Arrow
                    if current_page < total_pages - 1:
                        current_page += 1
                else:
                    # NEW: Catch-all DEBUG line to show exactly what key was received.
                    print_colored(f"\n[DEBUG] Unhandled Key Received: {repr(key)}. Tekan tombol apa saja untuk melanjutkan.", Fore.RED, Style.BRIGHT)
                    get_key_input(timeout=5) # Wait for another key press or 5 seconds to allow user to read the message.
        
        except KeyboardInterrupt:
            running = False

    clear_screen()
    print_colored("Trade viewer ditutup.", Fore.CYAN)


if __name__ == "__main__":
    run_viewer()
