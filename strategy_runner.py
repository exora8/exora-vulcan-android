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

# --- Tambahan untuk Live Chart ---
import http.server
import socketserver
import threading
import collections # Sudah ada, pastikan untuk deque
from urllib.parse import urlparse, parse_qs
import pathlib # Untuk menyajikan file statis
# --- End Tambahan untuk Live Chart ---

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

# --- Live Charting Globals ---
live_chart_data_store = {} # Dict: pair_id -> {"candles": deque, "annotations": list, "strategy_state_snapshot": {}}
live_chart_data_lock = threading.Lock()
MAX_CHART_CANDLES_DISPLAY = 300  # Max candles to display on the chart on initial load
MAX_CHART_CANDLES_STORE = MAX_CHART_CANDLES_DISPLAY + 200 # Store a bit more for history and updates
CHART_SERVER_PORT = 8008 # Port for the live chart
CHART_SERVER_THREAD = None
HTTPD = None

# --- HTML/JS Content untuk Live Chart ---
# **PENTING**: Pastikan file 'lightweight-charts.standalone.production.js' ada di direktori yang sama dengan skrip ini!
# Anda bisa mengunduhnya dari: https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js
LW_CHARTS_JS_FILENAME = "lightweight-charts.standalone.production.js"
LW_CHARTS_JS_PATH = pathlib.Path(__file__).parent / LW_CHARTS_JS_FILENAME

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Crypto Chart</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="/lightweight-charts.standalone.production.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; background-color: #131722; color: #D1D4DC; font-family: Arial, sans-serif;}}
        #chart-container {{ width: 98%; height: 80vh; margin: 10px auto; }} /* Adjusted height */
        #controls {{ text-align: center; padding: 10px; }}
        select, button {{ padding: 8px; margin: 5px; background-color: #2A2E39; color: #D1D4DC; border: 1px solid #404553; border-radius: 4px;}}
        #status-bar {{ text-align: center; padding: 5px; font-size: 0.9em; }}
        #strategy-info {{ margin-top:5px; padding: 5px; font-size: 0.8em; color: #aaa; background-color: #1e222d; border-radius:3px; min-height: 40px;}}
    </style>
</head>
<body>
    <div id="controls">
        <label for="pairSelector">Select Pair: </label>
        <select id="pairSelector"></select>
    </div>
    <div id="status-bar">
        <span id="lastUpdateTime"></span>
    </div>
    <div id="chart-container"></div>
    <div id="controls">
        <div id="strategy-info">Strategy details will appear here...</div>
    </div>

    <script>
        const chartContainer = document.getElementById('chart-container');
        const pairSelector = document.getElementById('pairSelector');
        const lastUpdateTimeElement = document.getElementById('lastUpdateTime');
        const strategyInfoElement = document.getElementById('strategy-info');
        
        let chart = null;
        let candlestickSeries = null;
        let currentPairId = null;
        let priceLinesStore = {{}}; // Object to store price lines by their unique ID

        const chartProperties = {{
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#D1D4DC',
            }},
            grid: {{
                vertLines: {{ color: '#2A2E39' }},
                horzLines: {{ color: '#2A2E39' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            priceScale: {{ borderColor: '#404553' }},
            timeScale: {{ borderColor: '#404553', timeVisible: true, secondsVisible: false }},
        }};

        function initChart() {{
            if (chart) chart.remove(); // Remove old chart if exists
            chart = LightweightCharts.createChart(chartContainer, chartProperties);
            candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            }});
        }}
        
        function updateAnnotations(annotations) {{
            const newMarkers = [];
            const currentPriceLineIds = new Set();

            annotations.forEach(ann => {{
                if (ann.type === 'marker') {{
                    newMarkers.push({{
                        time: ann.time,
                        position: ann.position,
                        color: ann.color,
                        shape: ann.shape,
                        text: ann.text,
                        size: ann.size || 1,
                    }});
                }} else if (ann.type === 'price_line') {{
                    const lineId = ann.id || ann.title; // Use 'id' if provided, else 'title' as a fallback unique key
                    currentPriceLineIds.add(lineId);

                    if (priceLinesStore[lineId]) {{ // Update existing line
                        priceLinesStore[lineId].applyOptions({{ price: ann.price, color: ann.color, title: ann.title, lineStyle: ann.lineStyle }});
                    }} else {{ // Create new line
                        priceLinesStore[lineId] = candlestickSeries.createPriceLine({{
                            price: ann.price,
                            color: ann.color || '#42A5F5',
                            lineWidth: ann.lineWidth || 2,
                            lineStyle: ann.lineStyle || LightweightCharts.LineStyle.Solid, // Solid = 0, Dotted = 1, Dashed = 2
                            axisLabelVisible: ann.axisLabelVisible !== undefined ? ann.axisLabelVisible : true,
                            title: ann.title || '',
                        }});
                    }}
                }}
            }});
            candlestickSeries.setMarkers(newMarkers);

            // Remove old price lines not in current annotations
            Object.keys(priceLinesStore).forEach(id => {{
                if (!currentPriceLineIds.has(id)) {{
                    candlestickSeries.removePriceLine(priceLinesStore[id]);
                    delete priceLinesStore[id];
                }}
            }});
        }}
        
        async function fetchDataForPair(pairId) {{
            if (!pairId || !candlestickSeries) return;
            try {{
                const response = await fetch(`/api/chart_data?pair_id=${{pairId}}`);
                if (!response.ok) {{
                    console.error('Failed to fetch chart data:', response.status);
                    lastUpdateTimeElement.textContent = `Error loading ${{pairId}}.`;
                    return;
                }}
                const data = await response.json();

                if (data.candles && data.candles.length > 0) {{
                    // Check if it's an update or full data set
                    const currentChartData = candlestickSeries.data();
                    if (currentChartData.length === 0 || data.candles.length > currentChartData.length || data.candles[0].time < currentChartData[0].time ) {{
                        candlestickSeries.setData(data.candles); // Full reset
                    }} else {{
                        data.candles.forEach(candle => candlestickSeries.update(candle)); // Incremental updates
                    }}
                }}

                if (data.annotations) {{
                    updateAnnotations(data.annotations);
                }}
                
                const now = new Date();
                lastUpdateTimeElement.textContent = `Last update for ${{pairId.replace('_',' ')}}: ${{now.toLocaleTimeString()}}`;

                if (data.strategy_state_snapshot) {{
                    let infoHtml = `<strong>${{pairId.replace('_',' ')}} State:</strong><br>`;
                    const state = data.strategy_state_snapshot;
                    infoHtml += `In Position: ${{state.in_position ? 'YES' : 'NO'}}<br>`;
                    if (state.in_position) {{
                        infoHtml += `Entry: ${{state.entry_price ? state.entry_price.toFixed(5) : 'N/A'}}<br>`;
                        infoHtml += `Current SL: ${{state.current_sl ? state.current_sl.toFixed(5) : 'N/A'}} (${{state.sl_type || ''}})<br>`;
                    }}
                    infoHtml += `Active FIB: ${{state.active_fib ? state.active_fib.toFixed(5) : 'N/A'}}`;
                    strategyInfoElement.innerHTML = infoHtml;
                }} else {{
                    strategyInfoElement.textContent = "No strategy data available.";
                }}
                
            }} catch (error) {{
                console.error('Error fetching or processing chart data:', error);
                lastUpdateTimeElement.textContent = `Error updating ${{pairId}}.`;
            }}
        }}

        async function initControls() {{
            try {{
                const response = await fetch('/api/pairs');
                const pairs = await response.json();
                pairSelector.innerHTML = ''; 
                if (pairs.length === 0) {{
                     pairSelector.innerHTML = '<option>No active pairs</option>';
                     strategyInfoElement.textContent = "Script is initializing or no pairs configured.";
                     return;
                }}
                pairs.forEach(pairId => {{
                    const option = document.createElement('option');
                    option.value = pairId;
                    option.textContent = pairId.replace('_', ' ');
                    pairSelector.appendChild(option);
                }});

                if (pairs.length > 0) {{
                    currentPairId = pairSelector.value || pairs[0]; 
                    pairSelector.value = currentPairId;
                    initChart();
                    await fetchDataForPair(currentPairId); 
                }}
            }} catch (error) {{
                console.error('Failed to fetch pair list:', error);
                 pairSelector.innerHTML = '<option>Error loading pairs</option>';
                 strategyInfoElement.textContent = "Error loading pair list from server.";
            }}
        }}

        pairSelector.addEventListener('change', async (event) => {{
            currentPairId = event.target.value;
            priceLinesStore = {{}}; // Reset stored price lines for new pair
            initChart(); // Re-initialize chart for the new pair
            await fetchDataForPair(currentPairId);
        }});
        
        window.addEventListener('resize', () => {{
            if (chart) {{
                chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
            }}
        }});

        // Initialize
        initControls(); // Load pairs initially
        setInterval(async () => {{
            if (currentPairId) {{
                await fetchDataForPair(currentPairId);
            }} else {{ 
                await initControls(); // Try to re-initialize if no pair was selected (e.g. script started after page load)
            }}
        }}, 3000); // Refresh data every 3 seconds
    </script>
</body>
</html>
"""

class ChartRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif parsed_path.path == f'/{LW_CHARTS_JS_FILENAME}':
            if LW_CHARTS_JS_PATH.exists():
                self.send_response(200)
                self.send_header("Content-type", "application/javascript")
                self.end_headers()
                with open(LW_CHARTS_JS_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"{LW_CHARTS_JS_FILENAME} not found. Please download it and place it in the script's directory.")
        elif parsed_path.path == '/api/pairs':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            with live_chart_data_lock:
                active_pairs = [pid for pid, data in live_chart_data_store.items() if data.get("candles")]
            self.wfile.write(json.dumps(active_pairs).encode('utf-8'))
        elif parsed_path.path == '/api/chart_data':
            query_components = parse_qs(parsed_path.query)
            pair_id_req = query_components.get("pair_id", [None])[0]
            
            data_to_send = {"candles": [], "annotations": [], "strategy_state_snapshot": {}}
            found = False
            if pair_id_req:
                with live_chart_data_lock:
                    if pair_id_req in live_chart_data_store:
                        pair_data = live_chart_data_store[pair_id_req]
                        data_to_send["candles"] = list(pair_data["candles"])
                        data_to_send["annotations"] = list(pair_data["annotations"])
                        data_to_send["strategy_state_snapshot"] = pair_data.get("strategy_state_snapshot", {})
                        found = True
            
            if found:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data_to_send).encode('utf-8'))
            else:
                self.send_error(404, f"Pair data not found for {pair_id_req}")
        else:
            self.send_error(404, "Resource not found")

def start_chart_server_thread():
    global HTTPD, CHART_SERVER_THREAD
    if CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive():
        log_info("Chart server thread already running.", pair_name="CHART_SYS")
        return

    def run_server():
        global HTTPD
        try:
            socketserver.TCPServer.allow_reuse_address = True
            HTTPD = socketserver.TCPServer(("0.0.0.0", CHART_SERVER_PORT), ChartRequestHandler)
            log_info(f"Live chart server starting on http://<your_ip>:{CHART_SERVER_PORT}", pair_name="CHART_SYS")
            HTTPD.serve_forever()
        except Exception as e:
            log_error(f"Chart server failed to start or crashed: {e}", pair_name="CHART_SYS")
            HTTPD = None # Ensure HTTPD is None if server fails

    CHART_SERVER_THREAD = threading.Thread(target=run_server, daemon=True)
    CHART_SERVER_THREAD.start()
    # Short delay to check if server started, though this is not foolproof
    time.sleep(0.5)
    if not HTTPD: # Check if server failed to initialize
         log_error(f"Chart server thread started but HTTPD instance is not available. Server might not be running.", pair_name="CHART_SYS")


def stop_chart_server_thread():
    global HTTPD, CHART_SERVER_THREAD
    if HTTPD:
        log_info("Attempting to stop live chart server...", pair_name="CHART_SYS")
        HTTPD.shutdown() # Signal serve_forever to stop
        HTTPD.server_close() # Close the server socket
        HTTPD = None
        log_info("Live chart server shut down.", pair_name="CHART_SYS")

    if CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive():
        log_info("Waiting for chart server thread to join...", pair_name="CHART_SYS")
        CHART_SERVER_THREAD.join(timeout=3) # Wait for the thread to finish
        if CHART_SERVER_THREAD.is_alive():
            log_warning("Chart server thread did not join in time.", pair_name="CHART_SYS")
    CHART_SERVER_THREAD = None
    log_info("Live chart server thread stopped.", pair_name="CHART_SYS")


# --- ANIMATION HELPER FUNCTIONS ---
def animated_text_display(text, delay=0.02, color=AnsiColors.CYAN, new_line=True):
    for char in text:
        sys.stdout.write(color + char + AnsiColors.ENDC if color else char)
        sys.stdout.flush()
        time.sleep(delay)
    if new_line:
        print()

def show_spinner(duration_seconds, message="Processing..."):
    spinner_chars = ['-', '\\', '|', '/']
    start_time = time.time()
    idx = 0
    sys.stdout.write(AnsiColors.MAGENTA)
    term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            pass

    while (time.time() - start_time) < duration_seconds:
        display_message = message[:term_width - 5]
        sys.stdout.write(f"\r{display_message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") # Clear spinner line
    sys.stdout.write(AnsiColors.ENDC)
    sys.stdout.flush()

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    term_width = 80
    if os.isatty(sys.stdout.fileno()):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            pass # Keep default

    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    sys.stdout.write(progress_line[:term_width])
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n') # New line on complete
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
    pass

# --- KONFIGURASI LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Can be set to DEBUG for more verbose output
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler("trading_log.txt", mode='a', encoding='utf-8') # Append mode
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
            record.pair_name = 'SYSTEM' # Default context
        return True
logger.addFilter(AddPairNameFilter())

def log_info(message, pair_name="SYSTEM"): logger.info(message, extra={'pair_name': pair_name})
def log_warning(message, pair_name="SYSTEM"): logger.warning(message, extra={'pair_name': pair_name})
def log_error(message, pair_name="SYSTEM"): logger.error(message, extra={'pair_name': pair_name})
def log_debug(message, pair_name="SYSTEM"): logger.debug(message, extra={'pair_name': pair_name}) # Added for finer logs
def log_exception(message, pair_name="SYSTEM"): logger.exception(message, extra={'pair_name': pair_name})


SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999 
TARGET_BIG_DATA_CANDLES = 2500 # Default, can be large
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 # Min seconds between refreshes after big data

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None):
        self.keys = []
        if primary_key and primary_key != "YOUR_API_KEY_HERE" and primary_key != "YOUR_PRIMARY_KEY":
            self.keys.append(primary_key)
        if recovery_keys_list:
            self.keys.extend([k for k in recovery_keys_list if k]) # Add only non-empty recovery keys

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}

        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys:
            return None
        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None # All keys used up or no keys

    def switch_to_next_key(self):
        if not self.keys: return None

        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # Global Email Notification Logic (Simplified for brevity)
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                email_subject = "Peringatan: API Key CryptoCompare Diganti Otomatis"
                email_body = f"Skrip trading telah mengganti API key. Index sekarang: {self.current_index} ({new_key_display})"
                # Simplified: Assume send_email_notification setup is correct
                send_email_notification(email_subject, email_body, {**self.global_email_settings, "enable_email_notifications": True, "pair_name": "API_KEY_MGMT"})
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                 send_email_notification("KRITIS: SEMUA API Key CryptoCompare Gagal!", "Semua API key gagal.", {**self.global_email_settings, "enable_email_notifications": True, "pair_name": "API_KEY_MGMT"})
            return None # All keys exhausted

    def has_valid_keys(self):
        return bool(self.keys)

    def total_keys(self):
        return len(self.keys)

    def get_current_key_index(self):
        return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION ---
def play_notification_sound():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 500) # frequency, duration
        else: # Linux, MacOS, Termux
            print('\a', end='', flush=True) # Bell character
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    # For global notifications, admin receiver might be different
    if settings_for_email.get('pair_name') == "API_KEY_MGMT" or settings_for_email.get('pair_name') == "GLOBAL_EMAIL":
        receiver_email = settings_for_email.get("email_receiver_address_admin", receiver_email)


    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL')) # Context for log
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
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
        log_info(f"{AnsiColors.CYAN}Notifikasi email berhasil dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=pair_name_ctx)
    except Exception as e:
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL'))
        log_error(f"{AnsiColors.RED}Gagal mengirim email notifikasi: {e}{AnsiColors.ENDC}", pair_name=pair_name_ctx)


def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    """Mengirim notifikasi menggunakan termux-notification jika diaktifkan."""
    api_settings = global_settings.get("api_settings", {}) # Ensure api_settings exists
    if not api_settings.get("enable_termux_notifications", False):
        return

    try:
        # check=False avoids exception on non-zero exit, but we can capture output
        result = subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                                check=False, # Don't raise error for non-zero exit
                                capture_output=True, text=True) # Capture output
        if result.returncode == 0:
            log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
        else:
            log_warning(f"{AnsiColors.ORANGE}Termux-notification command returned code {result.returncode}. Stderr: {result.stderr.strip()}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

    except FileNotFoundError:
        log_warning(f"{AnsiColors.ORANGE}Perintah 'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal dan paket termux-api sudah diinstal di Termux (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)


# --- FUNGSI PENGATURAN ---
def get_default_crypto_config():
    return {
        "id": str(uuid.uuid4()), "enabled": True, # Unique ID for each config
        "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG", # Core trading pair
        "timeframe": "hour", "refresh_interval_seconds": 60, # Data fetching
        "left_strength": 50, "right_strength": 150, # Pivot parameters
        "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0, # TP/Trailing
        "emergency_sl_percent": 10.0, "enable_secure_fib": True, "secure_fib_check_price": "Close", # SL / FIB
        "enable_email_notifications": False, # Email settings per pair
        "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""
    }

def load_settings():
    default_api_settings = {
        "primary_key": "YOUR_PRIMARY_KEY",
        "recovery_keys": [], # List of recovery keys
        "enable_global_email_notifications_for_key_switch": False,
        "email_sender_address": "pengirim.global@gmail.com", # For global system notifications
        "email_sender_app_password": "xxxx xxxx xxxx xxxx",
        "email_receiver_address_admin": "admin.penerima@example.com", # Admin receives system notifs
        "enable_termux_notifications": False # Global Termux notification toggle
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            try:
                settings = json.load(f)
                # Ensure api_settings and its keys exist
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v
                
                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = [] # Ensure cryptos is a list
                # Ensure each crypto config has default fields if missing
                for crypto_cfg in settings["cryptos"]:
                    default_single_cfg = get_default_crypto_config()
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True # Default to enabled
                    for key_default, val_default in default_single_cfg.items():
                        if key_default not in crypto_cfg:
                            crypto_cfg[key_default] = val_default
                return settings
            except json.JSONDecodeError:
                log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
                # Fallback to default if JSON is corrupted
                return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    # If file doesn't exist, return fresh default settings
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): # current_config is a copy
    clear_screen_animated()
    new_config = current_config.copy() # Work on a copy
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    # Enabled status
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    # Basic info
    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()
    
    # Timeframe
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}"); # Keep old/default

    # Refresh interval
    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        refresh_input = int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60)
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, refresh_input) # Enforce minimum
    except ValueError:
        print(f"{AnsiColors.RED}Input interval refresh tidak valid. Menggunakan default: {new_config.get('refresh_interval_seconds',60)}{AnsiColors.ENDC}")
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60)) # Fallback

    # Pivot Parameters
    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))
    except ValueError:
        print(f"{AnsiColors.RED}Input strength tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        # Keep old/default values
        new_config["left_strength"] = new_config.get('left_strength',50)
        new_config["right_strength"] = new_config.get('right_strength',150)

    # Trading Parameters
    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter trading tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        # Keep old/default
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',5.0)
        new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0)
        new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',10.0)

    # Secure FIB
    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower().strip()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))

    secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid. Menggunakan default: {new_config.get('secure_fib_check_price','Close')}{AnsiColors.ENDC}");


    # Email Notifications per pair
    animated_text_display("\n-- Notifikasi Email (Gmail) untuk Pair Ini --", color=AnsiColors.HEADER, delay=0.01)
    print(f"{AnsiColors.ORANGE}Kosongkan jika ingin menggunakan pengaturan email global dari API Settings (jika notif global aktif).{AnsiColors.ENDC}")
    email_enable_input = input(f"Aktifkan Notifikasi Email? (true/false) [{new_config.get('enable_email_notifications',False)}]: ").lower().strip()
    new_config["enable_email_notifications"] = True if email_enable_input == 'true' else (False if email_enable_input == 'false' else new_config.get('enable_email_notifications',False))

    new_config["email_sender_address"] = (input(f"{AnsiColors.BLUE}Email Pengirim (Gmail) [{new_config.get('email_sender_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_address','')).strip()
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip()
    new_config["email_receiver_address"] = (input(f"{AnsiColors.BLUE}Email Penerima [{new_config.get('email_receiver_address','')}]: {AnsiColors.ENDC}") or new_config.get('email_receiver_address','')).strip()
    
    return new_config

def settings_menu(current_settings):
    while True:
        clear_screen_animated()
        api_s = current_settings.get("api_settings", {}) # Ensure api_settings exists
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]

        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len([k for k in recovery_keys if k]) # Count only non-empty keys
        termux_notif_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"

        if not current_settings.get("cryptos"): # Ensure cryptos list exists
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

        # Options for 'pick' library
        selectable_options = [
            "Atur Primary API Key",
            "Kelola Recovery API Keys",
            "Atur Email Global untuk Notifikasi Sistem",
            "Aktifkan/Nonaktifkan Notifikasi Termux Realtime",
            "Tambah Konfigurasi Crypto Baru",
            "Ubah Konfigurasi Crypto",
            "Hapus Konfigurasi Crypto",
            "Kembali ke Menu Utama"
        ]
        
        action_choice = -1 # Default invalid choice
        try:
            _option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick: # Fallback for 'pick' library issues
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings) # Display the title manually
            for idx, opt_text in enumerate(selectable_options): print(f"  {idx + 1}. {opt_text}")
            try:
                choice_input = input("Pilih nomor opsi: ").strip()
                if not choice_input: raise ValueError("Input kosong")
                choice = int(choice_input) - 1
                if 0 <= choice < len(selectable_options): action_choice = choice
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka yang valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Memproses...") # Give time to read message
            if action_choice == -1: continue # Retry menu if choice was invalid

        # Process selected action
        try:
            clear_screen_animated()
            if action_choice == 0: # Atur Primary API Key
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s # Update settings dict
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 1: # Kelola Recovery API Keys
                while True: # Loop for recovery key management submenu
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k] # Filter out empty strings
                    api_s['recovery_keys'] = current_recovery # Update the list in api_s

                    if not current_recovery: recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"

                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]
                    rec_index = -1
                    try:
                        _rec_option_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e_pick_rec: # Fallback for 'pick'
                        log_error(f"Error dengan library 'pick' di menu recovery: {e_pick_rec}. Gunakan input manual.")
                        print(recovery_pick_title)
                        for idx_rec, opt_text_rec in enumerate(recovery_options_plain): print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice_input = input("Pilih nomor opsi: ").strip()
                            if not rec_choice_input: raise ValueError("Input Kosong")
                            rec_choice = int(rec_choice_input) - 1
                            if 0 <= rec_choice < len(recovery_options_plain): rec_index = rec_choice
                            else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                        except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                        show_spinner(1, "Memproses...")
                        if rec_index == -1: continue # Retry recovery menu

                    clear_screen_animated()
                    if rec_index == 0: # Tambah Recovery Key
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery # Already updated, just for clarity
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else: print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 1: # Hapus Recovery Key
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
                        else:
                            try:
                                for i_del, r_key_del in enumerate(current_recovery):
                                    r_key_del_display = r_key_del[:5] + "..." + r_key_del[-3:] if len(r_key_del) > 8 else r_key_del
                                    print(f"  {i_del+1}. {r_key_del_display}")
                                idx_del_str = input("Nomor recovery key yang akan dihapus: ").strip()
                                if not idx_del_str: raise ValueError("Input Kosong")
                                idx_del = int(idx_del_str) - 1
                                if 0 <= idx_del < len(current_recovery):
                                    removed = current_recovery.pop(idx_del)
                                    save_settings(current_settings)
                                    print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                                else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                            except ValueError: print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 2: break # Kembali ke Pengaturan Utama

            elif action_choice == 2: # Atur Email Global
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global (API Key switch, dll)? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 3: # Notifikasi Termux
                animated_text_display("-- Pengaturan Notifikasi Termux Realtime --", color=AnsiColors.HEADER)
                current_status = api_s.get('enable_termux_notifications', False)
                new_status_input = input(f"Aktifkan Notifikasi Termux? (true/false) [{current_status}]: ").lower().strip()
                if new_status_input == 'true':
                    api_s['enable_termux_notifications'] = True
                    print(f"{AnsiColors.GREEN}Notifikasi Termux diaktifkan.{AnsiColors.ENDC}")
                    print(f"{AnsiColors.ORANGE}Pastikan Termux:API terinstal dan `pkg install termux-api` sudah dijalankan.{AnsiColors.ENDC}")
                elif new_status_input == 'false':
                    api_s['enable_termux_notifications'] = False
                    print(f"{AnsiColors.GREEN}Notifikasi Termux dinonaktifkan.{AnsiColors.ENDC}")
                else: 
                    print(f"{AnsiColors.ORANGE}Input tidak valid. Status tidak berubah: {current_status}.{AnsiColors.ENDC}")
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(2, "Menyimpan & Kembali...")

            elif action_choice == 4: # Tambah Crypto
                new_crypto_conf = get_default_crypto_config() # Get a fresh default
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) # Populate it
                current_settings.setdefault("cryptos", []).append(new_crypto_conf) # Add to list
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")

            elif action_choice == 5: # Ubah Crypto
                if not current_settings.get("cryptos"): # Check if list is empty or not existing
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                else:
                    animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                    for i, crypto_conf in enumerate(current_settings["cryptos"]):
                        print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")
                    
                    idx_choice_str = input("Nomor konfigurasi crypto yang akan diubah: ").strip()
                    if not idx_choice_str: raise ValueError("Input Kosong")
                    try:
                        idx_choice = int(idx_choice_str) - 1
                        if 0 <= idx_choice < len(current_settings["cryptos"]):
                            # Pass a copy of the specific config to be modified
                            current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice].copy())
                            save_settings(current_settings)
                            log_info(f"Konfigurasi # {idx_choice+1} diubah.")
                        else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                    except ValueError: print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")

            elif action_choice == 6: # Hapus Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                else:
                    animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                    for i, crypto_conf in enumerate(current_settings["cryptos"]):
                        print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")

                    idx_choice_str = input("Nomor konfigurasi crypto yang akan dihapus: ").strip()
                    if not idx_choice_str: raise ValueError("Input Kosong")
                    try:
                        idx_choice = int(idx_choice_str) - 1
                        if 0 <= idx_choice < len(current_settings["cryptos"]):
                            removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                            current_settings["cryptos"].pop(idx_choice) # Remove from list
                            save_settings(current_settings)
                            log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                        else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                    except ValueError: print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            
            elif action_choice == 7: # Kembali ke Menu Utama
                break # Exit settings_menu loop

        except ValueError: # Catch empty inputs or non-integer inputs where int expected
            print(f"{AnsiColors.RED}Input tidak valid atau kosong.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e: # Generic error catch for safety
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:") # Log full traceback
            show_spinner(1.5, "Error, kembali...")
    return current_settings # Return potentially modified settings

# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name_log="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name_log)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None # Start with current time, go backwards
    api_endpoint = "histohour" # Default
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10 # Simple flag for logging verbosity

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name_log)

    # Progress bar for large fetches over multiple API calls
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT :
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # API limit per call is CRYPTOCOMPARE_MAX_LIMIT (e.g. 2000)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        # When fetching historical data (current_to_ts is set), we might need one extra candle
        # to ensure overlap detection if the API returns data up to toTs (inclusive).
        # However, CryptoCompare's `toTs` is exclusive for histoday/hour/minute.
        # So, if toTs is candle X's time, API returns candles BEFORE X.
        # The +1 logic below was for inclusive toTs, may not be strictly needed but harmless.
        if current_to_ts is not None and candles_still_needed > 1 : 
            limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT) # Keep it simple


        if limit_for_this_api_call <= 0: break # Should not happen if loop condition is correct

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts # Fetch candles before this timestamp

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: 
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call}, ToTs: {current_to_ts})", pair_name=pair_name_log)

            response = requests.get(url, params=params, timeout=20) # 20s timeout

            # Handle specific HTTP errors that indicate API key issues
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass # Ignore if error response is not JSON
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name_log)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}") # Trigger key switch

            response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'Unknown API Error')
                key_related_error_messages = [ # Keywords indicating key-related issues
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit" 
                ]
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name_log)
                    raise APIKeyError(f"JSON Error: {error_message}") # Trigger key switch
                else: # Other API errors not related to keys
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name_log)
                    break # Stop fetching for this pair on non-key API errors

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name_log)
                break # No more data or unexpected format

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: # API returned empty list of candles
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name_log)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                # Ensure all essential keys are present and not None before creating candle
                if any(item.get(k) is None for k in ['time', 'open', 'high', 'low', 'close']):
                    log_warning(f"Skipping candle with missing OHLC or time: {item}", pair_name=pair_name_log)
                    continue
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom', 0) # Default volume to 0 if missing
                }
                batch_candles_list.append(candle)

            # Prepend new batch to existing candles (since we fetch backwards in time)
            # Overlap handling: if the first candle of the new batch is the same as the last one fetched
            # (This logic is primarily for fetching newest data, not historical.
            # For historical, `toTs` should prevent major overlaps if API is consistent)
            # CryptoCompare's `toTs` is exclusive, so data[0]['time'] will be older than previous `toTs`.
            # No overlap expected if `toTs` is managed correctly.
            # The old overlap logic:
            # if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
            #     if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
            #         if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name_log)
            #         batch_candles_list.pop() 

            if not batch_candles_list and current_to_ts is not None : # If batch becomes empty (e.g. after some processing)
                if is_large_fetch: log_info("Batch menjadi kosong. Kemungkinan akhir data historis.", pair_name=pair_name_log)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles 

            if raw_candles_from_api: # Set `toTs` for the next iteration to the timestamp of the oldest candle in this batch
                current_to_ts = raw_candles_from_api[0]['time'] # Oldest is at index 0 (API returns newest first)
            else: # Should not happen if `raw_candles_from_api` was checked earlier
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): 
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles', length=40)

            # If API returns fewer candles than requested, it's likely the end of available history
            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name_log)
                break 

            if len(all_accumulated_candles) >= total_limit_desired: break # Target reached

            # Small delay between API calls during large fetches to be polite
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name_log)
                time.sleep(0.3) # 0.3s delay

        except APIKeyError: # Propagate APIKeyError to trigger key switching in caller
            raise 
        except requests.exceptions.RequestException as e: # Network errors
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name_log)
            break # Stop fetching for this pair on network errors
        except Exception as e: # Other unexpected errors
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name_log)
            log_exception("Traceback Error:", pair_name=pair_name_log) # Log full traceback
            break
    
    # Trim if we overshot (e.g. last batch made it > total_limit_desired)
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:] # Keep the most recent ones

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
            # Final progress bar update
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name_log)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 1 for high, -1 for low
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, # Temporary high for FIB calculation
        "high_bar_index_for_fib": None, # Index of the high_price_for_fib
        "active_fib_level": None, # The 0.5 FIB level price once confirmed
        "active_fib_line_start_index": None, # Index where FIB became active
        "entry_price_custom": None, # Actual entry price
        "highest_price_for_trailing": None, # Tracks highest price since entry for trailing TP
        "trailing_tp_active_custom": False, # Flag if trailing TP is active
        "current_trailing_stop_level": None, # Current calculated trailing SL
        "emergency_sl_level_custom": None, # Fixed SL based on entry
        "position_size": 0, # 1 if in position, 0 otherwise
        "last_op_timestamp": None, # Timestamp of last confirmed operation (pivot, fib, entry, exit) for chart
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list) # Initialize with Nones
    # Need enough data points for left + current + right
    if len(series_list) < left_strength + right_strength + 1:
        return pivots

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue # Skip if current point is None

        # Check left side
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break # Missing data in lookback
            if is_high: # For pivot high, current must be > all left points
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # For pivot low, current must be < all left points
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue # Move to next point if left side fails

        # Check right side
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break # Missing data in lookforward
            if is_high: # For pivot high, current must be >= all right points (standard is >)
                      # CryptoCompare seems to use >= for right strength on highs, < for lows
                if series_list[i] < series_list[i+j]: is_pivot = False; break # Strict greater for highs
            else: # For pivot low, current must be <= all right points
                if series_list[i] > series_list[i+j]: is_pivot = False; break # Strict lesser for lows
        
        if is_pivot:
            pivots[i] = series_list[i] # Store the pivot price at index i
    return pivots

# Modified run_strategy_logic to accept current_pair_id for chart updates
def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings, current_pair_id_for_chart):
    # Use current_pair_id_for_chart for logging and live_chart_data_store key
    log_pair_ctx = current_pair_id_for_chart 

    # Clear dynamic annotations (lines) for this pair before processing logic
    # Markers will accumulate or be managed by max count later if needed.
    with live_chart_data_lock:
        if current_pair_id_for_chart in live_chart_data_store:
            current_annotations = live_chart_data_store[current_pair_id_for_chart]["annotations"]
            # Keep markers, remove price_lines. Price lines will be re-added if still valid.
            live_chart_data_store[current_pair_id_for_chart]["annotations"] = [
                ann for ann in current_annotations if ann['type'] == 'marker' 
            ]
            # Optional: Limit number of markers to prevent chart clutter over time
            MAX_MARKERS_PER_CHART = 75 
            live_chart_data_store[current_pair_id_for_chart]["annotations"] = \
                live_chart_data_store[current_pair_id_for_chart]["annotations"][-MAX_MARKERS_PER_CHART:]
        else: # Should not happen if initialized correctly
            live_chart_data_store[current_pair_id_for_chart] = {"candles": collections.deque(maxlen=MAX_CHART_CANDLES_STORE), "annotations": [], "strategy_state_snapshot":{}}


    # Reset confirmations for this run
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
        return strategy_state # Return state unchanged

    # Extract price series
    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]

    # Find all potential pivots across the series
    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    current_bar_index_in_list = len(candles_history) - 1 # Index of the most recent (current) candle
    if current_bar_index_in_list < 0 : return strategy_state # No candles

    # A pivot is confirmed 'right_strength' bars after it occurs.
    # So, to check for a pivot that just got confirmed *by the current bar*,
    # we look at the pivot calculation for the bar at index `current_bar_index_in_list - right_strength`.
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength
    
    # Check if a high pivot was confirmed by the current bar's close
    if 0 <= idx_pivot_event_high < len(raw_pivot_highs) and raw_pivot_highs[idx_pivot_event_high] is not None:
        # Ensure it's a new pivot type or no previous signal
        if strategy_state["last_signal_type"] != 1: # Avoid re-confirming same type immediately
            strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_highs[idx_pivot_event_high]
            strategy_state["last_signal_type"] = 1 # Mark that a high pivot was last
            pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
            strategy_state["last_op_timestamp"] = pivot_timestamp.timestamp() # For chart

            log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': pivot_timestamp.timestamp(),
                        'price': strategy_state['final_pivot_high_price_confirmed'],
                        'position': 'aboveBar', 'shape': 'arrowDown', 'color': '#FF5252', # Reddish
                        'text': f"PH {strategy_state['final_pivot_high_price_confirmed']:.2f}"
                    })

    # Check if a low pivot was confirmed
    if 0 <= idx_pivot_event_low < len(raw_pivot_lows) and raw_pivot_lows[idx_pivot_event_low] is not None:
        if strategy_state["last_signal_type"] != -1:
            strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_lows[idx_pivot_event_low]
            strategy_state["last_signal_type"] = -1 # Mark that a low pivot was last
            pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
            strategy_state["last_op_timestamp"] = pivot_timestamp.timestamp()

            log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            with live_chart_data_lock:
                 if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': pivot_timestamp.timestamp(),
                        'price': strategy_state['final_pivot_low_price_confirmed'],
                        'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', # Greenish
                        'text': f"PL {strategy_state['final_pivot_low_price_confirmed']:.2f}"
                    })

    current_candle = candles_history[current_bar_index_in_list]
    # Ensure current candle has necessary data
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=log_pair_ctx)
        return strategy_state

    # FIBONACCI LOGIC
    # If a new high pivot was just confirmed, store it for potential FIB calculation
    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high # Index where this high occurred
        
        # If a FIB was active, a new High pivot invalidates it.
        if strategy_state["active_fib_level"] is not None:
            log_debug(f"Resetting active FIB {strategy_state['active_fib_level']:.5f} due to new High Pivot.", pair_name=log_pair_ctx)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None
            # Chart: Old FIB line will be removed by the clearing logic at the start of this function.

    # If a new low pivot was just confirmed AND we have a preceding high pivot stored
    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low # Index where this low occurred

            # Ensure Low came AFTER the High used for FIB
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                if strategy_state["high_price_for_fib"] is None or current_low_price is None: 
                    log_warning("Harga untuk kalkulasi FIB tidak valid (None).", pair_name=log_pair_ctx)
                else:
                    calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0

                    is_fib_late = False
                    if crypto_config["enable_secure_fib"]:
                        # Price to check against FIB level (e.g., current close or current high)
                        price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                        if price_val_current_candle is not None and calculated_fib_level is not None and price_val_current_candle > calculated_fib_level:
                            is_fib_late = True

                    if is_fib_late:
                        log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
                        strategy_state["active_fib_level"] = None # Do not activate this FIB
                        strategy_state["active_fib_line_start_index"] = None
                    elif calculated_fib_level is not None : 
                        log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=log_pair_ctx)
                        strategy_state["active_fib_level"] = calculated_fib_level
                        strategy_state["active_fib_line_start_index"] = current_low_bar_index
                        strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()
                        
                        # Add FIB line to chart annotations
                        with live_chart_data_lock:
                            if current_pair_id_for_chart in live_chart_data_store:
                                live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                                    'type': 'price_line', 'id': 'fib_0.5_level', # Unique ID for this line type
                                    'price': strategy_state["active_fib_level"],
                                    'color': '#2962FF', 'lineWidth': 1, 'lineStyle': 2, # Dashed line (0=Solid, 1=Dotted, 2=Dashed)
                                    'title': f'FIB 0.5: {strategy_state["active_fib_level"]:.5f}'
                                })
            
            # This FIB calculation cycle is done, reset the high_price_for_fib
            # so it requires a new High pivot before another FIB can be formed.
            strategy_state["high_price_for_fib"] = None
            strategy_state["high_bar_index_for_fib"] = None

    # ENTRY LOGIC
    # If a FIB 0.5 level is active and we are not already in a position
    if strategy_state["active_fib_level"] is not None and \
       strategy_state["active_fib_line_start_index"] is not None and \
       strategy_state["position_size"] == 0:

        if current_candle.get('close') is None or current_candle.get('open') is None:
            log_warning("Nilai close atau open tidak ada di candle saat ini. Skip entry check.", pair_name=log_pair_ctx)
            return strategy_state # Or continue if other logic follows

        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            strategy_state["position_size"] = 1 # Enter position
            entry_px = current_candle['close'] # Entry at close of breakout candle
            strategy_state["entry_price_custom"] = entry_px
            strategy_state["highest_price_for_trailing"] = entry_px # Initialize for trailing
            strategy_state["trailing_tp_active_custom"] = False # Trailing not yet active
            strategy_state["current_trailing_stop_level"] = None # No trailing SL yet

            emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
            strategy_state["emergency_sl_level_custom"] = emerg_sl
            strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()

            log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
            log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            play_notification_sound() # Sound notification
            
            # Termux & Email Notifications
            termux_title = f"BUY Signal: {log_pair_ctx}"
            termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}"
            send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=log_pair_ctx)

            email_subject = f"BUY Signal: {log_pair_ctx}"
            email_body = (f"New BUY signal for {log_pair_ctx} on {crypto_config['exchange']}.\n\n"
                          f"Entry Price: {entry_px:.5f}\nFIB Level: {strategy_state['active_fib_level']:.5f}\n"
                          f"Emergency SL: {emerg_sl:.5f}\nTime: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': log_pair_ctx}) # Pass context

            # Add Entry marker and SL line to chart
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    # Entry Marker
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': current_candle['timestamp'].timestamp(),
                        'price': entry_px, 'position': 'belowBar', 'shape': 'arrowUp', 
                        'color': '#26A69A', 'text': f'BUY\n{entry_px:.2f}'
                    })
                    # SL Line
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'price_line', 'id': 'sl_level', # Unique ID
                        'price': emerg_sl, 'color': '#EF5350', 'lineWidth': 2,
                        'title': f'Emerg SL: {emerg_sl:.5f}'
                    })
            
            # Once entry taken, FIB level is consumed for this trade
            strategy_state["active_fib_level"] = None 
            strategy_state["active_fib_line_start_index"] = None
            # (Chart: FIB line will be removed by clearing logic + not re-adding it)

    # POSITION MANAGEMENT (if in a position)
    if strategy_state["position_size"] > 0:
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None).", pair_name=log_pair_ctx)
        else: # Update highest price seen since entry
            strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        # Activate Trailing TP if profit target met
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None: profit_percent = 0.0
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
                strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()


        # Update Trailing SL if active
        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            # Trail SL only upwards
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=log_pair_ctx)
                strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()


        # Determine final SL for this bar (Emergency or Trailing)
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color_ann = AnsiColors.RED # For console log
        sl_chart_color = '#EF5350' # Red for SL line on chart
        sl_type_for_chart = "Emerg SL"

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            # If trailing SL is higher (better) than emergency SL, use trailing
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color_ann = AnsiColors.BLUE
                sl_chart_color = '#2962FF' # Blue for trailing SL line
                sl_type_for_chart = "Trail SL"
        
        # Update SL line on chart if in position
        if final_stop_for_exit is not None:
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    # annotations list already cleared of old price_lines at function start
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'price_line', 'id': 'sl_level', # Keep ID consistent
                        'price': final_stop_for_exit, 'color': sl_chart_color, 'lineWidth': 2,
                        'title': f'{sl_type_for_chart}: {final_stop_for_exit:.5f}'
                    })

        # Check if SL was hit
        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price_open = current_candle.get('open')
            if exit_price_open is None: # Should not happen with earlier checks
                log_warning("Harga open candle tidak ada untuk exit. Menggunakan SL sebagai harga exit.", pair_name=log_pair_ctx)
                exit_price = final_stop_for_exit 
            else: # Assume SL hit on open of next bar or slippage to SL level
                exit_price = min(exit_price_open, final_stop_for_exit) 
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if exit_comment == "Trailing Stop" and pnl < 0: # Trailing stop hit at a loss
                exit_color_ann = AnsiColors.RED # Change log color to red

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color_ann}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            play_notification_sound()
            strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()


            termux_title_exit = f"EXIT Signal: {log_pair_ctx}"
            termux_content_exit = f"{exit_comment} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=log_pair_ctx)

            email_subject = f"Trade Closed: {log_pair_ctx} ({exit_comment})"
            email_body = (f"Trade closed for {log_pair_ctx} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\nReason: {exit_comment}\n"
                          f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\nPnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': log_pair_ctx})

            # Add Exit marker to chart
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': current_candle['timestamp'].timestamp(),
                        'price': exit_price, 
                        'position': 'aboveBar' if pnl >=0 else 'belowBar', # Conventional: loss above, profit below (or vice-versa)
                        'shape': 'arrowDown', 'color': '#FFCA28', # Amber/Orange
                        'text': f'EXIT\n{exit_price:.2f}\nPnL:{pnl:.1f}%'
                    })
                    # SL line is removed by the clearing logic at start of next call since position_size becomes 0

            # Reset state after exiting position
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            # active_fib_level was already reset on entry or new High pivot

    # Debug log for active position status
    if strategy_state["position_size"] > 0:
        plot_stop_level = strategy_state.get("emergency_sl_level_custom")
        stop_type_info = "Emergency SL"
        if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state.get("current_trailing_stop_level")
                stop_type_info = "Trailing SL"
        
        entry_price_display = strategy_state.get('entry_price_custom', 0)
        sl_display_str = f'{plot_stop_level:.5f} ({stop_type_info})' if plot_stop_level is not None else 'N/A'
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=log_pair_ctx)

    # Update strategy state snapshot for the chart UI
    with live_chart_data_lock:
        if current_pair_id_for_chart in live_chart_data_store:
            sl_level_info = None
            sl_type_info = None
            if strategy_state["position_size"] > 0:
                if strategy_state.get("trailing_tp_active_custom") and strategy_state.get("current_trailing_stop_level") is not None:
                    sl_level_info = strategy_state.get("current_trailing_stop_level")
                    sl_type_info = "Trail SL"
                elif strategy_state.get("emergency_sl_level_custom") is not None:
                    sl_level_info = strategy_state.get("emergency_sl_level_custom")
                    sl_type_info = "Emerg SL"

            live_chart_data_store[current_pair_id_for_chart]["strategy_state_snapshot"] = {
                "in_position": strategy_state["position_size"] > 0,
                "entry_price": strategy_state["entry_price_custom"],
                "current_sl": sl_level_info,
                "sl_type": sl_type_info,
                "active_fib": strategy_state.get("active_fib_level"), # This will be None if consumed by entry
                "last_op_ts": strategy_state.get("last_op_timestamp")
            }
            # Ensure active FIB line is plotted if state says it's active (and not in position yet)
            if strategy_state.get("active_fib_level") and strategy_state["position_size"] == 0:
                # Check if it's already added by the FIB logic earlier. This is a fallback.
                fib_exists = any(ann.get('id') == 'fib_0.5_level' for ann in live_chart_data_store[current_pair_id_for_chart]["annotations"])
                if not fib_exists:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'price_line', 'id': 'fib_0.5_level', 
                        'price': strategy_state["active_fib_level"],
                        'color': '#2962FF', 'lineWidth': 1, 'lineStyle': 2, 
                        'title': f'FIB 0.5: {strategy_state["active_fib_level"]:.5f}'
                    })


    return strategy_state


# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Pass full api_settings for email config during key switch
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali...", color=AnsiColors.ORANGE)
        input()
        return

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali...", color=AnsiColors.ORANGE)
        input()
        return

    # Start chart server if not already running (moved to main_menu for earlier start)
    # start_chart_server_thread() # Ensure it's running

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = current_key_display_val[:5] + "..." + current_key_display_val[-3:] if len(current_key_display_val) > 8 else current_key_display_val
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM_INIT")

    crypto_data_manager = {} # Holds state and candles for each pair
    for config in all_crypto_configs:
        # Construct a unique ID for each pair including timeframe
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_id_for_logic'] = pair_id # Store it in config for easy access
        log_pair_ctx = pair_id # Use this for logging context

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{log_pair_ctx}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [], # Full history for logic
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, 
            "big_data_email_sent": False, # To avoid spamming "download complete"
            "last_candle_fetch_time": datetime.min, # Initialize to far past
            "data_fetch_failed_consecutively": 0 # Tracks consecutive fetch failures for a pair
        }
        
        # Initialize data store for live chart for this pair_id
        with live_chart_data_lock:
            live_chart_data_store[pair_id] = {
                "candles": collections.deque(maxlen=MAX_CHART_CANDLES_STORE), 
                "annotations": [],
                "strategy_state_snapshot": {} # Initial empty snapshot
            }

        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles = []
        # Max retries for initial fetch = number of available keys, or 1 if no keys (though caught earlier)
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: # Should be caught by api_key_manager.has_valid_keys earlier
                log_error(f"BIG DATA: Semua API key habis saat mengambil data awal untuk {log_pair_ctx}.", pair_name=log_pair_ctx)
                break 

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=log_pair_ctx)
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name_log=log_pair_ctx # Pass context for logging inside fetch_candles
                )
                initial_fetch_successful = True # Mark as successful if no exception
            except APIKeyError: # Specific error from fetch_candles indicating key issue
                log_warning(f"BIG DATA: API Key gagal untuk {log_pair_ctx}. Mencoba key berikutnya.", pair_name=log_pair_ctx)
                if not api_key_manager.switch_to_next_key(): break # No more keys, break retry loop
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e: # Network errors
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {log_pair_ctx}: {e}. Tidak mengganti key.", pair_name=log_pair_ctx)
                break # Stop trying for this pair on network error
            except Exception as e_gen: # Other errors during fetch
                log_error(f"BIG DATA: Error umum saat mengambil data awal {log_pair_ctx}: {e_gen}. Tidak mengganti key.", pair_name=log_pair_ctx)
                log_exception("Traceback Error Initial Fetch:", pair_name=log_pair_ctx)
                break

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {log_pair_ctx} setelah semua upaya. Pair ini mungkin tidak diproses.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Skip this pair essentially
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Mark as 'processed' for timing
            continue # Move to next pair in all_crypto_configs

        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {log_pair_ctx}.", pair_name=log_pair_ctx)

        # Populate chart data with initial candles
        with live_chart_data_lock:
            chart_candles_deque = live_chart_data_store[pair_id]["candles"]
            chart_candles_deque.clear()
            # Add only the most recent N candles to chart initially for performance
            for c in initial_candles[-MAX_CHART_CANDLES_DISPLAY:]: 
                chart_candles_deque.append({
                    "time": c['timestamp'].timestamp(), "open": c['open'], "high": c['high'],
                    "low": c['low'], "close": c['close']
                })
        
        # Warm-up strategy state with historical data (optional, can be time-consuming)
        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1 
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state ({log_pair_ctx})...", pair_name=log_pair_ctx)
                # Iterate up to the second to last candle for warm-up, last candle is for live processing.
                # Suppress buy/sell signals during warm-up by temporarily disabling position changes.
                temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # Exclude the very last candle
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue 

                    # Run logic, but don't allow it to take trades during warm-up
                    temp_state_for_warmup["position_size"] = 0 # Ensure no trades taken
                    # Pass pair_id for potential chart annotation during warm-up (e.g. historical pivots)
                    temp_state_for_warmup = run_strategy_logic(historical_slice, config, temp_state_for_warmup, global_settings_dict, pair_id)
                
                # Restore actual strategy state, keeping pivots/FIBs found, but reset trade state
                crypto_data_manager[pair_id]["strategy_state"] = {
                    **temp_state_for_warmup, # Keep learned pivots, FIB states
                    **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                       "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                       "current_trailing_stop_level":None} # Reset trade-specific parts
                }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai untuk {log_pair_ctx}.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            else: # Not enough data for warm-up
                log_warning(f"Data awal ({len(initial_candles)}) untuk {log_pair_ctx} tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=log_pair_ctx)
        else: # No initial candles
            log_warning(f"Tidak ada data awal untuk warm-up {log_pair_ctx}.", pair_name=log_pair_ctx)

        # Check if big data collection is complete
        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {log_pair_ctx}!{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            if not crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(f"Data Downloading Complete: {log_pair_ctx}", 
                                        f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {log_pair_ctx}.", 
                                        {**config, 'pair_name': log_pair_ctx})
                crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) untuk {log_pair_ctx} ----------{AnsiColors.ENDC}", pair_name=log_pair_ctx)

    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    # MAIN TRADING LOOP
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf') # Shortest wait time for next refresh
            any_data_fetched_this_cycle = False

            for pair_id_loop, data_loop in crypto_data_manager.items(): # Iterate over each configured crypto pair
                config_loop = data_loop["config"]
                log_pair_ctx_loop = config_loop['pair_id_for_logic'] # Use the unique pair_id for logging context

                # Cooldown logic if all keys failed for this pair previously
                if data_loop.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data_loop.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # 1 hour cooldown
                        log_debug(f"Pair {log_pair_ctx_loop} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=log_pair_ctx_loop)
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600) # Cooldown contributes to wait time
                        continue # Skip processing this pair
                    else: # Cooldown finished
                        data_loop["data_fetch_failed_consecutively"] = 0 # Reset failure count
                        log_info(f"Cooldown 1 jam untuk {log_pair_ctx_loop} selesai. Mencoba fetch lagi.", pair_name=log_pair_ctx_loop)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data_loop["last_candle_fetch_time"]).total_seconds()

                # Determine required refresh interval for this pair
                required_interval_for_this_pair = 0
                if data_loop["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Faster refresh during big data collection phase
                    if config_loop.get('timeframe') == "minute": required_interval_for_this_pair = 55 # Almost every minute
                    elif config_loop.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 # Almost daily
                    else: required_interval_for_this_pair = 3580 # Almost hourly
                else: # Live trading phase
                    required_interval_for_this_pair = config_loop.get('refresh_interval_seconds', 60) 

                # If not time to refresh this pair yet, calculate remaining time and skip
                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Skip to next pair

                log_info(f"Memproses {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                data_loop["last_candle_fetch_time"] = current_loop_time # Update last fetch time
                num_candles_before_fetch = len(data_loop["all_candles_list"])

                if data_loop["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data_loop['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) untuk {log_pair_ctx_loop} ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA LIVE ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data_loop['all_candles_list'])} candles | {log_pair_ctx_loop} ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                max_retries_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_update = 0

                while retries_done_update < max_retries_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt: # No more keys globally
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                        break # Stop trying for this pair in this cycle

                    limit_fetch = 3 # Default for live updates (fetch last few candles to catch updates)
                    if data_loop["big_data_collection_phase_active"]:
                        limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data_loop["all_candles_list"])
                        if limit_fetch_needed <=0 : # Already have enough or more
                             fetch_update_successful_for_this_pair = True 
                             new_candles_batch = [] # No new candles needed
                             break
                        limit_fetch = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) # Fetch up to API max
                        limit_fetch = max(limit_fetch, 1) # Fetch at least 1

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()}) untuk {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                    try:
                        new_candles_batch = fetch_candles(
                            config_loop['symbol'], config_loop['currency'], limit_fetch, 
                            config_loop['exchange'], current_api_key_for_attempt, config_loop['timeframe'],
                            pair_name_log=log_pair_ctx_loop
                        )
                        fetch_update_successful_for_this_pair = True # Assume success if no exception
                        data_loop["data_fetch_failed_consecutively"] = 0 # Reset fail counter on success
                        any_data_fetched_this_cycle = True 
                    
                    except APIKeyError: # Key failed
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {log_pair_ctx_loop}. Mencoba key berikutnya.", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        if not api_key_manager.switch_to_next_key(): # Try next key
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                            break # No more keys, stop retrying
                        retries_done_update += 1 
                    except requests.exceptions.RequestException as e: # Network error
                        log_error(f"Error jaringan saat mengambil update {log_pair_ctx_loop}: {e}. Tidak mengganti key.", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        break # Stop retrying on network error for this cycle
                    except Exception as e_gen_update: # Other errors
                        log_error(f"Error umum saat mengambil update {log_pair_ctx_loop}: {e_gen_update}. Tidak mengganti key.", pair_name=log_pair_ctx_loop)
                        log_exception("Traceback Error Update Fetch:", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        break

                # If all keys failed for this pair in this cycle
                if data_loop.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data_loop["last_attempt_after_all_keys_failed"] = datetime.now() # Timestamp for cooldown
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {log_pair_ctx_loop}. Akan masuk cooldown.", pair_name=log_pair_ctx_loop)

                # If fetch failed or no new candles received
                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data_loop["big_data_collection_phase_active"]:
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {log_pair_ctx_loop} (fetch berhasil tapi kosong).{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {log_pair_ctx_loop} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) # Ensure this pair waits its interval
                    continue # Skip to next pair

                # Merge new candles with existing history
                # Using a dict for merging to handle updates to existing candles (by timestamp)
                merged_candles_dict = {c['timestamp']: c for c in data_loop["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 

                for candle in new_candles_batch: # new_candles_batch is newest first if from histoday etc.
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict: # New candle
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Existing candle, content updated
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1
                
                # Re-sort all candles by timestamp and update the list
                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data_loop["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data_loop['all_candles_list'])} untuk {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                elif new_candles_batch : # Fetched candles but they were identical to existing
                     log_info(f"Tidak ada candle dengan timestamp baru atau update konten untuk {log_pair_ctx_loop}. Data terakhir mungkin identik.", pair_name=log_pair_ctx_loop)

                # Update chart data (deque)
                with live_chart_data_lock:
                    if log_pair_ctx_loop in live_chart_data_store:
                        chart_deque = live_chart_data_store[log_pair_ctx_loop]["candles"]
                        # Simple update: clear and refill from the latest history
                        # More efficient would be to append/update last element of deque
                        chart_deque.clear()
                        for c_obj in data_loop["all_candles_list"][-MAX_CHART_CANDLES_STORE:]: # Use a slightly larger store for chart
                            chart_deque.append({
                                "time": c_obj['timestamp'].timestamp(), "open": c_obj['open'], 
                                "high": c_obj['high'], "low": c_obj['low'], "close": c_obj['close']
                            })


                # Handle big data collection phase completion
                if data_loop["big_data_collection_phase_active"]:
                    if len(data_loop["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {log_pair_ctx_loop}!{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                        # Trim if over target
                        if len(data_loop["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                            data_loop["all_candles_list"] = data_loop["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data_loop["big_data_email_sent"]: # Send email once
                            send_email_notification(f"Data Downloading Complete: {log_pair_ctx_loop}", 
                                                    f"Data downloading for {TARGET_BIG_DATA_CANDLES} candles complete! Now trading on {log_pair_ctx_loop}.", 
                                                    {**config_loop, 'pair_name': log_pair_ctx_loop})
                            data_loop["big_data_email_sent"] = True
                        
                        data_loop["big_data_collection_phase_active"] = False # Switch to live mode
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data_loop['all_candles_list'])} candles) untuk {log_pair_ctx_loop} ----------{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                else: # Live mode, ensure history doesn't grow indefinitely beyond target
                    if len(data_loop["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 50: # Allow some buffer
                        data_loop["all_candles_list"] = data_loop["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 50):]

                # Run strategy logic if enough data and new data arrived or phase changed
                min_len_for_pivots_logic = config_loop.get('left_strength',50) + config_loop.get('right_strength',150) + 1
                if len(data_loop["all_candles_list"]) >= min_len_for_pivots_logic:
                    # Process if new candles, or if just finished big data, or if in big data phase and got new ones
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data_loop["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_loop["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         (data_loop["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) )

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data_loop['all_candles_list'])} candle untuk {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                         data_loop["strategy_state"] = run_strategy_logic(data_loop["all_candles_list"], config_loop, data_loop["strategy_state"], global_settings_dict, log_pair_ctx_loop) # Pass pair_id
                    elif not data_loop["big_data_collection_phase_active"]: # Live mode but no new data
                         last_c_time_str = data_loop["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data_loop["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {log_pair_ctx_loop}. Data terakhir @ {last_c_time_str}.", pair_name=log_pair_ctx_loop)
                else: # Not enough data for logic
                    log_info(f"Data ({len(data_loop['all_candles_list'])}) untuk {log_pair_ctx_loop} belum cukup utk analisa (min: {min_len_for_pivots_logic}).", pair_name=log_pair_ctx_loop)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) # Update for sleep calculation
            
            # Determine sleep duration for the main loop
            sleep_duration = 15 # Default sleep if other calculations fail

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: # All keys failed globally
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM_LOOP")
                sleep_duration = 3600 # 1 hour global cooldown
            elif active_cryptos_still_in_big_data_collection > 0: # If some pairs still collecting big data
                min_big_data_interval = float('inf')
                for pid_bd, pdata_bd in crypto_data_manager.items(): # Find shortest interval for big data pairs
                    if pdata_bd["big_data_collection_phase_active"]:
                        pconfig_bd = pdata_bd["config"]
                        interval_bd_calc = 55 if pconfig_bd.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_bd.get('timeframe') == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval_bd_calc)
                
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) # Cap at 30s
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM_LOOP")
            else: # All pairs are in live mode
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM_LOOP")
                else: # Fallback if min_overall_next_refresh_seconds is not useful
                    default_refresh_from_config = 60 # Default
                    if all_crypto_configs : # Use first active config's refresh interval as a guess
                        default_refresh_from_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama).", pair_name="SYSTEM_LOOP")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: # Should not happen, but safety
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM_LOOP")
                time.sleep(1)

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e: # Catch-all for unexpected errors in the main loop
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM_CRASH")
        log_exception("Traceback Error:", pair_name="SYSTEM_CRASH") # Log full traceback
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()
        # Chart server is stopped in main_menu on full exit

# --- MENU UTAMA ---
def main_menu():
    global CHART_SERVER_THREAD # Ensure we can manage the global thread object
    settings = load_settings()

    # Start the chart server thread ONCE when main_menu is first called
    if not CHART_SERVER_THREAD or not CHART_SERVER_THREAD.is_alive():
        log_info("Starting chart server from main_menu...", pair_name="SYS_BOOT")
        start_chart_server_thread()
        # Check if it actually started
        time.sleep(1) # Give thread time to initialize server
        if not HTTPD:
            log_error("Chart server failed to initialize properly from main_menu. Chart functionality may be unavailable.",pair_name="SYS_BOOT")


    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Live Chart) =========", color=AnsiColors.HEADER, delay=0.005)

        pick_title_main = "" 
        active_configs = [cfg for cfg in settings.get("cryptos", []) if cfg.get("enabled", True)]
        if active_configs:
            pick_title_main += f"--- Crypto Aktif ({len(active_configs)}) ---\n"
            for i, cfg in enumerate(active_configs):
                pick_title_main += f"  {i+1}. {cfg.get('symbol','N/A')}-{cfg.get('currency','N/A')} (TF: {cfg.get('timeframe','N/A')}, Exch: {cfg.get('exchange','N/A')})\n"
        else:
            pick_title_main += "Tidak ada konfigurasi crypto yang aktif.\n"

        api_s = settings.get("api_settings", {})
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
             primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]
        num_recovery_keys = len([k for k in api_s.get('recovery_keys',[]) if k])
        termux_notif_main_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"
        chart_status = "Aktif" if HTTPD and (CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive()) else "Nonaktif/Error"


        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display} | Recovery Keys: {num_recovery_keys}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status} | Live Chart: {chart_status} (Port: {CHART_SERVER_PORT})\n"
        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"

        options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]
        selected_index = -1 # Default to invalid
        try:
            _option_text, selected_index = pick(options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception as e_pick_main: # Fallback for 'pick'
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main)
            for idx_main, opt_text_main in enumerate(options_plain): print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main_input = input("Pilih nomor opsi: ").strip()
                if not choice_main_input: raise ValueError("Input Kosong")
                choice_main = int(choice_main_input) - 1
                if 0 <= choice_main < len(options_plain): selected_index = choice_main
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
            show_spinner(1.5, "Memproses...")
            if selected_index == -1: continue # Retry main menu if invalid choice

        if selected_index == 0: # Mulai Analisa
            start_trading(settings) # This is a blocking call
        elif selected_index == 1: # Pengaturan
            settings = settings_menu(settings) # Potentially update settings
        elif selected_index == 2: # Keluar
            log_info("Aplikasi ditutup.", pair_name="SYSTEM_EXIT")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
            show_spinner(0.5, "Exiting")
            stop_chart_server_thread() # Stop chart server on exit
            break # Exit main_menu loop (and program)

if __name__ == "__main__":
    try:
        if not LW_CHARTS_JS_PATH.exists():
            print(f"{AnsiColors.RED}Error: File JavaScript '{LW_CHARTS_JS_FILENAME}' tidak ditemukan.{AnsiColors.ENDC}")
            print(f"{AnsiColors.ORANGE}Harap unduh dari https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js")
            print(f"dan letakkan di direktori yang sama dengan skrip ini: {pathlib.Path(__file__).parent}{AnsiColors.ENDC}")
            input("Tekan Enter untuk keluar...")
            sys.exit(1)
            
        main_menu()
    except KeyboardInterrupt:
        clear_screen_animated()
        animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan paksa. Bye!{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
        stop_chart_server_thread() # Ensure server stops on Ctrl+C
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        # Using logger for traceback for better formatting and file logging
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        stop_chart_server_thread() # Ensure server stops on critical error
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        # Ensure chart server is stopped if it was running, regardless of how __main__ exits
        if HTTPD or (CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive()):
            log_info("Final shutdown call for chart server.", pair_name="SYS_CLEANUP")
            stop_chart_server_thread()
