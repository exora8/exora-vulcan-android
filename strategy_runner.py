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
        #chart-container {{ width: 98%; height: 75vh; margin: 10px auto; }} /* Adjusted height */
        #controls {{ text-align: center; padding: 10px; }}
        select, button {{ padding: 8px; margin: 5px; background-color: #2A2E39; color: #D1D4DC; border: 1px solid #404553; border-radius: 4px;}}
        #status-bar {{ text-align: center; padding: 5px; font-size: 0.9em; }}
        #strategy-info {{ margin-top:5px; padding: 10px; font-size: 0.85em; color: #b0b3b8; background-color: #1e222d; border-radius:5px; min-height: 50px; width: 95%; margin-left:auto; margin-right:auto; text-align:left; box-sizing: border-box;}}
        .info-item {{ margin-bottom: 3px; }}
    </style>
</head>
<body>
    <div id="controls">
        <label for="pairSelector">Select Pair: </label>
        <select id="pairSelector"></select>
    </div>
    <div id="status-bar">
        <span id="lastUpdateTime">Waiting for data...</span>
    </div>
    <div id="chart-container"></div>
    <div id="strategy-info-container"> <!-- Wrapped strategy info for better control -->
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
        let priceLinesStore = {{}}; 
        let initialLoadComplete = false;
        let initControlsRetryTimeout = null;

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

        function createNewChart() {{
            if (chart) chart.remove();
            chart = LightweightCharts.createChart(chartContainer, chartProperties);
            candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            }});
            priceLinesStore = {{}}; // Reset price lines for the new chart
        }}
        
        function updateAnnotations(annotations) {{
            if (!candlestickSeries) return;
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
                    const lineId = ann.id || ann.title || `price_line_${{ann.price}}`; 
                    currentPriceLineIds.add(lineId);

                    if (priceLinesStore[lineId]) {{ 
                        priceLinesStore[lineId].applyOptions({{ price: ann.price, color: ann.color, title: ann.title, lineStyle: ann.lineStyle }});
                    }} else {{ 
                        priceLinesStore[lineId] = candlestickSeries.createPriceLine({{
                            price: ann.price,
                            color: ann.color || '#42A5F5',
                            lineWidth: ann.lineWidth || 2,
                            lineStyle: ann.lineStyle === 1 ? LightweightCharts.LineStyle.Dotted : (ann.lineStyle === 2 ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid),
                            axisLabelVisible: ann.axisLabelVisible !== undefined ? ann.axisLabelVisible : true,
                            title: ann.title || '',
                        }});
                    }}
                }}
            }});
            candlestickSeries.setMarkers(newMarkers);

            Object.keys(priceLinesStore).forEach(id => {{
                if (!currentPriceLineIds.has(id)) {{
                    candlestickSeries.removePriceLine(priceLinesStore[id]);
                    delete priceLinesStore[id];
                }}
            }});
        }}
        
        async function fetchDataForPair(pairId, isInitialCall = false) {{
            if (!pairId) {{
                console.warn("fetchDataForPair called without pairId");
                return;
            }}
            if (!candlestickSeries && isInitialCall) {{
                createNewChart(); // Create chart if it's the first call for this pair
            }}
            if (!candlestickSeries) {{
                 console.warn("Candlestick series not available for", pairId);
                 return;
            }}

            try {{
                const response = await fetch(`/api/chart_data?pair_id=${{pairId}}`);
                if (!response.ok) {{
                    console.error('Failed to fetch chart data for', pairId, 'Status:', response.status);
                    lastUpdateTimeElement.textContent = `Error loading ${{pairId.replace(/_/g, ' ')}}.`;
                    return;
                }}
                const data = await response.json();

                if (data.candles && data.candles.length > 0) {{
                    candlestickSeries.setData(data.candles); // Always set full data for simplicity on each fetch
                                                            // Lightweight charts is optimized for this.
                    if (isInitialCall) {{
                        chart.timeScale().fitContent(); // Fit content on initial load
                    }}
                }} else if (isInitialCall) {{
                    // No candles on initial call, might still be loading
                    lastUpdateTimeElement.textContent = `No candle data yet for ${{pairId.replace(/_/g, ' ')}}. Waiting...`;
                }}


                if (data.annotations) {{
                    updateAnnotations(data.annotations);
                }}
                
                const now = new Date();
                lastUpdateTimeElement.textContent = `Last update for ${{pairId.replace(/_/g, ' ')}}: ${{now.toLocaleTimeString()}}`;

                if (data.strategy_state_snapshot) {{
                    let infoHtml = `<div class="info-item"><strong>Strategy: ${{pairId.replace(/_/g, ' ')}}</strong></div>`;
                    const state = data.strategy_state_snapshot;
                    infoHtml += `<div class="info-item">In Position: <span style="color: ${{state.in_position ? '#26a69a' : '#ef5350'}}">${{state.in_position ? 'YES' : 'NO'}}</span></div>`;
                    if (state.in_position) {{
                        infoHtml += `<div class="info-item">Entry Price: ${{state.entry_price ? state.entry_price.toFixed(5) : 'N/A'}}</div>`;
                        infoHtml += `<div class="info-item">Current SL: ${{state.current_sl ? state.current_sl.toFixed(5) : 'N/A'}} (${{state.sl_type || 'N/A'}})</div>`;
                    }}
                    infoHtml += `<div class="info-item">Active FIB 0.5: ${{state.active_fib ? state.active_fib.toFixed(5) : 'N/A'}}</div>`;
                    // You can add more state details here
                    // infoHtml += `<div class="info-item">Last Op Time: ${{state.last_op_ts ? new Date(state.last_op_ts * 1000).toLocaleString() : 'N/A'}}</div>`;
                    strategyInfoElement.innerHTML = infoHtml;
                }} else {{
                    strategyInfoElement.innerHTML = `<div class="info-item">Strategy details for ${{pairId.replace(/_/g, ' ')}} are loading...</div>`;
                }}
                initialLoadComplete = true; // Mark that at least one successful fetch happened
                
            }} catch (error) {{
                console.error('Error fetching or processing chart data for', pairId, ':', error);
                lastUpdateTimeElement.textContent = `Error updating ${{pairId.replace(/_/g, ' ')}}.`;
            }}
        }}

        async function initControls() {{
            if (initControlsRetryTimeout) clearTimeout(initControlsRetryTimeout); // Clear any pending retry

            try {{
                const response = await fetch('/api/pairs');
                if (!response.ok) {{
                    console.error("Failed to fetch pair list, status:", response.status);
                    pairSelector.innerHTML = '<option>Error loading pairs...</option>';
                    strategyInfoElement.textContent = "Could not load pair list from server. Retrying...";
                    initControlsRetryTimeout = setTimeout(initControls, 3000); // Retry after 3 seconds
                    return;
                }}
                const pairs = await response.json();
                
                if (pairs.length === 0) {{
                    pairSelector.innerHTML = '<option>No active pairs yet...</option>';
                    strategyInfoElement.textContent = "Script is initializing or no pairs configured. Waiting for pairs...";
                    initControlsRetryTimeout = setTimeout(initControls, 3000); // Retry after 3 seconds
                    return; // Wait for pairs to be available
                }}

                pairSelector.innerHTML = ''; // Clear previous options
                pairs.forEach(pairId => {{
                    const option = document.createElement('option');
                    option.value = pairId;
                    option.textContent = pairId.replace(/_/g, ' '); // Replace underscore with space for display
                    pairSelector.appendChild(option);
                }});

                // If currentPairId is not set or not in the new list, select the first one
                if (!currentPairId || !pairs.includes(currentPairId)) {{
                    currentPairId = pairs[0];
                }}
                pairSelector.value = currentPairId; // Set dropdown to current/first pair

                if (currentPairId) {{
                    if (!chart) createNewChart(); // Ensure chart exists
                    await fetchDataForPair(currentPairId, true); // True for isInitialCall
                }}
                
            }} catch (error) {{
                console.error('Failed to initialize controls:', error);
                pairSelector.innerHTML = '<option>Error initializing</option>';
                strategyInfoElement.textContent = "Error initializing controls. Retrying...";
                initControlsRetryTimeout = setTimeout(initControls, 3000); // Retry
            }}
        }}

        pairSelector.addEventListener('change', async (event) => {{
            currentPairId = event.target.value;
            initialLoadComplete = false; // Reset for the new pair
            strategyInfoElement.innerHTML = `<div class="info-item">Loading data for ${{currentPairId.replace(/_/g, ' ')}}...</div>`;
            lastUpdateTimeElement.textContent = `Fetching data for ${{currentPairId.replace(/_/g, ' ')}}...`;
            
            // It's better to create a new chart instance for a new symbol
            // to ensure all old series/markers/lines are cleared properly.
            createNewChart(); 
            await fetchDataForPair(currentPairId, true); // isInitialCall = true
        }});
        
        window.addEventListener('resize', () => {{
            if (chart) {{
                chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
            }}
        }});

        // Initialize
        initControls(); 

        // Main refresh interval
        setInterval(async () => {{
            if (currentPairId && initialLoadComplete) {{ // Only refresh if initial load was successful for the current pair
                await fetchDataForPair(currentPairId, false); // Not an initial call anymore
            }} else if (!initialLoadComplete) {
                // If initial load hasn't completed, initControls will retry or eventually load the selected pair
                console.log("Waiting for initial load or pair selection...");
                if (!currentPairId && pairSelector.options.length > 0 && pairSelector.value) {
                    // This case can happen if initControls populated pairs but fetchDataForPair failed initially
                    currentPairId = pairSelector.value;
                    await fetchDataForPair(currentPairId, true); // Try initial fetch again
                } else if (!currentPairId) {
                    await initControls(); // If no current pair, try re-initializing controls
                }
            }
        }}, 5000); // Refresh data every 5 seconds (adjust as needed)
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
            active_pairs = []
            with live_chart_data_lock:
                # Only return pairs that have some candle data already, implying they are initialized
                active_pairs = [pid for pid, data in live_chart_data_store.items() if data.get("candles") and len(data["candles"]) > 0]
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
                        # Send a *copy* of the deque as a list for JSON serialization
                        data_to_send["candles"] = list(pair_data["candles"]) 
                        data_to_send["annotations"] = list(pair_data["annotations"]) # Also a copy
                        data_to_send["strategy_state_snapshot"] = pair_data.get("strategy_state_snapshot", {}).copy()
                        found = True
            
            if found:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                # Check if there are candles to send, log if not for debugging
                # if not data_to_send["candles"]:
                #     log_debug(f"API /api/chart_data: No candles to send for {pair_id_req}, though pair found.", pair_name="CHART_API")
                # else:
                #     log_debug(f"API /api/chart_data: Sending {len(data_to_send['candles'])} candles for {pair_id_req}.", pair_name="CHART_API")

                self.wfile.write(json.dumps(data_to_send).encode('utf-8'))
            else:
                log_warning(f"API /api/chart_data: Pair data not found in store for {pair_id_req}", pair_name="CHART_API")
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
        except OSError as e: # Specifically catch address in use
             if e.errno == 98: # Address already in use
                 log_error(f"Chart server port {CHART_SERVER_PORT} is already in use. Server not started.", pair_name="CHART_SYS")
             else:
                 log_error(f"Chart server OS error: {e}", pair_name="CHART_SYS")
             HTTPD = None
        except Exception as e:
            log_error(f"Chart server failed to start or crashed: {e}", pair_name="CHART_SYS")
            HTTPD = None 

    CHART_SERVER_THREAD = threading.Thread(target=run_server, daemon=True)
    CHART_SERVER_THREAD.start()
    time.sleep(0.5)
    if not HTTPD: 
         log_error(f"Chart server thread started but HTTPD instance is not available. Server might not be running (check for 'Address already in use' errors above).", pair_name="CHART_SYS")


def stop_chart_server_thread():
    global HTTPD, CHART_SERVER_THREAD
    if HTTPD:
        log_info("Attempting to stop live chart server...", pair_name="CHART_SYS")
        HTTPD.shutdown() 
        HTTPD.server_close() 
        HTTPD = None
        log_info("Live chart server shut down.", pair_name="CHART_SYS")

    if CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive():
        log_info("Waiting for chart server thread to join...", pair_name="CHART_SYS")
        CHART_SERVER_THREAD.join(timeout=3) 
        if CHART_SERVER_THREAD.is_alive():
            log_warning("Chart server thread did not join in time.", pair_name="CHART_SYS")
    CHART_SERVER_THREAD = None


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
    sys.stdout.write(f"\r{' ' * (len(display_message) + 3)}\r") 
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
            pass 

    progress_line = f'\r{AnsiColors.GREEN}{prefix} |{bar}| {percent}% {suffix}{AnsiColors.ENDC}'
    sys.stdout.write(progress_line[:term_width])
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n') 
        sys.stdout.flush()

# --- CUSTOM EXCEPTION ---
class APIKeyError(Exception):
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
CRYPTOCOMPARE_MAX_LIMIT = 1999 
TARGET_BIG_DATA_CANDLES = 2500 
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 

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
            self.keys.extend([k for k in recovery_keys_list if k]) 

        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}

        if not self.keys:
            log_warning("Tidak ada API key yang valid (primary atau recovery) yang dikonfigurasi.")

    def get_current_key(self):
        if not self.keys:
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
                email_body = f"Skrip trading telah mengganti API key. Index sekarang: {self.current_index} ({new_key_display})"
                send_email_notification(email_subject, email_body, {**self.global_email_settings, "enable_email_notifications": True, "pair_name": "API_KEY_MGMT"})
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL! Tidak dapat mengambil data.{AnsiColors.ENDC}")
            if self.global_email_settings.get("enable_global_email_notifications_for_key_switch", False):
                 send_email_notification("KRITIS: SEMUA API Key CryptoCompare Gagal!", "Semua API key gagal.", {**self.global_email_settings, "enable_email_notifications": True, "pair_name": "API_KEY_MGMT"})
            return None 

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
            winsound.Beep(1000, 500) 
        else: 
            print('\a', end='', flush=True) 
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    if settings_for_email.get('pair_name') == "API_KEY_MGMT" or settings_for_email.get('pair_name') == "GLOBAL_EMAIL":
        receiver_email = settings_for_email.get("email_receiver_address_admin", receiver_email)


    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('pair_name', settings_for_email.get('symbol', 'GLOBAL_EMAIL')) 
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
    api_settings = global_settings.get("api_settings", {}) 
    if not api_settings.get("enable_termux_notifications", False):
        return

    try:
        result = subprocess.run(['termux-notification', '--title', title, '--content', content_msg],
                                check=False, 
                                capture_output=True, text=True) 
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
        "email_receiver_address_admin": "admin.penerima@example.com", 
        "enable_termux_notifications": False 
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            try:
                settings = json.load(f)
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v
                
                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = [] 
                for crypto_cfg in settings["cryptos"]:
                    default_single_cfg = get_default_crypto_config()
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True 
                    for key_default, val_default in default_single_cfg.items():
                        if key_default not in crypto_cfg:
                            crypto_cfg[key_default] = val_default
                return settings
            except json.JSONDecodeError:
                log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
                return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")

def _prompt_crypto_config(current_config): 
    clear_screen_animated()
    new_config = current_config.copy() 
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()
    
    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}"); 

    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        refresh_input = int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60)
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, refresh_input) 
    except ValueError:
        print(f"{AnsiColors.RED}Input interval refresh tidak valid. Menggunakan default: {new_config.get('refresh_interval_seconds',60)}{AnsiColors.ENDC}")
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60)) 

    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))
    except ValueError:
        print(f"{AnsiColors.RED}Input strength tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["left_strength"] = new_config.get('left_strength',50)
        new_config["right_strength"] = new_config.get('right_strength',150)

    animated_text_display("\n-- Parameter Trading --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["profit_target_percent_activation"] = float(input(f"{AnsiColors.BLUE}Profit % Aktivasi Trailing TP [{new_config.get('profit_target_percent_activation',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"{AnsiColors.BLUE}Gap Trailing TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"{AnsiColors.RED}Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: {AnsiColors.ENDC}").strip() or new_config.get('emergency_sl_percent',10.0))
    except ValueError:
        print(f"{AnsiColors.RED}Input parameter trading tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["profit_target_percent_activation"] = new_config.get('profit_target_percent_activation',5.0)
        new_config["trailing_stop_gap_percent"] = new_config.get('trailing_stop_gap_percent',5.0)
        new_config["emergency_sl_percent"] = new_config.get('emergency_sl_percent',10.0)

    animated_text_display("\n-- Fitur Secure FIB --", color=AnsiColors.HEADER, delay=0.01)
    enable_sf_input = input(f"Aktifkan Secure FIB? (true/false) [{new_config.get('enable_secure_fib',True)}]: ").lower().strip()
    new_config["enable_secure_fib"] = True if enable_sf_input == 'true' else (False if enable_sf_input == 'false' else new_config.get('enable_secure_fib',True))

    secure_fib_price_input = (input(f"{AnsiColors.BLUE}Harga Cek Secure FIB (Close/High) [{new_config.get('secure_fib_check_price','Close')}]: {AnsiColors.ENDC}").strip() or new_config.get('secure_fib_check_price','Close')).capitalize()
    if secure_fib_price_input in ["Close", "High"]: new_config["secure_fib_check_price"] = secure_fib_price_input
    else: print(f"{AnsiColors.RED}Pilihan harga Secure FIB tidak valid. Menggunakan default: {new_config.get('secure_fib_check_price','Close')}{AnsiColors.ENDC}");


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
        api_s = current_settings.get("api_settings", {}) 
        primary_key_display = api_s.get('primary_key', 'BELUM DIATUR')
        if primary_key_display and len(primary_key_display) > 10 and primary_key_display not in ["YOUR_PRIMARY_KEY", "BELUM DIATUR"]:
            primary_key_display = primary_key_display[:5] + "..." + primary_key_display[-3:]

        recovery_keys = api_s.get('recovery_keys', [])
        num_recovery_keys = len([k for k in recovery_keys if k]) 
        termux_notif_status = "Aktif" if api_s.get("enable_termux_notifications", False) else "Nonaktif"

        pick_title_settings = "--- Menu Pengaturan Utama ---\n"
        pick_title_settings += f"Primary API Key: {primary_key_display}\n"
        pick_title_settings += f"Recovery API Keys: {num_recovery_keys} tersimpan\n"
        pick_title_settings += f"Notifikasi Termux: {termux_notif_status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Daftar Konfigurasi Crypto:\n"

        if not current_settings.get("cryptos"): 
            pick_title_settings += "  (Belum ada konfigurasi crypto)\n"
        else:
            for i, crypto_conf in enumerate(current_settings["cryptos"]):
                status = "Aktif" if crypto_conf.get('enabled', True) else "Nonaktif"
                pick_title_settings += f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')} ({crypto_conf.get('timeframe','N/A')}) - {status}\n"
        pick_title_settings += "------------------------------------\n"
        pick_title_settings += "Pilih tindakan:"

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
        
        action_choice = -1 
        try:
            _option_text, action_choice = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick: 
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings) 
            for idx, opt_text in enumerate(selectable_options): print(f"  {idx + 1}. {opt_text}")
            try:
                choice_input = input("Pilih nomor opsi: ").strip()
                if not choice_input: raise ValueError("Input kosong")
                choice = int(choice_input) - 1
                if 0 <= choice < len(selectable_options): action_choice = choice
                else: print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
            except ValueError: print(f"{AnsiColors.RED}Input harus berupa angka yang valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Memproses...") 
            if action_choice == -1: continue 

        try:
            clear_screen_animated()
            if action_choice == 0: 
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s 
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 1: 
                while True: 
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k] 
                    api_s['recovery_keys'] = current_recovery 

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
                    except Exception as e_pick_rec: 
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
                        if rec_index == -1: continue 

                    clear_screen_animated()
                    if rec_index == 0: 
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery 
                            save_settings(current_settings)
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else: print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 1: 
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
                    elif rec_index == 2: break 

            elif action_choice == 2: 
                animated_text_display("-- Pengaturan Email Global Notifikasi Sistem --", color=AnsiColors.HEADER)
                enable_g_email = input(f"Aktifkan notifikasi email global (API Key switch, dll)? (true/false) [{api_s.get('enable_global_email_notifications_for_key_switch',False)}]: ").lower().strip()
                api_s['enable_global_email_notifications_for_key_switch'] = True if enable_g_email == 'true' else (False if enable_g_email == 'false' else api_s.get('enable_global_email_notifications_for_key_switch',False))
                
                api_s['email_sender_address'] = (input(f"Email Pengirim Global [{api_s.get('email_sender_address','')}]: ").strip() or api_s.get('email_sender_address',''))
                api_s['email_sender_app_password'] = (input(f"App Password Pengirim Global [{api_s.get('email_sender_app_password','')}]: ").strip() or api_s.get('email_sender_app_password',''))
                api_s['email_receiver_address_admin'] = (input(f"Email Penerima Notifikasi Sistem (Admin) [{api_s.get('email_receiver_address_admin','')}]: ").strip() or api_s.get('email_receiver_address_admin',''))
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            
            elif action_choice == 3: 
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

            elif action_choice == 4: 
                new_crypto_conf = get_default_crypto_config() 
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) 
                current_settings.setdefault("cryptos", []).append(new_crypto_conf) 
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")

            elif action_choice == 5: 
                if not current_settings.get("cryptos"): 
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
                            current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice].copy())
                            save_settings(current_settings)
                            log_info(f"Konfigurasi # {idx_choice+1} diubah.")
                        else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                    except ValueError: print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")

            elif action_choice == 6: 
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
                            current_settings["cryptos"].pop(idx_choice) 
                            save_settings(current_settings)
                            log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                        else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                    except ValueError: print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            
            elif action_choice == 7: 
                break 

        except ValueError: 
            print(f"{AnsiColors.RED}Input tidak valid atau kosong.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e: 
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:") 
            show_spinner(1.5, "Error, kembali...")
    return current_settings 

# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name_log="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name_log)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None 
    api_endpoint = "histohour" 
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10 

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name_log)

    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT :
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)
        
        if current_to_ts is not None and candles_still_needed > 1 : 
            limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT) 


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
                log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call}, ToTs: {current_to_ts})", pair_name=pair_name_log)

            response = requests.get(url, params=params, timeout=20) 

            if response.status_code in [401, 403, 429]: 
                error_data = {}
                try: error_data = response.json()
                except json.JSONDecodeError: pass 
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name_log)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}") 

            response.raise_for_status() 
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'Unknown API Error')
                key_related_error_messages = [ 
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit" 
                ]
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name_log)
                    raise APIKeyError(f"JSON Error: {error_message}") 
                else: 
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name_log)
                    break 

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name_log)
                break 

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: 
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name_log)
                break

            batch_candles_list = []
            for item in raw_candles_from_api:
                if any(item.get(k) is None for k in ['time', 'open', 'high', 'low', 'close']):
                    log_warning(f"Skipping candle with missing OHLC or time: {item}", pair_name=pair_name_log)
                    continue
                candle = {
                    'timestamp': datetime.fromtimestamp(item['time']),
                    'open': item.get('open'), 'high': item.get('high'),
                    'low': item.get('low'), 'close': item.get('close'),
                    'volume': item.get('volumefrom', 0) 
                }
                batch_candles_list.append(candle)


            if not batch_candles_list and current_to_ts is not None : 
                if is_large_fetch: log_info("Batch menjadi kosong. Kemungkinan akhir data historis.", pair_name=pair_name_log)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles 

            if raw_candles_from_api: 
                current_to_ts = raw_candles_from_api[0]['time'] 
            else: 
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): 
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles', length=40)

            if len(raw_candles_from_api) < limit_for_this_api_call:
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name_log)
                break 

            if len(all_accumulated_candles) >= total_limit_desired: break 

            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name_log)
                time.sleep(0.3) 

        except APIKeyError: 
            raise 
        except requests.exceptions.RequestException as e: 
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name_log)
            break 
        except Exception as e: 
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name_log)
            log_exception("Traceback Error:", pair_name=pair_name_log) 
            break
    
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:] 

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT:
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name_log} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name_log)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, 
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, 
        "high_bar_index_for_fib": None, 
        "active_fib_level": None, 
        "active_fib_line_start_index": None, 
        "entry_price_custom": None, 
        "highest_price_for_trailing": None, 
        "trailing_tp_active_custom": False, 
        "current_trailing_stop_level": None, 
        "emergency_sl_level_custom": None, 
        "position_size": 0, 
        "last_op_timestamp": None, 
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list) 
    if len(series_list) < left_strength + right_strength + 1:
        return pivots

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue 

        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break 
            if is_high: 
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: 
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue 

        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break 
            if is_high: 
                if series_list[i] < series_list[i+j]: is_pivot = False; break 
            else: 
                if series_list[i] > series_list[i+j]: is_pivot = False; break 
        
        if is_pivot:
            pivots[i] = series_list[i] 
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings, current_pair_id_for_chart):
    log_pair_ctx = current_pair_id_for_chart 

    with live_chart_data_lock:
        if current_pair_id_for_chart in live_chart_data_store:
            current_annotations = live_chart_data_store[current_pair_id_for_chart]["annotations"]
            live_chart_data_store[current_pair_id_for_chart]["annotations"] = [
                ann for ann in current_annotations if ann['type'] == 'marker' 
            ]
            MAX_MARKERS_PER_CHART = 75 
            live_chart_data_store[current_pair_id_for_chart]["annotations"] = \
                live_chart_data_store[current_pair_id_for_chart]["annotations"][-MAX_MARKERS_PER_CHART:]
        else: 
            live_chart_data_store[current_pair_id_for_chart] = {"candles": collections.deque(maxlen=MAX_CHART_CANDLES_STORE), "annotations": [], "strategy_state_snapshot":{}}


    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
        return strategy_state 

    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]

    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    current_bar_index_in_list = len(candles_history) - 1 
    if current_bar_index_in_list < 0 : return strategy_state 

    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength
    
    if 0 <= idx_pivot_event_high < len(raw_pivot_highs) and raw_pivot_highs[idx_pivot_event_high] is not None:
        if strategy_state["last_signal_type"] != 1: 
            strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_highs[idx_pivot_event_high]
            strategy_state["last_signal_type"] = 1 
            pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
            strategy_state["last_op_timestamp"] = pivot_timestamp.timestamp() 

            log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': pivot_timestamp.timestamp(),
                        'price': strategy_state['final_pivot_high_price_confirmed'],
                        'position': 'aboveBar', 'shape': 'arrowDown', 'color': '#FF5252', 
                        'text': f"PH {strategy_state['final_pivot_high_price_confirmed']:.2f}"
                    })

    if 0 <= idx_pivot_event_low < len(raw_pivot_lows) and raw_pivot_lows[idx_pivot_event_low] is not None:
        if strategy_state["last_signal_type"] != -1:
            strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_lows[idx_pivot_event_low]
            strategy_state["last_signal_type"] = -1 
            pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
            strategy_state["last_op_timestamp"] = pivot_timestamp.timestamp()

            log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            with live_chart_data_lock:
                 if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': pivot_timestamp.timestamp(),
                        'price': strategy_state['final_pivot_low_price_confirmed'],
                        'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', 
                        'text': f"PL {strategy_state['final_pivot_low_price_confirmed']:.2f}"
                    })

    current_candle = candles_history[current_bar_index_in_list]
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close', 'timestamp']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=log_pair_ctx)
        return strategy_state

    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high 
        
        if strategy_state["active_fib_level"] is not None:
            log_debug(f"Resetting active FIB {strategy_state['active_fib_level']:.5f} due to new High Pivot.", pair_name=log_pair_ctx)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and strategy_state["high_bar_index_for_fib"] is not None:
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low 

            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                if strategy_state["high_price_for_fib"] is None or current_low_price is None: 
                    log_warning("Harga untuk kalkulasi FIB tidak valid (None).", pair_name=log_pair_ctx)
                else:
                    calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0

                    is_fib_late = False
                    if crypto_config["enable_secure_fib"]:
                        price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                        if price_val_current_candle is not None and calculated_fib_level is not None and price_val_current_candle > calculated_fib_level:
                            is_fib_late = True

                    if is_fib_late:
                        log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
                        strategy_state["active_fib_level"] = None 
                        strategy_state["active_fib_line_start_index"] = None
                    elif calculated_fib_level is not None : 
                        log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=log_pair_ctx)
                        strategy_state["active_fib_level"] = calculated_fib_level
                        strategy_state["active_fib_line_start_index"] = current_low_bar_index
                        strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()
                        
                        with live_chart_data_lock:
                            if current_pair_id_for_chart in live_chart_data_store:
                                live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                                    'type': 'price_line', 'id': 'fib_0.5_level', 
                                    'price': strategy_state["active_fib_level"],
                                    'color': '#2962FF', 'lineWidth': 1, 'lineStyle': 2, 
                                    'title': f'FIB 0.5: {strategy_state["active_fib_level"]:.5f}'
                                })
            
            strategy_state["high_price_for_fib"] = None
            strategy_state["high_bar_index_for_fib"] = None

    if strategy_state["active_fib_level"] is not None and \
       strategy_state["active_fib_line_start_index"] is not None and \
       strategy_state["position_size"] == 0:

        if current_candle.get('close') is None or current_candle.get('open') is None:
            log_warning("Nilai close atau open tidak ada di candle saat ini. Skip entry check.", pair_name=log_pair_ctx)
            return strategy_state 

        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            strategy_state["position_size"] = 1 
            entry_px = current_candle['close'] 
            strategy_state["entry_price_custom"] = entry_px
            strategy_state["highest_price_for_trailing"] = entry_px 
            strategy_state["trailing_tp_active_custom"] = False 
            strategy_state["current_trailing_stop_level"] = None 

            emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
            strategy_state["emergency_sl_level_custom"] = emerg_sl
            strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()

            log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
            log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            play_notification_sound() 
            
            termux_title = f"BUY Signal: {log_pair_ctx}"
            termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}"
            send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=log_pair_ctx)

            email_subject = f"BUY Signal: {log_pair_ctx}"
            email_body = (f"New BUY signal for {log_pair_ctx} on {crypto_config['exchange']}.\n\n"
                          f"Entry Price: {entry_px:.5f}\nFIB Level: {strategy_state['active_fib_level']:.5f}\n"
                          f"Emergency SL: {emerg_sl:.5f}\nTime: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, {**crypto_config, 'pair_name': log_pair_ctx}) 

            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': current_candle['timestamp'].timestamp(),
                        'price': entry_px, 'position': 'belowBar', 'shape': 'arrowUp', 
                        'color': '#26A69A', 'text': f'BUY\n{entry_px:.2f}'
                    })
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'price_line', 'id': 'sl_level', 
                        'price': emerg_sl, 'color': '#EF5350', 'lineWidth': 2,
                        'title': f'Emerg SL: {emerg_sl:.5f}'
                    })
            
            strategy_state["active_fib_level"] = None 
            strategy_state["active_fib_line_start_index"] = None

    if strategy_state["position_size"] > 0:
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing is None or current_candle.get('high') is None:
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None).", pair_name=log_pair_ctx)
        else: 
            strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None: profit_percent = 0.0
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=log_pair_ctx)
                strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()


        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=log_pair_ctx)
                strategy_state["last_op_timestamp"] = current_candle['timestamp'].timestamp()


        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color_ann = AnsiColors.RED 
        sl_chart_color = '#EF5350' 
        sl_type_for_chart = "Emerg SL"

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color_ann = AnsiColors.BLUE
                sl_chart_color = '#2962FF' 
                sl_type_for_chart = "Trail SL"
        
        if final_stop_for_exit is not None:
            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'price_line', 'id': 'sl_level', 
                        'price': final_stop_for_exit, 'color': sl_chart_color, 'lineWidth': 2,
                        'title': f'{sl_type_for_chart}: {final_stop_for_exit:.5f}'
                    })

        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            exit_price_open = current_candle.get('open')
            if exit_price_open is None: 
                log_warning("Harga open candle tidak ada untuk exit. Menggunakan SL sebagai harga exit.", pair_name=log_pair_ctx)
                exit_price = final_stop_for_exit 
            else: 
                exit_price = min(exit_price_open, final_stop_for_exit) 
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            if exit_comment == "Trailing Stop" and pnl < 0: 
                exit_color_ann = AnsiColors.RED 

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

            with live_chart_data_lock:
                if current_pair_id_for_chart in live_chart_data_store:
                    live_chart_data_store[current_pair_id_for_chart]["annotations"].append({
                        'type': 'marker', 'time': current_candle['timestamp'].timestamp(),
                        'price': exit_price, 
                        'position': 'aboveBar' if pnl >=0 else 'belowBar', 
                        'shape': 'arrowDown', 'color': '#FFCA28', 
                        'text': f'EXIT\n{exit_price:.2f}\nPnL:{pnl:.1f}%'
                    })

            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
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
        log_debug(f"Posisi Aktif. Entry: {entry_price_display:.5f}, SL Saat Ini: {sl_display_str}", pair_name=log_pair_ctx)

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
                "active_fib": strategy_state.get("active_fib_level"), 
                "last_op_ts": strategy_state.get("last_op_timestamp")
            }
            if strategy_state.get("active_fib_level") and strategy_state["position_size"] == 0:
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
        api_settings 
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


    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = current_key_display_val[:5] + "..." + current_key_display_val[-3:] if len(current_key_display_val) > 8 else current_key_display_val
    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM_INIT")

    crypto_data_manager = {} 
    for config in all_crypto_configs:
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}"
        config['pair_id_for_logic'] = pair_id 
        log_pair_ctx = pair_id 

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{log_pair_ctx}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [], 
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, 
            "big_data_email_sent": False, 
            "last_candle_fetch_time": datetime.min, 
            "data_fetch_failed_consecutively": 0 
        }
        
        with live_chart_data_lock:
            live_chart_data_store[pair_id] = {
                "candles": collections.deque(maxlen=MAX_CHART_CANDLES_STORE), 
                "annotations": [],
                "strategy_state_snapshot": {} 
            }

        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: 
                log_error(f"BIG DATA: Semua API key habis saat mengambil data awal untuk {log_pair_ctx}.", pair_name=log_pair_ctx)
                break 

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=log_pair_ctx)
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name_log=log_pair_ctx 
                )
                initial_fetch_successful = True 
            except APIKeyError: 
                log_warning(f"BIG DATA: API Key gagal untuk {log_pair_ctx}. Mencoba key berikutnya.", pair_name=log_pair_ctx)
                if not api_key_manager.switch_to_next_key(): break 
                retries_done_initial +=1 
            except requests.exceptions.RequestException as e: 
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {log_pair_ctx}: {e}. Tidak mengganti key.", pair_name=log_pair_ctx)
                break 
            except Exception as e_gen: 
                log_error(f"BIG DATA: Error umum saat mengambil data awal {log_pair_ctx}: {e_gen}. Tidak mengganti key.", pair_name=log_pair_ctx)
                log_exception("Traceback Error Initial Fetch:", pair_name=log_pair_ctx)
                break

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {log_pair_ctx} setelah semua upaya. Pair ini mungkin tidak diproses.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False 
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() 
            continue 

        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima untuk {log_pair_ctx}.", pair_name=log_pair_ctx)

        with live_chart_data_lock:
            chart_candles_deque = live_chart_data_store[pair_id]["candles"]
            chart_candles_deque.clear()
            # Add all initial candles to the chart's deque (up to MAX_CHART_CANDLES_STORE)
            # The chart will decide how many to display via MAX_CHART_CANDLES_DISPLAY or fitContent
            candles_for_chart_init = initial_candles[-MAX_CHART_CANDLES_STORE:]
            for c in candles_for_chart_init: 
                chart_candles_deque.append({
                    "time": c['timestamp'].timestamp(), "open": c['open'], "high": c['high'],
                    "low": c['low'], "close": c['close']
                })
            log_debug(f"Added {len(chart_candles_deque)} initial candles to chart store for {pair_id}", pair_name="CHART_INIT")
        
        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1 
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state ({log_pair_ctx})...", pair_name=log_pair_ctx)
                temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): 
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue 

                    temp_state_for_warmup["position_size"] = 0 
                    temp_state_for_warmup = run_strategy_logic(historical_slice, config, temp_state_for_warmup, global_settings_dict, pair_id)
                
                crypto_data_manager[pair_id]["strategy_state"] = {
                    **temp_state_for_warmup, 
                    **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                       "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                       "current_trailing_stop_level":None} 
                }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai untuk {log_pair_ctx}.{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            else: 
                log_warning(f"Data awal ({len(initial_candles)}) untuk {log_pair_ctx} tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=log_pair_ctx)
        else: 
            log_warning(f"Tidak ada data awal untuk warm-up {log_pair_ctx}.", pair_name=log_pair_ctx)

        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {log_pair_ctx}!{AnsiColors.ENDC}", pair_name=log_pair_ctx)
            if not crypto_data_manager[pair_id]["big_data_email_sent"]:
                send_email_notification(f"Data Downloading Complete: {log_pair_ctx}", 
                                        f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {log_pair_ctx}.", 
                                        {**config, 'pair_name': log_pair_ctx})
                crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) untuk {log_pair_ctx} ----------{AnsiColors.ENDC}", pair_name=log_pair_ctx)
        
        # After each pair is initialized, explicitly run logic once to populate chart annotations based on full initial data
        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= (config.get('left_strength', 50) + config.get('right_strength', 150) + 1):
            log_info(f"Running initial logic pass for {log_pair_ctx} to populate chart.", pair_name=log_pair_ctx)
            crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(
                crypto_data_manager[pair_id]["all_candles_list"], 
                config, 
                crypto_data_manager[pair_id]["strategy_state"], 
                global_settings_dict, 
                pair_id
            )


    animated_text_display(f"{AnsiColors.HEADER}----------------- SEMUA PAIR DIINISIALISASI -----------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf') 
            any_data_fetched_this_cycle = False

            for pair_id_loop, data_loop in crypto_data_manager.items(): 
                config_loop = data_loop["config"]
                log_pair_ctx_loop = config_loop['pair_id_for_logic'] 

                if data_loop.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : 
                    if (datetime.now() - data_loop.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: 
                        log_debug(f"Pair {log_pair_ctx_loop} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=log_pair_ctx_loop)
                        min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, 3600) 
                        continue 
                    else: 
                        data_loop["data_fetch_failed_consecutively"] = 0 
                        log_info(f"Cooldown 1 jam untuk {log_pair_ctx_loop} selesai. Mencoba fetch lagi.", pair_name=log_pair_ctx_loop)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data_loop["last_candle_fetch_time"]).total_seconds()

                required_interval_for_this_pair = 0
                if data_loop["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    if config_loop.get('timeframe') == "minute": required_interval_for_this_pair = 55 
                    elif config_loop.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 
                    else: required_interval_for_this_pair = 3580 
                else: 
                    required_interval_for_this_pair = config_loop.get('refresh_interval_seconds', 60) 

                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue 

                log_info(f"Memproses {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                data_loop["last_candle_fetch_time"] = current_loop_time 
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
                    if not current_api_key_for_attempt: 
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                        break 

                    limit_fetch = 3 
                    if data_loop["big_data_collection_phase_active"]:
                        limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data_loop["all_candles_list"])
                        if limit_fetch_needed <=0 : 
                             fetch_update_successful_for_this_pair = True 
                             new_candles_batch = [] 
                             break
                        limit_fetch = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) 
                        limit_fetch = max(limit_fetch, 1) 

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()}) untuk {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                    try:
                        new_candles_batch = fetch_candles(
                            config_loop['symbol'], config_loop['currency'], limit_fetch, 
                            config_loop['exchange'], current_api_key_for_attempt, config_loop['timeframe'],
                            pair_name_log=log_pair_ctx_loop
                        )
                        fetch_update_successful_for_this_pair = True 
                        data_loop["data_fetch_failed_consecutively"] = 0 
                        any_data_fetched_this_cycle = True 
                    
                    except APIKeyError: 
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {log_pair_ctx_loop}. Mencoba key berikutnya.", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        if not api_key_manager.switch_to_next_key(): 
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                            break 
                        retries_done_update += 1 
                    except requests.exceptions.RequestException as e: 
                        log_error(f"Error jaringan saat mengambil update {log_pair_ctx_loop}: {e}. Tidak mengganti key.", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        break 
                    except Exception as e_gen_update: 
                        log_error(f"Error umum saat mengambil update {log_pair_ctx_loop}: {e_gen_update}. Tidak mengganti key.", pair_name=log_pair_ctx_loop)
                        log_exception("Traceback Error Update Fetch:", pair_name=log_pair_ctx_loop)
                        data_loop["data_fetch_failed_consecutively"] = data_loop.get("data_fetch_failed_consecutively", 0) + 1
                        break

                if data_loop.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data_loop["last_attempt_after_all_keys_failed"] = datetime.now() 
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {log_pair_ctx_loop}. Akan masuk cooldown.", pair_name=log_pair_ctx_loop)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data_loop["big_data_collection_phase_active"]:
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {log_pair_ctx_loop} (fetch berhasil tapi kosong).{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {log_pair_ctx_loop} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) 
                    continue 

                merged_candles_dict = {c['timestamp']: c for c in data_loop["all_candles_list"]}
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
                data_loop["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data_loop['all_candles_list'])} untuk {log_pair_ctx_loop}.", pair_name=log_pair_ctx_loop)
                elif new_candles_batch : 
                     log_info(f"Tidak ada candle dengan timestamp baru atau update konten untuk {log_pair_ctx_loop}. Data terakhir mungkin identik.", pair_name=log_pair_ctx_loop)

                with live_chart_data_lock:
                    if log_pair_ctx_loop in live_chart_data_store:
                        chart_deque = live_chart_data_store[log_pair_ctx_loop]["candles"]
                        chart_deque.clear()
                        # Populate with the latest N candles from the full history for logic
                        candles_for_chart_update = data_loop["all_candles_list"][-MAX_CHART_CANDLES_STORE:]
                        for c_obj in candles_for_chart_update: 
                            chart_deque.append({
                                "time": c_obj['timestamp'].timestamp(), "open": c_obj['open'], 
                                "high": c_obj['high'], "low": c_obj['low'], "close": c_obj['close']
                            })


                if data_loop["big_data_collection_phase_active"]:
                    if len(data_loop["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {log_pair_ctx_loop}!{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                        if len(data_loop["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                            data_loop["all_candles_list"] = data_loop["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data_loop["big_data_email_sent"]: 
                            send_email_notification(f"Data Downloading Complete: {log_pair_ctx_loop}", 
                                                    f"Data downloading for {TARGET_BIG_DATA_CANDLES} candles complete! Now trading on {log_pair_ctx_loop}.", 
                                                    {**config_loop, 'pair_name': log_pair_ctx_loop})
                            data_loop["big_data_email_sent"] = True
                        
                        data_loop["big_data_collection_phase_active"] = False 
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data_loop['all_candles_list'])} candles) untuk {log_pair_ctx_loop} ----------{AnsiColors.ENDC}", pair_name=log_pair_ctx_loop)
                else: 
                    if len(data_loop["all_candles_list"]) > TARGET_BIG_DATA_CANDLES + 50: 
                        data_loop["all_candles_list"] = data_loop["all_candles_list"][-(TARGET_BIG_DATA_CANDLES + 50):]

                min_len_for_pivots_logic = config_loop.get('left_strength',50) + config_loop.get('right_strength',150) + 1
                if len(data_loop["all_candles_list"]) >= min_len_for_pivots_logic:
                    process_logic_now = (actual_new_or_updated_count > 0 or
                                         (not data_loop["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data_loop["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         (data_loop["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) )

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data_loop['all_candles_list'])} candle untuk {log_pair_ctx_loop}...", pair_name=log_pair_ctx_loop)
                         data_loop["strategy_state"] = run_strategy_logic(data_loop["all_candles_list"], config_loop, data_loop["strategy_state"], global_settings_dict, log_pair_ctx_loop) 
                    elif not data_loop["big_data_collection_phase_active"]: 
                         last_c_time_str = data_loop["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data_loop["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {log_pair_ctx_loop}. Data terakhir @ {last_c_time_str}.", pair_name=log_pair_ctx_loop)
                else: 
                    log_info(f"Data ({len(data_loop['all_candles_list'])}) untuk {log_pair_ctx_loop} belum cukup utk analisa (min: {min_len_for_pivots_logic}).", pair_name=log_pair_ctx_loop)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair) 
            
            sleep_duration = 15 

            if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None: 
                log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM_LOOP")
                sleep_duration = 3600 
            elif active_cryptos_still_in_big_data_collection > 0: 
                min_big_data_interval = float('inf')
                for pid_bd, pdata_bd in crypto_data_manager.items(): 
                    if pdata_bd["big_data_collection_phase_active"]:
                        pconfig_bd = pdata_bd["config"]
                        interval_bd_calc = 55 if pconfig_bd.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_bd.get('timeframe') == "day" else 3580)
                        min_big_data_interval = min(min_big_data_interval, interval_bd_calc)
                
                sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) 
                log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM_LOOP")
            else: 
                if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                    log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM_LOOP")
                else: 
                    default_refresh_from_config = 60 
                    if all_crypto_configs : 
                        default_refresh_from_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)
                    sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_config)
                    log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama).", pair_name="SYSTEM_LOOP")

            if sleep_duration > 0:
                show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
            else: 
                log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM_LOOP")
                time.sleep(1)

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e: 
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM_CRASH")
        log_exception("Traceback Error:", pair_name="SYSTEM_CRASH") 
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input()

# --- MENU UTAMA ---
def main_menu():
    global CHART_SERVER_THREAD 
    settings = load_settings()

    if not CHART_SERVER_THREAD or not CHART_SERVER_THREAD.is_alive():
        log_info("Starting chart server from main_menu...", pair_name="SYS_BOOT")
        start_chart_server_thread()
        time.sleep(1) 
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
        selected_index = -1 
        try:
            _option_text, selected_index = pick(options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception as e_pick_main: 
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
            if selected_index == -1: continue 

        if selected_index == 0: 
            start_trading(settings) 
        elif selected_index == 1: 
            settings = settings_menu(settings) 
        elif selected_index == 2: 
            log_info("Aplikasi ditutup.", pair_name="SYSTEM_EXIT")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
            show_spinner(0.5, "Exiting")
            stop_chart_server_thread() 
            break 

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
    except Exception as e:
        clear_screen_animated()
        print(f"{AnsiColors.RED}Terjadi error tak terduga di level utama: {e}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL UNHANDLED EXCEPTION:", pair_name="SYSTEM_CRITICAL")
        animated_text_display("Tekan Enter untuk keluar...", color=AnsiColors.RED, delay=0.01)
        input()
    finally:
        if HTTPD or (CHART_SERVER_THREAD and CHART_SERVER_THREAD.is_alive()):
            log_info("Final shutdown call for chart server.", pair_name="SYS_CLEANUP")
            stop_chart_server_thread()
