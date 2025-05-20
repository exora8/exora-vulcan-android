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

# --- LIBRARY BARU UNTUK FITUR CHART ---
import threading
from flask import Flask, jsonify, render_template_string, request
import plotly.graph_objects as go
from collections import deque # Baik untuk menyimpan data dengan ukuran tetap

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

# --- KONSTANTA SKRIP & CHART ---
SETTINGS_FILE = "settings_multiple_recovery.json"
CRYPTOCOMPARE_MAX_LIMIT = 1999
TARGET_BIG_DATA_CANDLES = 2500 # Sudah ada
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15 # Sudah ada

# NEW: CHART FEATURE Constants
CHART_WINDOW_SIZE = 200  # Jumlah candle yang ditampilkan di chart
LIVE_CHART_DATA = {} # { 'BTC-USD': { 'candles': deque(), 'fib': None, 'buy': None, 'sl': None, 'pivots': {'high': [], 'low': []}, 'last_update': datetime }, ... }
FLASK_PORT = 5001 # Port untuk web server chart

# --- FUNGSI UNTUK FITUR CHART ---
# NEW: CHART FEATURE - Data store & update function
def update_live_chart_data(pair_name_chart, candles_history_chart, strategy_state_chart, config_chart):
    global LIVE_CHART_DATA
    if pair_name_chart not in LIVE_CHART_DATA:
        LIVE_CHART_DATA[pair_name_chart] = {
            'candles': deque(maxlen=CHART_WINDOW_SIZE),
            'fib_level': None,
            'buy_price': None,
            'sl_price': None,
            'pivots_high': [],
            'pivots_low': [],
            'last_update': datetime.now()
        }

    # 1. Update Candles
    start_index = max(0, len(candles_history_chart) - CHART_WINDOW_SIZE)
    candles_to_display = candles_history_chart[start_index:]
    
    LIVE_CHART_DATA[pair_name_chart]['candles'].clear()
    for c in candles_to_display:
        if c.get('timestamp') and c.get('open') is not None and c.get('high') is not None and \
           c.get('low') is not None and c.get('close') is not None:
            LIVE_CHART_DATA[pair_name_chart]['candles'].append({
                'time': c['timestamp'].timestamp() * 1000, # Plotly JS butuh ms
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close']
            })

    # 2. Update FIB Level
    if strategy_state_chart.get("active_fib_level") is not None:
        LIVE_CHART_DATA[pair_name_chart]['fib_level'] = strategy_state_chart["active_fib_level"]
    elif strategy_state_chart.get("position_size", 0) == 0:
        LIVE_CHART_DATA[pair_name_chart]['fib_level'] = None

    # 3. Update Buy and SL Prices
    if strategy_state_chart.get("position_size", 0) > 0:
        LIVE_CHART_DATA[pair_name_chart]['buy_price'] = strategy_state_chart.get("entry_price_custom")
        
        plot_stop_level = strategy_state_chart.get("emergency_sl_level_custom")
        if strategy_state_chart.get("trailing_tp_active_custom") and strategy_state_chart.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state_chart.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state_chart.get("current_trailing_stop_level")
        LIVE_CHART_DATA[pair_name_chart]['sl_price'] = plot_stop_level
    else:
        LIVE_CHART_DATA[pair_name_chart]['buy_price'] = None
        LIVE_CHART_DATA[pair_name_chart]['sl_price'] = None

    # 4. Update Pivots (Placeholder - bisa dikembangkan)
    LIVE_CHART_DATA[pair_name_chart]['pivots_high'] = [] 
    LIVE_CHART_DATA[pair_name_chart]['pivots_low'] = []

    LIVE_CHART_DATA[pair_name_chart]['last_update'] = datetime.now()
    # log_debug(f"Chart data for {pair_name_chart} updated. Candles: {len(LIVE_CHART_DATA[pair_name_chart]['candles'])}", pair_name=pair_name_chart)


# NEW: CHART FEATURE - Flask App
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crypto Charts</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #2c2c2c; color: #e0e0e0; }
        h1 { text-align: center; color: #e0e0e0; }
        ul { list-style-type: none; padding: 0; }
        li { margin: 10px 0; }
        a { 
            text-decoration: none; 
            color: #58a6ff; 
            padding: 10px 15px; 
            border: 1px solid #58a6ff; 
            border-radius: 5px; 
            display: block; 
            text-align: center;
            transition: background-color 0.3s, color 0.3s;
        }
        a:hover { background-color: #58a6ff; color: #2c2c2c; }
        .container { max-width: 600px; margin: auto; background: #3c3c3c; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.5); }
        .note {text-align:center; margin-top:20px; font-size: 0.9em; color: #aaa;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Available Live Charts</h1>
        {% if pairs %}
        <ul>
            {% for pair in pairs %}
            <li><a href="{{ url_for('show_chart_page', pair_name_url=pair.replace('-', '_')) }}">{{ pair }}</a></li>
            {% endfor %}
        </ul>
        {% else %}
        <p class="note">No active trading pairs found or data not yet available. Please wait for the script to initialize and fetch data.</p>
        {% endif %}
        <p class="note">Charts will be available after initial data fetch for each pair. Refresh this page if a pair is missing.</p>
    </div>
</body>
</html>
"""

CHART_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Chart: {{ pair_name }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #1e1e1e; color: #d4d4d4; }
        #chartContainer { width: 98%; height: 80vh; margin: 10px auto; }
        h1 { text-align: center; padding: 10px; color: #d4d4d4; margin-bottom: 5px; }
        .info-bar { text-align: center; padding: 5px; font-size: 0.9em; color: #aaa; }
        .loader {
            border: 8px solid #333; /* Darker grey */
            border-top: 8px solid #58a6ff; /* Blue */
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 50px auto; /* More margin for loader */
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .message { text-align:center; margin-top: 50px; font-size: 1.1em; color: #aaa;}
    </style>
</head>
<body>
    <h1>Live Chart: {{ pair_name }}</h1>
    <div class="info-bar">Server Data Last Updated: <span id="lastServerUpdate">-</span> | Chart Last Refreshed: <span id="lastChartRefresh">-</span></div>
    <div id="chartContainer"><div class="loader"></div></div>

    <script>
        const pairNameUrl = "{{ pair_name_url }}";
        const chartContainer = document.getElementById('chartContainer');
        let chartInitialized = false;

        async function fetchData() {
            try {
                const response = await fetch(`/chart_data/${pairNameUrl}`);
                if (!response.ok) {
                    console.error('Error fetching chart data:', response.status, await response.text());
                    if(chartContainer.querySelector('.loader')) chartContainer.innerHTML = '<p class="message">Error loading data. Is the trading script running and the pair active?</p>';
                    return null;
                }
                const data = await response.json();
                if (data.last_update) {
                    document.getElementById('lastServerUpdate').textContent = new Date(data.last_update).toLocaleString();
                }
                return data;
            } catch (error) {
                console.error('Failed to fetch:', error);
                if(chartContainer.querySelector('.loader')) chartContainer.innerHTML = '<p class="message">Could not connect to data source. Ensure the script is running.</p>';
                return null;
            }
        }

        function plotChart(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                if(chartContainer.querySelector('.loader')) chartContainer.innerHTML = '<p class="message">No candle data available yet for {{ pair_name }}. Waiting for data...</p>';
                console.warn("No candle data to plot for {{ pair_name }}");
                return;
            }
            
            if(chartContainer.querySelector('.loader')) {
                chartContainer.innerHTML = ''; // Remove loader only if it exists
            }


            const traceCandle = {
                x: data.candles.map(c => new Date(c.time)),
                open: data.candles.map(c => c.open),
                high: data.candles.map(c => c.high),
                low: data.candles.map(c => c.low),
                close: data.candles.map(c => c.close),
                type: 'candlestick',
                name: 'Candles',
                xaxis: 'x', yaxis: 'y'
            };

            const layout = {
                title: { 
                    text: `{{ pair_name }} Live Chart (Candles: ${data.candles.length})`,
                    font: { color: '#d4d4d4' }
                },
                xaxis: {
                    type: 'date',
                    rangeslider: { visible: false },
                    color: '#aaa',
                    gridcolor: '#444'
                },
                yaxis: {
                    autorange: true, 
                    type: 'linear',
                    color: '#aaa',
                    gridcolor: '#444',
                    tickformat: '.8f' // Adjust precision as needed
                },
                shapes: [],
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                font: { color: '#d4d4d4' },
                legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: {color: '#d4d4d4'} }
            };

            const plotDataArray = [traceCandle]; // Renamed to avoid conflict with function parameter `data`

            if (data.fib_level) {
                layout.shapes.push({
                    type: 'line', x0: traceCandle.x[0], y0: data.fib_level,
                    x1: traceCandle.x[traceCandle.x.length - 1], y1: data.fib_level,
                    line: { color: 'rgba(255, 165, 0, 0.7)', width: 2, dash: 'dash' },
                    name: 'FIB 0.5'
                });
                 plotDataArray.push({ // Trace for legend
                    x: [null], y: [null], mode: 'lines', name: `FIB 0.5 (${data.fib_level.toFixed(5)})`,
                    line: { color: 'rgba(255, 165, 0, 0.7)', width: 2, dash: 'dash' }
                });
            }
            if (data.buy_price) {
                layout.shapes.push({
                    type: 'line', x0: traceCandle.x[0], y0: data.buy_price,
                    x1: traceCandle.x[traceCandle.x.length - 1], y1: data.buy_price,
                    line: { color: 'rgba(0, 255, 0, 0.7)', width: 2 },
                    name: 'Buy Price'
                });
                plotDataArray.push({
                    x: [null], y: [null], mode: 'lines', name: `BUY (${data.buy_price.toFixed(5)})`,
                    line: { color: 'rgba(0, 255, 0, 0.7)', width: 2 }
                });
            }
            if (data.sl_price) {
                layout.shapes.push({
                    type: 'line', x0: traceCandle.x[0], y0: data.sl_price,
                    x1: traceCandle.x[traceCandle.x.length - 1], y1: data.sl_price,
                    line: { color: 'rgba(255, 0, 0, 0.7)', width: 2 },
                    name: 'SL Price'
                });
                plotDataArray.push({
                    x: [null], y: [null], mode: 'lines', name: `SL (${data.sl_price.toFixed(5)})`,
                    line: { color: 'rgba(255, 0, 0, 0.7)', width: 2 }
                });
            }

            if (!chartInitialized) {
                Plotly.newPlot('chartContainer', plotDataArray, layout, {responsive: true, displaylogo: false});
                chartInitialized = true;
            } else {
                Plotly.react('chartContainer', plotDataArray, layout, {responsive: true, displaylogo: false});
            }
            document.getElementById('lastChartRefresh').textContent = new Date().toLocaleString();
        }

        async function updateLoop() {
            const chartData = await fetchData(); // Renamed to avoid conflict
            if (chartData) {
                plotChart(chartData);
            }
            
            let refreshInterval = 15000; 
            if (chartData && chartData.last_update) {
                const age_ms = new Date().getTime() - new Date(chartData.last_update).getTime();
                if (age_ms < 20000) refreshInterval = 3000;      // Very recent data (within 20s), refresh every 3s
                else if (age_ms < 60000) refreshInterval = 5000; // Recent data (within 1min), refresh every 5s
                else if (age_ms < 300000) refreshInterval = 10000; // Data within 5 mins, refresh every 10s
            }
            setTimeout(updateLoop, refreshInterval);
        }
        
        // Initial call with a slight delay to allow page elements to settle
        setTimeout(updateLoop, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    global LIVE_CHART_DATA
    active_pairs_with_data = sorted([
        pair for pair, data in LIVE_CHART_DATA.items() 
        if data.get('candles') and len(data['candles']) > 0
    ])
    return render_template_string(INDEX_HTML, pairs=active_pairs_with_data)

@app.route('/chart/<pair_name_url>')
def show_chart_page(pair_name_url):
    actual_pair_name_candidate = pair_name_url.replace('_', '-')
    
    found_key = None
    if actual_pair_name_candidate in LIVE_CHART_DATA:
        found_key = actual_pair_name_candidate
    else: # Try a case-insensitive match if direct fails
        for key in LIVE_CHART_DATA.keys():
            if key.lower() == actual_pair_name_candidate.lower():
                found_key = key
                break
    
    if not found_key:
        return f"Chart data not found for {actual_pair_name_candidate}. Ensure the pair is configured, running, and has fetched initial data.", 404

    return render_template_string(CHART_PAGE_HTML, pair_name=found_key, pair_name_url=found_key.replace('-', '_'))


@app.route('/chart_data/<pair_name_url>')
def get_chart_data(pair_name_url):
    global LIVE_CHART_DATA
    actual_pair_name_candidate = pair_name_url.replace('_', '-')

    found_key = None
    if actual_pair_name_candidate in LIVE_CHART_DATA:
        found_key = actual_pair_name_candidate
    else: # Try a case-insensitive match
        for key in LIVE_CHART_DATA.keys():
            if key.lower() == actual_pair_name_candidate.lower():
                found_key = key
                break

    if not found_key or not LIVE_CHART_DATA.get(found_key):
        return jsonify({"error": f"Pair data not found for {actual_pair_name_candidate}", "candles": []}), 404

    pair_data = LIVE_CHART_DATA[found_key]
    return jsonify({
        "candles": list(pair_data['candles']),
        "fib_level": pair_data.get('fib_level'),
        "buy_price": pair_data.get('buy_price'),
        "sl_price": pair_data.get('sl_price'),
        "pivots_high": pair_data.get('pivots_high', []),
        "pivots_low": pair_data.get('pivots_low', []),
        "last_update": pair_data.get('last_update').isoformat() if pair_data.get('last_update') else None
    })

def run_flask_app():
    try:
        log_info(f"Starting live chart web server on http://0.0.0.0:{FLASK_PORT}", "WEBSERVER")
        # Setting use_reloader=False is important if starting Flask in a thread from another script
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)
    except OSError as e:
        if e.errno == 98: # Address already in use
            log_error(f"Web server port {FLASK_PORT} is already in use. Chart server cannot start.", "WEBSERVER")
        else:
            log_error(f"Failed to start web server: {e}", "WEBSERVER")
    except Exception as e:
        log_error(f"An unexpected error occurred with the web server: {e}", "WEBSERVER")
        log_exception("Traceback Web Server Error:", "WEBSERVER")

# --- FUNGSI CLEAR SCREEN ---
def clear_screen_animated():
    show_spinner(0.1, "Clearing screen")
    os.system('cls' if os.name == 'nt' else 'clear')

# --- API KEY MANAGER ---
class APIKeyManager:
    def __init__(self, primary_key, recovery_keys_list, global_settings_for_email=None): #init diubah ke __init__
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
                email_body = (f"Skrip trading telah secara otomatis mengganti API key CryptoCompare.\n\n"
                              f"API Key sebelumnya mungkin telah mencapai limit atau tidak valid.\n"
                              f"Sekarang menggunakan API key dengan index: {self.current_index}\n"
                              f"Key: ...{new_key_display[-8:] if len(new_key_display) > 8 else new_key_display} (bagian akhir ditampilkan untuk identifikasi)\n\n"
                              f"Harap periksa status API key Anda di CryptoCompare.")
                dummy_email_cfg = {
                    "enable_email_notifications": True, 
                    "email_sender_address": self.global_email_settings.get("email_sender_address"),
                    "email_sender_app_password": self.global_email_settings.get("email_sender_app_password"),
                    "email_receiver_address": self.global_email_settings.get("email_receiver_address_admin", self.global_email_settings.get("email_receiver_address"))
                }
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi pergantian API key (APIKeyManager).")
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
                if all(dummy_email_cfg.get(k) for k in ["email_sender_address", "email_sender_app_password", "email_receiver_address"]):
                    send_email_notification(email_subject, email_body, dummy_email_cfg)
                else:
                    log_warning("Konfigurasi email global tidak lengkap untuk notifikasi KRITIS semua API key gagal (APIKeyManager).")
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
            print('\a', end='', flush=True) # Standard bell for POSIX
    except Exception as e:
        log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False):
        return

    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")

    if not all([sender_email, sender_password, receiver_email]):
        pair_name_ctx = settings_for_email.get('pair_name', 
                                             settings_for_email.get('symbol', 'GLOBAL_EMAIL')) 
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
        # Menggunakan check=False agar tidak error jika termux-notification tidak ada,
        # tapi tetap mencoba menjalankannya.
        # Lebih baik menangkap FileNotFoundError secara eksplisit.
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg], 
                       check=False, # Diubah ke False untuk penanganan error yang lebih baik
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE) # stdout/stderr untuk debug jika perlu
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
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
        "enable_termux_notifications": False  # Default untuk Termux
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            try:
                settings = json.load(f)
                # Ensure api_settings exists and has all default keys
                if "api_settings" not in settings:
                    settings["api_settings"] = default_api_settings.copy()
                else:
                    for k, v in default_api_settings.items():
                        if k not in settings["api_settings"]:
                            settings["api_settings"][k] = v
                
                if "cryptos" not in settings or not isinstance(settings["cryptos"], list):
                    settings["cryptos"] = []
                for crypto_cfg in settings["cryptos"]: # Ensure existing cryptos have id and enabled
                    if "id" not in crypto_cfg: crypto_cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in crypto_cfg: crypto_cfg["enabled"] = True # Default to true if missing
                return settings
            except json.JSONDecodeError:
                log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default atau membuat file baru.")
                return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api_settings.copy(), "cryptos": [get_default_crypto_config()]}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")


def _prompt_crypto_config(current_config): # Underscore to indicate helper
    clear_screen_animated()
    new_config = current_config.copy() # Work on a copy
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)

    # Enabled status
    enabled_input = input(f"Aktifkan analisa untuk pair ini? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))

    new_config["symbol"] = (input(f"{AnsiColors.BLUE}Simbol Crypto Dasar (misal BTC) [{new_config.get('symbol','BTC')}]: {AnsiColors.ENDC}") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"{AnsiColors.BLUE}Simbol Mata Uang Quote (misal USDT, USD) [{new_config.get('currency','USD')}]: {AnsiColors.ENDC}") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"{AnsiColors.BLUE}Exchange (misal Binance, Coinbase, atau CCCAGG untuk agregat) [{new_config.get('exchange','CCCAGG')}]: {AnsiColors.ENDC}") or new_config.get('exchange','CCCAGG')).strip()

    tf_input = (input(f"{AnsiColors.BLUE}Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: {AnsiColors.ENDC}") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute', 'hour', 'day']: new_config["timeframe"] = tf_input
    else: print(f"{AnsiColors.RED}Timeframe tidak valid. Menggunakan default: {new_config.get('timeframe','hour')}{AnsiColors.ENDC}"); # No change if invalid

    refresh_input_str = input(f"{AnsiColors.BLUE}Interval Refresh (detik, setelah {TARGET_BIG_DATA_CANDLES} candle) [{new_config.get('refresh_interval_seconds',60)}]: {AnsiColors.ENDC}").strip()
    try:
        refresh_input = int(refresh_input_str) if refresh_input_str else new_config.get('refresh_interval_seconds',60)
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, refresh_input) # Enforce minimum
    except ValueError:
        print(f"{AnsiColors.RED}Input interval refresh tidak valid. Menggunakan default: {new_config.get('refresh_interval_seconds',60)}{AnsiColors.ENDC}")
        new_config["refresh_interval_seconds"] = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, new_config.get('refresh_interval_seconds',60))


    animated_text_display("\n-- Parameter Pivot --", color=AnsiColors.HEADER, delay=0.01)
    try:
        new_config["left_strength"] = int(input(f"{AnsiColors.BLUE}Left Strength [{new_config.get('left_strength',50)}]: {AnsiColors.ENDC}").strip() or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"{AnsiColors.BLUE}Right Strength [{new_config.get('right_strength',150)}]: {AnsiColors.ENDC}").strip() or new_config.get('right_strength',150))
    except ValueError:
        print(f"{AnsiColors.RED}Input strength tidak valid. Menggunakan default.{AnsiColors.ENDC}")
        new_config["left_strength"] = new_config.get('left_strength',50) # Revert to existing or default
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
    new_config["email_sender_app_password"] = (input(f"{AnsiColors.BLUE}App Password Email Pengirim [{new_config.get('email_sender_app_password','')}]: {AnsiColors.ENDC}") or new_config.get('email_sender_app_password','')).strip() # No masking
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
        num_recovery_keys = len([k for k in recovery_keys if k]) # Count non-empty keys
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

        # Structure to build the menu text more flexibly for pick
        # Options for 'pick' library
        original_options_structure = [
            ("header", "--- Pengaturan API & Global ---"),
            ("option", "Atur Primary API Key"),                                         # Index 0
            ("option", "Kelola Recovery API Keys"),                                     # Index 1
            ("option", "Atur Email Global untuk Notifikasi Sistem"),                    # Index 2
            ("option", "Aktifkan/Nonaktifkan Notifikasi Termux Realtime"),              # Index 3
            ("header", "--- Pengaturan Crypto Pair ---"),
            ("option", "Tambah Konfigurasi Crypto Baru"),                               # Index 4
            ("option", "Ubah Konfigurasi Crypto"),                                      # Index 5
            ("option", "Hapus Konfigurasi Crypto"),                                     # Index 6
            ("header", "-----------------------------"),
            ("option", "Kembali ke Menu Utama")                                         # Index 7
        ]
        
        # Extract only selectable options for `pick`
        selectable_options = [text for type, text in original_options_structure if type == "option"]
        
        # Reconstruct title for `pick`
        # pick_title_settings is already built above, good.
        
        try:
            option_text, index = pick(selectable_options, pick_title_settings, indicator='=>', default_index=0)
        except Exception as e_pick: # Catch potential errors from pick, e.g., if terminal too small or other issues
            log_error(f"Error dengan library 'pick': {e_pick}. Gunakan input manual.")
            print(pick_title_settings) # Show the menu text
            for idx, opt_text in enumerate(selectable_options):
                print(f"  {idx + 1}. {opt_text}")
            try:
                choice = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice < len(selectable_options):
                    index = choice # This 'index' corresponds to the index in 'selectable_options'
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue
        
        action_choice = index # This is the index from the selectable_options list

        # --- Handle Actions ---
        try:
            clear_screen_animated()
            if action_choice == 0: # Atur Primary API Key
                animated_text_display("--- Atur Primary API Key ---", color=AnsiColors.HEADER)
                api_s["primary_key"] = (input(f"Masukkan Primary API Key CryptoCompare baru [{api_s.get('primary_key','')}]: ").strip() or api_s.get('primary_key',''))
                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 1: # Kelola Recovery API Keys
                while True:
                    clear_screen_animated()
                    recovery_pick_title = "\n-- Kelola Recovery API Keys --\n"
                    current_recovery = [k for k in api_s.get('recovery_keys', []) if k] # Filter out empty strings
                    api_s['recovery_keys'] = current_recovery # Update api_s with cleaned list

                    if not current_recovery:
                        recovery_pick_title += "  (Tidak ada recovery key tersimpan)\n"
                    else:
                        for i, r_key in enumerate(current_recovery):
                            r_key_display = r_key[:5] + "..." + r_key[-3:] if len(r_key) > 8 else r_key
                            recovery_pick_title += f"  {i+1}. {r_key_display}\n"
                    recovery_pick_title += "\nPilih tindakan:"

                    recovery_options_plain = ["Tambah Recovery Key", "Hapus Recovery Key", "Kembali ke Pengaturan Utama"]
                    
                    try:
                        rec_option_text, rec_index = pick(recovery_options_plain, recovery_pick_title, indicator='=>', default_index=0)
                    except Exception as e_pick_rec:
                        log_error(f"Error dengan library 'pick' di menu recovery: {e_pick_rec}. Gunakan input manual.")
                        print(recovery_pick_title)
                        for idx_rec, opt_text_rec in enumerate(recovery_options_plain):
                             print(f"  {idx_rec + 1}. {opt_text_rec}")
                        try:
                            rec_choice = int(input("Pilih nomor opsi: ")) -1
                            if 0 <= rec_choice < len(recovery_options_plain):
                                rec_index = rec_choice
                            else:
                                print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                                show_spinner(1, "Kembali...")
                                continue
                        except ValueError:
                            print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                            show_spinner(1, "Kembali...")
                            continue

                    clear_screen_animated()

                    if rec_index == 0: # Tambah Recovery Key
                        animated_text_display("-- Tambah Recovery Key --", color=AnsiColors.HEADER)
                        new_r_key = input("Masukkan Recovery API Key baru: ").strip()
                        if new_r_key:
                            current_recovery.append(new_r_key)
                            api_s['recovery_keys'] = current_recovery
                            save_settings(current_settings) # Save immediately
                            print(f"{AnsiColors.GREEN}Recovery key ditambahkan.{AnsiColors.ENDC}")
                        else:
                            print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 1: # Hapus Recovery Key
                        animated_text_display("-- Hapus Recovery Key --", color=AnsiColors.HEADER)
                        if not current_recovery:
                            print(f"{AnsiColors.ORANGE}Tidak ada recovery key untuk dihapus.{AnsiColors.ENDC}")
                            show_spinner(1, "Kembali...")
                            continue
                        try:
                            for i_del, r_key_del in enumerate(current_recovery):
                                r_key_del_display = r_key_del[:5] + "..." + r_key_del[-3:] if len(r_key_del) > 8 else r_key_del
                                print(f"  {i_del+1}. {r_key_del_display}")
                            idx_del_str = input("Nomor recovery key yang akan dihapus: ").strip()
                            if not idx_del_str: # Check for empty input
                                print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                                show_spinner(1, "Kembali...")
                                continue
                            idx_del = int(idx_del_str) - 1
                            if 0 <= idx_del < len(current_recovery):
                                removed = current_recovery.pop(idx_del)
                                api_s['recovery_keys'] = current_recovery
                                save_settings(current_settings) # Save immediately
                                print(f"{AnsiColors.GREEN}Recovery key '{removed[:5]}...' dihapus.{AnsiColors.ENDC}")
                            else:
                                print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                        except ValueError:
                            print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                        show_spinner(1, "Kembali...")
                    elif rec_index == 2: # Kembali
                        break # Exit recovery key management loop
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
            
            elif action_choice == 3: # Aktifkan/Nonaktifkan Notifikasi Termux
                animated_text_display("-- Pengaturan Notifikasi Termux Realtime --", color=AnsiColors.HEADER)
                current_status = api_s.get('enable_termux_notifications', False)
                new_status_input = input(f"Aktifkan Notifikasi Termux? (true/false) [{current_status}]: ").lower().strip()
                if new_status_input == 'true':
                    api_s['enable_termux_notifications'] = True
                    print(f"{AnsiColors.GREEN}Notifikasi Termux diaktifkan.{AnsiColors.ENDC}")
                    print(f"{AnsiColors.ORANGE}Pastikan Termux:API terinstal dan `pkg install termux-api` sudah dijalankan di Termux.{AnsiColors.ENDC}")
                elif new_status_input == 'false':
                    api_s['enable_termux_notifications'] = False
                    print(f"{AnsiColors.GREEN}Notifikasi Termux dinonaktifkan.{AnsiColors.ENDC}")
                else: # Invalid input, no change
                    print(f"{AnsiColors.ORANGE}Input tidak valid. Status Notifikasi Termux tidak berubah: {current_status}.{AnsiColors.ENDC}")

                current_settings["api_settings"] = api_s
                save_settings(current_settings)
                show_spinner(2, "Menyimpan & Kembali...") # Longer spinner for reading message

            elif action_choice == 4: # Tambah Konfigurasi Crypto Baru
                new_crypto_conf = get_default_crypto_config() # Start with defaults
                new_crypto_conf = _prompt_crypto_config(new_crypto_conf) # Fill/update
                current_settings.setdefault("cryptos", []).append(new_crypto_conf)
                save_settings(current_settings)
                log_info(f"Konfigurasi untuk {new_crypto_conf['symbol']}-{new_crypto_conf['currency']} ditambahkan.")
                show_spinner(1, "Menyimpan & Kembali...")
            elif action_choice == 5: # Ubah Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk diubah.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali...");
                    continue
                animated_text_display("-- Ubah Konfigurasi Crypto --", color=AnsiColors.HEADER)
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                    print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")
                
                idx_choice_str = input("Nomor konfigurasi crypto yang akan diubah: ").strip()
                if not idx_choice_str:
                    print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue
                try:
                    idx_choice = int(idx_choice_str) - 1
                    if 0 <= idx_choice < len(current_settings["cryptos"]):
                        current_settings["cryptos"][idx_choice] = _prompt_crypto_config(current_settings["cryptos"][idx_choice])
                        save_settings(current_settings)
                        log_info(f"Konfigurasi untuk {current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']} diubah.")
                    else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                except ValueError:
                     print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 6: # Hapus Konfigurasi Crypto
                if not current_settings.get("cryptos"):
                    print(f"{AnsiColors.ORANGE}Tidak ada konfigurasi untuk dihapus.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali...");
                    continue
                animated_text_display("-- Hapus Konfigurasi Crypto --", color=AnsiColors.HEADER)
                for i, crypto_conf in enumerate(current_settings["cryptos"]):
                    print(f"  {i+1}. {crypto_conf.get('symbol','N/A')}-{crypto_conf.get('currency','N/A')}")

                idx_choice_str = input("Nomor konfigurasi crypto yang akan dihapus: ").strip()
                if not idx_choice_str:
                    print(f"{AnsiColors.RED}Input tidak boleh kosong.{AnsiColors.ENDC}")
                    show_spinner(1, "Kembali..."); continue
                try:
                    idx_choice = int(idx_choice_str) - 1
                    if 0 <= idx_choice < len(current_settings["cryptos"]):
                        removed_pair = f"{current_settings['cryptos'][idx_choice]['symbol']}-{current_settings['cryptos'][idx_choice]['currency']}"
                        current_settings["cryptos"].pop(idx_choice)
                        save_settings(current_settings)
                        log_info(f"Konfigurasi untuk {removed_pair} dihapus.")
                    else: print(f"{AnsiColors.RED}Nomor tidak valid.{AnsiColors.ENDC}")
                except ValueError:
                    print(f"{AnsiColors.RED}Input nomor tidak valid.{AnsiColors.ENDC}")
                show_spinner(1, "Kembali...")
            elif action_choice == 7: # Kembali ke Menu Utama
                break # Exit settings menu loop
        except ValueError: # Catch general ValueErrors from int conversions if any slip through
            print(f"{AnsiColors.RED}Input angka tidak valid.{AnsiColors.ENDC}")
            show_spinner(1.5, "Error, kembali...")
        except Exception as e: # Catch-all for other unexpected errors in menu logic
            log_error(f"Terjadi kesalahan di menu pengaturan: {e}")
            log_exception("Traceback Error Settings Menu:")
            show_spinner(1.5, "Error, kembali...")
    return current_settings


# --- FUNGSI PENGAMBILAN DATA ---
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use:
        log_error(f"Tidak ada API key yang diberikan untuk fetch_candles.", pair_name=pair_name)
        raise APIKeyError("API Key tidak tersedia untuk request.")

    all_accumulated_candles = []
    current_to_ts = None # Timestamp UNIX untuk candle terakhir yang diambil (untuk paginasi mundur)
    api_endpoint = "histohour"
    if timeframe == "minute": api_endpoint = "histominute"
    elif timeframe == "day": api_endpoint = "histoday"

    url = f"https://min-api.cryptocompare.com/data/v2/{api_endpoint}"
    is_large_fetch = total_limit_desired > 10 # Heuristic for logging/progress bar

    if is_large_fetch:
        log_info(f"Memulai pengambilan data: target {total_limit_desired} TF {timeframe}.", pair_name=pair_name)

    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT : # Only show progress bar for very large multi-batch fetches
        simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

    fetch_loop_count = 0 
    while len(all_accumulated_candles) < total_limit_desired:
        candles_still_needed = total_limit_desired - len(all_accumulated_candles)
        # CryptoCompare API limit per call is 2000.
        limit_for_this_api_call = min(candles_still_needed, CRYPTOCOMPARE_MAX_LIMIT)

        # If paginating (toTs is set), and we need more than 1 candle, request one extra candle
        # This extra candle is the one at toTs, which helps verify continuity or overlap.
        if current_to_ts is not None and candles_still_needed > 1 : #
            limit_for_this_api_call = min(candles_still_needed + 1, CRYPTOCOMPARE_MAX_LIMIT)
        
        if limit_for_this_api_call <= 0: break # Should not happen if loop condition is correct

        params = {
            "fsym": symbol, "tsym": currency,
            "limit": limit_for_this_api_call,
            "api_key": current_api_key_to_use
        }
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if current_to_ts is not None: params["toTs"] = current_to_ts

        try:
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Log for multi-batch only
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_debug(f"Fetching batch (Key: ...{key_display}, Limit: {limit_for_this_api_call})", pair_name=pair_name)

            response = requests.get(url, params=params, timeout=20) # Increased timeout

            # Check for HTTP errors that often indicate API key issues or rate limits
            if response.status_code in [401, 403, 429]: # Unauthorized, Forbidden, Too Many Requests
                error_data = {}
                try: # Try to get JSON error message from API if available
                    error_data = response.json()
                except json.JSONDecodeError:
                    pass # No JSON body, or not valid JSON
                error_message = error_data.get('Message', f"HTTP Error {response.status_code}")
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {response.status_code}): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {response.status_code}: {error_message}")

            response.raise_for_status() # Check for other HTTP errors (4xx, 5xx)
            data = response.json()

            if data.get('Response') == 'Error':
                error_message = data.get('Message', 'N/A')
                key_related_error_messages = [
                    "api key is invalid", "apikey_is_missing", "apikey_invalid",
                    "your_monthly_calls_are_over_the_limit", "rate limit exceeded",
                    "your_pro_tier_has_expired_or_is_not_active",
                    "you are over your rate limit" # Penambahan
                ]
                key_display = current_api_key_to_use[-5:] if len(current_api_key_to_use) > 5 else current_api_key_to_use
                if any(keyword.lower() in error_message.lower() for keyword in key_related_error_messages):
                    log_warning(f"{AnsiColors.RED}API Key Error (JSON): {error_message}{AnsiColors.ENDC} Key: ...{key_display}", pair_name=pair_name)
                    raise APIKeyError(f"JSON Error: {error_message}")
                else:
                    # Other API errors not related to keys (e.g., pair not found on exchange)
                    log_error(f"{AnsiColors.RED}API Error CryptoCompare: {error_message}{AnsiColors.ENDC} (Params: {params})", pair_name=pair_name)
                    break # Stop fetching for this pair if non-key API error

            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
                if is_large_fetch: log_info(f"Tidak ada lagi data candle dari API atau format data tidak sesuai. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break # No more data or unexpected format

            raw_candles_from_api = data['Data']['Data']

            if not raw_candles_from_api: # Should be caught by above, but defensive check
                if is_large_fetch: log_info(f"API mengembalikan list candle kosong. Total diambil: {len(all_accumulated_candles)}.", pair_name=pair_name)
                break

            # Process candles in this batch
            batch_candles_list = []
            for item in raw_candles_from_api:
                # Ensure all essential fields are present and are numbers (or can be converted)
                if not all(k in item for k in ['time', 'open', 'high', 'low', 'close']): continue
                try:
                    candle = {
                        'timestamp': datetime.fromtimestamp(item['time']),
                        'open': float(item['open']), 'high': float(item['high']),
                        'low': float(item['low']), 'close': float(item['close']),
                        'volume': float(item.get('volumefrom',0)) # Default volume to 0 if missing
                    }
                    batch_candles_list.append(candle)
                except (TypeError, ValueError) as e:
                    log_warning(f"Skipping candle with invalid data: {item}, Error: {e}", pair_name=pair_name)
                    continue


            # Overlap handling: if toTs was used, the first candle of new batch might be same as last of previous.
            # CryptoCompare returns data UP TO toTs (exclusive of toTs if toTs is a candle time).
            # If we requested `limit+1` and `toTs` was set to the timestamp of the first candle
            # in `all_accumulated_candles`, then `batch_candles_list`'s last item should be that candle.
            if current_to_ts is not None and all_accumulated_candles and batch_candles_list:
                # The new batch is prepended, so the *last* candle of batch_candles_list
                # could be the *first* candle of all_accumulated_candles.
                if batch_candles_list[-1]['timestamp'] == all_accumulated_candles[0]['timestamp']:
                    if is_large_fetch: log_debug(f"Menghapus candle tumpang tindih: {batch_candles_list[-1]['timestamp']}", pair_name=pair_name)
                    batch_candles_list.pop() # Remove the duplicate

            if not batch_candles_list and current_to_ts is not None : # If batch became empty after overlap removal
                if is_large_fetch: log_info("Batch menjadi kosong setelah overlap removal. Kemungkinan akhir data.", pair_name=pair_name)
                break

            all_accumulated_candles = batch_candles_list + all_accumulated_candles # Prepend new batch

            if raw_candles_from_api: # Set toTs for the next iteration from the *earliest* candle in this raw batch
                current_to_ts = raw_candles_from_api[0]['time'] # Earliest timestamp from this batch
            else: # Should not happen if previous checks are fine
                break

            fetch_loop_count +=1
            if is_large_fetch and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and (fetch_loop_count % 2 == 0 or len(all_accumulated_candles) >= total_limit_desired): # Update progress bar periodically
                simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles', length=40)

            # If API returns fewer items than requested (and it's not the max limit, indicating end of data)
            if len(raw_candles_from_api) < limit_for_this_api_call :
                if is_large_fetch: log_info(f"API mengembalikan < limit ({len(raw_candles_from_api)} vs {limit_for_this_api_call}). Akhir histori tercapai.", pair_name=pair_name)
                break # Reached the end of available history

            if len(all_accumulated_candles) >= total_limit_desired: break # Target met

            # Small delay if making multiple calls for large data fetch
            if len(all_accumulated_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT and is_large_fetch:
                log_debug(f"Diambil {len(batch_candles_list)} baru. Total: {len(all_accumulated_candles)}. Target: {total_limit_desired}. Delay...", pair_name=pair_name)
                time.sleep(0.3) # Be nice to the API

        except APIKeyError: # Re-raise to be caught by the caller (APIKeyManager logic)
            raise 
        except requests.exceptions.RequestException as e:
            log_error(f"{AnsiColors.RED}Kesalahan koneksi/permintaan saat mengambil batch: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            break # Stop fetching for this pair on connection issues
        except Exception as e: # Catch-all for other unexpected errors
            log_error(f"{AnsiColors.RED}Error tak terduga dalam fetch_candles: {e}{AnsiColors.ENDC}", pair_name=pair_name)
            log_exception("Traceback Error:", pair_name=pair_name) # Log full traceback
            break # Stop
            
    # Ensure we don't have more candles than desired (e.g. if last batch made it slightly over)
    if len(all_accumulated_candles) > total_limit_desired:
        all_accumulated_candles = all_accumulated_candles[-total_limit_desired:]

    if is_large_fetch:
        if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: # Final update for progress bar
            simple_progress_bar(len(all_accumulated_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix='Candles Complete', length=40)
        log_info(f"Pengambilan data selesai. Total {len(all_accumulated_candles)} (target: {total_limit_desired}).", pair_name=pair_name)

    return all_accumulated_candles


# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, # 1 for high, -1 for low
        "final_pivot_high_price_confirmed": None,
        "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, # Stores the high of PH before PL for FIB calc
        "high_bar_index_for_fib": None,
        "active_fib_level": None, # The 0.5 FIB level price once calculated
        "active_fib_line_start_index": None, # Bar index where FIB line starts (PL)
        "entry_price_custom": None,
        "highest_price_for_trailing": None,
        "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None,
        "emergency_sl_level_custom": None,
        "position_size": 0, # 1 if in position, 0 otherwise
    }

def find_pivots(series_list, left_strength, right_strength, is_high=True):
    pivots = [None] * len(series_list) # Initialize with None
    if len(series_list) < left_strength + right_strength + 1:
        return pivots # Not enough data

    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True
        if series_list[i] is None: continue # Skip if current point is None

        # Check left side
        for j in range(1, left_strength + 1):
            if series_list[i-j] is None: is_pivot = False; break # Skip if any surrounding point is None
            if is_high: # For pivot high
                if series_list[i] <= series_list[i-j]: is_pivot = False; break
            else: # For pivot low
                if series_list[i] >= series_list[i-j]: is_pivot = False; break
        if not is_pivot: continue # Move to next 'i' if left side fails

        # Check right side
        for j in range(1, right_strength + 1):
            if series_list[i+j] is None: is_pivot = False; break
            if is_high: # For pivot high
                if series_list[i] < series_list[i+j]: is_pivot = False; break # Note: strictly < for PH right side
            else: # For pivot low
                if series_list[i] > series_list[i+j]: is_pivot = False; break # Note: strictly > for PL right side
        
        if is_pivot:
            pivots[i] = series_list[i] # Store the pivot price at its index
    return pivots


def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    
    # Reset confirmed pivots at the start of each call, they are determined fresh
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None

    left_strength = crypto_config['left_strength']
    right_strength = crypto_config['right_strength']

    # Validate candles_history structure
    required_keys = ['high', 'low', 'open', 'close', 'timestamp']
    if not candles_history or not all(key in candles_history[0] for key in required_keys if candles_history and candles_history[0]):
        log_warning(f"{AnsiColors.ORANGE}Data candle kosong atau kunci OHLC tidak lengkap di run_strategy_logic.{AnsiColors.ENDC}", pair_name=pair_name)
        return strategy_state

    high_prices = [c.get('high') for c in candles_history] # Robust: handles if a candle is missing 'high'
    low_prices = [c.get('low') for c in candles_history]   # Robust: handles if a candle is missing 'low'

    # Calculate raw pivots for the entire history
    raw_pivot_highs = find_pivots(high_prices, left_strength, right_strength, True)
    raw_pivot_lows = find_pivots(low_prices,  left_strength, right_strength, False)

    # Pivot is confirmed 'right_strength' bars after it occurs.
    # So, to check for a pivot confirmed on the *current* bar (latest - 1),
    # we look at the pivot status from 'right_strength' bars ago.
    current_bar_index_in_list = len(candles_history) - 1 # Index of the most recent (possibly incomplete) candle
    if current_bar_index_in_list < 0 : return strategy_state # Should not happen with guard above

    # The bar on which the pivot *event* (the actual high/low) occurred.
    # This is 'right_strength' bars before the bar where it's *confirmed*.
    # We are checking based on the latest data, so confirmation happens for past bars.
    idx_pivot_event_high = current_bar_index_in_list - right_strength
    idx_pivot_event_low = current_bar_index_in_list - right_strength

    # Get the pivot price if one was confirmed at that event index
    raw_pivot_high_price_at_event = raw_pivot_highs[idx_pivot_event_high] if 0 <= idx_pivot_event_high < len(raw_pivot_highs) else None
    raw_pivot_low_price_at_event = raw_pivot_lows[idx_pivot_event_low] if 0 <= idx_pivot_event_low < len(raw_pivot_lows) else None

    # --- Pivot High Logic ---
    if raw_pivot_high_price_at_event is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_high_price_at_event
        strategy_state["last_signal_type"] = 1 # Mark that a PH was the last significant pivot
        pivot_timestamp = candles_history[idx_pivot_event_high]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    # --- Pivot Low Logic ---
    if raw_pivot_low_price_at_event is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_low_price_at_event
        strategy_state["last_signal_type"] = -1 # Mark that a PL was the last significant pivot
        pivot_timestamp = candles_history[idx_pivot_event_low]['timestamp']
        log_info(f"{AnsiColors.CYAN}PIVOT LOW:  {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {pivot_timestamp.strftime('%Y-%m-%d %H:%M')}{AnsiColors.ENDC}", pair_name=pair_name)

    # --- Fibonacci and Entry Logic ---
    current_candle = candles_history[current_bar_index_in_list] # The latest candle data

    # Ensure current candle has necessary data
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']):
        log_warning(f"Data OHLC tidak lengkap untuk candle terbaru @ {current_candle.get('timestamp', 'N/A')}. Skip evaluasi.", pair_name=pair_name)
        return strategy_state

    # If a new PH is confirmed, store its details for potential FIB calculation
    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event_high # Bar index of the PH

        # If there was an active FIB level from a previous setup, a new PH invalidates it.
        if strategy_state["active_fib_level"] is not None:
            log_debug("Resetting active FIB due to new High.", pair_name=pair_name)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    # If a new PL is confirmed AND we have a stored PH to pair with it:
    if strategy_state["final_pivot_low_price_confirmed"] is not None:
        if strategy_state["high_price_for_fib"] is not None and \
           strategy_state["high_bar_index_for_fib"] is not None:
            
            current_low_price = strategy_state["final_pivot_low_price_confirmed"]
            current_low_bar_index = idx_pivot_event_low # Bar index of the PL

            # Ensure PL occurred *after* the PH for a valid downward swing
            if current_low_bar_index > strategy_state["high_bar_index_for_fib"]:
                if strategy_state["high_price_for_fib"] is None or current_low_price is None: # Should be caught by prior checks
                     log_warning("Harga untuk kalkulasi FIB tidak valid (None).", pair_name=pair_name)
                else: 
                    calculated_fib_level = (strategy_state["high_price_for_fib"] + current_low_price) / 2.0

                    is_fib_late = False
                    if crypto_config["enable_secure_fib"]:
                        # Check if current price (close or high) is already above the calculated FIB
                        # This check happens on the bar where PL is *confirmed*
                        price_val_current_candle = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle.get('close'))
                        if price_val_current_candle is not None and calculated_fib_level is not None and price_val_current_candle > calculated_fib_level:
                            is_fib_late = True

                    if is_fib_late:
                        log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calculated_fib_level:.5f}), Harga Cek ({crypto_config['secure_fib_check_price']}: {price_val_current_candle:.5f}) > FIB.{AnsiColors.ENDC}", pair_name=pair_name)
                        strategy_state["active_fib_level"] = None # Do not activate this FIB
                        strategy_state["active_fib_line_start_index"] = None
                    elif calculated_fib_level is not None : # Activate FIB
                        log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calculated_fib_level:.5f}{AnsiColors.ENDC} (H: {strategy_state['high_price_for_fib']:.2f}, L: {current_low_price:.2f})", pair_name=pair_name)
                        strategy_state["active_fib_level"] = calculated_fib_level
                        strategy_state["active_fib_line_start_index"] = current_low_bar_index # FIB line starts from PL bar
            
            # Reset the stored PH details, as this PH-PL pair has been processed (or was invalid)
            strategy_state["high_price_for_fib"] = None
            strategy_state["high_bar_index_for_fib"] = None

    # Check for entry if a FIB level is active
    if strategy_state["active_fib_level"] is not None and \
       strategy_state["active_fib_line_start_index"] is not None:
        
        # Ensure current candle has necessary data for entry check (already checked above, but good for clarity)
        if current_candle.get('close') is None or current_candle.get('open') is None:
            log_warning("Nilai close atau open tidak ada di candle saat ini. Skip entry check.", pair_name=pair_name)
            return strategy_state

        is_bullish_candle = current_candle['close'] > current_candle['open']
        is_closed_above_fib = current_candle['close'] > strategy_state["active_fib_level"]

        if is_bullish_candle and is_closed_above_fib:
            if strategy_state["position_size"] == 0: # Only enter if not already in a position
                strategy_state["position_size"] = 1 # Enter position
                entry_px = current_candle['close'] # Entry at close of breakout candle
                strategy_state["entry_price_custom"] = entry_px
                strategy_state["highest_price_for_trailing"] = entry_px # Initialize for trailing
                strategy_state["trailing_tp_active_custom"] = False # Trailing not active yet
                strategy_state["current_trailing_stop_level"] = None

                emerg_sl = entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0)
                strategy_state["emergency_sl_level_custom"] = emerg_sl

                log_msg = f"BUY ENTRY @ {entry_px:.5f} (FIB {strategy_state['active_fib_level']:.5f} dilewati). Emerg SL: {emerg_sl:.5f}"
                log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                play_notification_sound()
                
                # Termux & Email Notifications
                termux_title = f"BUY Signal: {pair_name}"
                termux_content = f"Entry @ {entry_px:.5f}. SL: {emerg_sl:.5f}"
                send_termux_notification(termux_title, termux_content, global_settings, pair_name_for_log=pair_name)

                email_subject = f"BUY Signal: {pair_name}"
                email_body = (f"New BUY signal for {pair_name} on {crypto_config['exchange']}.\n\n"
                              f"Entry Price: {entry_px:.5f}\n"
                              f"FIB Level: {strategy_state['active_fib_level']:.5f}\n"
                              f"Emergency SL: {emerg_sl:.5f}\n"
                              f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                send_email_notification(email_subject, email_body, crypto_config)
            
            # Deactivate FIB level after entry (or attempted entry)
            strategy_state["active_fib_level"] = None
            strategy_state["active_fib_line_start_index"] = None

    # --- Position Management (if in position) ---
    if strategy_state["position_size"] > 0: 
        # Update highest price seen since entry for trailing stop
        current_high_for_trailing = strategy_state.get("highest_price_for_trailing", current_candle.get('high'))
        if current_high_for_trailing is None or current_candle.get('high') is None: 
            log_warning("Harga tertinggi untuk trailing atau high candle tidak valid (None).", pair_name=pair_name)
        else:
             strategy_state["highest_price_for_trailing"] = max(current_high_for_trailing , current_candle['high'])

        # Activate Trailing TP if profit target is met
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"] is not None:
            if strategy_state["entry_price_custom"] == 0: # Avoid division by zero
                profit_percent = 0.0
            elif strategy_state.get("highest_price_for_trailing") is None: # Should not happen if entry_price_custom is set
                profit_percent = 0.0
                log_warning("highest_price_for_trailing is None saat kalkulasi profit.", pair_name=pair_name)
            else:
                profit_percent = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            
            if profit_percent >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                log_info(f"{AnsiColors.BLUE}Trailing TP Aktif. Profit: {profit_percent:.2f}%, High: {strategy_state.get('highest_price_for_trailing',0):.5f}{AnsiColors.ENDC}", pair_name=pair_name)

        # Update Trailing Stop Level if active
        if strategy_state["trailing_tp_active_custom"] and strategy_state.get("highest_price_for_trailing") is not None:
            potential_new_stop_price = strategy_state["highest_price_for_trailing"] * (1 - (crypto_config["trailing_stop_gap_percent"] / 100.0))
            # Trailing stop can only move up
            if strategy_state["current_trailing_stop_level"] is None or potential_new_stop_price > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = potential_new_stop_price
                log_debug(f"Trailing SL update: {strategy_state['current_trailing_stop_level']:.5f}", pair_name=pair_name)

        # Determine final stop loss (Emergency SL or Trailing SL)
        final_stop_for_exit = strategy_state["emergency_sl_level_custom"]
        exit_comment = "Emergency SL"
        exit_color = AnsiColors.RED

        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            # If trailing stop is higher (better) than emergency SL, use it
            if final_stop_for_exit is None or strategy_state["current_trailing_stop_level"] > final_stop_for_exit :
                final_stop_for_exit = strategy_state["current_trailing_stop_level"]
                exit_comment = "Trailing Stop"
                exit_color = AnsiColors.BLUE # Blue for trailing stop, could be profit or small loss
        
        # Check for Exit
        if final_stop_for_exit is not None and current_candle.get('low') is not None and current_candle['low'] <= final_stop_for_exit:
            # Exit price: if candle opens below SL, exit at open. Otherwise, exit at SL.
            exit_price_open = current_candle.get('open')
            if exit_price_open is None: # Should be validated earlier
                log_warning("Harga open candle tidak ada untuk exit. Menggunakan SL sebagai harga exit.", pair_name=pair_name)
                exit_price = final_stop_for_exit # Fallback
            else:
                exit_price = min(exit_price_open, final_stop_for_exit) # Exit at SL or open if gapped down
            
            pnl = 0.0
            if strategy_state["entry_price_custom"] is not None and strategy_state["entry_price_custom"] != 0:
                pnl = ((exit_price - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0

            # If trailing stop hit but resulted in loss, color it red
            if exit_comment == "Trailing Stop" and pnl < 0:
                exit_color = AnsiColors.RED

            log_msg = f"EXIT ORDER @ {exit_price:.5f} by {exit_comment}. PnL: {pnl:.2f}%"
            log_info(f"{exit_color}{AnsiColors.BOLD}{log_msg}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()

            # Termux & Email Notifications for Exit
            termux_title_exit = f"EXIT Signal: {pair_name}"
            termux_content_exit = f"{exit_comment} @ {exit_price:.5f}. PnL: {pnl:.2f}%"
            send_termux_notification(termux_title_exit, termux_content_exit, global_settings, pair_name_for_log=pair_name)

            email_subject = f"Trade Closed: {pair_name} ({exit_comment})"
            email_body = (f"Trade closed for {pair_name} on {crypto_config['exchange']}.\n\n"
                          f"Exit Price: {exit_price:.5f}\n"
                          f"Reason: {exit_comment}\n"
                          f"Entry Price: {strategy_state.get('entry_price_custom', 0):.5f}\n"
                          f"PnL: {pnl:.2f}%\n"
                          f"Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            send_email_notification(email_subject, email_body, crypto_config)

            # Reset position state
            strategy_state["position_size"] = 0
            strategy_state["entry_price_custom"] = None
            strategy_state["highest_price_for_trailing"] = None
            strategy_state["trailing_tp_active_custom"] = False
            strategy_state["current_trailing_stop_level"] = None
            strategy_state["emergency_sl_level_custom"] = None
            # strategy_state["last_signal_type"] = 0 # Reset signal type after trade closure

    # Logging for active position status (if any)
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
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(
        api_settings.get("primary_key"),
        api_settings.get("recovery_keys", []),
        api_settings # Pass global api_settings for email notifications from manager
    )

    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key (primary/recovery) yang valid dikonfigurasi. Tidak dapat memulai.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    # NEW: CHART FEATURE - Start Flask server in a thread
    is_flask_running = any(t.name == "FlaskWebAppThread" for t in threading.enumerate())
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_app, name="FlaskWebAppThread", daemon=True)
        flask_thread.start()
    else:
        log_info("Flask web server thread sepertinya sudah berjalan.", "WEBSERVER")


    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto yang aktif untuk dijalankan.{AnsiColors.ENDC}")
        animated_text_display("Tekan Enter untuk kembali ke menu...", color=AnsiColors.ORANGE)
        input()
        return

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    current_key_display_val = api_key_manager.get_current_key()
    current_key_display = "N/A"
    if current_key_display_val:
        current_key_display = current_key_display_val[:5] + "..." + current_key_display_val[-3:] if len(current_key_display_val) > 8 else current_key_display_val

    log_info(f"Menggunakan API Key Index: {api_key_manager.get_current_key_index()} ({current_key_display}). Total keys: {api_key_manager.total_keys()}", pair_name="SYSTEM")

    crypto_data_manager = {} # Stores all data per crypto pair
    for config in all_crypto_configs:
        # Use a unique ID that includes symbol, currency, and timeframe for the manager key
        pair_id = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}_{config.get('timeframe','DEF')}" 
        config['pair_name'] = f"{config.get('symbol','DEF')}-{config.get('currency','DEF')}" # For logging and display

        animated_text_display(f"\nMenginisialisasi untuk {AnsiColors.BOLD}{config['pair_name']}{AnsiColors.ENDC} | Exch: {config.get('exchange','DEF')} | TF: {config.get('timeframe','DEF')}", color=AnsiColors.MAGENTA, delay=0.01)

        crypto_data_manager[pair_id] = {
            "config": config,
            "all_candles_list": [],
            "strategy_state": get_initial_strategy_state(),
            "big_data_collection_phase_active": True, # Start in big data collection mode
            "big_data_email_sent": False,
            "last_candle_fetch_time": datetime.min, # Time of last successful fetch for this pair
            "data_fetch_failed_consecutively": 0 # Consecutive fetch failures for this pair
        }
        
        # --- Initial Big Data Fetch ---
        initial_candles_target = TARGET_BIG_DATA_CANDLES
        initial_candles = []
        max_retries_initial = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
        retries_done_initial = 0
        initial_fetch_successful = False

        while retries_done_initial < max_retries_initial and not initial_fetch_successful:
            current_api_key = api_key_manager.get_current_key()
            if not current_api_key: # No more keys available globally
                log_error(f"BIG DATA: Semua API key habis saat mencoba mengambil data awal untuk {config['pair_name']}.", pair_name=config['pair_name'])
                break # Stop trying for this pair

            try:
                log_info(f"BIG DATA: Mengambil data awal (target {initial_candles_target} candle) dengan key index {api_key_manager.get_current_key_index()}...", pair_name=config['pair_name'])
                initial_candles = fetch_candles(
                    config['symbol'], config['currency'], initial_candles_target,
                    config['exchange'], current_api_key, config['timeframe'],
                    pair_name=config['pair_name']
                )
                initial_fetch_successful = True # Success for this attempt
            except APIKeyError: # Key failed, try next
                log_warning(f"BIG DATA: API Key gagal untuk {config['pair_name']}. Mencoba key berikutnya.", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break # No more keys to switch to
                retries_done_initial +=1 # Count this as a retry using a new key
            except requests.exceptions.RequestException as e: # Network error, don't switch key, just log and potentially retry later
                log_error(f"BIG DATA: Error jaringan saat mengambil data awal {config['pair_name']}: {e}. Tidak mengganti key.", pair_name=config['pair_name'])
                break # Stop trying for this pair on network error for now
            except Exception as e_gen: # Other unexpected error
                log_error(f"BIG DATA: Error umum saat mengambil data awal {config['pair_name']}: {e_gen}. Tidak mengganti key.", pair_name=config['pair_name'])
                log_exception("Traceback Error Initial Fetch:", pair_name=config['pair_name'])
                break

        if not initial_candles:
            log_error(f"{AnsiColors.RED}BIG DATA: Gagal mengambil data awal untuk {config['pair_name']} setelah semua upaya. Pair ini mungkin tidak diproses dengan benar.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False # Cannot collect if initial failed
            crypto_data_manager[pair_id]["last_candle_fetch_time"] = datetime.now() # Mark attempt time
            continue # Move to next crypto pair configuration

        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        log_info(f"BIG DATA: {len(initial_candles)} candle awal diterima.", pair_name=config['pair_name'])

        # --- Warm-up Strategy State with Initial Data ---
        if initial_candles:
            min_len_for_pivots = config.get('left_strength', 50) + config.get('right_strength', 150) + 1 # Min candles for pivot calc
            if len(initial_candles) >= min_len_for_pivots:
                log_info(f"Memproses {max(0, len(initial_candles) - 1)} candle historis awal untuk inisialisasi state...", pair_name=config['pair_name'])
                
                # Iterate through historical data to "warm up" strategy state, but don't take trades
                for i in range(min_len_for_pivots -1, len(initial_candles) - 1): # Up to second to last candle
                    historical_slice = initial_candles[:i+1]
                    if len(historical_slice) < min_len_for_pivots: continue # Ensure slice is large enough

                    temp_state_for_warmup = crypto_data_manager[pair_id]["strategy_state"].copy()
                    temp_state_for_warmup["position_size"] = 0 # Ensure no trades during warm-up
                    
                    # Run logic on this slice, result updates temp_state_for_warmup
                    crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(historical_slice, config, temp_state_for_warmup, global_settings_dict)
                    
                    # Crucially, reset any trade-related state that might have been set by logic
                    # if it accidentally simulated a trade. The goal is to set up pivots, FIBs, etc.
                    if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: # Should not happen with above override but defensive
                        crypto_data_manager[pair_id]["strategy_state"] = {
                            **crypto_data_manager[pair_id]["strategy_state"], # Keep structural elements
                            **{"position_size":0, "entry_price_custom":None, "emergency_sl_level_custom":None, 
                               "highest_price_for_trailing":None, "trailing_tp_active_custom":False, 
                               "current_trailing_stop_level":None}
                        }
                log_info(f"{AnsiColors.CYAN}Inisialisasi state (warm-up) dengan data awal selesai.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            else:
                log_warning(f"Data awal ({len(initial_candles)}) tidak cukup untuk warm-up pivot (min: {min_len_for_pivots}).", pair_name=config['pair_name'])
        else:
            log_warning("Tidak ada data awal untuk warm-up.", pair_name=config['pair_name'])

        # Check if big data collection target met after initial fetch
        if len(crypto_data_manager[pair_id]["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI setelah pengambilan awal!{AnsiColors.ENDC}", pair_name=config['pair_name'])
            if not crypto_data_manager[pair_id]["big_data_email_sent"]: # Send email on first completion
                send_email_notification(f"Data Downloading Complete: {config['pair_name']}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {config['pair_name']}.", config)
                crypto_data_manager[pair_id]["big_data_email_sent"] = True
            log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(crypto_data_manager[pair_id]['all_candles_list'])} candles) ----------{AnsiColors.ENDC}", pair_name=config['pair_name'])
        
        # NEW: CHART FEATURE - Initial update for this pair's chart
        update_live_chart_data(config['pair_name'], crypto_data_manager[pair_id]["all_candles_list"], crypto_data_manager[pair_id]["strategy_state"], config)


    animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)

    # --- Main Trading Loop ---
    try:
        while True:
            active_cryptos_still_in_big_data_collection = 0
            min_overall_next_refresh_seconds = float('inf') # Min wait time across all pairs
            any_data_fetched_this_cycle = False # Used for global API key failure cooldown

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]
                pair_name = config['pair_name'] # Convenience

                # Cooldown logic if all keys failed for this pair previously
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) + 1 : # +1 to ensure all keys tried
                    if (datetime.now() - data.get("last_attempt_after_all_keys_failed", datetime.min)).total_seconds() < 3600: # 1 hour cooldown
                        log_debug(f"Pair {pair_name} sedang dalam cooldown 1 jam setelah semua key gagal.", pair_name=pair_name)
                        continue # Skip this pair for now
                    else:
                        data["data_fetch_failed_consecutively"] = 0 # Reset failure count after cooldown
                        log_info(f"Cooldown 1 jam untuk {pair_name} selesai. Mencoba fetch lagi.", pair_name=pair_name)

                current_loop_time = datetime.now()
                time_since_last_fetch_seconds = (current_loop_time - data["last_candle_fetch_time"]).total_seconds()

                # Determine required refresh interval for this pair
                required_interval_for_this_pair = 0
                if data["big_data_collection_phase_active"]:
                    active_cryptos_still_in_big_data_collection += 1
                    # Faster refresh during big data collection
                    if config.get('timeframe') == "minute": required_interval_for_this_pair = 55 # Slightly less than 1 min
                    elif config.get('timeframe') == "day": required_interval_for_this_pair = 3600 * 23.8 # Slightly less than 1 day
                    else: required_interval_for_this_pair = 3580 # Slightly less than 1 hour
                else: # Live trading phase
                    required_interval_for_this_pair = config.get('refresh_interval_seconds', 60) 

                if time_since_last_fetch_seconds < required_interval_for_this_pair:
                    remaining_time_for_this_pair = required_interval_for_this_pair - time_since_last_fetch_seconds
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, remaining_time_for_this_pair)
                    continue # Not time to refresh this pair yet

                log_info(f"Memproses {pair_name}...", pair_name=pair_name)
                data["last_candle_fetch_time"] = current_loop_time # Mark attempt time
                num_candles_before_fetch = len(data["all_candles_list"])

                if data["big_data_collection_phase_active"]:
                    animated_text_display(f"\n--- PENGUMPULAN BIG DATA ({len(data['all_candles_list'])}/{TARGET_BIG_DATA_CANDLES}) ---", color=AnsiColors.BOLD + AnsiColors.MAGENTA, delay=0.005)
                else:
                    animated_text_display(f"\n--- ANALISA ({current_loop_time.strftime('%Y-%m-%d %H:%M:%S')}) | {len(data['all_candles_list'])} candles ---", color=AnsiColors.BOLD + AnsiColors.CYAN, delay=0.005)

                # --- Fetch Update Data for This Pair ---
                new_candles_batch = []
                fetch_update_successful_for_this_pair = False
                
                max_retries_for_this_pair_update = api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1
                retries_done_for_this_pair_update = 0

                while retries_done_for_this_pair_update < max_retries_for_this_pair_update and not fetch_update_successful_for_this_pair:
                    current_api_key_for_attempt = api_key_manager.get_current_key()
                    if not current_api_key_for_attempt: # No more keys globally
                        log_error(f"Semua API key habis secara global saat mencoba mengambil update untuk {pair_name}.", pair_name=pair_name)
                        break 

                    limit_fetch = 3 # Default for live updates (get current + a couple of recent ones)
                    if data["big_data_collection_phase_active"]:
                        limit_fetch_needed = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"])
                        if limit_fetch_needed <=0 : # Should be caught by phase check, but defensive
                             fetch_update_successful_for_this_pair = True # No new data needed
                             new_candles_batch = []
                             break
                        limit_fetch = min(limit_fetch_needed, CRYPTOCOMPARE_MAX_LIMIT) # Fetch up to API limit
                        limit_fetch = max(limit_fetch, 1) # Fetch at least 1

                    log_info(f"Mengambil {limit_fetch} candle (Key Idx: {api_key_manager.get_current_key_index()})...", pair_name=pair_name)
                    try:
                        new_candles_batch = fetch_candles(
                            config['symbol'], config['currency'], limit_fetch, 
                            config['exchange'], current_api_key_for_attempt, config['timeframe'],
                            pair_name=pair_name
                        )
                        fetch_update_successful_for_this_pair = True
                        data["data_fetch_failed_consecutively"] = 0 # Reset failure count on success
                        any_data_fetched_this_cycle = True # Mark that at least one fetch worked globally
                    
                    except APIKeyError: # Key failed, try next global key
                        log_warning(f"API Key (Idx: {api_key_manager.get_current_key_index()}) gagal untuk update {pair_name}. Mencoba key berikutnya.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        
                        if not api_key_manager.switch_to_next_key(): # Switch global key
                            log_error(f"Tidak ada lagi API key tersedia secara global setelah kegagalan pada {pair_name}.", pair_name=pair_name)
                            break # No more keys to try globally
                        retries_done_for_this_pair_update += 1 # This counts as a retry *for this pair* with a new *global* key

                    except requests.exceptions.RequestException as e: # Network error
                        log_error(f"Error jaringan saat mengambil update {pair_name}: {e}. Tidak mengganti key.", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break # Stop trying for this pair in this cycle on network error
                    except Exception as e_gen_update: # Other error
                        log_error(f"Error umum saat mengambil update {pair_name}: {e_gen_update}. Tidak mengganti key.", pair_name=pair_name)
                        log_exception("Traceback Error Update Fetch:", pair_name=pair_name)
                        data["data_fetch_failed_consecutively"] = data.get("data_fetch_failed_consecutively", 0) + 1
                        break

                # If all keys failed for this pair in this cycle
                if data.get("data_fetch_failed_consecutively", 0) >= (api_key_manager.total_keys() if api_key_manager.total_keys() > 0 else 1) +1 :
                    data["last_attempt_after_all_keys_failed"] = datetime.now() # For cooldown timer
                    log_warning(f"Semua API key telah dicoba dan gagal untuk {pair_name}. Akan masuk cooldown.", pair_name=pair_name)

                if not fetch_update_successful_for_this_pair or not new_candles_batch:
                    if fetch_update_successful_for_this_pair and not new_candles_batch and not data["big_data_collection_phase_active"]:
                        log_warning(f"{AnsiColors.ORANGE}Tidak ada data candle baru diterima untuk {pair_name} meskipun fetch (dianggap) berhasil.{AnsiColors.ENDC}", pair_name=pair_name)
                    elif not fetch_update_successful_for_this_pair:
                         log_error(f"{AnsiColors.RED}Gagal mengambil update untuk {pair_name} setelah semua upaya di siklus ini.{AnsiColors.ENDC}", pair_name=pair_name)
                    min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
                    # NEW: CHART FEATURE - Update chart even on failed fetch, to update "last_update" timestamp
                    update_live_chart_data(pair_name, data["all_candles_list"], data["strategy_state"], config)
                    continue # Move to next pair

                # --- Merge New Candles and Process Logic ---
                merged_candles_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                newly_added_count_this_batch = 0
                updated_count_this_batch = 0 

                for candle in new_candles_batch:
                    ts = candle['timestamp']
                    if ts not in merged_candles_dict:
                        merged_candles_dict[ts] = candle
                        newly_added_count_this_batch +=1
                    elif merged_candles_dict[ts] != candle : # Check if content changed for existing timestamp
                        merged_candles_dict[ts] = candle
                        updated_count_this_batch +=1

                all_candles_list_temp = sorted(list(merged_candles_dict.values()), key=lambda c: c['timestamp'])
                actual_new_or_updated_count = newly_added_count_this_batch + updated_count_this_batch
                data["all_candles_list"] = all_candles_list_temp

                if actual_new_or_updated_count > 0:
                     log_info(f"{actual_new_or_updated_count} candle baru/diupdate. Total: {len(data['all_candles_list'])}.", pair_name=pair_name)
                elif new_candles_batch : # new_candles_batch was not empty, but no new timestamps or content changes
                     log_info("Tidak ada candle dengan timestamp baru atau update konten. Data terakhir mungkin identik.", pair_name=pair_name)

                # Update phase if big data collected
                if data["big_data_collection_phase_active"]:
                    if len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES:
                        log_info(f"{AnsiColors.GREEN}TARGET {TARGET_BIG_DATA_CANDLES} CANDLE TERCAPAI untuk {pair_name}!{AnsiColors.ENDC}", pair_name=pair_name)
                        if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Trim if over
                            data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:] 
                        
                        if not data["big_data_email_sent"]: # Send email
                            send_email_notification(f"Data Downloading Complete: {pair_name}", f"Data downloading complete for {TARGET_BIG_DATA_CANDLES} candles! Now trading on {pair_name}.", config)
                            data["big_data_email_sent"] = True
                        
                        data["big_data_collection_phase_active"] = False # Switch to live mode
                        active_cryptos_still_in_big_data_collection = max(0, active_cryptos_still_in_big_data_collection -1)
                        log_info(f"{AnsiColors.HEADER}---------- MULAI LIVE ANALYSIS ({len(data['all_candles_list'])} candles) untuk {pair_name} ----------{AnsiColors.ENDC}", pair_name=pair_name)
                else: # Live trading, ensure list doesn't grow indefinitely
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: 
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]

                # --- Run Strategy Logic ---
                min_len_for_pivots = config.get('left_strength',50) + config.get('right_strength',150) + 1
                if len(data["all_candles_list"]) >= min_len_for_pivots:
                    # Determine if logic should be run
                    process_logic_now = (actual_new_or_updated_count > 0 or # New/updated candles came in
                                         # Or, just transitioned from big data to live
                                         (not data["big_data_collection_phase_active"] and num_candles_before_fetch < TARGET_BIG_DATA_CANDLES and len(data["all_candles_list"]) >= TARGET_BIG_DATA_CANDLES) or
                                         # Or, still in big data but got new candles
                                         (data["big_data_collection_phase_active"] and newly_added_count_this_batch > 0) 
                                         )

                    if process_logic_now:
                         log_info(f"Menjalankan logika strategi dengan {len(data['all_candles_list'])} candle...", pair_name=pair_name)
                         data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"], global_settings_dict)
                    elif not data["big_data_collection_phase_active"]: # Live mode, but no new candles this cycle
                         last_c_time_str = data["all_candles_list"][-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if data["all_candles_list"] else "N/A"
                         log_info(f"Tidak ada candle baru untuk diproses untuk {pair_name}. Data terakhir @ {last_c_time_str}.", pair_name=pair_name)
                else: # Not enough data for pivot logic
                    log_info(f"Data ({len(data['all_candles_list'])}) untuk {pair_name} belum cukup utk analisa (min: {min_len_for_pivots}).", pair_name=pair_name)
                
                # NEW: CHART FEATURE - Update chart data for this pair after processing
                update_live_chart_data(pair_name, data["all_candles_list"], data["strategy_state"], config)
                
                min_overall_next_refresh_seconds = min(min_overall_next_refresh_seconds, required_interval_for_this_pair)
            # --- End of loop for one pair ---
        
        # --- Determine Sleep Duration for Main Loop ---
        sleep_duration = 15 # Default sleep if other calculations fail

        if not any_data_fetched_this_cycle and api_key_manager.get_current_key() is None:
            # All API keys failed globally, and no data was fetched in the entire cycle
            log_error("Semua API key gagal secara global dan tidak ada data berhasil di-fetch. Menunggu 1 jam sebelum mencoba lagi semua proses.", pair_name="SYSTEM")
            sleep_duration = 3600 # Wait 1 hour
        elif active_cryptos_still_in_big_data_collection > 0:
            # If some pairs are still collecting big data, use their (faster) interval,
            # but don't sleep too long to allow other live pairs to update if needed.
            min_big_data_interval = float('inf')
            for pid_loop, pdata_loop in crypto_data_manager.items():
                if pdata_loop["big_data_collection_phase_active"]:
                    pconfig_loop = pdata_loop["config"]
                    interval_bd = 55 if pconfig_loop.get('timeframe') == "minute" else (3600 * 23.8 if pconfig_loop.get('timeframe') == "day" else 3580)
                    min_big_data_interval = min(min_big_data_interval, interval_bd)
            
            # Sleep for the minimum of the big data intervals, or a general short interval if that's too long.
            # This ensures pairs in big data phase get priority.
            sleep_duration = min(min_big_data_interval if min_big_data_interval != float('inf') else 30, 30) # Max 30s sleep if big data active
            log_debug(f"Masih ada {active_cryptos_still_in_big_data_collection} pair dalam pengumpulan BIG DATA. Sleep {sleep_duration}s.", pair_name="SYSTEM")
        else: # All pairs are in live trading mode
            if min_overall_next_refresh_seconds != float('inf') and min_overall_next_refresh_seconds > 0 :
                # Sleep until the next pair needs a refresh, but at least MIN_REFRESH_INTERVAL_AFTER_BIG_DATA
                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_overall_next_refresh_seconds))
                log_debug(f"Semua pair live. Tidur ~{sleep_duration}s sampai refresh berikutnya.", pair_name="SYSTEM")
            else: # Fallback if min_overall_next_refresh_seconds is weird
                default_refresh_from_config = 60 # Default
                if all_crypto_configs : # Use first active config's interval as a guess
                    default_refresh_from_config = all_crypto_configs[0].get('refresh_interval_seconds', 60)

                sleep_duration = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, default_refresh_from_config)
                log_debug(f"Default sleep {sleep_duration}s (fallback atau interval pair pertama).", pair_name="SYSTEM")

        if sleep_duration > 0:
            show_spinner(sleep_duration, f"Menunggu {int(sleep_duration)}s...")
        else: # Should not happen, but defensive
            log_debug("Sleep duration 0 atau negatif, menggunakan 1s default.", pair_name="SYSTEM")
            time.sleep(1)

    except KeyboardInterrupt:
        animated_text_display(f"\n{AnsiColors.ORANGE}Proses trading dihentikan oleh pengguna.{AnsiColors.ENDC}", color=AnsiColors.ORANGE, delay=0.01)
    except Exception as e:
        log_error(f"{AnsiColors.RED}Error tak terduga di loop trading utama: {e}{AnsiColors.ENDC}", pair_name="SYSTEM")
        log_exception("Traceback Error:", pair_name="SYSTEM") # Log full traceback
    finally:
        animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}", color=AnsiColors.HEADER, delay=0.005)
        animated_text_display("Tekan Enter untuk kembali ke menu utama...", color=AnsiColors.ORANGE, delay=0.01)
        input() # Wait for user before returning to main menu


# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()

    while True:
        clear_screen_animated() 
        animated_text_display("========= Crypto Strategy Runner (Multi + Key Recovery + Chart) =========", color=AnsiColors.HEADER, delay=0.005)

        pick_title_main = "" # Build title for pick
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

        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += f"Target Data per Pair: {TARGET_BIG_DATA_CANDLES} candle\n"
        pick_title_main += f"Primary API Key: {primary_key_display} | Recovery Keys: {num_recovery_keys}\n"
        pick_title_main += f"Notifikasi Termux: {termux_notif_main_status}\n"
        pick_title_main += f"Live Chart URL: http://localhost:{FLASK_PORT} (atau IP perangkat Anda)\n" # NEW: Chart URL info
        pick_title_main += "-----------------------------------------------\n"
        pick_title_main += "Pilih Opsi:"

        options_plain = [
            "Mulai Analisa Realtime Semua Pair Aktif",
            "Pengaturan",
            "Keluar"
        ]
        
        selected_index = -1 # Default to invalid index
        try:
            _option_text, selected_index = pick(options_plain, pick_title_main, indicator='=>', default_index=0)
        except Exception as e_pick_main: # Fallback for pick library issues
            log_error(f"Error dengan library 'pick' di menu utama: {e_pick_main}. Gunakan input manual.")
            print(pick_title_main) # Show the menu text
            for idx_main, opt_text_main in enumerate(options_plain):
                print(f"  {idx_main + 1}. {opt_text_main}")
            try:
                choice_main = int(input("Pilih nomor opsi: ")) -1
                if 0 <= choice_main < len(options_plain):
                    selected_index = choice_main
                else:
                    print(f"{AnsiColors.RED}Pilihan tidak valid.{AnsiColors.ENDC}")
                    show_spinner(1.5, "Kembali...")
                    continue
            except ValueError:
                print(f"{AnsiColors.RED}Input harus berupa angka.{AnsiColors.ENDC}")
                show_spinner(1.5, "Kembali...")
                continue

        if selected_index == 0: # Mulai Analisa
            start_trading(settings)
        elif selected_index == 1: # Pengaturan
            settings = settings_menu(settings) # Update settings if they were changed
        elif selected_index == 2: # Keluar
            log_info("Aplikasi ditutup.", pair_name="SYSTEM")
            clear_screen_animated()
            animated_text_display("Terima kasih telah menggunakan skrip ini! Sampai jumpa!", color=AnsiColors.MAGENTA)
            show_spinner(0.5, "Exiting")
            break # Exit main loop

if __name__ == "__main__":
    try:
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
        # Ensure Flask server thread is handled if necessary, though daemon should make it stop.
        # For a very robust shutdown, you might want to signal the Flask thread to stop.
        # However, for Termux, daemon=True is usually sufficient.
        log_info("Skrip utama selesai.", "SYSTEM")
