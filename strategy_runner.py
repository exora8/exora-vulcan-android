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
import subprocess

# --- LIBRARY BARU UNTUK FITUR CHART ---
import threading
from flask import Flask, jsonify, render_template_string, request
import plotly.graph_objects as go
from collections import deque

# --- ANSI COLOR CODES ---
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
TARGET_BIG_DATA_CANDLES = 2500
MIN_REFRESH_INTERVAL_AFTER_BIG_DATA = 15

# CHART FEATURE Constants
CHART_WINDOW_SIZE = TARGET_BIG_DATA_CANDLES # PERMINTAAN PENGGUNA: Menampilkan semua 2500 candle di chart
LIVE_CHART_DATA = {}
FLASK_PORT = 5001

# --- FUNGSI UNTUK FITUR CHART ---
def update_live_chart_data(pair_name_chart, candles_history_chart, strategy_state_chart, config_chart):
    global LIVE_CHART_DATA
    if pair_name_chart not in LIVE_CHART_DATA:
        LIVE_CHART_DATA[pair_name_chart] = {
            'candles': deque(maxlen=CHART_WINDOW_SIZE), # deque akan otomatis trim jika lebih dari maxlen
            'fib_level': None,
            'buy_info': None,
            'sl_price': None,
            'pivots_high_chart': [],
            'pivots_low_chart': [],
            'last_update': datetime.now()
        }

    # 1. Update Candles (gunakan CHART_WINDOW_SIZE yang sudah diset ke TARGET_BIG_DATA_CANDLES)
    start_index = max(0, len(candles_history_chart) - CHART_WINDOW_SIZE)
    candles_to_display_objects = candles_history_chart[start_index:]
    
    LIVE_CHART_DATA[pair_name_chart]['candles'].clear()
    for c_obj in candles_to_display_objects:
        if c_obj.get('timestamp') and c_obj.get('open') is not None and \
           c_obj.get('high') is not None and c_obj.get('low') is not None and \
           c_obj.get('close') is not None:
            LIVE_CHART_DATA[pair_name_chart]['candles'].append({
                'time': c_obj['timestamp'].timestamp() * 1000,
                'open': c_obj['open'], 'high': c_obj['high'],
                'low': c_obj['low'], 'close': c_obj['close']
            })

    # 2. Update FIB Level
    if strategy_state_chart.get("active_fib_level") is not None:
        LIVE_CHART_DATA[pair_name_chart]['fib_level'] = strategy_state_chart["active_fib_level"]
    elif strategy_state_chart.get("position_size", 0) == 0:
        LIVE_CHART_DATA[pair_name_chart]['fib_level'] = None

    # 3. Update Buy Info and SL Prices
    LIVE_CHART_DATA[pair_name_chart]['buy_info'] = None
    if strategy_state_chart.get("position_size", 0) > 0:
        entry_price = strategy_state_chart.get("entry_price_custom")
        entry_ts_from_state = strategy_state_chart.get("entry_timestamp_custom")
        if entry_price is not None:
            LIVE_CHART_DATA[pair_name_chart]['buy_info'] = {'price': entry_price}
            if entry_ts_from_state:
                 LIVE_CHART_DATA[pair_name_chart]['buy_info']['time'] = entry_ts_from_state.timestamp() * 1000
        
        plot_stop_level = strategy_state_chart.get("emergency_sl_level_custom")
        if strategy_state_chart.get("trailing_tp_active_custom") and strategy_state_chart.get("current_trailing_stop_level") is not None:
            if plot_stop_level is None or strategy_state_chart.get("current_trailing_stop_level") > plot_stop_level:
                plot_stop_level = strategy_state_chart.get("current_trailing_stop_level")
        LIVE_CHART_DATA[pair_name_chart]['sl_price'] = plot_stop_level
    else:
        LIVE_CHART_DATA[pair_name_chart]['sl_price'] = None

    # 4. Update Pivots untuk chart window saat ini
    LIVE_CHART_DATA[pair_name_chart]['pivots_high_chart'] = []
    LIVE_CHART_DATA[pair_name_chart]['pivots_low_chart'] = []
    
    min_pivot_data_len = config_chart.get('left_strength',0) + config_chart.get('right_strength',0) + 1
    if candles_to_display_objects and len(candles_to_display_objects) >= min_pivot_data_len:
        display_high_prices = [c.get('high') for c in candles_to_display_objects]
        display_low_prices = [c.get('low') for c in candles_to_display_objects]
        
        if not (any(p is None for p in display_high_prices) or any(p is None for p in display_low_prices)):
            chart_pivots_h_prices = find_pivots(display_high_prices, config_chart['left_strength'], config_chart['right_strength'], True)
            chart_pivots_l_prices = find_pivots(display_low_prices, config_chart['left_strength'], config_chart['right_strength'], False)

            for i, price in enumerate(chart_pivots_h_prices):
                if price is not None and i < len(candles_to_display_objects):
                    LIVE_CHART_DATA[pair_name_chart]['pivots_high_chart'].append({
                        'time': candles_to_display_objects[i]['timestamp'].timestamp() * 1000, 'price': price
                    })
            for i, price in enumerate(chart_pivots_l_prices):
                if price is not None and i < len(candles_to_display_objects):
                    LIVE_CHART_DATA[pair_name_chart]['pivots_low_chart'].append({
                        'time': candles_to_display_objects[i]['timestamp'].timestamp() * 1000, 'price': price
                    })
    LIVE_CHART_DATA[pair_name_chart]['last_update'] = datetime.now()

# --- FLASK APP ---
app = Flask(__name__)
INDEX_HTML = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Crypto Charts</title><style>body{font-family:Arial,sans-serif;margin:20px;background-color:#2c2c2c;color:#e0e0e0}h1{text-align:center;color:#e0e0e0}ul{list-style-type:none;padding:0}li{margin:10px 0}a{text-decoration:none;color:#58a6ff;padding:10px 15px;border:1px solid #58a6ff;border-radius:5px;display:block;text-align:center;transition:background-color .3s,color .3s}a:hover{background-color:#58a6ff;color:#2c2c2c}.container{max-width:600px;margin:auto;background:#3c3c3c;padding:20px;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,.5)}.note{text-align:center;margin-top:20px;font-size:.9em;color:#aaa}</style></head>
<body><div class="container"><h1>Available Live Charts</h1>
{% if pairs %}<ul>{% for pair in pairs %}<li><a href="{{ url_for('show_chart_page', pair_name_url=pair.replace('-', '_')) }}">{{ pair }}</a></li>{% endfor %}</ul>
{% else %}<p class="note">No active trading pairs found or data not yet available. Please wait for the script to initialize and fetch data.</p>{% endif %}
<p class="note">Charts will be available after initial data fetch. Displaying {{ chart_candle_count }} candles per chart. Large candle counts may impact browser performance.</p>
<p class="note">Refresh this page if a pair is missing.</p></div></body></html>
"""
CHART_PAGE_HTML = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Chart: {{ pair_name }}</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{font-family:Arial,sans-serif;margin:0;padding:0;background-color:#1e1e1e;color:#d4d4d4}#chartContainer{width:98%;height:85vh;margin:10px auto}h1{text-align:center;padding:10px;color:#d4d4d4;margin-bottom:5px}.info-bar{text-align:center;padding:5px;font-size:.9em;color:#aaa}.loader{border:8px solid #333;border-top:8px solid #58a6ff;border-radius:50%;width:50px;height:50px;animation:spin 1s linear infinite;margin:50px auto}@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}.message{text-align:center;margin-top:50px;font-size:1.1em;color:#aaa}</style></head>
<body><h1>Live Chart: {{ pair_name }}</h1><div class="info-bar">Server Data Last Updated: <span id="lastServerUpdate">-</span> | Chart Last Refreshed: <span id="lastChartRefresh">-</span></div>
<div id="chartContainer"><div class="loader"></div></div>
<script>
    const pairNameUrl="{{pair_name_url}}";const chartContainer=document.getElementById('chartContainer');let chartInitialized=false;
    async function fetchData(){try{const response=await fetch(`/chart_data/${pairNameUrl}`);if(!response.ok){console.error('Error fetching chart data:',response.status,await response.text());if(chartContainer.querySelector('.loader'))chartContainer.innerHTML='<p class="message">Error loading data. Is the trading script running and the pair active?</p>';return null}
    const data=await response.json();if(data.last_update){document.getElementById('lastServerUpdate').textContent=new Date(data.last_update).toLocaleString()}return data}catch(error){console.error('Failed to fetch:',error);if(chartContainer.querySelector('.loader'))chartContainer.innerHTML='<p class="message">Could not connect to data source. Ensure the script is running.</p>';return null}}
    function plotChart(data){if(!data||!data.candles||data.candles.length===0){if(chartContainer.querySelector('.loader'))chartContainer.innerHTML='<p class="message">No candle data available yet for {{pair_name}}. Waiting for data...</p>';console.warn("No candle data to plot for {{pair_name}}");return}
    if(chartContainer.querySelector('.loader')){chartContainer.innerHTML=''}
    const traceCandle={x:data.candles.map(c=>new Date(c.time)),open:data.candles.map(c=>c.open),high:data.candles.map(c=>c.high),low:data.candles.map(c=>c.low),close:data.candles.map(c=>c.close),type:'candlestick',name:'Candles',xaxis:'x',yaxis:'y'};
    const layout={title:{text:`{{pair_name}} Live Chart (Candles: ${data.candles.length})`,font:{color:'#d4d4d4'}},xaxis:{type:'date',rangeslider:{visible:false},color:'#aaa',gridcolor:'#444'},yaxis:{autorange:true,type:'linear',color:'#aaa',gridcolor:'#444',tickformat:'.8f'},shapes:[],annotations:[],paper_bgcolor:'#1e1e1e',plot_bgcolor:'#1e1e1e',font:{color:'#d4d4d4'},legend:{orientation:'h',yanchor:'bottom',y:1.02,xanchor:'right',x:1,font:{color:'#d4d4d4'}}};
    const plotDataArray=[traceCandle];
    if(data.fib_level){layout.shapes.push({type:'line',x0:traceCandle.x[0],y0:data.fib_level,x1:traceCandle.x[traceCandle.x.length-1],y1:data.fib_level,line:{color:'rgba(255,165,0,0.7)',width:2,dash:'dash'}});plotDataArray.push({x:[null],y:[null],mode:'lines',name:`FIB 0.5 (${data.fib_level.toFixed(5)})`,line:{color:'rgba(255,165,0,0.7)',width:2,dash:'dash'}})}
    if(data.buy_info&&data.buy_info.price){layout.shapes.push({type:'line',x0:traceCandle.x[0],y0:data.buy_info.price,x1:traceCandle.x[traceCandle.x.length-1],y1:data.buy_info.price,line:{color:'rgba(0,255,0,0.7)',width:2}});plotDataArray.push({x:[null],y:[null],mode:'lines',name:`BUY (${data.buy_info.price.toFixed(5)})`,line:{color:'rgba(0,255,0,0.7)',width:2}});if(data.buy_info.time){plotDataArray.push({x:[new Date(data.buy_info.time)],y:[data.buy_info.price],mode:'markers',type:'scatter',name:'Entry Point',marker:{color:'lime',size:10,symbol:'triangle-up'},hoverinfo:'text',text:`Entry@${data.buy_info.price.toFixed(5)}<br>${new Date(data.buy_info.time).toLocaleString()}`})}}
    if(data.sl_price){layout.shapes.push({type:'line',x0:traceCandle.x[0],y0:data.sl_price,x1:traceCandle.x[traceCandle.x.length-1],y1:data.sl_price,line:{color:'rgba(255,0,0,0.7)',width:2}});plotDataArray.push({x:[null],y:[null],mode:'lines',name:`SL (${data.sl_price.toFixed(5)})`,line:{color:'rgba(255,0,0,0.7)',width:2}})}
    if(data.pivots_high_chart&&data.pivots_high_chart.length>0){data.pivots_high_chart.forEach(ph=>{layout.annotations.push({x:new Date(ph.time),y:ph.price,xref:'x',yref:'y',text:'PH',showarrow:true,arrowhead:0,ax:0,ay:-25,font:{color:'orange',size:10},bordercolor:'orange',borderwidth:1,bgcolor:'rgba(0,0,0,0.5)'})})}
    if(data.pivots_low_chart&&data.pivots_low_chart.length>0){data.pivots_low_chart.forEach(pl=>{layout.annotations.push({x:new Date(pl.time),y:pl.price,xref:'x',yref:'y',text:'PL',showarrow:true,arrowhead:0,ax:0,ay:25,font:{color:'lightblue',size:10},bordercolor:'lightblue',borderwidth:1,bgcolor:'rgba(0,0,0,0.5)'})})}
    if(!chartInitialized){Plotly.newPlot('chartContainer',plotDataArray,layout,{responsive:true,displaylogo:false});chartInitialized=true}else{Plotly.react('chartContainer',plotDataArray,layout,{responsive:true,displaylogo:false})}
    document.getElementById('lastChartRefresh').textContent=new Date().toLocaleString()}
    async function updateLoop(){const chartData=await fetchData();if(chartData){plotChart(chartData)}
    let refreshInterval=15000;if(chartData&&chartData.last_update){const age_ms=new Date().getTime()-new Date(chartData.last_update).getTime();if(age_ms<20000)refreshInterval=3000;else if(age_ms<60000)refreshInterval=5000;else if(age_ms<300000)refreshInterval=10000}
    setTimeout(updateLoop,refreshInterval)}
    setTimeout(updateLoop,500);
</script></body></html>
"""
@app.route('/')
def index():
    global LIVE_CHART_DATA, CHART_WINDOW_SIZE
    active_pairs_with_data = sorted([pair for pair, data in LIVE_CHART_DATA.items() if data.get('candles') and len(data['candles']) > 0])
    return render_template_string(INDEX_HTML, pairs=active_pairs_with_data, chart_candle_count=CHART_WINDOW_SIZE)

@app.route('/chart/<pair_name_url>')
def show_chart_page(pair_name_url):
    actual_pair_name_candidate = pair_name_url.replace('_', '-')
    found_key = None
    if actual_pair_name_candidate in LIVE_CHART_DATA: found_key = actual_pair_name_candidate
    else:
        for key in LIVE_CHART_DATA.keys():
            if key.lower() == actual_pair_name_candidate.lower(): found_key = key; break
    if not found_key: return f"Chart data not found for {actual_pair_name_candidate}.", 404
    return render_template_string(CHART_PAGE_HTML, pair_name=found_key, pair_name_url=found_key.replace('-', '_'))

@app.route('/chart_data/<pair_name_url>')
def get_chart_data(pair_name_url):
    global LIVE_CHART_DATA
    actual_pair_name_candidate = pair_name_url.replace('_', '-')
    found_key = None
    if actual_pair_name_candidate in LIVE_CHART_DATA: found_key = actual_pair_name_candidate
    else:
        for key in LIVE_CHART_DATA.keys():
            if key.lower() == actual_pair_name_candidate.lower(): found_key = key; break
    if not found_key or not LIVE_CHART_DATA.get(found_key):
        return jsonify({"error": f"Pair data not found for {actual_pair_name_candidate}", "candles": []}), 404
    pair_data = LIVE_CHART_DATA[found_key]
    return jsonify({
        "candles": list(pair_data['candles']),
        "fib_level": pair_data.get('fib_level'),
        "buy_info": pair_data.get('buy_info'),
        "sl_price": pair_data.get('sl_price'),
        "pivots_high_chart": pair_data.get('pivots_high_chart', []),
        "pivots_low_chart": pair_data.get('pivots_low_chart', []),
        "last_update": pair_data.get('last_update').isoformat() if pair_data.get('last_update') else None
    })

def run_flask_app():
    try:
        log_info(f"Starting live chart web server on http://0.0.0.0:{FLASK_PORT}", "WEBSERVER")
        if CHART_WINDOW_SIZE > 500: # Peringatan jika chart window sangat besar
            log_warning(f"{AnsiColors.ORANGE}PERINGATAN: Chart di web akan menampilkan {CHART_WINDOW_SIZE} candle. Ini bisa sangat berat untuk browser Anda, terutama pada perangkat seluler atau PC dengan spesifikasi rendah. Pertimbangkan mengurangi `CHART_WINDOW_SIZE` jika ada masalah performa.{AnsiColors.ENDC}", "WEBSERVER")
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)
    except OSError as e:
        if e.errno == 98: log_error(f"Port {FLASK_PORT} sudah digunakan. Server chart tidak bisa dimulai.", "WEBSERVER")
        else: log_error(f"Gagal memulai server web: {e}", "WEBSERVER")
    except Exception as e:
        log_error(f"Error tak terduga pada server web: {e}", "WEBSERVER")
        log_exception("Traceback Web Server Error:", "WEBSERVER")

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
        self.current_index = 0
        self.global_email_settings = global_settings_for_email or {}
        if not self.keys: log_warning("Tidak ada API key valid dikonfigurasi.")
    def get_current_key(self):
        if not self.keys or self.current_index >= len(self.keys): return None
        return self.keys[self.current_index]
    def switch_to_next_key(self):
        if not self.keys: return None
        self.current_index += 1
        if self.current_index < len(self.keys):
            new_key_display = self.keys[self.current_index][:5] + "..." + self.keys[self.current_index][-3:] if len(self.keys[self.current_index]) > 8 else self.keys[self.current_index]
            log_info(f"{AnsiColors.ORANGE}Beralih ke API key berikutnya: Index {self.current_index} ({new_key_display}){AnsiColors.ENDC}")
            # ... (logika email notifikasi untuk switch key) ...
            return self.keys[self.current_index]
        else:
            log_error(f"{AnsiColors.RED}{AnsiColors.BOLD}SEMUA API KEY TELAH HABIS/GAGAL!{AnsiColors.ENDC}")
            # ... (logika email notifikasi untuk semua key gagal) ...
            return None
    def has_valid_keys(self): return bool(self.keys)
    def total_keys(self): return len(self.keys)
    def get_current_key_index(self): return self.current_index

# --- FUNGSI BEEP, EMAIL & TERMUX NOTIFICATION ---
def play_notification_sound():
    try:
        if sys.platform == "win32": import winsound; winsound.Beep(1000, 500)
        else: print('\a', end='', flush=True)
    except Exception as e: log_warning(f"Tidak bisa memainkan suara notifikasi: {e}")

def send_email_notification(subject, body_text, settings_for_email):
    if not settings_for_email.get("enable_email_notifications", False): return
    sender_email = settings_for_email.get("email_sender_address")
    sender_password = settings_for_email.get("email_sender_app_password")
    receiver_email = settings_for_email.get("email_receiver_address")
    if not all([sender_email, sender_password, receiver_email]):
        log_warning(f"Konfigurasi email tidak lengkap. Notifikasi email dilewati.", pair_name=settings_for_email.get('pair_name', 'GLOBAL'))
        return
    msg = MIMEText(body_text); msg['Subject'] = subject; msg['From'] = sender_email; msg['To'] = receiver_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s: s.login(sender_email, sender_password); s.sendmail(sender_email, receiver_email, msg.as_string())
        log_info(f"{AnsiColors.CYAN}Email notifikasi dikirim ke {receiver_email}{AnsiColors.ENDC}", pair_name=settings_for_email.get('pair_name', 'GLOBAL'))
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal mengirim email: {e}{AnsiColors.ENDC}", pair_name=settings_for_email.get('pair_name', 'GLOBAL'))

def send_termux_notification(title, content_msg, global_settings, pair_name_for_log="SYSTEM"):
    if not global_settings.get("api_settings", {}).get("enable_termux_notifications", False): return
    try:
        subprocess.run(['termux-notification', '--title', title, '--content', content_msg], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_info(f"{AnsiColors.CYAN}Notifikasi Termux dikirim: '{title}'{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except FileNotFoundError: log_warning(f"{AnsiColors.ORANGE}'termux-notification' tidak ditemukan. Pastikan Termux:API terinstal (pkg install termux-api).{AnsiColors.ENDC}", pair_name=pair_name_for_log)
    except Exception as e: log_error(f"{AnsiColors.RED}Gagal mengirim notifikasi Termux: {e}{AnsiColors.ENDC}", pair_name=pair_name_for_log)

# --- FUNGSI PENGATURAN (SETTINGS) ---
def get_default_crypto_config():
    return {"id": str(uuid.uuid4()), "enabled": True, "symbol": "BTC", "currency": "USD", "exchange": "CCCAGG", "timeframe": "hour", "refresh_interval_seconds": 60, "left_strength": 50, "right_strength": 150, "profit_target_percent_activation": 5.0, "trailing_stop_gap_percent": 5.0, "emergency_sl_percent": 10.0, "enable_secure_fib": True, "secure_fib_check_price": "Close", "enable_email_notifications": False, "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address": ""}
def load_settings():
    default_api = {"primary_key": "YOUR_PRIMARY_KEY", "recovery_keys": [], "enable_global_email_notifications_for_key_switch": False, "email_sender_address": "", "email_sender_app_password": "", "email_receiver_address_admin": "", "enable_termux_notifications": False}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            try:
                s = json.load(f)
                if "api_settings" not in s: s["api_settings"] = default_api.copy()
                else:
                    for k, v in default_api.items():
                        if k not in s["api_settings"]: s["api_settings"][k] = v
                if "cryptos" not in s or not isinstance(s["cryptos"], list): s["cryptos"] = []
                for cfg in s["cryptos"]:
                    if "id" not in cfg: cfg["id"] = str(uuid.uuid4())
                    if "enabled" not in cfg: cfg["enabled"] = True
                return s
            except json.JSONDecodeError: log_error(f"Error membaca {SETTINGS_FILE}. Menggunakan default."); return {"api_settings": default_api.copy(), "cryptos": [get_default_crypto_config()]}
    return {"api_settings": default_api.copy(), "cryptos": [get_default_crypto_config()]}
def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
    log_info(f"{AnsiColors.CYAN}Pengaturan disimpan ke {SETTINGS_FILE}{AnsiColors.ENDC}")
def _prompt_crypto_config(current_config):
    # ... (Implementasi _prompt_crypto_config dari jawaban sebelumnya, tidak perlu diubah signifikan untuk ini)
    # Cukup pastikan ia mengembalikan dictionary new_config yang valid
    clear_screen_animated()
    new_config = current_config.copy()
    animated_text_display(f"--- Konfigurasi Crypto Pair ({new_config.get('symbol','BARU')}-{new_config.get('currency','BARU')}) ---", color=AnsiColors.HEADER)
    enabled_input = input(f"Aktifkan pair? (true/false) [{new_config.get('enabled',True)}]: ").lower().strip()
    new_config["enabled"] = True if enabled_input == 'true' else (False if enabled_input == 'false' else new_config.get('enabled',True))
    new_config["symbol"] = (input(f"Simbol Crypto [{new_config.get('symbol','BTC')}]: ") or new_config.get('symbol','BTC')).upper().strip()
    new_config["currency"] = (input(f"Mata Uang Quote [{new_config.get('currency','USD')}]: ") or new_config.get('currency','USD')).upper().strip()
    new_config["exchange"] = (input(f"Exchange [{new_config.get('exchange','CCCAGG')}]: ") or new_config.get('exchange','CCCAGG')).strip()
    tf_input = (input(f"Timeframe (minute/hour/day) [{new_config.get('timeframe','hour')}]: ") or new_config.get('timeframe','hour')).lower().strip()
    if tf_input in ['minute','hour','day']: new_config["timeframe"] = tf_input
    # ... Lanjutkan dengan semua field lainnya seperti di skrip asli Anda ...
    try:
        new_config["refresh_interval_seconds"] = int(input(f"Interval Refresh (s) [{new_config.get('refresh_interval_seconds',60)}]: ") or new_config.get('refresh_interval_seconds',60))
        new_config["left_strength"] = int(input(f"Left Strength [{new_config.get('left_strength',50)}]: ") or new_config.get('left_strength',50))
        new_config["right_strength"] = int(input(f"Right Strength [{new_config.get('right_strength',150)}]: ") or new_config.get('right_strength',150))
        new_config["profit_target_percent_activation"] = float(input(f"Profit % Aktivasi Trail TP [{new_config.get('profit_target_percent_activation',5.0)}]: ") or new_config.get('profit_target_percent_activation',5.0))
        new_config["trailing_stop_gap_percent"] = float(input(f"Gap Trail TP % [{new_config.get('trailing_stop_gap_percent',5.0)}]: ") or new_config.get('trailing_stop_gap_percent',5.0))
        new_config["emergency_sl_percent"] = float(input(f"Emergency SL % [{new_config.get('emergency_sl_percent',10.0)}]: ") or new_config.get('emergency_sl_percent',10.0))
    except ValueError: print(f"{AnsiColors.RED}Input angka tidak valid, menggunakan default.{AnsiColors.ENDC}")
    # ... (Secure FIB dan Email settings) ...
    return new_config
def settings_menu(current_settings):
    # ... (Implementasi settings_menu dari jawaban sebelumnya, tidak perlu diubah signifikan) ...
    # Cukup pastikan ia memanggil _prompt_crypto_config dan save_settings dengan benar
    while True:
        clear_screen_animated()
        # ... (Tampilkan status API, crypto, dll.) ...
        options = [ "Atur Primary API Key", "Kelola Recovery API Keys", "Atur Email Global", "Notifikasi Termux", "Tambah Crypto", "Ubah Crypto", "Hapus Crypto", "Kembali"]
        # Gunakan `pick` atau input manual seperti di skrip asli Anda
        try:
            option_text, index = pick(options, "--- Menu Pengaturan ---", indicator='=>')
        except: # Fallback jika pick error
            for i, opt in enumerate(options): print(f"{i+1}. {opt}")
            try: index = int(input("Pilih: ")) - 1
            except: continue

        if index == 7: break # Kembali
        # ... (Handle pilihan lainnya) ...
        if index == 0: # Atur Primary API
            current_settings["api_settings"]["primary_key"] = input("Primary API Key baru: ").strip() or current_settings["api_settings"]["primary_key"]
        # ... (Lanjutkan implementasi untuk semua opsi) ...
        save_settings(current_settings) # Simpan setelah perubahan
    return current_settings


# --- FUNGSI PENGAMBILAN DATA (fetch_candles) ---
# Implementasi fetch_candles dari jawaban sebelumnya (sudah cukup baik)
def fetch_candles(symbol, currency, total_limit_desired, exchange_name, current_api_key_to_use, timeframe="hour", pair_name="N/A"):
    if not current_api_key_to_use: raise APIKeyError("API Key tidak tersedia.")
    all_candles, to_ts, endpoint = [], None, "histohour"
    if timeframe == "minute": endpoint = "histominute"
    elif timeframe == "day": endpoint = "histoday"
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    is_large = total_limit_desired > 10
    if is_large: log_info(f"Fetching {total_limit_desired} {timeframe} candles for {pair_name}...", pair_name=pair_name)
    if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(0, total_limit_desired, prefix=f'{pair_name} Data:', length=30)

    while len(all_candles) < total_limit_desired:
        needed = total_limit_desired - len(all_candles)
        limit = min(needed + 1 if to_ts and needed > 1 else needed, CRYPTOCOMPARE_MAX_LIMIT)
        if limit <= 0: break
        params = {"fsym": symbol, "tsym": currency, "limit": limit, "api_key": current_api_key_to_use}
        if exchange_name and exchange_name.upper() != "CCCAGG": params["e"] = exchange_name
        if to_ts: params["toTs"] = to_ts
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code in [401, 403, 429]:
                err_msg = r.json().get('Message', f"HTTP Error {r.status_code}")
                log_warning(f"{AnsiColors.RED}API Key Error (HTTP {r.status_code}): {err_msg}{AnsiColors.ENDC}", pair_name=pair_name)
                raise APIKeyError(f"HTTP {r.status_code}: {err_msg}")
            r.raise_for_status()
            data = r.json()
            if data.get('Response') == 'Error':
                err_msg = data.get('Message', 'Unknown API Error')
                # ... (pengecekan key_related_error_messages) ...
                if any(k.lower() in err_msg.lower() for k in ["api key is invalid", "over_the_limit", "rate limit", "pro_tier_has_expired"]):
                    raise APIKeyError(f"JSON Error: {err_msg}")
                else: log_error(f"{AnsiColors.RED}API Error: {err_msg}{AnsiColors.ENDC}", pair_name=pair_name); break
            if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']: break
            
            raw_api_candles = data['Data']['Data']
            batch = []
            for item in raw_api_candles:
                if not all(k in item for k in ['time','open','high','low','close']): continue
                try: batch.append({'timestamp':datetime.fromtimestamp(item['time']),'open':float(item['open']),'high':float(item['high']),'low':float(item['low']),'close':float(item['close']),'volume':float(item.get('volumefrom',0))})
                except: continue # Skip bad data
            
            if to_ts and all_candles and batch and batch[-1]['timestamp'] == all_candles[0]['timestamp']: batch.pop()
            if not batch and to_ts: break
            all_candles = batch + all_candles
            if raw_api_candles: to_ts = raw_api_candles[0]['time']
            else: break
            if total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_candles), total_limit_desired, prefix=f'{pair_name} Data:', length=30)
            if len(raw_api_candles) < limit: break
            if len(all_candles) >= total_limit_desired: break
            if len(all_candles) < total_limit_desired and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: time.sleep(0.2)
        except APIKeyError: raise
        except requests.exceptions.RequestException as e: log_error(f"{AnsiColors.RED}Connection error: {e}{AnsiColors.ENDC}", pair_name=pair_name); break
        except Exception as e: log_error(f"{AnsiColors.RED}Fetch error: {e}{AnsiColors.ENDC}", pair_name=pair_name); log_exception("Trace:", pair_name=pair_name); break
    if len(all_candles) > total_limit_desired: all_candles = all_candles[-total_limit_desired:]
    if is_large and total_limit_desired > CRYPTOCOMPARE_MAX_LIMIT: simple_progress_bar(len(all_candles), total_limit_desired, prefix=f'{pair_name} Data:', suffix=' Done', length=30)
    elif is_large: log_info(f"Fetched {len(all_candles)} candles.", pair_name=pair_name)
    return all_candles

# --- LOGIKA STRATEGI ---
def get_initial_strategy_state():
    return {
        "last_signal_type": 0, "final_pivot_high_price_confirmed": None, "final_pivot_low_price_confirmed": None,
        "high_price_for_fib": None, "high_bar_index_for_fib": None, "active_fib_level": None,
        "active_fib_line_start_index": None, "entry_price_custom": None, "entry_timestamp_custom": None, # Ditambahkan entry_timestamp_custom
        "highest_price_for_trailing": None, "trailing_tp_active_custom": False,
        "current_trailing_stop_level": None, "emergency_sl_level_custom": None, "position_size": 0,
    }
def find_pivots(series_list, left_strength, right_strength, is_high=True):
    # ... (Implementasi find_pivots dari jawaban sebelumnya, sudah baik) ...
    pivots = [None] * len(series_list)
    if len(series_list) < left_strength + right_strength + 1: return pivots
    for i in range(left_strength, len(series_list) - right_strength):
        is_pivot = True; current_val = series_list[i]
        if current_val is None: continue
        for j in range(1, left_strength + 1): # Left side
            if series_list[i-j] is None or (is_high and current_val <= series_list[i-j]) or (not is_high and current_val >= series_list[i-j]): is_pivot=False; break
        if not is_pivot: continue
        for j in range(1, right_strength + 1): # Right side
            if series_list[i+j] is None or (is_high and current_val < series_list[i+j]) or (not is_high and current_val > series_list[i+j]): is_pivot=False; break
        if is_pivot: pivots[i] = current_val
    return pivots

def run_strategy_logic(candles_history, crypto_config, strategy_state, global_settings):
    pair_name = f"{crypto_config['symbol']}-{crypto_config['currency']}"
    strategy_state["final_pivot_high_price_confirmed"] = None
    strategy_state["final_pivot_low_price_confirmed"] = None
    # ... (Validasi candles_history)
    if not candles_history or not all(k in candles_history[0] for k in ['high','low','open','close','timestamp']): return strategy_state

    high_prices = [c.get('high') for c in candles_history]
    low_prices = [c.get('low') for c in candles_history]
    raw_pivot_highs = find_pivots(high_prices, crypto_config['left_strength'], crypto_config['right_strength'], True)
    raw_pivot_lows = find_pivots(low_prices, crypto_config['left_strength'], crypto_config['right_strength'], False)
    
    current_bar_idx = len(candles_history) - 1
    if current_bar_idx < 0: return strategy_state
    
    idx_pivot_event = current_bar_idx - crypto_config['right_strength']
    
    # Pivot High Confirmation
    if 0 <= idx_pivot_event < len(raw_pivot_highs) and raw_pivot_highs[idx_pivot_event] is not None and strategy_state["last_signal_type"] != 1:
        strategy_state["final_pivot_high_price_confirmed"] = raw_pivot_highs[idx_pivot_event]
        strategy_state["last_signal_type"] = 1
        # ... (log_info untuk PH)
        log_info(f"{AnsiColors.CYAN}PIVOT HIGH: {strategy_state['final_pivot_high_price_confirmed']:.5f} @ {candles_history[idx_pivot_event]['timestamp']:%Y-%m-%d %H:%M}{AnsiColors.ENDC}", pair_name=pair_name)


    # Pivot Low Confirmation
    if 0 <= idx_pivot_event < len(raw_pivot_lows) and raw_pivot_lows[idx_pivot_event] is not None and strategy_state["last_signal_type"] != -1:
        strategy_state["final_pivot_low_price_confirmed"] = raw_pivot_lows[idx_pivot_event]
        strategy_state["last_signal_type"] = -1
        # ... (log_info untuk PL)
        log_info(f"{AnsiColors.CYAN}PIVOT LOW: {strategy_state['final_pivot_low_price_confirmed']:.5f} @ {candles_history[idx_pivot_event]['timestamp']:%Y-%m-%d %H:%M}{AnsiColors.ENDC}", pair_name=pair_name)

    current_candle = candles_history[current_bar_idx]
    if any(current_candle.get(k) is None for k in ['open', 'high', 'low', 'close']): return strategy_state # Skip jika data candle tidak lengkap

    # Fibonacci Logic
    if strategy_state["final_pivot_high_price_confirmed"] is not None:
        strategy_state["high_price_for_fib"] = strategy_state["final_pivot_high_price_confirmed"]
        strategy_state["high_bar_index_for_fib"] = idx_pivot_event # Index PH
        if strategy_state["active_fib_level"] is not None: strategy_state["active_fib_level"] = None # Reset FIB

    if strategy_state["final_pivot_low_price_confirmed"] is not None and \
       strategy_state["high_price_for_fib"] is not None and \
       idx_pivot_event > strategy_state.get("high_bar_index_for_fib", -1): # PL setelah PH
        
        ph_price = strategy_state["high_price_for_fib"]
        pl_price = strategy_state["final_pivot_low_price_confirmed"]
        calc_fib = (ph_price + pl_price) / 2.0
        
        is_late = False
        if crypto_config["enable_secure_fib"]:
            check_price = current_candle.get(crypto_config["secure_fib_check_price"].lower(), current_candle['close'])
            if check_price is not None and check_price > calc_fib: is_late = True
        
        if is_late: log_info(f"{AnsiColors.ORANGE}FIB Terlambat ({calc_fib:.5f}){AnsiColors.ENDC}", pair_name=pair_name)
        else:
            log_info(f"{AnsiColors.CYAN}FIB 0.5 Aktif: {calc_fib:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
            strategy_state["active_fib_level"] = calc_fib
            strategy_state["active_fib_line_start_index"] = idx_pivot_event # Index PL
        
        strategy_state["high_price_for_fib"] = None # Reset PH untuk FIB

    # Entry Logic
    if strategy_state["active_fib_level"] is not None and strategy_state["position_size"] == 0:
        if current_candle['close'] > current_candle['open'] and current_candle['close'] > strategy_state["active_fib_level"]:
            entry_px = current_candle['close']
            strategy_state.update({
                "position_size": 1, "entry_price_custom": entry_px,
                "entry_timestamp_custom": current_candle['timestamp'], # Simpan timestamp entry
                "highest_price_for_trailing": entry_px, "trailing_tp_active_custom": False,
                "current_trailing_stop_level": None,
                "emergency_sl_level_custom": entry_px * (1 - crypto_config["emergency_sl_percent"] / 100.0),
                "active_fib_level": None # Deaktivasi FIB setelah entry
            })
            log_info(f"{AnsiColors.GREEN}{AnsiColors.BOLD}BUY ENTRY @ {entry_px:.5f}{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()
            # ... (Termux & Email notifikasi untuk BUY) ...

    # Position Management (Trailing TP & SL)
    if strategy_state["position_size"] > 0:
        strategy_state["highest_price_for_trailing"] = max(strategy_state.get("highest_price_for_trailing", current_candle['high']), current_candle['high'])
        
        if not strategy_state["trailing_tp_active_custom"] and strategy_state["entry_price_custom"]:
            profit_pct = ((strategy_state["highest_price_for_trailing"] - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            if profit_pct >= crypto_config["profit_target_percent_activation"]:
                strategy_state["trailing_tp_active_custom"] = True
                # ... (log_info untuk aktivasi Trailing TP) ...
        
        if strategy_state["trailing_tp_active_custom"]:
            new_trail_sl = strategy_state["highest_price_for_trailing"] * (1 - crypto_config["trailing_stop_gap_percent"] / 100.0)
            if strategy_state["current_trailing_stop_level"] is None or new_trail_sl > strategy_state["current_trailing_stop_level"]:
                strategy_state["current_trailing_stop_level"] = new_trail_sl
        
        final_sl = strategy_state["emergency_sl_level_custom"]
        exit_reason = "Emergency SL"
        if strategy_state["trailing_tp_active_custom"] and strategy_state["current_trailing_stop_level"] is not None:
            if final_sl is None or strategy_state["current_trailing_stop_level"] > final_sl:
                final_sl = strategy_state["current_trailing_stop_level"]
                exit_reason = "Trailing Stop"
        
        if final_sl is not None and current_candle['low'] <= final_sl:
            exit_px = min(current_candle['open'], final_sl)
            pnl = ((exit_px - strategy_state["entry_price_custom"]) / strategy_state["entry_price_custom"]) * 100.0
            log_info(f"{AnsiColors.RED if pnl < 0 else AnsiColors.BLUE}{AnsiColors.BOLD}EXIT @ {exit_px:.5f} by {exit_reason}. PnL: {pnl:.2f}%{AnsiColors.ENDC}", pair_name=pair_name)
            play_notification_sound()
            # ... (Termux & Email notifikasi untuk EXIT) ...
            strategy_state.update({ # Reset state
                "position_size": 0, "entry_price_custom": None, "entry_timestamp_custom": None,
                "highest_price_for_trailing": None, "trailing_tp_active_custom": False,
                "current_trailing_stop_level": None, "emergency_sl_level_custom": None,
                "last_signal_type": 0
            })
    return strategy_state

# --- FUNGSI UTAMA TRADING LOOP ---
def start_trading(global_settings_dict):
    clear_screen_animated()
    api_settings = global_settings_dict.get("api_settings", {})
    api_key_manager = APIKeyManager(api_settings.get("primary_key"), api_settings.get("recovery_keys", []), api_settings)
    if not api_key_manager.has_valid_keys():
        log_error(f"{AnsiColors.RED}Tidak ada API key valid. Tidak dapat memulai.{AnsiColors.ENDC}"); input(); return

    is_flask_running = any(t.name == "FlaskWebAppThread" for t in threading.enumerate())
    if not is_flask_running:
        flask_thread = threading.Thread(target=run_flask_app, name="FlaskWebAppThread", daemon=True)
        flask_thread.start()
    else: log_info("Flask web server thread sudah berjalan.", "WEBSERVER")

    all_crypto_configs = [cfg for cfg in global_settings_dict.get("cryptos", []) if cfg.get("enabled", True)]
    if not all_crypto_configs:
        log_warning(f"{AnsiColors.ORANGE}Tidak ada konfigurasi crypto aktif.{AnsiColors.ENDC}"); input(); return

    animated_text_display("================ MULTI-CRYPTO STRATEGY START ================", color=AnsiColors.HEADER, delay=0.005)
    # ... (log_info untuk API key) ...

    crypto_data_manager = {}
    for config in all_crypto_configs:
        pair_id = f"{config['symbol']}-{config['currency']}_{config['timeframe']}"
        config['pair_name'] = f"{config['symbol']}-{config['currency']}"
        animated_text_display(f"\nMenginisialisasi {config['pair_name']}...", color=AnsiColors.MAGENTA)
        crypto_data_manager[pair_id] = {"config": config, "all_candles_list": [], "strategy_state": get_initial_strategy_state(), "big_data_collection_phase_active": True, "big_data_email_sent": False, "last_candle_fetch_time": datetime.min, "data_fetch_failed_consecutively": 0}
        
        # Initial Big Data Fetch
        initial_candles = []
        # ... (Logika retry fetch_candles seperti di jawaban sebelumnya) ...
        retries = 0
        max_retries = api_key_manager.total_keys() or 1
        initial_fetch_ok = False
        while retries < max_retries and not initial_fetch_ok:
            current_key = api_key_manager.get_current_key()
            if not current_key: break
            try:
                initial_candles = fetch_candles(config['symbol'], config['currency'], TARGET_BIG_DATA_CANDLES, config['exchange'], current_key, config['timeframe'], config['pair_name'])
                initial_fetch_ok = True
            except APIKeyError:
                log_warning(f"Initial fetch: API Key failed for {config['pair_name']}", pair_name=config['pair_name'])
                if not api_key_manager.switch_to_next_key(): break
                retries += 1
            except Exception as e: log_error(f"Initial fetch error for {config['pair_name']}: {e}", pair_name=config['pair_name']); break
        
        if not initial_candles:
            log_error(f"{AnsiColors.RED}Gagal mengambil data awal untuk {config['pair_name']}. Pair ini mungkin tidak diproses.{AnsiColors.ENDC}", pair_name=config['pair_name'])
            continue

        crypto_data_manager[pair_id]["all_candles_list"] = initial_candles
        # ... (Warm-up strategy state seperti di jawaban sebelumnya) ...
        if len(initial_candles) >= (config['left_strength'] + config['right_strength'] + 1):
            for i in range(config['left_strength'] + config['right_strength'], len(initial_candles) -1):
                # ... (panggil run_strategy_logic dengan slice dan reset posisi)
                temp_state = crypto_data_manager[pair_id]["strategy_state"].copy(); temp_state["position_size"]=0
                crypto_data_manager[pair_id]["strategy_state"] = run_strategy_logic(initial_candles[:i+1], config, temp_state, global_settings_dict)
                if crypto_data_manager[pair_id]["strategy_state"]["position_size"] > 0: # Pastikan reset
                    crypto_data_manager[pair_id]["strategy_state"].update({"position_size":0, "entry_price_custom":None, "entry_timestamp_custom":None})


        if len(initial_candles) >= TARGET_BIG_DATA_CANDLES:
            crypto_data_manager[pair_id]["big_data_collection_phase_active"] = False
            # ... (log & email notifikasi big data selesai) ...
        
        update_live_chart_data(config['pair_name'], initial_candles, crypto_data_manager[pair_id]["strategy_state"], config) # Initial chart update
    
    animated_text_display(f"{AnsiColors.HEADER}-----------------------------------------------{AnsiColors.ENDC}", delay=0.005)

    # Main Trading Loop
    try:
        while True:
            active_in_big_data = 0
            min_next_refresh_s = float('inf')
            data_fetched_this_cycle = False

            for pair_id, data in crypto_data_manager.items():
                config = data["config"]; pair_name = config['pair_name']
                # ... (Cooldown logic jika semua key gagal sebelumnya) ...
                
                now = datetime.now()
                time_since_last_fetch = (now - data["last_candle_fetch_time"]).total_seconds()
                
                interval_s = 0
                if data["big_data_collection_phase_active"]:
                    active_in_big_data +=1
                    # ... (interval lebih cepat untuk big data collection) ...
                    interval_s = 60 if config['timeframe'] == 'minute' else 3600 # Contoh
                else: interval_s = config['refresh_interval_seconds']

                if time_since_last_fetch < interval_s:
                    min_next_refresh_s = min(min_next_refresh_s, interval_s - time_since_last_fetch)
                    continue

                data["last_candle_fetch_time"] = now
                # ... (Fetch update data untuk pair ini, mirip initial fetch dengan retry API key) ...
                new_candles = []
                update_fetch_ok = False; retries_update = 0
                while retries_update < (api_key_manager.total_keys() or 1) and not update_fetch_ok:
                    current_key = api_key_manager.get_current_key()
                    if not current_key: break
                    try:
                        limit = TARGET_BIG_DATA_CANDLES - len(data["all_candles_list"]) if data["big_data_collection_phase_active"] else 5 # Ambil beberapa candle terakhir untuk update
                        if limit <=0 and data["big_data_collection_phase_active"]: data["big_data_collection_phase_active"] = False; break # Selesai big data
                        if limit <=0: limit = 5 # Default
                        
                        new_candles = fetch_candles(config['symbol'], config['currency'], limit , config['exchange'], current_key, config['timeframe'], pair_name)
                        update_fetch_ok = True; data_fetched_this_cycle = True
                        data["data_fetch_failed_consecutively"] = 0
                    except APIKeyError:
                        data["data_fetch_failed_consecutively"] +=1
                        if not api_key_manager.switch_to_next_key(): break
                        retries_update +=1
                    except Exception: break # Gagal fetch update

                if not new_candles and update_fetch_ok: log_info(f"Tidak ada candle baru untuk {pair_name}", pair_name=pair_name)
                
                if new_candles:
                    # ... (Merge new_candles ke data["all_candles_list"]) ...
                    # Contoh merge sederhana:
                    merged_dict = {c['timestamp']: c for c in data["all_candles_list"]}
                    for nc in new_candles: merged_dict[nc['timestamp']] = nc
                    data["all_candles_list"] = sorted(list(merged_dict.values()), key=lambda c: c['timestamp'])
                    if len(data["all_candles_list"]) > TARGET_BIG_DATA_CANDLES: # Jaga ukuran
                        data["all_candles_list"] = data["all_candles_list"][-TARGET_BIG_DATA_CANDLES:]
                    log_info(f"{len(new_candles)} candle baru/diupdate untuk {pair_name}. Total: {len(data['all_candles_list'])}", pair_name=pair_name)


                # Run strategy logic
                if len(data["all_candles_list"]) >= (config['left_strength'] + config['right_strength'] + 1):
                    data["strategy_state"] = run_strategy_logic(data["all_candles_list"], config, data["strategy_state"], global_settings_dict)
                
                update_live_chart_data(pair_name, data["all_candles_list"], data["strategy_state"], config) # Update chart
                min_next_refresh_s = min(min_next_refresh_s, interval_s)
            
            # Sleep duration logic
            sleep_s = 15 # Default
            if not data_fetched_this_cycle and api_key_manager.get_current_key() is None: sleep_s = 3600 # Semua key gagal global
            elif active_in_big_data > 0: sleep_s = min(min_next_refresh_s if min_next_refresh_s != float('inf') else 30, 30)
            else: sleep_s = max(MIN_REFRESH_INTERVAL_AFTER_BIG_DATA, int(min_next_refresh_s) if min_next_refresh_s != float('inf') else 60)
            
            if sleep_s > 0: show_spinner(sleep_s, f"Menunggu {int(sleep_s)}s...")
            else: time.sleep(1)

    except KeyboardInterrupt: animated_text_display(f"\n{AnsiColors.ORANGE}Proses dihentikan.{AnsiColors.ENDC}")
    except Exception as e: log_error(f"{AnsiColors.RED}Error loop utama: {e}{AnsiColors.ENDC}"); log_exception("Trace:", "SYSTEM")
    finally: animated_text_display(f"{AnsiColors.HEADER}================ STRATEGY STOP ================{AnsiColors.ENDC}"); input()

# --- MENU UTAMA ---
def main_menu():
    settings = load_settings()
    while True:
        clear_screen_animated()
        animated_text_display("========= Crypto Strategy Runner (Chart v2500) =========", color=AnsiColors.HEADER)
        # ... (Tampilkan info API, crypto, URL chart seperti di jawaban sebelumnya) ...
        pick_title_main = f"--- Crypto Aktif ({len([c for c in settings.get('cryptos',[]) if c.get('enabled')])}) ---\n"
        # ... (lanjutkan build pick_title_main) ...
        pick_title_main += f"Live Chart: http://localhost:{FLASK_PORT} (atau IP perangkat Anda)\n"
        pick_title_main += f"Chart menampilkan {CHART_WINDOW_SIZE} candle.\n"

        options = ["Mulai Analisa Realtime", "Pengaturan", "Keluar"]
        try: _, index = pick(options, pick_title_main, indicator='=>')
        except: # Fallback
            for i, opt in enumerate(options): print(f"{i+1}. {opt}")
            try: index = int(input("Pilih: ")) -1
            except: continue
        
        if index == 0: start_trading(settings)
        elif index == 1: settings = settings_menu(settings)
        elif index == 2: log_info("Aplikasi ditutup."); break
if __name__ == "__main__":
    try: main_menu()
    except KeyboardInterrupt: clear_screen_animated(); animated_text_display(f"{AnsiColors.ORANGE}Aplikasi dihentikan.{AnsiColors.ENDC}")
    except Exception as e:
        clear_screen_animated(); print(f"{AnsiColors.RED}Error utama: {e}{AnsiColors.ENDC}")
        log_exception("MAIN LEVEL ERROR:", "SYSTEM_CRITICAL"); input()
    finally: log_info("Skrip utama selesai.", "SYSTEM")
