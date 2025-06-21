import time
from flask import Flask, jsonify, render_template_string, Response
import cloudscraper
from bs4 import BeautifulSoup
from threading import Lock, Thread
import feedparser
import collections
import json
import random

# =================================================================
# BAGIAN 1: KONFIGURASI
# =================================================================
app = Flask(__name__)
scraper = cloudscraper.create_scraper()

# --- DAFTAR NEGARA ---
FULL_COUNTRY_MAP = {
    # Amerika
    'Argentina': 'AR', 'Bolivia': 'BO', 'Brazil': 'BR', 'Canada': 'CA', 'Chile': 'CL', 'Colombia': 'CO',
    'Cuba': 'CU', 'Ecuador': 'EC', 'Mexico': 'MX', 'Panama': 'PA', 'Paraguay': 'PY', 'Peru': 'PE',
    'USA': 'US', 'United States': 'US', 'Uruguay': 'UY', 'Venezuela': 'VE',
    # Eropa
    'Albania': 'AL', 'Austria': 'AT', 'Belarus': 'BY', 'Belgium': 'BE', 'Bosnia and Herzegovina': 'BA', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR',
    'Germany': 'DE', 'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT', 'Moldova': 'MD', 'Netherlands': 'NL',
    'North Macedonia': 'MK', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
    'Russia': 'RU', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE',
    'Switzerland': 'CH', 'Ukraine': 'UA', 'United Kingdom': 'GB',
    # Asia & Oseania
    'Afghanistan': 'AF', 'Australia': 'AU', 'Bangladesh': 'BD', 'Cambodia': 'KH', 'China': 'CN', 'Hong Kong': 'HK',
    'India': 'IN', 'Indonesia': 'ID', 'Japan': 'JP', 'Kazakhstan': 'KZ', 'Kyrgyzstan': 'KG', 'Malaysia': 'MY',
    'Mongolia': 'MN', 'Myanmar': 'MM', 'Nepal': 'NP', 'New Zealand': 'NZ', 'North Korea': 'KP', 'Pakistan': 'PK',
    'Philippines': 'PH', 'Singapore': 'SG', 'South Korea': 'KR', 'Sri Lanka': 'LK', 'Taiwan': 'TW',
    'Tajikistan': 'TJ', 'Thailand': 'TH', 'Turkmenistan': 'TM', 'Uzbekistan': 'UZ', 'Vietnam': 'VN',
    # Timur Tengah
    'Armenia': 'AM', 'Azerbaijan': 'AZ', 'Bahrain': 'BH', 'Egypt': 'EG', 'Georgia': 'GE', 'Iran': 'IR',
    'Iraq': 'IQ', 'Israel': 'IL', 'Jordan': 'JO', 'Kuwait': 'KW', 'Lebanon': 'LB', 'Oman': 'OM', 'Qatar': 'QA',
    'Saudi Arabia': 'SA', 'Syria': 'SY', 'Türkiye': 'TR', 'Turkey': 'TR', 'UAE': 'AE', 'Yemen': 'YE',
    # Afrika
    'Algeria': 'DZ', 'Angola': 'AO', 'Botswana': 'BW', 'Burkina Faso': 'BF', 'Cameroon': 'CM', 'DR Congo': 'CD', 'Ethiopia': 'ET',
    'Ghana': 'GH', 'Ivory Coast': 'CI', "Côte d'Ivoire": 'CI', 'Kenya': 'KE', 'Libya': 'LY', 'Madagascar': 'MG',
    'Mali': 'ML', 'Mauritius': 'MU', 'Morocco': 'MA', 'Mozambique': 'MZ', 'Namibia': 'NA', 'Niger': 'NE',
    'Nigeria': 'NG', 'Rwanda': 'RW', 'Senegal': 'SN', 'Somalia': 'SO', 'South Africa': 'ZA', 'Sudan': 'SD',
    'Tanzania': 'TZ', 'Tunisia': 'TN', 'Uganda': 'UG', 'Zambia': 'ZM', 'Zimbabwe': 'ZW'
}

REVERSE_COUNTRY_MAP = {v: k for k, v in FULL_COUNTRY_MAP.items()}
COUNTRY_MAP = FULL_COUNTRY_MAP
ACTIVE_CONFLICTS = [ ('RU', 'UA'), ('IL', 'IR'), ('IL', 'SY'), ('YE', 'SA'), ('CN', 'TW'), ('IL', 'LB') ]
TENSION_KEYWORDS = { 'war': 15, 'conflict': 10, 'sanction': 8, 'protest': 3, 'crisis': 7, 'attack': 12, 'dispute': 6, 'tension': 8, 'unrest': 4, 'mobilization': 10, 'threat': 9, 'retaliate': 11 }
G20_CODES = ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT', 'JP', 'KR', 'MX', 'RU', 'SA', 'ZA', 'TR', 'GB', 'US']
FED_HIKE_KEYWORDS = ['INFLATION HIGH', 'STRONG ECONOMY', 'ROBUST GROWTH', 'WAGE GROWTH', 'OVERHEATING']
FED_CUT_KEYWORDS = ['RECESSION', 'SLOWING', 'INFLATION EASING', 'WEAKNESS', 'UNEMPLOYMENT', 'DOWNTURN']

# --- Variabel Global & Konfigurasi Real-time ---
data_lock = Lock()
global_data = {
    'rates': [], 'decisions': [], 'meetings': [],
    'all_news': {code: [] for code in REVERSE_COUNTRY_MAP.keys()},
    'geopolitical': {
        "tension_scores": [], "conflicts": ACTIVE_CONFLICTS, "news_links": []
    },
    'last_updated_code': None
}
MONETARY_UPDATE_INTERVAL_SECONDS = 900

# =================================================================
# BAGIAN 2: LOGIKA AI (TIDAK BERUBAH)
# =================================================================
def generate_ai_summary(all_data):
    geo_data = all_data.get('geopolitical', {})
    all_scores = [s['value'] for s in geo_data.get('tension_scores', [])]
    world_tension = int(sum(all_scores) / len(all_scores)) if all_scores else 30
    war_probability = int(world_tension * 0.8 + len(geo_data.get('conflicts', [])) * 4)
    war_probability = min(war_probability, 95)
    hike_score, cut_score = 0, 0
    us_headlines = all_data.get('all_news', {}).get('US', [])
    for headline in us_headlines:
        if any(keyword in headline for keyword in FED_HIKE_KEYWORDS): hike_score += 1
        if any(keyword in headline for keyword in FED_CUT_KEYWORDS): cut_score += 1
    monetary_stance = "NEUTRAL"
    if hike_score > cut_score + 1: monetary_stance = "HAWKISH"
    if cut_score > hike_score + 1: monetary_stance = "DOVISH"
    market_outlook, strategic_posture = "", ""
    if world_tension > 65:
        if monetary_stance == "HAWKISH":
            market_outlook = "SANGAT BEARISH. Kombinasi ketegangan global yang ekstrim dan kebijakan moneter yang ketat menciptakan badai sempurna bagi aset berisiko. Pelarian modal ke aset paling aman sangat mungkin terjadi."
            strategic_posture = "POSTUR SANGAT DEFensif. Pertimbangkan untuk mengurangi eksposur secara signifikan pada saham dan aset spekulatif. Alihkan ke Dolar AS (tunai), obligasi pemerintah jangka pendek, dan Emas."
        else:
            market_outlook = "NETRAL cenderung BEARISH. Kebijakan moneter yang lebih longgar memberikan sedikit bantalan, namun ketegangan geopolitik yang tinggi mendominasi sentimen. Pasar diperkirakan akan sangat volatil dan bergerak sideways."
            strategic_posture = "POSTUR DEFensif. Fokus pada saham-saham defensif (utilitas, kesehatan) dan Emas sebagai lindung nilai utama terhadap risiko geopolitik. Hindari aset berisiko tinggi."
    else:
        if monetary_stance == "HAWKISH":
            market_outlook = "NETRAL cenderung BEARISH. Lingkungan geopolitik yang tenang diredam oleh kebijakan moneter yang ketat. Pertumbuhan pasar saham kemungkinan akan terbatas."
            strategic_posture = "POSTUR HATI-HATI. Fokus pada saham berkualitas dengan fundamental kuat. Hindari perusahaan dengan utang tinggi. Pasar dapat mengalami rotasi dari sektor pertumbuhan ke sektor nilai."
        elif monetary_stance == "DOVISH":
            market_outlook = "SANGAT BULLISH. Skenario 'Goldilocks' di mana stabilitas geopolitik bertemu dengan kebijakan moneter yang akomodatif. Kondisi ideal untuk kenaikan aset berisiko."
            strategic_posture = "POSTUR AGRESIF. Ini adalah waktu untuk meningkatkan eksposur ke aset berisiko seperti saham (terutama sektor teknologi dan pertumbuhan) dan aset kripto. Sentimen risk-on mendominasi."
        else:
            market_outlook = "BULLISH. Dengan tidak adanya kejutan geopolitik dan kebijakan moneter yang stabil, pasar cenderung untuk melanjutkan tren naiknya secara bertahap."
            strategic_posture = "POSTUR AGRESIF TERUKUR. Tetap berinvestasi pada aset berisiko, namun tetap waspada terhadap perubahan data ekonomi yang dapat mengubah sikap bank sentral."
    active_hotspots = [f"{REVERSE_COUNTRY_MAP.get(c1, c1)} vs {REVERSE_COUNTRY_MAP.get(c2, c2)}" for c1, c2 in geo_data.get('conflicts', [])]
    active_conflict_countries = {code for pair in geo_data.get('conflicts', []) for code in pair}
    sorted_scores = sorted(geo_data.get('tension_scores', []), key=lambda x: x['value'], reverse=True)
    potential_hotspots = [f"{REVERSE_COUNTRY_MAP.get(score['id'], score['id'])} (Skor: {score['value']})" for score in sorted_scores if score['id'] not in active_conflict_countries and len(potential_hotspots) < 5]
    return { "world_tension": world_tension, "war_probability": war_probability, "market_outlook": market_outlook, "strategic_posture": strategic_posture, "active_hotspots": active_hotspots, "potential_hotspots": potential_hotspots }

# =================================================================
# BAGIAN 3: FUNGSI-FUNGSI HELPER (TIDAK BERUBAH)
# =================================================================
def parse_world_rates(soup):
    data = []
    table = soup.select_one('table#AutoNumber3')
    if not table: return []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) < 7: continue
        try:
            country_name_raw = cols[4].get_text(strip=True).split('|')[0].strip()
            country_name = country_name_raw.replace('The', '').strip()
            if country_name.lower() == 'türkiye': country_name = 'Türkiye'
            data.append({ "country_name": country_name.upper(), "country_code": COUNTRY_MAP.get(country_name, None), "rate": cols[1].get_text(strip=True), "change": cols[2].get_text(strip=True), "date": cols[5].get_text(strip=True), "rate_name": cols[4].get_text(strip=True).replace(country_name_raw, '').lstrip('| ').strip().upper() })
        except (IndexError, AttributeError): continue
    return data

def parse_decisions_or_meetings(soup, type):
    data = []
    table = soup.select_one('table#AutoNumber3')
    if not table: return []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) < 4: continue
        try:
            date_text = cols[1].get_text(strip=True)
            full_text = cols[2].get_text(strip=True)
            country_name = "UNKNOWN"
            action = "MEETING"
            for name, code in COUNTRY_MAP.items():
                if name.lower() in full_text.lower(): country_name = name; break
            if country_name == "UNKNOWN" and "eurozone" in full_text.lower(): country_name = "Eurozone"
            if type == 'decisions':
                if 'cuts' in full_text.lower() or 'cut' in full_text.lower(): action = "CUT"
                elif 'raises' in full_text.lower() or 'hikes' in full_text.lower(): action = "HIKE"
                elif 'unchanged' in full_text.lower() or 'holds' in full_text.lower(): action = "UNCHANGED"
                else: action = "DECISION"
            data.append({"country_name": country_name.upper(), "country_code": COUNTRY_MAP.get(country_name, None), "date": date_text.upper(), "description": full_text.upper(), "action": action.upper()})
        except (IndexError, AttributeError): continue
    return data

def generate_analysis(country_code, decision_data):
    if not decision_data: return {"title": "ANALYSIS: NO DATA", "summary": "NO RECENT DIRECTIVES FOUND. TERMINATING DATALINK.", "impact": "IMPACT CANNOT BE DETERMINED."}
    latest_decision = decision_data[0]
    action = latest_decision.get("action", "UNKNOWN").lower()
    country_name = latest_decision.get("country_name", "Negara")
    major_economies, emerging_markets = ['US', 'EU', 'JP', 'GB', 'CN'], ['ID', 'BR', 'IN', 'RU', 'TR', 'ZA', 'MX']
    title = f"ANALYSIS :: {action.upper()} EVENT :: {country_name.upper()}"
    summary = f"SYSTEM DETECTED MONETARY POLICY: **{action.upper()}**. RAW DATA: {latest_decision['description']}."
    impact_paragraphs = []
    if action == 'cut': impact_paragraphs.append(">> GENERAL PROTOCOL: RATE CUT AIMS TO STIMULATE ECONOMY. POTENTIAL DOMESTIC CURRENCY WEAKNESS.")
    elif action == 'hike': impact_paragraphs.append(">> GENERAL PROTOCOL: RATE HIKE AIMS TO COMBAT INFLATION. POTENTIAL DOMESTIC CURRENCY STRENGTH.")
    else: impact_paragraphs.append(">> GENERAL PROTOCOL: RATE UNCHANGED INDICATES 'WAIT-AND-SEE' STANCE.")
    crypto_impact_text = ""
    if country_code in major_economies:
        if action == 'cut': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BEARISH]: RATE CUT BY {country_name.upper()} SUGGESTS REDUCED GLOBAL RISK APPETITE."
        elif action == 'hike': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BULLISH]: TRADITIONAL MARKETS STRESSED. INVESTORS MAY SEEK ALTERNATIVE ASSETS/HEDGES."
    elif country_code in emerging_markets:
        if action == 'cut': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BULLISH]: LOWER RATES IN {country_name.upper()} PUSH LOCAL INVESTORS TOWARDS HIGHER-YIELD ASSETS LIKE CRYPTO."
        elif action == 'hike': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BEARISH]: HIGHER LOCAL YIELDS ON SAFE ASSETS BECOME ATTRACTIVE, REDUCING DEMAND FOR CRYPTO."
    if crypto_impact_text: impact_paragraphs.append(crypto_impact_text)
    impact_paragraphs.append("<small><i>DISCLAIMER: PROTOTYPE MODEL. CONDUCT INDEPENDENT RESEARCH. END TRANSMISSION.</i></small>")
    return {"title": title, "summary": summary, "impact": "<br><br>".join(impact_paragraphs)}

def get_news_from_rss(country_code):
    try:
        url = f"https://news.google.com/rss?gl={country_code.upper()}&hl=en-US&ceid={country_code.upper()}:en"
        feed = feedparser.parse(url)
        if feed.bozo: raise Exception(feed.bozo_exception)
        return [entry.title.upper() for entry in feed.entries[:20]]
    except Exception as e:
        print(f"Error fetching RSS for {country_code}: {e}")
        return []

# =================================================================
# BAGIAN 4: PROSES BACKGROUND (TIDAK BERUBAH DARI SEBELUMNYA)
# =================================================================
def background_update_task():
    country_codes = list(REVERSE_COUNTRY_MAP.keys())
    random.shuffle(country_codes)
    country_index = 0
    last_monetary_update_time = 0
    
    while True:
        try:
            current_time = time.time()
            if (current_time - last_monetary_update_time) > MONETARY_UPDATE_INTERVAL_SECONDS:
                print(">> [TIMER] Performing periodic monetary data scan...")
                cbrates_urls = {'rates': 'https://www.cbrates.com/', 'decisions': 'https://www.cbrates.com/decisions.htm', 'meetings': 'https://www.cbrates.com/meetings.htm'}
                with data_lock:
                    for key, url in cbrates_urls.items():
                        try:
                            response = scraper.get(url, timeout=20); response.raise_for_status()
                            soup = BeautifulSoup(response.text, 'html.parser')
                            if key == 'rates': global_data['rates'] = parse_world_rates(soup)
                            elif key == 'decisions': global_data['decisions'] = parse_decisions_or_meetings(soup, 'decisions')
                            elif key == 'meetings': global_data['meetings'] = parse_decisions_or_meetings(soup, 'meetings')
                        except Exception as e: print(f"!! Failed to fetch monetary data for {key}: {e}")
                last_monetary_update_time = current_time
                print(">> [TIMER] Monetary scan complete.")

            country_code = country_codes[country_index]
            country_name = REVERSE_COUNTRY_MAP.get(country_code, country_code)
            print(f">> MAX SPEED SCAN: [{country_index + 1}/{len(country_codes)}] {country_name}")
            latest_news = get_news_from_rss(country_code)

            with data_lock:
                if latest_news and (country_code not in global_data['all_news'] or global_data['all_news'][country_code] != latest_news):
                    global_data['all_news'][country_code] = latest_news
                    global_data['last_updated_code'] = country_code
                
                tension_scores = collections.defaultdict(int)
                news_links_set = set()
                for c1, c2 in ACTIVE_CONFLICTS:
                    tension_scores[c1] += 50; tension_scores[c2] += 50
                
                for source_code, headlines in global_data['all_news'].items():
                    for headline in headlines:
                        headline_tension_weight = sum(weight for keyword, weight in TENSION_KEYWORDS.items() if keyword.upper() in headline)
                        if headline_tension_weight > 0:
                            for target_code, target_name in REVERSE_COUNTRY_MAP.items():
                                if source_code != target_code and target_name.upper() in headline:
                                    tension_scores[source_code] += headline_tension_weight
                                    tension_scores[target_code] += headline_tension_weight
                                    news_links_set.add(tuple(sorted((source_code, target_code))))
                
                global_data['geopolitical']['tension_scores'] = [{"id": code, "value": min(score, 100)} for code, score in tension_scores.items()]
                global_data['geopolitical']['news_links'] = [list(link) for link in news_links_set]

            country_index = (country_index + 1) % len(country_codes)
        except Exception as e:
            print(f"!! ERROR in background task: {e}")
            time.sleep(10)

# =================================================================
# BAGIAN 5: API ENDPOINTS (TIDAK BERUBAH DARI SEBELUMNYA)
# =================================================================
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/stream-updates')
def stream_updates():
    def event_stream():
        while True:
            time.sleep(0.5)
            with data_lock:
                data_to_send = {
                    'geopolitical': global_data['geopolitical'],
                    'updated_country_code': global_data.get('last_updated_code')
                }
                yield f"data: {json.dumps(data_to_send)}\n\n"
                if global_data['last_updated_code']:
                    global_data['last_updated_code'] = None
                    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/ai-summary')
def get_ai_summary():
    with data_lock: summary = generate_ai_summary(global_data)
    return jsonify(summary)

@app.route('/api/country-data/<country_code>')
def get_country_data(country_code):
    with data_lock: all_data = dict(global_data)
    country_code_upper = country_code.upper()
    is_eurozone = country_code_upper == 'EU'
    def filter_by_code(data_list):
        if is_eurozone: return [d for d in data_list if d.get('country_name') == 'EUROZONE']
        return [d for d in data_list if d.get('country_code') == country_code_upper]
    decisions_data = filter_by_code(all_data.get('decisions', []))
    return jsonify({
        "rates": filter_by_code(all_data.get('rates', [])), "decisions": decisions_data,
        "meetings": filter_by_code(all_data.get('meetings', [])),
        "news": all_data.get('all_news', {}).get(country_code_upper, []),
        "analysis": generate_analysis(country_code_upper, decisions_data)
    })

# =================================================================
# BAGIAN 6: FRONTEND HTML DENGAN FITUR NOTIFIKASI
# =================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G.I.M.P.S - Global Intelligence Monetary Policy System</title>
    <style>
        :root {
            --main-bg: #000; --panel-bg: rgba(10, 10, 10, 0.9); --border-color: rgba(255, 255, 255, 0.2);
            --accent-color: #FFF; --text-color: #dcdcdc; --red-alert: #ff4d4d; --blue-link: #64b5f6;
            --flash-color: #FFFF00;
        }
        @keyframes scanlines { 0% { background-position: 0 0; } 100% { background-position: 0 50px; } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes typing { from { width: 0; } to { width: 100%; } }
        @keyframes blink-caret { from, to { border-color: transparent; } 50% { border-color: var(--accent-color); } }
        @keyframes glitch-text { 2%,64%{ transform: translate(2px,0) skew(0deg); } 4%,60%{ transform: translate(-2px,0) skew(0deg); } 62%{ transform: translate(0,0) skew(5deg); } }
        @keyframes glitch-appear { 0% { clip-path: inset(20% 0 70% 0); } 20% { clip-path: inset(80% 0 10% 0); } 40% { clip-path: inset(40% 0 40% 0); } 60% { clip-path: inset(90% 0 5% 0); } 80% { clip-path: inset(10% 0 85% 0); } 100% { clip-path: inset(0 0 0 0); } }

        body, html { 
            font-family: 'Courier New', Courier, monospace; 
            margin: 0; padding: 0; height: 100%; 
            overflow: hidden; 
            background-color: var(--main-bg); 
            color: var(--text-color); 
        }
        
        body::before {
            content: ''; position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            box-shadow: inset 0 0 150px rgba(0,0,0,0.9);
            pointer-events: none;
            z-index: 9999;
        }

        #chartdiv { 
            width: 100%; height: 100vh; 
            background-image: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), linear-gradient(0deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px); 
            background-size: 100%, 3px 3px; 
            position: relative; 
            transform: scale(1.05);
            border-radius: 40px; 
        }

        #chartdiv::after { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(0deg, rgba(0, 0, 0, 0) 50%, rgba(255, 255, 255, 0.05) 50%); background-size: 100% 4px; animation: scanlines 0.2s linear infinite; pointer-events: none; }
        #infopanel { position: fixed; bottom: 0; left: 0; width: 100%; height: 50vh; background-color: var(--panel-bg); border-top: 1px solid var(--border-color); box-shadow: 0 -2px 10px rgba(0,0,0,0.5); visibility: hidden; opacity: 0; z-index: 1000; display: flex; flex-direction: column; backdrop-filter: blur(5px); }
        #infopanel.visible { visibility: visible; opacity: 1; animation: glitch-appear 0.2s steps(8, end) forwards; }
        #panel-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background-color: rgba(0,0,0,0.3); border-bottom: 1px solid var(--border-color); }
        .glitch { position: relative; animation: glitch-text 2s infinite; }
        #close-panel { background: none; border: none; color: var(--accent-color); font-size: 28px; cursor: pointer; transition: transform 0.2s; }
        #close-panel:hover { transform: scale(1.5); }
        #info-content { padding: 20px; overflow-y: auto; flex-grow: 1; animation: fadeIn 1s; }
        .info-container { display: flex; gap: 20px; }
        .data-column, .analysis-column, .news-column { flex: 1; }
        .analysis-column, .news-column { border-left: 1px solid var(--border-color); padding-left: 20px; }
        h2, h3 { color: var(--accent-color); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-top: 0; }
        h3 { font-size: 1.1em; border: none; padding-bottom: 5px; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 0.9em; }
        th, td { border: 1px solid var(--border-color); padding: 8px; text-align: left; }
        th { background-color: rgba(255, 255, 255, 0.05); }
        ul#news-list, ul.hotspot-list { list-style-type: '>> '; padding-left: 20px; margin: 0; font-size: 0.85em; max-height: 40vh; overflow-y: auto; }
        ul#news-list li, ul.hotspot-list li { margin-bottom: 8px; }
        .placeholder { color: var(--accent-color); text-align: center; padding-top: 40px; font-size: 1.2em; overflow: hidden; white-space: nowrap; margin: 0 auto; letter-spacing: .15em; animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite; }
        ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #555; }
    </style>
</head>
<body>
<div id="chartdiv"></div>
<div id="infopanel">
    <div id="panel-header">
        <h2 id="panel-title" style="border:none; margin:0; padding:0;">// SYSTEM STANDBY //</h2>
        <button id="close-panel" title="TERMINATE CONNECTION">×</button>
    </div>
    <div id="info-content"><p class="placeholder">// AWAITING COMMAND //</p></div>
</div>

<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/map.js"></script>
<script src="https://cdn.amcharts.com/lib/5/geodata/worldLow.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

<script>
    const infoPanel = document.getElementById('infopanel'), infoContent = document.getElementById('info-content'), panelTitle = document.getElementById('panel-title'), closeButton = document.getElementById('close-panel');

    // << FUNGSI BARU UNTUK MENAMPILKAN NOTIFIKASI BROWSER
    function showNotification(title, body) {
        // Cek dulu apakah izin sudah diberikan
        if (Notification.permission === 'granted') {
            new Notification(title, { body: body, icon: '/favicon.ico' }); // Ganti icon jika punya
        }
    }

    am5.ready(function() {
        // << PENAMBAHAN: MEMINTA IZIN NOTIFIKASI SAAT HALAMAN DIBUKA
        if ('Notification' in window) {
            if (Notification.permission !== 'granted' && Notification.permission !== 'denied') {
                Notification.requestPermission().then(function(permission) {
                    if (permission === 'granted') {
                        console.log('Notification permission granted.');
                        showNotification('G.I.M.P.S.', 'System notifications are now active.');
                    }
                });
            }
        }

        var root = am5.Root.new("chartdiv");
        root.setThemes([am5themes_Animated.new(root)]);
        var chart = root.container.children.push(am5map.MapChart.new(root, { panX: "pan", panY: "pan", projection: am5map.geoEquirectangular(), wheelY: "zoom" }));
        
        chart.chartContainer.set("background", am5.Rectangle.new(root, { fill: am5.color(0x000000), fillOpacity: 1 }));

        var polygonSeries = chart.series.push(am5map.MapPolygonSeries.new(root, { geoJSON: am5geodata_worldLow, exclude: ["AQ"], valueField: "value", calculateAggregates: true }));
        
        polygonSeries.mapPolygons.template.setAll({ interactive: true, fill: am5.color(0x550000), stroke: am5.color(0x444444), strokeWidth: 0.5, transitionDuration: 500 });
        polygonSeries.mapPolygons.template.adapters.add("tooltipText", function(text, target) {
          if (target.dataItem.get("value") == null || target.dataItem.get("value") === 0) { return "{name}: Tension Unknown / Stable"; }
          return "{name}: TENSION SCORE {value}";
        });
        polygonSeries.mapPolygons.template.states.create("hover", { fill: am5.color(0x64b5f6) });
        polygonSeries.set("heatRules", [{ target: polygonSeries.mapPolygons.template, dataField: "value", min: am5.color(0x000000), max: am5.color(0xff0000), key: "fill", logarithmic: true }]);

        var conflictLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        conflictLineSeries.mapLines.template.setAll({ stroke: am5.color(0xff0000), strokeOpacity: 0.6, strokeWidth: 2, strokeDasharray: [4,2] });
        var newsLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        newsLineSeries.mapLines.template.setAll({ stroke: am5.color(0x64b5f6), strokeOpacity: 0.6, strokeWidth: 1, arc: -0.2 });
        
        function updateMap(geoData) {
            if (!geoData || !polygonSeries.data.length) return;
            (geoData.tension_scores || []).forEach(score => {
                const dataItem = polygonSeries.getDataItemById(score.id);
                if (dataItem) { dataItem.set("value", score.value); }
            });
            conflictLineSeries.data.clear(); newsLineSeries.data.clear();
            (geoData.conflicts || []).forEach(conflict => renderLine(conflict[0], conflict[1], conflictLineSeries, true));
            (geoData.news_links || []).forEach(link => renderLine(link[0], link[1], newsLineSeries, false));
        }

        // << MODIFIKASI: DURASI FLASH SESUAI PERMINTAAN
        function flashCountry(countryCode) {
            let polygon = polygonSeries.getPolygonById(countryCode);
            if (polygon) {
                const originalColor = polygon.get("fill");
                const flashColor = am5.color(0xFFFF00);
                let animation = polygon.animate({
                    key: "fill",
                    to: flashColor,
                    duration: 5400
                });
                if (animation) {
                    animation.events.on("stopped", function() {
                        polygon.animate({
                            key: "fill",
                            to: originalColor,
                            duration: 1800
                        });
                    });
                }
            }
        }
        
        function renderLine(code1, code2, series, isConflict) {
            let p1 = polygonSeries.getPolygonById(code1); let p2 = polygonSeries.getPolygonById(code2);
            if(p1 && p2) {
                let lineDataItem = series.pushDataItem({ geometry: { type: "LineString", coordinates: [[p1.visualCentroid.longitude, p1.visualCentroid.latitude], [p2.visualCentroid.longitude, p2.visualCentroid.latitude]] } });
                if (isConflict) {
                     let bullet = am5.Bullet.new(root, { sprite: am5.Circle.new(root, { radius: 3, fill: am5.color(0xff0000) }) });
                     bullet.animate({ key: "location", from: 0, to: 1, duration: 2000, loops: Infinity });
                     lineDataItem.bullets.push(bullet);
                }
            }
        }
        
        const eventSource = new EventSource("/api/stream-updates");
        eventSource.onmessage = function(event) { 
            const receivedData = JSON.parse(event.data);
            updateMap(receivedData.geopolitical); 
            
            // << MODIFIKASI: PANGGIL FUNGSI FLASH DAN NOTIFIKASI
            if(receivedData.updated_country_code) {
                const countryCode = receivedData.updated_country_code;
                const polygon = polygonSeries.getPolygonById(countryCode);
                if (polygon) {
                    const countryName = polygon.dataItem.dataContext.name;
                    flashCountry(countryCode);
                    showNotification('G.I.M.P.S. Intel Update', `New headlines detected for ${countryName}.`);
                }
            }
        };
        eventSource.onerror = function(err) { console.error("EventSource failed:", err); };

        polygonSeries.mapPolygons.template.events.on("click", function(ev) {
            if (ev.target.dataItem.get("value") != null) { fetchCountryData(ev.target.dataItem.dataContext.id, ev.target.dataItem.dataContext.name); }
        });
        chart.chartContainer.get("background").events.on("click", () => closeInfoPanel());
    });
    
    closeButton.addEventListener('click', closeInfoPanel);
    
    function showInfoPanel() { infopanel.classList.add('visible'); }
    function closeInfoPanel() { infoPanel.classList.remove('visible'); }
    
    async function fetchCountryData(countryCode, countryName) {
        infoContent.innerHTML = `<p class="placeholder">ESTABLISHING DATALINK: ${countryName.toUpperCase()}...</p>`;
        panelTitle.innerText = countryName.toUpperCase();
        showInfoPanel();
        try {
            const response = await fetch(`/api/country-data/${countryCode}`);
            if (!response.ok) throw new Error('CONNECTION FAILED.');
            const data = await response.json();
            displayCountryData(data, countryName);
        } catch (error) { infoContent.innerHTML = `<p class="placeholder" style="color: var(--red-alert);">${error.message}</p>`; }
    }

    function displayCountryData(data, countryName) {
        const ratesHtml = data.rates && data.rates.length > 0 ? createTable(data.rates, ['Rate', 'Change', 'Date', 'Rate Name']) : '<p>// NO MONETARY RATE DATA //</p>';
        const decisionsHtml = data.decisions && data.decisions.length > 0 ? createTable(data.decisions, ['Date', 'Description', 'Action']) : '<p>// NO RECENT DIRECTIVE DATA //</p>';
        const meetingsHtml = data.meetings && data.meetings.length > 0 ? createTable(data.meetings, ['Date', 'Description']) : '<p>// NO UPCOMING TRANSMISSIONS DATA //</p>';
        let newsHtml = '<p>// NO INTEL FEED //</p>';
        if(data.news && data.news.length > 0) {
            newsHtml = '<ul id="news-list">'; data.news.slice(0, 15).forEach(headline => newsHtml += `<li>${headline}</li>`); newsHtml += '</ul>';
        }
        panelTitle.innerText = countryName.toUpperCase();
        panelTitle.classList.add('glitch');
        setTimeout(() => { panelTitle.classList.remove('glitch'); }, 2000);
        infoContent.innerHTML = `
            <div class="info-container">
                <div class="data-column"><h3>// MONETARY DATA</h3>${ratesHtml}<h3>// RECENT DIRECTIVES</h3>${decisionsHtml}<h3>// UPCOMING TRANSMISSIONS</h3>${meetingsHtml}</div>
                <div class="news-column"><h3>// RECENT INTEL</h3>${newsHtml}</div>
                <div class="analysis-column"><h3>${data.analysis.title}</h3><p><b>SUMMARY:</b> ${data.analysis.summary.replace(/\\*\\*(.*?)\\*\\*/g, '<span>$1</span>')}</p><br><p><b>IMPACT ANALYSIS:</b><br>${data.analysis.impact.replace(/\\*\\*(.*?)\\*\\*/g, '<span>$1</span>')}</p></div>
            </div>`;
    }

    function createTable(data, headers) {
        let table = '<table><thead><tr>'; headers.forEach(h => table += `<th>${h.toUpperCase()}</th>`); table += '</tr></thead><tbody>';
        data.forEach(row => {
            table += '<tr>';
            headers.forEach(h => { table += `<td>${row[h.toLowerCase().replace(/ /g, '_')] || 'N/A'}</td>`; });
            table += '</tr>';
        });
        return table + '</tbody></table>';
    }
</script>
</body>
</html>
"""

# ===============================================
# BAGIAN 7: MENJALANKAN APLIKASI
# ===============================================
if __name__ == '__main__':
    print("===============================================================")
    print(">> G.I.M.P.S (vStrategic AI) :: BOOTING...")
    print(">> FITUR BARU: Notifikasi browser diaktifkan.")
    print(">> WARNING: Scan delay is set to 0. This will perform requests at maximum speed.")
    print(">> High frequency requests may lead to temporary IP blocking from data sources.")
    print("===============================================================")
    update_thread = Thread(target=background_update_task, daemon=True)
    update_thread.start()
    print(">> BACKGROUND SCANNER INITIALIZED. SYSTEM IS NOW LIVE.")
    print(">> Open your browser and access the following address:")
    print(">> http://127.0.0.1:5001")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
