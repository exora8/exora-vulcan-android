import time
from flask import Flask, jsonify, render_template_string
import cloudscraper
from bs4 import BeautifulSoup
from threading import Lock
import feedparser
import collections

# =================================================================
# BAGIAN 1: KONFIGURASI DENGAN DAFTAR NEGARA MAKSIMAL
# =================================================================
app = Flask(__name__)
scraper = cloudscraper.create_scraper()
app_cache = {}
CACHE_LIFETIME_SECONDS = 1800
data_lock = Lock()

# --- DAFTAR NEGARA YANG DIPERLUAS SECARA MAKSIMAL ---
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
# SKOR TINGGI UNTUK KONFLIK AKTIF
ACTIVE_CONFLICTS = [ ('RU', 'UA'), ('IL', 'IR'), ('IL', 'SY'), ('YE', 'SA'), ('CN', 'TW'), ('IL', 'LB') ]
TENSION_KEYWORDS = { 'war': 15, 'conflict': 10, 'sanction': 8, 'protest': 3, 'crisis': 7, 'attack': 12, 'dispute': 6, 'tension': 8, 'unrest': 4, 'mobilization': 10, 'threat': 9, 'retaliate': 11 }
G20_CODES = ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT', 'JP', 'KR', 'MX', 'RU', 'SA', 'ZA', 'TR', 'GB', 'US']
FED_HIKE_KEYWORDS = ['INFLATION HIGH', 'STRONG ECONOMY', 'ROBUST GROWTH', 'WAGE GROWTH', 'OVERHEATING']
FED_CUT_KEYWORDS = ['RECESSION', 'SLOWING', 'INFLATION EASING', 'WEAKNESS', 'UNEMPLOYMENT', 'DOWNTURN']


# =================================================================
# BAGIAN 2: LOGIKA BACKEND (Sedikit Perubahan)
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
                if name.lower() in full_text.lower():
                    country_name = name; break
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
        return [entry.title.upper() for entry in feed.entries]
    except Exception as e:
        print(f"Error fetching RSS for {country_code}: {e}")
        return [f"ERROR FETCHING RSS FEED FOR {country_code.upper()}"]

def generate_ai_summary(all_data):
    geo_data = all_data.get('geopolitical', {})
    g20_scores = [s['value'] for s in geo_data.get('tension_scores', []) if s['id'] in G20_CODES]
    world_tension = int(sum(g20_scores) / len(g20_scores)) if g20_scores else 0
    hike_score, cut_score = 0, 0
    # Menggunakan all_news karena lebih andal daripada news per negara
    us_headlines = all_data.get('all_news', {}).get('US', [])
    for headline in us_headlines:
        if any(keyword in headline for keyword in FED_HIKE_KEYWORDS): hike_score += 1
        if any(keyword in headline for keyword in FED_CUT_KEYWORDS): cut_score += 1
    total_score = hike_score + cut_score
    if total_score == 0:
        hike_prob, cut_prob, hold_prob = 10, 10, 80
    else:
        hike_prob = int((hike_score / total_score) * 80)
        cut_prob = int((cut_score / total_score) * 80)
        hold_prob = 100 - hike_prob - cut_prob
    market_impact = "MARKET IN WAIT-AND-SEE MODE. VOLATILITY EXPECTED AROUND DATA RELEASES."
    if hike_prob > hold_prob: market_impact = "BEARISH OUTLOOK. HIGHER RATES TIGHTEN LIQUIDITY, HURTING RISK ASSETS LIKE STOCKS AND CRYPTO."
    if cut_prob > hold_prob: market_impact = "BULLISH OUTLOOK. EASIER MONETARY POLICY BOOSTS LIQUIDITY, FAVORING RISK ASSETS."
    avg_global_tension = sum(s['value'] for s in geo_data.get('tension_scores', [])) / len(geo_data.get('tension_scores', [])) if geo_data.get('tension_scores') else 0
    war_prob = int(avg_global_tension * 0.6 + len(geo_data.get('conflicts', [])) * 5 + len(geo_data.get('news_links',[])) * 0.1)
    war_prob = min(war_prob, 95)
    return { "world_tension": world_tension, "fed_hike_prob": hike_prob, "fed_hold_prob": hold_prob, "fed_cut_prob": cut_prob, "market_impact": market_impact, "war_probability": war_prob }

# =================================================================
# BAGIAN INI DIROMBAK TOTAL UNTUK LOGIKA SKOR YANG LEBIH BAIK
# =================================================================
def get_all_data():
    with data_lock:
        # Caching tetap sama
        if app_cache and (time.time() - app_cache.get('timestamp', 0)) < CACHE_LIFETIME_SECONDS:
            return app_cache.get('data')

        print(">> INITIALIZING GLOBAL SCAN... THIS WILL TAKE SEVERAL MINUTES.")
        all_data = {'all_news': {}} # Tambahkan all_news untuk digunakan AI summary

        # 1. Ambil data moneter (tidak berubah)
        cbrates_urls = {'rates': 'https://www.cbrates.com/', 'decisions': 'https://www.cbrates.com/decisions.htm', 'meetings': 'https://www.cbrates.com/meetings.htm'}
        for key, url in cbrates_urls.items():
            try:
                response = scraper.get(url, timeout=20); response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                if key == 'rates': all_data['rates'] = parse_world_rates(soup)
                elif key == 'decisions': all_data['decisions'] = parse_decisions_or_meetings(soup, 'decisions')
                elif key == 'meetings': all_data['meetings'] = parse_decisions_or_meetings(soup, 'meetings')
            except Exception as e: print(f"Failed to fetch monetary data for {key}: {e}"); all_data[key] = []

        # 2. Ambil SEMUA berita terlebih dahulu
        country_codes_to_scan = list(REVERSE_COUNTRY_MAP.keys())
        all_headlines_by_country = {}
        for i, code in enumerate(country_codes_to_scan):
            print(f">> GATHERING INTEL: {REVERSE_COUNTRY_MAP.get(code, code)} ({i+1}/{len(country_codes_to_scan)})")
            all_headlines_by_country[code] = get_news_from_rss(code)
            time.sleep(0.1) # Tetap ada jeda untuk menghindari blokir
        
        all_data['all_news'] = all_headlines_by_country # Simpan berita untuk API

        # 3. Hitung skor ketegangan berdasarkan interaksi
        print(">> ANALYZING INTER-STATE TENSIONS...")
        tension_scores = collections.defaultdict(int)
        news_links_set = set()

        # Tambahkan skor dasar untuk konflik yang diketahui
        for country1, country2 in ACTIVE_CONFLICTS:
            tension_scores[country1] += 50
            tension_scores[country2] += 50

        # Analisis interaksi dari berita
        for source_code, headlines in all_headlines_by_country.items():
            for headline in headlines:
                # Cari dulu kata kunci ketegangan dalam headline
                headline_tension_weight = 0
                for keyword, weight in TENSION_KEYWORDS.items():
                    if keyword.upper() in headline:
                        headline_tension_weight += weight
                
                # Jika ada kata kunci ketegangan, baru cari negara lain yang disebut
                if headline_tension_weight > 0:
                    for target_code, target_name in REVERSE_COUNTRY_MAP.items():
                        if source_code == target_code:
                            continue
                        # Cek apakah nama negara target ada di headline
                        if target_name.upper() in headline:
                            # Jika ya, naikkan skor kedua negara
                            tension_scores[source_code] += headline_tension_weight
                            tension_scores[target_code] += headline_tension_weight
                            # Tambahkan ke link berita untuk visualisasi garis
                            link = tuple(sorted((source_code, target_code)))
                            news_links_set.add(link)

        # 4. Finalisasi data untuk dikirim ke frontend
        all_data['geopolitical'] = {
            "conflicts": ACTIVE_CONFLICTS,
            "news_links": [list(link) for link in news_links_set],
            # Ubah format skor ke format yang dibutuhkan amCharts
            "tension_scores": [{"id": code, "value": min(score, 100)} for code, score in tension_scores.items()],
             # 'news' sekarang menjadi 'all_news' untuk diakses endpoint lain
            "news": all_headlines_by_country 
        }

        app_cache['data'] = all_data
        app_cache['timestamp'] = time.time()
        print(">> GLOBAL SCAN COMPLETE. INTEL CACHE UPDATED.")
        return all_data


# =================================================================
# BAGIAN 3: API ENDPOINTS (Tidak Berubah)
# =================================================================
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/global-status')
def get_global_status():
    all_data = get_all_data()
    if not all_data or 'geopolitical' not in all_data: return jsonify({"error": "GLOBAL DATA NOT AVAILABLE."}), 500
    # Pastikan data yang dikirim hanya yang relevan untuk peta
    geo_data_for_map = {
        'tension_scores': all_data['geopolitical'].get('tension_scores', []),
        'conflicts': all_data['geopolitical'].get('conflicts', []),
        'news_links': all_data['geopolitical'].get('news_links', [])
    }
    return jsonify(geo_data_for_map)

@app.route('/api/ai-summary')
def get_ai_summary():
    all_data = get_all_data()
    if not all_data: return jsonify({"error": "AI OFFLINE. CANNOT GENERATE SUMMARY."}), 500
    summary = generate_ai_summary(all_data)
    return jsonify(summary)

@app.route('/api/country-data/<country_code>')
def get_country_data(country_code):
    all_data = get_all_data()
    if not all_data: return jsonify({"error": "DATA SOURCE OFFLINE."}), 500
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
# BAGIAN 4: FRONTEND HTML (Perubahan Warna Sudah Diterapkan)
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
        }
        @keyframes scanlines { 0% { background-position: 0 0; } 100% { background-position: 0 50px; } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes typing { from { width: 0; } to { width: 100%; } }
        @keyframes blink-caret { from, to { border-color: transparent; } 50% { border-color: var(--accent-color); } }
        @keyframes glitch-text { 2%,64%{ transform: translate(2px,0) skew(0deg); } 4%,60%{ transform: translate(-2px,0) skew(0deg); } 62%{ transform: translate(0,0) skew(5deg); } }
        @keyframes glitch-appear { 0% { clip-path: inset(20% 0 70% 0); } 20% { clip-path: inset(80% 0 10% 0); } 40% { clip-path: inset(40% 0 40% 0); } 60% { clip-path: inset(90% 0 5% 0); } 80% { clip-path: inset(10% 0 85% 0); } 100% { clip-path: inset(0 0 0 0); } }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255,255,255, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255,255,255, 0); } 100% { box-shadow: 0 0 0 0 rgba(255,255,255, 0); } }

        body, html {
            font-family: 'Courier New', Courier, monospace; margin: 0; padding: 0;
            height: 100%; overflow: hidden; background-color: var(--main-bg); color: var(--text-color);
        }
        #chartdiv {
            width: 100%; height: 100vh;
            background-image: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)),
                              linear-gradient(0deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100%, 3px 3px; position: relative;
        }
        #chartdiv::after {
            content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(0deg, rgba(0, 0, 0, 0) 50%, rgba(255, 255, 255, 0.05) 50%);
            background-size: 100% 4px; animation: scanlines 0.2s linear infinite; pointer-events: none;
        }
        #infopanel {
            position: fixed; bottom: 0; left: 0; width: 100%; height: 50vh;
            background-color: var(--panel-bg);
            border-top: 1px solid var(--border-color); box-shadow: 0 -2px 10px rgba(0,0,0,0.5);
            visibility: hidden; opacity: 0; z-index: 1000; display: flex; flex-direction: column;
            backdrop-filter: blur(5px);
        }
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
        ul#news-list { list-style-type: '>> '; padding-left: 20px; margin: 0; font-size: 0.85em; max-height: 40vh; overflow-y: auto; }
        ul#news-list li { margin-bottom: 8px; }
        .placeholder { color: var(--accent-color); text-align: center; padding-top: 40px; font-size: 1.2em; overflow: hidden; white-space: nowrap; margin: 0 auto; letter-spacing: .15em; animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite; }
        
        .modal-backdrop { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); backdrop-filter: blur(5px); z-index: 2000; display: none; }
        .ai-modal { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 90%; max-width: 600px; background: var(--panel-bg); border: 1px solid var(--border-color); z-index: 2001; display: none; animation: fadeIn 0.5s; }
        .modal-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background-color: rgba(0,0,0,0.3); border-bottom: 1px solid var(--border-color); }
        .modal-content { padding: 20px; }
        .summary-item { margin-bottom: 20px; } .summary-item b { color: var(--accent-color); }
        .prob-bar-container { background: #333; height: 20px; border: 1px solid var(--border-color); padding: 2px; }
        .prob-bar { background: var(--accent-color); height: 100%; transition: width 1s ease-out; }
        #ai-button { position: fixed; bottom: 20px; right: 20px; z-index: 1500; width: 50px; height: 50px; border-radius: 50%; background: var(--accent-color); color: var(--main-bg); border: 2px solid var(--main-bg); font-weight: bold; font-size: 1.2em; cursor: pointer; animation: pulse 2s infinite; }
        #ai-button:hover { animation: none; }

        ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #555; } ::-webkit-scrollbar-thumb:hover { background: #888; }
        @media (max-width: 768px) {
            #infopanel { height: 75vh; } .info-container { flex-direction: column; }
            .analysis-column, .news-column { border-left: none; padding-left: 0; border-top: 1px solid var(--border-color); padding-top: 20px; margin-top: 20px; }
            .ai-modal { width: 95%; }
        }
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
    const aiButton = document.getElementById('ai-button'), modalBackdrop = document.getElementById('modal-backdrop'), aiModal = document.getElementById('ai-modal'), aiCloseBtn = document.getElementById('ai-close-btn'), aiModalContent = document.getElementById('ai-modal-content');

    am5.ready(function() {
        var root = am5.Root.new("chartdiv");
        root.setThemes([am5themes_Animated.new(root)]);
        var chart = root.container.children.push(am5map.MapChart.new(root, { panX: "pan", panY: "pan", projection: am5map.geoEquirectangular(), wheelY: "zoom" }));
        
        chart.chartContainer.set("background", am5.Rectangle.new(root, { fill: am5.color(0x000000), fillOpacity: 1 }));

        var polygonSeries = chart.series.push(am5map.MapPolygonSeries.new(root, { geoJSON: am5geodata_worldLow, exclude: ["AQ"], valueField: "value", calculateAggregates: true }));
        
        polygonSeries.mapPolygons.template.setAll({
            interactive: true,
            // =========================================================================================
            // === PERUBAHAN DI SINI: Warna default diubah menjadi merah tua gelap ===
            // =========================================================================================
            fill: am5.color(0x550000), 
            stroke: am5.color(0x444444), 
            strokeWidth: 0.5,
            transitionDuration: 500
        });

        polygonSeries.mapPolygons.template.adapters.add("tooltipText", function(text, target) {
          if (target.dataItem.get("value") == null) {
            return "{name}: Tension Unknown";
          }
          return "{name}: TENSION SCORE {value}";
        });

        polygonSeries.mapPolygons.template.states.create("hover", {
             fill: am5.color(0x64b5f6) 
        });
        
        polygonSeries.set("heatRules", [{
            target: polygonSeries.mapPolygons.template,
            dataField: "value",
            min: am5.color(0x000000), 
            max: am5.color(0xff0000), 
            key: "fill",
            logarithmic: true 
        }]);

        var conflictLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        conflictLineSeries.mapLines.template.setAll({ stroke: am5.color(0xff0000), strokeOpacity: 0.6, strokeWidth: 2, strokeDasharray: [4,2] });
        
        var newsLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        newsLineSeries.mapLines.template.setAll({ stroke: am5.color(0x64b5f6), strokeOpacity: 0.6, strokeWidth: 1, arc: -0.2 });

        fetch('/api/global-status').then(res => res.json()).then(data => {
            if (!data) return;
            polygonSeries.data.setAll(data.tension_scores);
            (data.conflicts || []).forEach(conflict => renderLine(conflict[0], conflict[1], conflictLineSeries, true));
            (data.news_links || []).forEach(link => renderLine(link[0], link[1], newsLineSeries, false));
        }).catch(err => console.error("Failed to load global status:", err));

        function renderLine(code1, code2, series, isConflict) {
            let p1 = polygonSeries.getPolygonById(code1); let p2 = polygonSeries.getPolygonById(code2);
            if(p1 && p2) {
                let lineDataItem = series.pushDataItem({
                    geometry: {
                        type: "LineString",
                        coordinates: [
                            [p1.visualCentroid.longitude, p1.visualCentroid.latitude],
                            [p2.visualCentroid.longitude, p2.visualCentroid.latitude]
                        ]
                    }
                });
                
                if (isConflict) {
                     let bullet = am5.Bullet.new(root, {
                        sprite: am5.Circle.new(root, {
                            radius: 3,
                            fill: am5.color(0xff0000)
                        })
                    });
                    bullet.animate({ key: "location", from: 0, to: 1, duration: 2000, loops: Infinity });
                    lineDataItem.bullets.push(bullet);
                }
            }
        }
        
        polygonSeries.mapPolygons.template.events.on("click", function(ev) {
            if (ev.target.dataItem.get("value") != null) {
                const countryId = ev.target.dataItem.dataContext.id;
                const countryName = ev.target.dataItem.dataContext.name;
                fetchCountryData(countryId, countryName);
            }
        });

        chart.chartContainer.get("background").events.on("click", () => closeInfoPanel());
    });
    
    closeButton.addEventListener('click', closeInfoPanel);
    aiButton.addEventListener('click', fetchAiSummary);
    aiCloseBtn.addEventListener('click', closeAiModal);
    modalBackdrop.addEventListener('click', closeAiModal);
    function showInfoPanel() { infoPanel.classList.add('visible'); }
    function closeInfoPanel() { infoPanel.classList.remove('visible'); panelTitle.classList.remove('glitch'); }
    function showAiModal() { modalBackdrop.style.display = 'block'; aiModal.style.display = 'block'; }
    function closeAiModal() { modalBackdrop.style.display = 'none'; aiModal.style.display = 'none'; }
    
    async function fetchAiSummary() {
        aiModalContent.innerHTML = '<p class="placeholder">// AI ANALYZING GLOBAL DATA... //</p>';
        showAiModal();
        try {
            const response = await fetch('/api/ai-summary');
            if (!response.ok) throw new Error('AI OFFLINE. CANNOT GENERATE SUMMARY.');
            const data = await response.json();
            displayAiSummary(data);
        } catch (error) { aiModalContent.innerHTML = `<p class="placeholder" style="color: var(--red-alert);">${error.message}</p>`; }
    }

    function displayAiSummary(data) {
        aiModalContent.innerHTML = `
            <div class="summary-item"><b>GLOBAL TENSION LEVEL:</b> ${data.world_tension}% <div class="prob-bar-container"><div class="prob-bar" style="width: ${data.world_tension}%; background-color: var(--red-alert);"></div></div></div>
            <div class="summary-item"><b>MAJOR POWER CONFLICT PROBABILITY:</b> ${data.war_probability}% <div class="prob-bar-container"><div class="prob-bar" style="width: ${data.war_probability}%; background-color: var(--red-alert);"></div></div></div>
            <div class="summary-item"><b>FED NEXT MOVE PROBABILITY:</b><br> CUT: ${data.fed_cut_prob}% <div class="prob-bar-container"><div class="prob-bar" style="width: ${data.fed_cut_prob}%;"></div></div> HOLD: ${data.fed_hold_prob}% <div class="prob-bar-container"><div class="prob-bar" style="width: ${data.fed_hold_prob}%;"></div></div> HIKE: ${data.fed_hike_prob}% <div class="prob-bar-container"><div class="prob-bar" style="width: ${data.fed_hike_prob}%;"></div></div></div>
            <div class="summary-item"><b>MARKET IMPACT ANALYSIS:</b><br><p>${data.market_impact}</p></div>`;
    }

    async function fetchCountryData(countryCode, countryName) {
        infoContent.innerHTML = `<p class="placeholder">ESTABLISHING DATALINK: ${countryName.toUpperCase()}...</p>`;
        panelTitle.innerText = countryName.toUpperCase();
        showInfoPanel();
        try {
            const response = await fetch(`/api/country-data/${countryCode}`);
            if (!response.ok) throw new Error('CONNECTION FAILED. DATA STREAM CORRUPTED.');
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
                <div class="data-column">
                    <h3>// MONETARY DATA</h3>${ratesHtml}
                    <h3>// RECENT DIRECTIVES</h3>${decisionsHtml}
                    <h3>// UPCOMING TRANSMISSIONS</h3>${meetingsHtml}
                </div>
                <div class="news-column"><h3>// RECENT INTEL</h3>${newsHtml}</div>
                <div class="analysis-column">
                    <h3>${data.analysis.title}</h3>
                    <p><b>SUMMARY:</b> ${data.analysis.summary.replace(/\*\*(.*?)\*\*/g, '<span>$1</span>')}</p><br>
                    <p><b>IMPACT ANALYSIS:</b><br>${data.analysis.impact.replace(/\*\*(.*?)\*\*/g, '<span>$1</span>')}</p>
                </div>
            </div>`;
    }

    function createTable(data, headers) {
        let table = '<table><thead><tr>'; headers.forEach(h => table += `<th>${h.toUpperCase()}</th>`); table += '</tr></thead><tbody>';
        data.forEach(row => {
            table += '<tr>';
            headers.forEach(h => {
                const key = h.toLowerCase().replace(/ /g, '_'); table += `<td>${row[key] || 'N/A'}</td>`;
            });
            table += '</tr>';
        });
        table += '</tbody></table>'; return table;
    }
</script>

</body>
</html>
"""

# ===============================================
# BAGIAN 5: MENJALANKAN APLIKASI
# ===============================================
if __name__ == '__main__':
    print("===============================================================")
    print(">> G.I.M.P.S (vMaximum Coverage) :: BOOTING...")
    print(">> Menginisialisasi koneksi dan memuat data intelijen global...")
    get_all_data()
    print(">> SISTEM ONLINE. Menunggu perintah di command interface...")
    print(">> Buka browser Anda dan akses alamat berikut:")
    print(">> http://127.0.0.1:5001")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
