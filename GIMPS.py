import time
from flask import Flask, jsonify, render_template_string, Response
import cloudscraper
from bs4 import BeautifulSoup
from threading import Lock, Thread
import feedparser
import collections
import json
import random
import datetime
import requests
# --- MODIFIKASI TERMUX ---: Library 'openai' sudah tidak digunakan lagi.

# Coba impor plyer untuk notifikasi desktop
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("[WARNING] Library 'plyer' tidak ditemukan. Notifikasi desktop tidak akan berfungsi. Install dengan 'pip install plyer'")


# =================================================================
# BAGIAN 1: KONFIGURASI
# =================================================================
app = Flask(__name__)
scraper = cloudscraper.create_scraper()

# --- KONFIGURASI OPENROUTER AI ---
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"
OPENROUTER_SITE_URL = "https://openrouter.ai/api/v1"

# --- KONFIGURASI NOTIFIKASI GLOBAL ---
NTFY_TOPIC = "gimps-global-notificationA87X"
GLOBAL_ALERT_KEYWORDS = [
    'WAR', 'MILITARY', 'CONFLICT', 'INVASION', 'ATTACK', 'BOMB', 'DRONE', 'MISSILE',
    'CRISIS', 'RECESSION', 'CRASH', 'DEFAULT', 'COLLAPSE', 'MARKET PANIC',
    'CENTRAL BANK', 'THE FED', 'ECB', 'INTEREST RATE', 'EMERGENCY MEETING', 'QUANTITATIVE EASING',
    'PANDEMIC', 'OUTBREAK', 'QUARANTINE', 'LOCKDOWN'
]
GLOBAL_ALERT_THRESHOLD = 3

# --- DAFTAR NEGARA ---
FULL_COUNTRY_MAP = {
    # Amerika
    'Argentina': 'AR', 'Bolivia': 'BO', 'Brazil': 'BR', 'Canada': 'CA', 'Chile': 'CL', 'Colombia': 'CO', 'Cuba': 'CU', 'Ecuador': 'EC', 'Mexico': 'MX', 'Panama': 'PA', 'Paraguay': 'PY', 'Peru': 'PE', 'USA': 'US', 'United States': 'US', 'Uruguay': 'UY', 'Venezuela': 'VE',
    # Eropa
    'Albania': 'AL', 'Austria': 'AT', 'Belarus': 'BY', 'Belgium': 'BE', 'Bosnia and Herzegovina': 'BA', 'Bulgaria': 'BG', 'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT', 'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT', 'Moldova': 'MD', 'Netherlands': 'NL', 'North Macedonia': 'MK', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Russia': 'RU', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH', 'Ukraine': 'UA', 'United Kingdom': 'GB',
    # Asia & Oseania
    'Afghanistan': 'AF', 'Australia': 'AU', 'Bangladesh': 'BD', 'Cambodia': 'KH', 'China': 'CN', 'Hong Kong': 'HK', 'India': 'IN', 'Indonesia': 'ID', 'Japan': 'JP', 'Kazakhstan': 'KZ', 'Kyrgyzstan': 'KG', 'Malaysia': 'MY', 'Mongolia': 'MN', 'Myanmar': 'MM', 'Nepal': 'NP', 'New Zealand': 'NZ', 'North Korea': 'KP', 'Pakistan': 'PK', 'Philippines': 'PH', 'Singapore': 'SG', 'South Korea': 'KR', 'Sri Lanka': 'LK', 'Taiwan': 'TW', 'Tajikistan': 'TJ', 'Thailand': 'TH', 'Turkmenistan': 'TM', 'Uzbekistan': 'UZ', 'Vietnam': 'VN',
    # Timur Tengah
    'Armenia': 'AM', 'Azerbaijan': 'AZ', 'Bahrain': 'BH', 'Egypt': 'EG', 'Georgia': 'GE', 'Iran': 'IR', 'Iraq': 'IQ', 'Israel': 'IL', 'Jordan': 'JO', 'Kuwait': 'KW', 'Lebanon': 'LB', 'Oman': 'OM', 'Qatar': 'QA', 'Saudi Arabia': 'SA', 'Syria': 'SY', 'Türkiye': 'TR', 'Turkey': 'TR', 'UAE': 'AE', 'Yemen': 'YE',
    # Afrika
    'Algeria': 'DZ', 'Angola': 'AO', 'Botswana': 'BW', 'Burkina Faso': 'BF', 'Cameroon': 'CM', 'DR Congo': 'CD', 'Ethiopia': 'ET', 'Ghana': 'GH', 'Ivory Coast': 'CI', "Côte d'Ivoire": 'CI', 'Kenya': 'KE', 'Libya': 'LY', 'Madagascar': 'MG', 'Mali': 'ML', 'Mauritius': 'MU', 'Morocco': 'MA', 'Mozambique': 'MZ', 'Namibia': 'NA', 'Niger': 'NE', 'Nigeria': 'NG', 'Rwanda': 'RW', 'Senegal': 'SN', 'Somalia': 'SO', 'South Africa': 'ZA', 'Sudan': 'SD', 'Tanzania': 'TZ', 'Tunisia': 'TN', 'Uganda': 'UG', 'Zambia': 'ZM', 'Zimbabwe': 'ZW'
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
global_data = { 'rates': [], 'decisions': [], 'meetings': [], 'all_news': {code: [] for code in REVERSE_COUNTRY_MAP.keys()}, 'geopolitical': { "tension_scores": [], "conflicts": ACTIVE_CONFLICTS, "news_links": [] }, 'last_updated_code': None, 'headline_tracker': collections.defaultdict(lambda: {'countries': set()}), 'sent_notifications': set() }
MONETARY_UPDATE_INTERVAL_SECONDS = 900

# =================================================================
# BAGIAN 2: LOGIKA AI (TANPA LIBRARY OPENAI)
# =================================================================

def call_openrouter_api(system_prompt, user_prompt):
    """
    --- MODIFIKASI TERMUX ---
    Fungsi pusat untuk memanggil API OpenRouter menggunakan library 'requests'.
    Ini menggantikan kebutuhan akan library 'openai' dan lebih portabel.
    """
    api_url = f"{OPENROUTER_SITE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Bisa diisi apa saja, praktik yang baik
        "X-Title": "GIMPS AI System"       # Nama aplikasi Anda
    }
    
    payload = {
        "model": OPENROUTER_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 700
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()  # Akan melempar error untuk status 4xx/5xx
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"[!!] OPENROUTER API NETWORK ERROR: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[!!] OPENROUTER API RESPONSE PARSE ERROR: {e}")
        return None


def generate_ai_summary(all_data):
    """Menghasilkan ringkasan global menggunakan LLM."""
    geo_data = all_data.get('geopolitical', {})
    all_scores = [s['value'] for s in geo_data.get('tension_scores', [])]
    world_tension = int(sum(all_scores) / len(all_scores)) if all_scores else 30
    war_probability = min(int(world_tension * 0.8 + len(geo_data.get('conflicts', [])) * 4), 95)
    hike_score, cut_score = 0, 0
    us_headlines = all_data.get('all_news', {}).get('US', [])
    for headline in us_headlines:
        if any(keyword in headline for keyword in FED_HIKE_KEYWORDS): hike_score += 1
        if any(keyword in headline for keyword in FED_CUT_KEYWORDS): cut_score += 1
    monetary_stance = "NEUTRAL"
    if hike_score > cut_score + 1: monetary_stance = "HAWKISH"
    if cut_score > hike_score + 1: monetary_stance = "DOVISH"
    active_hotspots = [f"{REVERSE_COUNTRY_MAP.get(c1, c1)} vs {REVERSE_COUNTRY_MAP.get(c2, c2)}" for c1, c2 in geo_data.get('conflicts', [])]
    active_conflict_countries = {code for pair in geo_data.get('conflicts', []) for code in pair}
    sorted_scores = sorted(geo_data.get('tension_scores', []), key=lambda x: x['value'], reverse=True)
    potential_hotspots = [f"{REVERSE_COUNTRY_MAP.get(score['id'], score['id'])} (Skor: {score['value']})" for score in sorted_scores if score['id'] not in active_conflict_countries and len(potential_hotspots) < 5]
    system_prompt = "You are G.I.M.P.S, a global intelligence analysis system. Your task is to provide a concise market analysis based ONLY on the data provided. Do not use external knowledge. Your output MUST be a valid JSON object with two keys: \"market_outlook\" and \"strategic_posture\". The analysis should be sharp, direct, and in English."
    user_prompt = f"""Analyze the following global intelligence data:\n- Global Tension Score: {world_tension}/100\n- Implied War Probability: {war_probability}%\n- US Monetary Stance (Proxy for Global Markets): {monetary_stance}\n- Active Military Hotspots: {', '.join(active_hotspots) if active_hotspots else 'None'}\n- Potential Escalation Zones (High Tension): {', '.join(potential_hotspots) if potential_hotspots else 'None'}\n\nBased ONLY on this data, generate the JSON output."""
    ai_response_str = call_openrouter_api(system_prompt, user_prompt)
    analysis_result = {"market_outlook": "AI analysis failed or is pending.", "strategic_posture": "Could not determine posture due to API error."}
    if ai_response_str:
        try:
            json_start = ai_response_str.find('{'); json_end = ai_response_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                clean_json_str = ai_response_str[json_start:json_end]
                analysis_result = json.loads(clean_json_str)
            else: raise ValueError("No JSON object found in the AI response.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!!] AI-SUMMARY JSON PARSE ERROR: {e}"); analysis_result['market_outlook'] = "AI response was not in the correct format."
    return {"world_tension": world_tension, "war_probability": war_probability, "market_outlook": analysis_result.get("market_outlook", "N/A"), "strategic_posture": analysis_result.get("strategic_posture", "N/A"), "active_hotspots": active_hotspots, "potential_hotspots": potential_hotspots}

# =================================================================
# BAGIAN 3: FUNGSI-FUNGSI HELPER
# =================================================================
def send_global_alert(headline):
    print(f"\n[!!!] GLOBAL ALERT TRIGGERED: {headline}\n")
    if PLYER_AVAILABLE:
        try:
            notification.notify(title='G.I.M.P.S. GLOBAL ALERT', message=headline, app_name='G.I.M.P.S.', timeout=20)
        except Exception as e: print(f"[ERROR] Gagal mengirim notifikasi desktop: {e}")
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=headline.encode(encoding='utf-8'), headers={"Title": "G.I.M.P.S. Global Alert", "Priority": "high", "Tags": "warning"})
    except Exception as e: print(f"[ERROR] Gagal mengirim notifikasi ntfy: {e}")

def parse_world_rates(soup):
    data = []; table = soup.select_one('table#AutoNumber3')
    if not table: return []
    for row in table.find_all('tr'):
        cols = row.find_all('td');
        if len(cols) < 7: continue
        try:
            country_name_raw = cols[4].get_text(strip=True).split('|')[0].strip(); country_name = country_name_raw.replace('The', '').strip()
            if country_name.lower() == 'türkiye': country_name = 'Türkiye'
            data.append({ "country_name": country_name.upper(), "country_code": COUNTRY_MAP.get(country_name, None), "rate": cols[1].get_text(strip=True), "change": cols[2].get_text(strip=True), "date": cols[5].get_text(strip=True), "rate_name": cols[4].get_text(strip=True).replace(country_name_raw, '').lstrip('| ').strip().upper() })
        except (IndexError, AttributeError): continue
    return data

def parse_decisions_or_meetings(soup, type):
    data = []; table = soup.select_one('table#AutoNumber3')
    if not table: return []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) < 4: continue
        try:
            date_text = cols[1].get_text(strip=True); full_text = cols[2].get_text(strip=True); country_name = "UNKNOWN"; action = "MEETING"
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

def generate_analysis(country_code, country_name, decision_data, meetings_data, news_headlines):
    latest_decision = decision_data[0] if decision_data else {"action": "N/A", "description": "No recent directive data."}
    upcoming_meetings_str = ', '.join([m['description'] for m in meetings_data]) if meetings_data else "No upcoming meetings scheduled."
    news_headlines_str = ', '.join(news_headlines[:5]) if news_headlines else "No recent headlines available."
    system_prompt = "You are a top-tier geopolitical and financial market strategist. Your task is to provide a sharp, professional analysis based ONLY on the data provided. Your output MUST be a valid JSON object with four string keys: \"title\", \"summary\", \"impact_analysis\", and \"futures_outlook\". Do not use external knowledge. The analysis must be in English."
    user_prompt = f"""
    Analyze the following intelligence data for {country_name.upper()} ({country_code}):
    
    1.  **Latest Central Bank Directive:**
        - Action: {latest_decision.get('action', 'N/A')}
        - Description: "{latest_decision.get('description', 'N/A')}"

    2.  **Recent News Headlines (for context):**
        - {news_headlines_str}

    3.  **Upcoming Transmissions (Scheduled Meetings):**
        - {upcoming_meetings_str}

    Based ONLY on this data, perform the following analysis and generate the JSON output:
    - **title:** A concise, impactful title for the analysis.
    - **summary:** A brief summary of the current situation based on the latest directive and news.
    - **impact_analysis:** Analyze the immediate impact of the latest directive on the local economy, currency, and potential spillover to global/crypto markets.
    - **futures_outlook:** This is crucial. Based on the recent news and the upcoming transmissions, forecast the likely market sentiment and potential central bank actions in the near future. What should be watched for during the upcoming meetings?
    """
    ai_response_str = call_openrouter_api(system_prompt, user_prompt)
    fallback_analysis = { "title": f"AI ANALYSIS FAILED for {country_name.upper()}", "summary": "Could not generate analysis due to an API or parsing error.", "impact_analysis": f"The raw data indicates a '{latest_decision.get('action', 'N/A')}' action. Please check server logs for API error details.", "futures_outlook": "Forecasting is unavailable due to the analysis failure." }
    if ai_response_str:
        try:
            json_start = ai_response_str.find('{'); json_end = ai_response_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                clean_json_str = ai_response_str[json_start:json_end]
                return json.loads(clean_json_str)
            else: raise ValueError("No JSON object found in the AI response.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!!] AI-ANALYSIS JSON PARSE ERROR: {e}")
            return fallback_analysis
    return fallback_analysis

# =================================================================
# BAGIAN 4: PROSES BACKGROUND
# =================================================================
def get_news_from_rss(country_code):
    try:
        url = f"https://news.google.com/rss?gl={country_code.upper()}&hl=en-US&ceid={country_code.upper()}:en"
        feed = feedparser.parse(url)
        if feed.bozo: raise Exception(feed.bozo_exception)
        processed_news = []; date_str = datetime.datetime.now().strftime('%d/%m/%Y')
        for entry in feed.entries[:20]:
            core_headline = entry.title.upper().rsplit(' - ', 1)[0].strip()
            formatted_headline = f"{core_headline} ({date_str})"
            processed_news.append({"core": core_headline, "formatted": formatted_headline})
        return processed_news
    except Exception as e:
        print(f"Error fetching RSS for {country_code}: {e}"); return []

def background_update_task():
    country_codes = list(REVERSE_COUNTRY_MAP.keys()); random.shuffle(country_codes); country_index = 0; last_monetary_update_time = 0
    while True:
        try:
            current_time = time.time()
            if (current_time - last_monetary_update_time) > MONETARY_UPDATE_INTERVAL_SECONDS:
                print(">> [TIMER] Performing periodic monetary data scan..."); cbrates_urls = {'rates': 'https://www.cbrates.com/', 'decisions': 'https://www.cbrates.com/decisions.htm', 'meetings': 'https://www.cbrates.com/meetings.htm'}
                with data_lock:
                    for key, url in cbrates_urls.items():
                        try:
                            response = scraper.get(url, timeout=20); response.raise_for_status(); soup = BeautifulSoup(response.text, 'html.parser')
                            if key == 'rates': global_data['rates'] = parse_world_rates(soup)
                            elif key == 'decisions': global_data['decisions'] = parse_decisions_or_meetings(soup, 'decisions')
                            elif key == 'meetings': global_data['meetings'] = parse_decisions_or_meetings(soup, 'meetings')
                        except Exception as e: print(f"!! Failed to fetch monetary data for {key}: {e}")
                last_monetary_update_time = current_time; print(">> [TIMER] Monetary scan complete.")

            country_code = country_codes[country_index]; country_name = REVERSE_COUNTRY_MAP.get(country_code, country_code); print(f">> MAX SPEED SCAN: [{country_index + 1}/{len(country_codes)}] {country_name}")
            news_items = get_news_from_rss(country_code)
            
            with data_lock:
                formatted_headlines = [item['formatted'] for item in news_items]
                if formatted_headlines and (country_code not in global_data['all_news'] or global_data['all_news'][country_code] != formatted_headlines):
                    global_data['all_news'][country_code] = formatted_headlines; global_data['last_updated_code'] = country_code
                
                for item in news_items:
                    core_headline = item['core']; global_data['headline_tracker'][core_headline]['countries'].add(country_code); count = len(global_data['headline_tracker'][core_headline]['countries'])
                    is_above_threshold = count >= GLOBAL_ALERT_THRESHOLD; is_not_sent_yet = core_headline not in global_data['sent_notifications']; has_alert_keyword = any(keyword in core_headline for keyword in GLOBAL_ALERT_KEYWORDS)
                    if is_above_threshold and is_not_sent_yet and has_alert_keyword:
                        send_global_alert(core_headline); global_data['sent_notifications'].add(core_headline)

                tension_scores = collections.defaultdict(int)
                news_links_set = set()
                
                for c1, c2 in ACTIVE_CONFLICTS: 
                    tension_scores[c1] += 50
                    tension_scores[c2] += 50
                
                for source_code, headlines_list in global_data['all_news'].items():
                    for headline_text in headlines_list:
                        core_headline = headline_text.rsplit(' (', 1)[0]
                        headline_tension_weight = sum(weight for keyword, weight in TENSION_KEYWORDS.items() if keyword.upper() in core_headline)

                        if headline_tension_weight > 0:
                            tension_scores[source_code] += headline_tension_weight
                            for target_code, target_name in REVERSE_COUNTRY_MAP.items():
                                if source_code != target_code and target_name.upper() in core_headline:
                                    tension_scores[target_code] += headline_tension_weight
                                    news_links_set.add(tuple(sorted((source_code, target_code))))
                
                global_data['geopolitical']['tension_scores'] = [{"id": code, "value": min(score, 100)} for code, score in tension_scores.items()]
                global_data['geopolitical']['news_links'] = [list(link) for link in news_links_set]

            country_index = (country_index + 1) % len(country_codes)
            
        except Exception as e:
            print(f"!! ERROR in background task: {e}"); time.sleep(10)

# =================================================================
# BAGIAN 5: API ENDPOINTS
# =================================================================
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/stream-updates')
def stream_updates():
    def event_stream():
        while True:
            time.sleep(0.5)
            with data_lock:
                data_to_send = {'geopolitical': global_data['geopolitical'], 'updated_country_code': global_data.get('last_updated_code')}
                yield f"data: {json.dumps(data_to_send)}\n\n"
                if global_data['last_updated_code']: global_data['last_updated_code'] = None
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/ai-summary')
def get_ai_summary():
    with data_lock: summary = generate_ai_summary(global_data)
    return jsonify(summary)

@app.route('/api/country-data/<country_code>')
def get_country_data(country_code):
    with data_lock:
        all_data = dict(global_data); country_name = REVERSE_COUNTRY_MAP.get(country_code.upper(), country_code.upper())
    country_code_upper = country_code.upper(); is_eurozone = country_code_upper == 'EU'
    def filter_by_code(data_list):
        if is_eurozone: return [d for d in data_list if d.get('country_name') == 'EUROZONE']
        return [d for d in data_list if d.get('country_code') == country_code_upper]
    decisions_data = filter_by_code(all_data.get('decisions', []))
    meetings_data = filter_by_code(all_data.get('meetings', []))
    news_data_formatted = all_data.get('all_news', {}).get(country_code_upper, [])
    news_headlines_core = [headline.rsplit(' (', 1)[0] for headline in news_data_formatted]
    analysis = generate_analysis(country_code_upper, country_name, decisions_data, meetings_data, news_headlines_core)
    return jsonify({"rates": filter_by_code(all_data.get('rates', [])), "decisions": decisions_data, "meetings": meetings_data, "news": news_data_formatted, "analysis": analysis})

# =================================================================
# BAGIAN 6: FRONTEND HTML
# =================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G.I.M.P.S - Global Intelligence Monetary Policy System</title>
    <style>
        :root { --main-bg: #000; --panel-bg: rgba(10, 10, 10, 0.9); --border-color: rgba(255, 255, 255, 0.2); --accent-color: #FFF; --text-color: #dcdcdc; --red-alert: #ff4d4d; --blue-link: #64b5f6; --flash-color: #FFFF00; }
        @keyframes scanlines { 0% { background-position: 0 0; } 100% { background-position: 0 50px; } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes typing { from { width: 0; } to { width: 100%; } }
        @keyframes blink-caret { from, to { border-color: transparent; } 50% { border-color: var(--accent-color); } }
        @keyframes glitch-text { 2%,64%{ transform: translate(2px,0) skew(0deg); } 4%,60%{ transform: translate(-2px,0) skew(0deg); } 62%{ transform: translate(0,0) skew(5deg); } }
        @keyframes glitch-appear { 0% { clip-path: inset(20% 0 70% 0); } 20% { clip-path: inset(80% 0 10% 0); } 40% { clip-path: inset(40% 0 40% 0); } 60% { clip-path: inset(90% 0 5% 0); } 80% { clip-path: inset(10% 0 85% 0); } 100% { clip-path: inset(0 0 0 0); } }
        body, html { font-family: 'Courier New', Courier, monospace; margin: 0; padding: 0; height: 100%; overflow: hidden; background-color: var(--main-bg); color: var(--text-color); }
        body::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; box-shadow: inset 0 0 150px rgba(0,0,0,0.9); pointer-events: none; z-index: 9999; }
        #chartdiv { width: 100%; height: 100vh; background-image: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), linear-gradient(0deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px); background-size: 100%, 3px 3px; position: relative; transform: scale(1.05); border-radius: 40px; }
        #chartdiv::after { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(0deg, rgba(0, 0, 0, 0) 50%, rgba(255, 255, 255, 0.05) 50%); background-size: 100% 4px; animation: scanlines 0.2s linear infinite; pointer-events: none; }
        #infopanel { position: fixed; bottom: 0; left: 0; width: 100%; height: 60vh; background-color: var(--panel-bg); border-top: 1px solid var(--border-color); box-shadow: 0 -2px 10px rgba(0,0,0,0.5); visibility: hidden; opacity: 0; z-index: 1000; display: flex; flex-direction: column; backdrop-filter: blur(5px); }
        #infopanel.visible { visibility: visible; opacity: 1; animation: glitch-appear 0.2s steps(8, end) forwards; }
        #panel-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background-color: rgba(0,0,0,0.3); border-bottom: 1px solid var(--border-color); }
        #panel-title-container { display: flex; align-items: center; gap: 15px; }
        .glitch { position: relative; animation: glitch-text 2s infinite; }
        #close-panel, #export-pdf-btn { background: none; border: none; color: var(--accent-color); font-size: 28px; cursor: pointer; transition: transform 0.2s; }
        #close-panel:hover, #export-pdf-btn:hover { transform: scale(1.5); }
        #export-pdf-btn { font-size: 22px; display: none; }
        #info-content { padding: 20px; overflow-y: hidden; flex-grow: 1; animation: fadeIn 1s; }
        .info-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 20px; height: 100%; }
        .data-container, .analysis-container { overflow-y: auto; padding-right: 10px;}
        .analysis-container { border-left: 1px solid var(--border-color); padding-left: 20px; }
        h3 { color: var(--accent-color); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-top: 0; margin-bottom: 15px; font-size: 1.2em; text-transform: uppercase;}
        h4 { color: var(--accent-color); margin-top: 0; margin-bottom: 5px; font-size: 1.1em; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.9em; }
        th, td { border: 1px solid var(--border-color); padding: 8px; text-align: left; }
        th { background-color: rgba(255, 255, 255, 0.05); }
        #news-list { list-style-type: '>> '; padding-left: 20px; margin: 0; font-size: 0.85em; }
        #news-list li { margin-bottom: 8px; }
        .placeholder { color: var(--accent-color); text-align: center; padding-top: 40px; font-size: 1.2em; overflow: hidden; white-space: nowrap; margin: 0 auto; letter-spacing: .15em; animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite; }
        .analysis-content p { margin-top: 0; margin-bottom: 1.2em; white-space: pre-wrap; line-height: 1.5; }
        .analysis-content strong { color: var(--blue-link); }
        ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #555; }
    </style>
</head>
<body>
<div id="chartdiv"></div>
<div id="infopanel">
    <div id="panel-header">
        <div id="panel-title-container">
            <h2 id="panel-title" style="border:none; margin:0; padding:0;">// SYSTEM STANDBY //</h2>
            <button id="export-pdf-btn" title="Export Briefing to PDF">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            </button>
        </div>
        <button id="close-panel" title="TERMINATE CONNECTION">×</button>
    </div>
    <div id="info-content">
        <p class="placeholder">// AWAITING COMMAND //</p>
    </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdn.amcharts.com/lib/5/index.js"></script><script src="https://cdn.amcharts.com/lib/5/map.js"></script><script src="https://cdn.amcharts.com/lib/5/geodata/worldLow.js"></script><script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>
<script>
    const infoPanel = document.getElementById('infopanel'), infoContent = document.getElementById('info-content'), panelTitle = document.getElementById('panel-title'), closeButton = document.getElementById('close-panel');
    const exportPdfBtn = document.getElementById('export-pdf-btn');

    function showNotification(title, body) { if (Notification.permission === 'granted') { new Notification(title, { body: body, icon: '/favicon.ico' }); } }
    am5.ready(function() {
        if ('Notification' in window) { if (Notification.permission !== 'granted' && Notification.permission !== 'denied') { Notification.requestPermission().then(function(permission) { if (permission === 'granted') { console.log('Notification permission granted.'); showNotification('G.I.M.P.S.', 'System notifications are now active.'); } }); } }
        var root = am5.Root.new("chartdiv"); root.setThemes([am5themes_Animated.new(root)]); var chart = root.container.children.push(am5map.MapChart.new(root, { panX: "pan", panY: "pan", projection: am5map.geoEquirectangular(), wheelY: "zoom" })); chart.chartContainer.set("background", am5.Rectangle.new(root, { fill: am5.color(0x000000), fillOpacity: 1 }));
        var polygonSeries = chart.series.push(am5map.MapPolygonSeries.new(root, { geoJSON: am5geodata_worldLow, exclude: ["AQ"], valueField: "value", calculateAggregates: true }));
        polygonSeries.mapPolygons.template.setAll({ interactive: true, fill: am5.color(0x550000), stroke: am5.color(0x444444), strokeWidth: 0.5, transitionDuration: 500 });
        polygonSeries.mapPolygons.template.adapters.add("tooltipText", function(text, target) { if (target.dataItem.get("value") == null || target.dataItem.get("value") === 0) { return "{name}: Tension Unknown / Stable"; } return "{name}: TENSION SCORE {value}"; });
        polygonSeries.mapPolygons.template.states.create("hover", { fill: am5.color(0x64b5f6) }); polygonSeries.set("heatRules", [{ target: polygonSeries.mapPolygons.template, dataField: "value", min: am5.color(0x000000), max: am5.color(0xff0000), key: "fill", logarithmic: true }]);
        var conflictLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {})); conflictLineSeries.mapLines.template.setAll({ stroke: am5.color(0xff0000), strokeOpacity: 0.6, strokeWidth: 2, strokeDasharray: [4,2] });
        var newsLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {})); newsLineSeries.mapLines.template.setAll({ stroke: am5.color(0x64b5f6), strokeOpacity: 0.6, strokeWidth: 1, arc: -0.2 });
        function updateMap(geoData) { if (!geoData || !polygonSeries.data.length) return; (geoData.tension_scores || []).forEach(score => { const dataItem = polygonSeries.getDataItemById(score.id); if (dataItem) { dataItem.set("value", score.value); } }); conflictLineSeries.data.clear(); newsLineSeries.data.clear(); (geoData.conflicts || []).forEach(conflict => renderLine(conflict[0], conflict[1], conflictLineSeries, true)); (geoData.news_links || []).forEach(link => renderLine(link[0], link[1], newsLineSeries, false)); }
        function flashCountry(countryCode) { let polygon = polygonSeries.getPolygonById(countryCode); if (polygon) { const originalColor = polygon.get("fill"); const flashColor = am5.color(0xFFFF00); let animation = polygon.animate({ key: "fill", to: flashColor, duration: 5400 }); if (animation) { animation.events.on("stopped", function() { polygon.animate({ key: "fill", to: originalColor, duration: 1800 }); }); } } }
        function renderLine(code1, code2, series, isConflict) { let p1 = polygonSeries.getPolygonById(code1); let p2 = polygonSeries.getPolygonById(code2); if(p1 && p2) { let lineDataItem = series.pushDataItem({ geometry: { type: "LineString", coordinates: [[p1.visualCentroid.longitude, p1.visualCentroid.latitude], [p2.visualCentroid.longitude, p2.visualCentroid.latitude]] } }); if (isConflict) { let bullet = am5.Bullet.new(root, { sprite: am5.Circle.new(root, { radius: 3, fill: am5.color(0xff0000) }) }); bullet.animate({ key: "location", from: 0, to: 1, duration: 2000, loops: Infinity }); lineDataItem.bullets.push(bullet); } } }
        const eventSource = new EventSource("/api/stream-updates");
        eventSource.onmessage = function(event) { const receivedData = JSON.parse(event.data); updateMap(receivedData.geopolitical); if(receivedData.updated_country_code) { const countryCode = receivedData.updated_country_code; const polygon = polygonSeries.getPolygonById(countryCode); if (polygon) { const countryName = polygon.dataItem.dataContext.name; flashCountry(countryCode); showNotification('G.I.M.P.S. Intel Update', `New headlines detected for ${countryName}.`); } } };
        eventSource.onerror = function(err) { console.error("EventSource failed:", err); };
        polygonSeries.mapPolygons.template.events.on("click", function(ev) { if (ev.target.dataItem.get("value") != null) { fetchCountryData(ev.target.dataItem.dataContext.id, ev.target.dataItem.dataContext.name); } });
        chart.chartContainer.get("background").events.on("click", () => closeInfoPanel());
    });
    closeButton.addEventListener('click', closeInfoPanel);
    function showInfoPanel() { infopanel.classList.add('visible'); }
    function closeInfoPanel() { infoPanel.classList.remove('visible'); }
    
    async function fetchCountryData(countryCode, countryName) {
        infoContent.innerHTML = `<p class="placeholder">ESTABLISHING DATALINK: ${countryName.toUpperCase()}... AI ANALYSIS IN PROGRESS...</p>`; panelTitle.innerText = countryName.toUpperCase(); exportPdfBtn.style.display = 'none';
        showInfoPanel();
        try {
            const response = await fetch(`/api/country-data/${countryCode}`); if (!response.ok) throw new Error('CONNECTION FAILED.'); const data = await response.json(); displayCountryData(data, countryName);
        } catch (error) { infoContent.innerHTML = `<p class="placeholder" style="color: var(--red-alert);">${error.message}</p>`; }
    }
    
    function displayCountryData(data, countryName) {
        const ratesHtml = data.rates && data.rates.length > 0 ? createTable(data.rates, ['Rate', 'Change', 'Date', 'Rate Name']) : '<p>// NO MONETARY RATE DATA //</p>';
        const decisionsHtml = data.decisions && data.decisions.length > 0 ? createTable(data.decisions, ['Date', 'Description', 'Action']) : '<p>// NO RECENT DIRECTIVE DATA //</p>';
        const meetingsHtml = data.meetings && data.meetings.length > 0 ? createTable(data.meetings, ['Date', 'Description']) : '<p>// NO UPCOMING TRANSMISSIONS DATA //</p>';
        let newsHtml = '<p>// NO INTEL FEED //</p>'; if(data.news && data.news.length > 0) { newsHtml = '<ul id="news-list">'; data.news.slice(0, 15).forEach(headline => newsHtml += `<li>${headline}</li>`); newsHtml += '</ul>'; }
        panelTitle.innerText = countryName.toUpperCase(); panelTitle.classList.add('glitch'); setTimeout(() => { panelTitle.classList.remove('glitch'); }, 2000);
        
        infoContent.innerHTML = `
            <div class="info-grid">
                <div class="data-container">
                    <h3>Raw Data Feed</h3>
                    <h4>Monetary Rates</h4>${ratesHtml}
                    <h4>Recent Directives</h4>${decisionsHtml}
                    <h4>Upcoming Transmissions</h4>${meetingsHtml}
                    <h4>Recent Intel</h4>${newsHtml}
                </div>
                <div class="analysis-container">
                    <h3>AI Strategic Analysis</h3>
                    <div class="analysis-content">
                        <h4>${data.analysis.title || 'ANALYSIS PENDING'}</h4>
                        <p><strong>Summary:</strong><br>${data.analysis.summary || 'N/A'}</p>
                        <p><strong>Impact Analysis:</strong><br>${data.analysis.impact_analysis || 'N/A'}</p>
                        <p><strong>Futures Outlook:</strong><br>${data.analysis.futures_outlook || 'N/A'}</p>
                    </div>
                </div>
            </div>`;
            
        exportPdfBtn.style.display = 'block';
        exportPdfBtn.onclick = () => exportDataToPdf(data, countryName);
    }
    
    function exportDataToPdf(data, countryName) {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        const currentDate = new Date().toISOString().slice(0, 10);
        let y = 15;
        const leftMargin = 15;
        const rightMargin = 195;
        
        doc.setFont("courier", "bold");
        doc.setFontSize(14);
        doc.text("G.I.M.P.S. STRATEGIC BRIEFING", doc.internal.pageSize.getWidth() / 2, y, { align: "center" });
        y += 5;
        doc.setLineWidth(0.5);
        doc.line(leftMargin, y, rightMargin, y);
        y += 10;
        
        doc.setFont("courier", "normal");
        doc.setFontSize(12);
        doc.text(`SUBJECT: ${countryName.toUpperCase()}`, leftMargin, y);
        y += 7;
        doc.text(`DATE: ${currentDate}`, leftMargin, y);
        y += 10;
        doc.line(leftMargin, y, rightMargin, y);
        y += 10;
        
        const addSection = (title, content) => {
            if (y > 260) { doc.addPage(); y = 20; }
            doc.setFont("courier", "bold");
            doc.setFontSize(11);
            doc.text(title, leftMargin, y);
            y += 6;
            doc.setFont("courier", "normal");
            doc.setFontSize(10);
            
            const splitContent = doc.splitTextToSize(content.replace(/\\n/g, '\\n'), rightMargin - leftMargin);
            doc.text(splitContent, leftMargin, y);
            y += (splitContent.length * 4) + 8;
        };
        
        doc.setFont("courier", "bold");
        doc.setFontSize(12);
        doc.text("I. RAW DATA FEED", leftMargin, y);
        y += 8;

        let ratesText = "No data available.";
        if (data.rates && data.rates.length > 0) { ratesText = data.rates.map(r => `- ${r.rate_name}: ${r.rate} (Change: ${r.change || 'N/A'}, Date: ${r.date})`).join('\\n'); }
        addSection("Monetary Rates:", ratesText);
        
        let decisionsText = "No data available.";
        if (data.decisions && data.decisions.length > 0) { decisionsText = data.decisions.map(d => `- [${d.date}] ${d.description}`).join('\\n'); }
        addSection("Recent Directives:", decisionsText);

        let meetingsText = "No data available.";
        if (data.meetings && data.meetings.length > 0) { meetingsText = data.meetings.map(m => `- [${m.date}] ${m.description}`).join('\\n'); }
        addSection("Upcoming Transmissions:", meetingsText);
        
        y+= 5;
        doc.line(leftMargin, y, rightMargin, y);
        y += 10;
        
        doc.setFont("courier", "bold");
        doc.setFontSize(12);
        doc.text("II. STRATEGIC AI ANALYSIS", leftMargin, y);
        y += 8;
        
        addSection(data.analysis.title || 'ANALYSIS PENDING', "");
        y -= 8;
        addSection("Summary:", data.analysis.summary || 'N/A');
        addSection("Impact Analysis:", data.analysis.impact_analysis || 'N/A');
        addSection("Futures Outlook:", data.analysis.futures_outlook || 'N/A');
        
        const pageCount = doc.internal.getNumberOfPages();
        for(let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.text(`Page ${i} of ${pageCount}`, doc.internal.pageSize.getWidth() / 2, 287, { align: 'center' });
            doc.text('//CONFIDENTIAL//GENERATED_BY_GIMPS//', leftMargin, 287);
        }

        doc.save(`GIMPS_Briefing_${countryName.replace(/ /g, "_")}_${currentDate}.pdf`);
    }

    function createTable(data, headers) { let table = '<table><thead><tr>'; headers.forEach(h => table += `<th>${h.toUpperCase()}</th>`); table += '</tr></thead><tbody>'; data.forEach(row => { table += '<tr>'; headers.forEach(h => { table += `<td>${row[h.toLowerCase().replace(/ /g, '_')] || 'N/A'}</td>`; }); table += '</tr>'; }); return table + '</tbody></table>'; }
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
    print(">> MODIFIKASI: Library 'openai' dihapus, diganti 'requests' untuk kompatibilitas Termux.")
    print(f">> Model AI yang digunakan: {OPENROUTER_MODEL_NAME}")
    print(">> PERINGATAN: API Key yang digunakan adalah kunci publik gratis. Performa bisa tidak stabil.")
    print(f">> Notifikasi Ponsel DIKONFIGURASI untuk topik: '{NTFY_TOPIC}'")
    print("===============================================================")
    update_thread = Thread(target=background_update_task, daemon=True)
    update_thread.start()
    print(">> BACKGROUND SCANNER INITIALIZED. SYSTEM IS NOW LIVE.")
    print(">> Open your browser and access the following address:")
    print(">> http://127.0.0.1:5001")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
