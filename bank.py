import time
from flask import Flask, jsonify, render_template_string
import cloudscraper
from bs4 import BeautifulSoup
from threading import Lock
import feedparser # Library baru untuk RSS

# =================================================================
# BAGIAN 1: KONFIGURASI (TANPA API KEY)
# =================================================================
app = Flask(__name__)
scraper = cloudscraper.create_scraper()
app_cache = {}
CACHE_LIFETIME_SECONDS = 1800 
data_lock = Lock()

# =================================================================
# BAGIAN 2: LOGIKA BACKEND DENGAN RSS FEED
# =================================================================

FULL_COUNTRY_MAP = {
    'Argentina': 'AR', 'Australia': 'AU', 'Austria': 'AT', 'Belgium': 'BE', 'Brazil': 'BR', 'Bulgaria': 'BG',
    'Canada': 'CA', 'China': 'CN', 'Colombia': 'CO', 'Cuba': 'CU', 'Czech Republic': 'CZ', 'Denmark': 'DK',
    'Egypt': 'EG', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR', 'Hong Kong': 'HK', 'Hungary': 'HU',
    'India': 'IN', 'Indonesia': 'ID', 'Ireland': 'IE', 'Israel': 'IL', 'Italy': 'IT', 'Japan': 'JP',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Malaysia': 'MY', 'Mexico': 'MX', 'Morocco': 'MA', 'Netherlands': 'NL',
    'New Zealand': 'NZ', 'Nigeria': 'NG', 'Norway': 'NO', 'Philippines': 'PH', 'Poland': 'PL',
    'Portugal': 'PT', 'Romania': 'RO', 'Russia': 'RU', 'Saudi Arabia': 'SA', 'Serbia': 'RS',
    'Singapore': 'SG', 'Slovakia': 'SK', 'Slovenia': 'SI', 'South Africa': 'ZA', 'South Korea': 'KR',
    'Sweden': 'SE', 'Switzerland': 'CH', 'Taiwan': 'TW', 'Thailand': 'TH', 'Türkiye': 'TR', 'Turkey': 'TR',
    'UAE': 'AE', 'Ukraine': 'UA', 'United Kingdom': 'GB', 'USA': 'US', 'Venezuela': 'VE', 'United States': 'US'
}
REVERSE_COUNTRY_MAP = {v: k for k, v in FULL_COUNTRY_MAP.items()}
COUNTRY_MAP = FULL_COUNTRY_MAP 
ACTIVE_CONFLICTS = [ ('RU', 'UA'), ('IL', 'PS') ] 
TENSION_KEYWORDS = { 'war': 15, 'conflict': 10, 'sanction': 8, 'protest': 5, 'crisis': 7, 'attack': 12, 'dispute': 6, 'tension': 8, 'unrest': 5, 'mobilization': 10 }

# --- Fungsi-fungsi parsing moneter (tidak berubah) ---
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

def generate_analysis(country_code, decision_data): # Teks diubah jadi kapital
    if not decision_data: return {"title": "ANALYSIS: NO DATA", "summary": "NO RECENT DIRECTIVES FOUND. TERMINATING DATALINK.", "impact": "IMPACT CANNOT BE DETERMINED."}
    latest_decision = decision_data[0]
    action = latest_decision.get("action", "UNKNOWN").lower()
    country_name = latest_decision.get("country_name", "Negara")
    major_economies, emerging_markets = ['US', 'EU', 'JP', 'GB', 'CN'], ['ID', 'BR', 'IN', 'RU', 'TR', 'ZA', 'MX']
    title = f"ANALYSIS :: {action.upper()} EVENT :: {country_name.upper()}"
    summary = f"SYSTEM DETECTED MONETARY POLICY: **{action.upper()}**. RAW DATA: {latest_decision['description']}."
    impact_paragraphs = []
    if action == 'cut': impact_paragraphs.append(">> GENERAL PROTOCOL: RATE CUT AIMS TO STIMULATE ECONOMY, REDUCE LENDING COSTS. POTENTIAL DOMESTIC CURRENCY WEAKNESS.")
    elif action == 'hike': impact_paragraphs.append(">> GENERAL PROTOCOL: RATE HIKE AIMS TO COMBAT INFLATION, COOLING ECONOMIC ACTIVITY. POTENTIAL DOMESTIC CURRENCY STRENGTH.")
    else: impact_paragraphs.append(">> GENERAL PROTOCOL: RATE UNCHANGED INDICATES 'WAIT-AND-SEE' STANCE. MONITORING CONDITIONS.")
    crypto_impact_text = ""
    if country_code in major_economies:
        if action == 'cut': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BEARISH]: RATE CUT BY {country_name.upper()} SUGGESTS REDUCED GLOBAL RISK APPETITE. CAPITAL MAY EXIT SPECULATIVE ASSETS (CRYPTO) FOR 'SAFE-HAVENS'. INCREASED SELL PRESSURE EXPECTED."
        elif action == 'hike': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BULLISH]: TRADITIONAL MARKETS STRESSED BY {country_name.upper()} HIKE. INVESTORS MAY SEEK ALTERNATIVE ASSETS/HEDGES LIKE CRYPTO, DRIVING DEMAND."
    elif country_code in emerging_markets:
        if action == 'cut': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BULLISH]: SUKU BUNGA RENDAH DI {country_name.upper()} MENDORONG INVESTOR LOKAL MENCARI IMBAL HASIL LEBIH TINGGI. CRYPTO MENJADI ALTERNATIF MENARIK, MEMICU ALIRAN MODAL DOMESTIK."
        elif action == 'hike': crypto_impact_text = f">> CRYPTO SENTIMENT ANALYSIS [TENDENCY: BEARISH]: IMBAL HASIL INSTRUMEN PERBANKAN LOKAL MENJADI SANGAT MENARIK. TERJADI PERPINDAHAN MODAL DARI ASET BERISIKO (CRYPTO) KE ASET AMAN, MENGURANGI PERMINTAAN KRIPTO."
    if crypto_impact_text: impact_paragraphs.append(crypto_impact_text)
    impact_paragraphs.append("<small><i>DISCLAIMER: ANALYSIS IS A PROTOTYPE MODEL. FINANCIAL MARKETS ARE COMPLEX. CONDUCT INDEPENDENT RESEARCH. END TRANSMISSION.</i></small>")
    return {"title": title, "summary": summary, "impact": "<br><br>".join(impact_paragraphs)}

# --- Fungsi Berita Diubah Menggunakan RSS Feed ---
def get_news_and_tension_from_rss(country_code):
    try:
        # URL RSS Google News berdasarkan kode negara
        url = f"https://news.google.com/rss?gl={country_code.upper()}&hl=en-US&ceid={country_code.upper()}:en"
        feed = feedparser.parse(url)
        
        if feed.bozo: # feedparser menandai error dengan flag 'bozo'
            raise Exception(feed.bozo_exception)

        headlines = [entry.title.upper() for entry in feed.entries]
        score = 0
        for headline in headlines:
            for keyword, weight in TENSION_KEYWORDS.items():
                if keyword.upper() in headline:
                    score += weight
                    
        return {"headlines": headlines, "score": min(score, 100)}
    except Exception as e:
        print(f"Error fetching RSS for {country_code}: {e}")
        return {"headlines": [f"ERROR FETCHING RSS FEED FOR {country_code.upper()}"], "score": 0}

# --- Fungsi utama yang diperluas ---
def get_all_data():
    with data_lock:
        if app_cache and (time.time() - app_cache.get('timestamp', 0)) < CACHE_LIFETIME_SECONDS:
            return app_cache.get('data')

        print(">> RE-CALIBRATING GLOBAL INTEL CACHE... (THIS MAY TAKE A FEW MINUTES)")
        all_data = {}
        cbrates_urls = {'rates': 'https://www.cbrates.com/', 'decisions': 'https://www.cbrates.com/decisions.htm', 'meetings': 'https://www.cbrates.com/meetings.htm'}
        for key, url in cbrates_urls.items():
            try:
                response = scraper.get(url, timeout=20); response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                if key == 'rates': all_data['rates'] = parse_world_rates(soup)
                elif key == 'decisions': all_data['decisions'] = parse_decisions_or_meetings(soup, 'decisions')
                elif key == 'meetings': all_data['meetings'] = parse_decisions_or_meetings(soup, 'meetings')
            except Exception as e:
                print(f"Failed to fetch monetary data for {key}: {e}"); all_data[key] = []
        
        all_data['geopolitical'] = {"tension_scores": [], "conflicts": ACTIVE_CONFLICTS, "news": {}, "news_links": []}
        news_links_set = set()
        
        country_codes_to_scan = list(REVERSE_COUNTRY_MAP.keys())
        
        for i, source_code in enumerate(country_codes_to_scan):
            print(f">> SCANNING: {REVERSE_COUNTRY_MAP.get(source_code, source_code)} ({i+1}/{len(country_codes_to_scan)})")
            # Gunakan fungsi RSS baru
            news_data = get_news_and_tension_from_rss(source_code)
            all_data['geopolitical']['tension_scores'].append({"id": source_code, "value": news_data["score"]})
            all_data['geopolitical']['news'][source_code] = news_data["headlines"]
            
            for headline in news_data["headlines"]:
                for target_code, target_name in REVERSE_COUNTRY_MAP.items():
                    if source_code == target_code: continue
                    if target_name.upper() in headline:
                        link = tuple(sorted((source_code, target_code)))
                        news_links_set.add(link)
            # Jeda sopan agar tidak membanjiri server Google News
            time.sleep(0.2)

        all_data['geopolitical']['news_links'] = [list(link) for link in news_links_set]

        app_cache['data'] = all_data
        app_cache['timestamp'] = time.time()
        print(">> INTEL CACHE UPDATED.")
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
    return jsonify(all_data['geopolitical'])

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
        "news": all_data.get('geopolitical', {}).get('news', {}).get(country_code_upper, []),
        "analysis": generate_analysis(country_code_upper, decisions_data)
    })

# =================================================================
# BAGIAN 4: FRONTEND HTML (Tidak Berubah)
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
        @keyframes glitch-text {
            2%,64%{ transform: translate(2px,0) skew(0deg); } 4%,60%{ transform: translate(-2px,0) skew(0deg); }
            62%{ transform: translate(0,0) skew(5deg); }
        }
        @keyframes glitch-appear {
            0% { clip-path: inset(20% 0 70% 0); } 20% { clip-path: inset(80% 0 10% 0); }
            40% { clip-path: inset(40% 0 40% 0); } 60% { clip-path: inset(90% 0 5% 0); }
            80% { clip-path: inset(10% 0 85% 0); } 100% { clip-path: inset(0 0 0 0); }
        }

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
        ul#news-list { list-style-type: '>> '; padding-left: 20px; margin: 0; font-size: 0.85em; height: 100%; overflow-y: auto; }
        ul#news-list li { margin-bottom: 8px; }
        .placeholder { color: var(--accent-color); text-align: center; padding-top: 40px; font-size: 1.2em; overflow: hidden; white-space: nowrap; margin: 0 auto; letter-spacing: .15em; animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite; }
        ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #555; } ::-webkit-scrollbar-thumb:hover { background: #888; }
        @media (max-width: 768px) {
            #infopanel { height: 75vh; } .info-container { flex-direction: column; }
            .analysis-column, .news-column { border-left: none; padding-left: 0; border-top: 1px solid var(--border-color); padding-top: 20px; margin-top: 20px; }
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
    const infoPanel = document.getElementById('infopanel');
    const infoContent = document.getElementById('info-content');
    const panelTitle = document.getElementById('panel-title');
    const closeButton = document.getElementById('close-panel');

    am5.ready(function() {
        var root = am5.Root.new("chartdiv");
        root.setThemes([am5themes_Animated.new(root)]);
        var chart = root.container.children.push(am5map.MapChart.new(root, { panX: "pan", panY: "pan", projection: am5map.geoEquirectangular() }));
        
        var polygonSeries = chart.series.push(am5map.MapPolygonSeries.new(root, {
            geoJSON: am5geodata_worldLow, exclude: ["AQ"], valueField: "value", calculateAggregates: true,
            fill: am5.color(0x1a1a1a), stroke: am5.color(0xffffff), strokeOpacity: 0.15
        }));
        polygonSeries.mapPolygons.template.setAll({ tooltipText: "{name}: TENSION SCORE {value}", interactive: true, strokeWidth: 0.5, transitionDuration: 300 });
        polygonSeries.mapPolygons.template.states.create("hover", { fill: am5.color(0x666666) });
        polygonSeries.set("heatRules", [{ target: polygonSeries.mapPolygons.template, dataField: "value", min: am5.color(0x333333), max: am5.color(0xff0000), key: "fill", logarithmic: true }]);
        
        var conflictLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        conflictLineSeries.mapLines.template.setAll({ stroke: am5.color(0xff0000), strokeOpacity: 0.6, strokeWidth: 2, strokeDasharray: [4,2] });
        var newsLineSeries = chart.series.push(am5map.MapLineSeries.new(root, {}));
        newsLineSeries.mapLines.template.setAll({ stroke: am5.color(0x64b5f6), strokeOpacity: 0.7, strokeWidth: 1.5, arc: -0.2, strokeDasharray: [2, 2] });

        fetch('/api/global-status').then(res => res.json()).then(data => {
            if (!data) return;
            polygonSeries.data.setAll(data.tension_scores);
            (data.conflicts || []).forEach(conflict => renderLine(conflict[0], conflict[1], conflictLineSeries, true));
            (data.news_links || []).forEach(link => renderLine(link[0], link[1], newsLineSeries, false));
        });

        function renderLine(code1, code2, series, isConflict) {
            let p1 = polygonSeries.getPolygonById(code1); let p2 = polygonSeries.getPolygonById(code2);
            if(p1 && p2) {
                let line = series.pushDataItem({ geometry: { type: "LineString", coordinates: [[p1.visualCentroid.longitude, p1.visualCentroid.latitude], [p2.visualCentroid.longitude, p2.visualCentroid.latitude]] } });
                let bullet = am5.Bullet.new(root, { sprite: am5.Circle.new(root, { radius: isConflict ? 3 : 2, fill: isConflict ? am5.color(0xff0000) : am5.color(0xffffff) }) });
                bullet.animate({ key: "location", from: 0, to: 1, duration: isConflict ? 2000 : 4000, loops: Infinity });
                line.bullets.push(bullet);
            }
        }

        polygonSeries.mapPolygons.template.events.on("click", function(ev) {
            const countryId = ev.target.dataItem.dataContext.id;
            const countryName = ev.target.dataItem.dataContext.name;
            fetchCountryData(countryId, countryName);
        });
        
        infoPanel.addEventListener('pointerdown', e => e.stopPropagation());
        chart.chartContainer.get("background").events.on("click", () => closeInfoPanel());
    });
    
    closeButton.addEventListener('click', closeInfoPanel);
    function showInfoPanel() { infoPanel.classList.add('visible'); }
    function closeInfoPanel() { infoPanel.classList.remove('visible'); panelTitle.classList.remove('glitch'); }

    async function fetchCountryData(countryCode, countryName) {
        infoContent.innerHTML = `<p class="placeholder">ESTABLISHING DATALINK: ${countryName.toUpperCase()}...</p>`;
        panelTitle.innerText = countryName.toUpperCase();
        showInfoPanel();
        try {
            const response = await fetch(`/api/country-data/${countryCode}`);
            if (!response.ok) throw new Error('CONNECTION FAILED. DATA STREAM CORRUPTED.');
            const data = await response.json();
            displayCountryData(data, countryName);
        } catch (error) {
            infoContent.innerHTML = `<p class="placeholder" style="color: var(--red-alert);">${error.message}</p>`;
        }
    }

    function displayCountryData(data, countryName) {
        const ratesHtml = data.rates.length > 0 ? createTable(data.rates, ['Rate', 'Change', 'Date', 'Rate Name']) : '<p>// NO MONETARY RATE DATA //</p>';
        const decisionsHtml = data.decisions.length > 0 ? createTable(data.decisions, ['Date', 'Description', 'Action']) : '<p>// NO RECENT DIRECTIVE DATA //</p>';
        const meetingsHtml = data.meetings.length > 0 ? createTable(data.meetings, ['Date', 'Description']) : '<p>// NO UPCOMING TRANSMISSIONS DATA //</p>';
        let newsHtml = '<p>// NO INTEL FEED //</p>';
        if(data.news && data.news.length > 0) {
            newsHtml = '<ul id="news-list">';
            data.news.slice(0, 10).forEach(headline => newsHtml += `<li>${headline}</li>`);
            newsHtml += '</ul>';
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
                <div class="news-column">
                    <h3>// RECENT INTEL</h3>${newsHtml}
                </div>
                <div class="analysis-column">
                    <h3>${data.analysis.title}</h3>
                    <p><b>SUMMARY:</b> ${data.analysis.summary.replace(/\*\*(.*?)\*\*/g, '<span>$1</span>')}</p>
                    <br>
                    <p><b>IMPACT ANALYSIS:</b><br>${data.analysis.impact.replace(/\*\*(.*?)\*\*/g, '<span>$1</span>')}</p>
                </div>
            </div>`;
    }

    function createTable(data, headers) {
        let table = '<table><thead><tr>';
        headers.forEach(h => table += `<th>${h.toUpperCase()}</th>`);
        table += '</tr></thead><tbody>';
        data.forEach(row => {
            table += '<tr>';
            headers.forEach(h => {
                const key = h.toLowerCase().replace(/ /g, '_');
                table += `<td>${row[key] || 'N/A'}</td>`;
            });
            table += '</tr>';
        });
        table += '</tbody></table>';
        return table;
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
    print(">> G.I.M.P.S (vIntel-Link) :: BOOTING...")
    print(">> Menginisialisasi koneksi dan memuat data intelijen global...")
    get_all_data()
    print(">> SISTEM ONLINE. Menunggu perintah di command interface...")
    print(">> Buka browser Anda dan akses alamat berikut:")
    print(">> http://127.0.0.1:5001")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
