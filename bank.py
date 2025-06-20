import time
from flask import Flask, jsonify, render_template_string
import cloudscraper
from bs4 import BeautifulSoup
from threading import Lock

# =================================================================
# BAGIAN 1 & 2: BACKEND & PARSING (TIDAK ADA PERUBAHAN)
# Logika Python di bawah ini sama persis dengan versi sebelumnya.
# =================================================================
app = Flask(__name__)
scraper = cloudscraper.create_scraper()
app_cache = {}
CACHE_LIFETIME_SECONDS = 600
data_lock = Lock()

COUNTRY_MAP = {
    'Argentina': 'AR', 'Australia': 'AU', 'Brazil': 'BR', 'Canada': 'CA', 'Chile': 'CL', 
    'China': 'CN', 'Colombia': 'CO', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Eurozone': 'EU', 
    'Hungary': 'HU', 'Iceland': 'IS', 'India': 'IN', 'Indonesia': 'ID', 'Israel': 'IL', 
    'Japan': 'JP', 'Mexico': 'MX', 'New Zealand': 'NZ', 'Norway': 'NO', 'Poland': 'PL', 
    'Russia': 'RU', 'Saudi Arabia': 'SA', 'South Korea': 'KR', 'South Africa': 'ZA', 
    'Sweden': 'SE', 'Switzerland': 'CH', 'Türkiye': 'TR', 'United Kingdom': 'GB', 'USA': 'US',
    'Egypt': 'EG', 'Pakistan': 'PK', 'Kenya': 'KE', 'Ukraine': 'UA', 'Vietnam': 'VN', 
    'Thailand': 'TH', 'Philippines': 'PH', 'Malaysia': 'MY', 'Peru': 'PE', 'Nigeria': 'NG', 
    'Ghana': 'GH', 'Morocco': 'MA', 'Serbia': 'RS'
}

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
            data.append({
                "country_name": country_name.upper(), "country_code": COUNTRY_MAP.get(country_name, None),
                "rate": cols[1].get_text(strip=True), "change": cols[2].get_text(strip=True),
                "date": cols[5].get_text(strip=True), "rate_name": cols[4].get_text(strip=True).replace(country_name_raw, '').lstrip('| ').strip().upper()
            })
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

def get_all_data():
    with data_lock:
        if app_cache and (time.time() - app_cache.get('timestamp', 0)) < CACHE_LIFETIME_SECONDS: return app_cache['data']
        print(">> RE-CALIBRATING INTEL CACHE...")
        urls = {'rates': 'https://www.cbrates.com/', 'decisions': 'https://www.cbrates.com/decisions.htm', 'meetings': 'https://www.cbrates.com/meetings.htm'}
        all_data = {}
        try:
            for key, url in urls.items():
                response = scraper.get(url, timeout=20); response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                if key == 'rates': all_data['rates'] = parse_world_rates(soup)
                elif key == 'decisions': all_data['decisions'] = parse_decisions_or_meetings(soup, 'decisions')
                elif key == 'meetings': all_data['meetings'] = parse_decisions_or_meetings(soup, 'meetings')
            app_cache['data'], app_cache['timestamp'] = all_data, time.time()
            print(">> INTEL CACHE UPDATED.")
            return all_data
        except Exception as e:
            print(f">> CRITICAL ERROR: FAILED TO RETRIEVE INTEL: {e}"); return None

@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/country-data/<country_code>')
def get_country_data(country_code):
    all_data = get_all_data()
    if all_data is None: return jsonify({"error": "FAILED TO CONNECT TO DATA SOURCE."}), 500
    is_eurozone = country_code.upper() == 'EU'
    def filter_by_code(data_list):
        if is_eurozone: return [d for d in data_list if d.get('country_name') == 'EUROZONE']
        return [d for d in data_list if d.get('country_code') == country_code.upper()]
    decisions_data = filter_by_code(all_data.get('decisions', []))
    return jsonify({
        "rates": filter_by_code(all_data.get('rates', [])), "decisions": decisions_data,
        "meetings": filter_by_code(all_data.get('meetings', [])),
        "analysis": generate_analysis(country_code.upper(), decisions_data)
    })

# =======================================================
# BAGIAN 3: FRONTEND HTML DENGAN TAMPILAN STEALTH GLITCH
# =======================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G.I.M.P.S - Global Intelligence Monetary Policy System</title>
    <style>
        :root {
            --main-bg: #000000;
            --panel-bg: rgba(10, 10, 10, 0.9);
            --border-color: rgba(255, 255, 255, 0.2);
            --accent-color: #FFFFFF;
            --text-color: #dcdcdc;
            --scrollbar-thumb: #555;
            --scrollbar-track: #111;
        }
        @keyframes scanlines { 0% { background-position: 0 0; } 100% { background-position: 0 50px; } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes typing { from { width: 0; } to { width: 100%; } }
        @keyframes blink-caret { from, to { border-color: transparent; } 50% { border-color: var(--accent-color); } }
        @keyframes border-flicker {
            0%, 100% { border-top-color: rgba(255,255,255,0.2); }
            50% { border-top-color: rgba(255,255,255,0.5); }
        }
        @keyframes glitch-appear {
            0% { clip-path: inset(20% 0 70% 0); } 20% { clip-path: inset(80% 0 10% 0); }
            40% { clip-path: inset(40% 0 40% 0); } 60% { clip-path: inset(90% 0 5% 0); }
            80% { clip-path: inset(10% 0 85% 0); } 100% { clip-path: inset(0 0 0 0); }
        }
        @keyframes glitch-text-anim {
            0% { text-shadow: 1px 0 0 #fff, -1px 0 0 #fff; clip-path: inset(10% 0 80% 0); }
            20% { clip-path: inset(40% 0 40% 0); } 40% { clip-path: inset(80% 0 10% 0); }
            60% { clip-path: inset(20% 0 70% 0); } 80% { clip-path: inset(60% 0 30% 0); }
            100% { clip-path: inset(10% 0 80% 0); }
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
            position: fixed; bottom: 0; left: 0; width: 100%; height: 45vh;
            background-color: var(--panel-bg);
            border-top: 1px solid var(--border-color); box-shadow: 0 -2px 10px rgba(0,0,0,0.5);
            visibility: hidden; opacity: 0;
            z-index: 1000; display: flex; flex-direction: column;
            backdrop-filter: blur(5px);
        }
        #infopanel.visible { visibility: visible; opacity: 1; animation: glitch-appear 0.2s steps(8, end) forwards; }
        
        #panel-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 20px; background-color: rgba(0,0,0,0.3); border-bottom: 1px solid var(--border-color);
            animation: border-flicker 2s linear infinite;
        }
        .glitch { position: relative; }
        .glitch::before, .glitch::after {
            content: attr(data-text); position: absolute; top: 0; left: 0; width: 100%;
            height: 100%; background: var(--panel-bg); overflow: hidden;
        }
        .glitch::before { left: 2px; text-shadow: -1px 0 #FFFFFF; animation: glitch-text-anim 1.5s infinite linear alternate-reverse; }
        .glitch::after { left: -2px; text-shadow: -1px 0 #FFFFFF; animation: glitch-text-anim 2s infinite linear alternate-reverse; }

        #close-panel {
            background: none; border: none; color: var(--accent-color); font-size: 28px;
            cursor: pointer; line-height: 1; transition: transform 0.2s;
        }
        #close-panel:hover { transform: scale(1.5); }
        
        #info-content { padding: 20px; overflow-y: auto; flex-grow: 1; animation: fadeIn 1s; }
        .info-container { display: flex; gap: 25px; }
        .data-column, .analysis-column { flex: 1; }
        .analysis-column { border-left: 1px solid var(--border-color); padding-left: 25px; }
        h2, h3 { color: var(--accent-color); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-top: 0; }
        h3 { font-size: 1.1em; border: none; padding-bottom: 5px; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 0.9em; }
        th, td { border: 1px solid var(--border-color); padding: 8px; text-align: left; }
        th { background-color: rgba(255, 255, 255, 0.05); }
        .placeholder {
            color: var(--accent-color); text-align: center; padding-top: 40px; font-size: 1.2em;
            overflow: hidden; white-space: nowrap; margin: 0 auto; letter-spacing: .15em;
            animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite;
        }
        
        /* --- PERUBAHAN SCROLLBAR --- */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--scrollbar-track); }
        ::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #888; }

        @media (max-width: 768px) {
            #infopanel { height: 75vh; }
            .info-container { flex-direction: column; }
            .analysis-column { border-left: none; padding-left: 0; border-top: 1px solid var(--border-color); padding-top: 20px; margin-top: 20px; }
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
    const chartDiv = document.getElementById('chartdiv');

    am5.ready(function() {
        var root = am5.Root.new("chartdiv");
        root.setThemes([am5themes_Animated.new(root)]);
        var chart = root.container.children.push(am5map.MapChart.new(root, { panX: "pan", panY: "pan", projection: am5map.geoEquirectangular() }));
        
        var polygonSeries = chart.series.push(am5map.MapPolygonSeries.new(root, {
            geoJSON: am5geodata_worldLow, exclude: ["AQ"],
            fill: am5.color(0x0a0a0a), stroke: am5.color(0xffffff), strokeOpacity: 0.15
        }));

        polygonSeries.mapPolygons.template.setAll({
            tooltipText: "{name}", interactive: true, strokeWidth: 0.5,
            // --- PERUBAHAN HOVER SMOOTH ---
            transitionDuration: 300 // Durasi transisi 300ms
        });

        // --- PERUBAHAN HOVER STATE ---
        polygonSeries.mapPolygons.template.states.create("hover", {
            fill: am5.color(0x666666), // Warna abu-abu saat hover
            fillOpacity: 0.8 // Sedikit transparan
        });

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
    function closeInfoPanel() {
        infoPanel.classList.remove('visible');
        panelTitle.classList.remove('glitch');
        panelTitle.removeAttribute('data-text');
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
        } catch (error) {
            infoContent.innerHTML = `<p class="placeholder" style="color: #ff4d4d;">${error.message}</p>`;
        }
    }

    function displayCountryData(data, countryName) {
        const ratesHtml = data.rates.length > 0 ? createTable(data.rates, ['Rate', 'Change', 'Date', 'Rate Name']) : '<p>// NO CURRENT RATE DATA //</p>';
        const decisionsHtml = data.decisions.length > 0 ? createTable(data.decisions, ['Date', 'Description', 'Action']) : '<p>// NO RECENT DECISION DATA //</p>';
        const meetingsHtml = data.meetings.length > 0 ? createTable(data.meetings, ['Date', 'Description']) : '<p>// NO UPCOMING MEETINGS DATA //</p>';
        
        panelTitle.innerText = countryName.toUpperCase();
        panelTitle.classList.add('glitch');
        panelTitle.setAttribute('data-text', countryName.toUpperCase());
        setTimeout(() => { panelTitle.classList.remove('glitch'); }, 2500);

        infoContent.innerHTML = `
            <div class="info-container">
                <div class="data-column">
                    <h3>// CURRENT RATE DATA</h3>${ratesHtml}
                    <h3>// RECENT DIRECTIVES</h3>${decisionsHtml}
                    <h3>// UPCOMING TRANSMISSIONS</h3>${meetingsHtml}
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
# BAGIAN 4: MENJALANKAN APLIKASI
# ===============================================
if __name__ == '__main__':
    print("===============================================================")
    print(">> G.I.M.P.S (vStealth UI) :: ONLINE")
    print(">> Menginisialisasi koneksi dan memuat data intelijen awal...")
    get_all_data()
    print(">> Sistem Siap. Menunggu perintah di command interface...")
    print(">> Buka browser Anda dan akses alamat berikut:")
    print(">> http://127.0.0.1:5001")
    print("===============================================================")
    app.run(host='0.0.0.0', port=5001, debug=False)
