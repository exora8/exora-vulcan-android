import os
import json
import base64
import subprocess
import logging
import time

# Coba impor library, berikan pesan jika gagal
try:
    import requests
    from flask import Flask, request, render_template_string, redirect, url_for
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError as e:
    print(f"Error: Library yang dibutuhkan belum terinstal ({e}).")
    print("Jalankan perintah ini di Termux:")
    print("pip install requests flask apscheduler")
    exit()

# --- Konfigurasi Dasar ---
CONFIG_FILE = "config.json"
DEVICE_INFO_FILE = "device.json"
LOG_FILE = "uploader.log"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Fungsi untuk Konfigurasi ---
def load_config():
    """Memuat konfigurasi dari file JSON."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error("Gagal membaca config.json, file rusak.")
        return {}

def save_config(data):
    """Menyimpan konfigurasi ke file JSON."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info("Konfigurasi berhasil disimpan.")

# --- Fungsi untuk Interaksi dengan Termux & GitHub ---
def get_device_info():
    """Mengambil info baterai & suhu dari Termux API dan menyimpannya ke file."""
    try:
        result = subprocess.run(
            ['termux-battery-status'],
            capture_output=True, text=True, check=True
        )
        battery_data = json.loads(result.stdout)
        device_status = {
            "battery_level": battery_data.get("percentage"),
            "temperature_c": battery_data.get("temperature"),
            "status": battery_data.get("status"),
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(DEVICE_INFO_FILE, 'w') as f:
            json.dump(device_status, f, indent=4)
        logging.info(f"Info perangkat diperbarui: Baterai {device_status['battery_level']}%, Suhu {device_status['temperature_c']}°C")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Gagal mengambil info perangkat dari Termux API: {e}. Pastikan 'termux-api' sudah terinstal.")
        return False

def upload_to_github(file_path, commit_message):
    """Mengunggah atau memperbarui file ke GitHub menggunakan requests."""
    config = load_config()
    token = config.get("github_token")
    repo_name = config.get("github_repo_name")
    owner = config.get("github_owner")
    branch = config.get("github_branch", "main")

    if not all([token, owner, repo_name]):
        logging.warning(f"Upload untuk {file_path} dilewati. Konfigurasi GitHub belum lengkap.")
        return

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        with open(file_path, 'rb') as f:
            content_bytes = f.read()
        content_base64 = base64.b64encode(content_bytes).decode('utf-8')

        # 1. Cek apakah file sudah ada untuk mendapatkan SHA
        sha = None
        get_response = requests.get(api_url, headers=headers, params={'ref': branch})
        if get_response.status_code == 200:
            sha = get_response.json().get('sha')
            logging.info(f"File '{file_path}' sudah ada di repo, akan diperbarui.")
        elif get_response.status_code != 404:
            logging.error(f"Gagal memeriksa file di GitHub: {get_response.status_code} - {get_response.text}")
            return

        # 2. Buat payload untuk di-upload
        payload = {
            "message": commit_message,
            "content": content_base64,
            "branch": branch
        }
        if sha:
            payload["sha"] = sha  # Tambahkan SHA jika ini adalah update

        # 3. Upload file (create atau update)
        upload_response = requests.put(api_url, headers=headers, data=json.dumps(payload))

        if upload_response.status_code == 201:
            logging.info(f"Berhasil membuat file baru '{file_path}' di GitHub.")
        elif upload_response.status_code == 200:
            logging.info(f"Berhasil memperbarui file '{file_path}' di GitHub.")
        else:
            logging.error(f"Gagal mengunggah '{file_path}' ke GitHub: {upload_response.status_code} - {upload_response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Kesalahan jaringan saat menghubungi GitHub API: {e}")
    except Exception as e:
        logging.error(f"Terjadi kesalahan tak terduga saat proses upload: {e}")


# --- Fungsi untuk Job Terjadwal ---
def job_upload_trades():
    """Tugas untuk mengunggah trades.json."""
    logging.info("Menjalankan tugas: Upload trades.json")
    if os.path.exists("trades.json"):
        upload_to_github("trades.json", "Update: Data trades otomatis")
    else:
        logging.warning("File trades.json tidak ditemukan, upload dilewati.")

def job_upload_device_info():
    """Tugas untuk mengambil info perangkat dan mengunggahnya."""
    logging.info("Menjalankan tugas: Upload device.json")
    if get_device_info():
        upload_to_github(DEVICE_INFO_FILE, "Update: Status perangkat")

# --- Pengelola Jadwal (Scheduler) ---
scheduler = BackgroundScheduler(timezone="Asia/Jakarta")

def reschedule_jobs():
    """Mengatur ulang semua jadwal berdasarkan konfigurasi saat ini."""
    config = load_config()
    scheduler.remove_all_jobs()
    
    try:
        trades_interval = int(config.get("trades_interval_sec", 3600))
        if trades_interval > 0:
            scheduler.add_job(job_upload_trades, 'interval', seconds=trades_interval, id='trades_job')
            logging.info(f"Jadwal upload 'trades.json' diatur setiap {trades_interval} detik.")
    except (ValueError, TypeError):
        logging.warning("Interval trades.json tidak valid.")

    try:
        device_interval = int(config.get("device_interval_sec", 600))
        if device_interval > 0:
            scheduler.add_job(job_upload_device_info, 'interval', seconds=device_interval, id='device_job')
            logging.info(f"Jadwal upload 'device.json' diatur setiap {device_interval} detik.")
    except (ValueError, TypeError):
        logging.warning("Interval device.json tidak valid.")
        
    logging.info("Daftar pekerjaan aktif: " + str(scheduler.get_jobs()))


# --- Aplikasi Web Flask ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Konfigurasi Uploader Termux</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background: #2e3440; color: #d8dee9; }
        .container { max-width: 700px; margin: auto; background: #3b4252; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        h1, h2 { color: #88c0d0; border-bottom: 2px solid #4c566a; padding-bottom: 10px; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; color: #eceff4; font-weight: bold; }
        input[type="text"], input[type="password"], input[type="number"] { width: calc(100% - 20px); padding: 10px; border-radius: 4px; border: 1px solid #4c566a; background: #434c5e; color: #d8dee9; font-size: 16px; }
        input::placeholder { color: #a3b2cc; }
        .btn { display: block; width: 100%; background-color: #5e81ac; color: white; padding: 12px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 25px; font-size: 18px; font-weight: bold; }
        .btn:hover { background-color: #81a1c1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Konfigurasi Uploader Termux</h1>
        <form method="post">
            <h2>Pengaturan GitHub</h2>
            <label for="github_token">Personal Access Token</label>
            <input type="password" name="github_token" placeholder="Biarkan kosong jika tidak ingin mengubah" >
            
            <label for="github_owner">Username/Organisasi GitHub</label>
            <input type="text" name="github_owner" value="{{ config.get('github_owner', '') }}" required>

            <label for="github_repo_name">Nama Repositori</label>
            <input type="text" name="github_repo_name" value="{{ config.get('github_repo_name', '') }}" required>
            
            <label for="github_branch">Nama Branch (contoh: main)</label>
            <input type="text" name="github_branch" value="{{ config.get('github_branch', 'main') }}" required>

            <h2>Pengaturan Interval</h2>
            <label for="trades_interval_sec">Interval Upload trades.json (detik)</label>
            <input type="number" name="trades_interval_sec" value="{{ config.get('trades_interval_sec', 3600) }}" required>
            
            <label for="device_interval_sec">Interval Upload device.json (detik)</label>
            <input type="number" name="device_interval_sec" value="{{ config.get('device_interval_sec', 600) }}" required>

            <button type="submit" class="btn">Simpan & Terapkan Konfigurasi</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        current_config = load_config()
        # Ambil token dari form; jika kosong, gunakan token lama yang tersimpan
        new_token = request.form.get('github_token')
        
        new_config = {
            "github_token": new_token if new_token else current_config.get('github_token'),
            "github_owner": request.form.get('github_owner'),
            "github_repo_name": request.form.get('github_repo_name'),
            "github_branch": request.form.get('github_branch'),
            "trades_interval_sec": int(request.form.get('trades_interval_sec')),
            "device_interval_sec": int(request.form.get('device_interval_sec'))
        }
        save_config(new_config)
        reschedule_jobs()
        return redirect(url_for('index'))

    current_config = load_config()
    return render_template_string(HTML_TEMPLATE, config=current_config)

# --- Fungsi Utama ---
def main():
    if not os.path.exists("trades.json"):
        logging.info("Membuat file trades.json kosong karena tidak ditemukan.")
        with open("trades.json", "w") as f:
            json.dump({"message": "File ini dibuat secara otomatis"}, f)
            
    reschedule_jobs()
    
    scheduler.start()
    logging.info("Scheduler dimulai.")
    
    logging.info("Server Flask berjalan di http://0.0.0.0:5001")
    logging.info("Untuk konfigurasi, buka browser di HP/PC yang satu jaringan dan akses alamat IP perangkat ini.")
    logging.info("Contoh: http://192.168.1.5:5001 (cek IP Anda dengan perintah 'ifconfig')")
    
    app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Menghentikan scheduler...")
        scheduler.shutdown()
        logging.info("Skrip dihentikan.")
