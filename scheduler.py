import os
import json
import subprocess
import logging
import time
from threading import Thread

# Coba impor library yang dibutuhkan, berikan pesan jika gagal
try:
    from flask import Flask, request, render_template_string, redirect, url_for
    from apscheduler.schedulers.background import BackgroundScheduler
    from github import Github, UnknownObjectException
except ImportError:
    print("Error: Library yang dibutuhkan belum terinstal.")
    print("Jalankan perintah ini di Termux:")
    print("pip install Flask PyGithub apscheduler")
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
    """Mengunggah atau memperbarui file ke GitHub."""
    config = load_config()
    token = config.get("github_token")
    repo_name = config.get("github_repo_name")
    owner = config.get("github_owner")
    branch = config.get("github_branch", "main")

    if not all([token, repo_name, owner]):
        logging.warning(f"Upload untuk {file_path} dilewati. Konfigurasi GitHub belum lengkap.")
        return

    try:
        g = Github(token)
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        with open(file_path, 'r') as f:
            content = f.read()

        try:
            contents = repo.get_contents(file_path, ref=branch)
            repo.update_file(
                path=contents.path,
                message=commit_message,
                content=content,
                sha=contents.sha,
                branch=branch
            )
            logging.info(f"Berhasil memperbarui file '{file_path}' di GitHub.")
        except UnknownObjectException:
            repo.create_file(
                path=file_path,
                message=commit_message,
                content=content,
                branch=branch
            )
            logging.info(f"Berhasil membuat file baru '{file_path}' di GitHub.")

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat mengunggah {file_path} ke GitHub: {e}")


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
scheduler = BackgroundScheduler()

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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Termux Uploader Config</title>
    <style>
        body { font-family: sans-serif; margin: 2em; background: #2e3440; color: #d8dee9; }
        .container { max-width: 600px; margin: auto; background: #3b4252; padding: 20px; border-radius: 8px; }
        h1, h2 { color: #88c0d0; }
        label { display: block; margin-top: 10px; color: #eceff4; }
        input[type="text"], input[type="password"], input[type="number"] { width: 95%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #4c566a; background: #434c5e; color: #d8dee9; }
        .btn { background-color: #5e81ac; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 20px; font-size: 16px; }
        .btn:hover { background-color: #81a1c1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Konfigurasi Uploader Termux</h1>
        <form method="post">
            <h2>Pengaturan GitHub</h2>
            <label for="github_token">Personal Access Token</label>
            <input type="password" name="github_token" value="{{ config.get('github_token', '') }}" required>
            
            <label for="github_owner">Username/Organisasi GitHub</label>
            <input type="text" name="github_owner" value="{{ config.get('github_owner', '') }}" required>

            <label for="github_repo_name">Nama Repositori</label>
            <input type="text" name="github_repo_name" value="{{ config.get('github_repo_name', '') }}" required>
            
            <label for="github_branch">Nama Branch (e.g., main atau master)</label>
            <input type="text" name="github_branch" value="{{ config.get('github_branch', 'main') }}" required>

            <h2>Pengaturan Interval Upload</h2>
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
        # Ambil token dari form, jika kosong, gunakan token lama yang tersimpan
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
    if not scheduler.get_jobs():
        logging.warning("Tidak ada jadwal yang aktif. Periksa konfigurasi Anda.")
    
    scheduler.start()
    logging.info("Scheduler dimulai.")
    
    logging.info("Server Flask berjalan di http://0.0.0.0:5001")
    logging.info("Untuk konfigurasi, buka browser di HP/PC yang satu jaringan dan akses alamat IP perangkat ini, contoh: http://192.168.1.5:5001")
    
    # Jalankan Flask app dalam thread terpisah agar tidak memblokir
    # Ini adalah alternatif jika app.run() memblokir fungsi lain.
    # Namun, untuk kasus ini, BackgroundScheduler sudah berjalan di thread terpisah.
    app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Menghentikan scheduler...")
        scheduler.shutdown()
        logging.info("Skrip dihentikan.")
