import os
import json
import time
import threading
import subprocess
import requests
import base64
from flask import Flask, request, render_template_string, redirect, url_for

# --- Konfigurasi Awal ---
CONFIG_FILE = 'config.json'
config_lock = threading.Lock()

# --- Template HTML untuk Halaman Konfigurasi ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Konfigurasi Uploader</title>
    <style>
        body { font-family: sans-serif; margin: 2em; background-color: #f4f4f4; color: #333; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2 { color: #0056b3; }
        form { display: flex; flex-direction: column; }
        label { margin-top: 10px; font-weight: bold; }
        input[type="text"], input[type="password"], input[type="number"] {
            padding: 10px; margin-top: 5px; border-radius: 4px; border: 1px solid #ddd;
        }
        input[type="submit"] {
            margin-top: 20px; padding: 10px 15px; background-color: #0056b3; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-size: 16px;
        }
        input[type="submit"]:hover { background-color: #004494; }
        .note { font-size: 0.9em; color: #666; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pengaturan Uploader GitHub</h1>
        <p>Atur detail repositori GitHub dan interval unggah file.</p>
        <form method="post">
            <h2>Konfigurasi GitHub</h2>
            <label for="github_user">Username GitHub:</label>
            <input type="text" id="github_user" name="github_user" value="{{ config.github_user }}" required>

            <label for="github_repo">Nama Repositori:</label>
            <input type="text" id="github_repo" name="github_repo" value="{{ config.github_repo }}" required>

            <label for="github_token">Personal Access Token (PAT):</label>
            <input type="password" id="github_token" name="github_token" value="{{ config.github_token }}" required>
            <p class="note">Token tidak akan ditampilkan sepenuhnya setelah disimpan. Masukkan ulang jika ingin mengubah.</p>

            <h2>Konfigurasi Interval (detik)</h2>
            <label for="trades_interval">Interval Unggah `trades.json`:</label>
            <input type="number" id="trades_interval" name="trades_interval" value="{{ config.trades_interval }}" min="1" required>

            <label for="device_interval">Interval Unggah `device.json`:</label>
            <input type="number" id="device_interval" name="device_interval" value="{{ config.device_interval }}" min="1" required>

            <input type="submit" value="Simpan Konfigurasi">
        </form>
    </div>
</body>
</html>
"""

# --- Fungsi untuk mengelola file konfigurasi ---
def load_config():
    """Memuat konfigurasi dari file JSON."""
    with config_lock:
        if not os.path.exists(CONFIG_FILE):
            # Membuat file config default jika tidak ada
            default_config = {
                "github_user": "", "github_repo": "", "github_token": "",
                "trades_interval": 300, "device_interval": 600
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)

def save_config(new_config):
    """Menyimpan konfigurasi ke file JSON."""
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f, indent=4)

# --- Fungsi Inti ---

def get_device_info():
    """Mengambil info baterai dan suhu dari Termux API."""
    try:
        result = subprocess.run(['termux-battery-status'], capture_output=True, text=True, check=True)
        battery_data = json.loads(result.stdout)
        
        device_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "battery_level": battery_data.get("percentage", "N/A"),
            "temperature_celsius": battery_data.get("temperature", "N/A"),
            "status": battery_data.get("status", "N/A")
        }
        return device_info
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Gagal mendapatkan info perangkat: {e}")
        return None

def upload_to_github(file_path, content, config, commit_message):
    """Mengunggah atau memperbarui file ke GitHub menggunakan REST API."""
    github_user = config.get('github_user')
    github_repo = config.get('github_repo')
    token = config.get('github_token')

    if not all([github_user, github_repo, token]):
        print("[WARN] Konfigurasi GitHub belum lengkap. Lewati unggahan.")
        return

    api_url = f"https://api.github.com/repos/{github_user}/{github_repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Cek apakah file sudah ada untuk mendapatkan SHA
    sha = None
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            sha = response.json().get('sha')
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 404: # 404 berarti file belum ada, itu normal
            print(f"[ERROR] Gagal memeriksa file di GitHub: {e}")
            return
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Koneksi ke GitHub gagal: {e}")
        return

    # 2. Siapkan data untuk diunggah
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    data = {
        "message": commit_message,
        "content": encoded_content
    }
    if sha:
        data["sha"] = sha # Tambahkan SHA jika file akan diperbarui

    # 3. Lakukan request PUT untuk membuat/memperbarui file
    try:
        response = requests.put(api_url, headers=headers, json=data)
        response.raise_for_status()
        if response.status_code in [200, 201]:
            print(f"[SUCCESS] Berhasil mengunggah '{file_path}' ke GitHub. Pesan: '{commit_message}'")
        else:
            print(f"[WARN] Respon tidak terduga saat mengunggah: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal mengunggah file '{file_path}': {e}")


# --- Worker Threads untuk Unggahan Berkala ---

def trades_uploader():
    """Thread worker untuk mengunggah trades.json."""
    while True:
        config = load_config()
        interval = config.get('trades_interval', 300)
        
        if os.path.exists("trades.json"):
            try:
                with open("trades.json", "r") as f:
                    content = f.read()
                
                if content.strip(): # Hanya upload jika file tidak kosong
                    commit_message = f"Update trades.json at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    upload_to_github("trades.json", content, config, commit_message)
                else:
                    print("[INFO] `trades.json` kosong, unggahan dilewati.")

            except Exception as e:
                print(f"[ERROR] Gagal membaca `trades.json`: {e}")
        else:
            print("[WARN] File `trades.json` tidak ditemukan. Lewati unggahan.")
            
        time.sleep(interval)


def device_info_uploader():
    """Thread worker untuk mengunggah device.json."""
    while True:
        config = load_config()
        interval = config.get('device_interval', 600)

        info = get_device_info()
        if info:
            # Simpan info ke file lokal `device.json` terlebih dahulu
            try:
                content = json.dumps(info, indent=4)
                with open("device.json", "w") as f:
                    f.write(content)
                
                commit_message = f"Update device info at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                upload_to_github("device.json", content, config, commit_message)
            except Exception as e:
                print(f"[ERROR] Gagal memproses `device.json`: {e}")
        
        time.sleep(interval)


# --- Aplikasi Flask ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    config = load_config()
    if request.method == 'POST':
        # Simpan token saat ini jika input dikosongkan
        new_token = request.form.get('github_token')
        
        config['github_user'] = request.form.get('github_user')
        config['github_repo'] = request.form.get('github_repo')
        if new_token: # Hanya perbarui token jika ada isinya
            config['github_token'] = new_token
        config['trades_interval'] = int(request.form.get('trades_interval'))
        config['device_interval'] = int(request.form.get('device_interval'))
        
        save_config(config)
        print("[INFO] Konfigurasi telah diperbarui.")
        return redirect(url_for('index'))
    
    # Untuk keamanan, jangan kirim token kembali ke form
    display_config = config.copy()
    display_config['github_token'] = "" # Kosongkan field token di halaman web
    
    return render_template_string(HTML_TEMPLATE, config=display_config)


# --- Main Execution ---
if __name__ == "__main__":
    # Inisialisasi file config
    load_config()

    print("="*40)
    print("Memulai Uploader Service untuk Termux")
    print("="*40)

    # Jalankan thread uploader di background
    trades_thread = threading.Thread(target=trades_uploader, daemon=True)
    device_thread = threading.Thread(target=device_info_uploader, daemon=True)
    trades_thread.start()
    device_thread.start()
    print("[INFO] Thread uploader untuk `trades.json` dan `device.json` telah dimulai.")

    # Jalankan server Flask
    print("\n[INFO] Server konfigurasi berjalan.")
    print(f"[INFO] Buka browser Anda dan akses: http://localhost:5001")
    print("[INFO] Gunakan IP perangkat Anda jika mengakses dari komputer di jaringan yang sama.")
    print("="*40)
    app.run(host='0.0.0.0', port=5001)
