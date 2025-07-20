import requests
import time
import os
import json

# --- KONFIGURASI ---
URL = "https://raw.githubusercontent.com/exora8/exora-vulcan-android/refs/heads/main/device.json"
CHECK_INTERVAL_SECONDS = 60  # Periksa setiap 60 detik (1 menit)
BATTERY_THRESHOLD_LOW = 55   # Batas bawah level baterai untuk notifikasi
TEMP_THRESHOLD_HIGH = 37.0   # Batas atas suhu untuk notifikasi
# --------------------

def send_notification(title, content):
    """Fungsi untuk mengirim notifikasi menggunakan Termux API."""
    command = f'termux-notification -t "{title}" -c "{content}"'
    os.system(command)
    print(f"Notifikasi terkirim: {title} - {content}")

def fetch_and_check_device_status():
    """Mengambil data dari URL dan memeriksa status perangkat."""
    try:
        # Menambahkan timeout untuk menghindari script menggantung jika tidak ada koneksi
        response = requests.get(URL, timeout=10)
        
        # Memeriksa apakah request berhasil (status code 200)
        response.raise_for_status()
        
        # Mengurai data JSON
        data = response.json()
        
        battery = data.get('battery_level')
        temp = data.get('temperature_celsius')
        
        # Memastikan key ada di dalam JSON
        if battery is None or temp is None:
            print("Error: Key 'battery_level' atau 'temperature_celsius' tidak ditemukan di JSON.")
            return

        print(f"Pengecekan berhasil: Baterai={battery}%, Suhu={temp}°C")

        # Kondisi untuk notifikasi baterai
        if battery < BATTERY_THRESHOLD_LOW:
            title = "⚠️ Peringatan Baterai Lemah!"
            content = f"Level baterai di bawah {BATTERY_THRESHOLD_LOW}%. Saat ini: {battery}%"
            send_notification(title, content)

        # Kondisi untuk notifikasi suhu
        if temp > TEMP_THRESHOLD_HIGH:
            title = "🔥 Peringatan Suhu Tinggi!"
            content = f"Suhu perangkat di atas {TEMP_THRESHOLD_HIGH}°C. Saat ini: {temp}°C"
            send_notification(title, content)

    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil data: {e}")
    except json.JSONDecodeError:
        print("Error: Gagal mem-parsing data JSON dari server.")
    except Exception as e:
        print(f"Terjadi error yang tidak terduga: {e}")


if __name__ == "__main__":
    print("Memulai script monitoring perangkat...")
    print(f"URL: {URL}")
    print(f"Interval Pengecekan: {CHECK_INTERVAL_SECONDS} detik")
    print("-------------------------------------------------")
    
    while True:
        fetch_and_check_device_status()
        time.sleep(CHECK_INTERVAL_SECONDS)
