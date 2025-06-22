# Nama file: auto_private.py
# Skrip final untuk mengirim notifikasi ntfy ke PRIVATE channel Telegram
# MENGGUNAKAN ENVIRONMENT VARIABLE untuk keamanan token

import requests
import json
import os
import time

# --- KONFIGURASI ---
# Ambil konfigurasi dari environment variables untuk keamanan
# Pastikan kamu sudah mengatur environment variable ini sebelum menjalankan skrip
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "gimps-global-notificationA87X")
TELEGRAM_BOT_TOKEN = os.getenv("7590245062:AAH2sLwsDvukOrVPUF7-iXU45fnfd20UV2M")

# GANTI DENGAN ID CHANNEL PRIVAT KAMU (berupa angka, bukan string)
TELEGRAM_CHAT_ID = -1002705544292 # <--- UBAH BAGIAN INI DENGAN ID ASLI
# --------------------

# Validasi apakah token sudah di-set
if not TELEGRAM_BOT_TOKEN:
    print("!!! ERROR: Environment variable 'TELEGRAM_BOT_TOKEN' tidak ditemukan.")
    print("!!! Harap set token bot Anda sebelum menjalankan skrip.")
    exit() # Keluar dari skrip jika token tidak ada

def send_to_telegram(message_text):
    """Fungsi untuk mengirim pesan ke Telegram."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message_text,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        
        response_data = response.json()
        if not response_data.get('ok'):
            error_code = response_data.get('error_code')
            error_desc = response_data.get('description', 'Unknown error')
            print(f"!!! Error dari Telegram (Kode: {error_code}): {error_desc}")
            if error_code == 400 and 'chat not found' in error_desc:
                print("!!! PASTIKAN 'TELEGRAM_CHAT_ID' sudah benar dan bot sudah ditambahkan ke channel.")
            elif error_code == 403 and 'bot is not a member' in error_desc:
                 print("!!! PASTIKAN bot sudah menjadi anggota di channel/grup privat tersebut.")
            else:
                print("!!! PASTIKAN bot sudah menjadi admin di channel DAN memiliki izin 'Post Messages'.")
            return

        response.raise_for_status()
        print(f"Pesan berhasil dikirim ke private channel ID: {TELEGRAM_CHAT_ID}")
    except requests.exceptions.RequestException as e:
        print(f"Error saat request ke Telegram: {e}")

def listen_to_ntfy():
    """Fungsi utama untuk mendengarkan ntfy.sh dan meneruskan pesan."""
    ntfy_stream_url = f"https://ntfy.sh/{NTFY_TOPIC}/json"
    print(f"Mendengarkan notifikasi dari ntfy.sh di topik: {NTFY_TOPIC}")
    print(f"Pesan akan diteruskan ke Telegram Channel ID: {TELEGRAM_CHAT_ID}")

    while True:
        try:
            response = requests.get(ntfy_stream_url, stream=True, timeout=90)
            response.raise_for_status()
            
            print("Koneksi ke ntfy.sh berhasil. Menunggu pesan...")
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if data.get('event') == 'message':
                            title = data.get('title', 'Tanpa Judul')
                            message = data.get('message', '')
                            priority = data.get('priority', 3)
                            
                            priority_emojis = {1: "‼️", 2: "❗️", 3: "ℹ️", 4: "✅", 5: "🔥"}
                            priority_emoji = priority_emojis.get(priority, "⚪️")
                            
                            formatted_message = (
                                f"{priority_emoji} *{title}*\n\n"
                                f"{message}\n\n"
                            )

                            print(f"Menerima notifikasi: {title}")
                            send_to_telegram(formatted_message)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Terjadi error saat memproses pesan: {e}")

        except requests.exceptions.ConnectionError as e:
            print(f"Koneksi ke ntfy.sh terputus: {e}. Mencoba menyambung kembali dalam 15 detik...")
            time.sleep(15)
        except requests.exceptions.RequestException as e:
            print(f"Error pada request ke ntfy.sh: {e}. Mencoba lagi dalam 15 detik...")
            time.sleep(15)


if __name__ == "__main__":
    listen_to_ntfy()
