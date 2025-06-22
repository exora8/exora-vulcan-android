# Nama file: auto_private.py
# Script final untuk mengirim notifikasi ntfy ke channel Telegram PRIVATE

import requests
import json
import os
import time

# --- KONFIGURASI FINAL ---
NTFY_TOPIC = "gimps-global-notificationA87X"
TELEGRAM_BOT_TOKEN = "7590245062:AAH2sLwsDvukOrVPUF7-iXU45fnfd20UV2M"
# --- PERUBAHAN DI SINI ---
# Ganti dengan ID numerik channel private Anda (diawali dengan tanda minus).
# Contoh: -1001234567890
TELEGRAM_CHAT_ID = -1002705544292 # <-- GANTI DENGAN ID CHANNEL ANDA
# -------------------------


def send_to_telegram(message_text):
    """Fungsi untuk mengirim pesan ke Telegram."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Pastikan chat_id dikirim sebagai integer/string, keduanya didukung oleh API
    payload = {
        'chat_id': str(TELEGRAM_CHAT_ID), 
        'text': message_text,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        
        response_data = response.json()
        if not response_data.get('ok'):
            error_desc = response_data.get('description', 'Unknown error')
            print(f"!!! Error dari Telegram: {error_desc}")
            print("!!! PASTIKAN bot sudah menjadi admin di channel DAN ID channel sudah benar.")
            return

        response.raise_for_status()
        print(f"Pesan berhasil dikirim ke channel ID: {TELEGRAM_CHAT_ID}")
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
