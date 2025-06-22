# Nama file: auto.py
# Script final untuk mengirim notifikasi ntfy ke channel Telegram (TANPA TAGS)

import requests
import json
import os
import time

# --- KONFIGURASI FINAL ---
NTFY_TOPIC = "gimps-global-notificationA87X"
TELEGRAM_BOT_TOKEN = "7590245062:AAH2sLwsDvukOrVPUF7-iXU45fnfd20UV2M"
TELEGRAM_CHAT_ID = "@GlobalMacroPulse8" 
# -------------------------


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
            error_desc = response_data.get('description', 'Unknown error')
            print(f"!!! Error dari Telegram: {error_desc}")
            print("!!! PASTIKAN bot sudah menjadi admin di channel DAN memiliki izin 'Post Messages'.")
            return

        response.raise_for_status()
        print(f"Pesan berhasil dikirim ke channel: {TELEGRAM_CHAT_ID}")
    except requests.exceptions.RequestException as e:
        print(f"Error saat request ke Telegram: {e}")

def listen_to_ntfy():
    """Fungsi utama untuk mendengarkan ntfy.sh dan meneruskan pesan."""
    ntfy_stream_url = f"https://ntfy.sh/{NTFY_TOPIC}/json"
    print(f"Mendengarkan notifikasi dari ntfy.sh di topik: {NTFY_TOPIC}")
    print(f"Pesan akan diteruskan ke Telegram Channel: {TELEGRAM_CHAT_ID}")

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
                            # BARIS UNTUK TAGS DIHAPUS DARI SINI
                            
                            priority_emojis = {1: "‼️", 2: "❗️", 3: "ℹ️", 4: "✅", 5: "🔥"}
                            priority_emoji = priority_emojis.get(priority, "⚪️")
                            
                            # Pembuatan pesan sekarang tidak menyertakan tags
                            formatted_message = (
                                f"{priority_emoji} *{title}*\n\n"
                                f"{message}\n\n"
                            )
                            # BLOK 'if tags:' JUGA DIHAPUS DARI SINI

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
