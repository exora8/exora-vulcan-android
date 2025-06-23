# Nama file: auto_private_fixed.py
# Script final untuk mengirim notifikasi ntfy ke channel Telegram PRIVATE (dengan perbaikan parsing Markdown)

import requests
import json
import time

# --- KONFIGURASI FINAL ---
NTFY_TOPIC = "gimps-global-notificationA87XY" # Pastikan ini sama persis dengan topik di skrip GIMPS
TELEGRAM_BOT_TOKEN = "7590245062:AAH2sLwsDvukOrVPUF7-iXU45fnfd20UV2M"
# --- PERUBAHAN DI SINI ---
# Ganti dengan ID numerik channel private Anda (diawali dengan tanda minus).
# Contoh: -1001234567890
TELEGRAM_CHAT_ID = -1002705544292 # <-- GANTI DENGAN ID CHANNEL ANDA
# -------------------------

def sanitize_telegram_markdown(text: str) -> str:
    """
    Membersihkan teks dari karakter spesial Markdown yang bisa menyebabkan error di Telegram API.
    Telegram sangat ketat, terutama dengan karakter seperti '.', '-', '!', dll.
    Fungsi ini akan 'meloloskan' (escape) karakter-karakter tersebut.
    """
    # Daftar karakter yang harus di-escape untuk parse_mode 'MarkdownV2'.
    # Meskipun kita pakai 'Markdown', lebih aman untuk escape set yang lebih lengkap ini.
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    
    # Buat dictionary untuk replace yang lebih efisien jika diperlukan, tapi loop sederhana sudah cukup
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text

def send_to_telegram(message_text):
    """Fungsi untuk mengirim pesan ke Telegram."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': str(TELEGRAM_CHAT_ID), 
        'text': message_text,
        # Kita tetap menggunakan 'Markdown'. Versi V2 lebih ketat.
        'parse_mode': 'MarkdownV2' # Mengganti ke MarkdownV2 yang lebih konsisten dengan karakter escape kita
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=20) # Timeout ditambah
        
        # Selalu cek respons, bahkan jika tidak ada exception
        if response.status_code != 200:
            response_data = response.json()
            error_desc = response_data.get('description', 'Unknown error')
            print(f"!!! Error dari Telegram (Status {response.status_code}): {error_desc}")
            print("!!! Pesan yang gagal dikirim (setelah sanitasi):")
            print(message_text)
            print("!!! PASTIKAN bot sudah menjadi admin di channel DAN ID channel sudah benar.")
            return

        print(f"Pesan berhasil dikirim ke channel ID: {TELEGRAM_CHAT_ID}")
    except requests.exceptions.RequestException as e:
        print(f"Error saat request ke Telegram: {e}")
    except json.JSONDecodeError:
        print("!!! Gagal mem-parse respons dari Telegram. Respons mentah:")
        print(response.text)


def listen_to_ntfy():
    """Fungsi utama untuk mendengarkan ntfy.sh dan meneruskan pesan."""
    ntfy_stream_url = f"https://ntfy.sh/{NTFY_TOPIC}/json"
    print(f"Mendengarkan notifikasi dari ntfy.sh di topik: {NTFY_TOPIC}")
    print(f"Pesan akan diteruskan ke Telegram Channel ID: {TELEGRAM_CHAT_ID}")

    while True:
        try:
            # Menggunakan header untuk memastikan koneksi tetap hidup
            headers = {'Connection': 'keep-alive'}
            response = requests.get(ntfy_stream_url, stream=True, timeout=90, headers=headers)
            response.raise_for_status()
            
            print("Koneksi ke ntfy.sh berhasil. Menunggu pesan...")
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    if data.get('event') == 'message':
                        title = data.get('title', 'Tanpa Judul')
                        message = data.get('message', '')
                        
                        # --- PERBAIKAN UTAMA ADA DI SINI ---
                        # Sanitasi judul dan pesan secara terpisah.
                        # Judul akan kita buat bold secara manual.
                        # Pesan dari AI (yang sudah mengandung Markdown) harus disanitasi.
                        sanitized_title = sanitize_telegram_markdown(title)
                        sanitized_message = sanitize_telegram_markdown(message)
                        
                        # Format akhir untuk dikirim ke Telegram
                        # Kita gunakan format MarkdownV2
                        formatted_message = (
                            f"*{sanitized_title}*\n\n"
                            f"{sanitized_message}"
                        )

                        print(f"Menerima notifikasi: {title}")
                        send_to_telegram(formatted_message)

                except json.JSONDecodeError:
                    # Baris kosong atau baris 'keep-alive' dari ntfy bisa menyebabkan ini, aman untuk diabaikan.
                    continue
                except Exception as e:
                    print(f"Terjadi error saat memproses pesan: {e}")

        except requests.exceptions.ConnectionError:
            print("Koneksi ke ntfy.sh terputus. Mencoba menyambung kembali dalam 15 detik...")
            time.sleep(15)
        except requests.exceptions.ReadTimeout:
            print("Koneksi ke ntfy.sh timeout. Menyambung kembali...")
            # Tidak perlu sleep, langsung coba lagi
        except requests.exceptions.RequestException as e:
            print(f"Error pada request ke ntfy.sh: {e}. Mencoba lagi dalam 15 detik...")
            time.sleep(15)


if __name__ == "__main__":
    listen_to_ntfy()
