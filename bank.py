import requests
from bs4 import BeautifulSoup
import time
import hashlib
import subprocess
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align

# --- Konfigurasi ---
URL_MEETINGS = "https://www.cbrates.com/meetings.htm"
URL_RATES = "https://www.cbrates.com/rates/"
URL_DECISIONS = "https://www.cbrates.com/"
WAKTU_REFRESH_DETIK = 60  # Ganti sesuai kebutuhan (misal: 300 untuk 5 menit)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- State untuk Notifikasi (jangan diubah) ---
last_hashes = {
    "meetings": None,
    "decisions": None,
    "rates": None,
}

# --- Inisialisasi Rich Console ---
console = Console()

def kirim_notifikasi(judul, konten):
    """Mengirim notifikasi menggunakan termux-notification."""
    try:
        # Menjalankan perintah termux-notification di background
        subprocess.run([
            'termux-notification',
            '--title', judul,
            '--content', konten,
            '--led-color', '00FF00', # Warna LED notifikasi (hijau)
            '--vibrate', '500'      # Getar selama 500ms
        ], check=True, timeout=10)
    except FileNotFoundError:
        console.print("[bold red]Peringatan: 'termux-notification' tidak ditemukan. Install 'pkg install termux-api' untuk notifikasi.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Gagal mengirim notifikasi: {e}[/bold red]")

def get_data_hash(element):
    """Membuat hash SHA256 dari konten teks sebuah elemen HTML."""
    if element:
        return hashlib.sha256(element.get_text().strip().encode('utf-8')).hexdigest()
    return None

def fetch_data():
    """Mengambil dan mem-parsing semua data dari cbrates.com."""
    global last_hashes
    headers = {'User-Agent': USER_AGENT}
    
    data = {
        "meetings_table": None,
        "notifications": []
    }

    try:
        # 1. Ambil Data Rapat Moneter (Meetings)
        response_meetings = requests.get(URL_MEETINGS, headers=headers, timeout=15)
        response_meetings.raise_for_status()
        soup_meetings = BeautifulSoup(response_meetings.text, 'html.parser')
        
        table_meetings = soup_meetings.find('table', {'id': 'meetings'})
        new_meetings_hash = get_data_hash(table_meetings)
        
        if last_hashes["meetings"] is not None and new_meetings_hash != last_hashes["meetings"]:
            data["notifications"].append(("Update Jadwal Rapat", "Ada perubahan pada jadwal rapat moneter."))
        last_hashes["meetings"] = new_meetings_hash
        
        if table_meetings:
            rich_table_meetings = Table(title="[bold cyan]Jadwal Rapat Bank Sentral Mendatang[/bold cyan]", expand=True)
            headers_list = [th.text.strip() for th in table_meetings.find_all('th')]
            for header in headers_list:
                rich_table_meetings.add_column(header, justify="left", style="white", no_wrap=False)

            for row in table_meetings.find('tbody').find_all('tr'):
                cols = [td.text.strip() for td in row.find_all('td')]
                rich_table_meetings.add_row(*cols)
            data["meetings_table"] = rich_table_meetings

        # 2. Cek Keputusan Moneter (Decisions)
        response_decisions = requests.get(URL_DECISIONS, headers=headers, timeout=15)
        response_decisions.raise_for_status()
        soup_decisions = BeautifulSoup(response_decisions.text, 'html.parser')
        
        heading_decisions = soup_decisions.find('h2', string="Latest Monetary Policy Decisions")
        table_decisions = heading_decisions.find_next_sibling('table') if heading_decisions else None
        new_decisions_hash = get_data_hash(table_decisions)

        if last_hashes["decisions"] is not None and new_decisions_hash != last_hashes["decisions"]:
            data["notifications"].append(("Update Keputusan Moneter", "Ada keputusan kebijakan moneter baru."))
        last_hashes["decisions"] = new_decisions_hash

        # 3. Cek Suku Bunga Dunia (Rates)
        response_rates = requests.get(URL_RATES, headers=headers, timeout=15)
        response_rates.raise_for_status()
        soup_rates = BeautifulSoup(response_rates.text, 'html.parser')
        
        table_rates = soup_rates.find('table', {'id': 'datatable'})
        new_rates_hash = get_data_hash(table_rates)

        if last_hashes["rates"] is not None and new_rates_hash != last_hashes["rates"]:
             data["notifications"].append(("Update Suku Bunga", "Ada perubahan pada data suku bunga dunia."))
        last_hashes["rates"] = new_rates_hash

    except requests.exceptions.RequestException as e:
        error_message = Align.center(f"[bold red]Gagal mengambil data: {e}[/bold red]", vertical="middle")
        # Jika tabel sudah ada, tampilkan error di footer. Jika belum, tampilkan di konten utama.
        if data["meetings_table"] is None:
            data["meetings_table"] = error_message
        else:
            data["error_footer"] = str(e)

    return data

def generate_layout():
    """Membuat struktur layout untuk ditampilkan."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    return layout

def main():
    """Fungsi utama untuk menjalankan monitor."""
    layout = generate_layout()
    header, main_content, footer = layout["header"], layout["main"], layout["footer"]

    header.update(Panel(Align.center("[bold green]cbrates.com Realtime Monitor for Termux[/bold green]"), border_style="green"))
    
    console.print("Mengambil data awal, mohon tunggu...")
    initial_data = fetch_data()
    if "error_footer" not in initial_data and initial_data["meetings_table"] is not None:
        console.print("[green]Data awal berhasil diambil. Memulai monitor... (Tekan Ctrl+C untuk berhenti)[/green]")
    else:
        console.print("[red]Gagal mengambil data awal. Memulai monitor dengan status error...[/red]")
        main_content.update(initial_data.get("meetings_table", "Kesalahan tidak diketahui."))

    with Live(layout, screen=True, redirect_stderr=False, refresh_per_second=4) as live:
        try:
            while True:
                data = fetch_data()
                
                for judul, konten in data.get("notifications", []):
                    kirim_notifikasi(judul, konten)

                main_content.update(
                    Panel(
                        data.get("meetings_table") or Align.center("[yellow]Data rapat tidak tersedia.[/yellow]"),
                        border_style="cyan",
                        title="[bold]Meetings[/bold]"
                    )
                )

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                notif_status = "[bold green]Tidak ada update baru.[/bold green]"
                if data.get("notifications"):
                    notif_status = f"[bold yellow]Update terdeteksi: {', '.join([n[0] for n in data['notifications']])}[/bold yellow]"
                elif "error_footer" in data:
                    notif_status = f"[bold red]Error: {data['error_footer']}[/bold red]"

                footer.update(Panel(
                    Align.center(f"Last Update: {now} | Refresh setiap {WAKTU_REFRESH_DETIK} detik | {notif_status}"),
                    border_style="blue"
                ))

                live.update(layout)
                time.sleep(WAKTU_REFRESH_DETIK)

        except KeyboardInterrupt:
            pass # Live context manager akan handle cleanup
    
    console.print("\n[bold yellow]Monitor dihentikan oleh pengguna.[/bold yellow]")

if __name__ == "__main__":
    main()
