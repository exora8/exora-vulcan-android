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

# --- Konfigurasi (URL SUDAH DIPERBARUI) ---
URL_RATES = "https://www.cbrates.com/"
URL_DECISIONS = "https://www.cbrates.com/decisions.htm"
URL_MEETINGS = "https://www.cbrates.com/meetings.htm"
WAKTU_REFRESH_DETIK = 90  # Refresh setiap 1.5 menit
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- State untuk Notifikasi ---
last_hashes = {"rates": None, "decisions": None, "meetings": None}

# --- Inisialisasi Rich Console ---
console = Console()

def kirim_notifikasi(judul, konten):
    """Mengirim notifikasi menggunakan termux-notification."""
    try:
        subprocess.run(
            ['termux-notification', '--title', judul, '--content', konten, '--vibrate', '500'],
            check=True, timeout=10
        )
    except Exception:
        # Gagal secara diam-diam jika termux-api tidak ada
        pass

def get_data_hash(element):
    """Membuat hash SHA256 dari konten teks sebuah elemen HTML."""
    if element:
        return hashlib.sha256(element.get_text().strip().encode('utf-8')).hexdigest()
    return None

def parse_html_table_to_rich(html_table, title, title_style="white"):
    """Mengubah elemen tabel HTML menjadi tabel Rich."""
    if not html_table:
        return Align.center(f"[yellow]Data '{title}' tidak ditemukan di halaman sumber.[/yellow]", vertical="middle")

    rich_table = Table(title=f"[bold {title_style}]{title}[/bold {title_style}]", expand=True)
    try:
        headers = [th.text.strip() for th in html_table.find('thead').find_all('th')]
        for header in headers:
            rich_table.add_column(header, justify="left", style="white", no_wrap=False)

        for row in html_table.find('tbody').find_all('tr'):
            # Ambil hanya beberapa kolom pertama jika tabel terlalu lebar untuk HP
            cols = [td.text.strip() for td in row.find_all('td')][:5] # Batasi 5 kolom
            rich_table.add_row(*cols)
        return rich_table
    except Exception as e:
        return Align.center(f"[red]Gagal mem-parsing tabel '{title}': {e}[/red]", vertical="middle")


def fetch_all_data():
    """Mengambil dan mem-parsing semua data dari tiga URL."""
    global last_hashes
    headers = {'User-Agent': USER_AGENT}
    data = {"notifications": []}

    sources = {
        "rates": {"url": URL_RATES, "id": "datatable", "title": "World Interest Rates", "style": "green"},
        "decisions": {"url": URL_DECISIONS, "id": "decisions", "title": "Monetary Decisions", "style": "yellow"},
        "meetings": {"url": URL_MEETINGS, "id": "meetings", "title": "Monetary Meetings", "style": "cyan"},
    }

    for name, src in sources.items():
        try:
            response = requests.get(src["url"], headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            html_table = soup.find('table', {'id': src["id"]})

            new_hash = get_data_hash(html_table)
            if last_hashes[name] is not None and new_hash != last_hashes[name]:
                data["notifications"].append((f"Update {src['title']}", f"Ada perubahan data di bagian {src['title']}."))
            last_hashes[name] = new_hash

            data[f"{name}_table"] = parse_html_table_to_rich(html_table, src["title"], src["style"])

        except requests.RequestException as e:
            data[f"{name}_table"] = Align.center(f"[bold red]Gagal mengambil data {name}: {e}[/bold red]", vertical="middle")
            data["error"] = True

    return data

def generate_layout():
    """Menciptakan struktur layout dashboard."""
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    # Bagi area utama menjadi tiga bagian
    layout["main"].split_column(
        Layout(name="rates", ratio=1),
        Layout(name="decisions", ratio=1),
        Layout(name="meetings", ratio=1)
    )
    return layout

def main():
    layout = generate_layout()
    header, main, footer = layout["header"], layout["main"], layout["footer"]
    header.update(Panel(Align.center("[bold magenta]cbrates.com Dashboard for Termux v2[/bold magenta]"), border_style="magenta"))
    
    console.print("Mengambil data awal, mohon tunggu...")
    # Lakukan fetch awal untuk mengisi hash
    fetch_all_data()
    console.print("[green]Data awal berhasil diambil. Memulai dashboard... (Tekan Ctrl+C untuk berhenti)[/green]")
    time.sleep(1)

    with Live(layout, screen=True, redirect_stderr=False, refresh_per_second=4) as live:
        try:
            while True:
                all_data = fetch_all_data()

                for judul, konten in all_data.get("notifications", []):
                    kirim_notifikasi(judul, konten)

                # Update setiap panel di layout
                main["rates"].update(Panel(all_data.get("rates_table"), border_style="green"))
                main["decisions"].update(Panel(all_data.get("decisions_table"), border_style="yellow"))
                main["meetings"].update(Panel(all_data.get("meetings_table"), border_style="cyan"))

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                status = "[bold green]Monitoring...[/bold green]"
                if all_data.get("notifications"):
                    status = f"[bold yellow]Update Terdeteksi![/bold yellow]"
                elif all_data.get("error"):
                    status = "[bold red]Terjadi Kesalahan Jaringan[/bold red]"

                footer.update(Panel(
                    Align.center(f"Last Check: {now} | Refresh: {WAKTU_REFRESH_DETIK}s | Status: {status}"),
                    border_style="blue"
                ))
                
                live.update(layout)
                time.sleep(WAKTU_REFRESH_DETIK)

        except KeyboardInterrupt:
            pass # Live akan membersihkan layar secara otomatis

    console.print("\n[bold yellow]Dashboard dihentikan.[/bold yellow]")

if __name__ == "__main__":
    main()
