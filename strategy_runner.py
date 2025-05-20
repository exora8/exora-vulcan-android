def start_trading(global_settings_dict):
    global flask_thread, is_flask_running # Deklarasi global

    clear_screen_animated()
    # ... (kode API key manager dan inisialisasi konfigurasi crypto yang sudah ada) ...

    # Mulai server Flask jika belum berjalan
    if not is_flask_running:
        log_info("Attempting to start live chart server...", pair_name="SYSTEM_CHART")
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        time.sleep(2) # Beri waktu server Flask untuk mulai
        if is_flask_running:
            log_info(f"{AnsiColors.GREEN}Live chart server started. Buka http://127.0.0.1:5000 di browser.{AnsiColors.ENDC}", pair_name="SYSTEM_CHART")
            try:
                webbrowser.open_new_tab("http://127.0.0.1:5000")
            except Exception as e_wb:
                log_warning(f"Tidak bisa membuka browser secara otomatis: {e_wb}", pair_name="SYSTEM_CHART")
        else:
            log_error(f"{AnsiColors.RED}Gagal memulai server live chart (mungkin port sudah digunakan atau error lain).{AnsiColors.ENDC}", pair_name="SYSTEM_CHART")
    
    # ... (sisa kode dari fungsi start_trading, termasuk loop utama) ...

    # Bagian finally dari try-except KeyboardInterrupt di start_trading:
    # Tidak perlu secara eksplisit menghentikan flask_thread jika daemon=True,
    # karena ia akan berhenti ketika program utama keluar.
    # Jika kamu ingin menambahkan tombol stop di UI chart, itu akan lebih kompleks.
