strategy_runner.py -> untuk menjalankan AI secara live dan auto trading dengan data trades.json yang sudah dibuat dan di backtest


scheduler.py -> untuk membuat device AIbox meng upload device.json yang berisi info tentang suhu, level baterai dan trades.json langsung ke repo github dan bisa di fetch secara live dari html yg udh di hosting di playcode.io


maintance.py -> untuk device yang mau listen langsung ke server untuk menerima alert apakah suhu terlalu panas atau baterai sudah dibawah 50%

backtester.html -> membuat json untuk backtest AI
