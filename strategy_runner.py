document.addEventListener('DOMContentLoaded', function () {
    const socket = io(); // Terhubung ke server SocketIO Python

    const logOutput = document.getElementById('log-output');
    const settingsOutput = document.getElementById('settings-output');
    const loadSettingsBtn = document.getElementById('load-settings-btn');
    const chartContainer = document.getElementById('chart-container');
    const pairSelector = document.getElementById('pair-selector');

    let chart = null;
    let candleSeries = null;
    let chartsData = {}; // Untuk menyimpan data dan marker per pair_id { pair_id: { chart: obj, series: obj, markers: [] } }

    // --- Logging ---
    socket.on('new_log', function (data) {
        const logEntry = document.createElement('p');
        logEntry.innerHTML = `<code>[${data.timestamp}] [${data.level}] [${data.pair_name}]</code>: ${data.message}`;
        logEntry.classList.add(data.level); // Untuk styling CSS berdasarkan level
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight; // Auto-scroll
    });

    socket.on('server_message', function(data) {
        console.log('Server message:', data.data);
        const logEntry = document.createElement('p');
        logEntry.textContent = `SERVER: ${data.data}`;
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    });
    
    socket.on('settings_updated', function(data) {
        alert(data.message);
        fetchAndDisplaySettings(); // Muat ulang tampilan pengaturan
    });

    // --- Pengaturan ---
    if (loadSettingsBtn) {
        loadSettingsBtn.addEventListener('click', fetchAndDisplaySettings);
    }

    function fetchAndDisplaySettings() {
        fetch('/api/settings')
            .then(response => response.json())
            .then(data => {
                settingsOutput.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                console.error('Error fetching settings:', error);
                settingsOutput.textContent = 'Gagal memuat pengaturan.';
            });
    }
    // Panggil sekali saat load awal
    fetchAndDisplaySettings();


    // --- Charting ---
    function createOrUpdateChart(pairId) {
        if (chartsData[pairId] && chartsData[pairId].chart) {
            // Chart sudah ada, mungkin hanya perlu di-clear atau fokus
            chart = chartsData[pairId].chart;
            candleSeries = chartsData[pairId].series;
            // Hapus marker lama jika ada
            chartsData[pairId].markers.forEach(marker => candleSeries.removePriceLine(marker));
            chartsData[pairId].markers = [];
            return;
        }

        // Hapus chart lama jika ada yang beda pairId
        if (chart) {
            chart.remove();
            chart = null;
            candleSeries = null;
        }
        chartContainer.innerHTML = ''; // Bersihkan container

        chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight, // Biasanya 400 dari CSS
            layout: {
                backgroundColor: '#ffffff',
                textColor: 'rgba(33, 56, 77, 1)',
            },
            grid: {
                vertLines: { color: 'rgba(197, 203, 206, 0.5)' },
                horzLines: { color: 'rgba(197, 203, 206, 0.5)' },
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: 'rgba(197, 203, 206, 0.8)' },
            timeScale: { borderColor: 'rgba(197, 203, 206, 0.8)', timeVisible: true, secondsVisible: false },
        });

        candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
            wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        });
        
        chartsData[pairId] = { chart: chart, series: candleSeries, markers: [] };
    }
    
    socket.on('active_pairs_list', function(pairs) {
        pairSelector.innerHTML = '<option value="">-- Pilih Pair --</option>'; // Reset
        pairs.forEach(pair => {
            const option = document.createElement('option');
            option.value = pair.id;
            option.textContent = pair.name;
            pairSelector.appendChild(option);
        });
    });

    pairSelector.addEventListener('change', function() {
        const selectedPairId = this.value;
        if (selectedPairId) {
            createOrUpdateChart(selectedPairId); // Buat chart kosong dulu
            // Minta data awal untuk pair yang dipilih
            fetch(`/api/chart_data/${selectedPairId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.candles && data.candles.length > 0) {
                        chartsData[selectedPairId].series.setData(data.candles);
                        console.log(`Initial chart data loaded for ${selectedPairId}`);
                    } else {
                        console.log(`No initial candle data for ${selectedPairId}`);
                         chartsData[selectedPairId].series.setData([]); // Kosongkan jika tidak ada data
                    }
                    // Anda juga bisa memproses strategy_state awal di sini untuk marker awal
                })
                .catch(error => console.error(`Error fetching initial chart data for ${selectedPairId}:`, error));
        }
    });


    socket.on('candle_update', function (data) {
        const currentPairId = pairSelector.value;
        if (data.pair_id === currentPairId && chartsData[currentPairId] && chartsData[currentPairId].series) {
            chartsData[currentPairId].series.update(data.candle);
            // Jika ada event bersamaan dengan candle (misal, pivot terkonfirmasi di candle ini)
            if (data.event) {
                handleStrategyEvent(data.pair_id, data.event.type, data.event.data);
            }
        }
    });
    
    socket.on('strategy_event', function (payload) {
        const currentPairId = pairSelector.value;
        if (payload.pair_id === currentPairId) {
            handleStrategyEvent(payload.pair_id, payload.type, payload.data);
        }
    });

    function handleStrategyEvent(pairId, eventType, eventData) {
        if (!chartsData[pairId] || !chartsData[pairId].series) return;

        const series = chartsData[pairId].series;
        let markerLine;

        // Hapus marker lama dengan tipe dan ID yang sama jika perlu (belum diimplementasikan ID marker)
        // Contoh: cari marker 'entry_price' dan hapus sebelum menambah yang baru

        if (eventType === 'pivot_high' || eventType === 'pivot_low') {
            markerLine = series.createPriceLine({
                price: eventData.price,
                color: eventType === 'pivot_high' ? 'red' : 'green',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `${eventType.replace('_', ' ').toUpperCase()} @ ${eventData.price.toFixed(5)}`,
            });
        } else if (eventType === 'buy_entry') {
            markerLine = series.createPriceLine({
                price: eventData.price,
                color: 'blue',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true,
                title: `ENTRY @ ${eventData.price.toFixed(5)}`,
            });
            chartsData[pairId].markers.push(markerLine); // Simpan referensi marker

            if (eventData.sl) {
                const slLine = series.createPriceLine({
                    price: eventData.sl,
                    color: 'orange',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: true,
                    title: `SL @ ${eventData.sl.toFixed(5)}`,
                });
                chartsData[pairId].markers.push(slLine);
            }
        } else if (eventType === 'exit_order') {
             markerLine = series.createPriceLine({
                price: eventData.price,
                color: eventData.pnl_percent >= 0 ? 'darkgreen' : 'darkred',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true,
                title: `EXIT (${eventData.reason}) @ ${eventData.price.toFixed(5)} PnL: ${eventData.pnl_percent.toFixed(2)}%`,
            });
        } else if (eventType === 'fib_level_active') { // Contoh event baru
            markerLine = series.createPriceLine({
                price: eventData.price,
                color: 'purple',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.LongDashed,
                axisLabelVisible: true,
                title: `FIB 0.5 @ ${eventData.price.toFixed(5)}`,
            });
        }
        // Tambahkan tipe event lain sesuai kebutuhan (SL update, TP, dll.)

        if (markerLine) {
            chartsData[pairId].markers.push(markerLine);
        }
    }
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (chart) {
            chart.applyOptions({ width: chartContainer.clientWidth, height: chartContainer.clientHeight });
        }
    });

});
