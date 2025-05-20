<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Crypto Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #1e1e1e; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; padding: 10px;}
        #controls { background-color: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width:100%; max-width: 1200px; }
        #controls label { font-size: 0.9em; }
        select, button { padding: 8px 12px; font-size:0.9em; border-radius: 5px; border: 1px solid #444; background-color: #333; color: #e0e0e0; cursor:pointer; }
        button:hover { background-color: #444; }
        #chart-container { width: 100%; max-width: 1200px; background-color: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        h1 { text-align: center; color: #00bcd4; margin-top: 0; margin-bottom:15px; font-size:1.5em; }
        #lastUpdatedLabel { font-size: 0.8em; color: #aaa; margin-left: auto; /* Aligns to the right */ padding-right: 10px; }
        .apexcharts-tooltip-candlestick { background: #333 !important; color: #fff !important; border: 1px solid #555 !important;}
        .apexcharts-tooltip-candlestick .value { font-weight: bold; }
        .apexcharts-marker-inverted .apexcharts-marker-poly { transform: rotate(180deg); transform-origin: center; } /* For inverted triangle */
    </style>
</head>
<body>
    <h1>Live Strategy Chart</h1>
    <div id="controls">
        <label for="pairSelector">Pilih Pair:</label>
        <select id="pairSelector" onchange="handlePairSelectionChange()"></select>
        <button onclick="loadChartDataForCurrentPair()">Refresh Manual</button>
        <span id="lastUpdatedLabel">Memuat...</span>
    </div>
    <div id="chart-container">
        <div id="chart"></div>
    </div>

    <script>
        let activeChart;
        let currentSelectedPairId = '';
        let lastKnownDataTimestamp = null;
        let autoRefreshIntervalId = null;
        let isLoadingData = false; // Flag untuk mencegah fetch data ganda

        const initialChartOptions = {
            series: [{ name: 'Candlestick', data: [] }],
            chart: { 
                type: 'candlestick', 
                height: 550,
                id: 'mainCandlestickChart',
                background: '#2a2a2a',
                animations: { enabled: true, easing: 'easeinout', speed: 500, animateGradually: { enabled: false } },
                toolbar: { show: true, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } }
            },
            theme: { mode: 'dark' },
            title: { text: 'Memuat Data Pair...', align: 'left', style: { color: '#e0e0e0', fontSize: '16px'} },
            xaxis: { type: 'datetime', labels: { style: { colors: '#aaa'} }, tooltip: { enabled: false } },
            yaxis: { tooltip: { enabled: true }, labels: { style: { colors: '#aaa'}, formatter: function (value) { return value.toFixed(5); } } },
            grid: { borderColor: '#444' },
            annotations: { yaxis: [], points: [] },
            tooltip: { theme: 'dark', shared: true, 
                custom: function({series, seriesIndex, dataPointIndex, w}) {
                    if (w.globals.seriesCandleO && w.globals.seriesCandleO[seriesIndex] && w.globals.seriesCandleO[seriesIndex][dataPointIndex] !== undefined) {
                        const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
                        const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
                        const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
                        const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
                        return (
                            '<div class="apexcharts-tooltip-candlestick" style="padding:5px 10px;">' +
                            '<div>O: <span class="value">' + o.toFixed(5) + '</span></div>' +
                            '<div>H: <span class="value">' + h.toFixed(5) + '</span></div>' +
                            '<div>L: <span class="value">' + l.toFixed(5) + '</span></div>' +
                            '<div>C: <span class="value">' + c.toFixed(5) + '</span></div>' +
                            '</div>'
                        );
                    } return '';
                }
            },
            noData: { text: 'Tidak ada data untuk ditampilkan.', align: 'center', verticalAlign: 'middle', style: { color: '#ccc', fontSize: '14px' } }
        };

        async function fetchAvailablePairs() {
            try {
                const response = await fetch('/api/available_pairs');
                if (!response.ok) throw new Error(`Gagal memuat daftar pair: ${response.status}`);
                const pairs = await response.json();
                const selectorElement = document.getElementById('pairSelector');
                selectorElement.innerHTML = ''; 
                if (pairs.length > 0) {
                    pairs.forEach(pair => {
                        const optionEl = document.createElement('option');
                        optionEl.value = pair.id;
                        optionEl.textContent = pair.name;
                        selectorElement.appendChild(optionEl);
                    });
                    currentSelectedPairId = selectorElement.value || pairs[0].id; 
                    loadChartDataForCurrentPair(); 
                } else {
                     selectorElement.innerHTML = '<option value="">Tidak ada pair aktif</option>';
                     if(activeChart) activeChart.destroy();
                     activeChart = null; // Pastikan chart lama di-clear
                     document.getElementById('chart').innerHTML = '<p style="text-align:center; color:#aaa;">Tidak ada pair aktif yang dikonfigurasi di server.</p>';
                     document.getElementById('lastUpdatedLabel').textContent = "Tidak ada pair";
                }
            } catch (error) {
                console.error("Error fetching pair list:", error);
                document.getElementById('pairSelector').innerHTML = '<option value="">Error memuat pair</option>';
                if(activeChart) activeChart.destroy();
                activeChart = null;
                document.getElementById('chart').innerHTML = `<p style="text-align:center; color:red;">Error: ${error.message}</p>`;
                document.getElementById('lastUpdatedLabel').textContent = "Error";
            }
        }

        function handlePairSelectionChange() {
            currentSelectedPairId = document.getElementById('pairSelector').value;
            lastKnownDataTimestamp = null; 
            loadChartDataForCurrentPair();
        }

        async function loadChartDataForCurrentPair() {
            if (!currentSelectedPairId) {
                console.log("Tidak ada pair ID yang dipilih.");
                if(activeChart) activeChart.updateOptions(initialChartOptions);
                document.getElementById('lastUpdatedLabel').textContent = "Pilih pair";
                return;
            }
            if (isLoadingData) {
                console.log(`Sinkronisasi data untuk ${currentSelectedPairId} sedang berjalan. Lewati sementara.`);
                return;
            }

            isLoadingData = true;
            document.getElementById('lastUpdatedLabel').textContent = `Sinkronisasi ${currentSelectedPairId}...`;
            
            try {
                const fetchResponse = await fetch(`/api/chart_data/${currentSelectedPairId}`);
                if (!fetchResponse.ok) {
                     let errorMsgText = `Gagal mengambil data chart: ${fetchResponse.status}`;
                     try { const errorData = await fetchResponse.json(); errorMsgText = errorData.error || errorMsgText; } catch(e){}
                     throw new Error(errorMsgText);
                }
                const chartDataPayload = await fetchResponse.json();

                // Pemeriksaan data dari server (payload dan ohlc)
                if (!chartDataPayload || !chartDataPayload.ohlc || chartDataPayload.ohlc.length === 0) {
                    console.warn(`Data OHLC tidak diterima atau kosong untuk ${currentSelectedPairId}.`);
                    const pairDisplayName = chartDataPayload.pair_name || currentSelectedPairId;
                    const noDataOpts = {
                        ...initialChartOptions,
                        title: { ...initialChartOptions.title, text: `${pairDisplayName} - Tidak Ada Data Candle` },
                        series: [{ name: 'Candlestick', data: [] }],
                        annotations: { yaxis: [], points: [] },
                        noData: { text: 'Tidak ada data candle terbaru dari server.' }
                    };
                    if (!activeChart) {
                        activeChart = new ApexCharts(document.querySelector("#chart"), noDataOpts);
                        activeChart.render();
                    } else {
                        activeChart.updateOptions(noDataOpts);
                    }
                    lastKnownDataTimestamp = chartDataPayload.last_updated_tv || null;
                    document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data (kosong) @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "Tidak ada data";
                    isLoadingData = false; // Reset flag
                    return; 
                }

                // Cek jika data tidak berubah (berdasarkan timestamp terakhir)
                if (chartDataPayload.last_updated_tv && chartDataPayload.last_updated_tv === lastKnownDataTimestamp) {
                    console.log("Data chart tidak berubah, tidak perlu update render.");
                    document.getElementById('lastUpdatedLabel').textContent = `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}`;
                    isLoadingData = false; // Reset flag
                    return;
                }
                lastKnownDataTimestamp = chartDataPayload.last_updated_tv;
                document.getElementById('lastUpdatedLabel').textContent = lastKnownDataTimestamp ? `Data terakhir @${new Date(lastKnownDataTimestamp).toLocaleTimeString()}` : "N/A";

                const newChartOptions = {
                    ...initialChartOptions, 
                    title: { ...initialChartOptions.title, text: `${chartDataPayload.pair_name} Candlestick` },
                    series: [{ name: 'Candlestick', data: chartDataPayload.ohlc }], // Tidak perlu || [] karena sudah dicek di atas
                    annotations: { 
                        yaxis: chartDataPayload.annotations_yaxis || [], 
                        points: chartDataPayload.annotations_points || [] 
                    }
                };
                
                if (!activeChart) {
                    activeChart = new ApexCharts(document.querySelector("#chart"), newChartOptions);
                    activeChart.render();
                } else {
                    activeChart.updateOptions(newChartOptions);
                }

            } catch (error) {
                console.error("Error loading chart data:", error);
                const pairDisplayNameError = currentSelectedPairId || "Chart"; // Fallback display name
                if (activeChart) { 
                    activeChart.destroy();
                    activeChart = null; 
                }
                const errorChartOpts = { 
                    ...initialChartOptions, 
                    title: { ...initialChartOptions.title, text: `Error: ${pairDisplayNameError}` }, 
                    series: [], 
                    noData: { text: `Gagal memuat data: ${error.message}` } 
                };
                activeChart = new ApexCharts(document.querySelector("#chart"), errorChartOpts);
                activeChart.render();
                document.getElementById('lastUpdatedLabel').textContent = "Error update";
            } finally {
                isLoadingData = false; // Selalu reset flag setelah selesai (sukses atau error)
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            // Inisialisasi chart dasar saat DOM siap, sebelum data pertama dimuat
            if (!activeChart) {
                 activeChart = new ApexCharts(document.querySelector("#chart"), initialChartOptions);
                 activeChart.render();
            }

            fetchAvailablePairs(); 
            
            if (autoRefreshIntervalId) clearInterval(autoRefreshIntervalId); 
            autoRefreshIntervalId = setInterval(async () => {
                if (currentSelectedPairId && document.visibilityState === 'visible' && !isLoadingData) { 
                    console.log(`Auto-refresh chart untuk ${currentSelectedPairId}`);
                    await loadChartDataForCurrentPair();
                }
            }, 30000); 
        });

    </script>
</body>
</html>
