document.addEventListener('DOMContentLoaded', function() {
    const socket = io(); // Connect to Socket.IO server
    const logContent = document.getElementById('logContent');
    const settingsContent = document.getElementById('settingsContent');
    const chartContainer = document.getElementById('chartContainer');
    const pairSelector = document.getElementById('pairSelector');
    const refreshPairsButton = document.getElementById('refreshPairs');

    let currentSelectedPairId = null;
    const MAX_LOG_LINES_FRONTEND = 200; // Sesuaikan dengan MAX_LOG_LINES di Python

    function appendLog(message) {
        logContent.innerHTML += message + '\n';
        // Keep only MAX_LOG_LINES_FRONTEND
        const lines = logContent.innerHTML.split('\n');
        if (lines.length > MAX_LOG_LINES_FRONTEND) {
            logContent.innerHTML = lines.slice(lines.length - MAX_LOG_LINES_FRONTEND).join('\n');
        }
        logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll
    }

    // --- Initial Data Loading ---
    function loadInitialLogs() {
        fetch('/api/initial_logs')
            .then(response => response.json())
            .then(logs => {
                logContent.innerHTML = ''; // Clear existing
                logs.forEach(log => appendLog(log));
            });
    }

    function loadSettings() {
        fetch('/api/settings')
            .then(response => response.json())
            .then(settings => {
                settingsContent.textContent = JSON.stringify(settings, null, 2);
            });
    }

    function loadActivePairs() {
        fetch('/api/active_pairs')
            .then(response => response.json())
            .then(pairs => {
                pairSelector.innerHTML = '<option value="">-- Select Pair --</option>';
                let firstPairId = null;
                pairs.forEach(pair => {
                    const option = document.createElement('option');
                    option.value = pair.id;
                    option.textContent = pair.name;
                    pairSelector.appendChild(option);
                    if (!firstPairId) firstPairId = pair.id;
                });

                // Auto-select first pair or previously selected if available
                if (currentSelectedPairId && pairs.some(p => p.id === currentSelectedPairId)) {
                    pairSelector.value = currentSelectedPairId;
                } else if (firstPairId) {
                    pairSelector.value = firstPairId;
                }
                loadChartForSelectedPair(); // Load chart for the auto-selected pair
            });
    }

    function loadChartForSelectedPair() {
        const selectedPairId = pairSelector.value;
        if (!selectedPairId) {
            Plotly.purge(chartContainer); // Clear chart
            chartContainer.innerHTML = 'Please select a pair.';
            return;
        }
        currentSelectedPairId = selectedPairId; // Update global selected pair

        fetch(`/api/chart_data/${selectedPairId}`)
            .then(response => response.json())
            .then(chartData => {
                if (chartData.error) {
                    console.error('Error loading chart data:', chartData.error);
                    chartContainer.innerHTML = `Error: ${chartData.error}`;
                    return;
                }
                if (chartData.data && chartData.layout) {
                    Plotly.react(chartContainer, chartData.data, chartData.layout);
                } else {
                    chartContainer.innerHTML = 'No chart data available for this pair.';
                }
            })
            .catch(error => {
                console.error('Failed to fetch chart data:', error);
                chartContainer.innerHTML = 'Failed to load chart data.';
            });
    }

    // --- Socket.IO Event Handlers ---
    socket.on('connect', () => {
        console.log('Connected to Socket.IO server!');
        // You might want to re-request data or specific pair chart on reconnect
    });

    socket.on('new_log', function(data) {
        appendLog(data.log);
    });

    socket.on('chart_update', function(data) {
        // console.log('Received chart_update for pair_id:', data.pair_id);
        // Only update if the received update is for the currently selected pair
        if (data.pair_id === currentSelectedPairId) {
            if (data.chart_data && data.chart_data.data && data.chart_data.layout) {
                Plotly.react(chartContainer, data.chart_data.data, data.chart_data.layout);
                // console.log('Chart updated for:', data.pair_id);
            } else {
                console.warn('Received chart_update with invalid data structure for', data.pair_id, data.chart_data);
            }
        }
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from Socket.IO server.');
    });

    // --- Event Listeners ---
    pairSelector.addEventListener('change', loadChartForSelectedPair);
    refreshPairsButton.addEventListener('click', loadActivePairs);

    // --- Initial Load ---
    loadInitialLogs();
    loadSettings();
    loadActivePairs(); // This will also trigger initial chart load
});
