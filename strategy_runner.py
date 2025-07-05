//@version=5
indicator("AI Backtest JSON Generator (FINAL v3)", overlay=true)

//==============================================================================
// INPUTS
//==============================================================================
g_main = "PENGATURAN UTAMA"
i_entryTimestamp = input.time(timestamp("01 Jan 2024 00:00 +0000"), "1. Pilih Candle Sinyal", group = g_main, tooltip = "Klik ikon kalender, lalu klik candle sinyal yang Anda inginkan di chart.")
i_tradeType = input.string("LONG", "2. Tipe Perdagangan", options = ["LONG", "SHORT"], group = g_main)
i_tpPercent = input.float(2.0, "3. Take Profit (%)", step = 0.1, group = g_main)
i_slPercent = input.float(1.0, "4. Stop Loss (%)", step = 0.1, group = g_main)
CANDLE_HISTORY_COUNT = 15 // Jumlah candle historis yang akan diambil

//==============================================================================
// STATE MANAGEMENT
//==============================================================================
var bool    isSignalTriggered = false
var bool    isInTrade = false
var float   entryPrice = na
var float   tpLevel = na
var float   slLevel = na
var int     entryBarIndex = na
var string  tradeType = ""
var int     signalTimestamp = na
var float   signal_ema9_current = na, var float signal_ema9_prev = na
var float   signal_ema50 = na, var float signal_ema100 = na
var float   signal_current_candle_close = na, var float signal_prev_candle_close = na
var string  signal_bias = "", var string signal_solidity = "", var string signal_direction = ""

//==============================================================================
// INDIKATOR
//==============================================================================
ema9 = ta.ema(close, 9)
ema50 = ta.ema(close, 50)
ema100 = ta.ema(close, 100)
candleDirection = close > open ? "UP" : "DOWN"
candleBody = math.abs(close - open)
candleRange = high - low
candleSolidity = candleRange == 0 ? 0 : candleBody / candleRange
marketBias = ema50 > ema100 ? "BULLISH" : "BEARISH"

//==============================================================================
// FUNGSI BANTUAN
//==============================================================================
f_pad(number) => number < 10 ? "0" + str.tostring(number) : str.tostring(number)
f_formatTimestamp(ts) => str.tostring(year(ts)) + "-" + f_pad(month(ts)) + "-" + f_pad(dayofmonth(ts)) + "T" + f_pad(hour(ts)) + ":" + f_pad(minute(ts)) + ":" + f_pad(second(ts)) + ".000Z"
f_naToNull(value) => na(value) ? "null" : str.tostring(value)

//==============================================================================
// LOGIKA UTAMA
//==============================================================================

// --- BAGIAN 1: MENANGKAP SINYAL ---
bool isSignalCandle = time == i_entryTimestamp and not isInTrade and not isSignalTriggered

if isSignalCandle and not na(ema100[CANDLE_HISTORY_COUNT])
    isSignalTriggered := true
    signalTimestamp := time
    tradeType := i_tradeType
    
    signal_ema9_current := ema9
    signal_ema9_prev := ema9[1]
    signal_ema50 := ema50
    signal_ema100 := ema100
    signal_current_candle_close := close
    signal_prev_candle_close := close[1]
    signal_bias := marketBias
    
    string solidity_array_string = "["
    string direction_array_string = "["
    for i = 1 to CANDLE_HISTORY_COUNT
        float  prevSolidity  = candleSolidity[i]
        string prevDirection = candleDirection[i]
        solidity_array_string := solidity_array_string + f_naToNull(prevSolidity)
        if i < CANDLE_HISTORY_COUNT
            solidity_array_string := solidity_array_string + ","
        direction_array_string := direction_array_string + '"' + prevDirection + '"'
        if i < CANDLE_HISTORY_COUNT
            direction_array_string := direction_array_string + ","
    solidity_array_string := solidity_array_string + "]"
    direction_array_string := direction_array_string + "]"

    signal_solidity  := solidity_array_string
    signal_direction := direction_array_string

// --- BAGIAN 2: EKSEKUSI TRADE & MANAJEMEN ---
if isSignalTriggered[1] and not isInTrade
    isInTrade := true
    entryBarIndex := bar_index
    entryPrice := open 
    tpLevel := entryPrice * (1 + (tradeType == "LONG" ? i_tpPercent : -i_tpPercent) / 100)
    slLevel := entryPrice * (1 - (tradeType == "LONG" ? i_slPercent : -i_slPercent) / 100)
    isSignalTriggered := false

plotshape(isInTrade and not isInTrade[1] and tradeType == "LONG", "Entry Long", shape.triangleup, location.belowbar, color.new(color.green, 0), size=size.small, text="E")
plotshape(isInTrade and not isInTrade[1] and tradeType == "SHORT", "Entry Short", shape.triangledown, location.abovebar, color.new(color.red, 0), size=size.small, text="E")

// --- BAGIAN 3: LOGIKA EXIT ---
if isInTrade
    line.new(x1=entryBarIndex, y1=tpLevel, x2=bar_index, y2=tpLevel, color=color.new(color.green, 50), style=line.style_dashed)
    line.new(x1=entryBarIndex, y1=slLevel, x2=bar_index, y2=slLevel, color=color.new(color.red, 50), style=line.style_dashed)
    
    float exitPrice = na
    float plPercent = na
    bool isExit = false
    
    if tradeType == "LONG"
        if high >= tpLevel
            isExit    := true, exitPrice := tpLevel, plPercent := i_tpPercent
        else if low <= slLevel
            isExit    := true, exitPrice := slLevel, plPercent := -i_slPercent
    else // SHORT
        if low <= tpLevel
            isExit    := true, exitPrice := tpLevel, plPercent := i_tpPercent
        else if high >= slLevel
            isExit    := true, exitPrice := slLevel, plPercent := -i_slPercent
            
    if isExit
        // --- PERUBAHAN DIMULAI DI SINI: Format string JSON dirapikan ---
        string finalJson = 
         '[\n' +
         '    {\n' +
         '        "id": ' + str.tostring(time) + ',\n' +
         '        "instrumentId": "' + syminfo.tickerid + '",\n' +
         '        "type": "' + tradeType + '",\n' +
         '        "entryTimestamp": "' + f_formatTimestamp(signalTimestamp) + '",\n' +
         '        "entryPrice": ' + str.tostring(entryPrice) + ',\n' +
         '        "entryReason": "Manual Entry",\n' +
         '        "status": "CLOSED",\n' +
         '        "exitPrice": ' + str.tostring(exitPrice) + ',\n' +
         '        "pl_percent": ' + str.tostring(plPercent) + ',\n' +
         '        "entry_snapshot": {\n' +
         '            "ema9_current": ' + f_naToNull(signal_ema9_current) + ',\n' +
         '            "ema9_prev": ' + f_naToNull(signal_ema9_prev) + ',\n' +
         '            "ema50": ' + f_naToNull(signal_ema50) + ',\n' +
         '            "ema100": ' + f_naToNull(signal_ema100) + ',\n' +
         '            "current_candle_close": ' + f_naToNull(signal_current_candle_close) + ',\n' +
         '            "prev_candle_close": ' + f_naToNull(signal_prev_candle_close) + ',\n' +
         '            "bias": "' + signal_bias + '",\n' +
         '            "pre_entry_candle_solidity": ' + signal_solidity + ',\n' +
         '            "pre_entry_candle_direction": ' + signal_direction + ',\n' +
         '            "funding_rate": null\n' + // Catatan: funding_rate di-hardcode sebagai null karena tidak ada inputnya.
         '        },\n' +
         '        "exitTimestamp": "' + f_formatTimestamp(time) + '"\n' +
         '    }\n' +
         ']'
        // --- PERUBAHAN SELESAI ---
        
        label.new(x=bar_index, y=high, text=finalJson, color=color.new(color.blue, 20), textcolor=color.white, style=label.style_label_down, yloc=yloc.price, size=size.normal, textalign=text.align_left)
        
        isInTrade := false
