    # ...
    # Hitung EMA series lagi di sini untuk chart
    ema_length_chart = pair_config.get('ema_length', 200)
    close_prices_chart = [c['close'] for c in candles_for_chart_display if c and 'close' in c and c['close'] is not None]
    ema_series_for_chart = []
    if len(close_prices_chart) >= ema_length_chart:
        ema_pd_series_chart = pd.Series(close_prices_chart).ewm(span=ema_length_chart, adjust=False).mean()
        # Cocokkan dengan timestamp candle yang ada di ohlc_data_points
        timestamps_ohlc = [dp['x'] for dp in ohlc_data_points]
        ema_series_for_chart = [{'x': ts, 'y': val} for ts, val in zip(timestamps_ohlc[-len(ema_pd_series_chart):], ema_pd_series_chart) if val is not None]

    return {
        "ohlc": ohlc_data_points,
        "ema_line": ema_series_for_chart, # Data EMA baru
        # ... annotations lama ...
        "pair_name": pair_config.get('pair_name', pair_id_to_display),
        "last_updated_tv": candles_for_chart_display[-1]['timestamp'].timestamp() * 1000 if candles_for_chart_display else None
    }
