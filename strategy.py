"flow_bullish_votes": int(flow_data["bullish_votes"]),
        "flow_bearish_votes": int(flow_data["bearish_votes"]),
        "trend_direction": trend_direction,
        "structure_trend": structure_data["structure_trend"],
        "market_structure": structure_data,
        "last_confirmed_top": structure_data["last_top"],
        "previous_confirmed_top": structure_data["previous_top"],
        "last_confirmed_bottom": structure_data["last_bottom"],
        "previous_confirmed_bottom": structure_data["previous_bottom"],
        "trend_slope_atr": round(trend_data["slope_atr"], 4),
        "trend_component_strength": int(trend_data["strength"]),
        "direction_trend_vote": trend_direction,
        "direction_authority": "TREND_EMA100_STRUCTURE",
        "structure_direction": structure_direction,
        "direction_votes": direction_votes,
        "direction_conflict": bool(direction_conflict),
        "signal_hierarchy": "DIRECTION>CONFIRMATION>ENTRY>PROTECTION",
        "flow_conflict_block": bool(flow_conflict),
        "direction_bullish_votes": int(direction_bullish_votes),
        "direction_bearish_votes": int(direction_bearish_votes),
        "direction_vote_count": int(len(direction_votes)),
        "direction_strength": int(direction_strength),
        "trend_strength": trend_strength,
        "sideways_market": sideways,
        "atr_value": round(atr_value, 8),
        "atr_median_50": round(atr_median, 8),
        "atr_relative_ratio": round(atr_ratio, 3),
        "volatility_too_low": volatility_too_low,
        "volatility_spike": volatility_spike,
        "entry_filter_passed": not bool(entry_filter_reasons),
        "entry_filter_reasons": entry_filter_reasons,
        "entry_context_warnings": entry_context_warnings,
        "ema_spread_atr": round(ema_spread_atr, 3),
        "ema_slope_atr": round(ema_slope_atr, 3),
        "ema9": round(ma9_value, 6),
        "ema21": round(ma21_value, 6),
        "ema50": round(ma50_value, 6),
        "ema100": round(ma100_value, 6),
        "ema_bull_alignment": bool(ema_bull_alignment),
        "ema_bear_alignment": bool(ema_bear_alignment),
        "ema_retest_call": bool(ema_retest_call),
        "ema_retest_put": bool(ema_retest_put),
        "ema9_pullback_seen": bool(ema9_pullback_seen),
        "ema9_recovery_confirmed": bool(ema9_recovery_confirmed),
        "rsi_kdj_overbought_call": bool(rsi_kdj_overbought_call),
        "overbought_call_block": bool(overbought_call_block),
        "ema9_pullback_lookback": int(ema9_pullback_lookback),
        "ema_cross_up_recent": recent_cross_up is not None,
        "ema_cross_down_recent": recent_cross_down is not None,
        "ema_touch_count_call": int(touch_count_call),
        "ema_touch_count_put": int(touch_count_put),
        "ema_three_touches_call": bool(three_touches_call),
        "ema_three_touches_put": bool(three_touches_put),
        "ema_confirmation": ema_direction,
        "ema100_direction": ema100_direction,
        "ema100_filter": ema100_direction,
        "kdj_k": round(k_now, 2),
        "kdj_d": round(d_now, 2),
        "kdj_j": round(j_now, 2),
        "kdj_direction": kdj_direction,
        "macd": round(float(macd_line.iloc[i]), 6),
        "macd_signal": round(float(macd_signal.iloc[i]), 6),
        "macd_status": (
            "COMPRA" if macd_cross_up else
            "VENDA" if macd_cross_down else
            "ALTA" if macd_bull else
            "BAIXA"
        ),
        "rsi": round(rsi2_value, 2),
        "rsi2": round(rsi2_value, 2),
        "rsi2_status": (
            "SOBREVENDA" if rsi2_value <= 10 else
            "SOBRECOMPRA" if rsi2_value >= 90 else
            "NEUTRO"
        ),
        "rsi2_period": 2,
        "rsi2_oversold": 10,
        "rsi2_overbought": 90,
        "rsi_status": (
            "SOBRECOMPRA" if rsi2_value >= 90 else
            "SOBREVENDA" if rsi2_value <= 10 else
            "COMPRA" if rsi_bull else
            "VENDA" if rsi_bear else
            "NEUTRO"
        ),
        "rsi_period": 2,
        "rsi_overbought": 90,
        "rsi_oversold": 10,
        "rsi_kdj_call_block_threshold": 90,
        "kdj_call_block_threshold": 95,
        "adx": round(adx_value, 2),
        "adx_status": (
            "FORTE" if adx_value >= 25 else
            "MODERADO" if adx_value >= 20 else
            "FRACO"
        ),
        "di_plus": round(dp, 2),
        "di_minus": round(dm, 2),
        "didi_status": (
            "AGULHADA COMPRA" if di_cross_up else
            "AGULHADA VENDA" if di_cross_down else
            "DI+ > DI-" if di_bull else
            "DI- > DI+"
        ),
        "entry_confirmations": int(confirm_count),
        "brain_score": int(brain_score),
        "signal_mode": signal_mode if direction != "NEUTRO" else "NONE",
        "mean_reversion_direction": mean_reversion_direction,
        "mean_reversion_distance_atr": round(distance_from_ema21_atr, 3),
        "mean_reversion_threshold_atr": 1.0,
        "mean_reversion_confirmations": int(mean_reversion_confirmations),
        "fibonacci_zone": fib_zone,
        "fibonacci_direction": fib_direction,
        "fibonacci_confirmation": bool(fib_confirmation),
        "fibonacci_position": round(fib_position, 4),
        "fibonacci_38_2": round(float(fib_levels.get("38.2", 0.0)), 6),
        "fibonacci_50": round(float(fib_levels.get("50.0", 0.0)), 6),
        "fibonacci_61_8": round(float(fib_levels.get("61.8", 0.0)), 6),
        "fibonacci_tolerance": round(fib_tolerance, 6),
        "fibonacci_lookback": int(fib_window),
        "absorption_source": "ESTIMADO_OHLCV",
        **diagnostics,
    }

    return {
        "ready": True,
        "direction": direction,
        "confidence": confidence,
