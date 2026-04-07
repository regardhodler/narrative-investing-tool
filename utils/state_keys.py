"""Central registry of all session_state keys used across the app.

Usage:
    from utils.state_keys import SK
    val = st.session_state.get(SK.fear_composite)
    st.session_state[SK.regime_context] = result

Existing code using raw string literals ("_regime_context") still works — this is
additive. Use SK.* in new code so typos are caught at import time, not at runtime.
"""


class SK:
    # ── Regime ────────────────────────────────────────────────────────────────
    regime_context          = "_regime_context"
    regime_context_ts       = "_regime_context_ts"
    rp_plays_result         = "_rp_plays_result"
    rp_plays_last_tier      = "_rp_plays_last_tier"

    # ── Fed / Rate Path ────────────────────────────────────────────────────────
    fed_funds_rate          = "_fed_funds_rate"
    dominant_rate_path      = "_dominant_rate_path"
    rate_path_probs         = "_rate_path_probs"
    rate_path_probs_ts      = "_rate_path_probs_ts"
    fed_plays_result        = "_fed_plays_result"
    fed_plays_result_ts     = "_fed_plays_result_ts"
    fed_plays_engine        = "_fed_plays_engine"
    fed_plays_tier          = "_fed_plays_tier"
    chain_narration         = "_chain_narration"
    chain_narration_engine  = "_chain_narration_engine"
    calibrated_rate_probs   = "_calibrated_rate_probs"

    # ── Tail Risk ──────────────────────────────────────────────────────────────
    custom_swans            = "_custom_swans"
    custom_swans_ts         = "_custom_swans_ts"
    doom_briefing           = "_doom_briefing"
    doom_briefing_ts        = "_doom_briefing_ts"
    doom_briefing_engine    = "_doom_briefing_engine"

    # ── Whale / Institutional ──────────────────────────────────────────────────
    whale_summary           = "_whale_summary"
    whale_summary_ts        = "_whale_summary_ts"
    whale_summary_engine    = "_whale_summary_engine"
    institutional_bias      = "_institutional_bias"
    insider_net_flow        = "_insider_net_flow"
    congress_bias           = "_congress_bias"
    activism_digest         = "_activism_digest"
    activism_digest_ts      = "_activism_digest_ts"
    activism_digest_engine  = "_activism_digest_engine"

    # ── Discovery ──────────────────────────────────────────────────────────────
    plays_result            = "_plays_result"
    plays_engine            = "_plays_engine"
    macro_fit_results       = "_macro_fit_results"
    trending_narratives     = "_trending_narratives"
    trending_narratives_ts  = "_trending_narratives_ts"
    trending_narratives_tf  = "_trending_narratives_tf"
    auto_trending_groups    = "_auto_trending_groups"
    auto_trending_groups_ts = "_auto_trending_groups_ts"
    auto_trending_groups_engine = "_auto_trending_groups_engine"
    price_momentum          = "_price_momentum"

    # ── Current Events ─────────────────────────────────────────────────────────
    current_events_digest    = "_current_events_digest"
    current_events_digest_ts = "_current_events_digest_ts"
    current_events_engine    = "_current_events_engine"

    # ── Portfolio ──────────────────────────────────────────────────────────────
    portfolio_analysis         = "_portfolio_analysis"
    portfolio_analysis_ts      = "_portfolio_analysis_ts"
    portfolio_analysis_engine  = "_portfolio_analysis_engine"
    portfolio_risk_snapshot    = "_portfolio_risk_snapshot"
    portfolio_risk_snapshot_ts = "_portfolio_risk_snapshot_ts"
    risk_matrix_interpretation = "_risk_matrix_interpretation"
    qir_earnings_risk          = "_qir_earnings_risk"
    qir_earnings_risk_ts       = "_qir_earnings_risk_ts"

    # ── Tactical / Signals ─────────────────────────────────────────────────────
    tactical_context         = "_tactical_context"
    tactical_context_ts      = "_tactical_context_ts"
    tactical_analysis        = "_tactical_analysis"
    tactical_analysis_ts     = "_tactical_analysis_ts"
    options_flow_context     = "_options_flow_context"
    options_flow_context_ts  = "_options_flow_context_ts"
    options_sentiment        = "_options_sentiment"
    unusual_activity_sentiment = "_unusual_activity_sentiment"
    macro_synopsis           = "_macro_synopsis"
    macro_synopsis_ts        = "_macro_synopsis_ts"
    macro_synopsis_engine    = "_macro_synopsis_engine"

    # ── Sector ─────────────────────────────────────────────────────────────────
    sector_regime_digest        = "_sector_regime_digest"
    sector_regime_digest_ts     = "_sector_regime_digest_ts"
    sector_regime_digest_engine = "_sector_regime_digest_engine"

    # ── EDGAR ──────────────────────────────────────────────────────────────────
    filing_digest           = "_filing_digest"
    factor_analysis         = "_factor_analysis"
    factor_analysis_ts      = "_factor_analysis_ts"
    factor_analysis_engine  = "_factor_analysis_engine"

    # ── Free sentiment data ────────────────────────────────────────────────────
    fear_greed              = "_fear_greed"
    fear_greed_ts           = "_fear_greed_ts"
    aaii_sentiment          = "_aaii_sentiment"
    aaii_sentiment_ts       = "_aaii_sentiment_ts"
    vix_curve               = "_vix_curve"
    vix_curve_ts            = "_vix_curve_ts"
    stocktwits_digest       = "_stocktwits_digest"
    stocktwits_digest_ts    = "_stocktwits_digest_ts"

    # ── Quantified signal scores ───────────────────────────────────────────────
    stress_zscore           = "_stress_zscore"
    whale_flow_score        = "_whale_flow_score"
    events_sentiment_score  = "_events_sentiment_score"
    canary_score            = "_canary_score"
    fear_composite          = "_fear_composite"

    # ── Data quality ───────────────────────────────────────────────────────────
    data_quality            = "_data_quality"
    data_quality_ts         = "_data_quality_ts"

    # ── Forecast tracker ───────────────────────────────────────────────────────
    forecast_log            = "_forecast_log"
    forecast_log_ts         = "_forecast_log_ts"

    # ── Error surface (set by @api_call decorator) ─────────────────────────────
    api_errors              = "_api_errors"

    # ── Internal session flags (not persisted to cache) ───────────────────────
    signals_cache_loaded    = "_signals_cache_loaded"
    fred_cache_warmed       = "_fred_cache_warmed"
    signals_gist_saved_at   = "_signals_gist_saved_at"
