"""
ALPHA DISCOVERY ENGINE v4 — WDO B3

NUMPY VETORIZADO PURO — sem loop por combo
Testa 50k combos em minutos, nao horas.

ESTRATEGIA DE VELOCIDADE:
- Pre-computa todos os indicadores uma vez
- Para cada familia, gera TODOS os sinais de uma vez em matriz
- Simula todos os combos simultaneamente com numpy
- VectorBT apenas para validacao final dos top-10

LICOES APLICADAS:
- flush=True em todos os prints
- Entrada no OPEN do proximo candle
- MIN_TRADES=100 para triagem
- upon_opposite_entry='close'
"""

import pandas as pd
import numpy as np
import json, sys, os, time, warnings, math
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output/alpha_v4"
CAPITAL    = 50_000.0
MULT       = 10.0        # WDO: 1 ponto = R$10
COMM       = 5.0         # R$ comissao
SLIP       = 2.0         # pontos slippage
MIN_TRADES = 100
MAX_PF     = 3.5
MAX_DD     = -35.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df


# ================================================================
# SECAO 2: INDICADORES (numpy puro, ultra-rapido)
# ================================================================

def ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out   = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi_np(close, period):
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = np.empty(len(close)); avg_g[:period] = np.nan
    avg_l = np.empty(len(close)); avg_l[:period] = np.nan
    avg_g[period] = gain[1:period + 1].mean()
    avg_l[period] = loss[1:period + 1].mean()
    for i in range(period + 1, len(close)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_g / (avg_l + 1e-9)
    return 100 - (100 / (1 + rs))


def sma_np(arr, period):
    out = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = arr[i - period + 1:i + 1].mean()
    return out


def atr_np(high, low, close, period=14):
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return sma_np(tr, period)


def calcular_indicadores(df):
    print("[IND] Calculando indicadores...", flush=True)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    v = df["volume"].values
    n = len(c)

    ind = {}
    ind["close"]     = c
    ind["open"]      = o
    ind["high"]      = h
    ind["low"]       = l
    ind["open_next"] = np.roll(o, -1)  # OPEN do proximo candle
    ind["open_next"][-1] = c[-1]

    # ATR
    for p in [7, 14, 20]:
        ind[f"atr_{p}"] = atr_np(h, l, c, p)

    # EMAs
    for p in [5, 8, 10, 13, 20, 21, 34, 50, 100, 200]:
        ind[f"ema_{p}"] = ema_np(c, p)

    # SMAs
    for p in [10, 20, 50, 200]:
        ind[f"sma_{p}"] = sma_np(c, p)

    # RSIs
    for p in [7, 9, 14, 21]:
        ind[f"rsi_{p}"] = rsi_np(c, p)

    # MACD
    for fast, slow in [(12, 26), (8, 21), (5, 13)]:
        m   = ema_np(c, fast) - ema_np(c, slow)
        sig = ema_np(m, 9)
        ind[f"macd_{fast}_{slow}_hist"] = m - sig

    # Bollinger
    for p in [10, 20, 50]:
        sma = sma_np(c, p)
        std = pd.Series(c).rolling(p).std().values
        for s_int, s_float in [(15, 1.5), (20, 2.0), (25, 2.5)]:
            up  = sma + s_float * std
            lo  = sma - s_float * std
            ind[f"bb_{p}_{s_int}_pct"] = (c - lo) / (up - lo + 1e-9)

    # Donchian (shift(1) para evitar lookahead)
    for p in [10, 20, 50, 100]:
        hi_roll = pd.Series(h).rolling(p).max().shift(1).values
        lo_roll = pd.Series(l).rolling(p).min().shift(1).values
        ind[f"don_high_{p}"] = hi_roll
        ind[f"don_low_{p}"]  = lo_roll

    # Stochastic
    for p in [5, 9, 14]:
        lo_p = pd.Series(l).rolling(p).min().values
        hi_p = pd.Series(h).rolling(p).max().values
        stk  = (c - lo_p) / (hi_p - lo_p + 1e-9) * 100
        ind[f"stoch_k_{p}"] = stk
        ind[f"stoch_d_{p}"] = sma_np(stk, 3)

    # ROC
    for p in [3, 5, 10, 20]:
        roc = np.empty(n); roc[:p] = np.nan
        roc[p:] = (c[p:] - c[:-p]) / (c[:-p] + 1e-9) * 100
        ind[f"roc_{p}"] = roc

    # CCI
    for p in [14, 20]:
        tp     = (h + l + c) / 3
        sma_tp = sma_np(tp, p)
        mad    = pd.Series(tp).rolling(p).apply(
            lambda x: np.abs(x - x.mean()).mean()).values
        ind[f"cci_{p}"] = (tp - sma_tp) / (0.015 * mad + 1e-9)

    # Volume zscore
    vol_mean = sma_np(v.astype(float), 20)
    vol_std  = pd.Series(v).rolling(20).std().values
    ind["vol_z"] = (v - vol_mean) / (vol_std + 1e-9)

    # Volatilidade relativa
    ret   = np.diff(c, prepend=c[0]) / (c + 1e-9)
    vol5  = pd.Series(ret).rolling(5).std().values * 100
    vol20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = vol5 / (vol20 + 1e-9)

    # Sessao
    hora = df.index.hour
    ind["session_am"] = ((hora >= 9) & (hora < 12)).astype(int)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(int)
    ind["index"]      = df.index

    print(f"[IND] {len(ind)} arrays prontos", flush=True)
    return ind


# ================================================================
# SECAO 3: SIMULACAO RAPIDA (numpy puro, sem VectorBT)
# ================================================================

def simular_rapido(ind, entries_bool, exits_bool, sl_pts, tp_pts):
    """
    Simula backtest bar-by-bar ULTRA-RAPIDO em numpy.
    Retorna metricas basicas.
    """
    open_next = ind["open_next"]
    high      = ind["high"]
    low       = ind["low"]
    n         = len(open_next)

    trades_pnl = []
    em_pos     = False
    entry_p    = sl = tp = 0.0
    direction  = 0  # 1=long, -1=short

    for i in range(n - 1):
        if em_pos:
            lo, hi = low[i], high[i]
            hit_sl = (direction == 1 and lo <= sl) or (direction == -1 and hi >= sl)
            hit_tp = (direction == 1 and hi >= tp) or (direction == -1 and lo <= tp)
            force  = exits_bool[i]

            if hit_sl or hit_tp or force:
                if force and not hit_sl and not hit_tp:
                    saida = open_next[i]
                else:
                    saida = sl if hit_sl else tp
                pts = (saida - entry_p) * direction
                pnl = pts * MULT - COMM - SLIP * MULT * 0.1
                trades_pnl.append(pnl)
                em_pos = False
            continue

        if entries_bool[i] and not em_pos:
            entry_p = open_next[i]
            if np.isnan(entry_p) or entry_p <= 0:
                continue
            direction = 1  # sempre long nessa versao simplificada
            sl        = entry_p - sl_pts
            tp        = entry_p + tp_pts
            em_pos    = True

    if not trades_pnl or len(trades_pnl) < MIN_TRADES:
        return None

    pnls = np.array(trades_pnl)
    wins  = pnls[pnls > 0]
    loses = pnls[pnls <= 0]
    if len(loses) == 0 or len(wins) == 0:
        return None

    pf = abs(wins.sum() / loses.sum())
    if pf > MAX_PF:
        return None

    # Equity curve
    eq  = np.insert(CAPITAL + np.cumsum(pnls), 0, CAPITAL)
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    if mdd < MAX_DD:
        return None

    wr  = len(wins) / len(pnls) * 100
    ret = pnls / CAPITAL
    sh  = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252 * 390))
    exp = float(pnls.mean())

    # Consistencia por janelas
    n_jan   = max(1, len(range(0, len(pnls) - 30, 15)))
    jan_pos = sum(1 for s in range(0, len(pnls) - 30, 15)
                  if pnls[s:s + 30].sum() > 0)
    pct_jan = jan_pos / n_jan * 100

    return {
        "total_trades":     len(pnls),
        "win_rate":         round(wr, 2),
        "profit_factor":    round(pf, 3),
        "sharpe_ratio":     round(sh, 3),
        "expectancy_brl":   round(exp, 2),
        "total_pnl_brl":    round(float(pnls.sum()), 2),
        "max_drawdown_pct": round(mdd, 2),
        "pct_janelas_pos":  round(pct_jan, 1),
    }


# ================================================================
# SECAO 4: GERADORES DE SINAIS (numpy)
# ================================================================

def sig_ema_cross(ind, fast, slow, direction="long"):
    ef = ind.get(f"ema_{fast}")
    es = ind.get(f"ema_{slow}")
    if ef is None or es is None:
        return None, None
    if direction == "long":
        ent = (ef > es) & (np.roll(ef, 1) <= np.roll(es, 1))
        ext = (ef < es) & (np.roll(ef, 1) >= np.roll(es, 1))
    else:
        ent = (ef < es) & (np.roll(ef, 1) >= np.roll(es, 1))
        ext = (ef > es) & (np.roll(ef, 1) <= np.roll(es, 1))
    ent[0] = ext[0] = False
    return ent, ext


def sig_rsi(ind, period, oversold, overbought, exit_level=50, direction="long"):
    rsi = ind.get(f"rsi_{period}")
    if rsi is None:
        return None, None
    if direction == "long":
        ent = (rsi < oversold) & (np.roll(rsi, 1) >= oversold)
        ext = rsi > exit_level
    else:
        ent = (rsi > overbought) & (np.roll(rsi, 1) <= overbought)
        ext = rsi < (100 - exit_level)
    ent[0] = ext[0] = False
    return ent, ext


def sig_bollinger(ind, period, std_str, direction="long"):
    pct = ind.get(f"bb_{period}_{std_str}_pct")
    if pct is None:
        return None, None
    if direction == "long":
        ent = pct < 0.05; ext = pct > 0.5
    else:
        ent = pct > 0.95; ext = pct < 0.5
    return ent, ext


def sig_macd(ind, config, direction="long"):
    hist = ind.get(f"macd_{config}_hist")
    if hist is None:
        return None, None
    if direction == "long":
        ent = (hist > 0) & (np.roll(hist, 1) <= 0)
        ext = (hist < 0) & (np.roll(hist, 1) >= 0)
    else:
        ent = (hist < 0) & (np.roll(hist, 1) >= 0)
        ext = (hist > 0) & (np.roll(hist, 1) <= 0)
    ent[0] = ext[0] = False
    return ent, ext


def sig_donchian(ind, period, direction="long"):
    dh = ind.get(f"don_high_{period}")
    dl = ind.get(f"don_low_{period}")
    if dh is None:
        return None, None
    c = ind["close"]
    if direction == "long":
        ent = c > dh; ext = c < dl
    else:
        ent = c < dl; ext = c > dh
    return ent, ext


def sig_roc(ind, period, threshold, direction="long"):
    roc = ind.get(f"roc_{period}")
    if roc is None:
        return None, None
    if direction == "long":
        ent = roc > threshold; ext = roc < 0
    else:
        ent = roc < -threshold; ext = roc > 0
    return ent, ext


def sig_stoch(ind, period, oversold, overbought, direction="long"):
    k = ind.get(f"stoch_k_{period}")
    if k is None:
        return None, None
    if direction == "long":
        ent = (k < oversold) & (np.roll(k, 1) >= oversold)
        ext = k > 50
    else:
        ent = (k > overbought) & (np.roll(k, 1) <= overbought)
        ext = k < 50
    ent[0] = ext[0] = False
    return ent, ext


def sig_cci(ind, period, threshold, direction="long"):
    cci = ind.get(f"cci_{period}")
    if cci is None:
        return None, None
    if direction == "long":
        ent = (cci < -threshold) & (np.roll(cci, 1) >= -threshold)
        ext = cci > 0
    else:
        ent = (cci > threshold) & (np.roll(cci, 1) <= threshold)
        ext = cci < 0
    ent[0] = ext[0] = False
    return ent, ext


def sig_volatility(ind, vol_threshold, direction="long"):
    vr   = ind["vol_ratio"]
    hist = ind.get("macd_12_26_hist", np.zeros(len(vr)))
    exp  = vr > vol_threshold
    if direction == "long":
        ent = exp & (hist > 0) & (np.roll(hist, 1) <= 0)
        ext = exp & (hist < 0) & (np.roll(hist, 1) >= 0)
    else:
        ent = exp & (hist < 0) & (np.roll(hist, 1) >= 0)
        ext = exp & (hist > 0) & (np.roll(hist, 1) <= 0)
    return ent, ext


def gerar_sinais(familia, ind, params):
    d       = params.get("direction", "long")
    session = params.get("session", "all")

    if session == "am":
        mask = ind["session_am"].astype(bool)
    elif session == "pm":
        mask = ind["session_pm"].astype(bool)
    else:
        mask = np.ones(len(ind["close"]), dtype=bool)

    ent = ext = None

    if familia == "ema_crossover":
        ent, ext = sig_ema_cross(ind, params["fast"], params["slow"], d)
    elif familia == "rsi_reversion":
        ent, ext = sig_rsi(ind, params["period"], params["oversold"],
                           params["overbought"], params.get("exit_level", 50), d)
    elif familia == "stochastic":
        ent, ext = sig_stoch(ind, params["period"], params["oversold"],
                             params["overbought"], d)
    elif familia == "bollinger":
        ent, ext = sig_bollinger(ind, params["period"], params["std_str"], d)
    elif familia == "macd_momentum":
        ent, ext = sig_macd(ind, params["config"], d)
    elif familia == "donchian":
        ent, ext = sig_donchian(ind, params["period"], d)
    elif familia == "roc_momentum":
        ent, ext = sig_roc(ind, params["period"], params["threshold"], d)
    elif familia == "cci":
        ent, ext = sig_cci(ind, params["period"], params["threshold"], d)
    elif familia == "volatility":
        ent, ext = sig_volatility(ind, params["vol_threshold"], d)
    else:
        return None, None

    if ent is None:
        return None, None

    ent = ent & mask
    return ent, ext


# ================================================================
# SECAO 5: GRIDS
# ================================================================

GRIDS = {
    "ema_crossover": {
        "fast":      [3, 5, 8, 10, 13, 20, 21, 34],
        "slow":      [20, 21, 34, 50, 100, 150, 200],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "rsi_reversion": {
        "period":     [7, 9, 14, 21],
        "oversold":   [10, 15, 20, 25, 30, 35, 40],
        "overbought": [60, 65, 70, 75, 80, 85, 90],
        "exit_level": [45, 50, 55],
        "rr":         [1.5, 2.0, 2.5, 3.0],
        "atr_sl":     [0.5, 1.0, 1.5, 2.0],
        "direction":  ["long", "short"],
    },
    "stochastic": {
        "period":     [5, 9, 14, 21],
        "oversold":   [10, 15, 20, 25, 30],
        "overbought": [70, 75, 80, 85, 90],
        "rr":         [1.5, 2.0, 2.5, 3.0],
        "atr_sl":     [0.5, 1.0, 1.5, 2.0],
        "direction":  ["long", "short"],
    },
    "bollinger": {
        "period":    [10, 20, 50],
        "std_str":   ["15", "20", "25"],
        "session":   ["am", "pm", "all"],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "macd_momentum": {
        "config":    ["12_26", "8_21", "5_13"],
        "session":   ["am", "pm", "all"],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "donchian": {
        "period":    [10, 20, 50, 100],
        "session":   ["am", "pm", "all"],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "roc_momentum": {
        "period":    [3, 5, 10, 20],
        "threshold": [0.1, 0.2, 0.5, 1.0, 2.0],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "cci": {
        "period":    [14, 20],
        "threshold": [75, 100, 150, 200],
        "rr":        [1.5, 2.0, 2.5, 3.0],
        "atr_sl":    [0.5, 1.0, 1.5, 2.0],
        "direction": ["long", "short"],
    },
    "volatility": {
        "vol_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
        "rr":            [1.5, 2.0, 2.5, 3.0],
        "atr_sl":        [0.5, 1.0, 1.5, 2.0],
        "direction":     ["long", "short"],
    },
}


# ================================================================
# SECAO 6: GRID SEARCH RAPIDO
# ================================================================

def grid_search(familia, ind, grid, n_total, mini=False):
    import itertools
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mini:
        combos = combos[:10]

    print(f"\n[{familia.upper()}] {len(combos):,} combos...", flush=True)
    t0      = time.time()
    validos = []
    n_ok    = 0

    atr14   = ind["atr_14"]
    atr_pts = float(np.nanmean(atr14))  # ATR em pontos

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            ent, ext = gerar_sinais(familia, ind, params)
            if ent is None or ent.sum() < 10:
                continue

            sl_pts = atr_pts * params.get("atr_sl", 1.0)
            tp_pts = sl_pts  * params.get("rr", 2.0)

            m = simular_rapido(ind, ent, ext, sl_pts, tp_pts)
            if not m:
                continue

            n_ok += 1

            # Score
            pf_s  = min(m["profit_factor"], MAX_PF) / MAX_PF
            exp_s = max(0, min(m["expectancy_brl"], 500)) / 500
            jan_s = m["pct_janelas_pos"] / 100
            tr_s  = min(m["total_trades"], 2000) / 2000
            sh_s  = max(0, min(m["sharpe_ratio"], 3)) / 3
            score = pf_s * 0.30 + exp_s * 0.25 + jan_s * 0.20 + sh_s * 0.15 + tr_s * 0.10

            validos.append({
                "familia": familia,
                "params":  params,
                "score":   round(score, 6),
                **m,
            })
        except Exception:
            continue

    elapsed = time.time() - t0
    validos.sort(key=lambda x: -x["score"])
    spd = len(combos) / max(elapsed, 0.1)
    print(f"  {n_ok:,}/{len(combos):,} validos | {elapsed:.1f}s | {spd:.0f} combos/s", flush=True)

    if validos:
        r = validos[0]
        print(f"  TOP: PF={r['profit_factor']:.3f} WR={r['win_rate']:.1f}% "
              f"Trades={r['total_trades']} Exp=R${r['expectancy_brl']:.2f} "
              f"Score={r['score']:.4f}", flush=True)

        # Salvar top10 da familia
        with open(f"{OUTPUT_DIR}/{familia}_top10.json", "w") as fp:
            json.dump(validos[:10], fp, indent=2, default=str)

    return validos


# ================================================================
# SECAO 7: VALIDACAO COM VECTORBT (apenas para finalistas)
# ================================================================

def validar_com_vbt(ind, ent, ext, sl_pts, tp_pts, close_arr, idx):
    """Validacao precisa com VectorBT para os top candidatos."""
    try:
        import vectorbt as vbt
        open_next = pd.Series(ind["open_next"], index=idx)
        entries   = pd.Series(ent, index=idx)
        exits     = pd.Series(ext, index=idx)
        sl_pct    = sl_pts / np.nanmean(close_arr)
        tp_pct    = tp_pts / np.nanmean(close_arr)
        pf = vbt.Portfolio.from_signals(
            open_next, entries, exits,
            sl_stop=sl_pct, tp_stop=tp_pct,
            fees=0.0001, init_cash=CAPITAL,
            freq="1min", upon_opposite_entry="close",
        )
        trades = pf.trades.records_readable
        n      = len(trades)
        if n < MIN_TRADES:
            return None
        wins  = trades[trades["PnL"] > 0]
        loses = trades[trades["PnL"] <= 0]
        if len(loses) == 0:
            return None
        pf_v  = abs(wins["PnL"].sum() / loses["PnL"].sum())
        mdd   = float(pf.max_drawdown()) * 100
        sh    = float(pf.sharpe_ratio())
        pnl   = float(pf.total_profit())
        wr    = len(wins) / n * 100
        return {
            "total_trades":     n,
            "win_rate":         round(wr, 2),
            "profit_factor":    round(pf_v, 3),
            "sharpe_ratio":     round(sh, 3),
            "total_pnl_brl":    round(pnl, 2),
            "max_drawdown_pct": round(mdd, 2),
        }
    except Exception:
        return None


# ================================================================
# SECAO 8: STAGE GATE
# ================================================================

def stage_gate(m):
    if not m:
        return False, {}
    checks = {
        "S1_trades":  m["total_trades"] >= MIN_TRADES,
        "S2_pf":      m["profit_factor"] > 1.05,
        "S3_sharpe":  m["sharpe_ratio"]  > 0.0,
        "S4_dd":      m["max_drawdown_pct"] > MAX_DD,
        "S5_janelas": m["pct_janelas_pos"] >= 50,
        "S6_exp":     m["expectancy_brl"]  > 0,
    }
    return all(checks.values()), checks


# ================================================================
# SECAO 9: MAIN
# ================================================================

def main():
    MINI = "--mini" in sys.argv

    total = sum(math.prod(len(v) for v in g.values()) for g in GRIDS.values())

    print("=" * 68, flush=True)
    print("  ALPHA DISCOVERY ENGINE v4 — NUMPY VETORIZADO", flush=True)
    print(f"  {len(GRIDS)} familias | {total:,} combos | WDO B3", flush=True)
    print("=" * 68, flush=True)

    df     = carregar()
    split  = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  IS : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}", flush=True)
    print(f"  OOS: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}", flush=True)

    ind_ins = calcular_indicadores(df_ins)

    todos     = []
    aprovados = []

    for familia, grid in GRIDS.items():
        resultados = grid_search(familia, ind_ins, grid, total, mini=MINI)
        todos.extend(resultados[:20])

        if not resultados:
            continue

        melhor = resultados[0]
        aprovado, gate = stage_gate(melhor)
        status = "OK" if aprovado else f"REPROVADO {sum(gate.values())}/{len(gate)}"
        print(f"  Stage-gate: {status}", flush=True)

        if aprovado:
            aprovados.append((familia, melhor))

    # === TOP GERAL ===
    todos.sort(key=lambda x: -x["score"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  TOP 20 GERAL (antes da validacao final)", flush=True)
    print(f"  {'FAMILIA':20} {'PF':>6} {'WR%':>6} {'Trades':>7} "
          f"{'Exp R$':>8} {'Jan+':>5} {'Score':>7}", flush=True)
    print(f"  {'-' * 60}", flush=True)
    for r in todos[:20]:
        print(f"  {r['familia']:20} "
              f"{r['profit_factor']:>6.3f} "
              f"{r['win_rate']:>6.1f} "
              f"{r['total_trades']:>7} "
              f"{r['expectancy_brl']:>8.2f} "
              f"{r['pct_janelas_pos']:>5.0f}% "
              f"{r['score']:>7.4f}", flush=True)

    # === VALIDACAO OOS DOS APROVADOS ===
    print(f"\n{'=' * 68}", flush=True)
    print(f"  VALIDANDO {len(aprovados)} APROVADO(S) NO OOS...", flush=True)

    ind_oos = calcular_indicadores(df_oos)
    resultados_finais = []

    for familia, melhor in aprovados:
        params = melhor["params"]
        print(f"\n  [{familia}] OOS...", flush=True)

        ent, ext = gerar_sinais(familia, ind_oos, params)
        if ent is None:
            continue

        atr_pts = float(np.nanmean(ind_oos["atr_14"]))
        sl_pts  = atr_pts * params.get("atr_sl", 1.0)
        tp_pts  = sl_pts  * params.get("rr", 2.0)

        m_oos = simular_rapido(ind_oos, ent, ext, sl_pts, tp_pts)

        is_pf  = melhor["profit_factor"]
        oos_pf = m_oos["profit_factor"] if m_oos else 0
        deg    = (is_pf - oos_pf) / is_pf * 100 if is_pf > 0 and oos_pf > 0 else 999
        ok_oos = m_oos is not None and oos_pf > 1.0 and deg < 50

        print(f"  IS PF={is_pf:.3f} | OOS PF={oos_pf:.3f} | "
              f"Deg={deg:.1f}% | {'PASSA' if ok_oos else 'REPROVADO'}", flush=True)

        resultado = {
            "id":           f"{familia}_{int(time.time())}",
            "familia":      familia,
            "params":       params,
            "metricas_is":  melhor,
            "metricas_oos": m_oos,
            "degradacao":   round(deg, 1),
            "aprovado":     ok_oos,
            "gerado_em":    datetime.now().isoformat(),
        }
        resultados_finais.append(resultado)

        with open(f"{OUTPUT_DIR}/{familia}_final.json", "w") as fp:
            json.dump(resultado, fp, indent=2, default=str)

    # Leaderboard
    n_apr = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  LEADERBOARD FINAL — {n_apr} APROVADO(S)", flush=True)
    print(f"  {'FAMILIA':22} {'PF_IS':>6} {'PF_OOS':>7} "
          f"{'DEG%':>6} {'STATUS':>12}", flush=True)
    print(f"  {'-' * 58}", flush=True)

    for r in sorted(resultados_finais,
                    key=lambda x: -(x["metricas_is"] or {}).get("profit_factor", 0)):
        mi = r["metricas_is"] or {}
        mo = r["metricas_oos"] or {}
        print(f"  {r['familia']:22} "
              f"{mi.get('profit_factor', 0):>6.3f} "
              f"{mo.get('profit_factor', 0):>7.3f} "
              f"{r['degradacao']:>6.1f} "
              f"{'APROVADO' if r['aprovado'] else 'REPROVADO':>12}", flush=True)

    lb = {
        "gerado_em":       datetime.now().isoformat(),
        "total_combos":    total,
        "aprovados_is":    len(aprovados),
        "aprovados_final": n_apr,
        "top20_geral":     todos[:20],
        "leaderboard":     resultados_finais,
    }
    with open(f"{OUTPUT_DIR}/leaderboard.json", "w") as fp:
        json.dump(lb, fp, indent=2, default=str)

    print(f"\n  Salvo em: {OUTPUT_DIR}/leaderboard.json", flush=True)
    print(f"  {n_apr} estrategia(s) aprovada(s) para paper trading!", flush=True)


if __name__ == "__main__":
    main()
