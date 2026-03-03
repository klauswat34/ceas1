import os
from datetime import datetime, timedelta
import numpy as np
import json
import traceback
from sklearn.decomposition import PCA


import time as _time
time = _time

import pandas as pd
import joblib
from kiteconnect import KiteConnect
import os
import pytz


VWAP_BASE = 1050118598.8833321

  # your last notebook cumulative value



IST = pytz.timezone("Asia/Kolkata")

TELEGRAM_BOT_TOKEN = "8527999481:AAE322YF6z9mIfmY6WAUP-FNkPIzq8bxch4"
TELEGRAM_CHAT_ID = 6716775488

#KITE_API_KEY = "m90wl78wn764217q"
#KITE_API_SECRET = "foczp8ktwodnbmdsn1ri9avj7m58ad55"

KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
#TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
#TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "LT", "ITC", "BHARTIARTL", "SBIN", "AXISBANK",
    "HINDUNILVR", "KOTAKBANK", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "NTPC", "POWERGRID", "ULTRACEMCO",
    "M&M",
    "ASIANPAINT", "TITAN",
    "ADANIENT", "BAJAJ-AUTO", "DIVISLAB",
    "JSWSTEEL", "HCLTECH", "ONGC",
    "TATASTEEL", "GRASIM",
]

LOOKBACK  = 2
MAX_HOLD  = 8

STOP_R    = 1.2
TARGET_R  = 2.35
COST      = 2.0

LOOKBACK_BARS = 30
ADV_WINDOW = 20
BREADTH_THRESHOLD = 0.6

KITE_LATENCY_BUFFER = 35
INTERVAL = "5minute"
INTERVAL_MIN = 5
EPS = 1e-9

MAX_ROWS = 1500
GLOBAL_BUFFER = pd.read_csv(
    "seed.csv",
    parse_dates=True,
    index_col=0
)

# 🔥 FORCE TZ-NAIVE INDEX (CRITICAL FIX)
GLOBAL_BUFFER.index = pd.to_datetime(GLOBAL_BUFFER.index)

if GLOBAL_BUFFER.index.tz is not None:
    GLOBAL_BUFFER.index = GLOBAL_BUFFER.index.tz_localize(None)

LOG_FILE = "logs/live_predictions.csv"
import requests


 



def append_log(ts, system, data):

    os.makedirs("logs", exist_ok=True)

    row = {
        "timestamp": ts,
        "system": system,
        **data,
    }

    df = pd.DataFrame([row])

    write_header = not os.path.exists(LOG_FILE)

    df.to_csv(
        LOG_FILE,
        mode="a",
        header=write_header,
        index=False,
    )




INTERVAL = "5minute"
INTERVAL_MIN = 5
MAX_ROWS = 1500

LAST_EXECUTED_CANDLE = None

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg
            },
            timeout=10
        )

        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text)

    except Exception as e:
        print("Telegram failed:", e)



def kite_login():
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)
    return kite

def market_open():
    now = datetime.now(IST)
 #   if now.weekday() >= 5:
 #       return False

    start = datetime.strptime("09:10", "%H:%M").time()
    end   = datetime.strptime("23:25", "%H:%M").time()

    return start <= now.time() <= end


def sleep_to_next_bar():

    now = datetime.now(IST)

    delta = INTERVAL_MIN - (now.minute % INTERVAL_MIN)
    if delta == 0:
        delta = INTERVAL_MIN

    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=delta)

    sleep_seconds = (nxt - now).total_seconds()

    time.sleep(max(1, sleep_seconds))

    # small buffer so Kite finalizes candle
    time.sleep(30)


def fetch_latest_nifty_candle(kite, token):

    now = datetime.now(IST)
    start = now - timedelta(minutes=30)

    #minutes=15
    #days=1

    candles = kite.historical_data(
        instrument_token=token,
        from_date=start,
        to_date=now,
        interval=INTERVAL,
        continuous=False,
    )
    print("Now:", now)
    print("Candles returned:", len(candles))

    if not candles:
        return None

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    last_row = df.iloc[-1]

    t = last_row["date"].time()

    if t < datetime.strptime("09:10","%H:%M").time():
        return None

    if t > datetime.strptime("23:25","%H:%M").time():
        return None
    
    #df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.floor("5min")

    df = df.sort_values("date")
    print("\n--- RAW CANDLES FROM KITE ---")
    print(df[["date", "open", "high", "low", "close"]].tail(6))
    print("Last 3 timestamps:", df["date"].tail(3).tolist())
    print("Max timestamp in df:", df["date"].max())
    print("-----------------------------\n")
    last_row = df.iloc[-1]



    return last_row



def run_cycle(kite, MODELS, tokens):
    models = MODELS
    nifty_token = tokens["NIFTY50"]


    global GLOBAL_BUFFER
    global LAST_EXECUTED_CANDLE


    if not market_open():
        print("Market closed")
        return

    candle = fetch_latest_nifty_candle(kite, nifty_token)

    if candle is None:
        print("No candle")
        return

    candle_time = candle["date"]

    # Prevent duplicate execution
    if LAST_EXECUTED_CANDLE == candle_time:
        print("Already executed this candle")
        return

    LAST_EXECUTED_CANDLE = candle_time

    # Fetch equity cross-sectional features
    eq_feats = fetch_equity_features(kite, MODELS["pca"], tokens, candle_time)



    nifty_df = pd.DataFrame([candle]).set_index("date")

    nifty_df = nifty_df.rename(columns={
    "open": "nifty_open",
    "high": "nifty_high",
    "low": "nifty_low",
    "close": "nifty_close",})

    new_row = nifty_df.join(eq_feats)

    GLOBAL_BUFFER.index = pd.to_datetime(GLOBAL_BUFFER.index)

    if GLOBAL_BUFFER.index.tz is not None:
     GLOBAL_BUFFER.index = GLOBAL_BUFFER.index.tz_localize(None)

    nifty_df = pd.DataFrame([candle]).set_index("date")

    nifty_df = nifty_df.rename(columns={
    "open": "nifty_open",
    "high": "nifty_high",
    "low": "nifty_low",
    "close": "nifty_close",})




    print("Nifty index:", nifty_df.index)
    print("EQ index:", eq_feats.index) 
    print("EQ feats empty:", eq_feats.empty)
    print("EQ feats index:", eq_feats.index)
    print(new_row.columns)

    GLOBAL_BUFFER = (
     pd.concat([GLOBAL_BUFFER, new_row])
     .loc[lambda df: ~df.index.duplicated(keep="last")]
     .sort_index()
     .tail(MAX_ROWS))
    # 🔥 FORCE TZ-NAIVE INDEX

    print("\n===== LAST 5 ROWS AFTER CONCAT =====")
    print(GLOBAL_BUFFER.tail(5))
    print("Buffer length:", len(GLOBAL_BUFFER))
    print("Last index:", GLOBAL_BUFFER.index[-1])
    print("New row index:", new_row.index[0])
    print("Index unique:", GLOBAL_BUFFER.index.is_unique)
    print("====================================\n")

    if new_row.index[0] > GLOBAL_BUFFER.index[-1]:
      print("NEW CANDLE ARRIVED")




    if len(GLOBAL_BUFFER) < 20:
        print("Not enough data")
        return

    feat_df = build_features(GLOBAL_BUFFER)



    print("Before dropna shape:", GLOBAL_BUFFER.shape)
    tmp = build_features(GLOBAL_BUFFER.copy())
    print("After build_features shape:", tmp.shape)

    if feat_df.empty:
        return

    last_row = feat_df.iloc[[-1]]



    X_exp = prepare_features(last_row, EXP_FEATURES)
    exp_p = models["exp"].predict_proba(X_exp)[0, 1]


    X_dir = prepare_features(last_row, DIR_FEATURES)
    side_probs = models["side"].predict_proba(X_dir)[0]

    p_down = side_probs[0]
    p_up = side_probs[1]
    side = 1 if p_up > p_down else -1


    feature_row = {}

    for col in EXP_FEATURES:
      feature_row[col] = X_exp.iloc[0][col]

    for col in DIR_FEATURES:
      feature_row[col] = X_dir.iloc[0][col]

    feature_row["exp_p"] = exp_p
    feature_row["p_up"] = p_up
    feature_row["p_down"] = p_down
    feature_row["atr"] = float(last_row["atr"].iloc[0])
    feature_row["side"] = side

    X_meta = pd.DataFrame([feature_row])[feature_cols]


    p_win = models["meta"].predict_proba(X_meta)[0, 1]


    META_THRESH = 0.6  



    print("Buffer length:", len(GLOBAL_BUFFER))
    print("Feature rows:", len(feat_df))
    print("exp_p:", exp_p)
    print("p_up:", p_up, "p_down:", p_down)
    print("p_win:", p_win)

    if p_win < META_THRESH:
     print("Meta rejected:", p_win)
     return

    hi_ref = GLOBAL_BUFFER["nifty_high"].iloc[-LOOKBACK-1:-1].max()
    lo_ref = GLOBAL_BUFFER["nifty_low"].iloc[-LOOKBACK-1:-1].min()


    entry = hi_ref if side == 1 else lo_ref

    atr_val = float(last_row["atr"].iloc[0])

    stop   = entry - side * STOP_R * atr_val
    target = entry + side * TARGET_R * atr_val


    msg = (
      f"⏱ {candle_time}\n"
      f"Side: {'LONG' if side==1 else 'SHORT'}\n"
      f"Entry: {entry:.2f}\n"
      f"Stop: {stop:.2f}\n"
      f"Target: {target:.2f}\n"
      f"exp_p: {exp_p:.3f}\n"
      f"p_win: {p_win:.3f}"
       )

    send_telegram(msg)

    print("Signal sent")

    



def compute_atr(df, n=14):

    h,l,c = df["nifty_high"], df["nifty_low"], df["nifty_close"]

    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(n).mean()

def build_features(df):
  df=df.copy()
  if 'date' in df.columns:
    df = df.drop(columns=['date'])

  # Convert the 'date' index to a column named 'date'
  df = df.reset_index(names=['date'])

  # Ensure the 'date' column is datetime type
  df["date"] = pd.to_datetime(df["date"])

  # Create 'trade_date'
  df["trade_date"] = df["date"].dt.date
  df = df.sort_values("date").reset_index(drop=True)
  df["nifty_ret"] = df["nifty_close"].pct_change()

  # ---------------- 1. VWAP DISTANCE + SLOPE ----------------
  typical = (df["nifty_high"] + df["nifty_low"] + df["nifty_close"]) / 3
  df["nifty_vwap"] = VWAP_BASE + typical.cumsum()

  df["vwap_dist"] = (df["nifty_close"] - df["nifty_vwap"]) / (
    df["nifty_close"].rolling(50).std()
     )

  df["vwap_slope"] = df["nifty_vwap"].diff(3)

   # ---------------- 2. EMA STACK SLOPES ----------------
  for p in [5, 15, 30]:
    ema = df["nifty_close"].ewm(span=p).mean()
    df[f"ema{p}_slope"] = ema.diff()

  df["trend_stack"] = (
    np.sign(df["ema5_slope"]) *
    np.sign(df["ema15_slope"]) *
    np.sign(df["ema30_slope"])
    )

  # ---------------- 3. RANGE SKEW ----------------
  df["up_range"] = df["nifty_high"] - df["nifty_open"]
  df["dn_range"] = df["nifty_open"] - df["nifty_low"]
  df["range_skew"] = df["up_range"] - df["dn_range"]

  df["range_skew_30m"] = df["range_skew"].rolling(6).mean()

  # ---------------- 4. 30-MIN RETURN SKEW ----------------
  ret = df["nifty_ret"]

  df["pos_ret_30m"] = ret.clip(lower=0).rolling(6).sum()
  df["neg_ret_30m"] = (-ret.clip(upper=0)).rolling(6).sum()
  df["ret_skew_30m"] = df["pos_ret_30m"] - df["neg_ret_30m"]

  # ---------------- 5. BREAKOUT ACCEPTANCE ----------------
  prior_high = df["nifty_high"].rolling(20).max()
  prior_low  = df["nifty_low"].rolling(20).min()

  df["break_up"] = (df["nifty_close"] > prior_high).astype(int)
  df["break_dn"] = (df["nifty_close"] < prior_low).astype(int)

  df["break_hold"] = (
    (df["break_up"].shift(1) == 1) & # Modified to ensure boolean comparison
    (df["nifty_close"] > df["nifty_close"].shift(1))
    ).astype(int)

  # ---------------- 6. FLOW × PRICE EXHAUSTION ----------------
  df["flow_trend"] = df["adv_dec_volume_ratio"].rolling(6).sum()
  df["price_vel"] = df["nifty_close"].diff(3)

  df["exhaustion"] = df["flow_trend"] * df["price_vel"]

  # ---------------- 7. BREADTH × TREND INTERACTION ----------------
  df["breadth_trend_align"] = (
    df["pct_vwap_up"] *
    np.sign(df["ema15_slope"])
   )

# ---------------- 8. PCA MOMENTUM ----------------
  df["vwap_pca_slope"] = df["vwap_pca1"].diff(3)
  df["ret1"] = df["nifty_close"].pct_change()
  df["vwap"]= df["nifty_high"] + df["nifty_low"] + df["nifty_close"]
  df["vwap"]=df["vwap"]/3
  df["range"] = df["nifty_high"] - df["nifty_low"]

  for w in [12, 48]:
     df[f"vol_{w}"] = df["ret1"].rolling(w).std()

  df["mom_3"] = df["nifty_close"].pct_change(3)
  df["mom_6"] = df["nifty_close"].pct_change(6)

  df["vwap_dist"] = (df["nifty_close"] - df["vwap"]) / df["vwap"] 
  for w in [3,6,12]:
        df[f"vwap_pressure_ema{w}"] = df["vwap_pressure"].ewm(span=w,adjust=False).mean()
        df[f"breadth_ema{w}"] = df["liquidity_breadth"].ewm(span=w,adjust=False).mean()
        df[f"advdec_ema{w}"] = df["adv_dec_volume_ratio"].ewm(span=w,adjust=False).mean()

  df["vwap_pressure_accel"] = df["vwap_pressure"].diff(2)
  df["breadth_accel"] = df["liquidity_breadth"].diff(2)
  df["advdec_accel"] = df["adv_dec_volume_ratio"].diff(2)
  ROLL = 6   

  df["vwap_disp_ma"] = df["vwap_disp"].rolling(ROLL).mean()
  df["vwap_disp_chg"] = df["vwap_disp"] - df["vwap_disp_ma"]

  df["vwap_tb_ma"] = df["vwap_tb_spread"].rolling(ROLL).mean()
  df["vwap_tb_chg"] = df["vwap_tb_spread"] - df["vwap_tb_ma"]

  df["ret_tb_ma"] = df["ret_tb_spread"].rolling(ROLL).mean()
  df["ret_tb_slope"] = df["ret_tb_ma"].diff()

  df["flow_align_ma"] = df["pct_flow_align"].rolling(ROLL).mean()
  df["flow_align_slope"] = df["flow_align_ma"].diff()

  df["breadth_ma"] = df["liquidity_breadth"].rolling(ROLL).mean()
  df["breadth_slope"] = df["breadth_ma"].diff()

  minutes = df["date"].dt.hour * 60 + df["date"].dt.minute
  df["tod_norm"] = minutes / 390

  df["open_range"] = (
    df["nifty_high"].rolling(6).max() -
    df["nifty_low"].rolling(6).min()
   )
  ret = df["nifty_ret"]

  df["neg_ret_30m"] = (-ret.clip(upper=0)).rolling(6).sum()
  df["pos_ret_30m"] = (ret.clip(lower=0)).rolling(6).sum()

  df["down_pressure"] = df["neg_ret_30m"] - df["pos_ret_30m"]

  vwap_dist = df["nifty_close"] - df["nifty_vwap"]

  df["vwap_reject_up"] = (
    (vwap_dist > 0) &
    (vwap_dist.diff(2) < 0)
   ).astype(int)

  df["vwap_reject_dn"] = (
    (vwap_dist < 0) &
    (vwap_dist.diff(2) > 0)
   ).astype(int)

  df["vwap_reject_score"] = -vwap_dist.diff(2)

  body = (df["nifty_close"] - df["nifty_open"]).abs()

  df["lower_wick"] = (
    np.minimum(df["nifty_open"], df["nifty_close"])
    - df["nifty_low"]
   )

  df["upper_wick"] = (
    df["nifty_high"]
    - np.maximum(df["nifty_open"], df["nifty_close"])
   )

  df["wick_skew"] = df["upper_wick"] - df["lower_wick"]

  df["wick_skew_30m"] = df["wick_skew"].rolling(6).mean()

  df["sell_exhaustion"] = (
    (df["neg_ret_30m"] > df["neg_ret_30m"].rolling(50).quantile(0.7)) &
    (df["exhaustion"] > 0)
   ).astype(int)

  df["bear_flow"] = (
    (1 - df["pct_vwap_up"]) *
    df["down_pressure"]
   )

  roll_hi = df["nifty_high"].rolling(12).max()
  roll_lo = df["nifty_low"].rolling(12).min()

  df["lower_high"] = (df["nifty_high"] < roll_hi.shift()).astype(int)
  df["higher_low"] = (df["nifty_low"] > roll_lo.shift()).astype(int)

  df["down_mom"] = df["nifty_close"].diff(3)

  df["down_divergence"] = (
    np.sign(df["down_mom"]) !=
    np.sign(df["down_pressure"])
   ).astype(int)

  df["flow_trend"] = df["pct_vwap_up"] * df["trend_stack"]
  df["bear_align"] = (1 - df["pct_vwap_up"]) * df["down_pressure"]
  df["break_flow"] = df["break_hold"] * df["pct_flow_align"]
  df["wick_flow"]  = df["wick_skew_30m"] * df["down_pressure"]

  ZWIN = 48   # keep whatever window you intended

  for col in [
    "vwap_pressure",
    "liquidity_breadth",
    "adv_dec_volume_ratio",
    "eq_norm_volume",
      ]:

     mu = df[col].rolling(ZWIN, min_periods=ZWIN).mean()
     sd = df[col].rolling(ZWIN, min_periods=ZWIN).std()

     df[f"{col}_z"] = (df[col] - mu) / sd

  minutes = df["date"].dt.hour * 60 + df["date"].dt.minute

  df["minutes_since_open"] = minutes - 555
  df["session_frac"] = df["minutes_since_open"] / 375

  df["vol_ratio_12_48"] = df["vol_12"] / df["vol_48"]

  df["range_z20"] = (
    (df["range"] - df["range"].rolling(20).mean())
    / df["range"].rolling(20).std()
    )
  
  df["flow_dir"] = (
    np.sign(df["vwap_pressure"])
    + np.sign(df["adv_dec_volume_ratio"] - 1)
    + np.sign(df["liquidity_breadth"] - 0.5)
    )

  df["flow_dir_ema6"] = df["flow_dir"].ewm(span=6,adjust=False).mean()
  df["flow_dir_ema12"] = df["flow_dir"].ewm(span=12,adjust=False).mean()

  df["nifty_ret"] = df["nifty_close"].pct_change()

  typical = (df["nifty_high"] + df["nifty_low"] + df["nifty_close"]) / 3
  df["nifty_vwap"] = VWAP_BASE + typical.cumsum()  

  df["vwap_dist"] = (df["nifty_close"] - df["nifty_vwap"]) / (
    df["nifty_close"].rolling(50).std()
    )

  df["vwap_slope"] = df["nifty_vwap"].diff(3)


  for p in [5, 15, 30]:
    ema = df["nifty_close"].ewm(span=p).mean()
    df[f"ema{p}_slope"] = ema.diff()

  df["trend_stack"] = (
    np.sign(df["ema5_slope"]) *
    np.sign(df["ema15_slope"]) *
    np.sign(df["ema30_slope"])
    )

  df["up_range"] = df["nifty_high"] - df["nifty_open"]
  df["dn_range"] = df["nifty_open"] - df["nifty_low"]
  df["range_skew"] = df["up_range"] - df["dn_range"]

  df["range_skew_30m"] = df["range_skew"].rolling(6).mean()

  ret = df["nifty_ret"]

  df["pos_ret_30m"] = ret.clip(lower=0).rolling(6).sum()
  df["neg_ret_30m"] = (-ret.clip(upper=0)).rolling(6).sum()
  df["ret_skew_30m"] = df["pos_ret_30m"] - df["neg_ret_30m"]

  prior_high = df["nifty_high"].rolling(20).max()
  prior_low  = df["nifty_low"].rolling(20).min()

  df["break_up"] = (df["nifty_close"] > prior_high).astype(int)
  df["break_dn"] = (df["nifty_close"] < prior_low).astype(int)

  df["break_hold"] = (
    (df["break_up"].shift(1) == 1) & 
    (df["nifty_close"] > df["nifty_close"].shift(1))
    ).astype(int)

  df["flow_trend"] = df["adv_dec_volume_ratio"].rolling(6).sum()
  df["price_vel"] = df["nifty_close"].diff(3)

  df["exhaustion"] = df["flow_trend"] * df["price_vel"]

  df["breadth_trend_align"] = (
    df["pct_vwap_up"] *
    np.sign(df["ema15_slope"])
   )

  df["vwap_pca_slope"] = df["vwap_pca1"].diff(3)

  df["ret1"] = df["nifty_close"].pct_change()
  df["vwap"]= df["nifty_high"] + df["nifty_low"] + df["nifty_close"]
  df["vwap"]=df["vwap"]/3
  df["range"] = df["nifty_high"] - df["nifty_low"]

  for w in [12, 48]:
    df[f"vol_{w}"] = df["ret1"].rolling(w).std()

  df["mom_3"] = df["nifty_close"].pct_change(3)
  df["mom_6"] = df["nifty_close"].pct_change(6)

  df["vwap_dist"] = (df["nifty_close"] - df["vwap"]) / df["vwap"]

  for w in [3,6,12]:
        df[f"vwap_pressure_ema{w}"] = df["vwap_pressure"].ewm(span=w,adjust=False).mean()
        df[f"breadth_ema{w}"] = df["liquidity_breadth"].ewm(span=w,adjust=False).mean()
        df[f"advdec_ema{w}"] = df["adv_dec_volume_ratio"].ewm(span=w,adjust=False).mean()

  df["vwap_pressure_accel"] = df["vwap_pressure"].diff(2)
  df["breadth_accel"] = df["liquidity_breadth"].diff(2)
  df["advdec_accel"] = df["adv_dec_volume_ratio"].diff(2)

  ROLL = 6  

  df["vwap_disp_ma"] = df["vwap_disp"].rolling(ROLL).mean()
  df["vwap_disp_chg"] = df["vwap_disp"] - df["vwap_disp_ma"]

  df["vwap_tb_ma"] = df["vwap_tb_spread"].rolling(ROLL).mean()
  df["vwap_tb_chg"] = df["vwap_tb_spread"] - df["vwap_tb_ma"]

  df["ret_tb_ma"] = df["ret_tb_spread"].rolling(ROLL).mean()
  df["ret_tb_slope"] = df["ret_tb_ma"].diff()

  df["flow_align_ma"] = df["pct_flow_align"].rolling(ROLL).mean()
  df["flow_align_slope"] = df["flow_align_ma"].diff()

  df["breadth_ma"] = df["liquidity_breadth"].rolling(ROLL).mean()
  df["breadth_slope"] = df["breadth_ma"].diff()

  minutes = df["date"].dt.hour * 60 + df["date"].dt.minute
  df["tod_norm"] = minutes / 390

  df["open_range"] = (
    df["nifty_high"].rolling(6).max() -
    df["nifty_low"].rolling(6).min()
   )
  ret = df["nifty_ret"]

  df["neg_ret_30m"] = (-ret.clip(upper=0)).rolling(6).sum()
  df["pos_ret_30m"] = (ret.clip(lower=0)).rolling(6).sum()

  df["down_pressure"] = df["neg_ret_30m"] - df["pos_ret_30m"]

  vwap_dist = df["nifty_close"] - df["nifty_vwap"]

  df["vwap_reject_up"] = (
    (vwap_dist > 0) &
    (vwap_dist.diff(2) < 0)
    ).astype(int)

  df["vwap_reject_dn"] = (
    (vwap_dist < 0) &
    (vwap_dist.diff(2) > 0)
    ).astype(int)

  df["vwap_reject_score"] = -vwap_dist.diff(2)

  body = (df["nifty_close"] - df["nifty_open"]).abs()

  df["lower_wick"] = (
    np.minimum(df["nifty_open"], df["nifty_close"])
    - df["nifty_low"]
    )

  df["upper_wick"] = (
    df["nifty_high"]
    - np.maximum(df["nifty_open"], df["nifty_close"])
  )

  df["wick_skew"] = df["upper_wick"] - df["lower_wick"]

  df["wick_skew_30m"] = df["wick_skew"].rolling(6).mean()

  df["sell_exhaustion"] = (
    (df["neg_ret_30m"] > df["neg_ret_30m"].rolling(50).quantile(0.7)) &
    (df["exhaustion"] > 0)
    ).astype(int)

  df["bear_flow"] = (
    (1 - df["pct_vwap_up"]) *
    df["down_pressure"]
   )


  roll_hi = df["nifty_high"].rolling(12).max()
  roll_lo = df["nifty_low"].rolling(12).min()

  df["lower_high"] = (df["nifty_high"] < roll_hi.shift()).astype(int)
  df["higher_low"] = (df["nifty_low"] > roll_lo.shift()).astype(int)

  df["down_mom"] = df["nifty_close"].diff(3)

  df["down_divergence"] = (
    np.sign(df["down_mom"]) !=
    np.sign(df["down_pressure"])
    ).astype(int)

  df["flow_trend"] = df["pct_vwap_up"] * df["trend_stack"]
  df["bear_align"] = (1 - df["pct_vwap_up"]) * df["down_pressure"]
  df["break_flow"] = df["break_hold"] * df["pct_flow_align"]
  df["wick_flow"]  = df["wick_skew_30m"] * df["down_pressure"]

  minutes = df["date"].dt.hour * 60 + df["date"].dt.minute

  df["minutes_since_open"] = minutes - 555
  df["session_frac"] = df["minutes_since_open"] / 375

  df["vol_ratio_12_48"] = df["vol_12"] / df["vol_48"]

  df["range_z20"] = (
    (df["range"] - df["range"].rolling(20).mean())
    / df["range"].rolling(20).std()
    )
  df["flow_dir"] = (
    np.sign(df["vwap_pressure"])
    + np.sign(df["adv_dec_volume_ratio"] - 1)
    + np.sign(df["liquidity_breadth"] - 0.5)
   )

  df["flow_dir_ema6"] = df["flow_dir"].ewm(span=6,adjust=False).mean()
  df["flow_dir_ema12"] = df["flow_dir"].ewm(span=12,adjust=False).mean()

  ZWIN = 48   

  for col in [
    "vwap_pressure",
    "liquidity_breadth",
    "adv_dec_volume_ratio",
    "eq_norm_volume",
    ]:

     mu = df[col].rolling(ZWIN, min_periods=ZWIN).mean()
     sd = df[col].rolling(ZWIN, min_periods=ZWIN).std()

     df[f"{col}_z"] = (df[col] - mu) / sd


  df["atr"] = compute_atr(df)

  nan_counts = df.isna().sum()
  print("Columns fully NaN:")
  print(nan_counts[nan_counts == len(df)])

  last = df.iloc[-1]
  nan_cols = last[last.isna()]

  print("Last row NaN columns:")
  print(nan_cols)

  print("\n--- DEBUG ROLLING WINDOW CHECK ---")
  print("Non-NaN counts last 10 rows:")
  print(df.tail(10)[[
    "vwap_disp",
    "vwap_tb_spread",
    "ret_tb_spread",
    "pct_flow_align",
    "liquidity_breadth",
   ]].notna().sum())
  print("-----------------------------------\n")
  print("Consecutive non-NaN in last 10 rows:")
  print(df.tail(10)[[
    "vwap_disp",
       ]].notna())

  



  return df.dropna()  


def prepare_features(df, feats):

    X = df[feats].copy()
    X = X.replace([np.inf,-np.inf], np.nan)

    return X.fillna(0)


def assert_features(df, cols):

    missing = set(cols) - set(df.columns)

    if missing:
        raise RuntimeError(f"Missing features: {missing}")


EXP_FEATURES = [
    "ret1","range","vol_12","vol_48","vwap_dist",
    "vwap_pressure","liquidity_breadth",
    "adv_dec_volume_ratio","eq_norm_volume",
    "mom_3","mom_6",
    "vwap_pressure_ema3","vwap_pressure_ema6","vwap_pressure_ema12",
    "breadth_ema3","breadth_ema6","breadth_ema12",
    "advdec_ema3","advdec_ema6","advdec_ema12",
    "vwap_pressure_accel","breadth_accel","advdec_accel",
    "minutes_since_open","session_frac",
    "vol_ratio_12_48","range_z20",
    "vwap_pressure_z",
    "liquidity_breadth_z",
    "adv_dec_volume_ratio_z",
    "eq_norm_volume_z",
]

DIR_FEATURES = [
    "pct_vwap_up","vwap_tb_spread","ret_tb_spread",
    "pct_flow_align","vwap_pca1","vwap_pca_slope",
    "flow_trend","bear_align","break_flow","wick_flow",
    "vwap_dist","vwap_slope","trend_stack",
    "range_skew_30m","ret_skew_30m","break_hold",
    "exhaustion","breadth_trend_align",
    "down_pressure","vwap_reject_score",
    "wick_skew_30m","sell_exhaustion","bear_flow",
    "lower_high","higher_low","down_divergence",
]

feature_cols = list(dict.fromkeys(
    EXP_FEATURES +
    DIR_FEATURES +
    ["exp_p", "p_up", "p_down", "atr", "side"]
))


def fetch_equity_features(kite, pca, tokens, candle_time):

    end_time = candle_time + timedelta(minutes=INTERVAL_MIN)
    start = end_time - timedelta(minutes=125)
 #   start = now - timedelta(minutes=LOOKBACK_BARS * INTERVAL_MIN*2)

    dfs = []

    for sym in SYMBOLS:

#        token = kite.ltp(f"NSE:{sym}")[f"NSE:{sym}"]["instrument_token"]
        token = tokens[sym]

        candles = kite.historical_data(
            token,
            start,
            end_time,
            INTERVAL,
        )

        if candles:
            df = pd.DataFrame(candles)
            df["symbol"] = sym
            dfs.append(df)


        time.sleep(0.35)

    if not dfs:
        return pd.DataFrame()

    data = pd.concat(dfs)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.floor("5min")
    candle_time = pd.to_datetime(candle_time).tz_localize(None).floor("5min")

    data = data[data["date"] <= candle_time]
    print("\n--- SYMBOL COUNT PER TIMESTAMP ---")
    print(data.groupby("date")["symbol"].nunique().tail())
    print("-----------------------------------\n")
    print("\n--- SYMBOL ROW COUNTS ---")
    print(data.groupby("symbol").size().describe())
    print(data.groupby("symbol").size().head())
    print("-------------------------\n")

    data = (
        data.sort_values(["symbol", "date"])
            .drop_duplicates(["symbol", "date"])
    )

    # ---------------- VWAP ----------------
    data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3

    # ---------------- ADV ----------------
    data["adv"] = (
        data.groupby("symbol")["volume"]
            .transform(lambda x: x.rolling(ADV_WINDOW, min_periods=5).mean())
    )
    print("\n--- ADV DIAGNOSTIC ---")
    print("Total rows:", len(data))
    print("ADV NaN count:", data["adv"].isna().sum())
    print("ADV non-NaN count:", data["adv"].notna().sum())
    print("----------------------\n")

    data["adv"] = data["adv"].replace(0, np.nan)
    data["norm_vol"] = data["volume"] / data["adv"]
    print("\n--- NORM_VOL DIAGNOSTIC ---")
    print("norm_vol NaN count:", data["norm_vol"].isna().sum())
    print("norm_vol non-NaN count:", data["norm_vol"].notna().sum())
    print("---------------------------\n")

    data["ret"] = data.groupby("symbol")["close"].pct_change()

    # ---------------- BASE AGGREGATES ----------------
    eq_norm_volume = data.groupby("date")["norm_vol"].mean()

    data["active"] = (data["norm_vol"] > BREADTH_THRESHOLD).astype(int)
    liquidity_breadth = data.groupby("date")["active"].mean()

    adv_vol = data[data["ret"] > 0].groupby("date")["volume"].sum()
    dec_vol = data[data["ret"] < 0].groupby("date")["volume"].sum()

    adv_dec_vol_ratio = adv_vol / dec_vol

    data["vwap_pressure"] = (data["close"] - data["vwap"]) / data["vwap"]
    vwap_pressure = data.groupby("date")["vwap_pressure"].mean()

    # ---------------- CROSS-SECTIONAL PROXIES ----------------
    pct_vwap_up = data.groupby("date")["vwap_pressure"].apply(
        lambda x: (x > 0).mean()
    )

    vwap_disp = data.groupby("date")["vwap_pressure"].std()

    def top_bottom_spread(x, q=0.2):
        hi = x.quantile(1 - q)
        lo = x.quantile(q)
        return x[x >= hi].mean() - x[x <= lo].mean()

    vwap_tb = data.groupby("date")["vwap_pressure"].apply(top_bottom_spread)
    ret_tb  = data.groupby("date")["ret"].apply(top_bottom_spread)

    pct_flow_align = data.groupby("date")["vwap_pressure"].apply(
        lambda x: (np.sign(x) == np.sign(x.mean())).mean()
    )

    # ---------------- PCA FACTOR ----------------
    vwap_mat = (
        data.pivot(index="date", columns="symbol", values="vwap_pressure")
            .sort_index()
    )

    if len(vwap_mat) < -5:
        return pd.DataFrame()

    pca_factor = pd.Series(
    pca.transform(vwap_mat.fillna(0)).flatten(),
    index=vwap_mat.index,
    name="vwap_pca1",
)




    # ---------------- FINAL FEATURE FRAME ----------------
    feats = pd.concat(
        [
            eq_norm_volume.rename("eq_norm_volume"),
            liquidity_breadth.rename("liquidity_breadth"),
            adv_dec_vol_ratio.rename("adv_dec_volume_ratio"),
            vwap_pressure.rename("vwap_pressure"),
            pct_vwap_up.rename("pct_vwap_up"),
            vwap_disp.rename("vwap_disp"),
            vwap_tb.rename("vwap_tb_spread"),
            ret_tb.rename("ret_tb_spread"),
            pct_flow_align.rename("pct_flow_align"),
            pca_factor,
        ],
        axis=1,
    )
    print("\n--- EQ_NORM_VOLUME DIAG ---")
    print(eq_norm_volume)
    print("----------------------------\n")
    print("Raw feats shape before dropna:", feats.shape)
    print("NaN counts per column:")
    print(feats.isna().sum().sort_values(ascending=False).head(10))

    feats = feats.dropna()

    print("Feats shape AFTER dropna:", feats.shape)
    feats.index = feats.index.floor("5min")
    feats.index = feats.index.tz_localize(None)

    feats.index = pd.to_datetime(feats.index).floor("5min")
    candle_time = pd.to_datetime(candle_time).floor("5min")


    valid_times = feats.index[feats.index <= candle_time]
    if len(valid_times) == 0:
       print("No valid equity timestamp")
       return pd.DataFrame()
    last_valid_time = valid_times.max()
    if last_valid_time != candle_time:
      print(f"⚠ Equity lag detected. Using {last_valid_time} instead of {candle_time}")
    row = feats.loc[[last_valid_time]].copy()

    # 🔥 FORCE index to match candle_time
    row.index = [candle_time]

    return row





def resolve_tokens(kite):

    instruments = pd.DataFrame(kite.instruments())
    time.sleep(1)

    mapping = {}

    eq = instruments[
        (instruments["exchange"] == "NSE")
        & instruments["tradingsymbol"].isin(SYMBOLS)
        & (instruments["instrument_type"] == "EQ")
    ]

    for _, r in eq.iterrows():
        mapping[r["tradingsymbol"]] = r["instrument_token"]

    nifty = instruments[
        instruments["tradingsymbol"].isin(["NIFTY 50","NIFTY50"])
        & instruments["segment"].str.contains("INDICES", na=False)
    ]

    mapping["NIFTY50"] = nifty["instrument_token"].iloc[0]

    return mapping


def load_models():
    models = {
        "exp": joblib.load("models/exp.pkl"),
        "side": joblib.load("models/side.pkl"),
        "meta": joblib.load("models/meta.pkl"),
        "pca" : joblib.load("models/pca.pkl")

    }

    return models

#def load_models():
#    models = {
#        "exp": joblib.load("exp.pkl"),
#        "side": joblib.load("side.pkl"),
#        "meta": joblib.load("meta.pkl"),
#        "pca": joblib.load("pca.pkl"),
#    }
#    return models

def now_ist():
    return datetime.now(IST)



def minute_of_day(ts):
    return ts.hour * 60 + ts.minute

def warmup_today(kite, MODELS, tokens):
    """
    Backfill today's missing bars (09:15 -> now)
    for both NIFTY and cross-sectional equity features.
    """

    global GLOBAL_BUFFER

    now = datetime.now(IST)
    today_open = pd.Timestamp.now(tz="Asia/Kolkata") \
    .floor("D") \
    .replace(hour=9, minute=15) \
    .tz_localize(None)

    nifty_token = tokens["NIFTY50"]

    print("Starting warmup...")

    # -------------------------------------------------
    # 1️⃣ Fetch today's NIFTY candles
    # -------------------------------------------------
    nifty_candles = kite.historical_data(
        nifty_token,
        today_open,
        now,
        INTERVAL
    )

    if not nifty_candles:
        print("No NIFTY candles for warmup.")
        return

    nifty_df = pd.DataFrame(nifty_candles)
    nifty_df["date"] = (
        pd.to_datetime(nifty_df["date"])
        .dt.tz_localize(None)
        .dt.floor("5min")
    )

    nifty_df = nifty_df.rename(columns={
        "open": "nifty_open",
        "high": "nifty_high",
        "low": "nifty_low",
        "close": "nifty_close",
    })

    nifty_df = nifty_df.set_index("date").sort_index()

    print(f"NIFTY warmup rows: {len(nifty_df)}")

    # -------------------------------------------------
    # 2️⃣ Fetch equity candles once (all symbols)
    # -------------------------------------------------
    # Add buffer so rolling(20) etc work properly
    eq_start = today_open - timedelta(minutes=125)

    dfs = []

    for sym in SYMBOLS:
        token = tokens[sym]

        candles = kite.historical_data(
            token,
            eq_start,
            now + timedelta(minutes=1),
            INTERVAL
        )

        if candles:
            df = pd.DataFrame(candles)
            df["symbol"] = sym
            dfs.append(df)

        time.sleep(0.25)  # safe rate limit

    if not dfs:
        print("No equity candles for warmup.")
        return

    data = pd.concat(dfs)

    data["date"] = (
        pd.to_datetime(data["date"])
        .dt.tz_localize(None)
        .dt.floor("5min")
    )

    data = data.sort_values(["symbol", "date"])
    data = data.drop_duplicates(["symbol", "date"])

    print(f"Equity warmup rows: {len(data)}")

    # -------------------------------------------------
    # 3️⃣ Build cross-sectional features FOR FULL RANGE
    # -------------------------------------------------

    # VWAP
    data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3

    # ADV
    data["adv"] = (
        data.groupby("symbol")["volume"]
            .transform(lambda x: x.rolling(ADV_WINDOW, min_periods=5).mean())
    )

    data["adv"] = data["adv"].replace(0, np.nan)
    data["norm_vol"] = data["volume"] / data["adv"]

    data["ret"] = data.groupby("symbol")["close"].pct_change()

    # Aggregates
    eq_norm_volume = data.groupby("date")["norm_vol"].mean()

    data["active"] = (data["norm_vol"] > BREADTH_THRESHOLD).astype(int)
    liquidity_breadth = data.groupby("date")["active"].mean()

    adv_vol = data[data["ret"] > 0].groupby("date")["volume"].sum()
    dec_vol = data[data["ret"] < 0].groupby("date")["volume"].sum()
    adv_dec_vol_ratio = adv_vol / dec_vol

    data["vwap_pressure"] = (data["close"] - data["vwap"]) / data["vwap"]
    vwap_pressure = data.groupby("date")["vwap_pressure"].mean()

    pct_vwap_up = data.groupby("date")["vwap_pressure"].apply(
        lambda x: (x > 0).mean()
    )

    vwap_disp = data.groupby("date")["vwap_pressure"].std()

    def top_bottom_spread(x, q=0.2):
        hi = x.quantile(1 - q)
        lo = x.quantile(q)
        return x[x >= hi].mean() - x[x <= lo].mean()

    vwap_tb = data.groupby("date")["vwap_pressure"].apply(top_bottom_spread)
    ret_tb  = data.groupby("date")["ret"].apply(top_bottom_spread)

    pct_flow_align = data.groupby("date")["vwap_pressure"].apply(
        lambda x: (np.sign(x) == np.sign(x.mean())).mean()
    )

    # PCA
    vwap_mat = (
        data.pivot(index="date", columns="symbol", values="vwap_pressure")
        .sort_index()
    )

    pca_factor = pd.Series(
        MODELS["pca"].transform(vwap_mat.fillna(0)).flatten(),
        index=vwap_mat.index,
        name="vwap_pca1",
    )

    feats = pd.concat(
        [
            eq_norm_volume.rename("eq_norm_volume"),
            liquidity_breadth.rename("liquidity_breadth"),
            adv_dec_vol_ratio.rename("adv_dec_volume_ratio"),
            vwap_pressure.rename("vwap_pressure"),
            pct_vwap_up.rename("pct_vwap_up"),
            vwap_disp.rename("vwap_disp"),
            vwap_tb.rename("vwap_tb_spread"),
            ret_tb.rename("ret_tb_spread"),
            pct_flow_align.rename("pct_flow_align"),
            pca_factor,
        ],
        axis=1,
    )

    feats = feats.dropna()

    # Keep only today's timestamps
    feats = feats[feats.index >= today_open]

    print(f"Equity feature rows after dropna: {len(feats)}")

    # -------------------------------------------------
    # 4️⃣ Merge NIFTY + EQ features
    # -------------------------------------------------

    full_today = nifty_df.join(feats)

    # Remove rows with missing eq features
    full_today = full_today.dropna()

    print(f"Final warmup rows appended: {len(full_today)}")

    # -------------------------------------------------
    # 5️⃣ Append once
    # -------------------------------------------------

    GLOBAL_BUFFER = (
        pd.concat([GLOBAL_BUFFER, full_today])
        .loc[lambda df: ~df.index.duplicated(keep="last")]
        .sort_index()
        .tail(MAX_ROWS)
    )

    print("Warmup complete. Buffer synced.")

# ============================================================
# MAIN
# ============================================================
def main():
    global GLOBAL_BUFFER

    kite = kite_login()
    tokens = resolve_tokens(kite)
    MODELS = load_models()
    warmup_today(kite, MODELS, tokens)
    send_telegram("🟢 Railway bot started")
    # Align to next 5-min boundary before starting engine
    print("Aligning to next 5-min boundary...")
    sleep_to_next_bar()

    while True:
        try:
            print("Loop running...")

            if not market_open():
                print("Market closed. Sleeping 60s.")
                time.sleep(60)
                continue

            run_cycle(kite, MODELS, tokens)
            sleep_to_next_bar()

        except Exception:
            err = traceback.format_exc()
            print("SYSTEM ERROR:")
            print(err)

            try:
                send_telegram(f"🚨 SYSTEM CRASH\n{err[:3000]}")
            except:
                pass

            time.sleep(30)


if __name__ == "__main__":
    main()
