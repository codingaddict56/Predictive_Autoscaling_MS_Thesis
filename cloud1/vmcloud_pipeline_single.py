#!/usr/bin/env python3
import os, argparse, warnings, pickle, json
from math import sqrt
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# optional TF
HAS_TF = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    HAS_TF = False

# statsmodels
from statsmodels.tsa.arima.model import ARIMA

# ==== YOUR CSV (change if you relocate) ====
DEFAULT_VM_CLOUD_CSV = Path("/Users/chaithratelkar/Documents/predictive-autoscaling/cloud1/data/vmcloud_small.csv")

# ==== Try to import your modules (optional) ====
try:
    import data_preprocessing as dp
except Exception:
    dp = None

try:
    import advanced_features as af
except Exception:
    af = None

# ---------- Utilities ----------
def rmse_backcompat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return sqrt(mean_squared_error(y_true, y_pred))

def print_scores(name, actual, preds):
    if actual is None or preds is None:
        print(f"[{name}] skipped (dependency missing)")
        return
    n = min(len(actual), len(preds))
    actual = actual[:n]; preds = preds[:n]
    mae = mean_absolute_error(actual, preds)
    try:
        rmse = mean_squared_error(actual, preds, squared=False)
    except TypeError:
        rmse = sqrt(mean_squared_error(actual, preds))
    print(f"[{name}] MAE={mae:.4f}  RMSE={rmse:.4f}  N={n}")

# ---------- 1) Load ----------
def load_vmcloud_df(csv_path: Path, timestamp_col="timestamp"):
    """
    Preferred: use user's data_preprocessing.load_vmcloud_df if present.
    Fallback: internal loader.
    """
    if dp and hasattr(dp, "load_vmcloud_df"):
        df = dp.load_vmcloud_df(csv_path)
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        df = pd.read_csv(csv_path)
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.sort_values(timestamp_col)
    return df

# ---------- 1b) Optional: synthesize latency p95 from execution_time ----------
def ensure_latency_target(
    df: pd.DataFrame,
    target: str = "response_time_p95",
    timestamp_col: str = "timestamp",
    base_col: str = "execution_time",
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    If `target` is missing and `base_col` exists, create a rolling p95 latency:
      response_time_p95 = rolling_quantile_0.95(execution_time, window)
    """
    if target in df.columns:
        return df

    if target == "response_time_p95":
        if base_col not in df.columns:
            raise ValueError(
                f"Cannot synthesize {target}: '{base_col}' not in dataframe. "
                f"Available: {list(df.columns)[:20]}"
            )

        df = df.copy()
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.sort_values(timestamp_col)

        rt = df[base_col].rolling(window=window, min_periods=min_periods).quantile(0.95)
        df[target] = rt.bfill().ffill()
        return df

    raise ValueError(
        f"Target '{target}' not found and no synthesis rule defined. "
        f"Available: {list(df.columns)[:20]}"
    )

# ---------- 2) Feature Engineering ----------
def default_build_features(df: pd.DataFrame, target="cpu_usage", timestamp_col="timestamp", lags=(1,2,3), roll_windows=(3,5)):
    """
    Minimal, safe feature engineering if advanced_features.build_features is not available.
    - numeric only (except timestamp)
    - lags for target
    - rolling means for a few key metrics if present
    Returns: X (DataFrame of features), y (Series), df_out (aligned with features)
    """
    df = df.copy()
    for c in df.columns:
        if c == timestamp_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available: {list(df.columns)[:20]}")

    candidates = [c for c in ["cpu_usage","memory_usage","network_traffic","power_consumption","execution_time"] if c in df.columns]
    if target not in candidates:
        candidates = [target] + candidates

    for L in lags:
        df[f"{target}_lag{L}"] = df[target].shift(L)

    for w in roll_windows:
        for c in candidates:
            df[f"{c}_roll{w}"] = df[c].rolling(window=w, min_periods=1).mean()

    df_fe = df.dropna().reset_index(drop=True)

    y = df_fe[target].astype(float)
    feature_cols = [c for c in df_fe.columns if c not in [target]]
    X = df_fe[feature_cols].select_dtypes(include=[np.number]).copy()

    return X, y, df_fe

def build_features(df: pd.DataFrame, target="cpu_usage", timestamp_col="timestamp"):
    if af and hasattr(af, "build_features"):
        out = af.build_features(df, target, timestamp_col)
        if isinstance(out, tuple) and len(out) == 3:
            return out
        else:
            return default_build_features(df, target, timestamp_col)
    else:
        return default_build_features(df, target, timestamp_col)

# ---------- 3) Scaling ----------
def scale_X_y(X: pd.DataFrame, y: pd.Series):
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    Xs = sx.fit_transform(X.values.astype(float))
    ys = sy.fit_transform(y.values.reshape(-1,1).astype(float))
    return Xs, ys, sx, sy

# ---------- 4) Model training ----------
def create_seq_xy(Xs, ys, seq_len):
    Xseq, yseq = [], []
    for i in range(len(ys) - seq_len):
        Xseq.append(Xs[i:i+seq_len, :])
        yseq.append(ys[i+seq_len, 0])
    return np.array(Xseq), np.array(yseq)

def train_lstm_multivar(Xs, ys, seq_len, epochs, save_path: Path):
    if not HAS_TF:
        print("[LSTM] TensorFlow not installed; skipping LSTM training.")
        return None
    Xseq, yseq = create_seq_xy(Xs, ys, seq_len)
    model = keras.Sequential([
        layers.Input(shape=(seq_len, Xs.shape[1])),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xseq, yseq, epochs=epochs, batch_size=64, verbose=0)
    model.save(save_path)
    print(f"[LSTM] saved: {save_path}")
    return model

def train_rf_tabular(Xs, ys, seq_len, save_path: Path):
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xs, ys.ravel())
    with open(save_path, "wb") as f:
        pickle.dump({"model": rf}, f)
    print(f"[RF] saved: {save_path}")
    return rf

def train_arima_univar(ys, order, save_path: Path):
    p,d,q = order
    model = ARIMA(ys.reshape(-1), order=(p,d,q))
    fitted = model.fit()
    with open(save_path, "wb") as f:
        pickle.dump({"model": fitted, "order": (p,d,q)}, f)
    print(f"[ARIMA] saved: {save_path} (order={p,d,q})")
    return fitted

# ---------- 5) Predict next (one-step) ----------
def predict_next_lstm_multivar(model, Xs, sy, seq_len):
    if model is None: return None
    last_seq = Xs[-seq_len:,:].reshape(1, seq_len, -1)
    yhat_s = model.predict(last_seq, verbose=0)[0,0]
    return float(sy.inverse_transform([[yhat_s]])[0,0])

def predict_next_rf(rf_path, Xs, sy):
    with open(rf_path, "rb") as f:
        obj = pickle.load(f)
    rf = obj["model"]
    yhat_s = rf.predict(Xs[-1:,:])[0]
    return float(sy.inverse_transform([[yhat_s]])[0,0])

def predict_next_arima(arima_path, sy):
    with open(arima_path, "rb") as f:
        obj = pickle.load(f)
    fitted = obj["model"]
    yhat_s = fitted.forecast(steps=1)[0]
    return float(sy.inverse_transform([[yhat_s]])[0,0])

# ---------- 6) Rolling evaluation ----------
def rolling_eval_lstm(Xs, ys, model, seq_len, sy):
    if model is None: return None, None
    preds_s = []
    for i in range(seq_len, len(ys)):
        window = Xs[i-seq_len:i,:].reshape(1, seq_len, -1)
        preds_s.append(model.predict(window, verbose=0)[0,0])
    preds_s = np.array(preds_s).reshape(-1,1)
    actual = sy.inverse_transform(ys[seq_len:].reshape(-1,1)).reshape(-1)
    preds  = sy.inverse_transform(preds_s).reshape(-1)
    return actual, preds

def rolling_eval_rf(Xs, ys, rf_path, sy):
    with open(rf_path, "rb") as f:
        obj = pickle.load(f)
    rf = obj["model"]
    preds_s = rf.predict(Xs).reshape(-1,1)
    actual = sy.inverse_transform(ys.reshape(-1,1)).reshape(-1)
    preds  = sy.inverse_transform(preds_s).reshape(-1)
    return actual, preds

def rolling_eval_arima(ys, arima_path, sy):
    with open(arima_path, "rb") as f:
        obj = pickle.load(f)
    fitted = obj["model"]
    n_preds = len(ys) - 1
    preds_s = fitted.forecast(steps=n_preds).reshape(-1,1)
    actual = sy.inverse_transform(ys[1:].reshape(-1,1)).reshape(-1)
    preds  = sy.inverse_transform(preds_s).reshape(-1)
    return actual, preds

# ---------- Predictive uncertainty helpers ----------
def rf_predict_quantiles(rf, X_last, qs=(0.5, 0.9, 0.95)):
    leaves = np.column_stack([t.predict(X_last) for t in rf.estimators_])
    quants = np.percentile(leaves, [q*100 for q in qs], axis=1).reshape(-1)
    return {q: quants[i] for i, q in enumerate(qs)}

def _load_rf(rf_path):
    with open(rf_path, "rb") as f:
        return pickle.load(f)["model"]

def lstm_mc_predict(model, Xs, sy, seq_len, T=30):
    last_seq = Xs[-seq_len:,:].reshape(1, seq_len, -1)
    preds_s = []
    for _ in range(T):
        yhat_s = model(last_seq, training=True).numpy()[0,0]  # dropout ON
        preds_s.append(yhat_s)
    preds = sy.inverse_transform(np.array(preds_s).reshape(-1,1)).reshape(-1)
    return float(np.mean(preds)), float(np.std(preds))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="One-command VMCloud pipeline (data -> features -> train -> predict -> evaluate)")
    ap.add_argument("--target", default="cpu_usage")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--arima-order", default="2,1,2")
    ap.add_argument("--save-dir", default="artifacts")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] CSV path:", os.path.abspath(str(DEFAULT_VM_CLOUD_CSV)))

    # 1) DATA PROCESSING
    print("Data processing started ...")
    df = load_vmcloud_df(DEFAULT_VM_CLOUD_CSV, timestamp_col=args.timestamp_col)
    # if you ask for latency and it's missing, synthesize from execution_time
    df = ensure_latency_target(df, target=args.target, timestamp_col=args.timestamp_col)
    print(f"[preprocess] df shape={df.shape} target={args.target}")
    print("Data processing completed ✅")

    # 2) FEATURE ENGINEERING
    print("Feature engineering started ...")
    X, y, df_out = build_features(df, target=args.target, timestamp_col=args.timestamp_col)
    print(f"[features] X shape={X.shape}, y len={len(y)}")
    Xs, ys, sx, sy = scale_X_y(X, y)
    joblib.dump({"scaler_X": sx, "scaler_y": sy, "target": args.target, "feature_names": list(X.columns)}, save_dir / "scalers.pkl")
    np.save(save_dir / "X_scaled.npy", Xs)
    np.save(save_dir / "y_scaled.npy", ys)
    print("Feature engineering completed ✅")

    # 3) TRAIN
    print("Model training started ...")
    lstm_path = save_dir / "lstm_model.keras"
    rf_path   = save_dir / "rf_model.pkl"
    arima_path= save_dir / "arima_model.pkl"

    lstm_model = train_lstm_multivar(Xs, ys, args.seq_len, args.epochs, lstm_path) if HAS_TF else None
    _ = train_rf_tabular(Xs, ys, args.seq_len, rf_path)
    p, d, q = [int(x.strip()) for x in args.arima_order.split(",")]
    _ = train_arima_univar(ys, (p, d, q), arima_path)
    print("Model training completed ✅")

    # 4) PREDICT (one-step ahead + uncertainty)
    print("Prediction started ...")
    signal = {"metric": args.target, "model": None, "horizon_seconds": None}

    # LSTM (point + optional MC)
    if HAS_TF and lstm_model is not None and len(ys) > args.seq_len:
        yhat_lstm = predict_next_lstm_multivar(lstm_model, Xs, sy, args.seq_len)
        print(f"[predict:LSTM]  next={yhat_lstm:.4f}")
        try:
            lstm_mean, lstm_sigma = lstm_mc_predict(lstm_model, Xs, sy, args.seq_len, T=30)
            print(f"[predict:LSTM-MC] mean={lstm_mean:.4f} sigma={lstm_sigma:.4f}")
            signal.update({"model":"lstm_mc", "mean": lstm_mean, "sigma": lstm_sigma})
        except Exception:
            pass
    else:
        print("[predict:LSTM]  skipped")

    # RF: point + quantiles -> mean & sigma proxy
    rf_point = predict_next_rf(rf_path, Xs, sy)
    rf_obj   = _load_rf(rf_path)
    q_scaled = rf_predict_quantiles(rf_obj, Xs[-1:,:])
    q_inv = {f"p{int(q*100)}": float(sy.inverse_transform([[q_scaled[q]]])[0,0]) for q in q_scaled}
    p50 = q_inv.get("p50", rf_point); p90 = q_inv.get("p90", rf_point); p95 = q_inv.get("p95", rf_point)
    sigma_proxy = max(1e-6, (p95 - p50) / 1.645)
    print(f"[predict:RF]    p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  (point={rf_point:.4f})")

    # ARIMA (baseline)
    yhat_arima = predict_next_arima(arima_path, sy)
    print(f"[predict:ARIMA] next={yhat_arima:.4f}")

    # prefer LSTM uncertainty if available, else RF quantile proxy
    if "sigma" not in signal or signal["sigma"] is None:
        signal.update({"model":"rf_quantile", "mean": p50, "sigma": sigma_proxy, "p95": p95})


    # --- UNIT NORMALIZATION (ms -> seconds) ---------------------------------
    # Convert only if the target column is actually in milliseconds
    # (e.g., latency_p95_ms or rt_p95_ms). 
    # For response_time_p95 (already in seconds), skip conversion.
    if args.target in ("latency_p95_ms", "rt_p95_ms"):
        if "mean" in signal and signal["mean"] is not None:
            signal["mean"] = signal["mean"] / 1000.0
        if "sigma" in signal and signal["sigma"] is not None:
            signal["sigma"] = signal["sigma"] / 1000.0
        for k in ("p50", "p90", "p95"):
            if k in signal and signal[k] is not None:
                signal[k] = signal[k] / 1000.0
        signal["units"] = "seconds"
    else:
        # Already in seconds, just label it
        signal["units"] = "seconds"

    # Save predictive signal for controller
    out_json = save_dir / "predictive_signal.json"
    with open(out_json, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"📝 saved predictive signal -> {out_json}")
    print("Prediction completed ✅ ")

    # 5) EVALUATE
    print("Evaluation started ...")
    if HAS_TF and lstm_model is not None and len(ys) > args.seq_len:
        actual, preds = rolling_eval_lstm(Xs, ys, lstm_model, args.seq_len, sy)
        print_scores("LSTM", actual, preds)
    else:
        print("[eval:LSTM] skipped")

    actual, preds = rolling_eval_rf(Xs, ys, rf_path, sy)
    print_scores("RF", actual, preds)

    actual, preds = rolling_eval_arima(ys, arima_path, sy)
    print_scores("ARIMA", actual, preds)
    print("Evaluation completed ✅ ")

    # --- Display last and next 5 workload predictions (LSTM) ---
    if HAS_TF and lstm_model is not None and len(ys) > args.seq_len:
        print("\n[LSTM] Workload Forecast Analysis:")
        # Get last 5 actual workload values
        last_actuals = sy.inverse_transform(ys[-5:].reshape(-1, 1)).reshape(-1)
        print("  Last 5 actual workload values (response_time_p95):")
        for i, val in enumerate(last_actuals, start=1):
            print(f"    t-{5 - i}: {val:.4f}")

        # Predict next 5 workload values recursively
        next_preds = []
        last_seq = Xs[-args.seq_len:, :].copy()
        for step in range(5):
            yhat_scaled = lstm_model.predict(last_seq.reshape(1, args.seq_len, -1), verbose=0)[0, 0]
            yhat = sy.inverse_transform([[yhat_scaled]])[0, 0]
            next_preds.append(yhat)
            # Slide window and append predicted value
            last_seq = np.roll(last_seq, -1, axis=0)
            last_seq[-1, 0] = yhat_scaled  # assuming first column is target feature

        print("\n  Predicted next 5 workload values:")
        for i, val in enumerate(next_preds, start=1):
            print(f"    t+{i}: {val:.4f}")
        print("LSTM workload forecast completed ✅\n")
    else:
        print("\n[LSTM] Workload forecast skipped (model unavailable or sequence too short)\n")

    # --- Display Random Forest prediction ---
    print("\n[RF] Workload Forecast Analysis:")
    last_actuals_rf = sy.inverse_transform(ys[-5:].reshape(-1, 1)).reshape(-1)
    print("  Last 5 actual workload values (response_time_p95):")
    for i, val in enumerate(last_actuals_rf, start=1):
        print(f"    t-{5 - i}: {val:.4f}")

    # Predict next single-step workload
    rf_point = predict_next_rf(rf_path, Xs, sy)
    rf_obj   = _load_rf(rf_path)
    q_scaled = rf_predict_quantiles(rf_obj, Xs[-1:,:])
    q_inv = {f"p{int(q*100)}": float(sy.inverse_transform([[q_scaled[q]]])[0,0]) for q in q_scaled}

    print("\n  Predicted next-step workload (Random Forest):")
    print(f"    p50 (median): {q_inv.get('p50', rf_point):.4f}")
    print(f"    p90 (upper bound): {q_inv.get('p90', rf_point):.4f}")
    print(f"    p95 (high confidence bound): {q_inv.get('p95', rf_point):.4f}")
    print("Random Forest next-step forecast completed ✅\n")


if __name__ == "__main__":
    import time

    while True:
        print("\n==============================")
        print("Running predictive pipeline update...")
        print("==============================")
        main()
        print("\n✅ Predictive signal updated successfully!")
        print("Next update in 60 seconds...\n")
        time.sleep(60)  # wait 1 minute before running again
