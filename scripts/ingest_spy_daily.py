def fetch_spy_daily():
    sym = SYMBOL[0] if isinstance(SYMBOL, (list, tuple)) else SYMBOL

    df = yf.download(sym, period="2y", interval="1d", auto_adjust=False, progress=False)
    df = df.reset_index()

    # tuple-safe column normalization (handles MultiIndex from yfinance)
    def norm(c):
        if isinstance(c, tuple):
            c = "_".join(str(x) for x in c if x not in (None, ""))
        return str(c).strip().lower().replace(" ", "_")

        df.columns = [norm(c) for c in df.columns]

    # pick correct columns across yfinance variants
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(f"Missing columns. Have: {df.columns.tolist()}")

    date_col = pick("date", "datetime")
    open_col = pick("open", "spy_open", "open_spy")
    high_col = pick("high", "spy_high", "high_spy")
    low_col  = pick("low",  "spy_low",  "low_spy")
    close_col= pick("close","spy_close","close_spy")
    vol_col  = pick("volume","spy_volume","volume_spy")

    df[date_col] = pd.to_datetime(df[date_col]).dt.date.astype(str)

    out = df[[date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]

    out["symbol"] = sym
    out["source"] = "yfinance"
    out["ingest_ts"] = datetime.now(timezone.utc).isoformat()

    return out
