import pandas as pd

inp = "/Users/benni/Documents/personal/work/money/trading/bots/25_dec_13/forward equal to backward/live_1m.csv"
out = "/Users/benni/Documents/personal/work/money/trading/bots/25_dec_13/forward equal to backward/live_3m_forward.csv"

df = pd.read_csv(inp)
# handle EpochMs if present
if "EpochMs" in df.columns:
    df["Epoch"] = (df["EpochMs"] // 1000).astype(int)
else:
    df["Epoch"] = df["Epoch"].astype(int)

rows = []
buf = []

for _, row in df.iterrows():
    buf.append(row)
    if len(buf) == 3:
        rows.append({
            "Epoch": int(buf[0]["Epoch"]),
            "Open": float(buf[0]["Open"]),
            "High": float(max(b["High"] for b in buf)),
            "Low": float(min(b["Low"] for b in buf)),
            "Close": float(buf[-1]["Close"]),
            "Volume": float(sum(b["Volume"] for b in buf)),
        })
        buf = []

pd.DataFrame(rows).to_csv(out, index=False)
print("Wrote", out)
