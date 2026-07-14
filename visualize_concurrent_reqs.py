"""Visualize concurrent_reqs.csv — SGLang engine concurrency & throughput over time."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = "/workspace/slime/concurrent_reqs.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# Split into sessions by running_req==0 boundaries
session_breaks = df[df["running_req"] == 0].index
sessions = []
prev = 0
for brk in session_breaks:
    seg = df.iloc[prev:brk]
    if len(seg) > 0:
        sessions.append(seg)
    prev = brk + 1  # skip the zero row
if prev < len(df):
    sessions.append(df.iloc[prev:])

fig, axes = plt.subplots(len(sessions), 2, figsize=(16, 4 * len(sessions)),
                          squeeze=False, sharex=False)
fig.suptitle("SGLang Rollout Engine — Concurrency & Throughput per Session",
             fontsize=14, fontweight="bold", y=1.01)

for i, sess in enumerate(sessions):
    t = sess["timestamp"]
    left = axes[i, 0]
    right = axes[i, 1]

    # --- Left: running requests (area) + token usage (line) ---
    left.fill_between(t, sess["running_req"], alpha=0.3, color="steelblue", label="running_req")
    left.plot(t, sess["running_req"], color="steelblue", linewidth=1.2)
    left.set_ylabel("Running Requests", color="steelblue")
    left.tick_params(axis="y", labelcolor="steelblue")

    ax2 = left.twinx()
    ax2.plot(t, sess["token_usage"], color="coral", linewidth=1.2, marker=".", markersize=4, label="token_usage")
    ax2.set_ylabel("Token Usage", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(-0.02, max(sess["token_usage"].max() * 1.3, 0.4))

    left.set_title(f"Session {i + 1}  ({t.iloc[0].strftime('%H:%M:%S')} ~ {t.iloc[-1].strftime('%H:%M:%S')})")
    left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    left.tick_params(axis="x", rotation=30)

    lines1, labels1 = left.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    left.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # --- Right: throughput ---
    right.plot(t, sess["throughput"], color="seagreen", linewidth=1.4, marker="o", markersize=4)
    right.fill_between(t, sess["throughput"], alpha=0.2, color="seagreen")
    right.set_ylabel("Throughput (tokens/s)")
    right.set_title(f"Session {i + 1} — Throughput")
    right.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    right.tick_params(axis="x", rotation=30)

    # Annotate peak
    peak_idx = sess["throughput"].idxmax()
    peak_val = sess["throughput"].loc[peak_idx]
    peak_time = sess["timestamp"].loc[peak_idx]
    right.annotate(f"peak: {peak_val:.0f} tok/s",
                   xy=(peak_time, peak_val),
                   xytext=(15, 10), textcoords="offset points",
                   fontsize=8, color="darkgreen",
                   arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8))

plt.tight_layout()
out_path = "/workspace/slime/concurrent_reqs_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
