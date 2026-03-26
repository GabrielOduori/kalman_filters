"""
Demo: Kalman filter + RTS smoother on real NO2 data.
UCI Air Quality dataset — Italy.

Run:
    python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src import KalmanFilter1D, load_uci_airquality, calibrate_sensor


df         = load_uci_airquality()
sensor_cal = calibrate_sensor(df["no2_sensor"], df["no2_ref"])

R_ref, R_sensor = 4.0, 50.0

kf = KalmanFilter1D(
    x0=float(df["no2_ref"].iloc[0]), P0=20.0,
    A=1, H=1, Q=0.5, R=R_ref
)

for z_ref, z_lc in zip(df["no2_ref"], sensor_cal):
    kf.step([(z_ref, R_ref), (z_lc, R_sensor)])

smoothed = kf.smooth()

# --- Plot ---
t = np.arange(len(df))
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

ax = axes[0]
ax.plot(t, sensor_cal,    color="tomato",    alpha=0.4, linewidth=1,      label="Low-cost (calibrated)")
ax.plot(t, df["no2_ref"], color="steelblue", alpha=0.8, linewidth=1.5,    label="Reference analyser")
ax.plot(t, kf.estimates,  color="limegreen", linewidth=2.5,               label="Kalman filtered")
ax.plot(t, smoothed,      color="gold",      linewidth=2, linestyle="--", label="RTS smoothed")
ax.fill_between(t,
    kf.estimates - 2 * kf.uncertainty,
    kf.estimates + 2 * kf.uncertainty,
    alpha=0.15, color="limegreen", label="±2σ")
ax.set_ylabel("NO2 (μg/m³)")
ax.set_title("Real NO2 Data — UCI Air Quality Italy")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
err_lc = sensor_cal.values - df["no2_ref"].values
err_kf = kf.estimates      - df["no2_ref"].values
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.plot(t, err_lc, color="tomato",    alpha=0.5, linewidth=1,
        label=f"Low-cost  RMSE = {np.sqrt(np.mean(err_lc**2)):.1f} μg/m³")
ax.plot(t, err_kf, color="limegreen", linewidth=1.5,
        label=f"Kalman    RMSE = {np.sqrt(np.mean(err_kf**2)):.1f} μg/m³")
ax.set_xlabel("Hour")
ax.set_ylabel("Error (μg/m³)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("real_fusion.png", dpi=150)
plt.show()
