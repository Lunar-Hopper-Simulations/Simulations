
"""
LUNAR HOPPER RK4 TRAJECTORY SIMULATOR V7
=======================================

V7 uses Runge-Kutta 4th order integration to generate:
- trajectory state history
- velocity profile
- acceleration / acceleration-demand profile
- trajectory plot
- velocity graph
- acceleration graph
- 3D dynamic-camera video/GIF
- CSV outputs

The simulator starts with CALCULATION-1 values.
Use the GUI preset button to switch to CALCULATIONS-2 (1).
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from matplotlib.animation import FFMpegWriter
    HAS_FFMPEG = True
except Exception:
    HAS_FFMPEG = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TERRAIN = os.path.join(SCRIPT_DIR, "terrain_base_to_isru_clean.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "rk4_outputs")


PRESETS = {
    "CALCULATION-1": {
        "segment_mode": "same_corridor",
        "g_moon": 1.62,
        "g0": 9.80665,
        "base_x": 0.0,
        "base_h": 1261.2,
        "isru_x": 12933.01,
        "isru_h": -2790.0,
        "one_vx": 71.85,
        "two1_x0": 0.0, "two1_h0": 1261.2, "two1_x1": 6466.51, "two1_h1": -764.4, "two1_T": 90.0,
        "two2_x0": 6466.51, "two2_h0": -764.4, "two2_x1": 12933.01, "two2_h1": -2790.0, "two2_T": 90.0,
        "three1_x0": 0.0, "three1_h0": 1261.2, "three1_x1": 4311.0, "three1_h1": -89.2, "three1_T": 60.0,
        "three2_x0": 4311.0, "three2_h0": -89.2, "three2_x1": 8622.01, "three2_h1": -1439.6, "three2_T": 60.0,
        "three3_x0": 8622.01, "three3_h0": -1439.6, "three3_x1": 12933.01, "three3_h1": -2790.0, "three3_T": 60.0,
        "dry_mass": 900.0,
        "payload_mass": 650.0,
        "propellant_mass": 441.07,
        "isp": 450.0,
        "thrust": 8060.0,
    },
    "CALCULATIONS-2 (1)": {
        "segment_mode": "segmented",
        "g_moon": 1.62,
        "g0": 9.80665,
        "base_x": 0.0,
        "base_h": 1261.2,
        "isru_x": 12933.01,
        "isru_h": -2790.0,
        "one_vx": 115.9,
        "two1_x0": 0.0, "two1_h0": 1261.2, "two1_x1": 6467.0, "two1_h1": -603.3, "two1_T": 123.2,
        "two2_x0": 6467.0, "two2_h0": -603.3, "two2_x1": 12933.01, "two2_h1": -2790.0, "two2_T": 100.6,
        "three1_x0": 0.0, "three1_h0": 1261.2, "three1_x1": 4311.0, "three1_h1": 689.6, "three1_T": 96.6,
        "three2_x0": 4311.0, "three2_h0": 689.6, "three2_x1": 8622.0, "three2_h1": -1834.4, "three2_T": 71.3,
        "three3_x0": 8622.0, "three3_h0": -1834.4, "three3_x1": 12933.01, "three3_h1": -2790.0, "three3_T": 86.1,
        "dry_mass": 900.0,
        "payload_mass": 650.0,
        "propellant_mass": 423.3,
        "isp": 450.0,
        "thrust": 8060.0,
    }
}


SCENARIO_STYLE = {
    "Base to ISRU | 1-hop RK4": ("#00FFFF", "o"),
    "ISRU to Base | 1-hop RK4": ("#FFA500", "D"),
    "ISRU to Base | 2-hop RK4": ("#7CFC00", "^"),
    "ISRU to Base | 3-hop RK4": ("#FF00FF", "*"),
}


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Terrain file must be CSV, XLSX, or XLS.")


def load_terrain(path, distance_col, elevation_col, distance_unit, elevation_unit):
    df = read_table(path)
    if distance_col not in df.columns:
        raise ValueError(f"Distance column '{distance_col}' not found. Available: {list(df.columns)}")
    if elevation_col not in df.columns:
        raise ValueError(f"Elevation column '{elevation_col}' not found. Available: {list(df.columns)}")

    x = df[distance_col].to_numpy(dtype=float)
    z = df[elevation_col].to_numpy(dtype=float)

    good = np.isfinite(x) & np.isfinite(z)
    x = x[good]
    z = z[good]

    if distance_unit.lower().strip() == "km":
        x *= 1000.0
    if elevation_unit.lower().strip() == "km":
        z *= 1000.0

    idx = np.argsort(x)
    return x[idx], z[idx]


def rk4_step(state, dt, g):
    """
    RK4 step for ballistic lunar motion.
    State = [x, h, vx, vy]
    """
    def f(s):
        x, h, vx, vy = s
        return np.array([vx, vy, 0.0, -g], dtype=float)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_segment(x0, h0, x1, h1, T, g, n_steps):
    if T <= 0:
        raise ValueError("Segment time must be positive.")

    n_steps = max(20, int(n_steps))
    dt = T / n_steps

    vx0 = (x1 - x0) / T
    vy0 = ((h1 - h0) + 0.5 * g * T**2) / T

    state = np.array([x0, h0, vx0, vy0], dtype=float)

    time = [0.0]
    xs = [x0]
    hs = [h0]
    vxs = [vx0]
    vys = [vy0]

    for i in range(n_steps):
        state = rk4_step(state, dt, g)
        time.append((i + 1) * dt)
        xs.append(state[0])
        hs.append(state[1])
        vxs.append(state[2])
        vys.append(state[3])

    time = np.array(time)
    xs = np.array(xs)
    hs = np.array(hs)
    vxs = np.array(vxs)
    vys = np.array(vys)
    speed = np.sqrt(vxs**2 + vys**2)
    acc = np.full_like(time, g)

    peak_i = int(np.argmax(hs))

    return {
        "time": time,
        "x": xs,
        "h": hs,
        "vx": vxs,
        "vy": vys,
        "speed": speed,
        "acc": acc,
        "peak_x": xs[peak_i],
        "peak_h": hs[peak_i],
        "vx0": vx0,
        "vy0": vy0,
    }


def rk4_one_hop(name, x0, h0, x1, h1, horizontal_velocity, g, n_steps):
    T = abs(x1 - x0) / abs(horizontal_velocity)
    seg = rk4_segment(x0, h0, x1, h1, T, g, n_steps)

    return {
        "name": name,
        "mode": "RK4_one_hop",
        "segments": 1,
        "time": seg["time"],
        "x": seg["x"],
        "h": seg["h"],
        "vx": seg["vx"],
        "vy": seg["vy"],
        "speed": seg["speed"],
        "acceleration": seg["acc"],
        "waypoint_x": np.array([x0, x1]),
        "waypoint_h": np.array([h0, h1]),
        "peak_x": np.array([seg["peak_x"]]),
        "peak_h": np.array([seg["peak_h"]]),
        "total_time": T,
    }


def rk4_segmented(name, rows, g, n_steps_per_segment):
    all_t, all_x, all_h, all_vx, all_vy, all_v, all_a = [], [], [], [], [], [], []
    wp_x, wp_h, peaks_x, peaks_h = [], [], [], []
    t_offset = 0.0

    for i, row in enumerate(rows):
        x0, h0, x1, h1, T = row
        seg = rk4_segment(x0, h0, x1, h1, T, g, n_steps_per_segment)
        t = seg["time"] + t_offset

        if i > 0:
            t = t[1:]
            x = seg["x"][1:]
            h = seg["h"][1:]
            vx = seg["vx"][1:]
            vy = seg["vy"][1:]
            v = seg["speed"][1:]
            a = seg["acc"][1:]
        else:
            x = seg["x"]
            h = seg["h"]
            vx = seg["vx"]
            vy = seg["vy"]
            v = seg["speed"]
            a = seg["acc"]

        all_t.extend(t)
        all_x.extend(x)
        all_h.extend(h)
        all_vx.extend(vx)
        all_vy.extend(vy)
        all_v.extend(v)
        all_a.extend(a)

        if i == 0:
            wp_x.append(x0)
            wp_h.append(h0)
        wp_x.append(x1)
        wp_h.append(h1)

        peaks_x.append(seg["peak_x"])
        peaks_h.append(seg["peak_h"])

        t_offset = float(t[-1])

    return {
        "name": name,
        "mode": "RK4_segmented",
        "segments": len(rows),
        "time": np.array(all_t),
        "x": np.array(all_x),
        "h": np.array(all_h),
        "vx": np.array(all_vx),
        "vy": np.array(all_vy),
        "speed": np.array(all_v),
        "acceleration": np.array(all_a),
        "waypoint_x": np.array(wp_x),
        "waypoint_h": np.array(wp_h),
        "peak_x": np.array(peaks_x),
        "peak_h": np.array(peaks_h),
        "total_time": t_offset,
    }


def same_corridor_profile(name, reference, control_segments):
    p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in reference.items()}
    p["name"] = name
    p["mode"] = "RK4_same_corridor_mid_air_control"
    p["segments"] = control_segments

    fractions = np.linspace(0, 1, control_segments + 1)
    wp_t = reference["time"][0] + fractions * (reference["time"][-1] - reference["time"][0])
    p["waypoint_x"] = np.interp(wp_t, reference["time"], reference["x"])
    p["waypoint_h"] = np.interp(wp_t, reference["time"], reference["h"])

    peak_i = int(np.argmax(reference["h"]))
    p["peak_x"] = np.array([reference["x"][peak_i]])
    p["peak_h"] = np.array([reference["h"][peak_i]])
    return p


def powered_acceleration_demand(profile, g, ascent_extra, correction_extra, landing_extra):
    s = (profile["time"] - profile["time"][0]) / max(1e-9, profile["time"][-1] - profile["time"][0])
    a = np.full_like(s, g)
    a += ascent_extra * np.exp(-((s - 0.07) / 0.050)**2)
    a += landing_extra * np.exp(-((s - 0.93) / 0.055)**2)

    if profile["segments"] == 2:
        a += correction_extra * np.exp(-((s - 0.50) / 0.035)**2)
    elif profile["segments"] == 3:
        a += correction_extra * np.exp(-((s - 1/3) / 0.035)**2)
        a += correction_extra * np.exp(-((s - 2/3) / 0.035)**2)

    return a


def profile_color(name):
    return SCENARIO_STYLE.get(name, ("white", "o"))[0]


def profile_marker(name):
    return SCENARIO_STYLE.get(name, ("white", "o"))[1]


def save_trajectory_graph(outdir, terrain_x, terrain_z, profiles, clearance):
    path = os.path.join(outdir, "01_rk4_trajectory_profile.png")
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(terrain_x/1000, terrain_z, color="dimgray", linewidth=2.8, label="CSV lunar terrain")
    ax.plot(terrain_x/1000, terrain_z + clearance, color="gray", linestyle="--", linewidth=1.2, label=f"{clearance:.0f} m clearance reference")

    for p in profiles:
        color = profile_color(p["name"])
        marker = profile_marker(p["name"])
        ax.plot(p["x"]/1000, p["h"], color=color, linewidth=2.6, label=p["name"])
        ax.scatter(p["waypoint_x"]/1000, p["waypoint_h"], color="white", edgecolor="black", s=120, marker=marker, zorder=5)
        ax.scatter(p["waypoint_x"]/1000, p["waypoint_h"], color=color, edgecolor="black", s=65, marker=marker, zorder=6)

    base_x = profiles[0]["waypoint_x"][0]
    base_h = profiles[0]["waypoint_h"][0]
    isru_x = profiles[0]["waypoint_x"][-1]
    isru_h = profiles[0]["waypoint_h"][-1]

    ax.scatter([base_x/1000], [base_h], s=160, marker="^", color="gold", edgecolor="black", label="Base")
    ax.scatter([isru_x/1000], [isru_h], s=160, marker="s", color="red", edgecolor="black", label="ISRU")

    ax.set_title("RK4 Lunar Hopper Trajectories over CSV Terrain", fontsize=15, weight="bold")
    ax.set_xlabel("Distance from Base along route (km)")
    ax.set_ylabel("Altitude / elevation (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_velocity_graph(outdir, terrain_x, terrain_z, profiles):
    path = os.path.join(outdir, "02_rk4_professional_velocity_profile.png")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.15]})

    ax_top.plot(terrain_x/1000, terrain_z, color="dimgray", linewidth=2.5, label="CSV lunar terrain")
    for p in profiles:
        ax_top.plot(p["x"]/1000, p["h"], color=profile_color(p["name"]), linewidth=2.0, label=p["name"])

    ax_top.set_ylabel("Altitude / elevation (m)")
    ax_top.set_title("RK4 Trajectory with Velocity Profile")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", ncol=2, fontsize=8)

    for p in profiles:
        ax_bot.plot(p["x"]/1000, p["speed"], color=profile_color(p["name"]), linewidth=2.0, label=p["name"])

    ax_bot.axhline(183.0, color="black", linestyle="--", linewidth=1.4, label="183 m/s reference")
    ax_bot.axhline(200.0, color="gray", linestyle=":", linewidth=1.4, label="200 m/s suggested limit")
    ax_bot.axhline(5.0, color="tab:blue", linestyle="-.", linewidth=1.4, label="5 m/s soft landing target")
    ax_bot.set_xlabel("Distance from Base along route (km)")
    ax_bot.set_ylabel("Velocity magnitude (m/s)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_acceleration_graph(outdir, terrain_x, terrain_z, profiles, g, g0, clearance, ascent_extra, correction_extra, landing_extra):
    path = os.path.join(outdir, "03_rk4_professional_acceleration_profile.png")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.15]})

    ax_top.plot(terrain_x/1000, terrain_z, color="dimgray", linewidth=2.5, label="CSV lunar terrain")
    ax_top.plot(terrain_x/1000, terrain_z + clearance, color="gray", linestyle="--", linewidth=1.1, label=f"{clearance:.0f} m clearance")
    for p in profiles:
        ax_top.plot(p["x"]/1000, p["h"], color=profile_color(p["name"]), linewidth=2.0, label=p["name"])

    ax_top.set_ylabel("Altitude / elevation (m)")
    ax_top.set_title("RK4 Trajectory with Powered Acceleration Demand")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", ncol=2, fontsize=8)

    for p in profiles:
        demand = powered_acceleration_demand(p, g, ascent_extra, correction_extra, landing_extra)
        ax_bot.plot(p["x"]/1000, demand, color=profile_color(p["name"]), linewidth=2.0, label=p["name"])

    ax_bot.axhline(g, color="black", linestyle="--", linewidth=1.4, label="Lunar gravity")
    ax_bot.axhline(g0, color="gray", linestyle=":", linewidth=1.4, label="1 Earth-g")
    ax_bot.axhline(3*g0, color="tab:red", linestyle="-.", linewidth=1.4, label="3 g limit")

    ax_bot.set_xlabel("Distance from Base along route (km)")
    ax_bot.set_ylabel("Acceleration demand (m/s²)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right", ncol=2, fontsize=8)

    ax_g = ax_bot.twinx()
    ymin, ymax = ax_bot.get_ylim()
    ax_g.set_ylim(ymin/g0, ymax/g0)
    ax_g.set_ylabel("Equivalent g-load (Earth-g)")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def resample_profile(p, frames):
    t_new = np.linspace(p["time"][0], p["time"][-1], frames)
    return {
        "name": p["name"],
        "x": np.interp(t_new, p["time"], p["x"]),
        "h": np.interp(t_new, p["time"], p["h"]),
        "speed": np.interp(t_new, p["time"], p["speed"]),
    }


def save_3d_video(outdir, terrain_x, terrain_z, profiles, frames, save_gif, save_mp4):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    frames = max(120, int(frames))
    terrain_km = terrain_x / 1000.0
    lanes = [-0.66, -0.22, 0.22, 0.66]
    lane_map = {p["name"]: lanes[i] for i, p in enumerate(profiles)}
    rs = [resample_profile(p, frames) for p in profiles]

    y_vals = np.linspace(-1.05, 1.05, 28)
    X, Y = np.meshgrid(terrain_km, y_vals)
    Z = np.tile(terrain_z, (len(y_vals), 1)) + 45.0 * np.cos(2*np.pi*Y/2.1)

    fig = plt.figure(figsize=(15, 9), facecolor="#04070b")
    ax = fig.add_subplot(111, projection="3d", facecolor="#04070b")

    try:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((0.03, 0.04, 0.08, 1.0))
            axis.pane.set_edgecolor((0.5, 0.5, 0.5, 0.35))
    except Exception:
        pass

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.90, linewidth=0, antialiased=True, shade=True)
    ax.plot(terrain_km, np.zeros_like(terrain_km), terrain_z, color="white", linewidth=1.3, alpha=0.92, label="Terrain centerline")

    for p in profiles:
        color = profile_color(p["name"])
        marker = profile_marker(p["name"])
        lane = lane_map[p["name"]]
        ax.plot(p["x"]/1000, np.full_like(p["x"], lane), p["h"], color=color, linewidth=1.9, alpha=0.88, label=p["name"])
        wp_x = p["waypoint_x"]/1000
        wp_y = np.full_like(p["waypoint_x"], lane)
        wp_h = p["waypoint_h"]
        ax.scatter(wp_x, wp_y, wp_h, color="white", marker=marker, s=180, edgecolors="black", linewidths=1.0)
        ax.scatter(wp_x, wp_y, wp_h, color=color, marker=marker, s=84, edgecolors="black", linewidths=1.0)

    base_x = profiles[0]["waypoint_x"][0]/1000
    base_h = profiles[0]["waypoint_h"][0]
    isru_x = profiles[0]["waypoint_x"][-1]/1000
    isru_h = profiles[0]["waypoint_h"][-1]
    ax.scatter([base_x], [0], [base_h+80], s=380, marker="^", color="white", edgecolors="black")
    ax.scatter([base_x], [0], [base_h+80], s=230, marker="^", color="gold", edgecolors="black", label="BASE marker")
    ax.scatter([isru_x], [0], [isru_h+80], s=380, marker="s", color="white", edgecolors="black")
    ax.scatter([isru_x], [0], [isru_h+80], s=230, marker="s", color="red", edgecolors="black", label="ISRU marker")
    ax.text(base_x, 0.07, base_h+320, "BASE", color="gold", fontsize=12, weight="bold")
    ax.text(isru_x, 0.07, isru_h+320, "ISRU", color="red", fontsize=12, weight="bold")

    movers = []
    trails = []
    for r in rs:
        color = profile_color(r["name"])
        marker = profile_marker(r["name"])
        halo, = ax.plot([], [], [], marker=marker, markersize=18, linestyle="None", color="white", markeredgecolor="black")
        core, = ax.plot([], [], [], marker=marker, markersize=11.5, linestyle="None", color=color, markeredgecolor="black")
        trail, = ax.plot([], [], [], linewidth=3.3, color=color)
        movers.append((r, halo, core))
        trails.append((r, trail))

    fig.text(0.02, 0.975, "RK4 LUNAR HOPPER SIMULATION", color="white", fontsize=14, weight="bold")
    fig.text(0.02, 0.03, "RK4 trajectory mode | Colored terrain | High-visibility neon markers | Live velocity table", color="white", fontsize=10)
    info_text = fig.text(0.02, 0.90, "", color="white", fontsize=10, family="monospace", bbox=dict(facecolor="black", alpha=0.42, edgecolor="white"))
    table_text = fig.text(0.70, 0.68, "", color="white", fontsize=10, family="monospace", bbox=dict(facecolor="black", alpha=0.62, edgecolor="white"))

    ax.set_xlabel("Distance from Base (km)", labelpad=12)
    ax.set_ylabel("Scenario lane", labelpad=12)
    ax.set_zlabel("Altitude (m)", labelpad=12)
    ax.set_title("Professional 3D RK4 Lunar Hopper Trajectory", pad=18)
    ax.set_xlim(float(np.min(terrain_km)), float(np.max(terrain_km)))
    ax.set_ylim(-1.15, 1.15)
    zmin = min(float(np.min(terrain_z)), min(float(np.min(p["h"])) for p in profiles)) - 450
    zmax = max(float(np.max(terrain_z)), max(float(np.max(p["h"])) for p in profiles)) + 950
    ax.set_zlim(zmin, zmax)

    legend = ax.legend(loc="upper left", fontsize=7.6, framealpha=0.58)
    for t in legend.get_texts():
        t.set_color("white")

    def camera(frame):
        q = frame / max(1, frames - 1)
        if q < 0.25:
            local = q / 0.25
            return 18 + 10*np.sin(np.pi*local/2), -75 + 45*local
        if q < 0.55:
            local = (q - 0.25)/0.30
            return 24 + 10*np.sin(np.pi*local), -30 + 175*local
        if q < 0.80:
            local = (q - 0.55)/0.25
            return 32 + 8*np.sin(np.pi*local), 145 - 100*local
        local = (q - 0.80)/0.20
        return 34 + 18*local, 45 + 120*local

    def update(frame):
        elev, azim = camera(frame)
        ax.view_init(elev=elev, azim=azim)

        table_rows = []
        artists = []

        for r, halo, core in movers:
            y = lane_map[r["name"]]
            x = r["x"][frame]/1000
            h = r["h"][frame]
            halo.set_data([x], [y])
            halo.set_3d_properties([h])
            core.set_data([x], [y])
            core.set_3d_properties([h])
            artists.extend([halo, core])
            table_rows.append((r["name"], r["speed"][frame]))

        for r, trail in trails:
            y = np.full(frame+1, lane_map[r["name"]])
            x = r["x"][:frame+1]/1000
            h = r["h"][:frame+1]
            trail.set_data(x, y)
            trail.set_3d_properties(h)
            artists.append(trail)

        progress = 100*frame/max(1, frames-1)
        info_text.set_text(
            "Simulation status\n"
            f"Progress : {progress:5.1f}%\n"
            f"Camera   : elev {elev:5.1f}° | azim {azim:6.1f}°\n"
            f"Base alt : {base_h:8.2f} m\n"
            f"ISRU alt : {isru_h:8.2f} m"
        )

        table = "LIVE VELOCITY TABLE\n"
        table += "Color key: Cyan / Orange / Lime / Magenta\n"
        table += "Marker | Scenario                 | Velocity\n"
        table += "-------+--------------------------+-----------\n"
        symbols = ["●", "◆", "▲", "★"]
        for sym, (name, vel) in zip(symbols, table_rows):
            table += f"{sym:6s} | {name[:24]:24s} | {vel:8.2f} m/s\n"
        table_text.set_text(table)
        artists.extend([info_text, table_text])
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=55, blit=False)

    saved = []
    if save_gif:
        gif_path = os.path.join(outdir, "04_rk4_professional_3d_simulation_v7.gif")
        ani.save(gif_path, writer=PillowWriter(fps=18))
        saved.append(gif_path)

    if save_mp4 and HAS_FFMPEG:
        try:
            mp4_path = os.path.join(outdir, "04_rk4_professional_3d_simulation_v7.mp4")
            ani.save(mp4_path, writer=FFMpegWriter(fps=18, bitrate=3000))
            saved.append(mp4_path)
        except Exception:
            pass

    plt.close(fig)
    return saved


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Hopper RK4 Trajectory Simulator V7")
        self.root.geometry("1160x900")
        self.entries = {}
        self.terrain_file = tk.StringVar(value=DEFAULT_TERRAIN if os.path.exists(DEFAULT_TERRAIN) else "")
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT)
        self.preset_name = tk.StringVar(value="CALCULATION-1")
        self.segment_mode = tk.StringVar(value="same_corridor")
        self.save_gif = tk.BooleanVar(value=True)
        self.save_mp4 = tk.BooleanVar(value=True)
        self.build()

    def build(self):
        tk.Label(self.root, text="Lunar Hopper RK4 Trajectory Simulator V7", font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(self.root, text="Runge-Kutta 4th order integration for trajectory, velocity, graphs, CSV, and 3D video", font=("Arial", 10)).pack()

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        canvas = tk.Canvas(frame)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.section("1. Terrain CSV / Excel")
        self.file_row("Terrain file", self.terrain_file, self.browse_terrain)
        self.entry("distance_col", "Distance column", "distance_m")
        self.entry("elevation_col", "Elevation column", "elevation_m")
        self.entry("distance_unit", "Distance unit (m or km)", "m")
        self.entry("elevation_unit", "Elevation unit (m or km)", "m")
        self.file_row("Output folder", self.output_dir, self.browse_output)

        self.section("2. Word-file preset loader")
        self.preset_row()

        self.section("3. RK4 route and physics")
        self.entry("g_moon", "Lunar gravity (m/s²)", "1.62")
        self.entry("g0", "Earth g0 for g-load axis (m/s²)", "9.80665")
        self.entry("base_x", "Base distance from route start (m)", "0")
        self.entry("base_h", "Base elevation (m)", "1261.2")
        self.entry("isru_x", "ISRU distance from Base (m)", "12933.01")
        self.entry("isru_h", "ISRU elevation (m)", "-2790.0")
        self.entry("clearance", "Clearance reference line (m)", "100")
        self.entry("one_vx", "1-hop horizontal velocity (m/s)", "71.85")
        self.entry("points_one", "RK4 steps for 1-hop", "900")
        self.entry("points_segment", "RK4 steps per segment", "350")

        self.section("4. RK4 2-hop inputs, Base→ISRU table basis")
        self.segment_entries("two1", "2-hop segment 1", "0", "1261.2", "6466.51", "-764.4", "90.0")
        self.segment_entries("two2", "2-hop segment 2", "6466.51", "-764.4", "12933.01", "-2790.0", "90.0")

        self.section("5. RK4 3-hop inputs, Base→ISRU table basis")
        self.segment_entries("three1", "3-hop segment 1", "0", "1261.2", "4311.0", "-89.2", "60.0")
        self.segment_entries("three2", "3-hop segment 2", "4311.0", "-89.2", "8622.01", "-1439.6", "60.0")
        self.segment_entries("three3", "3-hop segment 3", "8622.01", "-1439.6", "12933.01", "-2790.0", "60.0")

        self.section("6. Acceleration graph demand model")
        self.entry("ascent_extra", "Extra acceleration peak during ascent (m/s²)", "3.4")
        self.entry("correction_extra", "Extra acceleration peak during correction burn (m/s²)", "1.1")
        self.entry("landing_extra", "Extra acceleration peak during landing braking (m/s²)", "4.8")

        self.section("7. Vehicle values for summary only")
        self.entry("dry_mass", "Dry mass (kg)", "900")
        self.entry("payload_mass", "Payload mass (kg)", "650")
        self.entry("propellant_mass", "Propellant mass (kg)", "441.07")
        self.entry("isp", "Specific impulse Isp (s)", "450")
        self.entry("thrust", "Total engine thrust (N)", "8060")

        self.section("8. Professional video settings")
        self.entry("animation_frames", "Animation frames, 420 recommended", "420")
        tk.Checkbutton(self.form, text="Save GIF video", variable=self.save_gif).grid(row=9991, column=0, sticky="w", padx=6, pady=3)
        tk.Checkbutton(self.form, text="Try saving MP4 if ffmpeg exists", variable=self.save_mp4).grid(row=9992, column=0, sticky="w", padx=6, pady=3)

        tk.Button(self.form, text="RUN RK4 TRAJECTORY SIMULATION V7", command=self.run, bg="#2e7d32", fg="white", font=("Arial", 13, "bold"), height=2).grid(row=9993, column=0, columnspan=3, sticky="ew", padx=6, pady=16)

        self.logbox = tk.Text(self.form, height=20, width=124)
        self.logbox.grid(row=9994, column=0, columnspan=3, sticky="ew", padx=6, pady=8)

    def section(self, text):
        row = len(self.form.grid_slaves()) + 1
        tk.Label(self.form, text=text, font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 6), padx=6)

    def entry(self, key, text, default):
        row = len(self.form.grid_slaves()) + 1
        tk.Label(self.form, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        var = tk.StringVar(value=str(default))
        tk.Entry(self.form, textvariable=var, width=34).grid(row=row, column=1, sticky="w", padx=6, pady=3)
        self.entries[key] = var

    def file_row(self, text, var, command):
        row = len(self.form.grid_slaves()) + 1
        tk.Label(self.form, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        tk.Entry(self.form, textvariable=var, width=65).grid(row=row, column=1, sticky="w", padx=6, pady=3)
        tk.Button(self.form, text="Browse", command=command).grid(row=row, column=2, sticky="w", padx=6, pady=3)

    def segment_entries(self, prefix, label, x0, h0, x1, h1, T):
        self.entry(f"{prefix}_x0", f"{label}: start distance x0 (m)", x0)
        self.entry(f"{prefix}_h0", f"{label}: start elevation h0 (m)", h0)
        self.entry(f"{prefix}_x1", f"{label}: end distance x1 (m)", x1)
        self.entry(f"{prefix}_h1", f"{label}: end elevation h1 (m)", h1)
        self.entry(f"{prefix}_T", f"{label}: segment time T (s)", T)

    def preset_row(self):
        row = len(self.form.grid_slaves()) + 1
        tk.Label(self.form, text="Select Word-file preset").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        combo = ttk.Combobox(self.form, textvariable=self.preset_name, values=list(PRESETS.keys()), state="readonly", width=34)
        combo.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        tk.Button(self.form, text="Load selected preset", command=lambda: self.apply_preset(self.preset_name.get())).grid(row=row, column=2, sticky="w", padx=6, pady=3)

        row = len(self.form.grid_slaves()) + 1
        tk.Button(self.form, text="Load CALCULATION-1", command=lambda: self.apply_preset("CALCULATION-1")).grid(row=row, column=0, sticky="ew", padx=6, pady=3)
        tk.Button(self.form, text="Load CALCULATIONS-2 (1)", command=lambda: self.apply_preset("CALCULATIONS-2 (1)")).grid(row=row, column=1, sticky="ew", padx=6, pady=3)

        row = len(self.form.grid_slaves()) + 1
        tk.Label(self.form, text="2-hop / 3-hop mode").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        combo_mode = ttk.Combobox(self.form, textvariable=self.segment_mode, values=["same_corridor", "segmented"], state="readonly", width=34)
        combo_mode.grid(row=row, column=1, sticky="w", padx=6, pady=3)

    def browse_terrain(self):
        p = filedialog.askopenfilename(title="Select terrain file", filetypes=[("Terrain files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if p:
            self.terrain_file.set(p)

    def browse_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def f(self, key):
        return float(self.entries[key].get())

    def i(self, key):
        return int(float(self.entries[key].get()))

    def log(self, text):
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")
        self.root.update_idletasks()

    def apply_preset(self, preset):
        data = PRESETS[preset]
        self.preset_name.set(preset)
        self.segment_mode.set(data["segment_mode"])
        for key, value in data.items():
            if key in self.entries:
                self.entries[key].set(str(value))
        self.log(f"Loaded preset: {preset}")

    def get_segment(self, prefix):
        return (self.f(f"{prefix}_x0"), self.f(f"{prefix}_h0"), self.f(f"{prefix}_x1"), self.f(f"{prefix}_h1"), self.f(f"{prefix}_T"))

    def reverse_segment(self, seg):
        x0, h0, x1, h1, T = seg
        return (x1, h1, x0, h0, T)

    def run(self):
        self.logbox.delete("1.0", "end")
        try:
            outdir = self.output_dir.get().strip()
            os.makedirs(outdir, exist_ok=True)

            self.log("Loading terrain...")
            terrain_x, terrain_z = load_terrain(
                self.terrain_file.get().strip(),
                self.entries["distance_col"].get().strip(),
                self.entries["elevation_col"].get().strip(),
                self.entries["distance_unit"].get().strip(),
                self.entries["elevation_unit"].get().strip()
            )

            g = self.f("g_moon")
            g0 = self.f("g0")
            base_x = self.f("base_x")
            base_h = self.f("base_h")
            isru_x = self.f("isru_x")
            isru_h = self.f("isru_h")
            clearance = self.f("clearance")

            self.log("Integrating RK4 trajectories...")

            inbound_1 = rk4_one_hop("Base to ISRU | 1-hop RK4", base_x, base_h, isru_x, isru_h, self.f("one_vx"), g, self.i("points_one"))
            return_1 = rk4_one_hop("ISRU to Base | 1-hop RK4", isru_x, isru_h, base_x, base_h, self.f("one_vx"), g, self.i("points_one"))

            if self.segment_mode.get() == "same_corridor":
                return_2 = same_corridor_profile("ISRU to Base | 2-hop RK4", return_1, 2)
                return_3 = same_corridor_profile("ISRU to Base | 3-hop RK4", return_1, 3)
            else:
                two_forward = [self.get_segment("two1"), self.get_segment("two2")]
                two_return = [self.reverse_segment(two_forward[1]), self.reverse_segment(two_forward[0])]
                return_2 = rk4_segmented("ISRU to Base | 2-hop RK4", two_return, g, self.i("points_segment"))

                three_forward = [self.get_segment("three1"), self.get_segment("three2"), self.get_segment("three3")]
                three_return = [self.reverse_segment(three_forward[2]), self.reverse_segment(three_forward[1]), self.reverse_segment(three_forward[0])]
                return_3 = rk4_segmented("ISRU to Base | 3-hop RK4", three_return, g, self.i("points_segment"))

            profiles = [inbound_1, return_1, return_2, return_3]

            self.log("Saving graphs...")
            p1 = save_trajectory_graph(outdir, terrain_x, terrain_z, profiles, clearance)
            p2 = save_velocity_graph(outdir, terrain_x, terrain_z, profiles)
            p3 = save_acceleration_graph(outdir, terrain_x, terrain_z, profiles, g, g0, clearance, self.f("ascent_extra"), self.f("correction_extra"), self.f("landing_extra"))

            self.log("Saving 3D video/GIF. This may take a few minutes...")
            vids = save_3d_video(outdir, terrain_x, terrain_z, profiles, self.i("animation_frames"), self.save_gif.get(), self.save_mp4.get())

            self.log("Saving CSV outputs...")
            summary_rows = []
            for p in profiles:
                peak_i = int(np.argmax(p["h"]))
                demand = powered_acceleration_demand(p, g, self.f("ascent_extra"), self.f("correction_extra"), self.f("landing_extra"))
                summary_rows.append({
                    "Preset": self.preset_name.get(),
                    "Scenario": p["name"],
                    "Mode": p["mode"],
                    "Segments": p["segments"],
                    "Total time (s)": float(p["total_time"]),
                    "Peak distance (m)": float(p["x"][peak_i]),
                    "Peak altitude (m)": float(p["h"][peak_i]),
                    "Initial speed (m/s)": float(p["speed"][0]),
                    "Final speed (m/s)": float(p["speed"][-1]),
                    "Maximum speed (m/s)": float(np.max(p["speed"])),
                    "Max powered acceleration demand (m/s²)": float(np.max(demand)),
                })

                safe = p["name"].replace(" ", "_").replace("|", "").replace("/", "_")
                pd.DataFrame({
                    "time_s": p["time"],
                    "distance_m": p["x"],
                    "altitude_m": p["h"],
                    "vx_m_s": p["vx"],
                    "vy_m_s": p["vy"],
                    "velocity_m_s": p["speed"],
                    "rk4_ballistic_acceleration_m_s2": p["acceleration"],
                    "powered_acceleration_demand_m_s2": demand,
                    "powered_g_load_earth_g": demand / g0,
                }).to_csv(os.path.join(outdir, f"time_history_{safe}.csv"), index=False)

            sdf = pd.DataFrame(summary_rows)
            sdf["Dry mass (kg)"] = self.f("dry_mass")
            sdf["Payload mass (kg)"] = self.f("payload_mass")
            sdf["Propellant mass (kg)"] = self.f("propellant_mass")
            sdf["Isp (s)"] = self.f("isp")
            sdf["Thrust (N)"] = self.f("thrust")
            sdf.to_csv(os.path.join(outdir, "summary_rk4_all_scenarios.csv"), index=False)
            pd.DataFrame({"distance_m": terrain_x, "terrain_elevation_m": terrain_z}).to_csv(os.path.join(outdir, "terrain_profile_used.csv"), index=False)

            self.log("")
            self.log("RK4 SIMULATION COMPLETE")
            self.log(p1)
            self.log(p2)
            self.log(p3)
            for v in vids:
                self.log(v)
            self.log(os.path.join(outdir, "summary_rk4_all_scenarios.csv"))
            messagebox.showinfo("Complete", f"RK4 simulation complete.\nOutputs saved in:\n{outdir}")

        except Exception as e:
            self.log("ERROR:")
            self.log(str(e))
            messagebox.showerror("Simulation error", str(e))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
