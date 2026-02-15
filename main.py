import os
import sys
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pipeline.grid import MapaMieszkania
from pipeline.physics import Ogrzewanie
from constants import (
    DT, HX,
    APARTMENT_WIDTH_M, APARTMENT_HEIGHT_M,
    SETPOINT_HOME_C, SETPOINT_AWAY_C,
)


@dataclass
class SimulationResult:
    name: str
    time_h: np.ndarray
    mean_temp_C: np.ndarray
    std_temp_C: np.ndarray
    room_means_C: dict[int, np.ndarray]
    energy_kWh: np.ndarray
    comfort_recovery_h: float | None
    snapshots: dict[float, np.ndarray]


STALE = {
    "dt": DT,
    "hx": HX,
}

DATA_DIR = os.path.join(current_dir, 'data')
OUTPUT_DIR = os.path.join(current_dir, 'results')


def wczytaj_scenariusze(path = None):
    """Wczytuje scenariusze z CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, 'scenariusze.csv')
    scenariusze = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenariusze[row['nazwa']] = {
                'base_C': float(row['base_C']),
                'amp_C': float(row['amp_C']),
                'opis': row.get('opis', ''),
            }
    return scenariusze


SCENARIUSZE = wczytaj_scenariusze()


def outside_profile(times_s, kind):
    """Temperatura zewnętrzna [K] jako f(t). Scenariusze z CSV."""""
    if kind not in SCENARIUSZE:
        dostepne = ', '.join(SCENARIUSZE.keys())
        raise ValueError(f"Nieznany scenariusz '{kind}'. Dostępne: {dostepne}")

    sc = SCENARIUSZE[kind]
    t_h = times_s / 3600.0
    daily = np.cos((t_h - 14.0) * 2.0 * np.pi / 24.0)

    T_C = sc['base_C'] + sc['amp_C'] * daily
    return T_C + 273.15


def build_apartment():
    dom = MapaMieszkania(APARTMENT_WIDTH_M, APARTMENT_HEIGHT_M, STALE["hx"])
    dom.stworz_uklad_do_problemu_3()
    model = Ogrzewanie(dom, STALE)
    return dom, model


def draw_layout(ax, dom):
    """Rysuje plan mieszkania z legendą."""""
    grid = dom.grid

    # 0 ściana, 1 powietrze, 2 okno, 3 grzejnik, 4 drzwi
    cmap = ListedColormap([
        "black",      # ściana
        "white",      # powietrze
        "skyblue",    # okno
        "orange",     # grzejnik
        "lightgreen", # drzwi
    ])

    im = ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=4, interpolation="nearest")
    ax.set_title("Plan mieszkania")
    ax.axis("off")

    # legenda
    labels = ["Ściana", "Powietrze", "Okno", "Grzejnik", "Drzwi"]
    handles = []
    for i, lab in enumerate(labels):
        handles.append(plt.Line2D([0], [0], marker='s', linestyle='', markersize=10,
                                  markerfacecolor=cmap(i), markeredgecolor='gray', label=lab))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)


def draw_heatmap(ax, temp_K, title, *, vmin_C, vmax_C):
    temp_C = temp_K - 273.15
    im = ax.imshow(temp_C, cmap="coolwarm", vmin=vmin_C, vmax=vmax_C, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    return im, temp_C


def run_simulation(
    dom: MapaMieszkania,
    model: Ogrzewanie,
    *,
    scenario: str,
    strategy: str,
    total_hours: float = 24.0,
    away_from_h: float = 8.0,
    away_to_h: float = 16.0,
    setpoint_home_C: float = SETPOINT_HOME_C,
    setpoint_away_C: float = SETPOINT_AWAY_C,
):
    """
      - 'zawsze_grzeje'          : r=5 cały czas (cel 21C)
      - 'wylaczam_na_wyjscie'    : r=0 w czasie 8-16, potem r=5

    Zapisujemy:
      - średnią temperaturę w domu (po komórkach powietrza),
      - odchylenie standardowe temperatury (miara równomierności),
      - średnie temperatury per pokój,
      - energię skumulowaną [kWh],
      - czas powrotu do komfortu po powrocie.
    """

    rooms = dom.pokoje()
    masks = dom.daj_maski()

    # start z komfortu
    u = np.full(dom.grid.shape, setpoint_home_C + 273.15, dtype=float)

    dt = STALE["dt"]
    n_steps = int(round(total_hours * 3600.0 / dt))
    times_s = np.arange(n_steps) * dt
    times_h = times_s / 3600.0
    Tout = outside_profile(times_s, scenario)

    mean_temp_C = np.zeros(n_steps, dtype=float)
    std_temp_C = np.zeros(n_steps, dtype=float)
    room_means_C = {rid: np.zeros(n_steps, dtype=float) for rid in rooms.keys()}
    energy_kWh = np.zeros(n_steps, dtype=float)
    snapshots: dict[float, np.ndarray] = {}

    energy_J_acc = 0.0
    comfort_recovery_h = None
    snap_captured = set()

    for k in range(n_steps):
        t_h = float(times_h[k])

        # snapshoty
        for snap_t in (7.0, 15.0, 22.0):
            if snap_t not in snap_captured and abs(t_h - snap_t) < (dt / 3600.0):
                snapshots[snap_t] = u.copy()
                snap_captured.add(snap_t)

        # sterowanie
        setpoint = setpoint_home_C
        knob_by_room = {rid: 5 for rid in rooms.keys()}

        if strategy == "zawsze_grzeje":
            setpoint = setpoint_home_C
        elif strategy == "wylaczam_na_wyjscie":
            if away_from_h <= t_h < away_to_h:
                knob_by_room = {rid: 0 for rid in rooms.keys()}
                setpoint = setpoint_away_C
            else:
                setpoint = setpoint_home_C
        else:
            raise ValueError("strategy musi być: 'zawsze_grzeje' | 'wylaczam_na_wyjscie'")

        u, dE_J = model.jeden_krok(
            u,
            float(Tout[k]),
            setpoint_K=setpoint + 273.15,
            knob_by_room=knob_by_room,
        )

        energy_J_acc += dE_J
        energy_kWh[k] = energy_J_acc / 3_600_000.0

        # metryki
        air_vals_C = u[masks["powietrze"]] - 273.15
        mean_temp_C[k] = np.mean(air_vals_C)
        std_temp_C[k] = np.std(air_vals_C)

        for rid, info in rooms.items():
            room_means_C[rid][k] = np.mean(u[info["room_mask"]] - 273.15)

        if comfort_recovery_h is None and t_h >= away_to_h:
            if mean_temp_C[k] >= (setpoint_home_C - 0.2):
                comfort_recovery_h = t_h - away_to_h

    return SimulationResult(
        name=f"{scenario} / {strategy}",
        time_h=times_h,
        mean_temp_C=mean_temp_C,
        std_temp_C=std_temp_C,
        room_means_C=room_means_C,
        energy_kWh=energy_kWh,
        comfort_recovery_h=comfort_recovery_h,
        snapshots=snapshots,
    )


def plot_maps_and_curves(dom, res_a, res_b, Tout_func, title, *, save_tag=None):

    # wspólna skala kolorów
    all_snaps = []
    for res in (res_a, res_b):
        for t in (7.0, 15.0, 22.0):
            if t in res.snapshots:
                all_snaps.append(res.snapshots[t] - 273.15)
    if all_snaps:
        vmin = float(np.floor(min(np.min(s) for s in all_snaps)))
        vmax = float(np.ceil(max(np.max(s) for s in all_snaps)))
        vmin = max(vmin, -5.0)
        vmax = min(vmax, 30.0)
    else:
        vmin, vmax = 15.0, 22.0

    # plan + mapy 2x3
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1], wspace=0.15, hspace=0.15)

    ax_plan = fig.add_subplot(gs[:, 0])
    draw_layout(ax_plan, dom)

    times = [7.0, 15.0, 22.0]
    axes = []
    ims = []

    for row, res in enumerate([res_a, res_b]):
        for col, t in enumerate(times):
            ax = fig.add_subplot(gs[row, col + 1])
            axes.append(ax)
            snap = res.snapshots.get(t)
            if snap is None:
                ax.axis("off")
                continue

            # temp zewn.
            t_out_C = Tout_func(t) - 273.15
            im, _ = draw_heatmap(
                ax,
                snap,
                title=f"{res.name.split(' / ')[1]} @ {t:0.0f}:00\nT_out={t_out_C:.1f}°C",
                vmin_C=vmin,
                vmax_C=vmax,
            )
            ims.append(im)

    # belka skali
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Temperatura [°C]")

    fig.suptitle(title)
    plt.tight_layout()
    if save_tag:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, f'mapy_{save_tag}.png'), dpi=180)
    plt.show()

    # wykresy
    fig2, ax = plt.subplots(2, 2, figsize=(14, 8))

    ax00 = ax[0, 0]
    ax00.plot(res_a.time_h, res_a.mean_temp_C, label=res_a.name)
    ax00.plot(res_b.time_h, res_b.mean_temp_C, label=res_b.name)
    ax00.set_title("Średnia temperatura w mieszkaniu")
    ax00.set_xlabel("Czas [h]")
    ax00.set_ylabel("T̄ [°C]")
    ax00.grid(True)
    ax00.legend()

    ax01 = ax[0, 1]
    ax01.plot(res_a.time_h, res_a.std_temp_C, label=res_a.name)
    ax01.plot(res_b.time_h, res_b.std_temp_C, label=res_b.name)
    ax01.set_title("Odchylenie standardowe temperatury (równomierność)")
    ax01.set_xlabel("Czas [h]")
    ax01.set_ylabel("σ(T) [°C]")
    ax01.grid(True)
    ax01.legend()

    ax10 = ax[1, 0]
    ax10.plot(res_a.time_h, res_a.energy_kWh, label=res_a.name)
    ax10.plot(res_b.time_h, res_b.energy_kWh, label=res_b.name)
    ax10.set_title("Energia zużyta przez grzejniki")
    ax10.set_xlabel("Czas [h]")
    ax10.set_ylabel("E [kWh]")
    ax10.grid(True)
    ax10.legend()

    ax11 = ax[1, 1]
    for rid, series in res_a.room_means_C.items():
        ax11.plot(res_a.time_h, series, label=f"{res_a.name.split(' / ')[1]}: pokój {rid}")
    for rid, series in res_b.room_means_C.items():
        ax11.plot(res_b.time_h, series, linestyle="--", label=f"{res_b.name.split(' / ')[1]}: pokój {rid}")

    ax11.set_title("Średnia temperatura w pokojach (linia przerywana = wyłączam)")
    ax11.set_xlabel("Czas [h]")
    ax11.set_ylabel("T̄_pokój [°C]")
    ax11.grid(True)
    ax11.legend(fontsize=8, ncol=2)

    fig2.suptitle(title)
    plt.tight_layout()
    if save_tag:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig2.savefig(os.path.join(OUTPUT_DIR, f'wykresy_{save_tag}.png'), dpi=180)
    plt.show()

    print("\n--- PODSUMOWANIE ---")
    for res in (res_a, res_b):
        e_total = float(res.energy_kWh[-1])
        t_rec = res.comfort_recovery_h
        rec_txt = "(nie osiągnięto w 24h)" if t_rec is None else f"{t_rec:.2f} h"
        print(f"{res.name}:  E_total={e_total:.2f} kWh,  czas powrotu komfortu po 16:00: {rec_txt}")


def main():
    for scenario in SCENARIUSZE:
        dom, model = build_apartment()

        def Tout_at_hour(t_h: float, _sc=scenario) -> float:
            t_s = int(round(t_h * 3600.0))
            return float(outside_profile(np.array([t_s], dtype=float), _sc)[0])

        res_const = run_simulation(dom, model, scenario=scenario, strategy="zawsze_grzeje")
        res_off = run_simulation(dom, model, scenario=scenario, strategy="wylaczam_na_wyjscie")

        plot_maps_and_curves(
            dom,
            res_const,
            res_off,
            Tout_at_hour,
            title=f"Problem 3 {scenario}",
            save_tag=scenario,
        )


if __name__ == "__main__":
    main()
