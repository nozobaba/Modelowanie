import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    ALPHA, P_RADIATOR_W, RHO_AIR, C_AIR,
    CEILING_HEIGHT_M, WALL_LOSS_BETA,
)


class Ogrzewanie:

    def __init__(
        self,
        mapa_obj,
        stale: dict,
        *,
        alpha: float = ALPHA,
        P_radiator_W: float = P_RADIATOR_W,
        rho_air: float = RHO_AIR,
        c_air: float = C_AIR,
        ceiling_height_m: float = CEILING_HEIGHT_M,
        wall_loss_beta: float = WALL_LOSS_BETA,
    ):
        self.mapa = mapa_obj
        self.maski = mapa_obj.daj_maski()
        self.rooms = mapa_obj.pokoje()

        self.hx = float(mapa_obj.hx)
        self.dt = float(stale["dt"])

        self.alpha = float(alpha)
        self.P = float(P_radiator_W)
        self.rho = float(rho_air)
        self.c = float(c_air)
        self.H = float(ceiling_height_m)
        self.wall_beta = float(wall_loss_beta)

        self._check_stability()

    def _check_stability(self) -> None:
        factor = (self.alpha * self.dt) / (self.hx ** 2)
        if factor > 0.25:
            raise ValueError(
                f"Niestabilny krok: alpha*dt/hx^2 = {factor:.3f} > 0.25. "
                "Zmniejsz dt albo alpha, albo zwiększ hx."
            )

    def jeden_krok(
        self,
        u: np.ndarray,
        temp_zew: float,
        *,
        setpoint_K: float,
        knob_by_room: dict[int, int] | None = None,
    ):

        u_new = u.copy()

        # Dyfuzja
        center = u[1:-1, 1:-1]
        up = u[0:-2, 1:-1]
        down = u[2:, 1:-1]
        left = u[1:-1, 0:-2]
        right = u[1:-1, 2:]
        lap = up + down + left + right - 4.0 * center

        factor = (self.alpha * self.dt) / (self.hx ** 2)
        mask_center = self.maski["powietrze"][1:-1, 1:-1]
        u_new[1:-1, 1:-1][mask_center] = center[mask_center] + factor * lap[mask_center]

        # Okna = temperatura zewnętrzna
        u_new[self.maski["okno"]] = temp_zew

        # Ściany
        u_new = self._apply_walls(u_new, temp_zew)

        # Grzanie
        energy_J = 0.0
        if knob_by_room is None:
            knob_by_room = {rid: 5 for rid in self.rooms.keys()}

        for rid, info in self.rooms.items():
            rad_mask = info["radiator_mask"]
            if not np.any(rad_mask):
                continue
            r = int(knob_by_room.get(rid, 0))
            r = max(0, min(5, r))
            if r == 0:
                continue

            # Termostat
            room_mask = info["room_mask"]
            room_mean = np.mean(u_new[room_mask])
            if room_mean >= setpoint_K:
                continue

            # Moc efektywna
            P_eff = self.P * (r / 5.0)
            E = P_eff * self.dt
            energy_J += E

            n_air = int(np.sum(info["air_mask"]))
            if n_air <= 0:
                continue
            V_room = n_air * (self.hx ** 2) * self.H
            dT = E / (self.rho * self.c * V_room)

            u_new[info["air_mask"]] += dT
            u_new[rad_mask] += 0.25 * dT

        u_new = np.nan_to_num(u_new, nan=293.15)
        u_new = np.clip(u_new, 223.15, 373.15)

        return u_new, energy_J

    def _apply_walls(self, u, temp_zew):
        wall = self.maski["sciana"]
        if not np.any(wall):
            return u

        ys, xs = np.where(wall)
        for y, x in zip(ys.tolist(), xs.tolist()):
            neigh = []
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < u.shape[0] and 0 <= xx < u.shape[1]:
                    if not wall[yy, xx]:
                        neigh.append(u[yy, xx])
            if not neigh:
                continue

            t_in = np.mean(neigh)
            if self.maski["sciana_zew"][y, x]:
                u[y, x] = (1.0 - self.wall_beta) * t_in + self.wall_beta * temp_zew
            else:
                u[y, x] = t_in

        return u
