import numpy as np
from collections import deque


class MapaMieszkania:
    """Typy komórek (int): 0 ściana, 1 powietrze, 2 okno, 3 grzejnik, 4 drzwi.

    Dla zakrecania grzejników ważne jest:
    - mieć osobne pokoje (żeby liczyć średnią temperaturę w pokoju),
    - mieć przejścia (drzwi), żeby ciepło mogło się przenosić.
    """

    SCIANA = 0
    POWIETRZE = 1
    OKNO = 2
    GRZEJNIK = 3
    DRZWI = 4

    WALKABLE = {POWIETRZE, GRZEJNIK, DRZWI}

    def __init__(self, szerokosc_m: float, wysokosc_m: float, hx: float):
        self.hx = float(hx)
        self.nx = int(round(szerokosc_m / self.hx))
        self.ny = int(round(wysokosc_m / self.hx))
        if self.nx < 5 or self.ny < 5:
            raise ValueError("Za mała siatka. Zwiększ rozmiar mieszkania albo zmniejsz hx.")

        self.grid = np.full((self.ny, self.nx), self.POWIETRZE, dtype=np.int8)

        # Ściany zewnętrzne
        self.grid[0, :] = self.SCIANA
        self.grid[-1, :] = self.SCIANA
        self.grid[:, 0] = self.SCIANA
        self.grid[:, -1] = self.SCIANA

        # Cache
        self._room_id = None
        self._rooms = None

    def _zakres(self, start_proc: float, end_proc: float, limit: int):
        """Zamienia % na indeksy pilnując żeby nie nadpisać ramki."""
        s = int(limit * start_proc)
        e = int(limit * end_proc)
        return max(1, s), min(limit - 1, e)

    def stworz_uklad_do_problemu_3(self):
        """Tworzy układ 3 pokojowy z drzwiami, oknami i grzejnikami
        """

        mx = int(self.nx * 0.4)
        my = int(self.ny * 0.5)

        # Wewnętrzne ściany
        if 1 < mx < self.nx - 1:
            self.grid[1:-1, mx] = self.SCIANA
        if 1 < my < self.ny - 1:
            self.grid[my, 1:mx] = self.SCIANA

        # ---- DRZWI (otwory w ścianach) -----
        # Drzwi w pionowej ścianie (łączą lewo/prawo)
        dy = int(self.ny * 0.25)
        self._otwor_drzwi(y=dy, x=mx)
        # Drzwi w poziomej ścianie (łączą górę/dół po lewej)
        dx = int(self.nx * 0.2)
        self._otwor_drzwi(y=my, x=dx)

        # --- OKNA ---
        s1, e1 = self._zakres(0.15, 0.35, self.nx)
        s2, e2 = self._zakres(0.65, 0.85, self.nx)
        self.grid[0, s1:e1] = self.OKNO          # okno górne
        self.grid[-1, s1:e1] = self.OKNO         # okno dolne lewo
        self.grid[-1, s2:e2] = self.OKNO         # okno dolne prawo

        # --- GRZEJNIKI (pod oknami) ---
        # Góra (pokój 1)
        gy_gora = 2
        self.grid[gy_gora, s1:e1] = self.GRZEJNIK
        # Dół lewo (pokój 2)
        gy_dol = self.ny - 3
        self.grid[gy_dol, s1:e1] = self.GRZEJNIK
        # Dół prawo (pokój 3)
        self.grid[gy_dol, s2:e2] = self.GRZEJNIK

        # po zmianach unieważnij cache pokoi
        self._room_id = None
        self._rooms = None

    def _otwor_drzwi(self, y: int, x: int, szerokosc: int = 2):
        """Wstawia drzwi (przejście) w ścianie: zamienia kilka komórek na DRZWI."""
        if not (1 <= y < self.ny - 1 and 1 <= x < self.nx - 1):
            return
        for k in range(-(szerokosc // 2), (szerokosc // 2) + 1):
            yy, xx = y, x + k
            if 1 <= yy < self.ny - 1 and 1 <= xx < self.nx - 1:
                self.grid[yy, xx] = self.DRZWI

    def daj_maski(self):
        sciana = (self.grid == self.SCIANA)
        okno = (self.grid == self.OKNO)
        grzejnik = (self.grid == self.GRZEJNIK)
        drzwi = (self.grid == self.DRZWI)
        powietrze = (self.grid == self.POWIETRZE) | grzejnik | drzwi

        # ściany zewnętrzne = brzegi (bez okien)
        zew_sciana = np.zeros_like(sciana)
        zew_sciana[0, :] = True
        zew_sciana[-1, :] = True
        zew_sciana[:, 0] = True
        zew_sciana[:, -1] = True
        zew_sciana &= sciana

        wew_sciana = sciana & (~zew_sciana)

        return {
            "powietrze": powietrze,
            "okno": okno,
            "grzejnik": grzejnik,
            "drzwi": drzwi,
            "sciana": sciana,
            "sciana_zew": zew_sciana,
            "sciana_wew": wew_sciana,
        }

    def label_pokoje(self):
        if self._room_id is not None:
            return self._room_id

        room_id = np.full(self.grid.shape, -1, dtype=np.int16)
        visited = np.zeros(self.grid.shape, dtype=bool)

        def is_walkable(y: int, x: int) -> bool:
            return self.grid[y, x] in self.WALKABLE

        rid = 0
        for y in range(1, self.ny - 1):
            for x in range(1, self.nx - 1):
                if visited[y, x] or not is_walkable(y, x):
                    continue
                q = deque([(y, x)])
                visited[y, x] = True
                room_id[y, x] = rid
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if visited[ny, nx] or not is_walkable(ny, nx):
                            continue
                        visited[ny, nx] = True
                        room_id[ny, nx] = rid
                        q.append((ny, nx))
                rid += 1

        self._room_id = room_id
        return room_id

    def pokoje(self):
        if self._rooms is not None:
            return self._rooms

        rid = self.label_pokoje()
        masks = self.daj_maski()
        rooms: dict[int, dict[str, np.ndarray]] = {}

        for k in sorted(set(rid[rid >= 0].ravel().tolist())):
            room_mask = (rid == k) & masks["powietrze"]
            rad_mask = room_mask & masks["grzejnik"]
            air_mask = room_mask & (~masks["grzejnik"])  # "powietrze" bez samych grzejników
            rooms[k] = {
                "room_mask": room_mask,
                "air_mask": air_mask,
                "radiator_mask": rad_mask,
            }

        self._rooms = rooms
        return rooms
