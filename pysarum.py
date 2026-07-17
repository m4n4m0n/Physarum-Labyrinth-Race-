import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter

DR = np.array([-1, -1,  0,  1,  1,  1,  0, -1])
DC = np.array([ 0,  1,  1,  1,  0, -1, -1, -1])
richtungen = ["N", "NO", "O", "SO", "S", "SW", "W", "NW"]


class Schleimpilz:
    """Schleimpilz Ausbreitung"""

    def __init__(
        self,
        groesse=200,            # Kantenlaenge des Raumes
        n_pilze=2500,           # Anzahl der Agenten
        start=None,             # Startpunkt (zeile, spalte) (None = Mitte)
        exponent=7.0,           # Pheromon-Praeferenz (0 = reiner RandomWalk)
        zerfall=0.05,           # prozentualer Pheromon-Zerfall pro Zeitschritt
        deposit=0.6,            # frische Pheromonmenge auf Feld
        max_pheromon=4.0,       # Obergrenze fuer Pheromon je Feld
        w_geradeaus=1,          # RandomWalk-Basisgewicht "geradeaus"
        w_abbiegen=0.5,         # RandomWalk-Basisgewicht "links/rechts"
        wahrnehmung=1,          # Reichweite des Wahrnehmungsfeldes
        diffusion=0.0,          # Pheromon-Diffusion je Schritt (0 = aus, 1 = stark)
        periodisch=False,       # umlaufende Raender (Torus) statt Waenden
        seed=None,
    ):
        self.groesse = groesse
        self.n_pilze = n_pilze
        self.exponent = exponent
        self.retain = 1.0 - zerfall
        self.deposit = deposit
        self.max_pheromon = max_pheromon
        self.w_geradeaus = w_geradeaus
        self.w_abbiegen = w_abbiegen
        self.wahrnehmung = max(1, int(wahrnehmung))
        self.diffusion = float(np.clip(diffusion, 0.0, 1.0))
        self.periodisch = bool(periodisch)
        self.rng = np.random.default_rng(seed)

        self.pheromon = np.zeros((groesse, groesse), dtype=np.float32)
        self.hindernis = np.zeros((groesse, groesse), dtype=bool)

        if not self.periodisch:
            self.hindernis[0, :] = self.hindernis[-1, :] = True
            self.hindernis[:, 0] = self.hindernis[:, -1] = True

        if start is None:
            start = (groesse // 2, groesse // 2)
        self.start = start

        self.pos_r = np.full(n_pilze, start[0], dtype=np.int32)
        self.pos_c = np.full(n_pilze, start[1], dtype=np.int32)
        self.richtung = self.rng.integers(0, 8, size=n_pilze).astype(np.int32)

        self.besucht = np.zeros((groesse, groesse), dtype=bool)
        self.erkundet_bei = None

        self.schritt_nr = 0


    def _wrap(self, r, c):
        if self.periodisch:
            return r % self.groesse, c % self.groesse
        return np.clip(r, 0, self.groesse - 1), np.clip(c, 0, self.groesse - 1)

    # Hindernisse
    def rechteck(self, r0, c0, hoehe, breite):
        self.hindernis[r0:r0 + hoehe, c0:c0 + breite] = True

    def kreis(self, rc, cc, radius):
        rr, cc_ = np.ogrid[:self.groesse, :self.groesse]
        maske = (rr - rc) ** 2 + (cc_ - cc) ** 2 <= radius ** 2
        self.hindernis |= maske

    def linie(self, r0, c0, r1, c1, dicke=2):
        n = int(np.hypot(r1 - r0, c1 - c0)) + 1
        rs = np.linspace(r0, r1, n).round().astype(int)
        cs = np.linspace(c0, c1, n).round().astype(int)
        for dr in range(-dicke, dicke + 1):
            for dc in range(-dicke, dicke + 1):
                r = np.clip(rs + dr, 0, self.groesse - 1)
                c = np.clip(cs + dc, 0, self.groesse - 1)
                self.hindernis[r, c] = True

    def herz(self, rc, cc, skala, gefuellt=True, dicke=3):
        def maske(s):
            rr, cc_ = np.ogrid[:self.groesse, :self.groesse]
            x = (cc_ - cc) / s
            y = (rc - rr) / s
            return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

        if gefuellt:
            self.hindernis |= maske(skala)
        else:
            self.hindernis |= maske(skala) & ~maske(max(1.0, skala - dicke))

    def hindernisse_default(self):
        g = self.groesse
        self.rechteck(int(0.15 * g), int(0.20 * g), int(0.10 * g), int(0.25 * g))
        self.rechteck(int(0.62 * g), int(0.55 * g), int(0.22 * g), int(0.08 * g))
        self.herz(int(0.30 * g), int(0.72 * g), int(0.08 * g))
        self.kreis(int(0.75 * g), int(0.25 * g), int(0.06 * g))
        self.linie(int(0.45 * g), int(0.05 * g), int(0.45 * g), int(0.42 * g), dicke=1)
        self.hindernis[self.start[0], self.start[1]] = False

    def _diffundieren(self):
        """Verwischt das Pheromonfeld leicht"""
        modus = "wrap" if self.periodisch else "constant"
        verwischt = uniform_filter(self.pheromon, size=3, mode=modus, cval=0.0)
        self.pheromon = ((1.0 - self.diffusion) * self.pheromon
                         + self.diffusion * verwischt).astype(np.float32)
        self.pheromon[self.hindernis] = 0.0

    def schritt(self):
        if self.schritt_nr == 0:
            self.besucht[self.pos_r, self.pos_c] = True
        N = self.n_pilze

        rel = np.array([-1, 0, 1])      # relative Richtung
        basis = np.array([self.w_abbiegen, self.w_geradeaus, self.w_abbiegen])

        gewichte = np.zeros((N, 3), dtype=np.float64)
        ziel_r = np.zeros((N, 3), dtype=np.int32)
        ziel_c = np.zeros((N, 3), dtype=np.int32)

        for k in range(3):
            kdir = (self.richtung + rel[k]) % 8
            nr, nc = self._wrap(self.pos_r + DR[kdir], self.pos_c + DC[kdir])
            ziel_r[:, k] = nr
            ziel_c[:, k] = nc

            pher = self.pheromon[nr, nc].copy()
            for w in range(2, self.wahrnehmung + 1):
                sr, sc = self._wrap(self.pos_r + DR[kdir] * w,
                                    self.pos_c + DC[kdir] * w)
                pher += self.pheromon[sr, sc] / w

            gewichte[:, k] = basis[k] * (1.0 + pher) ** self.exponent
            gewichte[:, k] *= ~self.hindernis[nr, nc]

        summe = gewichte.sum(axis=1)
        blockiert = summe <= 0.0         # kein freies Feld im Wahrnehmungsfeld

        # gewichtete Zufallsauswahl
        u = self.rng.random(N) * np.where(blockiert, 1.0, summe)  # wuerfeln
        kum = np.cumsum(gewichte, axis=1)
        wahl = (kum < u[:, None]).sum(axis=1)
        wahl = np.clip(wahl, 0, 2)

        beweg = ~blockiert
        gewaehlte_dir = (self.richtung + rel[wahl]) % 8

        idx = np.arange(N)
        self.pos_r[beweg] = ziel_r[idx[beweg], wahl[beweg]]
        self.pos_c[beweg] = ziel_c[idx[beweg], wahl[beweg]]
        self.richtung[beweg] = gewaehlte_dir[beweg]

        # blockierte Agenten drehen um
        self.richtung[blockiert] = (self.richtung[blockiert] + 4) % 8

        # Pheromon-Spur
        np.add.at(self.pheromon, (self.pos_r, self.pos_c), self.deposit)
        np.minimum(self.pheromon, self.max_pheromon, out=self.pheromon)

        # Pheromon-Diffusion (nur wenn aktiviert)
        if self.diffusion > 0.0:
            self._diffundieren()

        # Pheromon-Zerfall
        self.pheromon *= self.retain
        self.pheromon[self.pheromon < 1e-3] = 0.0

        self.schritt_nr += 1

        self.besucht[self.pos_r, self.pos_c] = True
        if self.erkundet_bei is None and self.besucht[~self.hindernis].all():
            self.erkundet_bei = self.schritt_nr

    def distribute_random(self):
        """Verteilt alle Agenten gleichmaessig zufaellig auf die freien Zellen."""
        frei_r, frei_c = np.where(~self.hindernis)
        wahl = self.rng.integers(0, len(frei_r), size=self.n_pilze)
        self.pos_r = frei_r[wahl].astype(np.int32)
        self.pos_c = frei_c[wahl].astype(np.int32)
        self.richtung = self.rng.integers(0, 8, size=self.n_pilze).astype(np.int32)


def _farbkarte():
    """Dunkler Hintergrund -> gelb-gruene Pheromonspur (Physarum-Look)."""
    return LinearSegmentedColormap.from_list(
        "pilz", ["#0a0a12", "#1b3a2f", "#2f8f5b", "#a8e05f", "#f7ff9e"]
    )


def animation(sim, schritte=300, schritte_pro_frame=2, dateiname="schleimpilz_demo.gif",
              fps=25, dpi=90):
    """Erzeugt eine GIF-Animation der Simulation."""
    cmap = _farbkarte()

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0a0a12")
    ax.set_facecolor("#0a0a12")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Pheromon-Hintergrund
    bild = ax.imshow(sim.pheromon, cmap=cmap, vmin=0, vmax=sim.max_pheromon,
                     interpolation="nearest", origin="upper")

    # Hindernisse als halbtransparentes Overlay
    hind_rgba = np.zeros((sim.groesse, sim.groesse, 4))
    hind_rgba[..., 0] = 0.55; hind_rgba[..., 1] = 0.12; hind_rgba[..., 2] = 0.22
    hind_rgba[..., 3] = sim.hindernis * 0.9
    ax.imshow(hind_rgba, origin="upper", interpolation="nearest")

    # Agenten
    punkte = ax.scatter(sim.pos_c, sim.pos_r, s=1.2, c="#ffffff", alpha=0.6)
    titel = ax.set_title("", color="#a8e05f", fontsize=10, family="monospace")

    n_frames = schritte // schritte_pro_frame

    def update(frame):
        for _ in range(schritte_pro_frame):
            sim.schritt()
        bild.set_data(sim.pheromon)
        punkte.set_offsets(np.column_stack([sim.pos_c, sim.pos_r]))
        titel.set_text(
            f"Schritt {sim.schritt_nr:4d}  Pilze={sim.n_pilze}   "
            f"Exponent={sim.exponent:.1f}  Zerfall={1 - sim.retain:.2f}  "
            f"Diff={sim.diffusion:.2f}"
        )
        return bild, punkte, titel

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(dateiname, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return dateiname


if __name__ == "__main__":
    sim = Schleimpilz(
        groesse=200,
        n_pilze=10000,
        exponent=5.0,
        zerfall=0.2,
        deposit=0.5,
        w_geradeaus=0.4,
        w_abbiegen=0.4,
        wahrnehmung=5,
        diffusion=0.15,
        periodisch=True,
        seed=100,
    )
    sim.hindernisse_default()
    #sim.distribute_random()
    pfad = animation(sim, schritte=2000, schritte_pro_frame=20,
                     dateiname="schleimpilz_random_12.gif")
    print("Animation gespeichert:", pfad)
    print("Ganzes Feld erkundet nach", sim.erkundet_bei, "Schritten")
