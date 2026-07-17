"""
labyrinth_staemme.py
====================

Grosses Labyrinth, in dem 4 Pilzstaemme von je einer Ecke zur MITTE streben.
Jeder Stamm hat EIGENE Parameter (Exponent, Zerfallsrate, Sichtweite,
Bewegungs-Zufaelligkeit) -- diese sind in einer Tabelle festgehalten und
werden beim Start ausgegeben und als CSV gesichert.

Interaktion ueber eine Kopplungsmatrix: alle Staemme stossen sich ab, ausser
einem kooperierenden Paar, das sich anzieht.

Loesen des Labyrinths: Am Ziel (Mitte) sitzt Futter, dessen Lockstoff durch
die Gaenge diffundiert (per BFS-Distanzfeld). Jeder Agent bevorzugt unter den
drei Vorwaertszellen die dem Ziel naechste -- so finden die Staemme den Weg.
"""

import csv
import numpy as np
from collections import deque
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageSequence

DR = np.array([-1, -1,  0,  1,  1,  1,  0, -1])
DC = np.array([ 0,  1,  1,  1,  0, -1, -1, -1])


# ===========================================================================
# Stamm-Tabelle  (HIER die Parameter je Stamm; wird beim Start gesichert)
# ===========================================================================
STAEMME = [
    # name      farbe (RGB 0..1)       exponent zerfall wahrnehmung w_abbiegen
    # Hinweis: Gruen + Cyan sind das kooperierende Paar (siehe KOOP_PAAR)
    dict(name="Gruen",  farbe=(0.30, 1.00, 0.45), exponent=3.0,
         zerfall=0.06, wahrnehmung=5, w_abbiegen=0.30),
    dict(name="Cyan",   farbe=(0.30, 0.90, 0.95), exponent=4.0,
         zerfall=0.05, wahrnehmung=3, w_abbiegen=0.20),
    dict(name="Pink",   farbe=(1.00, 0.35, 0.75), exponent=2.0,
         zerfall=0.03, wahrnehmung=7, w_abbiegen=0.50),
    dict(name="Orange", farbe=(1.00, 0.70, 0.20), exponent=6.0,
         zerfall=0.06, wahrnehmung=1, w_abbiegen=0.15),
]

# Kooperierendes Paar (Indizes in STAEMME); alle anderen Paare stossen sich ab
KOOP_PAAR = (0, 1)
ANZIEHUNG = 2.0     # Staerke der Kooperation (>0)
ABSTOSSUNG = 2.5    # Staerke der Abstossung  (>0)


def kopplungsmatrix(n, koop_paar, anziehung, abstossung):
    """-abstossung ueberall, +anziehung nur fuer das kooperierende Paar."""
    K = np.full((n, n), -abstossung, dtype=float)
    a, b = koop_paar
    K[a, b] = K[b, a] = anziehung
    np.fill_diagonal(K, 0.0)
    return K


class LabyrinthStaemme:
    def __init__(self, groesse=100, zellen=14, n_pro_stamm=200,
                 staemme=STAEMME, kopplung=None, lock_gewicht=1.0,
                 deposit=0.6, max_pheromon=4.0, w_geradeaus=1.0,
                 diffusion=0.05, schleifen_anteil=0.0, seed=0):
        self.groesse = groesse
        self.S = len(staemme)
        self.deposit = deposit
        self.max_pheromon = max_pheromon
        self.w_geradeaus = w_geradeaus
        self.diffusion = float(np.clip(diffusion, 0.0, 1.0))
        self.lock_gewicht = lock_gewicht
        self.rng = np.random.default_rng(seed)

        # --- Stamm-Parameter als Arrays (per-Stamm) ----------------------
        self.namen      = [s["name"] for s in staemme]
        self.farben     = np.array([s["farbe"] for s in staemme])
        self.exponent   = np.array([s["exponent"] for s in staemme], float)
        self.retain     = np.array([1.0 - s["zerfall"] for s in staemme], float)
        self.wahrnehmung = np.array([int(s["wahrnehmung"]) for s in staemme])
        self.w_abbiegen = np.array([s["w_abbiegen"] for s in staemme], float)

        if kopplung is None:
            kopplung = kopplungsmatrix(self.S, KOOP_PAAR, ANZIEHUNG, ABSTOSSUNG)
        self.K = np.array(kopplung, float)
        np.fill_diagonal(self.K, 0.0)

        # --- Labyrinth bauen + Ziel (Mitte) + Lockstoff (BFS) ------------
        self._baue_labyrinth(zellen, seed, schleifen_anteil)

        # --- Agenten je Stamm an eine Ecke setzen ------------------------
        self.felder = np.zeros((self.S, groesse, groesse), dtype=np.float32)
        self.N = n_pro_stamm * self.S
        self.stamm = np.repeat(np.arange(self.S), n_pro_stamm).astype(np.int32)
        self.pos_r = np.empty(self.N, dtype=np.int32)
        self.pos_c = np.empty(self.N, dtype=np.int32)
        for s in range(self.S):
            r, c = self.ecken[s % len(self.ecken)]
            self.pos_r[self.stamm == s] = r
            self.pos_c[self.stamm == s] = c
        self.richtung = self.rng.integers(0, 8, size=self.N).astype(np.int32)
        self.schritt_nr = 0

    # -----------------------------------------------------------------
    def _baue_labyrinth(self, zellen, seed, schleifen_anteil=0.0):
        rng = np.random.default_rng(seed)
        nx = ny = int(zellen)
        L = np.ones((2 * ny + 1, 2 * nx + 1), dtype=bool)   # True = Wand
        besucht = np.zeros((ny, nx), dtype=bool)
        stapel = [(0, 0)]; besucht[0, 0] = True; L[1, 1] = False
        while stapel:
            i, j = stapel[-1]
            nb = [(i + di, j + dj, di, dj)
                  for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                  if 0 <= i + di < ny and 0 <= j + dj < nx and not besucht[i + di, j + dj]]
            if nb:
                ni, nj, di, dj = nb[rng.integers(len(nb))]
                L[2 * i + 1 + di, 2 * j + 1 + dj] = False
                L[2 * ni + 1, 2 * nj + 1] = False
                besucht[ni, nj] = True
                stapel.append((ni, nj))
            else:
                stapel.pop()

        # --- Schleifen erzeugen: einen Anteil der Trennwaende oeffnen -----
        # Das macht aus dem 'perfekten' Labyrinth ein Schleifen-Labyrinth
        # mit mehreren Wegen -> die Pilze verlaufen sich oefter und die
        # Staemme begegnen sich schon unterwegs, nicht erst in der Mitte.
        if schleifen_anteil > 0.0:
            kandidaten = []
            for r in range(1, 2 * ny):
                for c in range(1, 2 * nx):
                    if not L[r, c]:
                        continue
                    if r % 2 == 1 and c % 2 == 0:        # Wand zwischen 2 Zellen (horiz.)
                        kandidaten.append((r, c))
                    elif r % 2 == 0 and c % 2 == 1:      # Wand zwischen 2 Zellen (vert.)
                        kandidaten.append((r, c))
            kandidaten = np.array(kandidaten)
            if len(kandidaten):
                anzahl = int(schleifen_anteil * len(kandidaten))
                wahl = rng.permutation(len(kandidaten))[:anzahl]
                for r, c in kandidaten[wahl]:
                    L[r, c] = False

        H, W = L.shape
        rand = 2
        s = (self.groesse - 2 * rand) // max(H, W)
        if s < 3:
            raise ValueError("Gitter zu klein -- groesse erhoehen oder zellen senken.")
        wand = np.kron(L, np.ones((s, s), dtype=bool))
        self.hindernis = np.zeros((self.groesse, self.groesse), dtype=bool)
        self.hindernis[0, :] = self.hindernis[-1, :] = True
        self.hindernis[:, 0] = self.hindernis[:, -1] = True
        r0, c0 = rand, rand
        self.hindernis[r0:r0 + wand.shape[0], c0:c0 + wand.shape[1]] |= wand

        def mitte(i, j):
            return (r0 + (2 * i + 1) * s + s // 2, c0 + (2 * j + 1) * s + s // 2)
        # Ziel = Mittelzelle des Labyrinths
        self.ziel = mitte(ny // 2, nx // 2)
        # Startecken der Staemme
        self.ecken = [mitte(0, 0), mitte(0, nx - 1),
                      mitte(ny - 1, 0), mitte(ny - 1, nx - 1)]

        # Lockstoff = BFS-Distanz zur Mitte (Waende = grosser Wert)
        dist = self._bfs_distanz(self.ziel)
        gross = np.nanmax(dist[np.isfinite(dist)]) + 10.0
        self.lockstoff = np.where(np.isfinite(dist), dist, gross).astype(np.float32)

    def _bfs_distanz(self, quelle):
        dist = np.full((self.groesse, self.groesse), np.inf)
        qr, qc = quelle
        dist[qr, qc] = 0
        dq = deque([(qr, qc)])
        while dq:
            r, c = dq.popleft()
            d = dist[r, c] + 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.groesse and 0 <= nc < self.groesse
                        and not self.hindernis[nr, nc] and d < dist[nr, nc]):
                    dist[nr, nc] = d
                    dq.append((nr, nc))
        return dist

    # -----------------------------------------------------------------
    def schritt(self):
        N, g = self.N, self.groesse
        rel = np.array([-1, 0, 1])
        stamm = self.stamm
        exp_ag = self.exponent[stamm]          # Exponent je Agent
        wahr_ag = self.wahrnehmung[stamm]       # Sichtweite je Agent
        wabb_ag = self.w_abbiegen[stamm]        # Abbiege-Gewicht je Agent
        Kzeilen = self.K[stamm]                 # (N, S)
        wmax = int(self.wahrnehmung.max())

        gewichte = np.zeros((N, 3))
        ziel_r = np.zeros((N, 3), np.int32)
        ziel_c = np.zeros((N, 3), np.int32)
        d_ziel = np.zeros((N, 3))

        for k in range(3):
            kdir = (self.richtung + rel[k]) % 8
            dr, dc = DR[kdir], DC[kdir]
            nr = np.clip(self.pos_r + dr, 0, g - 1)
            nc = np.clip(self.pos_c + dc, 0, g - 1)
            ziel_r[:, k] = nr; ziel_c[:, k] = nc

            frei = ~self.hindernis[nr, nc]
            # kein diagonales Eckenschneiden (Labyrinth-Topologie wahren)
            diagonal = (dr != 0) & (dc != 0)
            ecke_frei = (~self.hindernis[np.clip(self.pos_r + dr, 0, g - 1), self.pos_c]) & \
                        (~self.hindernis[self.pos_r, np.clip(self.pos_c + dc, 0, g - 1)])
            frei &= (~diagonal | ecke_frei)

            # eigene Spur, aufsummiert ueber die (per-Stamm) Sichtweite
            eigen = self.felder[stamm, nr, nc].astype(np.float64)
            for w in range(2, wmax + 1):
                aktiv = wahr_ag >= w
                sr = np.clip(self.pos_r + dr * w, 0, g - 1)
                sc = np.clip(self.pos_c + dc * w, 0, g - 1)
                eigen += np.where(aktiv, self.felder[stamm, sr, sc] / w, 0.0)

            # Fremdspuren gewichtet (Abstossung/Kooperation)
            werte = self.felder[:, nr, nc]                  # (S, N)
            fremd = np.einsum("sn,ns->n", werte, Kzeilen)

            basis = np.where(k == 1, self.w_geradeaus, wabb_ag)
            w_k = basis * (1.0 + eigen) ** exp_ag
            w_k *= np.exp(np.clip(fremd, -15.0, 15.0))
            w_k *= frei
            gewichte[:, k] = w_k
            d_ziel[:, k] = self.lockstoff[nr, nc]

        # Chemotaxis zur Mitte: zielnaechste der drei Zellen bevorzugen
        dmin = d_ziel.min(axis=1, keepdims=True)
        gewichte *= np.exp(-self.lock_gewicht * (d_ziel - dmin))

        summe = gewichte.sum(axis=1)
        blockiert = summe <= 0.0
        u = self.rng.random(N) * np.where(blockiert, 1.0, summe)
        kum = np.cumsum(gewichte, axis=1)
        wahl = np.clip((kum < u[:, None]).sum(axis=1), 0, 2)

        beweg = ~blockiert
        idx = np.arange(N)
        self.pos_r[beweg] = ziel_r[idx[beweg], wahl[beweg]]
        self.pos_c[beweg] = ziel_c[idx[beweg], wahl[beweg]]
        self.richtung[beweg] = (self.richtung + rel[wahl])[beweg] % 8
        self.richtung[blockiert] = (self.richtung[blockiert] + 4) % 8

        # Pheromon je Stamm
        np.add.at(self.felder, (self.stamm, self.pos_r, self.pos_c), self.deposit)
        np.minimum(self.felder, self.max_pheromon, out=self.felder)
        if self.diffusion > 0.0:
            verwischt = uniform_filter(self.felder, size=(1, 3, 3),
                                       mode="constant", cval=0.0)
            self.felder = ((1.0 - self.diffusion) * self.felder
                           + self.diffusion * verwischt).astype(np.float32)
            self.felder[:, self.hindernis] = 0.0
        self.felder *= self.retain[:, None, None]        # per-Stamm Zerfall
        self.felder[self.felder < 1e-3] = 0.0
        self.schritt_nr += 1

    # -----------------------------------------------------------------
    def am_ziel(self, radius=None):
        """Anzahl Agenten je Stamm nahe der Mitte."""
        if radius is None:
            radius = max(4, self.groesse // 40)
        nah = ((self.pos_r - self.ziel[0]) ** 2
               + (self.pos_c - self.ziel[1]) ** 2) <= radius ** 2
        return [int(nah[self.stamm == s].sum()) for s in range(self.S)]

    def bild(self):
        g = self.groesse
        rgb = np.zeros((g, g, 3))
        for s in range(self.S):
            norm = np.sqrt(self.felder[s]) / np.sqrt(self.max_pheromon)
            rgb += norm[..., None] * self.farben[s][None, None, :]
        rgb[self.hindernis] = [0.16, 0.05, 0.09]        # Waende dunkelrot
        zr, zc = self.ziel                               # Ziel weiss markieren
        rgb[max(0, zr-2):zr+3, max(0, zc-2):zc+3] = [1, 1, 1]
        return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
def tabelle_sichern(staemme, K, pfad="staemme_tabelle.csv"):
    """Gibt die Stamm-Tabelle + Kopplung aus und speichert sie als CSV."""
    kopf = ["name", "exponent", "zerfall", "wahrnehmung", "w_abbiegen"]
    print("\n=== Stamm-Parameter =========================================")
    print(f"{'Stamm':<14}{'Exponent':>9}{'Zerfall':>9}{'Sicht':>7}{'Zufall':>8}")
    for s in staemme:
        print(f"{s['name']:<14}{s['exponent']:>9}{s['zerfall']:>9}"
              f"{s['wahrnehmung']:>7}{s['w_abbiegen']:>8}")
    print("\n=== Kopplungsmatrix K (>0 Kooperation, <0 Abstossung) ========")
    namen = [s["name"] for s in staemme]
    print(" " * 14 + "".join(f"{n[:6]:>8}" for n in namen))
    for i, n in enumerate(namen):
        print(f"{n:<14}" + "".join(f"{K[i, j]:>8.1f}" for j in range(len(namen))))

    with open(pfad, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(kopf)
        for s in staemme:
            wr.writerow([s[k] for k in kopf])
        wr.writerow([])
        wr.writerow(["Kopplungsmatrix"] + namen)
        for i, n in enumerate(namen):
            wr.writerow([n] + [f"{K[i, j]:.1f}" for j in range(len(namen))])
    print(f"\nTabelle gesichert: {pfad}\n")


def animation(sim, schritte=1500, schritte_pro_frame=4, dateiname="labyrinth_staemme.gif",
              fps=25, dpi=90, max_frames=240, farben=112):
    if schritte // schritte_pro_frame > max_frames:
        schritte_pro_frame = max(1, schritte // max_frames)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.patch.set_facecolor("#08080f"); ax.set_facecolor("#08080f")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    bild = ax.imshow(sim.bild(), interpolation="nearest", origin="upper")
    titel = ax.set_title("", color="#dddddd", fontsize=9, family="monospace")
    n_frames = schritte // schritte_pro_frame

    def update(frame):
        for _ in range(schritte_pro_frame):
            sim.schritt()
        bild.set_data(sim.bild())
        z = sim.am_ziel()
        titel.set_text(f"Schritt {sim.schritt_nr:4d}   am Ziel je Stamm: {z}")
        return bild, titel

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(dateiname, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    im = Image.open(dateiname)
    dauer = im.info.get("duration", 40)
    frames = [f.copy().convert("P", palette=Image.ADAPTIVE, colors=farben)
              for f in ImageSequence.Iterator(im)]
    frames[0].save(dateiname, save_all=True, append_images=frames[1:], loop=0,
                   duration=dauer, optimize=True, disposal=2)
    import os
    print(f"{dateiname}: {n_frames} Frames -> {os.path.getsize(dateiname)/1e6:.1f} MB")
    return dateiname


if __name__ == "__main__":
    K = kopplungsmatrix(len(STAEMME), KOOP_PAAR, ANZIEHUNG, ABSTOSSUNG)
    tabelle_sichern(STAEMME, K)

    # SEHR GROSSES Schleifen-Labyrinth:
    #   - groesse/zellen hoch  -> grosses, feines Labyrinth
    #   - schleifen_anteil     -> viele Alternativwege (Pilze verlaufen sich,
    #                             Staemme begegnen sich schon vor der Mitte)
    #   - lock_gewicht kleiner -> weniger Ziel-Sog, mehr Umherirren
    sim = LabyrinthStaemme(groesse=600, zellen=56, n_pro_stamm=500,
                           staemme=STAEMME, kopplung=K, lock_gewicht=0.6,
                           deposit=0.6, diffusion=0.05, schleifen_anteil=0.25,
                           seed=3)
    print("Weglaenge Ecke->Mitte (BFS, Beispiel):",
          int(sim.lockstoff[sim.ecken[0]]))

    # Grosses Labyrinth + mehr Umherirren -> viele Schritte noetig.
    animation(sim, schritte=2000, dateiname="labyrinth_staemme_xl.gif",
              max_frames=240, dpi=85, farben=100)
    print("Am Ziel je Stamm:", dict(zip(sim.namen, sim.am_ziel())))
