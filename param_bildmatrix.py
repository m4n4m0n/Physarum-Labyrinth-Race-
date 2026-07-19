"""
param_bildmatrix.py
===================

Qualitative Parameterstudie: variiert 'exponent' (Zeilen) gegen 'zerfall'
(Spalten) und zeigt jeweils das Pheromonfeld nach fester Schrittzahl.

Alle anderen Parameter bleiben konstant, ebenfalls der Zufalls-Seed, damit
Unterschiede allein vom variierten Parameterpaar stammen.
"""

import numpy as np
import matplotlib.pyplot as plt
from schleimpilz_j import Schleimpilz, _farbkarte

# --- variierte Werte -------------------------------------------------------
EXPONENTEN = [1.0, 2.0, 3.0, 5.0, 7.0]      # Zeilen: Spur-Praeferenz
DIFFUSION    = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]  # Spalten: Pheromon-Zerfall

# --- konstante Bedingungen -------------------------------------------------
GROESSE   = 200
N_PILZE   = 5000
SCHRITTE  = 500
KONSTANT  = dict(deposit=0.5, w_abbiegen=0.4, wahrnehmung=1,
                 zerfall=0.05, seed=100)

cmap = _farbkarte()
nz, ns = len(EXPONENTEN), len(DIFFUSION)
fig, axe = plt.subplots(nz, ns, figsize=(3.0 * ns, 3.0 * nz))
fig.patch.set_facecolor("#0a0a12")

for i, exp in enumerate(EXPONENTEN):
    for j, dif in enumerate(DIFFUSION):
        sim = Schleimpilz(groesse=GROESSE, n_pilze=N_PILZE,
                          exponent=exp, diffusion=dif, **KONSTANT)
        sim.distribute_random()               # gleichmaessiger Start ueber das Feld
        for _ in range(SCHRITTE):
            sim.schritt()

        ax = axe[i, j]
        ax.imshow(sim.pheromon, cmap=cmap, vmin=0, vmax=sim.max_pheromon,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0a0a12")
        if i == 0:
            ax.set_title(f"diffusion = {dif}", color="#a8e05f",
                         fontsize=11, family="monospace")
        if j == 0:
            ax.set_ylabel(f"exponent = {exp}", color="#a8e05f",
                          fontsize=11, family="monospace")

fig.suptitle("Pheromonfeld nach %d Schritten  (n=%d, Diffusion=%.2f, seed=%d)"
             % (SCHRITTE, N_PILZE, KONSTANT["zerfall"], KONSTANT["seed"]),
             color="#f7ff9e", fontsize=13, family="monospace", y=1.005)
plt.tight_layout()
plt.savefig("param_exponent_difussion2.png", dpi=95,
            facecolor="#0a0a12", bbox_inches="tight")
print("gespeichert")
