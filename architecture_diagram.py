"""
Schéma de l'architecture Doya (2000) — style article scientifique
Organisation anatomique : Cortex (haut) / BG (bas-gauche) / Cervelet (bas-droit)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Paramètres globaux ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_W, FIG_H = 16, 13
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Palette (style article) ─────────────────────────────────────
C_ENV  = dict(face="#F2F3F4", edge="#717D7E", text="#2C3E50")
C_CTX  = dict(face="#EBF5FB", edge="#1A5276", text="#1A5276")
C_BG   = dict(face="#FEF9E7", edge="#7D6608", text="#6E2F0A")
C_CB   = dict(face="#EAFAF1", edge="#1E8449", text="#1E8449")

# couleurs des flèches
A_OBS   = "#2471A3"   # observation → bleu
A_CTX   = "#1A5276"   # h_t → bleu foncé
A_ACT   = "#B7950B"   # action → or
A_REW   = "#CB4335"   # récompense → rouge
A_PRED  = "#1E8449"   # prédiction → vert
A_GREY  = "#717D7E"   # neutre


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def module_box(cx, cy, w, h, palette, title, subtitle="",
               lines=None, badge=None, badge_pos="bl"):
    """
    Dessine un module (boîte + contenu).
    lines : list of (text, color) tuples
    badge : (text, color)
    """
    x0, y0 = cx - w/2, cy - h/2
    rect = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.25",
        facecolor=palette["face"],
        edgecolor=palette["edge"],
        linewidth=1.8, zorder=3,
    )
    ax.add_patch(rect)

    # Barre de titre colorée
    title_bar = FancyBboxPatch(
        (x0, cy + h/2 - 0.52), w, 0.52,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=palette["edge"], edgecolor="none",
        linewidth=0, zorder=4, clip_on=True,
    )
    ax.add_patch(title_bar)

    ax.text(cx, cy + h/2 - 0.26, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=5)

    if subtitle:
        ax.text(cx, cy + h/2 - 0.78, subtitle,
                ha="center", va="center", fontsize=8.5,
                color=palette["text"], style="italic", zorder=5)

    if lines:
        base_y = cy + h/2 - 1.22
        for txt, col in lines:
            ax.text(cx, base_y, txt,
                    ha="center", va="center", fontsize=8,
                    color=col, zorder=5)
            base_y -= 0.42

    if badge:
        btext, bcol = badge
        bx = x0 + 0.18 if badge_pos.endswith("l") else x0 + w - 0.18
        by = y0 + 0.18 if badge_pos.startswith("b") else y0 + h - 0.18
        ax.text(bx, by, btext,
                ha="left" if badge_pos.endswith("l") else "right",
                va="bottom" if badge_pos.startswith("b") else "top",
                fontsize=6.5, color=bcol, zorder=5,
                bbox=dict(facecolor="white", edgecolor=bcol,
                          lw=0.8, pad=2, boxstyle="round,pad=0.2"))


def arrow(x1, y1, x2, y2, color, lw=1.8,
          label="", lside="center", lpad=(0, 0),
          cs="arc3,rad=0.0", headw=0.25, headt="->"):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=f"{headt},head_width={headw},head_length=0.15",
            color=color, lw=lw,
            connectionstyle=cs,
            shrinkA=4, shrinkB=4,
        ),
        zorder=6,
    )
    if label:
        mx = (x1 + x2) / 2 + lpad[0]
        my = (y1 + y2) / 2 + lpad[1]
        ax.text(mx, my, label,
                ha=lside, va="center", fontsize=8,
                color=color, zorder=7,
                bbox=dict(facecolor="white", edgecolor=color,
                          lw=0.7, pad=2.5, boxstyle="round,pad=0.2"))


def divider(y, label=""):
    ax.axhline(y, color="#D5D8DC", lw=1.0, xmin=0.03, xmax=0.97, zorder=1)
    if label:
        ax.text(FIG_W/2, y + 0.12, label,
                ha="center", va="bottom", fontsize=8,
                color="#808B96", style="italic")


# ════════════════════════════════════════════════════════════════
#  EN-TÊTE
# ════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 12.6,
        "Architecture des Systèmes d'Apprentissage Complémentaires",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#1C2833")
ax.text(FIG_W/2, 12.15,
        "Doya (2000)  ·  Cortex Préfrontal  /  Noyaux Gris Centraux  /  Cervelet  ·  CartPole-v1",
        ha="center", va="center", fontsize=9.5,
        color="#626567", style="italic")

# Ligne de séparation sous le titre
divider(11.85)


# ════════════════════════════════════════════════════════════════
#  ENVIRONNEMENT  (centré, haut)
# ════════════════════════════════════════════════════════════════
ENV_CX, ENV_CY, ENV_W, ENV_H = 8.0, 11.0, 7.5, 0.95
module_box(
    ENV_CX, ENV_CY, ENV_W, ENV_H,
    C_ENV,
    title="ENVIRONNEMENT  —  CartPole-v1",
    subtitle='Observation  s_t ∈ ℝ⁴  :  [x,  ẋ,  θ,  θ̇]    |    Action  a_t ∈ {0 (←), 1 (→)}    |    Récompense  r_t = +1',
)


# ════════════════════════════════════════════════════════════════
#  CORTEX  (large, milieu-haut)
# ════════════════════════════════════════════════════════════════
CTX_CX, CTX_CY, CTX_W, CTX_H = 8.0, 8.3, 13.5, 2.7
module_box(
    CTX_CX, CTX_CY, CTX_W, CTX_H,
    C_CTX,
    title="CORTEX  —  Cortex Préfrontal / Pariétal",
    subtitle="Encodeur d'état récurrent  ·  Mémoire de travail  ·  Apprentissage non-supervisé",
    lines=[
        ("LSTM   (n_layers=1,  hidden_dim=128,  batch_first=True)", "#1A5276"),
        ("Encoder : Linear(obs_dim → 128) + LayerNorm + Tanh", "#2E86C1"),
        ("Context head : Linear(128 → 64) + Tanh    →    h_t ∈ ℝ⁶⁴", "#2E86C1"),
        ("Prediction head : Linear(64 → 4)    →    ŝ_{t+1} ∈ ℝ⁴    (supervision : MSE)", "#2E86C1"),
    ],
    badge=("Apprentissage : min  ‖ŝ_{t+1} − s_{t+1}‖²", "#1A5276"),
    badge_pos="br",
)

# Étiquette anatomique gauche
ax.text(0.55, CTX_CY, "Cortex\ncérébral",
        ha="center", va="center", fontsize=7.5,
        color=C_CTX["edge"], rotation=90,
        bbox=dict(facecolor=C_CTX["face"], edgecolor=C_CTX["edge"],
                  lw=0.8, pad=3, boxstyle="round,pad=0.3"))


# ════════════════════════════════════════════════════════════════
#  BASAL GANGLIA  (bas-gauche)
# ════════════════════════════════════════════════════════════════
BG_CX, BG_CY, BG_W, BG_H = 3.8, 4.2, 6.8, 3.7
module_box(
    BG_CX, BG_CY, BG_W, BG_H,
    C_BG,
    title="NOYAUX GRIS CENTRAUX  (Basal Ganglia)",
    subtitle="Sélection d'action  ·  Apprentissage par renforcement  ·  A2C",
    lines=[
        ("Actor   : [h_t ∥ s_t] ∈ ℝ⁶⁸  →  π(a | s)  (Categorical)", "#7D6608"),
        ("Critic  : [h_t ∥ s_t] ∈ ℝ⁶⁸  →  V(s) ∈ ℝ", "#7D6608"),
        ("Avantage  A_t = G_t − V(s_t)   (Monte Carlo, normalisé)", "#A04000"),
        ("Signal dopamine  δ = r_t + γ V(s') − V(s)", "#CB4335"),
    ],
    badge=("Apprentissage : A2C  (γ=0.99, entropie 0.05)", "#7D6608"),
    badge_pos="br",
)

ax.text(0.55, BG_CY, "Ganglions\nde la base",
        ha="center", va="center", fontsize=7.5,
        color=C_BG["edge"], rotation=90,
        bbox=dict(facecolor=C_BG["face"], edgecolor=C_BG["edge"],
                  lw=0.8, pad=3, boxstyle="round,pad=0.3"))


# ════════════════════════════════════════════════════════════════
#  CERVELET  (bas-droite)
# ════════════════════════════════════════════════════════════════
CB_CX, CB_CY, CB_W, CB_H = 12.2, 4.2, 6.8, 3.7
module_box(
    CB_CX, CB_CY, CB_W, CB_H,
    C_CB,
    title="CERVELET",
    subtitle="Modèle interne  ·  Prédiction  ·  Apprentissage supervisé",
    lines=[
        ("Forward  : [s_t ∥ a_onehot] ∈ ℝ⁶  →  ŝ'_t ∈ ℝ⁴", "#1E8449"),
        ("Inverse   : [s_t ∥ s'_t] ∈ ℝ⁸  →  â_t  (action logits)", "#1E8449"),
        ("Erreur fibre grimpante : ‖ŝ'_t − s'_t‖²", "#117A65"),
        ("Replay Buffer (cap. 20k)  ·  batch=64  ·  train/10 steps", "#117A65"),
    ],
    badge=("Apprentissage : MSE + CrossEntropy (buffer)", "#1E8449"),
    badge_pos="br",
)

ax.text(15.45, CB_CY, "Cervelet",
        ha="center", va="center", fontsize=7.5,
        color=C_CB["edge"], rotation=90,
        bbox=dict(facecolor=C_CB["face"], edgecolor=C_CB["edge"],
                  lw=0.8, pad=3, boxstyle="round,pad=0.3"))


# ════════════════════════════════════════════════════════════════
#  FLÈCHES
# ════════════════════════════════════════════════════════════════

# 1. ENV → Cortex  :  s_t (observation brute)
arrow(ENV_CX - 1.0, ENV_CY - ENV_H/2,
      CTX_CX - 1.0, CTX_CY + CTX_H/2,
      A_OBS, lw=2.0,
      label="s_t ∈ ℝ⁴", lside="right", lpad=(0.15, 0))

# 2. Cortex → BG  :  h_t contexte (flèche principale)
arrow(CTX_CX - 4.0, CTX_CY - CTX_H/2,
      BG_CX + 0.8, BG_CY + BG_H/2,
      A_CTX, lw=2.4,
      label="h_t ∈ ℝ⁶⁴", lside="right", lpad=(0.1, 0.18))

# 3. BG → ENV  :  action a_t (remonte vers environnement)
arrow(BG_CX + BG_W/2 - 0.5, BG_CY + BG_H/2,
      ENV_CX - BG_W/2 + 0.3, ENV_CY - ENV_H/2,
      A_ACT, lw=2.0,
      label="a_t ∈ {0,1}", lside="left", lpad=(-0.15, 0.18),
      cs="arc3,rad=-0.15")

# 4. ENV → BG  :  r_t + s'_t (récompense + prochain état)
arrow(ENV_CX - 2.8, ENV_CY - ENV_H/2,
      BG_CX + 0.0, BG_CY + BG_H/2,
      A_REW, lw=1.6,
      label="r_t,  s'_t", lside="right", lpad=(-1.1, -0.05),
      cs="arc3,rad=0.12")

# 5. Cortex → Cervelet  :  h_t (feedback pour prédiction)
arrow(CTX_CX + 3.5, CTX_CY - CTX_H/2,
      CB_CX - 0.8, CB_CY + CB_H/2,
      A_CTX, lw=1.6,
      label="h_t", lside="left", lpad=(-0.1, 0.18),
      cs="arc3,rad=0.0")

# 6. ENV → Cervelet  :  s_t, s'_t
arrow(ENV_CX + 2.2, ENV_CY - ENV_H/2,
      CB_CX + 0.5, CB_CY + CB_H/2,
      A_OBS, lw=1.6,
      label="s_t,  s'_t", lside="left", lpad=(0.15, 0.0))

# 7. BG → Cervelet  :  a_t (copie d'efférence)
arrow(BG_CX + BG_W/2, BG_CY,
      CB_CX - CB_W/2, CB_CY,
      A_ACT, lw=1.6,
      label="a_t  (copie efférence)", lside="center", lpad=(0, 0.28))

# 8. Cervelet → Cortex  :  ŝ'_t (feedback prédiction)
arrow(CB_CX - CB_W/2 + 0.3, CB_CY + CB_H/2,
      CTX_CX + 4.5, CTX_CY - CTX_H/2,
      A_PRED, lw=1.8,
      label="ŝ'_t ∈ ℝ⁴", lside="right", lpad=(0.1, -0.28),
      cs="arc3,rad=0.25")


# ════════════════════════════════════════════════════════════════
#  SÉPARATEUR + LÉGENDE TYPE D'APPRENTISSAGE
# ════════════════════════════════════════════════════════════════
divider(2.05)

leg_y = 1.35
ax.text(FIG_W/2, 1.88, "Types d'apprentissage par module",
        ha="center", va="center", fontsize=9,
        fontweight="bold", color="#2C3E50")

legend_items = [
    (C_CTX["face"], C_CTX["edge"], "Non-supervisé",
     "Cortex  ·  Prédiction s_{t+1}  ·  SGD (Adam 3×10⁻⁴)"),
    (C_BG["face"],  C_BG["edge"],  "Par renforcement",
     "BG  ·  A2C, Monte Carlo returns  ·  Adam (actor 5×10⁻⁴, critic 10⁻³)"),
    (C_CB["face"],  C_CB["edge"],  "Supervisé",
     "Cervelet  ·  MSE + Cross-entropie  ·  Adam (10⁻³)"),
]

xs = [2.7, 7.2, 11.8]
for (face, edge, title, desc), x in zip(legend_items, xs):
    leg_box = FancyBboxPatch(
        (x - 2.5, leg_y - 0.52), 5.0, 1.05,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3,
    )
    ax.add_patch(leg_box)

    # Barre titre petite
    title_bar2 = FancyBboxPatch(
        (x - 2.5, leg_y + 0.32), 5.0, 0.21,
        boxstyle="round,pad=0.0,rounding_size=0.1",
        facecolor=edge, edgecolor="none", linewidth=0, zorder=4,
    )
    ax.add_patch(title_bar2)

    ax.text(x, leg_y + 0.425, title,
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=5)
    ax.text(x, leg_y - 0.1, desc,
            ha="center", va="center", fontsize=7.2,
            color="#2C3E50", zorder=5)


# ════════════════════════════════════════════════════════════════
#  NOTE DE BAS DE PAGE
# ════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 0.22,
        "Doya, K. (2000). Complementary roles of basal ganglia and cerebellum in learning and motor control. "
        "Current Opinion in Neurobiology, 10(6), 732–739.",
        ha="center", va="center", fontsize=7,
        color="#808B96", style="italic")


# ════════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ════════════════════════════════════════════════════════════════
plt.savefig("architecture_doya.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Schéma sauvegardé : architecture_doya.png")
