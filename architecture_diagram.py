"""
Schéma de l'architecture Doya (2000) — Cortex / Basal Ganglia / Cervelet
Génère architecture_doya.png pour présentation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

# ─── Palette ────────────────────────────────────────────────────
C_ENV   = "#4A4A6A"   # Environnement
C_CTX   = "#1A5276"   # Cortex
C_BG    = "#784212"   # Basal Ganglia
C_CB    = "#1E8449"   # Cervelet
C_EDGE  = "#AAB7B8"
C_WHITE = "#ECF0F1"
C_GOLD  = "#F4D03F"
C_GREEN = "#2ECC71"
C_RED   = "#E74C3C"
C_BLUE  = "#5DADE2"
C_ORANGE= "#F39C12"
C_PURPLE= "#9B59B6"

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.4, alpha=0.92):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.1,rounding_size={radius}",
        facecolor=color, edgecolor=C_EDGE, linewidth=1.8, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_WHITE, zorder=4)
    if sublabel:
        ax.text(x, y - 0.32, sublabel, ha="center", va="center",
                fontsize=8, color="#BDC3C7", zorder=4, style="italic")

def arrow(ax, x1, y1, x2, y2, color=C_EDGE, lw=1.8, label="", label_offset=(0, 0),
          style="->", connectionstyle="arc3,rad=0.0", alpha=1.0):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=2, alpha=alpha,
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=color, zorder=5,
                bbox=dict(facecolor="#0F1117", edgecolor="none", alpha=0.8, pad=1))

def dot_badge(ax, x, y, color, text, fontsize=7):
    ax.plot(x, y, "o", color=color, markersize=9, zorder=6)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=7)

# ═══════════════════════════════════════════════════════════════
#  TITRE
# ═══════════════════════════════════════════════════════════════
ax.text(8, 9.55, "Architecture des Systèmes d'Apprentissage Complémentaires",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=C_WHITE)
ax.text(8, 9.15, "Doya (2000)  —  Cortex Préfrontal / Noyaux Gris Centraux / Cervelet",
        ha="center", va="center", fontsize=10, color="#BDC3C7", style="italic")

# ═══════════════════════════════════════════════════════════════
#  ENVIRONNEMENT
# ═══════════════════════════════════════════════════════════════
box(ax, 8, 8.3, 4.2, 0.9, C_ENV,
    "ENVIRONNEMENT  —  CartPole-v1",
    "s_t ∈ ℝ⁴  :  [position, vitesse, angle, vit. angulaire]")

# ═══════════════════════════════════════════════════════════════
#  CORTEX
# ═══════════════════════════════════════════════════════════════
box(ax, 3.0, 5.8, 4.8, 2.6, C_CTX,
    "CORTEX  (Préfrontal / Pariétal)")

# Contenu interne Cortex
ax.text(3.0, 6.7, "LSTM  (hidden_dim = 128)", ha="center", fontsize=8.5,
        color=C_BLUE, fontweight="bold")
ax.text(3.0, 6.3,
        "Encoder → LayerNorm → Tanh\n"
        "Context head  →  h_t ∈ ℝ⁶⁴\n"
        "Prediction head  →  ŝ_{t+1} ∈ ℝ⁴",
        ha="center", va="center", fontsize=7.5, color="#BDC3C7",
        linespacing=1.5)

# Badge apprentissage
ax.text(1.0, 5.05,
        "Apprentissage\nnon-supervisé\n(prédiction obs.)",
        ha="center", va="center", fontsize=7, color=C_BLUE,
        bbox=dict(facecolor="#0D2137", edgecolor=C_BLUE, lw=1, pad=3,
                  boxstyle="round,pad=0.3"))

# ═══════════════════════════════════════════════════════════════
#  BASAL GANGLIA
# ═══════════════════════════════════════════════════════════════
box(ax, 8.0, 5.8, 4.8, 2.6, C_BG,
    "NOYAUX GRIS CENTRAUX  (Basal Ganglia)")

ax.text(8.0, 6.7, "Actor  +  Critic  (hidden_dim = 128)", ha="center",
        fontsize=8.5, color=C_ORANGE, fontweight="bold")
ax.text(8.0, 6.3,
        "Actor  :  [h_t ∥ s_t] ∈ ℝ⁶⁸  →  π(a | s)  (Categorical)\n"
        "Critic :  [h_t ∥ s_t] ∈ ℝ⁶⁸  →  V(s)  ∈ ℝ\n"
        "Signal dopamine  δ = r + γ V(s') − V(s)",
        ha="center", va="center", fontsize=7.5, color="#BDC3C7",
        linespacing=1.5)

ax.text(10.8, 5.05,
        "Apprentissage\npar renforcement\n(A2C, γ = 0.99)",
        ha="center", va="center", fontsize=7, color=C_ORANGE,
        bbox=dict(facecolor="#2C1503", edgecolor=C_ORANGE, lw=1, pad=3,
                  boxstyle="round,pad=0.3"))

# ═══════════════════════════════════════════════════════════════
#  CERVELET
# ═══════════════════════════════════════════════════════════════
box(ax, 13.0, 5.8, 4.8, 2.6, C_CB,
    "CERVELET")

ax.text(13.0, 6.7, "Forward Model  +  Inverse Model  (hidden_dim = 128)",
        ha="center", fontsize=8.5, color=C_GREEN, fontweight="bold")
ax.text(13.0, 6.3,
        "Forward  :  [s_t ∥ a_onehot] ∈ ℝ⁶  →  ŝ'_t ∈ ℝ⁴\n"
        "Inverse   :  [s_t ∥ s'_t] ∈ ℝ⁸  →  â_t  (logits)\n"
        "Erreur fibre grimpante  :  ‖ŝ' − s'‖²",
        ha="center", va="center", fontsize=7.5, color="#BDC3C7",
        linespacing=1.5)

ax.text(15.0, 5.05,
        "Apprentissage\nsupervisé\n(Replay Buffer)",
        ha="center", va="center", fontsize=7, color=C_GREEN,
        bbox=dict(facecolor="#0B2A14", edgecolor=C_GREEN, lw=1, pad=3,
                  boxstyle="round,pad=0.3"))

# ═══════════════════════════════════════════════════════════════
#  FLÈCHES PRINCIPALES
# ═══════════════════════════════════════════════════════════════

# ENV → Cortex  (s_t brut)
arrow(ax, 6.0, 8.3, 3.9, 7.1, color=C_BLUE, lw=2.0,
      label="s_t ∈ ℝ⁴", label_offset=(-0.6, 0.2))

# ENV → BG  (s_t brut aussi)
arrow(ax, 8.0, 7.85, 8.0, 7.1, color=C_ORANGE, lw=2.0,
      label="s_t ∈ ℝ⁴", label_offset=(0.55, 0.0))

# ENV → Cervelet
arrow(ax, 10.0, 8.3, 12.1, 7.1, color=C_GREEN, lw=2.0,
      label="s_t, s'_t", label_offset=(0.5, 0.2))

# Cortex → BG  (h_t context)
arrow(ax, 5.4, 5.8, 5.6, 5.8, color=C_BLUE, lw=2.5,
      label="h_t ∈ ℝ⁶⁴", label_offset=(0, 0.3))

# BG → ENV  (action a_t)
arrow(ax, 9.5, 7.85, 9.0, 8.3,  # remonte vers env
      color=C_ORANGE, lw=2.2,
      label="a_t ∈ {0,1}", label_offset=(0.8, 0))

# Cervelet → Cortex  (ŝ' prediction, feedback)
arrow(ax, 10.6, 5.2, 5.4, 5.2,
      color=C_GREEN, lw=1.6, style="->",
      connectionstyle="arc3,rad=-0.25",
      label="ŝ'_t ∈ ℝ⁴  (prédiction)", label_offset=(0, -0.45))

# BG → Cervelet  (a_t pour apprentissage)
arrow(ax, 10.4, 5.8, 10.6, 5.8, color=C_ORANGE, lw=1.6,
      label="a_t", label_offset=(0, 0.25))

# ENV → BG (reward)
arrow(ax, 8.8, 7.85, 8.4, 7.1, color=C_RED, lw=1.8,
      label="r_t, s'_t", label_offset=(1.0, 0.15))

# Dopamine δ (signal interne BG, annotation)
ax.annotate("",
    xy=(8.0, 7.1), xytext=(8.0, 6.5),
    arrowprops=dict(arrowstyle="-", color=C_ORANGE, lw=1.0, ls="dashed"),
    zorder=2)

# ═══════════════════════════════════════════════════════════════
#  ZONE BAS — Légende types d'apprentissage
# ═══════════════════════════════════════════════════════════════
y_leg = 3.5

ax.text(8, 4.25, "Types d'apprentissage", ha="center", fontsize=10,
        fontweight="bold", color=C_WHITE)

leg_items = [
    (C_BLUE,   "Non-supervisé",    "Cortex : prédiction de s_{t+1}"),
    (C_ORANGE, "Par renforcement", "BG : maximisation de la récompense cumul."),
    (C_GREEN,  "Supervisé",        "Cervelet : prédiction du prochain état"),
]
xs = [3.5, 8.0, 12.5]
for (color, title, desc), x in zip(leg_items, xs):
    rect = FancyBboxPatch((x - 2.0, y_leg - 0.55), 4.0, 1.2,
                          boxstyle="round,pad=0.15",
                          facecolor=color + "22", edgecolor=color,
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y_leg + 0.3, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color, zorder=4)
    ax.text(x, y_leg - 0.15, desc, ha="center", va="center",
            fontsize=7.5, color="#BDC3C7", zorder=4)

# ═══════════════════════════════════════════════════════════════
#  ZONE BAS — Flux de données résumé
# ═══════════════════════════════════════════════════════════════
ax.axhline(2.8, color="#333355", lw=0.8, xmin=0.02, xmax=0.98)
ax.text(8, 2.5, "Flux de données à chaque pas de temps t", ha="center",
        fontsize=9, fontweight="bold", color=C_WHITE)

steps = [
    ("①", "Env → Cortex",    "s_t → encode() → h_t",         C_BLUE),
    ("②", "Cortex → BG",     "[h_t ∥ s_t] → π(a|s), V(s)",   C_ORANGE),
    ("③", "BG → Env",        "a_t → step() → s'_t, r_t",      C_RED),
    ("④", "Env → Cervelet",  "(s_t, a_t) → ŝ'_t",             C_GREEN),
    ("⑤", "Cervelet → Ctx",  "ŝ'_t → encode(s_t, ŝ'_t)",      C_PURPLE),
]
for i, (num, src, data, col) in enumerate(steps):
    x = 1.4 + i * 2.7
    ax.text(x, 1.9, num, ha="center", va="center", fontsize=11,
            color=col, fontweight="bold")
    ax.text(x, 1.5, src, ha="center", va="center", fontsize=7.5,
            color=C_WHITE, fontweight="bold")
    ax.text(x, 1.1, data, ha="center", va="center", fontsize=7,
            color="#BDC3C7")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.5, 1.5), xytext=(x + 0.9, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#555577", lw=1.2))

plt.tight_layout(pad=0.3)
plt.savefig("architecture_doya.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Schéma sauvegardé : architecture_doya.png")
