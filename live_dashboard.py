"""
Dashboard en temps réel — Architecture Doya (2000)

Visualise en direct les activations des 3 modules (Cortex LSTM,
Basal Ganglia, Cervelet) pendant l'inférence sur CartPole-v1.

Layout 3×3 :
  [0,0] CartPole (rendu rgb_array)    [0,1] État caché LSTM [8×16]   [0,2] Vecteur contexte [64D]
  [1,0] Probabilités d'action         [1,1] Valeur V(s_t)             [1,2] Dopamine δ (+/-)
  [2,0] Prédiction Cervelet (angle)   [2,1] Erreur prédiction          [2,2] Stats épisode

Usage :
    python live_dashboard.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")      # GUI interactive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import gymnasium as gym

from basal_ganglia import BasalGanglia
from cerebellum   import Cerebellum
from cortex       import Cortex

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH    = "best_full_model.pt"
ENV_NAME           = "CartPole-v1"
N_EPISODES         = 5
ANIMATION_INTERVAL = 50       # ms entre frames
ROLLING_WINDOW     = 100
PRED_WINDOW        = 50       # fenêtre prédiction Cervelet

OBS_DIM      = 4
ACTION_DIM   = 2
CONTEXT_DIM  = 64
HIDDEN_DIM   = 128
BG_INPUT_DIM = CONTEXT_DIM + OBS_DIM

ACTION_LABELS = ["← Gauche", "Droite →"]
OBS_LABELS    = ["Position", "Vitesse", "Angle", "Vit.angul."]


# ─────────────────────────────────────────────────────────────────
#  Chargement des modules
# ─────────────────────────────────────────────────────────────────

def load_modules(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cortex = Cortex(
        obs_dim=OBS_DIM, context_dim=CONTEXT_DIM,
        hidden_dim=HIDDEN_DIM, lr=3e-4,
        use_cerebellum_pred=True,
    )
    bg = BasalGanglia(
        state_dim=BG_INPUT_DIM, action_dim=ACTION_DIM,
        discrete=True, gamma=0.99,
        lr_actor=5e-4, lr_critic=1e-3,
        hidden_dim=HIDDEN_DIM,
    )
    cb = Cerebellum(
        state_dim=OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, lr=1e-3,
        buffer_capacity=20_000, batch_size=64, train_every=10,
    )

    cortex.model.load_state_dict(ckpt["cortex"])
    bg.actor.load_state_dict(ckpt["bg_actor"])
    bg.critic.load_state_dict(ckpt["bg_critic"])
    cb.forward_model.load_state_dict(ckpt["cb_forward"])
    cb.inverse_model.load_state_dict(ckpt["cb_inverse"])

    cortex.model.eval()
    bg.actor.eval()
    bg.critic.eval()
    cb.forward_model.eval()
    cb.inverse_model.eval()

    return cortex, bg, cb


# ─────────────────────────────────────────────────────────────────
#  Initialisation de la figure
# ─────────────────────────────────────────────────────────────────

def init_figure():
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.patch.set_facecolor("#1A1A2E")

    titles = [
        ["CartPole — Environnement", "LSTM : État caché [128D → 8×16]", "Cortex : Vecteur contexte [64D]"],
        ["BG : Probabilités d'action", "BG : Valeur estimée V(s_t)", "BG : Signal Dopamine δ"],
        ["Cervelet : Prédiction ŝ' vs s' (angle)", "Cervelet : Erreur prédiction ||ŝ'−s'||²", "Statistiques épisode"],
    ]
    colors_title = [
        ["#ECF0F1", "#A9CCE3", "#A9CCE3"],
        ["#FAD7A0", "#FAD7A0", "#FAD7A0"],
        ["#A9DFBF", "#A9DFBF", "#ECF0F1"],
    ]
    bg_colors = [
        ["#16213E", "#16213E", "#16213E"],
        ["#16213E", "#16213E", "#16213E"],
        ["#16213E", "#16213E", "#16213E"],
    ]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.set_facecolor(bg_colors[i][j])
            ax.set_title(titles[i][j], fontsize=8, color=colors_title[i][j], pad=4)
            for spine in ax.spines.values():
                spine.set_color("#4A4A8A")
            ax.tick_params(colors="#AAAACC", labelsize=7)

    fig.suptitle(
        "Dashboard Temps Réel — Architecture Doya (Cortex / Basal Ganglia / Cervelet)",
        fontsize=12, color="white", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


# ─────────────────────────────────────────────────────────────────
#  Initialisation des artistes matplotlib
# ─────────────────────────────────────────────────────────────────

def init_artists(axes, env):
    artists = {}

    # [0,0] — CartPole frame
    env.reset()
    frame = env.render()
    ax = axes[0, 0]
    artists["frame_im"] = ax.imshow(frame)
    ax.axis("off")

    # [0,1] — LSTM hidden state heatmap
    ax = axes[0, 1]
    dummy_h = np.zeros((8, 16))
    artists["hidden_im"] = ax.imshow(dummy_h, cmap="coolwarm", vmin=-1, vmax=1,
                                      aspect="auto")
    plt.colorbar(artists["hidden_im"], ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

    # [0,2] — Contexte [64D]
    ax = axes[0, 2]
    ctx_vals = np.zeros(CONTEXT_DIM)
    artists["ctx_bars"] = ax.bar(range(CONTEXT_DIM), ctx_vals,
                                  color="#5DADE2", width=1.0, alpha=0.8)
    ax.set_xlim(-1, CONTEXT_DIM)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Neurone", fontsize=7, color="#AAAACC")
    ax.set_ylabel("Activation (Tanh)", fontsize=7, color="#AAAACC")

    # [1,0] — Probabilités d'action
    ax = axes[1, 0]
    artists["action_bars"] = ax.bar(
        range(ACTION_DIM), [0.5, 0.5],
        color=["#E74C3C", "#3498DB"], alpha=0.85,
    )
    ax.set_xticks(range(ACTION_DIM))
    ax.set_xticklabels(ACTION_LABELS, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probabilité", fontsize=7, color="#AAAACC")
    ax.axhline(0.5, color="white", lw=0.5, ls="--", alpha=0.4)

    # [1,1] — V(s_t) rolling
    ax = axes[1, 1]
    x_roll = np.arange(ROLLING_WINDOW)
    (artists["value_line"],) = ax.plot(x_roll, np.zeros(ROLLING_WINDOW),
                                        color="#F39C12", lw=1.5)
    ax.set_xlim(0, ROLLING_WINDOW)
    ax.set_ylim(-2, 6)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Pas (derniers 100)", fontsize=7, color="#AAAACC")
    ax.set_ylabel("V(s_t)", fontsize=7, color="#AAAACC")

    # [1,2] — Dopamine (redraw à chaque step)
    ax = axes[1, 2]
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Pas (derniers 100)", fontsize=7, color="#AAAACC")
    ax.set_ylabel("δ", fontsize=7, color="#AAAACC")

    # [2,0] — Prédiction Cervelet (angle = dim 2)
    ax = axes[2, 0]
    x_pred = np.arange(PRED_WINDOW)
    (artists["pred_actual"],) = ax.plot(x_pred, np.zeros(PRED_WINDOW),
                                         color="#2ECC71", lw=1.5, label="s' réel")
    (artists["pred_model"],)  = ax.plot(x_pred, np.zeros(PRED_WINDOW),
                                         color="#E74C3C", lw=1.5, ls="--",
                                         label="ŝ' prédit")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, PRED_WINDOW)
    ax.set_xlabel("Derniers pas", fontsize=7, color="#AAAACC")
    ax.set_ylabel("Angle (rad)", fontsize=7, color="#AAAACC")

    # [2,1] — Erreur prédiction rolling
    ax = axes[2, 1]
    (artists["pred_err_line"],) = ax.plot(x_roll, np.zeros(ROLLING_WINDOW),
                                           color="#9B59B6", lw=1.5)
    ax.set_xlim(0, ROLLING_WINDOW)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel("Pas (derniers 100)", fontsize=7, color="#AAAACC")
    ax.set_ylabel("||ŝ'−s'||²", fontsize=7, color="#AAAACC")

    # [2,2] — Stats texte
    axes[2, 2].axis("off")

    return artists


# ─────────────────────────────────────────────────────────────────
#  Fonction de mise à jour (un step par frame)
# ─────────────────────────────────────────────────────────────────

def make_update_fn(state, cortex, bg, cb, env, axes, artists, buffers):

    def update(frame):
        # ── Réinitialisation épisode ────────────────────────────────
        if state["done"]:
            if state["episode"] >= N_EPISODES:
                plt.close(artists["fig"])
                return
            obs, _ = env.reset(seed=42 + state["episode"])
            state["obs_t"]        = torch.FloatTensor(obs)
            state["total_reward"] = 0.0
            state["step"]         = 0
            state["done"]         = False
            state["episode"]     += 1
            state["ep_pred_actual"] = deque(maxlen=PRED_WINDOW)
            state["ep_pred_model"]  = deque(maxlen=PRED_WINDOW)
            cortex.reset()
            # Réinitialise rolling buffers (optionnel — garder l'historique inter-épisodes)

        obs_t = state["obs_t"]

        # ── Step simulation ────────────────────────────────────────
        with torch.no_grad():
            # Prédiction Cervelet
            cereb_pred = cb.predict_next_state(obs_t, 0) \
                         if len(cb.buffer) > 10 else None

            # Contexte Cortex
            context = cortex.encode(obs_t, cerebellum_pred=cereb_pred)

            # BG — action et valeur
            bg_state = torch.cat([context, obs_t], dim=-1)
            dist     = bg.actor(bg_state)
            action_t = dist.sample()
            probs    = dist.probs.detach().numpy()
            action   = action_t.item()
            v_s      = bg.critic(bg_state).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_obs_t = torch.FloatTensor(next_obs)

        # TD-error
        with torch.no_grad():
            next_context = cortex.peek(next_obs_t, cerebellum_pred=cereb_pred)
            next_bg_state = torch.cat([next_context, next_obs_t], dim=-1)
            v_next = bg.critic(next_bg_state).item()
        td_err = reward + 0.99 * v_next * (1.0 - float(done)) - v_s

        # Erreur Cervelet
        with torch.no_grad():
            cb_err = cb.prediction_error(obs_t, action, next_obs_t)
            if cereb_pred is not None:
                pred_angle  = cereb_pred[2].item()
            else:
                pred_angle  = 0.0
        actual_angle = next_obs_t[2].item()

        # Mise à jour state
        state["obs_t"]        = next_obs_t
        state["total_reward"] += reward
        state["step"]         += 1
        state["done"]          = done

        # Mise à jour buffers rolling
        buffers["value"].append(v_s)
        buffers["td"].append(td_err)
        buffers["pred_err"].append(cb_err)
        state["ep_pred_actual"].append(actual_angle)
        state["ep_pred_model"].append(pred_angle)

        # ── Mise à jour artistes ───────────────────────────────────

        # [0,0] CartPole frame
        frame_rgb = env.render()
        artists["frame_im"].set_data(frame_rgb)

        # [0,1] LSTM hidden state
        h = cortex._hidden[0].squeeze().detach().numpy().reshape(8, 16)
        artists["hidden_im"].set_data(h)

        # [0,2] Vecteur contexte
        ctx_vals = context.detach().numpy()
        for bar, val in zip(artists["ctx_bars"], ctx_vals):
            bar.set_height(val)
        axes[0, 2].set_ylim(min(-1.1, ctx_vals.min() - 0.1),
                             max(1.1, ctx_vals.max() + 0.1))

        # [1,0] Probabilités d'action
        for bar, p in zip(artists["action_bars"], probs):
            bar.set_height(float(p))

        # [1,1] V(s_t)
        v_arr = np.array(list(buffers["value"]))
        y = np.zeros(ROLLING_WINDOW)
        y[:len(v_arr)] = v_arr
        artists["value_line"].set_ydata(y)
        mn, mx = y.min(), y.max()
        margin = max(0.5, (mx - mn) * 0.1)
        axes[1, 1].set_ylim(mn - margin, mx + margin)

        # [1,2] Dopamine δ (redraw)
        ax_dopa = axes[1, 2]
        ax_dopa.cla()
        ax_dopa.set_facecolor("#16213E")
        ax_dopa.tick_params(colors="#AAAACC", labelsize=7)
        for spine in ax_dopa.spines.values():
            spine.set_color("#4A4A8A")
        td_arr = np.array(list(buffers["td"]))
        xs = np.arange(len(td_arr))
        ax_dopa.fill_between(xs, 0, td_arr, where=td_arr >= 0,
                             color="#27AE60", alpha=0.7)
        ax_dopa.fill_between(xs, 0, td_arr, where=td_arr < 0,
                             color="#E74C3C", alpha=0.7)
        ax_dopa.axhline(0, color="white", lw=0.5)
        ax_dopa.set_xlim(0, max(1, len(td_arr)))
        ax_dopa.set_title("BG : Signal Dopamine δ", fontsize=8,
                           color="#FAD7A0", pad=4)
        ax_dopa.set_xlabel("Pas (derniers 100)", fontsize=7, color="#AAAACC")
        ax_dopa.set_ylabel("δ", fontsize=7, color="#AAAACC")
        td_label = f"δ = {td_err:+.3f}"
        ax_dopa.text(0.98, 0.95, td_label, transform=ax_dopa.transAxes,
                     ha="right", va="top", fontsize=8,
                     color="#27AE60" if td_err >= 0 else "#E74C3C")

        # [2,0] Prédiction Cervelet
        ax_pred = axes[2, 0]
        ax_pred.cla()
        ax_pred.set_facecolor("#16213E")
        ax_pred.tick_params(colors="#AAAACC", labelsize=7)
        for spine in ax_pred.spines.values():
            spine.set_color("#4A4A8A")
        pa = list(state["ep_pred_actual"])
        pm = list(state["ep_pred_model"])
        xs2 = np.arange(len(pa))
        if len(pa) > 1:
            ax_pred.plot(xs2, pa, color="#2ECC71", lw=1.5, label="s' réel")
            ax_pred.plot(xs2, pm, color="#E74C3C", lw=1.5, ls="--", label="ŝ' prédit")
        ax_pred.legend(fontsize=7, loc="upper right")
        ax_pred.set_title("Cervelet : Prédiction ŝ' vs s' (angle)", fontsize=8,
                           color="#A9DFBF", pad=4)
        ax_pred.set_xlabel("Derniers pas", fontsize=7, color="#AAAACC")
        ax_pred.set_ylabel("Angle (rad)", fontsize=7, color="#AAAACC")

        # [2,1] Erreur prédiction
        pe_arr = np.array(list(buffers["pred_err"]))
        y_pe = np.zeros(ROLLING_WINDOW)
        y_pe[:len(pe_arr)] = pe_arr
        artists["pred_err_line"].set_ydata(y_pe)
        mx_pe = max(pe_arr.max() if len(pe_arr) else 0.1, 0.01)
        axes[2, 1].set_ylim(0, mx_pe * 1.2)

        # [2,2] Stats
        ax_stats = axes[2, 2]
        ax_stats.cla()
        ax_stats.axis("off")
        ax_stats.set_facecolor("#16213E")
        ax_stats.set_title("Statistiques épisode", fontsize=8,
                            color="#ECF0F1", pad=4)

        lines = [
            ("Épisode",       f"{state['episode']} / {N_EPISODES}"),
            ("Pas courant",   f"{state['step']}"),
            ("Récompense ep.",f"{state['total_reward']:.0f}"),
            ("Meilleure",     f"{state['best_reward']:.0f}"),
            ("",              ""),
            ("V(s_t)",        f"{v_s:+.3f}"),
            ("δ courant",     f"{td_err:+.3f}"),
            ("Cerv. err.",    f"{cb_err:.5f}"),
            ("Action",        ACTION_LABELS[action]),
            ("P(←)", f"{probs[0]:.3f}   P(→): {probs[1]:.3f}"),
        ]
        y_pos = 0.95
        for label, value in lines:
            if not label:
                y_pos -= 0.04
                continue
            ax_stats.text(0.05, y_pos, f"{label}:", transform=ax_stats.transAxes,
                          fontsize=9, color="#AAAACC", va="top")
            ax_stats.text(0.55, y_pos, value, transform=ax_stats.transAxes,
                          fontsize=9, color="#ECF0F1", va="top", fontweight="bold")
            y_pos -= 0.09

        if done and state["total_reward"] > state["best_reward"]:
            state["best_reward"] = state["total_reward"]

    return update


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Dashboard Temps Réel — Architecture Doya (2000)")
    print(f"  Modèle : {CHECKPOINT_PATH}")
    print(f"  Épisodes : {N_EPISODES}  |  Interval : {ANIMATION_INTERVAL}ms")
    print("=" * 60)
    print("  Ferme la fenêtre pour arrêter.\n")

    cortex, bg, cb = load_modules(CHECKPOINT_PATH)
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    fig, axes = init_figure()
    artists = init_artists(axes, env)
    artists["fig"] = fig

    # Rolling buffers
    buffers = {
        "value":    deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW),
        "td":       deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW),
        "pred_err": deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW),
    }

    # État mutable partagé
    state = {
        "obs_t":           None,
        "done":            True,
        "episode":         0,
        "total_reward":    0.0,
        "step":            0,
        "best_reward":     0.0,
        "ep_pred_actual":  deque(maxlen=PRED_WINDOW),
        "ep_pred_model":   deque(maxlen=PRED_WINDOW),
    }

    update_fn = make_update_fn(state, cortex, bg, cb, env, axes, artists, buffers)

    # Max frames : N_EPISODES × 500 steps max
    max_frames = N_EPISODES * 500
    anim = animation.FuncAnimation(
        fig,
        update_fn,
        frames=max_frames,
        interval=ANIMATION_INTERVAL,
        blit=False,
        cache_frame_data=False,
        repeat=False,
    )

    plt.show()
    env.close()
    print("Dashboard fermé.")


if __name__ == "__main__":
    main()
