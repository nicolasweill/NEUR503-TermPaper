"""
Protocole d'expérience neuroscientifique — Architecture Doya (2000)

Simule 5 conditions inspirées des études de lésion/inactivation sur
l'architecture Cortex–Basal Ganglia–Cervelet entraînée sur CartPole-v1.

Conditions :
  1. Baseline         — Modèle complet, tous modules actifs
  2. Lésion Cortex    — h_t remplacé par zeros(64), BG reçoit toujours 68D
  3. Lésion Cervelet  — observe/train_step désactivés, poids gelés
  4. Déplétion Dopamine — BG sélectionne mais aucun update de poids
  5. Inférence Pure   — Tous modules gelés (pure exploitation)

Produit :
  - experiment_report.png   : figure 12 panneaux
  - Rapport ASCII dans le terminal

Usage :
    python neuroscience_experiment.py
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

from basal_ganglia import BasalGanglia
from cerebellum   import Cerebellum
from cortex       import Cortex

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH  = "best_full_model.pt"
REPORT_PATH      = "experiment_report.png"
ENV_NAME         = "CartPole-v1"
N_EPISODES       = 30
GAMMA            = 0.99
ENTROPY_COEF     = 0.05
GRAD_CLIP        = 1.0
SEED_BASE        = 2000     # différent du seed d'entraînement (42)

OBS_DIM      = 4
ACTION_DIM   = 2
CONTEXT_DIM  = 64
HIDDEN_DIM   = 128
BG_INPUT_DIM = CONTEXT_DIM + OBS_DIM   # 68

CONDITIONS = [
    "Baseline",
    "Lésion Cortex",
    "Lésion Cervelet",
    "Déplétion Dopamine",
    "Inférence Pure",
]

COLORS = {
    "Baseline":            "#4A90D9",
    "Lésion Cortex":       "#E74C3C",
    "Lésion Cervelet":     "#F39C12",
    "Déplétion Dopamine":  "#8E44AD",
    "Inférence Pure":      "#27AE60",
}


# ─────────────────────────────────────────────────────────────────
#  Chargement des modules
# ─────────────────────────────────────────────────────────────────

def load_all_modules(path: str):
    """
    Reconstruit Cortex, BasalGanglia, Cerebellum depuis le checkpoint.
    Retourne (cortex, bg, cb, checkpoint_info).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cortex = Cortex(
        obs_dim=OBS_DIM, context_dim=CONTEXT_DIM,
        hidden_dim=HIDDEN_DIM, lr=3e-4,
        use_cerebellum_pred=True,
    )
    bg = BasalGanglia(
        state_dim=BG_INPUT_DIM, action_dim=ACTION_DIM,
        discrete=True, gamma=GAMMA,
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

    info = {
        "episode":     ckpt.get("episode", "?"),
        "mean_reward": ckpt.get("mean_reward", 0.0),
    }
    return cortex, bg, cb, info


def fresh_copies(base_cortex, base_bg, base_cb):
    """Deep-copy les modules pour repartir d'un état propre entre conditions."""
    return (
        copy.deepcopy(base_cortex),
        copy.deepcopy(base_bg),
        copy.deepcopy(base_cb),
    )


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def make_bg_state(context: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.cat([context, obs], dim=-1)


def compute_returns(rewards, gamma=GAMMA):
    """Monte Carlo returns normalisés (même logique que test_full_model.py)."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def step_td_error(bg, bg_state, reward, next_bg_state, done):
    """TD-error step-wise pour logging (sans mise à jour des poids)."""
    with torch.no_grad():
        v_s    = bg.critic(bg_state).item()
        v_next = bg.critic(next_bg_state).item()
    delta = reward + GAMMA * v_next * (1.0 - float(done)) - v_s
    return delta, v_s


def manual_pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA via SVD numpy, sans sklearn."""
    if data.shape[0] < 2:
        return np.zeros((data.shape[0], n_components))
    centered = data - data.mean(axis=0)
    # Limite à n_components pour éviter SVD complet sur grands N
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


# ─────────────────────────────────────────────────────────────────
#  Boucle d'expérience
# ─────────────────────────────────────────────────────────────────

def run_condition(
    condition_name: str,
    base_cortex, base_bg, base_cb,
    n_episodes: int = N_EPISODES,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Exécute une condition expérimentale.
    Retourne un dict de listes de métriques.
    """
    cortex, bg, cb = fresh_copies(base_cortex, base_bg, base_cb)

    lesion_cortex    = condition_name == "Lésion Cortex"
    lesion_cerebellum= condition_name == "Lésion Cervelet"
    no_bg_update     = condition_name in ("Déplétion Dopamine", "Inférence Pure")
    no_cortex_learn  = condition_name != "Baseline"          # cortex gelé sauf baseline
    no_cerb_learn    = condition_name in ("Lésion Cervelet", "Déplétion Dopamine", "Inférence Pure")
    inference_only   = condition_name == "Inférence Pure"

    env = gym.make(ENV_NAME)

    # Résultats
    rewards_list     = []
    td_errors_all    = []
    cb_errors_all    = []
    ctx_errors_all   = []
    entropies_all    = []
    values_all       = []
    contexts_all     = []
    ep_lengths       = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + seed_offset + ep)
        obs_t  = torch.FloatTensor(obs)
        cortex.reset()

        log_probs_ep = []
        rewards_ep   = []
        bg_states_ep = []
        ep_td_err    = []
        ep_cb_err    = []
        ep_ctx_err   = []
        ep_entropy   = []
        ep_values    = []
        ep_contexts  = []

        step = 0
        while True:
            # 1. Prédiction Cervelet
            cereb_pred = cb.predict_next_state(obs_t, 0) \
                         if (not lesion_cerebellum and len(cb.buffer) > 10) else None

            # 2. Contexte Cortex
            if lesion_cortex:
                context = torch.zeros(CONTEXT_DIM)
            else:
                context = cortex.encode(obs_t, cerebellum_pred=cereb_pred)

            ep_contexts.append(context.detach().numpy().copy())

            # 3. BG — sélection d'action
            bg_state = make_bg_state(context, obs_t)

            if inference_only:
                with torch.no_grad():
                    dist     = bg.actor(bg_state)
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t)
                    entropy  = dist.entropy().item()
            else:
                dist     = bg.actor(bg_state)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                entropy  = dist.entropy().item()

            action = action_t.item()

            # 4. Step environnement
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_obs_t = torch.FloatTensor(next_obs)

            # 5. Contexte suivant (peek → LSTM n'avance pas)
            if lesion_cortex:
                next_context = torch.zeros(CONTEXT_DIM)
            else:
                next_context = cortex.peek(next_obs_t, cerebellum_pred=cereb_pred)

            next_bg_state = make_bg_state(next_context, next_obs_t)

            # 6. TD-error step-wise (logging)
            with torch.no_grad():
                td_err, v_s = step_td_error(bg, bg_state, reward, next_bg_state, done)

            # 7. Apprentissage Cortex
            if not no_cortex_learn and not lesion_cortex:
                ctx_loss = cortex.learn(obs_t, next_obs_t, cerebellum_pred=cereb_pred)
                ep_ctx_err.append(ctx_loss)
            else:
                ep_ctx_err.append(0.0)

            # 8. Apprentissage Cervelet
            if not no_cerb_learn and not lesion_cerebellum:
                cb.observe(obs_t, action, next_obs_t)
                if cb._step % cb.train_every == 0:
                    cb.train_step()

            # 9. Erreur Cervelet (poids gelés ou actifs)
            with torch.no_grad():
                cb_err = cb.prediction_error(obs_t, action, next_obs_t)

            # Accumul pour A2C
            log_probs_ep.append(log_prob)
            rewards_ep.append(reward)
            bg_states_ep.append(bg_state.detach())

            ep_td_err.append(td_err)
            ep_cb_err.append(cb_err)
            ep_entropy.append(entropy)
            ep_values.append(v_s)

            obs_t = next_obs_t
            step += 1
            if done:
                break

        total_reward = sum(rewards_ep)

        # 10. Update A2C fin d'épisode
        if not no_bg_update and not inference_only and len(rewards_ep) > 0:
            returns_tensor = compute_returns(rewards_ep)
            states_tensor  = torch.stack(bg_states_ep)

            bg.actor.train()
            bg.critic.train()

            values_tensor = bg.critic(states_tensor).squeeze()
            advantages    = returns_tensor - values_tensor.detach()

            critic_loss = nn.functional.mse_loss(values_tensor, returns_tensor)
            bg.opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(bg.critic.parameters(), GRAD_CLIP)
            bg.opt_critic.step()

            lp_tensor = torch.stack(log_probs_ep)
            with torch.no_grad():
                entropy_sum = sum(
                    bg.actor(s.unsqueeze(0)).entropy() for s in bg_states_ep
                )
            actor_loss = -(lp_tensor * advantages).mean() \
                         - ENTROPY_COEF * entropy_sum / len(bg_states_ep)
            bg.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(bg.actor.parameters(), GRAD_CLIP)
            bg.opt_actor.step()

            bg.actor.eval()
            bg.critic.eval()

        rewards_list.append(total_reward)
        td_errors_all.extend(ep_td_err)
        cb_errors_all.extend(ep_cb_err)
        ctx_errors_all.extend(ep_ctx_err)
        entropies_all.extend(ep_entropy)
        values_all.extend(ep_values)
        contexts_all.extend(ep_contexts)
        ep_lengths.append(step)

        if verbose and (ep + 1) % 10 == 0:
            print(f"    [{condition_name}] Ep {ep+1:>3}/{n_episodes}  "
                  f"récomp.={total_reward:.0f}  "
                  f"moy.={np.mean(rewards_list):.1f}")

    env.close()

    return {
        "rewards":          rewards_list,
        "td_errors":        td_errors_all,
        "cb_pred_errors":   cb_errors_all,
        "ctx_pred_errors":  ctx_errors_all,
        "action_entropies": entropies_all,
        "value_estimates":  values_all,
        "context_vectors":  np.array(contexts_all, dtype=np.float32),
        "episode_lengths":  ep_lengths,
    }


# ─────────────────────────────────────────────────────────────────
#  Statistiques
# ─────────────────────────────────────────────────────────────────

def compute_stats(result: dict) -> dict:
    stats = {}
    for key, vals in result.items():
        if key == "context_vectors":
            continue
        arr = np.array(vals, dtype=float)
        stats[key] = {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
        }
    return stats


# ─────────────────────────────────────────────────────────────────
#  Figure de rapport
# ─────────────────────────────────────────────────────────────────

def generate_report_figure(all_results: dict, all_stats: dict, save_path: str):
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Rapport Expérience — Architecture Doya (2000)\n"
        "Cortex / Basal Ganglia / Cervelet — CartPole-v1",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    conds = CONDITIONS
    colors = [COLORS[c] for c in conds]

    # ── [0, 0:2] Violin plot récompenses ───────────────────────────
    ax0 = fig.add_subplot(gs[0, 0:2])
    data_violin = [all_results[c]["rewards"] for c in conds]
    parts = ax0.violinplot(data_violin, positions=range(len(conds)),
                           showmeans=True, showmedians=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for comp in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if comp in parts:
            parts[comp].set_color("black")
    ax0.set_xticks(range(len(conds)))
    ax0.set_xticklabels([c.replace(" ", "\n") for c in conds], fontsize=8)
    ax0.set_ylabel("Récompense par épisode")
    ax0.set_title("Distribution des récompenses par condition")
    ax0.axhline(475, color="red", ls="--", lw=1, alpha=0.6, label="Seuil résolu")
    ax0.legend(fontsize=8)

    # ── [0, 2] Timeseries dopamine Baseline ────────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    td_base = np.array(all_results["Baseline"]["td_errors"])
    steps   = np.arange(len(td_base))
    ax1.fill_between(steps, 0, td_base, where=td_base >= 0,
                     color="#27AE60", alpha=0.6, label="δ > 0")
    ax1.fill_between(steps, 0, td_base, where=td_base < 0,
                     color="#E74C3C", alpha=0.6, label="δ < 0")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title("Signal Dopamine — Baseline\n(TD-error δ step-wise)")
    ax1.set_xlabel("Pas de temps")
    ax1.set_ylabel("δ")
    ax1.legend(fontsize=8)

    # ── [0, 3] Bar δ moyen par condition ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 3])
    means_td = [all_stats[c]["td_errors"]["mean"] for c in conds]
    stds_td  = [all_stats[c]["td_errors"]["std"]  for c in conds]
    ax2.bar(range(len(conds)), means_td, color=colors, alpha=0.8, yerr=stds_td,
            capsize=4, error_kw={"linewidth": 1})
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels([c.replace(" ", "\n") for c in conds], fontsize=7)
    ax2.set_ylabel("δ moyen ± σ")
    ax2.set_title("Signal Dopamine moyen\npar condition")

    # ── [1, 0] Box plot erreur Cervelet ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    data_cb = [all_results[c]["cb_pred_errors"] for c in conds]
    bp = ax3.boxplot(data_cb, patch_artist=True, notch=False,
                     medianprops={"color": "black", "lw": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax3.set_xticks(range(1, len(conds) + 1))
    ax3.set_xticklabels([c.replace(" ", "\n") for c in conds], fontsize=7)
    ax3.set_ylabel("||ŝ' − s'||²")
    ax3.set_title("Erreur prédiction Cervelet\n(forward model)")

    # ── [1, 1] Bar erreur Cortex ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    means_ctx = []
    stds_ctx  = []
    hatches   = []
    for c in conds:
        arr = np.array(all_results[c]["ctx_pred_errors"])
        nonzero = arr[arr > 0]
        if len(nonzero) == 0 or c == "Lésion Cortex":
            means_ctx.append(0.0)
        else:
            means_ctx.append(float(np.mean(nonzero)))
        stds_ctx.append(float(np.std(nonzero)) if len(nonzero) > 0 else 0.0)
        hatches.append("///" if c == "Lésion Cortex" else "")

    bars = ax4.bar(range(len(conds)), means_ctx, color=colors, alpha=0.8,
                   yerr=stds_ctx, capsize=4, error_kw={"linewidth": 1})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax4.set_xticks(range(len(conds)))
    ax4.set_xticklabels([c.replace(" ", "\n") for c in conds], fontsize=7)
    ax4.set_ylabel("||ŝ_{t+1} − s_{t+1}||²")
    ax4.set_title("Erreur prédiction Cortex\n(/// = lésion, valeur=0)")

    # ── [1, 2:4] PCA des contextes h_t ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:4])
    all_ctx = np.vstack([all_results[c]["context_vectors"] for c in conds])
    # Subsample pour éviter surcharge visuelle
    max_pts = 2000
    all_ctx_proj = manual_pca(all_ctx)
    offset = 0
    for c in conds:
        n = len(all_results[c]["context_vectors"])
        proj = all_ctx_proj[offset:offset + n]
        # Sous-échantillonne
        idx = np.random.choice(len(proj), min(max_pts // len(conds), len(proj)),
                               replace=False)
        ax5.scatter(proj[idx, 0], proj[idx, 1], c=COLORS[c], alpha=0.35,
                    s=8, label=c)
        offset += n
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("PC2")
    ax5.set_title("PCA — Vecteurs contexte h_t [64D] → 2D\n(Représentations corticales par condition)")
    ax5.legend(fontsize=7, markerscale=2)

    # ── [2, 0] Bar entropie ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    means_ent = [all_stats[c]["action_entropies"]["mean"] for c in conds]
    stds_ent  = [all_stats[c]["action_entropies"]["std"]  for c in conds]
    ax6.bar(range(len(conds)), means_ent, color=colors, alpha=0.8,
            yerr=stds_ent, capsize=4, error_kw={"linewidth": 1})
    ax6.set_xticks(range(len(conds)))
    ax6.set_xticklabels([c.replace(" ", "\n") for c in conds], fontsize=7)
    ax6.set_ylabel("Entropie H[π(·|s)]")
    ax6.set_title("Entropie d'action moyenne\n(exploration vs exploitation)")

    # ── [2, 1] V(s_t) rolling mean ─────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    W = 100
    for c in conds:
        vals = np.array(all_results[c]["value_estimates"])
        if len(vals) >= W:
            smoothed = np.convolve(vals, np.ones(W) / W, mode="valid")
            ax7.plot(smoothed, color=COLORS[c], lw=1.5, alpha=0.8, label=c)
    ax7.set_xlabel("Pas de temps (rolling 100)")
    ax7.set_ylabel("V(s_t)")
    ax7.set_title("Valeur estimée V(s_t)\n(rolling mean 100 steps)")
    ax7.legend(fontsize=7)

    # ── [2, 2] Diagramme des modules ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)

    box_style = dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", edgecolor="#2C3E50", lw=1.5)
    arr_props = dict(arrowstyle="->", color="#2C3E50", lw=1.5)

    ax8.text(0.5, 0.88, "ENVIRONNEMENT\nCartPole-v1", ha="center", va="center",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="#BDC3C7",
                                   edgecolor="#2C3E50", lw=1.5))
    ax8.text(0.5, 0.65, "CORTEX (LSTM)\nEncode s_t → h_t [64D]", ha="center",
             va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
             facecolor="#D6EAF8", edgecolor="#2980B9", lw=1.5))
    ax8.text(0.5, 0.42, "BASAL GANGLIA\nActor-Critic A2C", ha="center", va="center",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
             facecolor="#FDEBD0", edgecolor="#E67E22", lw=1.5))
    ax8.text(0.5, 0.18, "CERVELET\nForward + Inverse Model", ha="center", va="center",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
             facecolor="#D5F5E3", edgecolor="#27AE60", lw=1.5))

    ax8.annotate("", xy=(0.5, 0.71), xytext=(0.5, 0.81),
                 arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))
    ax8.text(0.55, 0.76, "s_t", fontsize=7, color="#555")

    ax8.annotate("", xy=(0.5, 0.48), xytext=(0.5, 0.58),
                 arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))
    ax8.text(0.55, 0.52, "h_t + s_t [68D]", fontsize=7, color="#555")

    ax8.annotate("", xy=(0.5, 0.24), xytext=(0.5, 0.35),
                 arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))
    ax8.text(0.55, 0.29, "a_t", fontsize=7, color="#555")

    ax8.annotate("", xy=(0.15, 0.42), xytext=(0.15, 0.65),
                 arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2,
                                 connectionstyle="arc3,rad=0.3"))
    ax8.text(0.02, 0.53, "ŝ'_t\n(cereb\npred)", fontsize=6, color="#27AE60")

    ax8.set_title("Architecture des modules\n(flux d'information)", fontsize=9)

    # ── [2, 3] Tableau résumé ───────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 3])
    ax9.axis("off")

    col_labels = ["Condition", "Récomp.\nmoy ± σ", "δ moy", "Cerv.err\nmoy"]
    table_data = []
    for c in conds:
        r_m = all_stats[c]["rewards"]["mean"]
        r_s = all_stats[c]["rewards"]["std"]
        d_m = all_stats[c]["td_errors"]["mean"]
        cb_m = all_stats[c]["cb_pred_errors"]["mean"]
        row = [
            c.replace("Déplétion ", "Dépl."),
            f"{r_m:.0f} ± {r_s:.0f}",
            f"{d_m:+.3f}",
            f"{cb_m:.4f}",
        ]
        table_data.append(row)

    tbl = ax9.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.2, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F3F4")
    ax9.set_title("Tableau résumé statistique", fontsize=9, pad=15)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure sauvegardée : {save_path}")


# ─────────────────────────────────────────────────────────────────
#  Rapport texte
# ─────────────────────────────────────────────────────────────────

def print_text_report(all_stats: dict, ckpt_info: dict):
    w = 72
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + " RAPPORT EXPÉRIENCE NEUROSCIENCES — Architecture Doya (2000)".center(w - 2) + "║")
    print("║" + f" Modèle chargé : épisode {ckpt_info['episode']}, "
          f"moy. entraîn. {ckpt_info['mean_reward']:.1f}".center(w - 2) + "║")
    print("║" + f" CartPole-v1  |  {N_EPISODES} épisodes par condition".center(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")

    header = f"  {'Condition':<24} {'Récomp.±σ':>12}  {'δ moy':>8}  {'Cerv.err':>10}  {'Entropie':>8}"
    print("║" + header + "║")
    print("╠" + "─" * (w - 2) + "╣")

    for c in CONDITIONS:
        s = all_stats[c]
        r_m = s["rewards"]["mean"]
        r_s = s["rewards"]["std"]
        d_m = s["td_errors"]["mean"]
        cb_m = s["cb_pred_errors"]["mean"]
        e_m = s["action_entropies"]["mean"]

        solved = "✓" if r_m >= 475 else " "
        line = (f"  {c:<24} {r_m:>6.0f}±{r_s:<5.0f}  {d_m:>+8.4f}  "
                f"{cb_m:>10.5f}  {e_m:>8.4f} {solved}")
        print("║" + line + "║")

    print("╠" + "─" * (w - 2) + "╣")
    print("║" + "  ✓ = récompense moyenne ≥ 475 (problème résolu)".ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")

    print("\n  Interprétation neuroscientifique :")
    print("  ─────────────────────────────────")

    base_r  = all_stats["Baseline"]["rewards"]["mean"]
    ctx_r   = all_stats["Lésion Cortex"]["rewards"]["mean"]
    cerb_r  = all_stats["Lésion Cervelet"]["rewards"]["mean"]
    dopa_r  = all_stats["Déplétion Dopamine"]["rewards"]["mean"]
    inf_r   = all_stats["Inférence Pure"]["rewards"]["mean"]

    def pct_change(x, ref):
        return (x - ref) / max(ref, 1.0) * 100

    print(f"  • Lésion Cortex      : {pct_change(ctx_r, base_r):+.1f}% vs Baseline "
          f"→ {'impact majeur' if abs(pct_change(ctx_r, base_r)) > 20 else 'impact modéré'}")
    print(f"  • Lésion Cervelet    : {pct_change(cerb_r, base_r):+.1f}% vs Baseline "
          f"→ {'impact majeur' if abs(pct_change(cerb_r, base_r)) > 20 else 'impact modéré'}")
    print(f"  • Déplétion Dopamine : {pct_change(dopa_r, base_r):+.1f}% vs Baseline "
          f"→ {'apprentissage bloqué' if dopa_r < base_r * 0.7 else 'effet limité'}")
    print(f"  • Inférence Pure     : {pct_change(inf_r, base_r):+.1f}% vs Baseline "
          f"→ {'exploitation optimale' if inf_r >= base_r * 0.95 else 'dégradation inattendue'}")
    print()


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  EXPÉRIENCE NEUROSCIENTIFIQUE — Architecture Doya (2000)")
    print("  Chargement du modèle :", CHECKPOINT_PATH)
    print("=" * 72)

    torch.manual_seed(42)
    np.random.seed(42)

    base_cortex, base_bg, base_cb, ckpt_info = load_all_modules(CHECKPOINT_PATH)

    print(f"\n  Modèle chargé : épisode {ckpt_info['episode']}, "
          f"moy. entraîn. {ckpt_info['mean_reward']:.1f}")
    print(f"  {N_EPISODES} épisodes par condition × {len(CONDITIONS)} conditions\n")

    all_results = {}

    for i, condition in enumerate(CONDITIONS):
        print(f"\n[{i+1}/{len(CONDITIONS)}] Condition : {condition}")
        print("  " + "─" * 50)
        result = run_condition(
            condition_name=condition,
            base_cortex=base_cortex,
            base_bg=base_bg,
            base_cb=base_cb,
            n_episodes=N_EPISODES,
            seed_offset=i * N_EPISODES,
            verbose=True,
        )
        all_results[condition] = result
        mean_r = np.mean(result["rewards"])
        print(f"  → Récompense moyenne : {mean_r:.1f}")

    all_stats = {c: compute_stats(r) for c, r in all_results.items()}

    print_text_report(all_stats, ckpt_info)

    print("Génération de la figure de rapport...")
    generate_report_figure(all_results, all_stats, REPORT_PATH)
    print("Terminé.")


if __name__ == "__main__":
    main()
