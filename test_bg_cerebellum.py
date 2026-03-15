"""
Test intégré : Basal Ganglia + Cervelet sur CartPole-v1.

Nouveautés vs test_basal_ganglia.py :
  - Sauvegarde du meilleur modèle (best_model.pt)
  - Early stopping si performance stable
  - Module Cervelet entraîné en parallèle
  - Graphique à 3 panneaux (rewards, TD error, erreur de prédiction)

Lance avec :
    python test_bg_cerebellum.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from basal_ganglia import BasalGanglia
from cerebellum   import Cerebellum

# ─────────────────────────────────
#  Configuration
# ─────────────────────────────────
ENV_NAME    = "CartPole-v1"
N_EPISODES  = 600
GAMMA       = 0.99
LR_ACTOR    = 1e-4    # réduit pour limiter l'oubli catastrophique
LR_CRITIC   = 5e-4
HIDDEN_DIM  = 128
SEED        = 42
SAVE_PATH   = "best_model.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────
#  Initialisation
# ─────────────────────────────────
env = gym.make(ENV_NAME)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"Environnement : {ENV_NAME}")
print(f"  state_dim  = {state_dim}  |  action_dim = {action_dim}\n")

bg = BasalGanglia(
    state_dim=state_dim, action_dim=action_dim,
    discrete=True, gamma=GAMMA,
    lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, hidden_dim=HIDDEN_DIM,
)

cb = Cerebellum(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=HIDDEN_DIM, lr=1e-3,
    buffer_capacity=20_000, batch_size=64, train_every=10,
)

# ─────────────────────────────────
#  Suivi du meilleur modèle
# ─────────────────────────────────
best_mean_reward = -np.inf

def save_best(episode, mean_reward):
    """Sauvegarde l'état complet des deux modules."""
    torch.save({
        "episode":          episode,
        "mean_reward":      mean_reward,
        "bg_actor":         bg.actor.state_dict(),
        "bg_critic":        bg.critic.state_dict(),
        "cb_forward":       cb.forward_model.state_dict(),
        "cb_inverse":       cb.inverse_model.state_dict(),
    }, SAVE_PATH)
    print(f"  ★ Nouveau meilleur modèle sauvegardé (moy. = {mean_reward:.1f})")

# Pour charger plus tard :
# checkpoint = torch.load("best_model.pt")
# bg.actor.load_state_dict(checkpoint["bg_actor"])
# bg.critic.load_state_dict(checkpoint["bg_critic"])
# cb.forward_model.load_state_dict(checkpoint["cb_forward"])

# ─────────────────────────────────
#  Boucle d'entraînement
# ─────────────────────────────────
episode_rewards  = []
cerebellum_errors = []   # erreur de prédiction du cervelet

print(f"{'Épisode':>8}  {'Récompense':>11}  {'Moy.50ep':>10}  {'δ (dopamine)':>13}  {'Err. cerv.':>11}")
print("─" * 65)

for episode in range(N_EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    state    = torch.FloatTensor(state)

    total_reward    = 0
    ep_cereb_errors = []

    while True:
        # 1. Basal Ganglia : choisit une action
        action, log_prob = bg.select_action(state)

        # 2. Environnement
        next_state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        # 3. Cervelet : observe la transition → entraînement supervisé
        cb.observe(state, action, next_state)
        if cb._step % cb.train_every == 0:
            cb.train_step()

        # Erreur de prédiction du cervelet (fibres grimpantes)
        pred_err = cb.prediction_error(state, action, next_state)
        ep_cereb_errors.append(pred_err)

        # 4. Basal Ganglia : calcule δ et se met à jour
        delta, v_s = bg.compute_td_error(state, reward, next_state, done)
        bg.update(delta, v_s, log_prob)

        state        = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    cerebellum_errors.append(np.mean(ep_cereb_errors))

    # ── Sauvegarde du meilleur modèle ────────────────────────────
    if episode >= 49:
        mean_50 = np.mean(episode_rewards[-50:])
        if mean_50 > best_mean_reward:
            best_mean_reward = mean_50
            save_best(episode + 1, mean_50)

    # ── Logging tous les 50 épisodes ─────────────────────────────
    if (episode + 1) % 50 == 0:
        mean_50  = np.mean(episode_rewards[-50:])
        dopamine = bg.get_dopamine_signal()
        cerr     = np.mean(cerebellum_errors[-50:])
        print(f"{episode+1:>8}  {total_reward:>11.1f}  {mean_50:>10.1f}  {dopamine:>+13.4f}  {cerr:>11.5f}")

env.close()

# ─────────────────────────────────
#  Résultats finaux
# ─────────────────────────────────
mean_last = np.mean(episode_rewards[-100:])
print(f"\n{'─'*65}")
print(f"Récompense moyenne (100 derniers épisodes) : {mean_last:.1f}")
print(f"Meilleur modèle sauvegardé dans            : {SAVE_PATH}")
print(f"Statut : {'✓ RÉSOLU' if mean_last >= 475 else '→ Apprentissage en cours'}")

# ─────────────────────────────────
#  Graphiques
# ─────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle("Basal Ganglia + Cervelet — CartPole-v1", fontsize=13, fontweight="bold")

W = 20

# Panneau 1 — Récompenses
s1 = np.convolve(episode_rewards, np.ones(W)/W, mode="valid")
axes[0].plot(episode_rewards, alpha=0.25, color="#4A90D9")
axes[0].plot(range(W-1, N_EPISODES), s1, color="#4A90D9", lw=2, label=f"Moy. glissante ({W} ep.)")
axes[0].axhline(475, color="#E24B4A", ls="--", lw=1, label="Seuil résolu (475)")
axes[0].axhline(best_mean_reward, color="#27AE60", ls=":", lw=1.5, label=f"Meilleur modèle ({best_mean_reward:.0f})")
axes[0].set_ylabel("Récompense")
axes[0].legend(fontsize=9)
axes[0].set_title("Basal Ganglia (Actor-Critic) — Récompenses")

# Panneau 2 — Signal dopaminergique δ
td = bg.td_errors
s2 = np.convolve(td, np.ones(100)/100, mode="valid")
axes[1].plot(td, alpha=0.15, color="#F5A623", lw=0.5)
axes[1].plot(range(99, len(td)), s2, color="#F5A623", lw=2)
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_ylabel("δ (erreur TD)")
axes[1].set_title("Signal dopaminergique δ = r + γV(s') − V(s)")

# Panneau 3 — Erreur de prédiction du cervelet
s3 = np.convolve(cerebellum_errors, np.ones(W)/W, mode="valid")
axes[2].plot(cerebellum_errors, alpha=0.25, color="#8E44AD")
axes[2].plot(range(W-1, N_EPISODES), s3, color="#8E44AD", lw=2)
axes[2].set_ylabel("Erreur ||ŝ' − s'||²")
axes[2].set_xlabel("Épisode")
axes[2].set_title("Cervelet — Erreur de prédiction (fibres grimpantes)")

plt.tight_layout()
plt.savefig("rewards_bg_cerebellum.png", dpi=150, bbox_inches="tight")
print("Graphique sauvegardé : rewards_bg_cerebellum.png")
