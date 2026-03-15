"""
Test du module Basal Ganglia sur CartPole-v1.

Lance avec :
    python test_basal_ganglia.py

Résultat attendu :
  - La récompense moyenne augmente au fil des épisodes
  - L'erreur TD (δ) converge vers 0
  - Un graphique rewards.png est généré
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from basal_ganglia import BasalGanglia


# ─────────────────────────────────
#  Configuration
# ─────────────────────────────────
ENV_NAME    = "CartPole-v1"
N_EPISODES  = 500
GAMMA       = 0.99
LR_ACTOR    = 3e-4
LR_CRITIC   = 1e-3
HIDDEN_DIM  = 64
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────
#  Initialisation
# ─────────────────────────────────
env = gym.make(ENV_NAME)
state_dim  = env.observation_space.shape[0]   # 4 pour CartPole
action_dim = env.action_space.n               # 2 pour CartPole

print(f"Environnement : {ENV_NAME}")
print(f"  state_dim  = {state_dim}")
print(f"  action_dim = {action_dim}")
print(f"  discret    = True\n")

bg = BasalGanglia(
    state_dim  = state_dim,
    action_dim = action_dim,
    discrete   = True,
    gamma      = GAMMA,
    lr_actor   = LR_ACTOR,
    lr_critic  = LR_CRITIC,
    hidden_dim = HIDDEN_DIM,
)

# ─────────────────────────────────
#  Boucle d'entraînement
# ─────────────────────────────────
episode_rewards = []
episode_lengths = []

print(f"{'Épisode':>8}  {'Récompense':>12}  {'Moy. 50 ep.':>12}  {'δ (dopamine)':>13}")
print("─" * 55)

for episode in range(N_EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    state    = torch.FloatTensor(state)

    total_reward = 0
    step = 0

    while True:
        # 1. L'Actor choisit une action
        action, log_prob = bg.select_action(state)

        # 2. L'environnement répond
        next_state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        # 3. Le Critic calcule δ (signal dopaminergique)
        delta, v_s = bg.compute_td_error(state, reward, next_state, done)

        # 4. Mise à jour Actor + Critic
        bg.update(delta, v_s, log_prob)

        state         = next_state
        total_reward += reward
        step         += 1

        if done:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(step)

    # Logging tous les 50 épisodes
    if (episode + 1) % 50 == 0:
        mean_r = np.mean(episode_rewards[-50:])
        dopamine = bg.get_dopamine_signal()
        print(f"{episode+1:>8}  {total_reward:>12.1f}  {mean_r:>12.1f}  {dopamine:>+13.4f}")

env.close()

# ─────────────────────────────────
#  Résultats finaux
# ─────────────────────────────────
mean_last_100 = np.mean(episode_rewards[-100:])
print(f"\n{'─'*55}")
print(f"Récompense moyenne (100 derniers épisodes) : {mean_last_100:.1f}")
print(f"CartPole est 'résolu' si la moyenne ≥ 475.0")
solved = "✓ RÉSOLU" if mean_last_100 >= 475 else "→ Apprentissage en cours"
print(f"Statut : {solved}")

# ─────────────────────────────────
#  Graphiques
# ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("Basal Ganglia (Actor-Critic) — CartPole-v1", fontsize=13, fontweight="bold")

# Courbe de récompenses
window = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
axes[0].plot(episode_rewards, alpha=0.3, color="#4A90D9", label="Récompense brute")
axes[0].plot(range(window-1, N_EPISODES), smoothed, color="#4A90D9", linewidth=2, label=f"Moyenne glissante ({window} ep.)")
axes[0].axhline(475, color="#E24B4A", linestyle="--", linewidth=1, label="Seuil résolu (475)")
axes[0].set_ylabel("Récompense")
axes[0].set_xlabel("Épisode")
axes[0].legend(fontsize=9)
axes[0].set_title("Apprentissage de l'Actor (striatum dorsal)")

# Signal dopaminergique (TD errors)
td_smooth = np.convolve(bg.td_errors, np.ones(50)/50, mode="valid")
axes[1].plot(bg.td_errors, alpha=0.2, color="#F5A623", linewidth=0.5)
axes[1].plot(range(49, len(bg.td_errors)), td_smooth, color="#F5A623", linewidth=2)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("δ (erreur TD)")
axes[1].set_xlabel("Pas de temps")
axes[1].set_title("Signal dopaminergique δ = r + γV(s') - V(s)  →  converge vers 0")

plt.tight_layout()
plt.savefig("rewards.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : rewards.png")
