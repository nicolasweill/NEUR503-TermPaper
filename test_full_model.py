"""
Test complet : Cortex + Basal Ganglia + Cervelet — CartPole-v1

Architecture de Doya intégrée :

  Environnement
       │ s_t (obs brute)
       ▼
  ┌─────────────┐
  │    CORTEX   │  h_t = encode(s_t)      ← forward sans gradient
  │    (LSTM)   │  learn(s_t, s_{t+1})    ← entraînement auto-supervisé
  └──────┬──────┘
         │ h_t (contexte enrichi)
         ▼
  ┌─────────────────────┐
  │   BASAL GANGLIA     │  a_t ~ π(h_t)
  │   (Actor-Critic)    │  δ = r + γV(h') - V(h)
  └──────┬──────────────┘
         │ a_t
         ▼
  Environnement → s_{t+1}, r_t
         │
         ▼
  ┌─────────────┐
  │  CERVELET   │  ŝ' = f(s_t, a_t)       ← apprentissage supervisé
  └─────────────┘

Test complet : Cortex + Basal Ganglia + Cervelet — CartPole-v1
Architecture Doya intégrée — version corrigée

Corrections vs version précédente :
  - peek() pour h_{t+1} : le LSTM n'avance plus 2x par step
  - Gradient clipping sur Actor et Critic
  - Bonus d'entropie pour forcer l'exploration
  - Warmup du Cortex (50 ep.) avant d'entraîner les BG

Lance avec :
    python test_full_model.py
"""
"""
Test complet : Cortex + Basal Ganglia + Cervelet — CartPole-v1
Architecture Doya — version stable

Corrections vs version précédente :
  - Critic initialisé avec petits poids (évite sur-estimation de V)
  - BG reçoit concat[h_t, s_t] : contexte cortical + obs brute
    → ancrage stable même si h_t change (fidèle à Doya : BG reçoit
      cortex ET entrées sensorielles directes)
  - peek() pour h_{t+1} sans double avance du LSTM
  - Gradient clipping + entropie conservés
"""
"""
Test complet : Cortex + Basal Ganglia + Cervelet — CartPole-v1
Architecture Doya — version A2C + cortex gelé

Corrections clés :
  - Cortex gelé après warmup → features stationnaires pour le Critic
  - Monte Carlo returns (A2C) → plus stable que TD(0) sur épisodes courts
  - Normalisation des returns → δ dans une plage raisonnable
  - Cervelet continue d'apprendre même après gel du Cortex
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from basal_ganglia import BasalGanglia
from cerebellum   import Cerebellum
from cortex       import Cortex

# ─────────────────────────────────
#  Configuration
# ─────────────────────────────────
ENV_NAME     = "CartPole-v1"
N_EPISODES   = 800
WARMUP_EP    = 80       # épisodes de pré-entraînement du cortex
GAMMA        = 0.99
HIDDEN_DIM   = 128
CONTEXT_DIM  = 64
ENTROPY_COEF = 0.05
GRAD_CLIP    = 1.0
SEED         = 42
SAVE_PATH    = "best_full_model.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────
#  Modules
# ─────────────────────────────────
env = gym.make(ENV_NAME)
obs_dim    = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n               # 2
bg_input_dim = CONTEXT_DIM + obs_dim          # 68

print("=" * 62)
print("  Architecture Doya — Cortex / BG / Cervelet")
print("=" * 62)
print(f"  obs_dim={obs_dim}  action_dim={action_dim}")
print(f"  BG input = context({CONTEXT_DIM}) + obs({obs_dim}) = {bg_input_dim}")
print(f"  Warmup cortex : {WARMUP_EP} ep. → puis gelé")
print(f"  Méthode BG    : A2C (Monte Carlo returns)")
print("=" * 62 + "\n")

cortex = Cortex(
    obs_dim=obs_dim, context_dim=CONTEXT_DIM,
    hidden_dim=HIDDEN_DIM, lr=3e-4,
    use_cerebellum_pred=True,
)

bg = BasalGanglia(
    state_dim=bg_input_dim, action_dim=action_dim,
    discrete=True, gamma=GAMMA,
    lr_actor=5e-4, lr_critic=1e-3,
    hidden_dim=HIDDEN_DIM,
)

cb = Cerebellum(
    state_dim=obs_dim, action_dim=action_dim,
    hidden_dim=HIDDEN_DIM, lr=1e-3,
    buffer_capacity=20_000, batch_size=64, train_every=10,
)

n_ctx = sum(p.numel() for p in cortex.model.parameters())
n_bg  = sum(p.numel() for p in bg.actor.parameters()) + \
        sum(p.numel() for p in bg.critic.parameters())
n_cb  = sum(p.numel() for p in cb.forward_model.parameters()) + \
        sum(p.numel() for p in cb.inverse_model.parameters())
print(f"Paramètres — Cortex:{n_ctx:,}  BG:{n_bg:,}  Cerv:{n_cb:,}  "
      f"Total:{n_ctx+n_bg+n_cb:,}\n")

def make_bg_input(context, obs):
    return torch.cat([context, obs], dim=-1)

def compute_returns(rewards, gamma=GAMMA):
    """Monte Carlo : retourne les returns G_t = Σ γ^k r_{t+k}."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalisation → moyenne=0, std=1 → δ dans [-3, +3]
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# ─────────────────────────────────
#  Sauvegarde
# ─────────────────────────────────
best_mean = -np.inf

def save_best(ep, mean_r):
    torch.save({
        "episode": ep, "mean_reward": mean_r,
        "cortex":     cortex.model.state_dict(),
        "bg_actor":   bg.actor.state_dict(),
        "bg_critic":  bg.critic.state_dict(),
        "cb_forward": cb.forward_model.state_dict(),
        "cb_inverse": cb.inverse_model.state_dict(),
    }, SAVE_PATH)
    print(f"  ★  Meilleur modèle sauvegardé → moy. {mean_r:.1f}")

# ─────────────────────────────────
#  Boucle d'entraînement
# ─────────────────────────────────
episode_rewards   = []
cortex_errors     = []
cerebellum_errors = []
cortex_frozen     = False

print(f"{'Ep':>6}  {'Récomp.':>8}  {'Moy.50':>8}  {'δ moy':>8}  "
      f"{'Ctx.err':>8}  {'Cerv.err':>9}  {'Phase':>10}")
print("─" * 72)

for episode in range(N_EPISODES):

    # ── Gel du Cortex après warmup ────────────────────────────────
    if episode == WARMUP_EP and not cortex_frozen:
        for p in cortex.model.parameters():
            p.requires_grad_(False)
        cortex_frozen = True
        print(f"\n  [Ep {episode}] Cortex gelé — BG commence l'apprentissage\n")

    obs, _ = env.reset(seed=SEED + episode)
    obs_t  = torch.FloatTensor(obs)
    cortex.reset()

    # Buffers épisode
    log_probs, rewards_ep     = [], []
    bg_states, ep_ctx_errors  = [], []
    ep_cerv_errors            = []
    phase = "warmup" if episode < WARMUP_EP else "A2C"

    while True:
        # 1. Cervelet : prédiction courante
        cereb_pred = cb.predict_next_state(obs_t, 0) \
                     if len(cb.buffer) > 10 else None

        # 2. Cortex : encode s_t → h_t
        context = cortex.encode(obs_t, cerebellum_pred=cereb_pred)

        # 3. BG : sélectionne action depuis concat[h_t, s_t]
        bg_state = make_bg_input(context, obs_t)
        dist     = bg.actor(bg_state)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        action   = action_t.item()

        # 4. Environnement
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_obs_t = torch.FloatTensor(next_obs)

        # 5. Cortex : apprend (seulement pendant warmup)
        if not cortex_frozen:
            ctx_loss = cortex.learn(obs_t, next_obs_t, cerebellum_pred=cereb_pred)
            ep_ctx_errors.append(ctx_loss)

        # 6. Cervelet : toujours actif
        cb.observe(obs_t, action, next_obs_t)
        if cb._step % cb.train_every == 0:
            cb.train_step()
        ep_cerv_errors.append(cb.prediction_error(obs_t, action, next_obs_t))

        # Accumule pour A2C
        log_probs.append(log_prob)
        rewards_ep.append(reward)
        bg_states.append(bg_state)

        obs_t        = next_obs_t
        if done:
            break

    total_reward = sum(rewards_ep)

    # ── Mise à jour A2C (fin d'épisode, après warmup) ─────────────
    if episode >= WARMUP_EP:
        returns = compute_returns(rewards_ep)           # G_t normalisés

        # Valeurs V(s_t) pour tout l'épisode
        states_tensor = torch.stack(bg_states)
        values        = bg.critic(states_tensor).squeeze()

        # Avantages : A_t = G_t - V(s_t)
        advantages = returns - values.detach()

        # Critic loss : MSE(V(s_t), G_t)
        critic_loss = nn.functional.mse_loss(values, returns)
        bg.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(bg.critic.parameters(), GRAD_CLIP)
        bg.opt_critic.step()

        # Actor loss : -E[log π(a|s) * A_t] - entropie
        lp_tensor = torch.stack(log_probs)

        # Recalcule l'entropie pour tout l'épisode
        entropy_sum = sum(
            bg.actor(s.unsqueeze(0)).entropy() for s in bg_states
        )
        actor_loss = -(lp_tensor * advantages).mean() \
                     - ENTROPY_COEF * entropy_sum / len(bg_states)
        bg.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(bg.actor.parameters(), GRAD_CLIP)
        bg.opt_actor.step()

        # δ moyen de l'épisode (pour logging)
        delta_mean = advantages.mean().item()
        bg.td_errors.append(delta_mean)

    episode_rewards.append(total_reward)
    cortex_errors.append(np.mean(ep_ctx_errors) if ep_ctx_errors else 0)
    cerebellum_errors.append(np.mean(ep_cerv_errors))

    if episode >= WARMUP_EP + 49:
        mean_50 = np.mean(episode_rewards[-50:])
        if mean_50 > best_mean:
            best_mean = mean_50
            save_best(episode + 1, mean_50)

    if (episode + 1) % 50 == 0:
        mean_50  = np.mean(episode_rewards[-50:])
        dopamine = bg.td_errors[-1] if bg.td_errors else 0.0
        c_err    = np.mean(cortex_errors[-50:])
        cb_err   = np.mean(cerebellum_errors[-50:])
        print(f"{episode+1:>6}  {total_reward:>8.0f}  {mean_50:>8.1f}"
              f"  {dopamine:>+8.3f}  {c_err:>8.4f}  {cb_err:>9.5f}  {phase:>10}")

env.close()

mean_last = np.mean(episode_rewards[-100:])
print(f"\n{'─'*72}")
print(f"Récompense moyenne (100 derniers) : {mean_last:.1f}")
print(f"Meilleur modèle : {SAVE_PATH}  (moy. {best_mean:.1f})")
print(f"Statut : {'✓ RÉSOLU' if mean_last >= 475 else '→ Apprentissage en cours'}")

# ─────────────────────────────────
#  Graphiques
# ─────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(11, 13))
fig.suptitle("Architecture Doya — Cortex / BG / Cervelet\nCartPole-v1  (A2C + cortex gelé)",
             fontsize=13, fontweight="bold")

W = 20
def smooth(x, w=W):
    return np.convolve(x, np.ones(w)/w, mode="valid")

# Récompenses
axes[0].plot(episode_rewards, alpha=0.2, color="#4A90D9")
axes[0].plot(range(W-1, N_EPISODES), smooth(episode_rewards),
             color="#4A90D9", lw=2, label=f"Moy. glissante ({W} ep.)")
axes[0].axvline(WARMUP_EP, color="gray", ls=":", lw=1.5,
                label=f"Cortex gelé (ep. {WARMUP_EP})")
axes[0].axhline(475, color="#E24B4A", ls="--", lw=1, label="Seuil résolu")
if best_mean > 0:
    axes[0].axhline(best_mean, color="#27AE60", ls=":", lw=1.5,
                    label=f"Meilleur ({best_mean:.0f})")
axes[0].set_ylabel("Récompense")
axes[0].legend(fontsize=9)
axes[0].set_title("Basal Ganglia — Récompenses")

# Avantages A_t (= δ en MC)
td = bg.td_errors
if td:
    axes[1].plot(td, alpha=0.4, color="#F5A623", lw=1)
    w2 = min(30, len(td))
    if len(td) >= w2:
        axes[1].plot(range(w2-1, len(td)),
                     np.convolve(td, np.ones(w2)/w2, mode="valid"),
                     color="#F5A623", lw=2)
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_ylabel("Avantage A_t (normalisé)")
axes[1].set_title("Basal Ganglia — Avantage moyen par épisode A_t = G_t − V(s_t)")

# Erreur Cortex (warmup seulement)
axes[2].plot(cortex_errors[:WARMUP_EP], color="#8E44AD", lw=1.5,
             label="Warmup (actif)")
axes[2].axvline(WARMUP_EP, color="gray", ls=":", lw=1.5, label="Gel")
axes[2].set_ylabel("||ŝ_{t+1} − s_{t+1}||²")
axes[2].legend(fontsize=9)
axes[2].set_title("Cortex — Erreur de prédiction (gelé après warmup)")

# Erreur Cervelet
axes[3].plot(cerebellum_errors, alpha=0.25, color="#16A085")
axes[3].plot(range(W-1, N_EPISODES), smooth(cerebellum_errors),
             color="#16A085", lw=2)
axes[3].set_ylabel("||ŝ' − s'||²")
axes[3].set_xlabel("Épisode")
axes[3].set_title("Cervelet — Erreur de prédiction (actif tout l'entraînement)")

plt.tight_layout()
plt.savefig("full_model.png", dpi=150, bbox_inches="tight")
print("Graphique sauvegardé : full_model.png")