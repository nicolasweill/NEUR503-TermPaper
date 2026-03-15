"""
Basal Ganglia module — Actor-Critic with TD-learning
Inspired by Doya (2000): "Complementary roles of basal ganglia and cerebellum"

Architecture:
  - Critic  (striatum ventral / Nucleus Accumbens) : estime V(s)
  - Actor   (striatum dorsal → GPi → thalamus)     : produit π(a|s)
  - Delta   (dopamine SNc/VTA)                      : erreur TD = r + γV(s') - V(s)

Ce module est conçu pour être branché sur le Cortex (encodeur d'état)
et le Cervelet (forward model) dans l'architecture complète.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


# ─────────────────────────────────────────────
#  CRITIC  — estime la valeur d'un état V(s)
#  Correspond au striatum ventral (Nucleus Accumbens)
# ─────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # scalaire V(s)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne V(s) — valeur estimée de l'état."""
        return self.net(state).squeeze(-1)


# ─────────────────────────────────────────────
#  ACTOR  — produit la politique π(a|s)
#  Correspond au striatum dorsal → GPi → cortex moteur
#  Version discrète (CartPole) et continue (Pendulum)
# ─────────────────────────────────────────────

class ActorDiscrete(nn.Module):
    """Pour les espaces d'action discrets (ex: CartPole)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        """Retourne une distribution Categorical sur les actions."""
        logits = self.net(state)
        return Categorical(logits=logits)


class ActorContinuous(nn.Module):
    """Pour les espaces d'action continus (ex: Pendulum)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std   = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Normal:
        """Retourne une distribution Normale sur les actions."""
        h   = self.net(state)
        mu  = torch.tanh(self.mean_head(h))   # borné dans [-1, 1]
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


# ─────────────────────────────────────────────
#  BASAL GANGLIA — module complet Actor-Critic
#  Gère la boucle d'apprentissage TD
# ─────────────────────────────────────────────

class BasalGanglia(nn.Module):
    """
    Module Basal Ganglia complet.

    Paramètres
    ----------
    state_dim   : dimension du vecteur d'état (venant du Cortex)
    action_dim  : nombre d'actions (discret) ou dim action (continu)
    discrete    : True = CartPole, False = Pendulum/robotique
    gamma       : facteur d'escompte (0 < γ < 1)
    lr_actor    : taux d'apprentissage de l'Actor
    lr_critic   : taux d'apprentissage du Critic
    hidden_dim  : taille des couches cachées
    """
    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        discrete:   bool  = True,
        gamma:      float = 0.99,
        lr_actor:   float = 3e-4,
        lr_critic:  float = 1e-3,
        hidden_dim: int   = 64,
    ):
        super().__init__()
        self.gamma    = gamma
        self.discrete = discrete

        # Instanciation des sous-modules
        self.critic = Critic(state_dim, hidden_dim)
        if discrete:
            self.actor = ActorDiscrete(state_dim, action_dim, hidden_dim)
        else:
            self.actor = ActorContinuous(state_dim, action_dim, hidden_dim)

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Historique pour le logging
        self.td_errors = []

    def select_action(self, state: torch.Tensor):
        """
        Sélectionne une action depuis la politique courante.

        Retourne
        --------
        action    : int (discret) ou Tensor (continu)
        log_prob  : log π(a|s), nécessaire pour la mise à jour de l'Actor
        """
        dist      = self.actor(state)
        action    = dist.sample()
        log_prob  = dist.log_prob(action)
        if self.discrete:
            return action.item(), log_prob
        return action, log_prob

    def compute_td_error(
        self,
        state:      torch.Tensor,
        reward:     float,
        next_state: torch.Tensor,
        done:       bool,
    ) -> torch.Tensor:
        """
        Calcule l'erreur TD (signal dopaminergique).

        δ = r + γ · V(s') · (1 - done) - V(s)

        done = 1 en fin d'épisode (pas de bootstrap sur V(s'))
        """
        v_s      = self.critic(state)
        with torch.no_grad():
            v_next = self.critic(next_state) * (1.0 - float(done))
        delta    = reward + self.gamma * v_next - v_s
        self.td_errors.append(delta.item())
        return delta, v_s

    def update(
        self,
        delta:    torch.Tensor,
        v_s:      torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """
        Met à jour l'Actor et le Critic à partir de δ.

        Critic : minimise (δ)² → V(s) converge vers la vraie valeur
        Actor  : monte le gradient δ · ∇ log π(a|s)
                 si δ > 0 : l'action était meilleure que prévu → on l'encourage
                 si δ < 0 : elle était pire que prévu → on la décourage
        """
        # ── Critic loss ──────────────────────────────────────────────
        # On veut minimiser l'erreur de prédiction de valeur
        critic_loss = delta.pow(2)

        self.opt_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.opt_critic.step()

        # ── Actor loss ───────────────────────────────────────────────
        # On monte le gradient pondéré par δ (detach : δ est un signal, pas un gradient)
        actor_loss = -log_prob * delta.detach()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        return critic_loss.item(), actor_loss.item()

    def get_dopamine_signal(self) -> float:
        """Retourne le dernier δ (signal dopaminergique)."""
        return self.td_errors[-1] if self.td_errors else 0.0
