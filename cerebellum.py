"""
Cervelet — Forward Model (apprentissage supervisé)
Inspiré de Doya (2000)

Rôle dans l'architecture :
  Le cervelet reçoit (s, a) et prédit l'état suivant ŝ'.
  L'erreur de prédiction ||ŝ' - s'|| est le signal d'apprentissage.

  Dans le cerveau :
    - Entrée  : état s (cortex cérébral) + action a (cortex moteur)
    - Sortie  : prédiction ŝ' (état futur)
    - Erreur  : signal des cellules de Purkinje (fibres grimpantes)
    - Poids   : synapses parallèles → cellules de Purkinje (LTD/LTP)

Connexions avec les autres modules :
  - Reçoit s depuis le Cortex (LSTM)
  - Reçoit a depuis les Basal Ganglia (Actor)
  - Envoie ŝ' au Cortex pour améliorer la représentation d'état
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ─────────────────────────────────────────────
#  FORWARD MODEL — prédit l'état suivant
#  ŝ' = f(s, a)
# ─────────────────────────────────────────────

class ForwardModel(nn.Module):
    """
    Modèle forward du cervelet.
    Prédit l'état s' à partir de (s, a).

    Paramètres
    ----------
    state_dim  : dimension de l'état
    action_dim : dimension de l'action (1 si discret, encodé en one-hot)
    hidden_dim : taille des couches cachées
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Entrée : concat(s, a_onehot) → état suivant ŝ'
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)   # prédit Δs ou s' directement
        )

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Prédit l'état suivant ŝ'.

        Entrée  : state       [batch, state_dim]
                  action_onehot [batch, action_dim]
        Sortie  : ŝ'          [batch, state_dim]
        """
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
#  INVERSE MODEL — prédit l'action depuis (s, s')
#  â = g(s, s')  (optionnel, enrichit l'analyse)
# ─────────────────────────────────────────────

class InverseModel(nn.Module):
    """
    Modèle inverse : prédit l'action qui a causé la transition s → s'.
    Utile pour le contrôle et pour l'analyse des représentations.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # logits → softmax externe
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Prédit les logits de l'action depuis (s, s')."""
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
#  REPLAY BUFFER — nécessaire pour l'apprentissage supervisé
#  Le cervelet apprend sur des transitions passées (pas online)
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Stocke les transitions (s, a, s') pour entraîner le cervelet.
    Le cervelet apprend de l'expérience accumulée (apprentissage supervisé).
    """
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, next_state)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx   = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states      = torch.stack([b[0] for b in batch])
        actions     = torch.tensor([b[1] for b in batch], dtype=torch.long)
        next_states = torch.stack([b[2] for b in batch])
        return states, actions, next_states

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  CEREBELLUM — module complet
# ─────────────────────────────────────────────

class Cerebellum:
    """
    Module Cervelet complet.

    Apprentissage supervisé : minimise ||ŝ' - s'||²
    (erreur de prédiction = signal des fibres grimpantes)

    Paramètres
    ----------
    state_dim       : dimension de l'état
    action_dim      : nb d'actions discrètes
    hidden_dim      : taille des couches cachées
    lr              : taux d'apprentissage
    buffer_capacity : taille du replay buffer
    batch_size      : taille des mini-batchs pour l'entraînement
    train_every     : entraîne le cervelet tous les N pas
    """
    def __init__(
        self,
        state_dim:       int,
        action_dim:      int,
        hidden_dim:      int   = 64,
        lr:              float = 1e-3,
        buffer_capacity: int   = 10_000,
        batch_size:      int   = 64,
        train_every:     int   = 10,
    ):
        self.action_dim  = action_dim
        self.batch_size  = batch_size
        self.train_every = train_every
        self._step       = 0

        self.forward_model  = ForwardModel(state_dim, action_dim, hidden_dim)
        self.inverse_model  = InverseModel(state_dim, action_dim, hidden_dim)
        self.buffer         = ReplayBuffer(buffer_capacity)

        self.opt_forward = optim.Adam(self.forward_model.parameters(), lr=lr)
        self.opt_inverse = optim.Adam(self.inverse_model.parameters(), lr=lr)
        self.loss_fn     = nn.MSELoss()
        self.ce_loss     = nn.CrossEntropyLoss()

        # Historique pour le logging
        self.prediction_errors = []

    def _action_to_onehot(self, action: torch.Tensor) -> torch.Tensor:
        """Convertit une action discrète en vecteur one-hot."""
        onehot = torch.zeros(*action.shape, self.action_dim)
        onehot.scatter_(-1, action.unsqueeze(-1), 1.0)
        return onehot

    def observe(self, state: torch.Tensor, action: int, next_state: torch.Tensor):
        """
        Enregistre une transition dans le buffer.
        À appeler à chaque pas de temps depuis la boucle principale.
        """
        self.buffer.push(state.detach(), action, next_state.detach())
        self._step += 1

    def train_step(self) -> dict:
        """
        Effectue une mise à jour des modèles forward et inverse.
        Appelée automatiquement par observe() tous les train_every pas.

        Retourne un dict de métriques.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, next_states = self.buffer.sample(self.batch_size)
        action_onehot = self._action_to_onehot(actions)

        # ── Forward model loss ──────────────────────────────────────
        # Le cervelet prédit ŝ' et minimise l'erreur ||ŝ' - s'||²
        pred_next = self.forward_model(states, action_onehot)
        fwd_loss  = self.loss_fn(pred_next, next_states)

        self.opt_forward.zero_grad()
        fwd_loss.backward()
        self.opt_forward.step()

        # ── Inverse model loss ──────────────────────────────────────
        # Le modèle inverse prédit l'action depuis (s, s')
        pred_action_logits = self.inverse_model(states, next_states)
        inv_loss = self.ce_loss(pred_action_logits, actions)

        self.opt_inverse.zero_grad()
        inv_loss.backward()
        self.opt_inverse.step()

        self.prediction_errors.append(fwd_loss.item())

        return {
            "forward_loss":  fwd_loss.item(),
            "inverse_loss":  inv_loss.item(),
        }

    def predict_next_state(
        self, state: torch.Tensor, action: int
    ) -> torch.Tensor:
        """
        Prédit l'état suivant ŝ' depuis (s, a).
        Utilisé par le Cortex pour la planification.
        """
        action_t      = torch.tensor([action], dtype=torch.long)
        action_onehot = self._action_to_onehot(action_t)
        with torch.no_grad():
            return self.forward_model(state.unsqueeze(0), action_onehot).squeeze(0)

    def prediction_error(
        self, state: torch.Tensor, action: int, next_state: torch.Tensor
    ) -> float:
        """
        Calcule l'erreur de prédiction pour une transition donnée.
        Signal analogue aux fibres grimpantes (cellules de Purkinje).
        """
        pred = self.predict_next_state(state, action)
        return (pred - next_state).pow(2).mean().item()
