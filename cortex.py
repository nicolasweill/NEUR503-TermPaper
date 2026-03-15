"""
Cortex — Encodeur d'état récurrent (apprentissage non-supervisé)
Inspiré de Doya (2000)

Rôle dans l'architecture :
  Le cortex maintient une représentation interne de l'état du monde
  en intégrant l'historique des observations. Il fournit un contexte
  enrichi aux Basal Ganglia et au Cervelet.

  Dans le cerveau :
    - Cortex préfrontal     : mémoire de travail, contexte temporel
    - Cortex pariétal       : intégration sensorielle
    - Apprentissage         : non-supervisé (prédiction d'observation)

  Implémentation :
    - LSTM          : capture les dépendances temporelles
    - learn()       : entraînement sur transition (s_t, s_{t+1}) complète
    - encode()      : pur forward pass, sans gradient
    - Sortie h_t    → remplace s_t brut pour les BG

Connexions :
  - Reçoit s_t brut de l'environnement
  - Envoie h_t aux Basal Ganglia et au Cervelet
  - Reçoit optionnellement ŝ'_t du Cervelet
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────
#  CORTEX LSTM — encodeur de contexte temporel
# ─────────────────────────────────────────────

class CortexLSTM(nn.Module):
    """
    Encodeur cortical récurrent.

    Paramètres
    ----------
    obs_dim       : dimension de l'observation brute
    context_dim   : dimension du vecteur de contexte h_t (sortie)
    hidden_dim    : taille des cellules LSTM
    n_layers      : nombre de couches LSTM
    use_cerebellum_pred : si True, concatène ŝ'_{t-1} à l'entrée
    """
    def __init__(
        self,
        obs_dim:               int,
        context_dim:           int  = 64,
        hidden_dim:            int  = 128,
        n_layers:              int  = 1,
        use_cerebellum_pred:   bool = False,
    ):
        super().__init__()
        self.obs_dim             = obs_dim
        self.context_dim         = context_dim
        self.hidden_dim          = hidden_dim
        self.n_layers            = n_layers
        self.use_cerebellum_pred = use_cerebellum_pred

        lstm_input_dim = obs_dim * 2 if use_cerebellum_pred else obs_dim

        self.encoder = nn.Sequential(
            nn.Linear(lstm_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
        )

        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.Tanh(),
        )

        # Tête de prédiction : prédit s_{t+1} depuis h_t
        self.prediction_head = nn.Linear(context_dim, obs_dim)

    def forward(
        self,
        obs:             torch.Tensor,
        hidden_state:    tuple = None,
        cerebellum_pred: torch.Tensor = None,
    ):
        """
        Entrées
        -------
        obs             : [batch, obs_dim]
        hidden_state    : (h, c) état caché LSTM
        cerebellum_pred : [batch, obs_dim] (optionnel)

        Sorties
        -------
        context      : [batch, context_dim]
        hidden_state : (h, c)
        obs_pred     : [batch, obs_dim]  prédiction de s_{t+1}
        """
        if self.use_cerebellum_pred:
            if cerebellum_pred is not None:
                x = torch.cat([obs, cerebellum_pred], dim=-1)
            else:
                x = torch.cat([obs, torch.zeros_like(obs)], dim=-1)
        else:
            x = obs

        x = self.encoder(x)
        x = x.unsqueeze(1)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        context  = self.context_head(lstm_out)
        obs_pred = self.prediction_head(context)

        return context, hidden_state, obs_pred

    def init_hidden(self, batch_size: int = 1) -> tuple:
        """Initialise l'état caché à zéro."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return (h, c)


# ─────────────────────────────────────────────
#  CORTEX — module complet
# ─────────────────────────────────────────────

class Cortex:
    """
    Module Cortex complet.

    Deux méthodes bien séparées :
      encode()  — forward pur, sans gradient, met à jour l'état caché
      learn()   — entraînement sur transition complète (s_t, s_{t+1})

    Cette séparation évite de transporter des grad_fn entre les pas
    de temps (cause du RuntimeError précédent).
    """
    def __init__(
        self,
        obs_dim:             int,
        context_dim:         int   = 64,
        hidden_dim:          int   = 128,
        lr:                  float = 3e-4,
        use_cerebellum_pred: bool  = False,
    ):
        self.context_dim = context_dim
        self.model = CortexLSTM(
            obs_dim             = obs_dim,
            context_dim         = context_dim,
            hidden_dim          = hidden_dim,
            use_cerebellum_pred = use_cerebellum_pred,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        self._hidden   = self.model.init_hidden(batch_size=1)

        # Historique
        self.prediction_errors = []

    def reset(self):
        """Réinitialise l'état caché en début d'épisode."""
        self._hidden = self.model.init_hidden(batch_size=1)

    def encode(
        self,
        obs:             torch.Tensor,
        cerebellum_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode s_t → h_t. Pur forward pass sans mise à jour de poids.

        Retourne
        --------
        context : [context_dim]
        """
        obs_batch = obs.unsqueeze(0)
        cp_batch  = cerebellum_pred.unsqueeze(0) if cerebellum_pred is not None else None

        with torch.no_grad():
            context, new_hidden, _ = self.model(obs_batch, self._hidden, cp_batch)

        self._hidden = tuple(h.detach() for h in new_hidden)
        return context.squeeze(0)

    def learn(
        self,
        obs:             torch.Tensor,
        next_obs:        torch.Tensor,
        cerebellum_pred: torch.Tensor = None,
    ) -> float:
        """
        Entraîne le cortex sur une transition complète (s_t → s_{t+1}).

        On refait un forward pass frais avec gradient activé.
        L'état caché est détaché pour tronquer le BPTT —
        pas de transport de grad_fn entre les pas de temps.

        Retourne l'erreur de prédiction (scalaire).
        """
        obs_batch  = obs.unsqueeze(0)
        next_batch = next_obs.unsqueeze(0).detach()
        cp_batch   = cerebellum_pred.unsqueeze(0) if cerebellum_pred is not None else None

        # État caché détaché : graphe propre, pas d'accumulation infinie
        hidden_detached = tuple(h.detach() for h in self._hidden)

        _, _, obs_pred = self.model(obs_batch, hidden_detached, cp_batch)

        loss = self.loss_fn(obs_pred, next_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.prediction_errors.append(loss.item())
        return loss.item()

    def peek(
        self,
        obs:             torch.Tensor,
        cerebellum_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode s_t → h_t SANS mettre à jour l'état caché interne.
        Utilisé pour calculer h_{t+1} lors du TD-error des BG,
        sans avancer le LSTM d'un pas supplémentaire.
        """
        obs_batch = obs.unsqueeze(0)
        cp_batch  = cerebellum_pred.unsqueeze(0) if cerebellum_pred is not None else None
        with torch.no_grad():
            context, _, _ = self.model(obs_batch, self._hidden, cp_batch)
        return context.squeeze(0)