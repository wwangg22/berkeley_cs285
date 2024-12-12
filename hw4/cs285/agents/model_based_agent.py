from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.N = 0

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        e = 1e-6
        # obs = ptu.from_numpy(obs)
        # acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        obs_delta = next_obs - obs
        obs_delta_norm = (obs_delta - self.obs_delta_mean) /( self.obs_delta_std + e)

        obs_norm = ( obs - self.obs_acs_mean[:self.ob_dim].unsqueeze(dim=0) ) / (self.obs_acs_std[:self.ob_dim].unsqueeze(dim=0) + e)
        acs_norm = (acs - self.obs_acs_mean[self.ob_dim:].unsqueeze(dim=0)) / (self.obs_acs_std[self.ob_dim:].unsqueeze(dim=0) + e)
        
        # print('obs norm', obs_norm)
        obs_acs = torch.cat((obs_norm, acs_norm), dim=1)
        pred = self.dynamics_models[i](obs_acs)
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        # print("obs delta norm ", obs_delta_norm.shape)
        # print("pred shape ", pred.shape)
        loss = self.loss_fn(obs_delta_norm, pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('loss', loss)

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        delta_obs = next_obs - obs
        # self.N += 1
        # print("OBS", obs.shape)
        obs_acs = torch.cat((obs, acs), dim=1)
        # TODO(student): update the statistics

        self.obs_acs_mean = ((obs_acs)).mean(dim=0)
        self.obs_acs_std = (obs_acs).std(dim=0, unbiased=False)
        self.obs_delta_mean = (delta_obs).mean(dim=0)
        self.obs_delta_std = (delta_obs).std(dim=0, unbiased=False)

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        e = 1e-6
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
  
        obs_norm = ( obs - self.obs_acs_mean[:self.ob_dim].unsqueeze(dim=0) ) / (self.obs_acs_std[:self.ob_dim].unsqueeze(dim=0) + e)
        acs_norm = (acs - self.obs_acs_mean[self.ob_dim:].unsqueeze(dim=0)) / (self.obs_acs_std[self.ob_dim:].unsqueeze(dim=0) + e)
   
        # TODO(student): get the model's predicted `next_obs`
        obs_acs = torch.cat((obs_norm, acs_norm), dim = 1)
        # print("Obs acs requires grad ", obs_acs.requires_grad)

        pred = self.dynamics_models[i](obs_acs)


        pred_next_obs = pred * self.obs_delta_std + self.obs_delta_mean
        #         # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        # print("is this batch_size, ob_dim", pred_next_obs.shape)
        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        
        for index, acs in enumerate(action_sequences):
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )
            next_obs = np.zeros_like(obs)

            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            for i in range(self.ensemble_size):
                obs[i,:,:] = obs[i,:,:] + self.get_dynamics_predictions(i, obs[i, :, :], acs)
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.
            rewards = np.zeros_like(sum_of_rewards)
            for i in range(self.ensemble_size):
                rewards[i,:], dones = self.env.get_reward(obs[i,:,:], acs)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )
        # print('creating the sequence with shape ', self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim)
        action_sequences = action_sequences.transpose(1, 0, 2)

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            #og is action_sequences[best_index, :, :]
            return action_sequences[:, best_index, :]
        elif self.mpc_strategy == "cem":
            elite_mean = (self.env.action_space.high + self.env.action_space.low) / 2
            elite_std = (self.env.action_space.high - self.env.action_space.low)**2 / 12
            for i in range(self.cem_num_iters): 
                rewards = sorted(self.evaluate_action_sequences(obs, action_sequences))
                elites = rewards[-self.cem_num_elites:] 
                elite_mean = self.cem_alpha * np.mean(elites) + (1-self.cem_alpha)*elite_mean
                elite_std = self.cem_alpha * np.var(elites) + (1-self.cem_alpha) *elite_std
                samples = np.random.uniform(0, 1, size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim))
                cv = np.var(samples)
                sf = np.sqrt(elite_std / cv)
                samples = samples * sf

                cm = np.mean(samples)
                action_sequences = samples + (elite_mean - cm)

                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std

        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
