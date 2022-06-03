import logging
import random
import numpy as np


from lcs.agents import Agent
from lcs.agents.Agent import TrialMetrics
from lcs.agents.dqn import Configuration, Network
from .ReplayMemory import ReplayMemory
from .ReplayMemorySample import ReplayMemorySample

logger = logging.getLogger(__name__)


class DQN(Agent):

    def __init__(self,
                 cfg: Configuration,
                 network: Network = None) -> None:
        self.cfg = cfg
        self.network = network or self._initial_network()
        self.replay_memory = ReplayMemory(max_size=cfg.er_buffer_size)

    def get_population(self):
        return None

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        state = env.reset()
        state = [int(i) for i in state]
        action = env.action_space.sample()
        last_reward = 0
        done = False

        while not done:
            assert len(state) == self.cfg.classifier_length
            action = self.choose_action(state)
            logger.debug("\tExecuting action: [%d]", action)

            prev_state = state
            raw_state, last_reward, done, _ = env.step(action)
            raw_state = [int(i) for i in state]
            state = raw_state

            # Add new sample to the buffer, potenially remove if exceed max size
            new_sample = ReplayMemorySample(
                prev_state, action, last_reward, state, done)
            self.replay_memory.update(
                new_sample, self.cfg.er_weight_function(self.replay_memory, new_sample))

            if len(self.replay_memory) >= self.cfg.er_min_samples:

                # Rand samples indexes from the replay memory buffer
                mini_batch_size = self.cfg.er_samples_number
                samples = random.choices(
                    population=self.replay_memory, weights=self.replay_memory.weights, k=mini_batch_size)

                state_list = np.zeros(
                    (mini_batch_size, self.cfg.classifier_length))
                next_state_list = np.zeros(
                    (mini_batch_size, self.cfg.classifier_length))
                action_list, reward_list, done_list = [], [], []

                for i in range(mini_batch_size):
                    state_list[i] = samples[i].state
                    action_list.append(samples[i].action)
                    reward_list.append(samples[i].reward)
                    next_state_list[i] = samples[i].next_state
                    done_list.append(samples[i].done)

                # do batch prediction to save speed
                target = self.network.predict(state_list)
                target_next = self.network.predict(next_state_list)

                for i in range(mini_batch_size):
                    # correction on the Q value for the action used
                    if done_list[i]:
                        target[i][action_list[i]] = reward_list[i]
                    else:
                        # Standard - DQN
                        # DQN chooses the max Q value among next actions
                        # selection and evaluation of action is on the target Q Network
                        # Q_max = max_a' Q_target(s', a')
                        target[i][action_list[i]] = reward_list[i] + \
                            self.cfg.gamma * (np.amax(target_next[i]))

                # Train the Neural Network with batches
                self.network.fit(state_list, target, mini_batch_size)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        state = env.reset()
        state = [int(i) for i in state]
        action = int(env.action_space.sample())
        last_reward = 0
        done = False

        while not done:
            action = self.predict_action(state)
            raw_state, last_reward, done, _ = env.step(action)
            steps += 1

        return TrialMetrics(steps, last_reward)

    def _initial_network(self):
        return Network(self.cfg.classifier_length, self.cfg.number_of_possible_actions)

    def choose_action(self, state):
        if np.random.random() <= self.cfg.epsilon:
            return random.randrange(self.cfg.number_of_possible_actions)

        return self.predict_action(state)

    def predict_action(self, state):
        if not self.network.is_fitted():
            return random.randrange(self.cfg.number_of_possible_actions)

        return np.argmax(self.network.predict(np.reshape(state, [1, self.cfg.classifier_length])))
