import logging
import random
from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.agents.racser.ReplayMemory import ReplayMemory
from lcs.agents.racser.ReplayMemorySample import ReplayMemorySample
from lcs.strategies.action_selection.BestAction import BestAction
from lcs.agents.racs import ClassifierList
from lcs.agents.racser import Configuration
from lcs.agents.Agent import Agent
from lcs.agents.racs.action_selection import choose_action

logger = logging.getLogger(__name__)


class RACSER(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifierList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifierList()
        self.replay_memory = ReplayMemory(max_size=cfg.er_buffer_size)

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        state = env.reset()
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        done = False

        while not done:
            state = state
            assert len(state) == self.cfg.classifier_length

            match_set = self.population.form_match_set(state)
            # action = self.cfg.action_selector(match_set)
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon,
                self.cfg.biased_exploration
            )
            logger.debug("\tExecuting action: [%d]", action)

            prev_state = state
            raw_state, last_reward, done, _ = env.step(action)
            state = raw_state

            # Add new sample to the buffer, potenially remove if exceed max size
            self.replay_memory.update(ReplayMemorySample(
                prev_state, action, last_reward, state, done))

            if len(self.replay_memory) >= self.cfg.er_min_samples:

                # Rand samples indexes from the replay memory buffer
                samples = random.sample(
                    range(0, len(self.replay_memory)), self.cfg.er_samples_number)
                for sample_index in samples:
                    sample: ReplayMemorySample = self.replay_memory[sample_index]
                    er_match_set = self.population.form_match_set(
                        sample.state)
                    er_action_set = er_match_set.form_action_set(
                        sample.action)
                    er_next_match_set = self.population.form_match_set(
                        sample.next_state)
                    # Apply learning in the replied action set
                    ClassifierList.apply_alp(
                        self.population,
                        er_next_match_set,
                        er_action_set,
                        sample.state,
                        sample.action,
                        sample.next_state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                    ClassifierList.apply_reinforcement_learning(
                        er_action_set,
                        sample.reward,
                        0 if sample.done else er_next_match_set.get_maximum_fitness(),
                        self.cfg.beta,
                        self.cfg.gamma
                    )
                    if self.cfg.do_ga:
                        ClassifierList.apply_ga(
                            time + steps,
                            self.population,
                            ClassifierList() if sample.done else er_next_match_set,
                            er_action_set,
                            sample.next_state,
                            self.cfg.theta_ga,
                            self.cfg.mu,
                            self.cfg.chi,
                            self.cfg.theta_as,
                            self.cfg.do_subsumption,
                            self.cfg.theta_exp)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        state = env.reset()

        last_reward = 0
        action_set = ClassifierList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifierList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0,
                biased_exploration_prob=0.0)
            action_set = match_set.form_action_set(action)

            state, last_reward, done, _ = env.step(action)

            if done:
                ClassifierList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)
