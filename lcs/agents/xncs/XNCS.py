from typing import Optional
import random
import numpy as np
from copy import copy

from lcs.agents.xcs import XCS
from lcs.agents.Agent import TrialMetrics
from lcs.agents.xncs import Configuration, Backpropagation
# TODO: find a way to not require super in __init__
from lcs.agents.xncs import ClassifiersList, GeneticAlgorithm, Effect


class XNCS(XCS):

    def __init__(self,
                 cfg: Configuration,
                 population: Optional[ClassifiersList] = None
                 ) -> None:
        """
        :param cfg: object storing parameters of the experiment
        :param population: all classifiers at current time
        """

        if population is not None:
            self.population = population
        else:
            self.population = ClassifiersList(cfg=cfg)
        self.cfg = cfg
        self.ga = GeneticAlgorithm(
            population=self.population,
            cfg=self.cfg
        )
        self.back_propagation = Backpropagation(
            cfg=self.cfg,
            percentage=0.05
            )
        self.time_stamp = 0
        self.reward = 0

    def _form_sets_and_choose_action(self, state):
        match_set = self.population.generate_match_set(state, self.time_stamp)
        prediction_array = match_set.prediction_array
        action = self.select_action(prediction_array, match_set)
        action_set = match_set.generate_action_set(action)
        self.back_propagation.update_effect(action_set)
        return action_set, prediction_array, action

    def _distribute_and_update(self, action_set, current_situation, next_situation, p):
        if action_set is not None:
            self.back_propagation.update_cycle(
                action_set,
                Effect(next_situation)
            )
        super()._distribute_and_update(action_set, current_situation, next_situation, p)

