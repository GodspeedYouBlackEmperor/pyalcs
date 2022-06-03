from typing import Callable, Optional


class Configuration(object):

    def __init__(self, **kwargs) -> None:
        """
        Creates the configuration object used during training the DQN agent.
        """
        # number of possible actions to be executed
        self.number_of_possible_actions: int = kwargs[
            'number_of_possible_actions']

        # ER replay memory buffer size
        self.er_buffer_size = kwargs.get('er_buffer_size', 10000)

        # ER replay memory min samples
        self.er_min_samples = kwargs.get('er_min_samples', 1000)

        # ER replay memory samples number
        self.er_samples_number = kwargs.get('er_samples_number', 3)

        # ER weight_function
        self.er_weight_function = kwargs.get(
            'er_weight_function', lambda rm, sample: 1)

        # length of the condition and effect strings
        self.classifier_length: int = kwargs['classifier_length']

        # how often metric are collected
        self.metrics_trial_frequency: int = kwargs.get(
            'metrics_trial_frequency', 1)

        # custom function for collecting customized metrics
        self.user_metrics_collector_fcn: Optional[Callable] = kwargs.get(
            'user_metrics_collector_fcn', None)

        # custom fitness function
        self.fitness_fcn: Optional[Callable] = kwargs.get('fitness_fcn', None)

        # learning rate
        self.beta: float = kwargs.get('beta', 0.05)

        self.gamma: float = kwargs.get('gamma', 0.95)

        # inadequacy threshold
        self.theta_i: float = kwargs.get('theta_i', 0.1)

        # reliability threshold - quality level when the classifier
        # is treated as "reliable"
        self.theta_r: float = kwargs.get('theta_r', 0.9)

        # Probability of executing random action.
        # Otherwise, the action from the best classifier is selected.
        self.epsilon: float = kwargs.get('epsilon', 0.5)

        self.u_max: int = kwargs.get('u_max', 100000)
        self.theta_exp: int = kwargs.get('theta_exp', 20)
        self.theta_as: int = kwargs.get('theta_as', 20)

        # whether to use mlflow
        self.use_mlflow: bool = kwargs.get('use_mlflow', False)

        # how often dump model object with mlflow
        self.model_checkpoint_freq: Optional[int] = kwargs.get(
            'model_checkpoint_frequency', None)

    def __str__(self) -> str:
        return str(vars(self))
