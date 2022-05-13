import lcs.agents.acs2 as acs2
from lcs.agents.acs2er.ReplayMemory import ReplayMemory
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # ER replay memory buffer size
        self.er_buffer_size = kwargs.get('er_buffer_size', 10000)

        # ER replay memory min samples
        self.er_min_samples = kwargs.get('er_min_samples', 1000)

        # ER replay memory samples number
        self.er_samples_number = kwargs.get('er_samples_number', 3)

        # ER weight_function
        self.er_weight_function = kwargs.get(
            'er_weight_function', lambda rm, sample: 1)

    def __str__(self) -> str:
        return str(vars(self))
