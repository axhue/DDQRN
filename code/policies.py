
        
class LinearDecayGreedyEpsilonPolicy:
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, start_value, end_value, num_steps):  # noqa: D102
        self.start_value = start_value
        self.decay_rate = float(end_value - start_value) / num_steps
        self.end_value = end_value
        self.step = 0
        self.epsilon = start_value

    def update(self,is_training = True):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        Any:
          Selected action.
        """
        epsilon = self.start_value
        epsilon += self.decay_rate * self.step
        self.step += 1
        self.epsilon = max(epsilon, self.end_value)

    def reset(self):
        """Start the decay over at the start value."""
        self.step = 0
