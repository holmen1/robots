import numpy as np
from source.agent import BaseAgent
from source.tile_coder import PendulumTileCoder


class ActorCriticSoftmaxAgent(BaseAgent):
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size") / num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size.
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """

        # compute softmax probability
        softmax_prob = self.compute_softmax_prob(self.actor_w, active_tiles)

        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        angle, ang_vel = state
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(active_tiles)

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        angle, ang_vel = state
        active_tiles = self.tc.get_tiles(angle, ang_vel)

        v = self.critic_w[active_tiles].sum()
        v_prev = self.critic_w[self.prev_tiles].sum()
        delta = reward - self.avg_reward + v - v_prev

        self.avg_reward += self.avg_reward_step_size * delta

        self.critic_w[self.prev_tiles] += self.critic_step_size * delta  # * 1 #active_tiles

        # update actor weights
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        current_action = self.agent_policy(active_tiles)

        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action

    def compute_softmax_prob(self, actor_w, tiles):
        """
        Computes softmax probability for all actions

        Args:
        actor_w - np.array, an array of actor weights
        tiles - np.array, an array of active tiles

        Returns:
        softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
        """

        # First compute the list of state-action preferences
        state_action_preferences = []
        actions = list(range(3))
        state_action_preferences = [actor_w[a][tiles].sum() for a in actions]

        # Set the constant c by finding the maximum of state-action preferences
        c = np.max(state_action_preferences)

        numerator = [np.exp(state_action_preference - c) for state_action_preference in state_action_preferences]
        denominator = np.sum(numerator)

        softmax_prob = [num / denominator for num in numerator]

        return softmax_prob

    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward
