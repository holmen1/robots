#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""
from __future__ import print_function
#from abc import ABCMeta, abstractmethod
from source.agent import BaseAgent
import numpy as np


class ExpectedSarsaAgent(BaseAgent):
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions))  # The array of action-value estimates.

    def agent_start(self, observation):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        # Perform an update
        pi = np.ones(self.num_actions) * self.epsilon / self.num_actions
        pi[action] += 1 - self.epsilon
        q_exp = np.dot(pi, current_q)
        target = reward + self.discount * q_exp
        error = target - self.q[self.prev_state, self.prev_action]
        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * error

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode
        target = reward
        error = target - self.q[self.prev_state, self.prev_action]
        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * error
        return reward

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)


class DynaQAgent(BaseAgent):

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """

        # First, we get the relevant information from agent_info
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {} # model is a dictionary of dictionaries, which maps states to actions to
                        # (reward, next_state) tuples

    def update_model(self, past_state, past_action, state, reward):
        """updates the model

        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """

        # EAFP
        try:
            self.model[past_state][past_action] = (state, reward)
        except:
            self.model[past_state] = {past_action: (state, reward)}

        """
        if past_state not in self.model.keys():
            self.model[past_state] = {past_action: (state, reward)}
        else:
            self.model[past_state][past_action] = (state, reward)
        """

    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Args:
            None
        Returns:
            Nothing
        """

        # The indirect RL step:
        # - Choose a state and action from the set of experiences that are stored in the model. (~2 lines)
        # - Query the model with this state-action pair for the predicted next state and reward.(~1 line)
        # - Update the action values with this simulated experience.                            (2~4 lines)
        # - Repeat for the required number of planning steps.

        for n in range(self.planning_steps):
            # random-sample one-step tabular Q-planning method
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))

            s_next, r = self.model[s][a]
            if s_next == -1:  # terminate
                target = r
                error = target - self.q_values[s, a]
                self.q_values[s, a] = self.q_values[s, a] + self.step_size * error
            else:
                q_max = max(self.q_values[s_next])
                target = r + self.gamma * q_max
                error = target - self.q_values[s, a]
                self.q_values[s, a] = self.q_values[s, a] + self.step_size * error

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        action = self.choose_action_egreedy(state)

        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_step(self, reward, state):
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        # - Direct-RL step (~1-3 lines)
        current_q = self.q_values[state]
        q_max = max(current_q)
        target = reward + self.gamma * q_max
        error = target - self.q_values[self.past_state, self.past_action]
        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * error

        # - Model Update step (~1 line)
        self.update_model(self.past_state, self.past_action, state, reward)

        # - `planning_step` (~1 line)
        self.planning_step()

        # - Action Selection step (~1 line)
        action = self.choose_action_egreedy(state)

        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_end(self, reward):
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # - Direct RL update with this final transition (1~2 lines)
        target = reward
        error = target - self.q_values[self.past_state, self.past_action]
        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * error

        # - Model Update step with this final transition (~1 line)
        self.update_model(self.past_state, self.past_action, -1, reward)

        # - One final `planning_step` (~1 line)
        self.planning_step()
        # ----------------

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)


class DynaQPlusAgent(BaseAgent):

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction
                kappa (float): The scaling factor for the reward bonus

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)
        self.kappa = agent_info.get("kappa", 0.001)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, tau, etc.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.tau = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}

    def update_model(self, past_state, past_action, state, reward):
        """updates the model

        Args:
            past_state  (int): s
            past_action (int): a
            state       (int): s'
            reward      (int): r
        Returns:
            Nothing
        """

        # Recall that when adding a state-action to the model, if the agent is visiting the state
        #    for the first time, then the remaining actions need to be added to the model as well
        #    with zero reward and a transition into itself.

        if past_state not in self.model:
            self.model[past_state] = {past_action: (state, reward)}
            for a in self.actions:
                if a != past_action:
                    self.model[past_state][a] = (past_state, 0)
        else:
            self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Args:
            None
        Returns:
            Nothing
        """

        # The indirect RL step:
        for n in range(self.planning_steps):
            # random-sample one-step tabular Q-planning method
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))

            # - Query the model with this state-action pair for the predicted next state and reward.(~1 line)
            s_next, r = self.model[s][a]

            # - **Add the bonus to the reward** (~1 line)
            r += self.kappa * np.sqrt(self.tau[s, a])

            if s_next == -1:  # terminate
                target = r
                error = target - self.q_values[s, a]
                self.q_values[s, a] = self.q_values[s, a] + self.step_size * error
            else:
                q_max = max(self.q_values[s_next])
                target = r + self.gamma * q_max
                error = target - self.q_values[s, a]
                self.q_values[s, a] = self.q_values[s, a] + self.step_size * error

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) The first action the agent takes.
        """
        action = self.choose_action_egreedy(state)

        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent is taking.
        """

        # Update the last-visited counts (~2 lines)
        self.tau += 1
        self.tau[self.past_state, self.past_action] = 0

        # - Direct-RL step (~1-3 lines)
        current_q = self.q_values[state]
        q_max = max(current_q)
        target = reward + self.gamma * q_max
        error = target - self.q_values[self.past_state, self.past_action]
        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * error

        # - Model Update step (~1 line)
        self.update_model(self.past_state, self.past_action, state, reward)

        # - `planning_step` (~1 line)
        self.planning_step()

        # - Action Selection step (~1 line)
        action = self.choose_action_egreedy(state)

        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_end(self, reward):
        """Called when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # - Direct RL update with this final transition
        self.tau += 1
        self.tau[self.past_state, self.past_action] = 0

        target = reward + self.kappa * np.sqrt(self.tau[self.past_state, self.past_action])
        error = target - self.q_values[self.past_state, self.past_action]
        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * error

        # - Model Update step with this final transition (~1 line)
        self.update_model(self.past_state, self.past_action, -1, reward)

        # - One final `planning_step` (~1 line)
        self.planning_step()

