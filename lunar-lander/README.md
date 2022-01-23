
# lunar-lander

This library contains modified lab files from the Reinforcement Learning Specialization offered by
University of Alberta, Alberta Machine Intelligence Institute
https://www.coursera.org/specializations/reinforcement-learning

Topics explored:

* OpenAI gym environment

* Adam algorithm for neural network optimization

* Experience replay buffers

* Softmax action-selection

* Expected Sarsa


LunarLander-v2

Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

![error](https://github.com/holmen1/robots/blob/master/lunar-lander/images/lander.jpg)

https://gym.openai.com/envs/LunarLander-v2/


Expected SARSA update algorithm
![error](https://github.com/holmen1/robots/blob/master/lunar-lander/images/E-SARSA.png)



To run Jupyter from venv

$ sudo ./venv/bin/python3 -m ipykernel install --name=venv


AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLander'

$ sudo apt-get install build-essential python-dev swig python-pygame

$ pip install Box2D

https://stackoverflow.com/questions/44198228/install-pybox2d-for-python-3-6-with-conda-4-3-21




