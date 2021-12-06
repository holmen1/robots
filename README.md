# robots

## Reinforcement-Learning
a computational approach to understanding and automating goal-directed learning and decision making
![error](https://github.com/holmen1/robots/blob/master/RL.JPG)



* maze

    Learning agents to navigate with Q-learning, Expected SARSA, Dyna-Q and Dyna-Q+

* mountain-car

    On-policy Control with function approximation.
    Weak car climbing hill using episodic semi-gradient one-step SARSA

* pendulum

    Average Reward Softmax Actor-Critic on a continuing task.

![error](https://github.com/holmen1/robots/blob/master/map.png)

**References**
Reinforcement learning: an introduction (second edition)
Richard S Sutton; Andrew G Barto
[Full pdf](http://incompleteideas.net/book/RLbook2020.pdf)


* policy evaluation (MDP)

    V(s) <-  &Sigma; &pi;(a|s) &Sigma; p(s',r|s,a) (r + &gamma; V(s'))

* value iteration (MDP)

    V(s) <- max<sub>a</sub> &Sigma; p(s',r|s,a) (r + &gamma; V(s'))

* policy iteration (MDP)

    V(s) <-  &Sigma; p(s',r|s,&pi;(s)) (r + &gamma; V(s'))

    &pi;(s) <- argmax<sub>a</sub> &Sigma; p(s',r|s,a) (r + &gamma; V(s'))

* SARSA agent

    Q(s,a) <- Q(s,a) + &alpha; (r + &gamma; Q(s',a') - Q(s,a))

* q-learning agent

    Q(s,a) <- Q(s,a) + &alpha; (r + &gamma; max[Q(s') - Q(s,a)]

* linear function approximation

    &theta; <- &theta; + &alpha; (r + &gamma; (max[Q(s') - Q(s,a)])) * f(s,a)

    Q(s,a) = &theta; f(s,a)