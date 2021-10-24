# cliffworld

This library contains modified lab files from the course Sample-based Learning Methods offered by
University of Alberta, Alberta Machine Intelligence Institute
https://www.coursera.org/learn/sample-based-learning-methods

![error](https://github.com/holmen1/robots/blob/master/cliffworld/cliffworld.pnp)

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

