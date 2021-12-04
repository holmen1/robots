# pendulum

This library contains modified lab files from the course Sample-based Learning Methods offered by
University of Alberta, Alberta Machine Intelligence Institute
https://www.coursera.org/learn/sample-based-learning-methods

![error](https://github.com/holmen1/robots/blob/master/pendulum/images/actor-critic.png)

Modifying for continuous task using average reward:

$\delta_t = R_{t+1} - \bar{R} + \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_{t}, \mathbf{w}) \hspace{6em}$

To run Jupyter from venv

$ sudo ./venv/bin/python3 -m ipykernel install --name=venv


## pendulum.ipynb


