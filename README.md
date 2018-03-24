# DDQRN
A Dueling Double Deep Q Recurrent Network implemented to play video games!


## Source papers
 * [Human-level control through deep reinforcement learning ](https://www.nature.com/articles/nature14236)
 * [Dueling Network Architectures for Deep Reinforcement Learning ](https://arxiv.org/abs/1511.06581)
 * [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
 * [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)


## Results

![Playing Seaquest](https://github.com/axhue/minecraftRL/blob/master/results/Seaquest-R20-DDQRN.gif) ![Playing Qbert](https://github.com/axhue/minecraftRL/blob/master/results/QbertR20-results.gif)

And the rewards per episode (runtime was 2 days on P100)

![Rewards Graph](https://github.com/axhue/minecraftRL/blob/master/results/Seaquest-rewards-DDRQN20.png) ![Rewards Graph](https://github.com/axhue/minecraftRL/blob/master/results/Qbert-rewards-DDRQN20.png)

And the rewards per episode (runtime was 2 days on P100)



## How to run the code
First you need to create the conda environment `conda env create -f environment.yml` and activate

then you may adjust the parameters setup in execute.py

finally `python execute.py`

the code's default directory for tensorboard data is ./logs, to read these logs you need to run `tensorboard --logdir=./logs`
## Next up!
 * Attention Mechanisms
 * Hindsight Experience Replay
 * A2C methods
 * Metaheuristics

## Interesting Notes
 * I previously thought that pseudo huber loss was identical to huber loss, this [article](https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/) was very useful in helping me make the model work better
