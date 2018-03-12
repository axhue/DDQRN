from agent import Agent,AgentConfig
import gym
from preprocessors import AtariPreprocessor
from policies import LinearDecayGreedyEpsilonPolicy
import os


def _test(env,episodes):
    env.reset()
    for i in range(episodes):
        state = env.reset()
        for t in itertools.count():
            env.render()
            s,r,done, info = env.step(env.action_space.sample()) # take a random action

            if done:
                break
#_test(atari_env,10)


def get_output_folder(args, parent_dir, env_name, task_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id) + '-' + task_name
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)
    else:
        print('===== Folder exists; delete? %s'%parent_dir)
        input("Press Enter to continue...")
        os.system('rm -rf %s/' % (parent_dir))
    #s.makedirs(parent_dir+'/videos/')
    #os.makedirs(parent_dir+'/images/')
    return parent_dir



if __name__== '__main__':
    atari_env = gym.make('Seaquest-v0')
    doc = get_output_folder(None,"./logs","Seaquest-v0","DQRN-10")

    episodes = 1000000
    cfg = AgentConfig()
    cfg.stateCnt = atari_env.observation_space.shape
    cfg.actionCnt = atari_env.action_space.n
    cfg.mem_size = 1000000
    cfg.epsilon_policy = LinearDecayGreedyEpsilonPolicy(1.0,0.05,episodes)
    cfg.gamma = 0.99
    cfg.num_frames = 10
    cfg.learning_rate = 0.0001
    cfg.train_start = 50
    cfg.train_freq = 4
    cfg.target_update_freq = 10000
    cfg.batch_size = 32
    cfg.preprocessor = AtariPreprocessor((84,84,1))
    cfg.log_path = doc
    cfg.name = "R7"
    agent = Agent(cfg)
    agent.train(atari_env,episodes,render=False)