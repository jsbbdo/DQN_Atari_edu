import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import TransformReward
from scipy.signal import convolve, gaussian
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from base64 import b64encode
from IPython.display import HTML

def make_env(env_name, clip_rewards=True):
    env = gym.make(env_name,
                   render_mode='rgb_array',
                   frameskip=1
              )
    env = AtariPreprocessing(env, screen_size=84, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
    return env

def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step

def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for i in range(n_games):
        s, _ = env.reset(seed=i)
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, terminated, truncated, _ = env.step(action)
            reward += r
            if terminated:
                break

        rewards.append(reward)
    return np.mean(rewards)

def play_and_record(start_state, agent, env, exp_replay, n_steps=1):

    s = start_state
    sum_rewards = 0
    # Play the game for n_steps and record transitions in buffer
    for i in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        next_s, r, terminated, truncated, _ = env.step(a)
        sum_rewards += r
        done = terminated or truncated
        exp_replay.add(s, a, r, next_s, done)
        if terminated:
            s, _ = env.reset(seed=i)
        else:
            s = next_s

    return sum_rewards, s

def record_video(env_id, make_env, video_folder, video_length, agent):

    vec_env = DummyVecEnv([lambda: make_env(env_id, clip_rewards=False)])
    env_id_suffix = env_id.split("/")[-1]
    # Record the video starting at the first step
    vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"{type(agent).__name__}-{env_id_suffix}")

    obs = vec_env.reset()
    print(obs.shape)
    for _ in range(video_length + 1):
        qvalues = agent.get_qvalues(obs)
        action = qvalues.argmax(axis=-1)
        obs, _, dones, _ = vec_env.step(action)
    # video filename
    file_path = "./"+video_folder+vec_env.video_recorder.path.split("/")[-1]
    # Save the video
    vec_env.close()
    return file_path

def play_video(file_path):
    mp4 = open(file_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
        <video width=400 controls>
              <source src="%s" type="video/mp4">
        </video>
        """ % data_url)