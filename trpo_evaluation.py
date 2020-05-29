import os

import numpy as np
import retro
import tensorflow as tf
from gym import wrappers
from stable_baselines import TRPO
from stable_baselines.common.vec_env import SubprocVecEnv

from util import get_valid_filename


def make_vec_envs(env_id, is_eval, num_envs, model_name=None):
    # stable-baselines built-in make_vec_env converted to work with retro
    assert not is_eval or model_name is not None
    
    def make_env():
        def _init():
            env = retro.make(env_id)
            if is_eval:
                # Wraps the env in a Monitor wrapper to record runs
                monitor_dir = f"videos/{model_name}"
                os.makedirs(monitor_dir, exist_ok=True)
                env = wrappers.Monitor(env, directory=monitor_dir, force=True,
                                       mode="evaluation")
            return env
        
        return _init
    
    return SubprocVecEnv([make_env() for _ in range(num_envs)])


def train(game, num_timesteps, num_envs, dir_name, model_name,
          prev_model_name):
    dir_name = get_valid_filename(dir_name)
    model_name = get_valid_filename(model_name)
    
    log_dir = f"logs/{dir_name}/{model_name}-training"
    model_dir = f"models/{dir_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = make_vec_envs(game, False, num_envs)
    prev_model_path = f"{model_dir}/{prev_model_name}.zip"
    if prev_model_name is not None and os.path.exists(prev_model_path):
        model = TRPO.load(prev_model_path, env=env)
        model.tensorboard_log = log_dir
    else:
        model = TRPO(policy="MlpPolicy", env=env, gamma=0.8, verbose=1,
                     tensorboard_log=log_dir)
    model.learn(num_timesteps)
    model.save(f"{model_dir}/{model_name}.zip")
    env.close()


def evaluate(game, num_eps, num_envs, dir_name, model_name):
    dir_name = get_valid_filename(dir_name)
    model_name = get_valid_filename(model_name)
    
    log_dir = f"logs/{dir_name}/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_vec_envs(game, True, num_envs, model_name=model_name)
    model_path = f"models/{dir_name}/{model_name}.zip"
    model = TRPO.load(model_path, env=env)
    model.tensorboard_log = log_dir
    
    eps_done = 0
    ep_rewards = np.array([0] * num_eps)
    curr_rewards = [0] * num_envs
    obs = env.reset()
    while eps_done != num_eps:
        # For vectorised environments, they are automatically reset when done,
        # so returned obs would be the start state of next episode
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render(mode="human")
        
        for i in range(num_envs):
            curr_rewards[i] += reward[i]
            if done[i]:
                ep_rewards[eps_done] = curr_rewards[i]
                curr_rewards[i] = 0
                eps_done += 1
    print("All episodes completed")
    env.close()
    
    mean = ep_rewards.mean()
    std_dev = ep_rewards.std()
    # Outliers: outside of 3 standard deviations
    outlier_threshold_upper = mean + 3 * std_dev
    outlier_threshold_lower = mean - 3 * std_dev
    trimmed_rewards = np.array([
        rew for rew in ep_rewards
        if outlier_threshold_lower <= rew <= outlier_threshold_upper
    ])
    avg_reward = trimmed_rewards.mean()
    best_reward = ep_rewards.max()
    print(f"Average score over {num_eps} games: {avg_reward:.2f}")
    print(f"Best score: {best_reward}")
    
    summary_writer = tf.summary.FileWriter(log_dir)
    sess = tf.Session()
    rew_var = tf.Variable(0, dtype=tf.int64)
    rew_val = tf.summary.scalar(f"Reward / Episode ({model_name})", rew_var)
    for i in range(num_eps):
        rew = ep_rewards[i]
        sess.run(rew_var.assign(rew))
        summary_writer.add_summary(sess.run(rew_val), i)
    
    best_val = tf.summary.scalar(f"Best Reward", rew_var)
    sess.run(rew_var.assign(best_reward))
    summary_writer.add_summary(sess.run(best_val), 0)
    
    avg_var = tf.Variable(0.0, dtype=tf.float64)
    avg_val = tf.summary.scalar(f"Trimmed Average ({model_name})", avg_var)
    sess.run(avg_var.assign(avg_reward))
    summary_writer.add_summary(sess.run(avg_val), 0)
    
    summary_writer.flush()
    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    game_name = "SpaceInvaders-Snes"
    # for i in range(5):
    #     curr_model_iter = (i + 1) * 1000000
    #     retry = True
    #     while retry:
    #         try:
    #             train(game=game_name, num_timesteps=1000000, num_envs=1,
    #                   dir_name="trpo", model_name=f"{curr_model_iter}",
    #                   prev_model_name=None if i == 0 else str((i * 1000000)))
    #             retry = False
    #         except EOFError:
    #             pass
    
    for i in range(1):
        curr_model_iter = (i + 1) * 1000000
        retry = True
        while retry:
            try:
                evaluate(game=game_name, num_eps=100, num_envs=1,
                         dir_name="trpo", model_name=f"{curr_model_iter}")
                retry = False
            except EOFError:
                pass
