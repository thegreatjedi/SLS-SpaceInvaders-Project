import os

import numpy as np
import retro
import tensorflow as tf
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from util import get_valid_filename


def make_vec_env(env_id):
    # stable-baselines built-in make_vec_env converted to work with retro
    def make_env():
        def _init():
            return retro.make(env_id)
        
        return _init
    
    return DummyVecEnv([make_env()])


def main(game_name, num_timesteps, num_episodes, dir_name, model_name,
         policy, discount=0.8, batch_size=64, learn_rate=0.00025):
    dir_name = get_valid_filename(dir_name)
    model_name = get_valid_filename(model_name)
    
    eval_log_dir = f"logs/{dir_name}/{model_name}"
    tr_log_dir = f"{eval_log_dir}-training"
    model_dir = f"models/{dir_name}"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(tr_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = make_vec_env(game_name)
    env.seed(309)
    
    model = A2C(policy=policy, env=env, gamma=discount, n_steps=batch_size,
                learning_rate=learn_rate, verbose=1, seed=309,
                tensorboard_log=tr_log_dir, n_cpu_tf_sess=1)
    model.learn(total_timesteps=num_timesteps)
    model.save(f"{model_dir}/{model_name}")
    
    eps_done = 0
    ep_rewards = np.array([0] * num_episodes)
    curr_rewards = 0
    obs = env.reset()
    while eps_done != num_episodes:
        if eps_done % 10 == 0:
            print(f"Episodes completed: {eps_done} / {num_episodes}", end="\r")
        # For vectorised environments, they are automatically reset when done,
        # so returned obs would be the start state of next episode
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render(mode="human")
        curr_rewards += reward[0]
        if done[0]:
            ep_rewards[eps_done] = curr_rewards
            curr_rewards = 0
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
    print(f"Average score over {num_episodes} games: {avg_reward:.2f}")
    
    summary_writer = tf.summary.FileWriter(eval_log_dir)
    sess = tf.Session()
    rew_var = tf.Variable(0, dtype=tf.int64)
    rew_val = tf.summary.scalar(f"Reward / Episode ({model_name})", rew_var)
    for i in range(num_episodes):
        rew = ep_rewards[i]
        sess.run(rew_var.assign(rew))
        summary_writer.add_summary(sess.run(rew_val), i)
    
    avg_var = tf.Variable(0.0, dtype=tf.float64)
    avg_val = tf.summary.scalar(f"Trimmed Average ({model_name})", avg_var)
    sess.run(avg_var.assign(avg_reward))
    summary_writer.add_summary(sess.run(avg_val), 0)
    
    summary_writer.flush()
    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    game_name = "SpaceInvaders-Snes"
    
    policies = ["MlpPolicy"]
    for policy in policies:
        retry = True
        while retry:
            try:
                main(game_name=game_name, num_timesteps=100000,
                     num_episodes=100, dir_name="a2c-policies",
                     model_name=policy, policy=policy)
                retry = False
            except EOFError:
                pass
