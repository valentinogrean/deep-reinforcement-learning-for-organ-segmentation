import numpy as np
import tensorflow as tf
import datetime
from agentSEGMPPO import Agent
from envPPODUAL import CTEnv

if __name__ == '__main__':
    
    env = CTEnv(render=False, final_render = False)
    N = 1024
    batch_size = 64
    n_epochs = 10
    learning_rate = 0.0001
    agent = Agent(n_actions=2, batch_size=batch_size,
                  learning_rate=learning_rate, n_epochs=n_epochs,
                  input_dims=(256, 256, 1))

    total_steps = 0
    traj_length = 0

    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    best_episode_reward = -10000
    best_running_reward = -10000

    savedForThisLearnStep = False

    #logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './logs_ddpg_segmentation/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    #agent.load_models()

    
    while True:
        observation, slice, x, max_x = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, prob = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            total_steps += 1
            traj_length += 1
            episode_reward += reward
            agent.remember(observation, observation_, action,
                           prob, reward, done)
            if traj_length % N == 0:
                agent.learn()
                savedForThisLearnStep = False
                traj_length = 0
            observation = observation_

        # decrease the first 10 episodes for better average start
        if episode_count < 10:
            episode_reward -= 10
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        with summary_writer.as_default():
                tf.summary.scalar("Episode reward", episode_reward, step=episode_count)
                tf.summary.scalar("Running reward - mean of last 100 episodes", running_reward, step=episode_count)
                summary_writer.flush()

        episode_count += 1

        if(episode_reward > best_episode_reward):
            best_episode_reward = episode_reward

        if(running_reward > best_running_reward):    
            best_running_reward = running_reward
            if savedForThisLearnStep == False:
                agent.save_models()
                savedForThisLearnStep = True         