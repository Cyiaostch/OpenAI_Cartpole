import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

env=gym.make("CartPole-v0")

episodes=10000
max_steps=1000
score_threshold=100
env._max_episode_steps=max_steps
file_name="new_mined_data_1"

valid_scores=[]
training_data=[]

for episode in range(1,episodes+1):
    step_history=[]
    score=0
    previous_observation=env.reset()
    for _ in range(max_steps):
        action=env.action_space.sample()
        current_observation, reward, done, info = env.step(action)

        step_history.append([previous_observation,action])
        previous_observation=current_observation
        score+=reward

        if(done):
            break
    if score>=score_threshold:
        valid_scores.append(score)
        for step in step_history:
            if step[1]==1:
                training_data.append([step[0],1])
            else:
                training_data.append([step[0],0])
    if(episode%5000==0):
        print("Progress : {}%".format(round(episode*100/episodes,3)))

##sns.kdeplot(valid_scores)
##plt.title("Valid Scores Distribution")
##plt.show()
np.save(file_name,training_data)
print("-- Process Finished ")
print("-- Number of Episodes : {} ".format(episodes))
print("-- Average Score : {}".format(np.mean(score)))
print("-- Data Mined : {} record".format(len(training_data)))
