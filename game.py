import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import time

def transform_data(dat):
    func=lambda x : [
                             #1st Order
                             ##1 variable
                             x[0],x[1],x[2],x[3],
                             
                             #2nd Order
                             ##1 variable
                             x[0]**2,x[1]**2,x[2]**2,x[3]**2,
                             ##2 variable
                             x[0]*x[1],x[0]*x[2],x[0]*x[3],x[1]*x[2],x[1]*x[3],x[2]*x[3],
            
                             #3rd Order
                             ##1 variable
                             x[0]**3,x[1]**3,x[2]**3,x[3]**3,
                             ##2 variables
                             (x[0]**2)*x[1],(x[0]**2)*x[2],(x[0]**2)*x[3],
                             (x[1]**2)*x[0],(x[1]**2)*x[2],(x[1]**2)*x[3],
                             (x[2]**2)*x[0],(x[2]**2)*x[1],(x[2]**2)*x[3],
                             (x[3]**2)*x[0],(x[3]**2)*x[1],(x[3]**2)*x[2],
                             ##3 variables
                             x[0]*x[1]*x[2],x[0]*x[1]*x[3],x[0]*x[2]*x[3],x[1]*x[2]*x[3],
                             ]
    transformed_data = np.array(list(map(func,dat)))
    return transformed_data

env=gym.make("CartPole-v0")

episodes=100
max_steps=100000
env._max_episode_steps=max_steps
file_name="new_mined_data_1"

scores=[]
infinite_score_num=0

with open('logistic.pkl', 'rb') as f:
    classifier = pickle.load(f)   

for episode in range(1,episodes+1):
    step_history=[]
    score=0
    previous_observation=env.reset()
    for _ in range(max_steps):
        env.render()
##        time.sleep(0.02)
        transformed_observation=transform_data([previous_observation])
        action=classifier.predict(transformed_observation)[0]
        current_observation, reward, done, info = env.step(action)

        previous_observation=current_observation
        score+=reward

        if(done):
            break
    if(score==max_steps):
        infinite_score_num+=1
    scores.append(score)
    print("Episode {} || Score : {}".format(episode,score))
print("-- Process Finished ")
print("-- Number of Episodes : {} ".format(episodes))
print("-- Average Score : {}".format(np.mean(scores)))
print("-- Scores Standard Error : {}".format(np.std(scores)))
print("-- Infinite Score Percentage : {}".format(round(infinite_score_num*100/episodes),2))
