import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

env=gym.make("CartPole-v0")

data_transformer=PolynomialFeatures(3)

episodes=120
max_steps=1000
env._max_episode_steps=max_steps

model=LogisticRegression()

x=[]
y=[]

valx=[]
valy=[]



for episode in range(1,200+1):
    score=0
    previous_observation=env.reset()
    for _ in range(max_steps):
        action=env.action_space.sample()
        current_observation, reward, done, info = env.step(action)

        temp=data_transformer.fit_transform([previous_observation])
        x.append(temp[0])
        y.append(action)
        
        previous_observation=current_observation
        score+=reward
        if(done):
            break
    if(score>=20):
        for data in x:
            valx.append(data)
        for data in y:
            valy.append(data)
##    if score>=score_threshold:
model.fit(valx,valy)

for episode in range(1,10+1):
    score=0
    previous_observation=env.reset()
    for _ in range(max_steps):
        features=data_transformer.fit_transform([previous_observation])
        action=model.predict(features)[0]
        current_observation, reward, done, info = env.step(action)

        temp=data_transformer.fit_transform([previous_observation])
        x.append(temp[0])
        y.append(action)
        
        previous_observation=current_observation
        score+=reward

        if(done):
            break
##    if score>=score_threshold:
##    model.fit(x,y)
    print(episode,score,len(x))
