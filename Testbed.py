#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Feb 22 10:12:32 2018

@author: Cameron Hargreaves
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt

def generateTestbed(numValues, centreMean):
    trueValues = []
    for i in range(numValues):
        trueValues.append(np.random.normal(centreMean,1))
    return trueValues

def pullArm(trueValue, action):
    return np.random.normal(trueValue[action], 1) # return a random number with mean of the true value

def runTestbed(numBandits, numIterations, learningRate, trueValues, currentEstimate):
    predictions = np.full([numIterations, numBandits], np.nan) # Initialise predictions to nan
    optimalAction = np.max(trueValues)
    optimalReward = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # ignore warning for mean of vector with NaNs in

        for i in range(numIterations):
            if np.random.rand(1) < learningRate:     # perform an exploratory action
                action = np.random.randint(numBandits)
            else:                               # Perform an exploitative action
                action = np.nanargmax(currentEstimate)       # Take the index of largest value

            predictions[i, action] = pullArm(trueValues, action)
            currentEstimate = np.nanmean(predictions, 0)     # Average down the matrix to get the average value for each bandit
            optimalReward.append(np.random.normal(optimalAction, 1))
    return predictions, optimalReward

def averageRewards(predictions):
    rewards = []
    rewardsAverage = np.zeros(predictions.shape[0])
    for i in range(predictions.shape[0]):
        rewards.append(np.nanmax(predictions[i])) # append the reward from each row
        rewardsAverage[i] = np.mean(rewards)      # have a running mean of the rewards
    return rewardsAverage

numBandits = 10
numIterations = 1000
learningRates = [0, 0.01, 0.1]

trueValue = generateTestbed(numBandits, 1)   # Each true value is in the normal distribution, mean 0 var 1
currentEstimate = [0] * numBandits     # Initialise estimates to 0

summedReward = 0

actualReward = []
optimalReward = []
estimate = []
rewardsRunningAverage = np.zeros([3, 1000])
rewardsRunningAverageAllRuns = []

numRuns = 10

for i in range(numRuns):
    for j, e in enumerate(learningRates):
        prediction, optimal = runTestbed(numBandits, numIterations, e, trueValue, currentEstimate)
        actualReward.append(prediction)
        optimalReward.append(optimal)
        estimate.append(np.nanmean(actualReward[j], 0))                 # Take the mean of each column to estimate the true value
        rewardsRunningAverage[j] = np.add(rewardsRunningAverage[j], averageRewards(actualReward[j]))   # Takes cumulative average at each step

for i in range(len(learningRates)):
    rewardsRunningAverage[i] = np.divide(rewardsRunningAverage[i], numRuns)

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(range(rewardsRunningAverage[0].shape[0]), rewardsRunningAverage[0], label = "Ɛ = 0")
plt.plot(range(rewardsRunningAverage[1].shape[0]), rewardsRunningAverage[1], label = "Ɛ = 0.01")
plt.plot(range(rewardsRunningAverage[2].shape[0]), rewardsRunningAverage[2], label = "Ɛ = 0.1")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(rewardsRunningAverage[0].shape[0]), (np.cumsum(rewardsRunningAverage[0]) / np.cumsum(optimalReward[0])) * 100, label = "Ɛ = 0")
plt.plot(range(rewardsRunningAverage[1].shape[0]), (np.cumsum(rewardsRunningAverage[1]) / np.cumsum(optimalReward[1])) * 100, label = "Ɛ = 0.01")
plt.plot(range(rewardsRunningAverage[2].shape[0]), (np.cumsum(rewardsRunningAverage[2]) / np.cumsum(optimalReward[2])) * 100, label = "Ɛ = 0.1")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()



plt.show()





















