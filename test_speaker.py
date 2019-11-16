#test_gender.py
import os
#import cPickle
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import scipy
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

enableFigure = True

#path to training data
source   = "./development_set/"

modelpath = "./speaker_models/"

test_file = "./development_set_test.txt"

file_paths = open(test_file,'r')


gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]

maxAudioLength = 0
nFiles = 0
for path in file_paths:
    path = path.strip()
    sr, audio = read(source + path)
    maxAudioLength = np.maximum(maxAudioLength, audio.shape[0]) # samples
    nFiles += 1

maxAudioLength = maxAudioLength / sr # sec
logLikelihoodMatrix = np.zeros((int(np.ceil(maxAudioLength / 10e-3)), nFiles, len(models)))
speakersProbability = np.zeros((int(np.ceil(maxAudioLength / 10e-3)), nFiles, len(models)))

# Read the test directory and get the list of test audio files
file_paths = open(test_file,'r')
for fIdx, path in enumerate(file_paths):
    path = path.strip()   
    print(path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score_samples(vector))
        log_likelihood_alongTime = scores.cumsum()
        logLikelihoodMatrix[:scores.shape[0], fIdx, i] = log_likelihood_alongTime
        logLikelihoodMatrix[scores.shape[0]:, fIdx, i] = logLikelihoodMatrix[scores.shape[0] - 1, fIdx, i]
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    #time.sleep(1.0)

    speakersProbability[:, fIdx, :] = scipy.special.softmax(logLikelihoodMatrix[:, fIdx, :], axis=1)

    if enableFigure:
        tVec = np.arange(0, speakersProbability.shape[0]) * 10e-3 + (25e-3) / 2
        plt.plot(tVec, speakersProbability[:, fIdx, winner])
        plt.xlabel('sec')
        plt.ylim(-0.1, 1.1)
        plt.title('winner speaker prob vs time')


