#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import six.moves.urllib.request as request
import six
import os
import tarfile
import shutil
import hashlib
import sys
import librosa

import pickle

import numpy as np

def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6).reshape([80, 80, 1])

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo)


def oneHotVector(classIdx, numClasses):
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)
    v[np.arange(0, len(v)), classIdx] = 1
    return v


class GZTan2:
    IMG_WIDTH = 80
    IMG_HEIGHT = 80
    CLASS_COUNT = 10

    dataPath = ''
    batchSize = 128
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testLabels = np.array([])
    currentIndexTrain = 0
    currentIndexTest = 0
    nTrainSamples = 0
    nTestSamples = 0

    pTrain = []
    pTest = []

    def __init__(self, batchSize=128):
        self.batchSize = batchSize
        self.loadGZTan()

    def preprocess(self):
        """
        Convert pixel values to lie within [0, 1]
        """
        self.trainData = self._normaliseImages(self.trainData.astype(np.float32, copy=False))
        self.testData = self._normaliseImages(self.testData.astype(np.float32, copy=False))

    def _normaliseImages(self, imgs_flat):
        min = np.min(imgs_flat)
        max = np.max(imgs_flat)
        range = max - min
        return (imgs_flat - min) / range

    def _unflatten(self, imgs_flat):
        return imgs_flat.reshape(-1, self.IMG_WIDTH, self.IMG_HEIGHT)

    def _flatten(self, imgs):
        return imgs.reshape(-1, self.IMG_WIDTH * self.IMG_HEIGHT)

    def loadGZTan(self):
        with open('music_genres_dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
            test_set = pickle.load(f)

        self.trainData = np.array(train_set['data'])
        self.trainLabels = oneHotVector(train_set['labels'], 10)

        self.testData = np.array(test_set['data'])
        self.testLabels = oneHotVector(test_set['labels'], 10)

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

    def getTrainBatch(self, allowSmallerBatches=False):
        return self._getBatch('train', allowSmallerBatches)

    def getTestBatch(self, allowSmallerBatches=False):
        return self._getBatch('test', allowSmallerBatches)

    def _getBatch(self, dataSet, allowSmallerBatches=False):
        D = np.array([])
        L = np.array([])

        if dataSet == 'train':
            train = True
            test = False
        elif dataSet == 'test':
            train = False
            test = True
        else:
            raise ValueError('_getBatch: Unrecognised set: ' + dataSet)

        while True:
            if train:
                r = range(self.currentIndexTrain,
                          min(self.currentIndexTrain + self.batchSize - L.shape[0], self.nTrainSamples))
                self.currentIndexTrain = r[-1] + 1 if r[-1] < self.nTrainSamples - 1 else 0
                (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])
            elif test:
                r = range(self.currentIndexTest,
                          min(self.currentIndexTest + self.batchSize - L.shape[0], self.nTestSamples))
                self.currentIndexTest = r[-1] + 1 if r[-1] < self.nTestSamples - 1 else 0
                (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])

            d = np.apply_along_axis(melspectrogram, axis=1, arr=d)

            if D.size == 0:
                D = d
                L = l
            else:
                D = np.concatenate((D, d))
                L = np.concatenate((L, l))

            if D.shape[0] == self.batchSize or allowSmallerBatches:
                break

        return (D, L)

    def reset(self):
        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)
