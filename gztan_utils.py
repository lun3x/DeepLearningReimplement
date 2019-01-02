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

def oneHotVector(classIdx, numClasses):
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)
    v[np.arange(0, len(v)), classIdx] = 1
    return v

class GZTan2:
    IMG_WIDTH = 80
    IMG_HEIGHT = 80
    CLASS_COUNT = 10

    numBatches = 50
    batchSize = 0
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testLabels = np.array([])
    testTracks = np.array([])
    nTrainSamples = 0
    nTestSamples = 0
    nTracks = 0

    def __init__(self, numBatches=50):
        self.numBatches = numBatches
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
        self.testTracks = np.array(test_set['track_id'])

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)
        self.nTracks = len(np.lib.arraysetops.unique(self.testTracks))

        self.trainBatchSize = self.nTrainSamples // self.numBatches
        self.testBatchSize = self.nTestSamples // self.numBatches

    def getTrainBatch(self, batchNum):
        return self._getBatch2('train', batchNum)

    def getTestBatch(self, batchNum):
        return self._getBatch2('test', batchNum)

    def _getBatch2(self, dataSet, batchNum):
        if dataSet == 'train':
            start_idx = batchNum * self.trainBatchSize
            end_idx = (batchNum + 1) * self.trainBatchSize
            if batchNum == self.numBatches - 1:
                end_idx = self.nTrainSamples - 1
            r = range(start_idx, end_idx)

            (d, l) = (self.trainData[r][:], self.trainLabels[r][:])

        elif dataSet == 'test':
            start_idx = batchNum * self.testBatchSize
            end_idx = (batchNum + 1) * self.testBatchSize
            if batchNum == self.numBatches - 1:
                end_idx = self.nTestSamples - 1
            r = range(start_idx, end_idx)

            (d, l) = (self.testData[r][:], self.testLabels[r][:])

        d = np.apply_along_axis(melspectrogram, axis=-1, arr=d)

        return (d, l)

    def getTrackSamples(self, track_id):
        trackIndices = np.where(self.testTracks == track_id)[0]
        D = self.testData[trackIndices]
        D = np.apply_along_axis(melspectrogram, axis=1, arr=D)
        label = self.testLabels[trackIndices[0]]
        return D, label

    def reset(self):
        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)
