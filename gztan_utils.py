from __future__ import print_function, unicode_literals
import six.moves.urllib.request as request
import six
import os
import tarfile
import shutil
import hashlib
import sys
import librosa

import cPickle

import numpy as np

def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    # print('Mel Spec shape: {}'.format(spec.shape))
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    # print('Mel basis shape: {}'.format(mel_basis.shape))
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6).reshape([80, 80, 1])

def cqt(audio):
    cqt = librosa.core.cqt(audio, sr=22050, hop_length=256, n_bins=80, pad_mode='constant')
    cqt_spec = np.abs(cqt)

    return np.log(cqt_spec + 1e-6).reshape([80, 80, 1])

def binariseVector(classIdx, numClasses):
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)
    v[np.arange(0, len(v)), classIdx] = 1
    return v

class GZTan:
    IMG_WIDTH = 80
    IMG_HEIGHT = 80
    CLASS_COUNT = 10

    numBatches = 50
    batchSize = 0
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testDataOriginal = np.array([])
    testLabels = np.array([])
    testTracks = np.array([])
    nTrainSamples = 0
    nTestSamples = 0
    nTracks = 0
    representationFunc = melspectrogram
    pTrain = []
    pTest = []

    def __init__(self, numBatches=50, mel=True):
        self.numBatches = numBatches
        if mel:
            self.representationFunc = melspectrogram
        else:
            self.representationFunc = cqt

        self.loadGZTan()

    def loadGZTan(self):
        with open('music_genres_dataset.pkl', 'rb') as f:
            train_set = cPickle.load(f)
            test_set = cPickle.load(f)

        train_data = np.array(train_set['data'])
        self.trainLabels = binariseVector(train_set['labels'], 10)

        self.testDataOriginal = np.array(test_set['data'])
        self.testLabels = binariseVector(test_set['labels'], 10)

        self.testTracks = np.array(test_set['track_id'])
        self.trainTracks = np.array(train_set['track_id'])

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)
        self.nTracks = len(np.lib.arraysetops.unique(self.testTracks))
        self.nTrainTracks = len(np.lib.arraysetops.unique(self.trainTracks))

        self.trainBatchSize = self.nTrainSamples // self.numBatches
        self.testBatchSize = self.nTestSamples // self.numBatches

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

        self.trainData = np.apply_along_axis(self.representationFunc, axis=-1, arr=train_data)
        self.testData = np.apply_along_axis(self.representationFunc, axis=-1, arr=self.testDataOriginal)

        print('testBatchSize: {}, trainBatchSize: {}'.format(self.testBatchSize, self.trainBatchSize))
        print('trainData length: {}, testData length: {}'.format(len(self.trainData), len(self.testData)))

    def getTrainBatch(self, batchNum):
        return self._getBatch('train', batchNum)

    def getTestBatch(self, batchNum):
        return self._getBatch('test', batchNum)

    def _getBatch(self, dataSet, batchNum):
        if dataSet == 'train':
            start_idx = batchNum * self.trainBatchSize
            end_idx = (batchNum + 1) * self.trainBatchSize
            if batchNum == self.numBatches - 1:
                end_idx = self.nTrainSamples - 1
            r = range(start_idx, end_idx)

            (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])

        elif dataSet == 'test':
            start_idx = batchNum * self.testBatchSize
            end_idx = (batchNum + 1) * self.testBatchSize
            if batchNum == self.numBatches - 1:
                end_idx = self.nTestSamples - 1
            r = range(start_idx, end_idx)

            (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])

        return (d, l)

    def outputSample(self, track_id, sample_id):
        trackIndices = np.where(self.testTracks == track_id)[0]
        D = self.testDataOriginal[trackIndices]
        sample = np.array(D[sample_id])
        librosa.output.write_wav('incorrect_track{t}_sample{e}.wav'.format(t=track_id, e=sample_id), y=sample, sr=22050)
        return D

    def getClassSamples(self, class_label, batch_num):
        classIndices = np.where(self.testLabels == class_label)[0]
        batch_size = 15

        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size

        r = range(start_idx, end_idx)

        D = self.testData[classIndices[r]]
        return D

    def getTrackSamples(self, track_id):
        trackIndices = np.where(self.testTracks == track_id)[0]
        # print('shape of track indices: {}'.format(trackIndices.shape))
        D = self.testData[trackIndices]
        labels = self.testLabels[trackIndices]
        return D, labels

    def shuffle(self):
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)