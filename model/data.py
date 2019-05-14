#!/usr/bin/python
# author: Charlotte Bunne

# imports
import numpy as np
import random


def gaussians_8mode(sample_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1./np.sqrt(2), 1./np.sqrt(2)),
        (1./np.sqrt(2), -1./np.sqrt(2)),
        (-1./np.sqrt(2), 1./np.sqrt(2)),
        (-1./np.sqrt(2), -1./np.sqrt(2))
    ]
    centers = [(scale*x, scale*y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2)*.2
        index = random.randint(0, len(centers)-1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_5mode(sample_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (0, 0)
    ]
    centers = [(scale*x, scale*y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2)*.2
        index = random.randint(0, len(centers)-1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_4mode(sample_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]
    centers = [(scale*x, scale*y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2)*.2
        index = random.randint(0, len(centers)-1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_3d_4mode(sample_size):
    scale = 2.
    centers = [
        (0, 1, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, -1, -1),
    ]
    centers = [(scale*x, scale*y, scale*z) for x, y, z in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(3)*.2
        index = random.randint(0, len(centers)-1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        point[2] += center[2]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)
