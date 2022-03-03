import os
import numpy as np


def make_input_batch(nbatch, X, Y):
  ndata = X.shape[0]
  samples = np.random.permutation(np.arange(ndata))[:nbatch]
  x = X[samples] / 255.
  y = Y[samples]
  return (x.astype("float32").reshape(nbatch, -1), y)


def load_mnist(data_dir):
  path = data_dir
  fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
  loaded = np.fromfile(file=fd, dtype=np.uint8)
  trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

  fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
  loaded = np.fromfile(file=fd, dtype=np.uint8)
  trainY = loaded[8:].reshape((60000)).astype(np.int32)

  trX = trainX[:55000]
  trY = trainY[:55000]
  valX = trainX[55000:]
  valY = trainY[55000:]

  fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
  loaded = np.fromfile(file=fd, dtype=np.uint8)
  teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

  fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
  loaded = np.fromfile(file=fd, dtype=np.uint8)
  teY = loaded[8:].reshape((10000)).astype(np.int32)

  return trX, trY, valX, valY, teX, teY
