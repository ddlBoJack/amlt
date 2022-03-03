import os
import sys
import gzip
import shutil
from six.moves import urllib


# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"


def download_and_uncompress_zip(URL, dataset_dir, force=False):
  filename = URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
  extract_to = os.path.splitext(filepath)[0]

  def download_progress(count, block_size, total_size):
    sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
    sys.stdout.flush()

  if not force and os.path.exists(filepath):
    print("file %s already exist" % (filename))
  else:
    filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
    print()
    print('Successfully Downloaded', filename)

  # with zipfile.ZipFile(filepath) as fd:
  with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
    print('Extracting ', filename)
    shutil.copyfileobj(f_in, f_out)
    print('Successfully extracted')
    print()


def start_download(save_to, force=True):
  if not os.path.exists(save_to):
    os.makedirs(save_to)
  download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, save_to, force)
  download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, save_to, force)
  download_and_uncompress_zip(MNIST_TEST_IMGS_URL, save_to, force)
  download_and_uncompress_zip(MNIST_TEST_LABELS_URL, save_to, force)


if __name__ == '__main__':
  this_file_path = os.path.dirname(os.path.realpath(__file__))
  save_path = os.path.join(this_file_path, os.pardir, "data")
  print("Saving data to ", save_path)
  start_download(save_path)
