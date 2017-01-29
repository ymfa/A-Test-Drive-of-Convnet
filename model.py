import cv2
import numpy as np
import tensorflow as tf
import csv, argparse, re
from os.path import join
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from scipy.ndimage.filters import gaussian_filter1d

#img_shape = (66, 200, 3)
img_shape = (38, 160, 3)
outdims = ('YUV',)

def activation_func():
  return ELU()
  
def img_normalization(img):
  return img / tf.reduce_mean(img, axis=(0, 1)) - 1.
  
def nvidia_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(64, 3, 3, border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(64, 3, 3, border_mode='valid'))
  model.add(activation_func())
  model.add(Flatten())
  model.add(Dense(100))
  model.add(activation_func())
  model.add(Dense(50))
  model.add(activation_func())
  model.add(Dense(1))
  model.compile("adam", "mse")
  return model

def my_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(activation_func())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(activation_func())
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(activation_func())
  model.add(Convolution2D(64, 2, 2, border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(activation_func())
  model.add(Dense(200))
  model.add(Dropout(0.5))
  model.add(activation_func())
  model.add(Dense(20))
  model.add(activation_func())
  model.add(Dense(1))
  model.compile("adam", "mse")
  return model

def process_img(img, indim, flip=False, luma_alt=False):
  img = img[59:135]
  if 'edge' in outdims:
    if indim == 'RGB': s_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif indim == 'BGR': s_img = img
    else: raise NotImplementedError('Cannot convert %s into BGR' % indim)
    img_edge = np.reshape(cv2.Canny(cv2.GaussianBlur(s_img, (5, 5), 0), 50, 150), (img_shape[0], img_shape[1], 1)).astype('float32')
  img = img.astype('float32')
  if flip:
    img = cv2.flip(img, 1)
  if img.shape[:2] != img_shape[:2]:
    img = cv2.resize(img, (img_shape[1], img_shape[0]), cv2.INTER_AREA)
  if 'YUV' in outdims:
    if indim == 'BGR': img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif indim == 'RGB': img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else: raise NotImplementedError('Cannot convert %s into YUV' % indim)
    if luma_alt:
      max_luma_ratio = min(1.25, 255. / np.max(img_yuv[:,:,0]))
      luma_ratio = np.random.uniform(.75, max_luma_ratio)
      img_yuv[:,:,0] *= luma_ratio
  layers = []
  for dim in outdims:
    if dim == 'RGB':
      if indim == 'RGB': layers.append(img)
      elif indim == 'BGR': layers.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      else: raise NotImplementedError('Cannot convert %s into RGB' % indim)
    elif dim == 'YUV':
      layers.append(img_yuv)
    elif dim == 'edge':
      layers.append(img_edge)
    else:
      raise NotImplementedError('Invalid output dimension %s' % dim)
  if len(layers) == 1:
    return layers[0]
  else:
    return np.concatenate(layers, 2)
    
def filter_zero_steering(x, y, drop_prob):
  indices_to_remove = np.where(y == 0.)[0]
  np.random.shuffle(indices_to_remove)
  indices_to_remove = indices_to_remove[:np.rint(len(indices_to_remove) * drop_prob)]
  mask = np.ones(len(y), dtype=bool)
  mask[indices_to_remove] = False
  return np.array(x)[mask], y[mask]

def img_generator(x_data, y_data, training=True):
  batch_size = args.batch
  while True:
    x_epoch, y_epoch = filter_zero_steering(x_data, y_data, 0.8)
    if training:
      x_epoch, y_epoch = shuffle(x_epoch, y_epoch)
    for offset in range(0, len(y_epoch), batch_size):
      batch_paths, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
      batch_x = np.zeros((len(batch_paths), img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
      for i, (imgpath, flip) in enumerate(batch_paths):
        batch_x[i] = process_img(cv2.imread(imgpath), 'BGR', flip, training)
      yield batch_x, batch_y

def extend_data(x_old, x_new, y_old, y_new, shift=0):
  if not y_new: return x_old, y_old
  if shift > 0:
    if shift >= len(y_new): return x_old, y_old
    y_new = y_new[shift:]
    x_new = x_new[:-shift]
  y_new = np.array(y_new, dtype='float32')
  if y_old is None:
    return x_new, y_new
  else:
    return x_old + x_new, np.concatenate((y_old, y_new))

class Dataset(object):
  def __init__(self, path, columns=(0,), flips=(False,), time_shift=0):
    self.path = path
    self.columns = columns
    self.flips = flips
    self.frame_shift = time_shift * (len(columns) + sum(flips))

udacity_data = Dataset('data', (0,1,2), (True,False,False), 1)

def load_data(dataset_list):
  x_train, y_train = [], None
  ts_recognizer = re.compile(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', re.ASCII)
  for dataset in dataset_list:
    x_dataset, y_dataset = [], []
    previous_time = None
    print("Dataset:", dataset.path)
    with open(join(dataset.path, 'driving_log.csv'), 'r') as csvfile:
      for row in csv.reader(csvfile, delimiter=','):
        try: steering = float(row[3])
        except ValueError: continue
        # if there is a gap in the recording, treat the previous part as a seperate dataset
        filename = row[0].strip()
        current_time = datetime(*[int(v) for v in ts_recognizer.search(filename).groups()])
        if previous_time is not None:
          time_delta = current_time - previous_time
          if not (0 <= time_delta.seconds <= 1 and time_delta.days == 0):
            x_train, y_train = extend_data(x_train, x_dataset, y_train, y_dataset, dataset.frame_shift)
            x_dataset, y_dataset = [], []
            print("Time gap in training data:", time_delta)
        previous_time = current_time
        # append to x, y
        for column, flip in zip(dataset.columns, dataset.flips):
          filename = row[column].strip()
          if column == 0: adj_steering = steering
          elif column == 1: adj_steering = min(steering + .15, 1.)
          elif column == 2: adj_steering = max(steering - .15, -1.)
          else: raise ValueError('Image must be in column 0, 1, or 2')
          x_dataset.append((join(dataset.path, filename), False))
          y_dataset.append(adj_steering)
          if flip:
            x_dataset.append((join(dataset.path, filename), True))
            y_dataset.append(-adj_steering)
    # submit to the pool of training data
    x_train, y_train = extend_data(x_train, x_dataset, y_train, y_dataset, dataset.frame_shift)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
  return img_generator(x_train, y_train), img_generator(x_valid, y_valid, False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train a behavioural cloning model')
  parser.add_argument('--batch', type=int, default=256, help='Batch size')
  parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
  args = parser.parse_args()
  
  model = my_model()
  gen_train1, gen_valid1 = load_data([udacity_data])
  
  print("Training...")
  model.fit_generator(
      gen_train1, args.batch * 64, args.epoch,
      validation_data=gen_valid1, nb_val_samples=args.batch * 8
  )

  print("Saving model...")
  model.save_weights("./model.h5")
  with open('./model.json', 'w') as outfile:
    outfile.write(model.to_json())
