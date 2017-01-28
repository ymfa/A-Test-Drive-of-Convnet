import cv2
import numpy as np
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

img_shape = (75, 320, 3)
outdims = ('YUV',)

def activation_func():
  return ELU()

def define_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape))
  model.add(Convolution2D(24, 7, 7, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(36, 7, 7, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(48, 7, 7, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(64, 3, 3, border_mode='valid'))
  model.add(activation_func())
  model.add(Convolution2D(64, 3, 3, border_mode='valid'))
  model.add(Dropout(.2))
  model.add(Flatten())
  model.add(activation_func())
  model.add(Dense(200))
  model.add(Dropout(.5))
  model.add(activation_func())
  model.add(Dense(50))
  model.add(activation_func())
  model.add(Dense(10))
  model.add(activation_func())
  model.add(Dense(1))
  model.compile("adam", "mse")
  return model

def process_img(img, indim, flip=False, luma_alt=False):
  img = img[60:135]
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

def img_generator(x_data, y_data, training=True):
  num_examples = len(y_data)
  batch_size = args.batch
  while True:
    if training:
      x_data, y_data = shuffle(x_data, y_data)
    for offset in range(0, num_examples, batch_size):
      batch_paths, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
      batch_x = np.zeros((len(batch_paths), img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
      for i, (imgpath, flip) in enumerate(batch_paths):
        batch_x[i] = process_img(cv2.imread(imgpath), 'BGR', flip)
      yield batch_x, batch_y

def extend_y(y_old, y_new, smoothing=0.):
  if not y_new: return y_old
  y_new = np.array(y_new, dtype='float32')
  if smoothing > 0.:
    y_new = gaussian_filter1d(y_new, smoothing)
  if y_old is None:
    return y_new
  else:
    return np.concatenate((y_old, y_new))
    
datatypes = {
  'good': {'flip': True, 'smoothing': 0., 'keep0': 1.},
  'user': {'flip': False, 'smoothing': 0., 'keep0': .2}
}

def load_data(dataset_list):
  x_train, y_train = [], None
  ts_recognizer = re.compile(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', re.ASCII)
  for dataset, data_config in dataset_list:
    x_dataset, y_dataset = [], []
    previous_time = None
    print("Dataset:", dataset)
    with open(join(dataset, 'driving_log.csv'), 'r') as csvfile:
      for row in csv.reader(csvfile, delimiter=','):
        try: steering = float(row[3])
        except ValueError: continue
        filename = row[0].strip()
        writable = True
        if steering == 0. and data_config['keep0'] < 1. and np.random.uniform(0., 1.) > data_config['keep0']:
          writable = False
        if writable:
          x_dataset.append((join(dataset, filename), False))
          if data_config['flip']: x_dataset.append((join(dataset, filename), True))
        current_time = datetime(*[int(v) for v in ts_recognizer.search(filename).groups()])
        if previous_time is not None:
          time_delta = current_time - previous_time
          if not (0 <= time_delta.seconds <= 1 and time_delta.days == 0):
            y_train = extend_y(y_train, y_dataset, data_config['smoothing'])
            y_dataset = []
            print("Time gap in training data:", time_delta)
        previous_time = current_time
        if writable:
          y_dataset.append(steering)
          if data_config['flip']: y_dataset.append(-steering)
    x_train.extend(x_dataset)
    y_train = extend_y(y_train, y_dataset, data_config['smoothing'])
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
  return img_generator(x_train, y_train), len(y_train), img_generator(x_valid, y_valid, False), len(y_valid)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train a behavioural cloning model')
  parser.add_argument('--batch', type=int, default=256, help='Batch size')
  parser.add_argument('--epoch', type=int, default=5, help='Number of epochs')
  args = parser.parse_args()
  
  model = define_model()
  gen_train1, gen_train1_len, gen_valid1, gen_valid1_len = load_data([('data', datatypes['good']), ('session_data', datatypes['user'])])
  
  print("Training...")
  model.fit_generator(
      gen_train1, gen_train1_len, args.epoch,
      validation_data=gen_valid1, nb_val_samples=gen_valid1_len
  )

  print("Saving model...")
  model.save_weights("./model.h5")
  with open('./model.json', 'w') as outfile:
    outfile.write(model.to_json())
