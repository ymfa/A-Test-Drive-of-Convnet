import cv2
import numpy as np
import csv, argparse, re
from os.path import join
from datetime import datetime
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from scipy.ndimage.filters import gaussian_filter1d

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

img_shape = (38, 160, 3)
outdims = ('YUV',)

def activation_func():
  return ELU()

def nvidia_model():
  global img_shape
  img_shape = (66, 200, 3)
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
  # Normalization
  model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape))
  # Convolution 1
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(activation_func())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  # Convolution 2
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(activation_func())
  # Convolution 3
  model.add(Convolution2D(48, 2, 2, border_mode='valid'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(activation_func())
  # Dense 1
  model.add(Dense(100))
  model.add(Dropout(0.5))
  model.add(activation_func())
  # Dense 2
  model.add(Dense(10))
  model.add(activation_func())
  # Output
  model.add(Dense(1))
  model.compile("adam", "mse")
  return model

def process_img(img, indim, flip=False, augmentation=False, translation=0.):
  # warning: range after cropping [59, 135) are hard-coded for the image size
  img = img.astype('float32')
  # rotating and horizontal shift
  affine_matrix, transform = None, None
  if translation != 0.:
    affine_matrix = np.float32([[1, 0, translation], [0, 1, 0]])
  if augmentation:
    rotation_degree = np.random.normal(scale=1.)
    if rotation_degree >= .1 or rotation_degree <= -.1:  # save computation
      rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, (135+59)/2), rotation_degree, 1.)
      if affine_matrix is not None:  # prefer one perspective transform over consecutive affine transforms to minimize cropping
        affine_matrix = np.concatenate((affine_matrix, [[0., 0., 1.]]))  # 2x3 => 3x3 matrix
        rotation_matrix = np.concatenate((rotation_matrix, [[0., 0., 1.]]))  # 2x3 => 3x3 matrix
        transform = np.dot(affine_matrix, rotation_matrix)  # rotation will be performed before translation
      else:
        affine_matrix = rotation_matrix
  if transform is not None:
    img = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]))
  elif affine_matrix is not None:
    img = cv2.warpAffine(img, affine_matrix, (img.shape[1], img.shape[0]))
  # vertical shift, implemented as cropping
  vertical_shift = 0
  if augmentation:
    vertical_shift = np.rint(np.random.normal(scale=0.8)).astype(int)
    if vertical_shift > 4: vertical_shift = 4
    elif vertical_shift < -4:  vertical_shift = -4
  img = img[59+vertical_shift:135+vertical_shift]
  # flipping and resizing
  if flip:
    img = cv2.flip(img, 1)
  if img.shape[:2] != img_shape[:2]:
    img = cv2.resize(img, (img_shape[1], img_shape[0]), cv2.INTER_AREA)
  # edge detection
  if 'edge' in outdims:
    if indim == 'RGB': s_img = np.rint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).astype('uint8')
    elif indim == 'BGR': s_img = img.astype('uint8')
    else: raise NotImplementedError('Cannot convert %s into BGR' % indim)
    img_edge = np.reshape(cv2.Canny(cv2.GaussianBlur(s_img, (5, 5), 0), 50, 150), (img_shape[0], img_shape[1], 1)).astype('float32')
  # YUV colour space and random brightness
  if 'YUV' in outdims:
    if indim == 'BGR': img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif indim == 'RGB': img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else: raise NotImplementedError('Cannot convert %s into YUV' % indim)
    if augmentation:  # warning: brightness of RGB dimensions are not affected
      max_luma_ratio = min(1.25, 255. / np.max(img_yuv[:,:,0]))
      luma_ratio = np.random.uniform(.5, max_luma_ratio)
      img_yuv[:,:,0] *= luma_ratio
  # compose colour channels
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

def grade_steering(steering):
  if steering > 0.:
    if steering > 0.4: return 3
    elif steering > 0.2: return 2
    else: return 1
  elif steering < 0.:
    if steering < -0.4: return -3
    elif steering < -0.2: return -2
    else: return -1
  else: return 0

def img_generator(x_data, y_data, training=True):
  batch_size, total_size = args.batch, len(y_data)
  indices = np.array(range(total_size))
  batch_id = 0
  dw_exp, final_dw_exp, dw_exp_factor = 2., 0.75, 1.
  lr_prob, max_lr_prob, lr_prob_incr = 0., .5, 0.
  tr_prob, max_tr_prob, tr_prob_incr = 0., .5, 0.
  if training and args.epoch > 1:
    lr_prob_incr = (max_lr_prob - lr_prob) / (args.epoch - 1)
    tr_prob_incr = (max_tr_prob - tr_prob) / (args.epoch - 1)
    dw_exp_factor = (final_dw_exp / dw_exp) ** (1 / (args.epoch - 1))
  if training:
    sample_weights = np.float32([dataset.train_weight * (dataset.prob[grade_steering(y_data[i])] ** dw_exp)
                               for i, (_, dataset) in enumerate(x_data)])
  else:
    sample_weights = np.float32([dataset.test_weight for _, dataset in x_data])
  sample_weights /= np.sum(sample_weights)
  while True:
    batch_indices = np.random.choice(indices, batch_size, p=sample_weights)
    batch_info = [x_data[i] for i in batch_indices]
    batch_y = np.float32([y_data[i] for i in batch_indices])
    batch_x = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]), dtype='float32')
    for i in range(batch_size):
      filename, dataset = batch_info[i]
      flip, translation, variant = False, 0., 'center'
      if training:
        # use left/right cameras
        if dataset.side_cameras and lr_prob > 0. and np.random.rand() < lr_prob:
          dice = np.random.randint(2)
          if dice == 0:
            variant, batch_y[i] = 'left', min(batch_y[i] + .15, 1.)
          else:
            variant, batch_y[i] = 'right', max(batch_y[i] - .15, -1.)
        # horizontal translation
        if tr_prob == 1. or (tr_prob > 0. and np.random.rand() < tr_prob):
          translation = np.random.uniform(-20, 20)
          batch_y[i] += .005 * translation
          if batch_y[i] > 1.: batch_y[i] = 1.
          elif batch_y[i] < -1.: batch_y[i] = -1.
        # flipping
        if np.random.randint(2) == 1:
          flip, batch_y[i] = True, -batch_y[i]
      imgpath = join(dataset.path, "IMG", variant + filename)
      image = process_img(cv2.imread(imgpath), 'BGR', flip, training, translation)
      batch_x[i] = image
    yield batch_x, batch_y
    # adjust probability of the training samples
    batch_id += 1
    if training and batch_id == args.train:
      batch_id = 0
      lr_prob += lr_prob_incr
      tr_prob += tr_prob_incr
      if dw_exp_factor != 1.:
        dw_exp *= dw_exp_factor
        sample_weights = np.float32([dataset.train_weight * (dataset.prob[grade_steering(y_data[i])] ** dw_exp)
                                     for i, (_, dataset) in enumerate(x_data)])
        sample_weights /= np.sum(sample_weights)

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
  def __init__(self, path, side_cameras=False, weight=1., frame_shift=0):
    self.path = path
    self.side_cameras = side_cameras
    self.frame_shift = frame_shift
    self.test_weight = weight
    if side_cameras:
      self.train_weight = weight * 2.
    else:
      self.train_weight = weight

udacity_data = Dataset('data', True, weight=2., frame_shift=1)
my_data = Dataset('my_track1', False, frame_shift=1)

def load_data(dataset_list):
  x_train, y_train, y_count = [], None, Counter()
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
        filename = row[0].strip()[10:]
        current_time = datetime(*[int(v) for v in ts_recognizer.search(filename).groups()])
        if previous_time is not None:
          time_delta = current_time - previous_time
          if not (0 <= time_delta.seconds <= 1 and time_delta.days == 0):
            x_train, y_train = extend_data(x_train, x_dataset, y_train, y_dataset, dataset.frame_shift)
            x_dataset, y_dataset = [], []
            print("Time gap in training data:", time_delta)
        previous_time = current_time
        # append to x, y, and the counter
        x_dataset.append((filename, dataset))
        y_dataset.append(steering)
        y_count[grade_steering(steering)] += 1
    # submit to the pool of training data
    x_train, y_train = extend_data(x_train, x_dataset, y_train, y_dataset, dataset.frame_shift)
    # balance the steering angles in this dataset
    max_count = max(y_count.values())
    dataset.prob = {k: max_count/v for k, v in y_count.items()}
    y_count = Counter()
  # validation set
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05, random_state=42)
  print('Training set:', len(y_train), 'frames')
  print('Validation set:', len(y_valid), 'frames')
  return img_generator(x_train, y_train), img_generator(x_valid, y_valid, False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train a behavioural cloning model')
  parser.add_argument('--batch', type=int, default=256, help='Batch size')
  parser.add_argument('--train', type=int, default=64, help='Number of training batches per epoch')
  parser.add_argument('--valid', type=int, default=2, help='Number of validation batches per epoch')
  parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
  parser.add_argument('--refine', action='store_true', help='Use this to refine a trained model')
  args = parser.parse_args()
  
  model = my_model()
  if args.refine:  # argument 'initial_epoch' not usable in fit_generator due to Keras version
    gen_train1, gen_valid1 = load_data([refine_data])
    model.load_weights('model.h5')
    output_name = 'improved'
  else:
    gen_train1, gen_valid1 = load_data([udacity_data, my_data])
    output_name = 'model'
  
  print("Training...")
  model.fit_generator(
      gen_train1, args.batch * args.train, args.epoch,
      validation_data=gen_valid1, nb_val_samples=args.batch * args.valid
  )

  print("Saving model...")
  model.save_weights(output_name + ".h5")
  with open(output_name + '.json', 'w') as outfile:
    outfile.write(model.to_json())
