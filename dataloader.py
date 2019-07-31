import os
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import keras.backend as K


class DataLoader:

    def __init__(self,
                 batch_size=512,
                 input_width=200,
                 input_height=66,
                 input_channels=3,
                 delta_correction=0.25,
                 aug_steer_std=0.2,
                 aug_val_min=0.2,
                 aug_val_max=1.5,
                 bias=0.8,
                 crop_height=range(20, 140)):
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.delta_correction = delta_correction
        self.aug_steer_std = aug_steer_std
        self.aug_val_min = aug_val_min
        self.aug_val_max = aug_val_max
        self.bias = bias
        self.crop_height = crop_height

    def get_train_val(self, driving_data, test_size=0.2):
        with open(driving_data, 'r') as f:
            reader = csv.reader(f)
            driving_data = [row for row in reader][1:]
        train_data, val_data = train_test_split(driving_data, test_size=test_size, 
                                                random_state=1)
        return train_data, val_data

    def preprocess(self, frame, verbose=False):
        # training images resized shape
        h, w = self.input_height, self.input_width
        # crop image (remove useless information, e.g. tree, sky, etc. in top half)
        frame_cropped = frame[self.crop_height, :, :]
        # resize image
        frame_resized = cv2.resize(frame_cropped, dsize=(w, h))
        # change color space
        if self.input_channels == 1:
            frame_resized = np.expand_dims(cv2.cvtColor(frame_resized,
                                        cv2.COLOR_BGR2YUV)[:, :, 0], 2)
        if verbose:
            plt.figure(1), plt.imshow(cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB))
            plt.figure(2), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
            plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
            plt.show()
        return frame_resized.astype('float32')

    def load_data_batch(self, data, data_dir='data', augment_data=True, bias=0.5):
        batchsize = self.batch_size
        # training images resized shape
        h, w, c = self.input_height, self.input_width, self.input_channels
        # prepare output structures
        X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
        y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
        #y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)  # not using it

        # shuffle data
        shuffled_data = shuffle(data)
        loaded_elements = 0
        while loaded_elements < batchsize:
            ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()
            # cast strings to float32
            steer = np.float32(steer)
            throttle = np.float32(throttle)

            # randomly choose which camera to use among (central, left, right)
            # in case the chosen camera is not the frontal one, adjust steer accordingly
            delta_correction = self.delta_correction
            camera = random.choice(['frontal', 'left', 'right'])
            if camera == 'frontal':
                frame = self.preprocess(cv2.imread(os.path.join(data_dir, ct_path.strip())))
                steer = steer
            elif camera == 'left':
                frame = self.preprocess(cv2.imread(os.path.join(data_dir, lt_path.strip())))
                steer = steer + delta_correction
            elif camera == 'right':
                frame = self.preprocess(cv2.imread(os.path.join(data_dir, rt_path.strip())))
                steer = steer - delta_correction

            if augment_data:
                # mirror images with chance=0.5
                if random.choice([True, False]):
                    frame = frame[:, ::-1, :]
                    steer *= -1.
                # perturb slightly steering direction
                steer += np.random.normal(loc=0, scale=self.aug_steer_std)
                # if color images, randomly change brightness
                if self.input_channels == 3:
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                    frame[:, :, 2] *= random.uniform(self.aug_val_min, self.aug_val_max)
                    frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

            # check that each element in the batch meet the condition
            steer_magnitude_thresh = np.random.rand()
            if (abs(steer) + bias) < steer_magnitude_thresh:
                pass  
            else:
                X[loaded_elements] = frame
                y_steer[loaded_elements] = steer
                loaded_elements += 1

        return X, y_steer

    def generate_data_batch(self, data, data_dir='data', augment_data=True, bias=0.5):
       while True:
            X, y_steer = self.load_data_batch(data, data_dir, augment_data, bias)
            yield X, y_steer