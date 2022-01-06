import os
import shutil
import datetime
from copy import copy, deepcopy
import re
import itertools
from typing import *
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2 as cv


def calc_label_from_img_name(image_path):
    return 1 if 'right' in image_path.split(os.sep)[-1].lower() else 0 if 'left' in image_path.split(os.sep)[-1].lower() else -1

def read_img_and_attach_label(img_path: str):
    """
    read and resize an image, attaching the appropriate label (read from the filename)
    if the filename contains 'left' the label is 0, if it contains 'right' the label is 1, otherwise -1
    :param img_path:
    :return:
    """
    label = calc_label_from_img_name(image_path=img_path)
    img = cv.imread(img_path)
    img = cv.resize(img, (1024, 1024))
    return {'img': img, 'label': label}

train_images_dir_path = 'C:\\Users\\User\\PycharmProjects\\AEyeTechnicalInterview\\Data\\train'
all_train_img_files_paths = np.array(list(map(lambda filename: os.path.join(train_images_dir_path, filename), filter(lambda x: '.jpeg' in x, os.listdir(train_images_dir_path)))))
np.random.shuffle(all_train_img_files_paths)
validation_size = int((len(all_train_img_files_paths)/100) * 20)
train_paths, validation_paths = all_train_img_files_paths[:-validation_size], all_train_img_files_paths[validation_size:]

def extract_features_on_image(img: np.array):
    img_features = {}
    lines = cv.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)
    all_theta = np.array([l[0][1] for l in lines])
    img_features['theta_mean'] = all_theta.mean()
    img_features['theta_std'] = all_theta.std()
    all_rho = np.array([l[0][0] for l in lines])
    img_features['rho_mean'] = all_rho.mean()
    img_features['rho_std'] = all_rho.std()
    img_features['n_lines'] = len(lines)

    param1 = 300
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=param1, param2=0.9, minRadius=200, maxRadius=300)
    while circles is None:
        param1 = param1//2
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=param1, param2=0.9, minRadius=200, maxRadius=300)

    circles = np.squeeze(circles)
    img_features['most_left_circle_x'] = [np.min(circles[:, 0])]
    img_features['most_right_circle_x'] = [np.max(circles[:, 0])]
    img_features['most_left_circle_r'] = [circles[np.argmin(circles[:, 0]), 2]]

    img_features['most_left_circle_y'] = [np.min(circles[:, 1])]
    img_features['most_right_circle_y'] = [np.max(circles[:, 1])]
    img_features['most_right_circle_r'] = [circles[np.argmax(circles[:, 0]), 2]]

    img_features['biggest_circle_x'] = [circles[np.argmax(circles[:, 2]), 0]]
    img_features['biggest_circle_y'] = [circles[np.argmax(circles[:, 2]), 1]]
    img_features['avg_circle_r'] = [np.mean(circles[:, 2])]

    img_features['n_circles'] = len(circles)
    return img_features


def generate_img_features_to_labels_df():
    train_images_dir_path = 'C:\\Users\\User\\PycharmProjects\\AEyeTechnicalInterview\\Data\\train'
    all_train_img_files_paths = np.array(list(map(lambda filename: os.path.join(train_images_dir_path, filename),
                                                  filter(lambda x: '.jpeg' in x, os.listdir(train_images_dir_path)))))

    all_image_features_df = None
    for img_idx, img_path in enumerate(all_train_img_files_paths):
        print(f'extracting features {img_idx}/{len(all_train_img_files_paths)}')
        single_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        single_img_features = extract_features_on_image(single_img)
        single_img_features['img_path'] = img_path
        single_img_features['label'] = 1 if 'right' in img_path.split(os.sep)[-1].lower() else 0 if 'left' in img_path.split(os.sep)[-1].lower() else -1
        if all_image_features_df is None:
            all_image_features_df = pd.DataFrame(single_img_features).set_index('img_path')
        else:
            single_img_features = pd.DataFrame(single_img_features).set_index('img_path')
            all_image_features_df.append(single_img_features)

    return all_image_features_df
# takes a while
# imgs_features = generate_img_features_to_labels_df()
# imgs_features.head()
# imgs_features.to_csv('images_features_df.csv')

def my_heuristic(image_path):
    true_label = calc_label_from_img_name(image_path)
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    lines = cv.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)
    all_theta = np.array([l[0][1] for l in lines])
    mean_theta = np.mean(all_theta)
    # pred_label = 1 if mean_theta > 2 else 0

    return mean_theta


train_images_dir_path = 'C:\\Users\\User\\PycharmProjects\\AEyeTechnicalInterview\\Data\\train'
all_train_img_files_paths = np.array(list(map(lambda filename: os.path.join(train_images_dir_path, filename),
                                              filter(lambda x: '.jpeg' in x, os.listdir(train_images_dir_path)))))

left_images = list(filter(lambda x: 'left' in x, all_train_img_files_paths))
right_images = list(filter(lambda x: 'right' in x, all_train_img_files_paths))
left_mean_theta = np.array(list(map(lambda x: my_heuristic(x), left_images)))
right_mean_theta = np.array(list(map(lambda x: my_heuristic(x), right_images)))
print()