import pandas as pd
import numpy as np
import scipy
import itertools
import string
import glob
import os
import csv
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt


def main():
	"""
	Input: txt files in data folder
	Output: csv file
	"""
    number_of_files = int(len(os.listdir('data/')) / 3)
    all_features = []
    for i in range(0, number_of_files):
        features = read_files(i)
        reshaped_features = {}
        reshaped_features['rpm'] = features['rpm']
        for key in features['x_accel_features']:
            new_key = 'x_' + str(key)
            reshaped_features[new_key] = features['x_accel_features'][key]
        for key in features['y_accel_features']:
            new_key = 'y_' + str(key)
            reshaped_features[new_key] = features['y_accel_features'][key]
        reshaped_features['label'] = features['label']
        all_features.append(reshaped_features)
    keys = all_features[0].keys()
    with open('data.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_features)
    """
    SVM implementation
    """
    # fname = 'data.csv'
    # ffname = 'data_6.21.csv'
    # dataframe = load_data(fname)
    # positiveDF = dataframe[dataframe['label'] == True].copy()
    # negativeDF = dataframe[dataframe['label'] == False].copy()
    # X_train = pd.concat([positiveDF[:50], negativeDF[:50]]).reset_index(drop=True).copy()
    # Y_train = X_train['label'].values.copy()
    # dataframe = load_data(ffname)
    # positiveDF = dataframe[dataframe['label'] == True].copy()
    # negativeDF = dataframe[dataframe['label'] == False].copy()
    # X_test = pd.concat([positiveDF[:50], negativeDF[:50]]).reset_index(drop=True).copy()
    # Y_test = X_test['label'].values.copy()
    # X_train = generate_feature_matrix(X_train)
    # X_test = generate_feature_matrix(X_test)
    # clf = LinearSVC(penalty='l1', dual=False, C=1, class_weight='balanced')
    # clf.fit(X_train, Y_train)

    # start_time = time.time()
    # print(len(clf.predict(X_test)))
    # print(time.time() - start_time)

def generate_feature_matrix(csv_data):
    number_of_points = csv_data.shape[0]
    number_of_features = csv_data.shape[1] - 1
    feature_matrix = np.zeros((number_of_points, number_of_features))
    for i, point in enumerate(csv_data.values):
        feature_matrix[i] = point[:number_of_features]
    return feature_matrix

def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)

def read_files(FileNumber):
    balanced_file_name = 'data/rpm_' + str(FileNumber) + '_b.txt'
    if os.path.isfile(balanced_file_name):
        rpm_file = 'data/rpm_' + str(FileNumber) + '_b.txt'
        x_accel_file = 'data/x_accel_' + str(FileNumber) + '_b.txt'
        y_accel_file = 'data/y_accel_' + str(FileNumber) + '_b.txt'
        label = True
    else:
        rpm_file = 'data/rpm_' + str(FileNumber) + '_u.txt'
        x_accel_file = 'data/x_accel_' + str(FileNumber) + '_u.txt'
        y_accel_file = 'data/y_accel_' + str(FileNumber) + '_u.txt'
        label = False
    return get_features(rpm_file, x_accel_file, y_accel_file, label)

def get_features(rpm_file, x_accel_file, y_accel_file, balanced):
    features = {}
    rpm_data = parse_file(rpm_file)
    x_accel_data = parse_file(x_accel_file)
    y_accel_data = parse_file(y_accel_file)
    features['rpm'] = get_rpm(rpm_data, 10000)
    features['x_accel_features'] = get_accel_features(x_accel_data)
    features['y_accel_features'] = get_accel_features(y_accel_data)
    features['label'] = balanced
    return features

def get_rpm(rpm_data, frequency):
    number_of_points = len(rpm_data)
    time = number_of_points / frequency / 60 # time in minute
    number_of_peaks = 0
    old_data = 0
    for data in rpm_data:
        if old_data < 2.5 and data > 2.5:
            number_of_peaks += 1
        old_data = data
    rpm = number_of_peaks / 2 / time
    return rpm

def get_accel_features(accel_data):
    accel_features = {}
    accel_features['mean'] = np.mean(accel_data)
    accel_features['std'] = np.std(accel_data)
    accel_features['var'] = np.var(accel_data)
    sqr_rms = 0
    N = len(accel_data)
    for item in accel_data:
        sqr_rms += item**2
    rms = np.sqrt(sqr_rms / N)
    accel_features['rms'] = rms
    accel_features['skewness'] = scipy.stats.skew(accel_data)
    accel_features['kurtosis'] = scipy.stats.kurtosis(accel_data)
    accel_features['ppk'] = max(accel_data) - min(accel_data)
    accel_features['true_peak'] = max(accel_data)
    accel_features['crest_factor'] = accel_features['true_peak'] / accel_features['rms']
    return accel_features

def parse_file(filename):
    file = open(filename, 'r')
    num_lines = sum(1 for line in file)
    data = np.zeros(num_lines)
    file = open(filename, 'r')
    for line in file:
        (step_number, token) = line.split()
        data[int(step_number)] = float(token)
    return remove_outliers(data)

def remove_outliers(data):
    std = np.std(data)
    mean = np.mean(data)
    for index, item in enumerate(data):
        if abs(item - mean) > 3*std:
            data[index] = mean
    return data

if __name__ == '__main__':
    main()