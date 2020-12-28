from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset



def str_column_to_float(dataset, column):
    for row in dataset:
        column=int(column)
        row[column] = float(row[column].strip())



def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup



def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax



def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split




def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0



def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores




def euclidean_distance(r1, r2):
    dist = 0.0
    for i in range(len(r1) - 1):
        if i is int:
            dist += r1[i] - r2[i] ** 2
    return sqrt(dist)



def get_neighbors(train, test_r, num_n):
    distances = list()
    for train_r in train:
        dist = euclidean_distance(test_r, train_r)
        distances.append((train_r, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_n):
        neighbors.append(distances[i][0])
    return neighbors



def predict_classification(train, test_r, num_n):
    neighbors = get_neighbors(train, test_r, num_n)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def k_nearest_neighbors(train, test, num_n):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_n)
        predictions.append(output)
    return (predictions)


seed(1)

filename2 = '/Users/Pro/Desktop/COVID-19.csv'
Covid_19Dataset = load_csv(filename2)


Covid_19_data = pd.read_csv(filename2)
print(Covid_19_data.shape)
print()
print(Covid_19_data.head())
print()
print(Covid_19_data.groupby("Decision label").size())
print()
print(Covid_19_data)

Covid_19Dataset.remove(Covid_19Dataset[0])
#
# # I konvertojme te gjitha features kolonat ne tipin float
for i in range(len(Covid_19Dataset[0]) - 1):
    if i == 7 and i ==8 and i ==13:
        str_column_to_float(Covid_19Dataset, i)

str_column_to_int(Covid_19Dataset, len(Covid_19Dataset[0]) - 1)

n_folds = 5
num_neighbors = 5
print(evaluate_algorithm(Covid_19Dataset, k_nearest_neighbors, n_folds, num_neighbors))
scores2 = evaluate_algorithm(Covid_19Dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores per datasetin COVID-19: %s' % scores2)
print("Numri i neighbours K =", num_neighbors)
print('Mean Accuracy: %.3f%%' % (sum(scores2) / float(len(scores2))))
print()

