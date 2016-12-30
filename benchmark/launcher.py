import sys, getopt, os
import csv
import numpy as np
import benchmark
from random import randint, random, seed
from sklearn.model_selection import train_test_split
from math import ceil


_author_ = "Pierrick Calmels"


def main(argv):
    trainpath  = ''
    testpath   = ''
    outputname = ''
    noise = 0.0
    bags = 1
    c = 1
    stop_criterion = 0.01
    try:
        opts, args = getopt.getopt(argv, "ht:s:o:n:b:c:l:", ["trainpath=", "testpath=", "ofile=", "noise=", "bags="
            , "c=", "stop_criterion="])
    except getopt.GetoptError:
        print('launcher.py -t <train data path> -s <test path> -o <output name> -n <proportion of noisy points in the'
              ' dataset> -b <number of bags if needed> -c <decreasing age in Brownboost> '
              '-l <Browboost threshold to prevent divergence>')
        print('you may have only a datafile, if so, the dataset will be split into 80\%-20\%')
        sys.exit(2)
    for opt, arg in opts:
        print(opt)
        if opt == '-h':
            print(
                'launcher.py -t <train data path> -s <test path> -o <output name> -n <proportion of noisy points in the'
                ' dataset> -b <number of bags if needed> -c <decreasing age in Brownboost> '
                '-l <Browboost threshold to prevent divergence>')
            print('you may have only a datafile, if so, the dataset will be split into 80\%-20\%')
            sys.exit()
        elif opt in ("-t", "--trainpath"):
            trainpath = arg
        elif opt in ("-s", "--testpath"):
            testpath = arg
        elif opt in ("-o", "--ofile"):
            outputname = arg
        elif opt in ("-n", "--noise"):
            try:
                print("noise required {}".format(arg))
                noise = float(arg)
            except ValueError:
                print('error, please enter a float for randomness, 0.0 used')
        elif opt in ("-b", "--bags"):
            try:
                print("number of bags {}".format(arg))
                bags = int(arg)
            except ValueError:
                print('error, not a number given for bags, 1 used')
        elif opt in ("-c", "--c"):
            try:
                print("c {}".format(arg))
                c = float(arg)
            except ValueError:
                print('error, not a number given for stop criterion, 0.01 used')
        elif opt in ("-l", "--stop_criterion"):
            try:
                print("stop criterion {}".format(arg))
                stop_criterion = float(arg)
            except ValueError:
                print('error, not a number given for stop criterion, 0.01 used')

    return trainpath, testpath, outputname, noise, bags, c, stop_criterion


# Preprocess the data, loading it from a csv file
def preprocessor(trainpath, testpath, noise, label_position=-1, labels=None):
    with open(trainpath) as f:
        train_data = np.array(
            [[float(i) for i in line] for line in csv.reader(f, delimiter=",")])
        if label_position == 0:
            trainX = train_data[:, 1:]
            trainY = train_data[:, 0]
        elif label_position == -1:
            trainX = train_data[:, 0:label_position]
            trainY = train_data[:, label_position]
        else:
            trainX = train_data[:, 0:label_position].concat(train_data[:, label_position + 1:])
            trainY = train_data[:, label_position]
        # binarize the labels!
        if labels:
            trainY[trainY == labels[0]] = -1
            trainY[trainY == labels[1]] = 1
    f.close()

    x_train = randomize_data(trainX, noise)
    y_train = trainY
    print("Number of records: %d" % len(trainX))
    if testpath != '':
        with open(testpath) as tf:
            test_data = np.array([[float(i) for i in line] for line in csv.reader(tf, delimiter=",")])
            if label_position == 0:
                testX = test_data[:, 1:]
                testY = test_data[:, 0]
            elif label_position == -1:
                testX = test_data[:, 0:label_position]
                testY = test_data[:, label_position]
            else:
                testX = test_data[:, 0:label_position].concat(train_data[:, label_position + 1:])
                testY = test_data[:, label_position]
            # binarize the labels!
            if labels:
                testY[testY == labels[0]] = -1
                testY[testY == labels[1]] = 1
        tf.close()
        x_test = randomize_data(testX, noise)
        y_test = testY
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)
    return x_train, y_train, x_test, y_test


# randomizes a percentage of the training dataset
# for example : noise_percentage = 0.2 - makes 20% of the cells in the data random
def randomize_data(train_dataset, noise_percentage):
    seed(5)
    number_of_rows = train_dataset.shape[0]
    number_of_features = train_dataset.shape[1]
    number_of_cells = int(ceil(number_of_rows*number_of_features*noise_percentage))
    # print("num cells {}".format(number_of_cells))
    for i in range(number_of_cells):
        new_val = 1000.0*random()
        row = randint(0, number_of_rows-1)
        ft = randint(0, number_of_features-1)
        train_dataset[row][ft] = new_val
    return train_dataset


# # randomizes a data line uniformly, given the noise percentage
# def randomize_line(line, randomness, label_position):
#     p = uniform(0, 1)
#     if p < randomness:
#         if label_position == -1:
#             label_position = len(line)-1
#         while True:
#             index = randint(1, len(line)-1)
#             if index != label_position:
#                 val = random()
#                 line[index] = val
#                 break
#     return line


if __name__ == "__main__":
    trainpath, testpath, outputname, noise, bags, c, stop_criterion = main(sys.argv[1:])
    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open("results/{}.csv".format(outputname), "w") as f:
        f.write("c,stop criterion,bags,noise,classifier,accuracy,precision,recall,f1 score,learnT,testT \n")
    f.close()
    x_train, y_train, x_test, y_test = preprocessor(trainpath, testpath, noise, labels=[0, 1])
    ben = benchmark.Benchmark(x_train, y_train, x_test, y_test)
    ben.run_benchmark(outputname, noise, bags, c, stop_criterion)

    # to override the noise in parameter of the script and run the benchmark on different noise percentages
    # noises = [0.1, 0.15, 0.2, 0.25, 0.5]
    # cs = range(4, 12, 1)
    # for k in cs:
    #     for n in noises:
    #         x_train, y_train, x_test, y_test = preprocessor2(trainpath, testpath, n, labels=[0, 1])
    #         ben = benchmark.Benchmark(x_train, y_train, x_test, y_test)
    #         ben.run_benchmark(outputname, n, bags, c, stop_criterion)




