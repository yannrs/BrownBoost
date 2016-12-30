import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import brownboost
from copy import deepcopy


_author_ = "Pierrick Calmels"


class Benchmark:
    def __init__(self, train_data_x, train_data_y, test_data_x, test_data_y):
        self.x_train = train_data_x
        self.y_train = train_data_y
        self.x_test  = test_data_x
        self.y_test  = test_data_y

    def run_benchmark(self, output_name, noisiness, bags, c, stop_criterion):
        classif_list = [
            "AdaBoostClassifier",
            "BrownBoostClassifier"
        ]

        # with open("results/messidor/new_test/results_classif"+str(c)+"/" + str(output_name) + ".csv", "a") as f:
        with open("results/{}.csv".format(output_name), "a") as f:
            for classifier in classif_list:
                # metrics_f = open("results/" + str(classifier) + "_metrics.csv", "a")
                results = self.benchmark(classifier, bags, c, stop_criterion)
                res = str(c)+","+str(stop_criterion)+","+str(bags)+","+str(noisiness) + "," + classifier + "," + ",".join(results)+"\n"
                f.write(res)
                # f.write(str(results)+"\n")
                #metrics_f.close()
            f.close()
        return 0

    def benchmark(self, classifier_name, bags, c, stop_criterion, file=None):
        trainx = deepcopy(self.x_train)
        testx = deepcopy(self.x_test)
        trainy = deepcopy(self.y_train)
        testy = deepcopy(self.y_test)

        clf = classifier_factory(classifier_name, bags, c, stop_criterion)
        begin_time = time.time()
        clf.fit(trainx, trainy)
        learning_time = time.time() - begin_time

        begin_time = time.time()
        prediction = clf.predict(testx)
        testing_time = time.time() - begin_time

        # GET PREDICTION
        results_file = open("results/prediction_"+classifier_name+".csv", 'w')
        results_file.write("actual label, prediction\n")
        for y, p in zip(testy, prediction):
            results_file.write(str(y)+","+str(p)+"\n")
        results_file.close()
        # END PREDICTION

        # report = classification_report(testy, prediction, digits=4)
        # file.write(report)
        accuracy = accuracy_score(testy, prediction)
        precision = precision_score(testy, prediction, pos_label=1, labels=[-1, 1])
        recall = recall_score(testy, prediction, pos_label=1, labels=[-1, 1])
        clf_f1_score = f1_score(testy, prediction, pos_label=1, labels=[-1, 1])
        print("mean accuracy : %.4f \n" % accuracy)
        # file.write("clf {0} mean accuracy: {1}\n".format(classifier_name, accuracy))
        return str(accuracy), str(precision), str(recall), str(clf_f1_score), str(learning_time), str(testing_time)


def classifier_factory(name, bags=2, c=1, stop_criterion=0.01, weak_learner=DecisionTreeClassifier(max_depth=1)):
    return {
        "BrownBoostClassifier": brownboost.BrownBoost(weak_learner, c=c, stop_criterion=stop_criterion),
        # "AdaBoostClassifier": AdaBoostClassifier(base_estimator=weak_learner, n_estimators=bags),
        "AdaBoostClassifier": AdaBoostClassifier(weak_learner,
                                                 algorithm="SAMME.R",
                                                 n_estimators=bags
                                                 ),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=4, criterion='gini', min_samples_split=2)
    }.get(name)
