import numpy as np
import scipy
import csv

class PenParser:
    def __init__(self):
        self.UJI_PENCHARS_DATA = "../data/ujipenchars2.txt"
        self.PENDIGITS_TRAINING_DATA = "../data/pendigits.tra"
        self.PENDIGITS_TESTING_DATA = "../data/pendigits.tes"
    
    def parse_pendigits_csv(self, file_name):
        """
        Returns data, answers
        data is a numpy 2darray of floats of shape (num_samples, 16)
        answers is a numpy array of ints of shape (num_samples)
        """
        reader = csv.reader(open(file_name))
        lines = list()
        for row in reader:
            lines.append(row)
            
        data = np.zeros((len(lines), 16))
        answers = np.zeros(len(lines))

        for i in range(len(lines)):
            print lines[i]
            data[i,] = map(float, lines[i][:-1])
            answers[i] = int(lines[i][-1])

        return data, answers

    def parse_UJI_penchars(self, file_name):
        """ To be implemented """
        pass
    
    def retrieve_pendigits_data(self, labeled_fraction):
        """
        labeled_fraction (float between 0 and 1) is the fraction of labeled
        training data whose label should be kept

        Returns
        X_train_labeled - numpy 2darray of shape (num_labeled_samples, 16)
        Y_train_labeled - numpy 1darray of shape (num_labeled_samples)
        X_train_unlabeled - numpy 2darray of shape (num_unlabeled_samples, 16)
        X_test - numpy 2darray of shape (num_test_samples, 16)
        Y_test - numpy 1darray of shape (num_labeled_samples)
        """
        
        data_train, answers_train = self.parse_pendigits_csv(self.PENDIGITS_TRAINING_DATA)
        data_test, answers_test = self.parse_pendigits_csv(self.PENDIGITS_TESTING_DATA)
        
        num_train = data_train.shape[0]
        num_labeled = int(num_train * labeled_fraction)

        # should we randomize this?
        X_train_labeled = data_train[0:num_labeled, ]
        Y_train_labeled = answers_train[0:num_labeled]
        X_train_unlabeled = data_train[num_labeled:num_train, ]
        X_test = data_test
        Y_test = answers_test
        
        return X_train_labeled, Y_train_labeled, X_train_unlabeled, X_test, Y_test
