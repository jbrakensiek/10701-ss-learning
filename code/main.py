import parser, tsvm_sgt, tsvm
import numpy as np
import sys

def predict_digit(digit, X_train, Y_train, X_unlab, X_test, Y_test):
    print("Predicting "+ str(digit))
    for i in range(0, len(Y_train)):
        if(Y_train[i] != digit):
            Y_train[i] = -1
        else:
            Y_train[i] = 1
    for i in range(0, len(Y_test)):
        if(Y_test[i] != digit):
            Y_test[i] = -1
        else:
            Y_test[i] = 1
    T = tsvm.TSVM(X_train, Y_train, X_unlab)
    frac = int(0.1*len(X_unlab))
    T.learn(1, 0.9, frac, 10**-2)
    print("Test accuracy: " + str(T.score(X_test, Y_test)))
    return T.decision_function(X_test)

def predict_digit_2(digitclass, X_train, Y_train, X_unlab, X_test, Y_test):
    print("Predicting a class")
    for i in range(0, len(Y_train)):
        if (Y_train[i] not in digitclass):
            Y_train[i] = -1
        else:
            Y_train[i] = 1
    for i in range(0, len(Y_test)):
        if (Y_test[i] not in digitclass):
            Y_test[i] = -1
        else:
            Y_test[i] = 1
    T = tsvm.TSVM(X_train, Y_train, X_unlab)
    frac = int(0.1 * len(X_unlab))
    print("reaches here")
    T.learn(1, 0.9, frac, 10**-2)
    print("Test accuracy: " + str(T.score(X_test, Y_test)))
    return T.decision_function(X_test)

percent_lab = float(sys.argv[1])
which_classifier = int(sys.argv[2])
#def main():
pen = parser.PenParser()
X_train, Y_train, X_unlab, X_test, Y_test = pen.retrieve_pendigits_data(percent_lab)
c = tsvm_sgt.tsvm(X_train, Y_train, X_unlab)
#c.spectralgraphtransducer()

if (which_classifier == 1):
    boundaries = []
    for dig in range(0, 10):
        boundaries.append(predict_digit\
                (dig, X_train, np.matrix.copy(Y_train), X_unlab, X_test, np.matrix.copy(Y_test)))

    predictions = []
    for i in range(0, len(Y_test)):
        pred = 0
        max_boundary = boundaries[0][i]
        for dig in range(1, 10):
            if(boundaries[dig][i] > max_boundary):
                pred = dig
                max_boundary = boundaries[dig][i]
        predictions.append(pred)

    correct = 0
    for i in range(0, len(Y_test)):
        if(Y_test[i] == predictions[i]):
            correct += 1

    print(float(correct) / len(Y_test))
    confusion_matrix = np.zeros((10, 10))
    for a in range(0, len(predictions)):
        confusion_matrix[int(predictions[a])][int(Y_test[a])]+=1

    print(confusion_matrix)

else:
    boundaries = []
    digitclass1 = {1,2,3,4,5,8}
    digitclass2 = {1,2,8,9,0}
    digitclass3 = {1,2,5,6,9}
    digitclass4 = {1,3,5,7,9}
    digitclasses = [digitclass1, digitclass2, digitclass3, digitclass4]
    #try something like codewords of Hamming codes for digitclasses?
    for i in range(0, 4):
        raw_preds = predict_digit_2\
            (digitclasses[i], X_train, np.matrix.copy(Y_train), X_unlab,\
             X_test, np.matrix.copy(Y_test))
        toAdd = map(lambda x: 1 if (x > 0) else -1, raw_preds)
        boundaries.append(toAdd)

    predictions = []
    for i in range(0, len(Y_test)):
        hamming_best = 4
        best_pred = -1
        for dig in range(0, 10):
            hamming_dist = 0
            for j in range(0, 4):
                if ((boundaries[j][i] == 1 and (dig not in digitclasses[i]))\
                    or (boundaries[j][i] == -1 and (dig in digitclasses[i]))):
                    hamming_dist += 1
            if (hamming_dist < hamming_best):
                best_pred = dig
                hamming_best = hamming_dist
        predictions.append(best_pred)
    
    correct = 0
    for i in range(0, len(Y_test)):
        if (Y_test[i] == predictions[i]):
            correct += 1

    print(float(correct) / float(len(Y_test)))

"""
if __name__ == "__main__":
main()
"""

