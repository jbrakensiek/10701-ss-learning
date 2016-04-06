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


percent_lab = float(sys.argv[1])

#def main():
pen = parser.PenParser()
X_train, Y_train, X_unlab, X_test, Y_test = pen.retrieve_pendigits_data(percent_lab)
c = tsvm_sgt.tsvm(X_train, Y_train, X_unlab)
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

"""
if __name__ == "__main__":
main()
"""

