import parser, tsvm_sgt, tsvm

def main():
    pen = parser.PenParser()
    X_train, Y_train, X_unlab, X_test, Y_test = pen.retrieve_pendigits_data(0.5)
    c = tsvm_sgt.tsvm(X_train, Y_train, X_unlab)

    #Task 1: Differentiate 1s from non-1s
    for i in range(0, len(Y_train)):
        if(Y_train[i] != 1):
            Y_train[i] = -1
    for i in range(0, len(Y_test)):
        if(Y_test[i] != 1):
            Y_test[i] = -1
    T = tsvm.TSVM(X_train, Y_train, X_unlab)
    frac = int(0.1*len(X_unlab))
    T.learn(1, 0.8, frac)
    X_fits = T.predict(X_train)
    correct = 0
    for i in range(0, len(X_train)):
        if(X_fits[i] == Y_train[i]):
            correct+=1
    print(float(correct) / len(X_train))
    output = T.predict(X_test)
    correct = 0

    for i in range(0, len(Y_test)):
        if(Y_test[i] == output[i]):
            correct+=1
    print(float(correct) / len(Y_test))

if __name__ == "__main__":
    main()


