import parser, tsvm_sgt

def main():
    pen = parser.PenParser()
    X_train, Y_train, X_unlab, X_test, Y_test = pen.retrieve_pendigits_data(0.5)
    c = tsvm_sgt.tsvm(X_train, Y_train, X_unlab)
    # not sure what to do next

if __name__ == "__main__":
    main()


