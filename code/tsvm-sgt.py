import numpy as np
import math

class tsvm:
    ''' inputvec is a 2D numpy array, where the i-th entry of inputvec
        is the i-th datapoint. outputvec is a 1D numpy array, where the
        i-th entry of outputvec is the ouput label of the i-th point in
        inputvec. unlabeledvec is the unlabeled data as a 2D numpy array.
        Note: there is a difference between np.array and np.matrix.
        We will use both in this code, so let's be clear about the distinction
        to avoid confusion
    '''
    def __init__(inputvec, outputvec, unlabeledvec):
        (self.numpoints, self.datadimension) = inputvec.shape
        self.numunlabeled = unlabeledvec.shape
        self.labeled_data = inputvec
        self.unlabeled_data = unlabelvec
        self.labels = outputvec
        self.totaldata = np.concatenate(inputvec, unlabeledvec)
        self.labelcomplete = np.concatenate(outputvec, np.zeros(numunlabeled))

    self.laplacian = np.matrix([])
    ''' Boolean value necessary to specify is objective function should be
        minimized or maximized.
        The way the optimization function is designed, it can be expressed
        in the form s^{T}As + <b, s> where s is the vector of variables
        that we optimize over, A is some matrix, b is some vector.
        So the objective to minimize shall be represented
        as A, b. 'A' shall be called quadmat, 'b' shall be called
        linvec.
        The constraints also need some representation. For the labeled data
        a bunch of linear equations must be greater than or equal to something
        else. And the last constraint that makes the QP nonconvex is
        each y^* in {-1, +1}. The linear constraints for the labeled data
        can be succincted to Mx >= c.
        s is [weights, bias, slack variables]

        Generalize code so that it takes in the above matrices and runs the
        desired training algorithm on the code.
    '''

    ''' code for spectral graph transducer. Implemented as described in
        Joachims 2003. Optimizing this is shown to be equivalent to optimizing
        TSVMs.
    '''

    ''' This function generates the normalized Laplacian matrix of the
        similarity graph.
    '''
    def generateGraph():
        self.laplacian = np.zeros([self.numpoints + self.numunlabeled,\
            self.numpoints + numunlabeled])
        #find a faster way to do the following with numpy functions
        for i in range(self.numpoints + self.numlabeled):
            ithvec = self.totaldata[i]
            sum_ithrow = 0
            for j in range(self.numpoints + self.numlabeled):
                jthvec = self.totaldata[j]
                distancevec = ithvec - jthvec
                self.laplacian[i][j] = -1 * math.exp(-1 * np.dot(distancevec,\
                    distancevec))
                sum_ithrow += laplacian[i][j]
            self.laplacian[i] /= sum_ithrow
            self.laplacian[i][i] = 1

    def spectralgraphtransducer():
        generateGraph()
        #the next line because np.sum is faster than any for loop
        sumlabels = np.sum(self.labels)
        #length(outputvec) - this number is always even
        numpositive = ((numpoints + sumlabels) / 2)
        numnegative = ((numpoints - sumlabels) / 2)
        #the names of the parameters are from the paper, I will add
        #explanations for what these parameters mean soon.
        gammaplus = sqrt(float(numnegative)/float(numpositive))
        gammaminus = sqrt(float(numnegative)/float(numpositive))
        #more stuff to put here
