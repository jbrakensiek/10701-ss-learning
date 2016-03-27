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
        self.numtotal = self.numunlabeled + self.numpoints
        self.unlabeled_data = unlabelvec
        self.labels = outputvec
        self.totaldata = np.concatenate((inputvec, unlabeledvec))
        self.labelcomplete = np.concatenate((outputvec, np.zeros(numunlabeled)))
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

    ''' Code for spectral graph transducer. Implemented as described in
        Joachims 2003. Optimizing this is shown equivalent to optimizing
        TSVMs in Joachims 2003.
    '''

    ''' This function generates the normalized Laplacian matrix of the
        similarity graph.
    '''
    def generateGraph():
        self.laplacian = np.matrix(np.zeros([self.numtotal, self.numtotal]))
        #find a faster way to do the following with numpy functions
        for i in range(self.numtotal):
            ithvec = self.totaldata[i]
            sum_ithrow = 0
            for j in range(i):
                jthvec = self.totaldata[j]
                distancevec = ithvec - jthvec
                self.laplacian[i][j] = -1 * math.exp(-1 * np.dot(distancevec,\
                    distancevec))
                self.laplacian[j][i] = self.laplacian[i][j]
                sum_ithrow += laplacian[i][j]
            self.laplacian[i] /= sum_ithrow
            self.laplacian[i][i] = 1

    def spectralgraphtransducer():
        generateGraph()

        #using eigh over eig because it is optimized to work on symmetric
        #matrices
        laplace_eigval, laplace_eigvec = np.linalg.eigh(self.laplacian)

        #some eigenvectors/eigenvalues should potentially be discarded
        #figure out which ones!

        laplace_eigvec = laplace_eigvec.T
        #code to sort two arrays together taken from stackoverflow
        indexes = range(len(laplace_eigval))
        indexes.sort(key = laplace_eigval.__getitem__)
        sorted_laplace_eigval = map(laplace_eigval.__getitem__, indexes)
        sorted_laplace_eigvec = map(laplace_eigvec.__getitem__, indexes)
        sorted_laplace_eigvec = sorted_laplace_eigvec.T

        #don't have intuition for the following loop. Included it here because
        #the paper suggested it.
        dimension_eigval = len(sorted_laplace_eigval)
        for i in range(dimension_eigval):
            sorted_laplace_eigval[i] = (i + 1) * (i + 1)
        #the next line because np.sum is faster than any for loop
        sumlabels = np.sum(self.labels)

        #length(outputvec) - this number is always even
        numpositive = ((numpoints + sumlabels) / 2)
        numnegative = ((numpoints - sumlabels) / 2)

        #the names of the parameters are from the paper, I will add
        #explanations for what these parameters mean soon.
        gammaplus = sqrt(float(numnegative)/float(numpositive))
        gammaminus = -1 * sqrt(float(numpositive)/float(numnegative))

        gammavec = np.zeros(self.numtotal)
        for i in range(self.numpoints):
            if (self.labels[i] == 1):
                gammavec[i] = gammaplus
            else:
                gammavec[i] = gammaminus

        #more stuff to put here
        costsynthesis = np.zeros(self.numtotal)
        for i in range(self.numtotal):
            if (self.labels[i] == 1):
                costsynthesis[i] = float(numpoints) / (2.0 * float(numpositive))
            else:
                costsynthesis[i] = float(numpoints) / (2.0 * float(numnegative))
        costsynthesis = np.diag(costsynthesis) #this is the 'C' in the paper.

        eigvalmat = np.diag(sorted_laplace_eigval)
        tradeoff_param = 1 #I set it to 1 arbitrarily, tuning is necessary for
                           #better performance. This is the 'c' in the paper.

        #same eigenvectors as Laplacian, eigenvalues changed to
        #gammaplus, gammaminus or 0
        costweigh_eig = tradeoff_param * np.dot(sorted_laplace_eigvec.T,\
            costsynthesis)

        #the paper defines a matrix called G. matG represents that matrix
        matG = eigvalmat + np.dot(costweigh_eig, sorted_laplace_eigvec)

        #the paper defines a vector called b, vecb represents that vector
        vecb = np.dot(costweigh_eig, gammavec.T)
        
        numrowsinG = len(matG)
        negidentity = np.diag(-1 * np.ones(numrowsinG))

        #The following patch of code is to replicate the matrix whose
        #smallest eigenvalue is of interest to us
        topPart = np.concatenate((matG.T, negidentity)).T
        leftbottom = (float(-1)/float(numtotal)) * np.dot(vecb.T, vecb)
        #matG is the rightbottom
        bottomPart = np.concatenate((leftbottom.T, matG.t)).T
        interestmatrix = np.concatenate((topPart, bottomPart))

        ''' - Matrix not symmetric so not sure if eigenvalues are even real
            - Since I am using eigvals, even if the eigenvalues are real,
              eigvals might return complex numbers with a small imaginary
              part because of roundoff error. Hence the map.
        '''
        eigvalsOfInterest = np.linalg.eigvals(interestmatrix)
        eigvalsOfInterest = map(lambda x: sqrt(float((np.real(x) ** 2) +\
            (np.imag(x) ** 2))), eigvalsOfInterest)
        #eigenstar is the lambda^* variable
        eigenstar = min(eigvalsOfInterest)
        
        predictionhelper = np.dot(np.dot(sorted_laplace_eigvec,\
            np.linalg.inv(matG + (eigenstar * negidentity))), bvec)

        #find a better threshold
        threshold = 0.5 * float(gammaplus + gammaminus)

        predictions = map(lambda x: -1 if x < threshold else 1,\
            predictionhelper)
