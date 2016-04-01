import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
# We are going to use sklearn's SVM implementation

class tsvm:
    def __init__(self,x_t, y_t, x_u):
        self.x_train = x_t
        self.y_train = y_t
        self.x_unlabeled = x_u
        self.y_unlabeled = [0 for z in range(0, len(self.x_unlabeled))] 
    """ 
    The TSVM algorithm.
    C is the weight we should give to the labeled data's slacks variables.
    C_s is the weight we give to the unlabeled data's slack variables.
    num_p is the number of unlabeled examples we are allowed to assign to
    the "true" class.
    """
    def learn(C, C_s, num_p):
        clf = SVC()
        clf.fit(self.x_train, self.y_train)
        for pt in self.x_unlabeled:
            dists = clf.decision_function(self.x_unlabeled)
        #C_sn is how much weight we give to unlabeled negative examples
        C_sn = 10**-5
        C_sp = C_sn*num_p / (len(self.x_unlabeled)-num_p)

num_labeled = 3
num_unlabeled = 100
ax = [np.random.normal((2,1)) for x in range(0,num_labeled)]
for x in range(0, num_labeled):
    ax.append(np.random.normal(size=(2,1), loc=1))
X = np.array(ax)
ay = [0 for x in range(0, num_labeled)]
for x in range(0, num_labeled):
    ay.append(1)
y = np.array(ay)
au = [np.random.normal((2,1)) for x in range(0,num_unlabeled)]
for x in range(0, num_unlabeled):
    au.append(np.random.normal(size=(2,1), loc=1))
U = np.array(au)

T=tsvm(X, y, U)
