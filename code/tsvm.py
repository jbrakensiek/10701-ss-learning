import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
# We are going to use sklearn's SVM implementation

class tsvm:
    def __init__(self,x_t, y_t, x_u):
        self.x_train = x_t
        self.y_train = y_t
        self.x_unlabeled = x_u
        #We initialize to all -1s at first
        self.y_unlabeled = [-1 for z in range(0, len(self.x_unlabeled))] 
    """ 
    The TSVM algorithm.
    C is the weight we should give to the labeled data's slacks variables.
    C_s is the weight we give to the unlabeled data's slack variables.
    num_p is the number of unlabeled examples we are allowed to assign to
    the "true" class.
    """
    def learn(self,C, C_s, num_p):
        clf = svm.SVC()
        weights = []
        for x in range(0, len(self.x_train)):
            weights.append(C)
        clf.fit(self.x_train, self.y_train, sample_weight=weights)
        u_dists = []
        for i in range(0, len(self.x_unlabeled)):
            u_dists.append([clf.decision_function([self.x_unlabeled[i]])[0], i])
        #u_dists contains the margin sizes of the unlabeled points.
        #We take the largest sizes and classify them as 1
        u_dists.sort()
        print(u_dists)
        for x in range(len(u_dists)-1, int(max([len(u_dists)-1-num_p, 0])), -1):
            self.y_unlabeled[u_dists[x][1]] = 1
        #C_sn is how much weight we give to unlabeled negative examples
        C_sn = 10**-5
        C_sp = C_sn*num_p / (len(self.x_unlabeled)-num_p)

        #initialize the unlabeled weights
        u_weights = []
        for c in self.y_unlabeled:
            if(c == 1):
                u_weights.append(C_sp)
            else:
                u_weights.append(C_sn)
        while(C_sn < C or C_sp < C):
            #fit the unlabeled data with their labels
            clf.fit(np.concatenate((self.x_train, self.x_unlabeled), axis=0),\
                    np.concatenate((self.y_train, self.y_unlabeled), axis=0),\
                    sample_weight = np.concatenate((weights, u_weights), axis=0))
            u_dists = []
            #Same as above. We find the margins of the unlabeled data
            for i in range(0, len(self.x_unlabeled)):
                u_dists.append([clf.decision_function([self.x_unlabeled[i]])[0], i])

            C_sn = min(C, 2*C_sn)
            C_sp = min(C, 2*C_sp)


num_labeled = 3
num_unlabeled = 3
offset = 5
#put num_labelled points drawn from each Gaussian into ax
ax = [np.random.normal((2,1)) for x in range(0,num_labeled)]
for x in range(0, num_labeled):
    ax.append(np.random.normal(size=(2,1), loc=offset))
X = np.array(ax)
#put the labels into ay
ay = [-1 for x in range(0, num_labeled)]
for x in range(0, num_labeled):
    ay.append(1)
y = np.array(ay)
#Generate 2*num_unlabeled points, num_unlabeled from each Gaussian, into au
au = [np.random.normal((2,1)) for x in range(0,num_unlabeled)]
for x in range(0, num_unlabeled):
    au.append(np.random.normal(size=(2,1), loc=offset))
U = np.array(au)

Xp = []
Yp = []
C = []
for x in range(0, num_labeled):
    C.append('g')
for x in range(0, num_labeled):
    C.append('b')
for x in range(0, 2*num_unlabeled):
    C.append('r')

for x in X:
    Xp.append(x[0])
    Yp.append(x[1])

for x in U:
    Xp.append(x[0])
    Yp.append(x[1])

print(X)
print(y)
print(U)
T=tsvm(X, y, U)
T.learn(0.5, 0.3, 0.5*num_unlabeled)

plt.scatter(Xp, Yp, c=C)
plt.show()
