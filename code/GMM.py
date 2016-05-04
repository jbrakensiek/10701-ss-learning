import numpy as np
class GMM:
	def __init__(self, x_t, y_t, x_u, num_categories):
		self.x_train = x_t
        self.y_train = y_t
        self.x_unlabeled = x_u
        self.y_unlabeled = [-1 for z in range(0, len(x_u))]
        self.y_dist = [1.0/num_categories for x in range(0, num_categories)]
        self.x_gaussians = [[[0.0, 1.0] for x in range(0, len(x_u[0]))] for y in range(0, num_categories)] # The parameters of the Gaussians

    def EM(self):
    	#EM reduces to: Given the y_unlabeled, update the model to find the 