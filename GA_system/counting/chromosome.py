import random
import numpy as np
class Chromosome(object):
    def __init__(self, j_dim, i_dim, ranges, upbound_of_SD):

        # individual = > 'thita', 'w', 'm', 'sd', 'adapt_value'
        self.theta = np.array([random.uniform(-1, 1)])
        self.weight = np.zeros(j_dim)
        self.means = np.zeros(j_dim*i_dim)
        self.sd = np.zeros(j_dim)
        self.adapt_value = None
        # w initialization
        for i in range(j_dim):
            self.weight[i] = random.uniform(-1, 1)
        # m initialization
        for i in range(j_dim*i_dim):
            self.means[i] = random.uniform(ranges[1], ranges[0])
        # sd initialization
        for i in range(j_dim):
            self.sd[i] = random.uniform(1/2 * upbound_of_SD, upbound_of_SD)
    def printmyself(self):
        print('theta', self.theta)
        print('weight', self.weight)
        print('means', self.means)
        print('sd', self.sd)
        print('adapt_value', self.adapt_value)
     
