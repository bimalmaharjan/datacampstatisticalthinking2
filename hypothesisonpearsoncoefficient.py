import numpy as np
import sys
ecdfpath = '../'
sys.path.append(ecdfpath)
from statisticalthinkingpart1.covariance import  pearson_r
from linearregression import illiteracy,fertility

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) / float(len(perm_replicates))
print('p-val =', p)