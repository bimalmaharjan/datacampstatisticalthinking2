import numpy as np
from generatingpermutationreplicates import draw_perm_reps
# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yay_dems(dems, reps):
    """Compute fraction of Democrat yay votes."""
    frac = np.sum(dems) / float(len(dems))
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yay_dems, 10000)

print perm_replicates
# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / float(len(perm_replicates))
print('p-value =', p)