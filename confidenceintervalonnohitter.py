import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import optimalparaters as op
import generatingbootstrapreplicates as gbp 

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = gbp.draw_bs_reps(op.nohitter_times, np.mean, 10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates,[2.5,97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()