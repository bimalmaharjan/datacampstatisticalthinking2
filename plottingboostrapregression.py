
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from linearregression import illiteracy, fertility
from pairsbootstrapofilliteracyfertilitydata import draw_bs_pairs_linreg

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(np.array(illiteracy),np.array(fertility),1000)
# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.scatter(illiteracy,fertility,marker='.')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()