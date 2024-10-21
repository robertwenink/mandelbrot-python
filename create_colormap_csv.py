import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n = 1024

# Get the matplotlib twilight colormap
cmap = plt.get_cmap('twilight', n)

# Create an array of RGB values, without alpha channel
colors = cmap(np.linspace(0, 1, n))[:, :3] 

# Save to csv
np.savetxt('twilight.csv', colors, delimiter=',')