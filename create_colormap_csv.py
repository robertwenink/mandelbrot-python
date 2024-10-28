import numpy as np
import matplotlib.pyplot as plt

# Number of samples
# NB with 1024 we get repetitions in pairs, the max number is 512 in matplotlib, 
# except that the first and last are also repeated so actually 510, 
# which coincides with a scale of 256 colors (or values 0 to 255), now values 0 to 510;
# these are 511 integer values, correct for this at the cpp end to only have unique values
n = 510

# Get the matplotlib twilight colormap
cmap = plt.get_cmap('twilight', n)

# Create an array of RGB values, without alpha channel
colors = cmap(np.linspace(0, 1, n))[:, :3] 

# Save to csv
np.savetxt('twilight.csv', colors, delimiter=',')