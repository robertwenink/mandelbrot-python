###########################
##### Coloring scheme #####
###########################

import numpy as np

from settings import X_RESOLUTIE, Y_RESOLUTIE


def histogramColoring(iteration_counts):
    """
    Coloring schema that uses the escape algorithm iteration counts but relates them to all other counts, 
    for smoother behaviour independent of the maximum chosen number of iterations.
    """

    # pass 1, retrieve counts (iteration_counts)

    # pass 2, create bins filled with frequencies of the mandelbrot iteration counts; is of maximum size MAX_ITS
    (unique_its, inverese_indices, frequencies) = np.unique(iteration_counts, return_inverse = True, return_counts=True)

    # pass 3, count total, excluding the counts of pixels that reached bailout
    total = np.sum(frequencies[:-1])

    # pass 4, determine color/hue
    # basically, create a cumulative map of frequencies, divide by total, and index using location of a count in unique_its
    color_bins = np.cumsum(frequencies)/total
    color = np.take(color_bins,inverese_indices)

    return np.reshape(color,(X_RESOLUTIE,Y_RESOLUTIE))

    ## this is more like the implementation on wikipedia for step 4
    # color = np.zeros_like(iteration_counts)
    # for index, count in np.ndenumerate(iteration_counts):
    #     for idx, i in enumerate(unique_its):
    #         if i > count:
    #             break
    #         color[index] += frequencies[idx]/total     