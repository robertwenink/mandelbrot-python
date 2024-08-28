import numpy as np
from numba import njit, prange

from settings import MAX_ITS, X_RESOLUTIE, Y_RESOLUTIE


@njit(parallel=True, fastmath=True)
def mandelbrot_simple(x_cor, y_cor, max_its):
    """
    Escape time algorithm; simplest variant.
    Here x_corr are the real values, y_corr the imaginary in terms of the mandelbrot fractal
    """
    ESCAPE_RADIUS = 4 #
    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE),dtype="float64")

    for i in prange(X_RESOLUTIE):
        for j in prange(Y_RESOLUTIE):
            c = complex(x_cor[i],y_cor[j])  # complex coordinates / set
            z = complex(0, 0)               # complex answer
            n = 0                           # number of iterations
            for k in range(max_its):
                # this is the actual mandelbrot fractal formula! 
                # we let the calculation be handled by the 'complex' class
                z = z*z + c 

                n = n + 1
                if (abs(z) > ESCAPE_RADIUS):
                    break
            X[i,j] = n              
    return X


@njit(parallel=True, fastmath=True)
def mandelbrot(x_cor, y_cor, max_its):
    """
    Escape time algorithm; optimised variant.
    See https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
    Here x_corr are the real values, y_corr the imaginary in terms of the mandelbrot fractal
    """

    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE), dtype="float64")
    bailout = 4

    for i in prange(X_RESOLUTIE):
        for j in prange(Y_RESOLUTIE):
            # z = complex(0, 0)               # complex answer
            # complex answer z = x +yj
            x = 0.0
            y = 0.0
            
            # quadratised variants for the optimised version to limit multiplications
            x2 = 0.0
            y2 = 0.0
            w = 0.0

            # number of iterations
            n = 0                           
            while (x2 + y2 <= bailout and n < max_its):
                x = x2 - y2 + x_cor[i]
                y = w - x2 - y2 + y_cor[j]
                x2 = x * x
                y2 = y * y
                w = (x + y) * (x + y)
                n += 1

            # smooth coloring by making n fractional within its possible bounds based on closeness of |z| to the bailout
            # see https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
            # or https://blogen.pasithee.fr/2019/01/06/smooth-coloring-of-mandelbrot/ for an qualitative explanation
            if n < max_its:
                log_zn = np.log(x2 + y2) / 2 # in ln |z| because |z| of a complex number is just sqrt(x^2 + y^2) without its cross components
                nu = np.log(log_zn) / np.log(2)
                n_frac = n + 1 - nu
                X[i,j] = n_frac
            else:
                X[i,j] = n
    return X