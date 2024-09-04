import numpy as np
from numba import njit, prange

import pyopencl as cl
ctx = cl.create_some_context(interactive=False, answers=["0"])

from settings import MAX_ITS, X_RESOLUTIE, Y_RESOLUTIE


@njit(parallel=True, fastmath=True)
def mandelbrot_simple(x_cor, y_cor, max_its):
    """
    Escape time algorithm; simplest variant.
    Here x_corr are the real values, y_corr the imaginary in terms of the mandelbrot fractal
    """
    ESCAPE_RADIUS = 4 #
    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE),dtype="double64")

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

                # Take the modulus for cyclic coloring
                X[i,j] = n_frac % 255
            else:
                # Set in the set to darkest color in twilight cmap (middle of range)
                X[i,j] = 127.5
    return X


# https://gist.github.com/jfpuget/60e07a82dece69b011bb
def mandelbrot_gpu_v1(q, maxiter):

    global ctx
    
    queue = cl.CommandQueue(ctx)
    
    output = np.empty(q.shape, dtype=np.double)

    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global double2 *q,
                     __global double *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        double real = q[gid].x;
        double imag = q[gid].y;
        output[gid] = 0;
        for(int n = 0; n < maxiter; n++) {
            double real2 = real*real, imag2 = imag*imag;
            if (real2 + imag2 > 4.0f){
                double log_zn = log(real2 + imag2) / 2;
                double n_frac = n + 1 - log(log_zn) / log(2.0);

                // Take the modulus for cyclic coloring
                output[gid] = fmod(n_frac, 255.0);
                return;
            }
            imag = 2* real*imag + q[gid].y;
            real = real2 - imag2 + q[gid].x;
        }
        
        output[gid] = 127.5;
        return;
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()
    
    return output

def mandelbrot_gpu_v2(q, maxiter):

    global ctx
    
    queue = cl.CommandQueue(ctx)
    
    output = np.empty(q.shape, dtype=np.double)

    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global double2 *q,
                     __global double *output, ushort const max_its)
    {   
        int gid = get_global_id(0);
        output[gid] = 0;
        double x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0, w = 0.0;
        int n = 0;
        while(x2 + y2 <= 4.0f && n < max_its){
            x = x2 - y2 + q[gid].x;
            y = w - x2 - y2 + q[gid].y;
            x2 = x * x;
            y2 = y * y;
            w = (x + y) * (x + y);
            n += 1;
        }

        if (n < max_its) {
            double log_zn = log(x2 + y2) / 2;
            double n_frac = n + 1 - log(log_zn) / log(2.0);

            output[gid] = fmod(n_frac, 255.0);
            return;
        } else {
            output[gid] = 127.5;
            return;
        }
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()
    
    return output
            

def mandelbrot_gpu(x_cor, y_cor, max_its):
    c = np.add.outer(y_cor * 1j, x_cor).ravel('F')
    # c = x_cor + y_cor[:,None]*1j
    # c = c.flatten('F')
    
    # v2 is slightly slower for gpu, maybe because of the extra variables.
    X = mandelbrot_gpu_v1(c, max_its)
    X = X.reshape((X_RESOLUTIE,Y_RESOLUTIE))
    return X