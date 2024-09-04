# Mandelbrot
# example:  https://godhalakshmi.medium.com/a-simple-introduction-to-the-world-of-fractals-using-python-c8cb859bfd6d
# animeren: https://matplotlib.org/matplotblog/posts/animated-fractals/
# TODO: https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

####### IMPORT PACKAGES/LIBRARIES #########
import matplotlib.pyplot as plt
 
import time
from settings import ANIMATE, LIVEPLOTTING, MAKE_IMAGE, NR_FRAMES

from render import mandelbrotRender


if __name__ == "__main__":
    start = time.perf_counter()
    
    render = mandelbrotRender(NR_FRAMES, LIVEPLOTTING)
    if ANIMATE:
        render.animate()
    elif MAKE_IMAGE:
        render.image()
        
        plt.draw()
        plt.pause(10)
    else:
        render.frames_only()
    
    print("Calculation & render time: {}".format(time.perf_counter() - start))
