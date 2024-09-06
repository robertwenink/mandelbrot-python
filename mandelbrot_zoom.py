# Mandelbrot
# animeren: https://matplotlib.org/matplotblog/posts/animated-fractals/
# https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

####### IMPORT PACKAGES/LIBRARIES #########
import matplotlib.pyplot as plt
 
import time
from settings import ANIMATE, LIVEPLOTTING, RENDER, NR_FRAMES

from render import mandelbrotRender


if __name__ == "__main__":
    
    render = mandelbrotRender(NR_FRAMES, RENDER, LIVEPLOTTING)

    start = time.perf_counter()
    if ANIMATE:
        render.animate()
    else:
        render.image()
            
    print("Calculation & render time: {}".format(time.perf_counter() - start))
    # plt.pause(5)   
    plt.show()
