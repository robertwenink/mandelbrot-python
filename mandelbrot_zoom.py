# Mandelbrot
# animeren: https://matplotlib.org/matplotblog/posts/animated-fractals/
# https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

####### IMPORT PACKAGES/LIBRARIES #########
import matplotlib.pyplot as plt
 
import time
from settings import Settings

from render import mandelbrotRender


if __name__ == "__main__":
    settings = Settings()

    render = mandelbrotRender(settings)

    start = time.perf_counter()
    if settings.animate:
        render.run()

        render.log_performance(time.perf_counter() - start)
    else:
        render.image()    
        print("Calculation & render time: {}".format(time.perf_counter() - start))
    
    plt.pause(5)   
    # plt.show()
