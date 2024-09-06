

import os
import time
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from kernel import mandelbrot as mandelbrot, mandelbrot_gpu
from settings import *


#####################################
# Animation trajectory calculations #
#####################################

def get_trajectory_point(i):
    """
    Function to convert trajectory point from [x,y,zoom] -> [x,y,width,height],
    width and height based on the initial screen size and the zoom.
    """
    [x,y,zoom] = TRAJECTORY[i]
    traj_end_width = start_width / zoom
    traj_end_height = start_height / zoom
    return x, y, traj_end_width, traj_end_height


def get_max_its(current_frame_nr):
    """
    Linearly increase the max iteration count based on the zoom. 
    Higher zoom levels require deeper iterations.
    @param current_frame_nr, is a linear proxy for the zoom
    """

    current_frame_frac = current_frame_nr / NR_FRAMES

    min_its = 100

    # simple linear interpolation
    # based on: current_max_its = min_its + (MAX_ITS-min_its)*(current_zoom-min_zoom)/(max_zoom-min_zoom)
    # increase the x_scale variable to have a relatively higher rate of increase at the beginning and lower at the end of simulation.
    x_scale = 10
    y_scale = np.log(1+x_scale)
    current_max_its = min_its + (MAX_ITS-min_its)*np.log(1 + x_scale * current_frame_frac)/y_scale

    return np.floor(current_max_its)


#########################
## Render class #########
#########################

class mandelbrotRender:

    def __init__(self, nr_frames, render = True, liveplotting = True):
        self.nr_frames = nr_frames
        self.render = render
        self.liveplotting = liveplotting

        if self.render:
            self.init_figure()
            
        self.set_r_dim()
        self.set_trajectory_change_at_frame_number_list()


    def init_figure(self):
        # Create figure scladed for our given resolution
        self.fig, self.ax = plt.subplots(figsize=(X_RESOLUTION/100, Y_RESOLUTION/100),frameon=False) 

        # Stuff to make the image borderless             
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.ax.margins(0, 0)
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())

        # we willen geen grid of assenstelsel oid zien
        self.ax.axis("off")                                 

        # Create empty image, using a colorscheme, plot from bottom to top
        X = np.zeros((X_RESOLUTION, Y_RESOLUTION))    
        self.img = self.ax.imshow(X.T, cmap=CMAP, origin='lower') 

        # Access the underlying Tkinter window to position the window at the top-left corner (0, 0)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+0+0")

        # Create a cyclic normalization object
        # You can adjust vmin and vmax to control the range of values that get cycled through the colormap
        vmin, vmax = 0, 255  # Example range, adjust as needed for your specific application
        cyclic_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        self.img.set_norm(cyclic_norm)


        # als we liveplotting doen moeten we eerst de figure al tekenen!    
        if self.liveplotting or not ANIMATE:
            plt.draw()                                              
            plt.pause(0.001)   


    def set_r_dim(self):
        """
        Defining the ratios at which each frame consecutively changes in size; defined using the complete simulation range
        """
        
        _, _, _, start_height  = get_trajectory_point(0)
        _, _, _, end_height = get_trajectory_point(-1)

        # Each next frame should have the same rate of change in the width/height / size of the screen to have a uniform zoom feeling
            # Explanation: Stel we verplaatsen x, y, width en height lineair via interpolation afhankelijk van factor fac.
            # height en width tezamen vormen het oppervlak.
            # om de oppervlak verandering linear te laten verlopen moet je dus fac**(1/2) gebruiken
            # echter, stel je neemt 10 stappen van 0.1 tussen 1.01 en 0.01, 
            # is de eerste stap een verschil van 10% en de laatste stap van 99%
            # Het verloop moet dus relatief uniform zijn ook om visueel uniform te zijn, ofwel x^(n+1)=x^n * r met r constant.
            # we moeten dus een geometrische reeks hebben waarbij we voor k = self.nr_frames een r vinden waarvoor geldt dat
            # start*r^self.nr_frames = end ofwel r = kde wortel van end/start.
            # Verder moet x en y hetzelfde verlopen om overeen te komen!

        # NOTE equal in width and height
        # -1 because we apply it N-1 times for N frames
        self.r_dim = (end_height/start_height)**(1/(self.nr_frames-1)) 
        print(self.r_dim)


    def set_trajectory_change_at_frame_number_list(self):
        """ 
        Calculate at which iterations exactly we will be at the start of a new trajectory and so change the trajectory index.
        """
    
        self.trajectory_change_at_frame_number = []
        fac_temp = 1 
        j = 0
        for i in range(self.nr_frames):
            # fac is the inverse zoom here, and with respect to the start of simulation, not per trajectory as done later: 
            # start width or height multiplied with it will retrieve the width or height at zoom level xx
            fac_temp*= self.r_dim

            # if we have just crossed the zoomlevel of the current trajectory target, move to the next trajectory target
            if fac_temp <= 1/(TRAJECTORY[j][-1]):
                # indicate at which framenumber we will change the trajectory
                self.trajectory_change_at_frame_number.append(i)
                j += 1
            
            # if we are going to the last trajectory now, just break and add the last manually
            if j == (len(TRAJECTORY)-1):
                break   

        self.trajectory_change_at_frame_number.append(self.nr_frames)
       

    def next_trajectory_xy(self):
        """
        Generator for returning the next points and factor for interpolation and smooth trajectory

        @yield
        - traj_start_x, traj_start_y; x and y coordinates of the previous trajectory changepoint, for instance, the initial coords
        - traj_end_x, traj_end_y; x and y coordinates of the next trajectory changepoint
        - r_xy: the ratio used to converge the xy location from the one frame to the next
        - f: interpolation factor between the start and end of a trajectory (for the x and y location)
        - fac_xy: 1 + f
        """

        for i in range(len(TRAJECTORY)-1):
            traj_start_x, traj_start_y, _, _  = get_trajectory_point(i)
            traj_end_x, traj_end_y, _, _  = get_trajectory_point(i+1)

            # trajectory_change_at_frame_number (list): frame number at which we switch to the next trajecory target x, y and zoom
            traj_nr_frames = self.trajectory_change_at_frame_number[i+1] - self.trajectory_change_at_frame_number[i]

            # location ratio
            r_xy = self.r_dim**SMOOTHING_POWER # het ziet er iets soepeler uit als de locatie sneller convergeert dan de screen size

            # TODO A series of r^ni does not necessarily end up exactly at the zoom level of a new trajectory (very probably not)!
            # TODO So, we need to define a number f to compensate for this, to later correctly interpolate 
            # TODO (since there we will take the start x or y of the trajectory, not those of start of the complete simulation)

            # Below: with the smoothing_power applied (the effect of r_xy is stronger than globally needed), 
            # we need something that compensates for that in the interpolation, 
            # such that the endpoint in x and y correspond again compared to using the 'normal' r_dim at a trajectory switch.
            # we zoeken naar (1+f)*r^N = f , ofwel fac begint met 1+f eindigt met -f, en corrigeren hiervoor bij interpoleren
            # herschrijven geeft f = r^N/(1-r^N)
            rN = r_xy**(traj_nr_frames)
            
            f = rN/(1-rN)
            facxy = 1 + f

            yield traj_start_x, traj_start_y, traj_end_x, traj_end_y, r_xy, facxy, f


    def frame_helper(self):
        """
        Generator for the animation, returning iteration number, new centers and screen size.
        Defining the input for frame_builder
        """

        # get the start width and height
        _, _, width, height  = get_trajectory_point(0)

        # get the generator for the xy location, width and height
        gen = self.next_trajectory_xy()

        # interpolation function
        def intpol(start, end, fac, f):
            res = start * (fac-f) + end * ((1+f)-fac)
            return res

        j = 0
        for i in range(self.nr_frames):            
            if i == self.trajectory_change_at_frame_number[j] and not i == self.trajectory_change_at_frame_number[-1]:
                traj_start_x, traj_start_y, traj_end_x, traj_end_y, r_xy, facxy, f = next(gen)
                j += 1
                print("change of trajectory!")

            x = intpol(traj_start_x, traj_end_x, facxy, f)
            y = intpol(traj_start_y, traj_end_y, facxy, f)
            current_max_its = get_max_its(i)

            yield i, x, y, width, height, current_max_its

            # adjust parameters for next yield
            facxy *= r_xy
            width *= self.r_dim
            height *= self.r_dim
            
            # Print animation progress
            print(f"{(i+1)/self.nr_frames*100:.2f}% complete")
    

    def frame_builder(self, args):
        """
        Function that facilitates the animation.
        args: the output/yield of 'frames':  i, x, y, width, height
        """
        (x, y) = (args[1], args[2])
        (width, height) = (args[3], args[4])
        current_max_its = args[5]
        
        # Create a 'corrected' x and y linspace with sizes of the resolution and values within the mandelbrot domain of interest.
        x_cor = np.linspace(x-width/2,x+width/2,X_RESOLUTION, dtype="float64")
        y_cor = np.linspace(y-height/2,y+height/2,Y_RESOLUTION, dtype="float64")

        # main calculation
        t_0 = time.perf_counter()

        if GPU:
            X = mandelbrot_gpu(x_cor, y_cor, current_max_its)
        else:
            X = mandelbrot(x_cor, y_cor, current_max_its)

        print(f"Main done in: {time.perf_counter() - t_0:.3f}")
        
        if self.render:
            self.img.set_data(X.T)

            # flush events to speed up rendering
            if self.liveplotting:
                self.fig.canvas.flush_events()

            return [self.img]


    def animate(self):
        """
        Function that calls the matplotlib animation function, and saves the result to a gif.
        """

        if self.render:
            if FILE_FORMAT.lower() == "gif":        
                output_path = 'renders/mandelbrot.gif'
            else: 
                output_path = 'renders/mandelbrot.mp4'

            if os.path.exists(output_path):
                os.remove(output_path)

            # TODO https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
            anim = animation.FuncAnimation(self.fig, 
                                        lambda args : self.frame_builder(args), 
                                        frames=self.frame_helper, 
                                        interval=0, 
                                        blit=True,
                                        save_count = self.nr_frames,
                                        repeat = False
                                        )
            
            if FILE_FORMAT.lower() == "gif":        
                anim.save(output_path, writer='pillow')
            else:
                # TODO make fps related to the amount of zoom
                anim.save(output_path, writer='ffmpeg', fps=30, dpi=80)
        
        else:
            self.frames_only()

        # this print because 'frames' does not print the last after the yield
        print("{:.2f}% complete".format(100))


    def frames_only(self):
        """
        Function that runs all the frames, but calculations only, no saveing or rendering.
        """
        frame_helper = self.frame_helper()
        for f in range(NR_FRAMES):
            args = next(frame_helper)
            img = self.frame_builder(args)


    def image(self):
        traj_end_x, traj_end_y, traj_end_width, zoom_height = get_trajectory_point(0)
        
        start = time.perf_counter()

        x_cor = np.linspace(traj_end_x-traj_end_width/2,traj_end_x+traj_end_width/2,X_RESOLUTION, dtype="float64")
        y_cor = np.linspace(traj_end_y-zoom_height/2,traj_end_y+zoom_height/2,Y_RESOLUTION, dtype="float64")
        
        # main calculation
        X = mandelbrot(x_cor,y_cor,MAX_ITS)

        print("Calculation time: {}".format(time.perf_counter() - start))
        
        if self.render:
            self.img.set_data(X.T)
            plt.draw()   
            plt.pause(0.01)   


            output_path = 'renders/mandelbrot.png'

            if os.path.exists(output_path):
                os.remove(output_path)

            plt.imsave(output_path, X.T, cmap=CMAP, origin='lower')


            
