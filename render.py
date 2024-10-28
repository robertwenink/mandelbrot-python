

import csv
import os
import time
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from kernel import mandelbrot as mandelbrot, mandelbrot_gpu
from trajectory import Trajectory



#########################
## Render class #########
#########################

class mandelbrotRender(Trajectory):

    def __init__(self, settings):
        super().__init__(settings)

        attributes = ["x_resolution","y_resolution","min_its","max_its","fps","gpu",
                      "colormap_name","output_filename","animate","render","liveplotting"]
        
        for attr in attributes:
            setattr(self, attr, getattr(settings, attr))

        self.total_calculation_time = 0

        # matplotlib only saves what it actually 'sees', so take a high dpi ...
        self.dpi = 200  
        self.agg = False

        if self.render:
            self.init_figure()
            


    def init_figure(self):
        """
        Initialise the figure to draw to if rendering is on.
        If the non-interactive agg backend is not used, 
        after all this weird setup, the image still gets compressed 26px vertically... 
        """
        
        # use the non-interactive backend for a bit less overhead
        if not self.liveplotting and self.animate:
            self.agg = True
            matplotlib.use('Agg')

        # Create figure scaled for our given resolution
        self.fig, self.ax = plt.subplots(figsize=(self.x_resolution/self.dpi, self.y_resolution/self.dpi), frameon=False) 
        self.fig.set_size_inches(self.x_resolution/self.dpi, self.y_resolution/self.dpi)
        self.fig.set_dpi(self.dpi)  # Set DPI to match
        self.ax.set_position([0, 0, 1, 1])

        # Stuff to make the image borderless             
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.ax.margins(0, 0)
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())

        # we willen geen grid of assenstelsel oid zien
        self.ax.axis("off")                                 

        # Create empty image, using a colorscheme, plot from bottom to top
        X = np.zeros((self.x_resolution, self.y_resolution))
        self.img = self.ax.imshow(X.T, cmap=self.colormap_name, origin='lower', aspect='auto') 

        if not self.agg:
            # Access the underlying Tkinter window to position the window at the top-left corner (0, 0)
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry("+0+0")
            
            # remove the toolbar (which gets included in the figsize and distorts the actual image ....)
            fig_manager.toolbar.pack_forget() 

        # Create a cyclic normalization object
        # You can adjust vmin and vmax to control the range of values that get cycled through the colormap
        vmin, vmax = 0, 255  # Example range, adjust as needed for your specific application
        cyclic_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        self.img.set_norm(cyclic_norm)

        # if we liveplot the video, or when animating the image, we should already draw the figure at setup    
        if not self.agg:        
            plt.draw()                                              
            plt.pause(0.001)   


    def get_current_max_its(self, current_frame_nr):
        """
        Linearly increase the max iteration count based on the zoom. 
        Higher zoom levels require deeper iterations.
        @param current_frame_nr, is a linear proxy for the zoom
        """

        current_frame_frac = current_frame_nr / self.nr_frames

        # simple linear interpolation
        # based on: current_max_its = min_its + (self.max_its-min_its)*(current_zoom-min_zoom)/(max_zoom-min_zoom)
        # increase the x_scale variable to have a relatively higher rate of increase at the beginning and lower at the end of simulation.
        x_scale = 10
        y_scale = np.log(1+x_scale)
        current_max_its = self.min_its + (self.max_its-self.min_its)*np.log(1 + x_scale * current_frame_frac)/y_scale

        return np.floor(current_max_its)

    def frame_helper(self):
        """
        Generator for the animation, returning iteration number, new centers and screen size.
        Defining the input for frame_builder
        """

        # get the start width and height
        _, _, width, height  = self.get_trajectory_point(0)

        # get the generator for the xy location, width and height
        gen = self.next_trajectory_xy()

        j = 0
        for i in range(self.nr_frames):  
            t_0 = time.perf_counter()

            # get new trajectory simulation parameters at a trajectory change!
            if i == self.trajectory_change_at_frame_numbers[j] and not i == self.trajectory_change_at_frame_numbers[-1]:
                traj_start_x, traj_start_y, traj_end_x, traj_end_y, r_xy, f_xy, f_er = next(gen)
                j += 1
                print("change of trajectory!")

            x = self.corrected_interpolation(traj_start_x, traj_end_x, f_xy, f_er)
            y = self.corrected_interpolation(traj_start_y, traj_end_y, f_xy, f_er)
            current_max_its = self.get_current_max_its(i)

            # time this part of the calculation
            self.total_calculation_time += time.perf_counter() - t_0

            yield i, x, y, width, height, current_max_its

            # adjust parameters for next yield
            f_xy *= r_xy
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

        t_0 = time.perf_counter()
        
        # Create a 'corrected' x and y linspace with sizes of the resolution and values within the mandelbrot domain of interest.
        x_cor = np.linspace(x-width/2,x+width/2, self.x_resolution, dtype="float64")
        y_cor = np.linspace(y-height/2,y+height/2, self.y_resolution, dtype="float64")

        # main calculation
        if self.gpu:
            X = mandelbrot_gpu(x_cor, y_cor, current_max_its)
        else:
            X = mandelbrot(x_cor, y_cor, current_max_its)

        # record the calculation time
        calc_time = time.perf_counter() - t_0
        self.total_calculation_time += calc_time
        print(f"Main done in: {calc_time:.3f}s, for x={x}, y={y}")
        
        if self.render:
            self.img.set_data(X.T)

            # flush events to speed up rendering
            if not self.agg:
                self.fig.canvas.flush_events()

            return [self.img]


    def run(self):
        """
        Function that calls the matplotlib animation function, and saves the result to a gif.
        """

        if self.render:
            output_path = f'renders/{self.output_filename}'

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
            
            if os.path.splitext(self.output_filename)[1].lower() == ".gif":       
                anim.save(output_path, writer='pillow')
            else:
                # TODO make fps related to the rate of zoom for an automatic smooth feeling?
                anim.save(output_path, writer='ffmpeg', fps=self.fps, dpi=self.dpi)
        else:
            self.frames_only()

        # this print because 'frames' does not print the last after the yield
        print("{:.2f}% complete".format(100))


    def frames_only(self):
        """
        Function that runs all the frames, but calculations only, no saveing or rendering.
        """
        frame_helper = self.frame_helper()
        for f in range(self.nr_frames):
            args = next(frame_helper)
            img = self.frame_builder(args)


    def image(self):
        traj_end_x, traj_end_y, target_width, target_heigth = self.get_trajectory_point(0)
        
        start = time.perf_counter()

        x_cor = np.linspace(traj_end_x-target_width/2,traj_end_x+target_width/2,self.x_resolution, dtype="float64")
        y_cor = np.linspace(traj_end_y-target_heigth/2,traj_end_y+target_heigth/2,self.y_resolution, dtype="float64")
        
        if self.gpu:
            X = mandelbrot_gpu(x_cor, y_cor, self.max_its)
            print("Calculation time: {} s".format(time.perf_counter() - start))
        else:
            # main calculation
            X = mandelbrot(x_cor,y_cor,self.max_its)
            print("Calculation time (numba not initialised): {} s".format(time.perf_counter() - start))

            second = time.perf_counter()
            
            # main calculation, second time, to correct the timings for the numba initialisation
            x_cor = np.linspace(traj_end_x-target_width/2,traj_end_x+target_width/2,self.x_resolution, dtype="float64")
            y_cor = np.linspace(traj_end_y-target_heigth/2,traj_end_y+target_heigth/2,self.y_resolution, dtype="float64")
            X = mandelbrot(x_cor,y_cor,self.max_its)

            print("Calculation time (numba initialised): {} s".format(time.perf_counter() - second))
        
        if self.render:
            self.img.set_data(X.T)
            plt.draw()   
            plt.pause(0.01)   


            output_path = f'renders/{self.output_name}.png'

            if os.path.exists(output_path):
                os.remove(output_path)

            plt.imsave(output_path, X.T, cmap=self.colormap_name, origin='lower')


    def log_performance(self, total_elapsed_seconds, log_filename="performance_log.csv"):
        """
        Log the time performance measurements
        """

        # Calculate average times per iteration
        average_time_per_iteration = total_elapsed_seconds / self.nr_frames
        calc_time = self.total_calculation_time 
        render_time = total_elapsed_seconds - self.total_calculation_time
        average_calc_time_per_iteration = calc_time / self.nr_frames

        # Check if the file is empty to add header
        file_is_empty = not os.path.exists(log_filename) or os.stat(log_filename).st_size == 0

        # Open log file in append mode
        with open(log_filename, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)

            # Write header if the file is empty
            if file_is_empty:
                log_writer.writerow([
                    "Elapsed time (s)", "Calculation time (s)", "Render time (s)", 
                    "Number of frames", "Resolution (nx x ny)", 
                    "Avg time per Iteration (s)", "Avg calc time per Iteration (s)", "GPU", "liveplotting", "comments"
                ]) 

            # Write data in CSV format
            log_writer.writerow([
                f"{total_elapsed_seconds:.2f}",
                f"{calc_time:.2f}",
                f"{render_time:.2f}",
                self.nr_frames,
                f"{self.x_resolution}x{self.y_resolution}",
                f"{average_time_per_iteration:.4f}",
                f"{average_calc_time_per_iteration:.4f}", 
                self.gpu,
                self.liveplotting and self.render
            ])

        print("Total simulation time: {}".format(total_elapsed_seconds))
        print(f"Performance data logged to {log_filename}")
        
                
