class Trajectory:
    """
    Class that defines the trajectory in x, y, height, and width during the mandelbrot video animation.
    """

    def __init__(self, settings):
        self.trajectory_vector = settings.trajectory_vector
        self.nr_frames = settings.nr_frames
        self.start_width = settings.start_width
        self.start_height = settings.start_height
        self.xy_smoothing_power = settings.xy_smoothing_power

        self.set_r_dim()
        self.set_trajectory_change_at_frame_numbers()


    def get_trajectory_point(self, i):
        """
        Function to convert trajectory point from [x,y,zoom] -> [x,y,width,height],
        width and height based on the initial screen size and the zoom.
        """
        [x,y,zoom] = self.trajectory_vector[i]
        target_width = self.start_width / zoom
        target_heigth = self.start_height / zoom
        return x, y, target_width, target_heigth


    def corrected_interpolation(self, x0, x1, f_xy, f_err):
        """
        Interpolation function that takes correction factor f into account, 
        correcting for the difference between self.r_dim and the smoothed self.r_dim**xy_smoothing_power

        x0, x1: start and end values between which the interpolation takes place. 
        f_xy: starts at 1, gets multiplied by r_xy after each iteration.
        f_err:  constant, equal to 1-r_xy^N, or
                the error wrt to interpolation factor 0 after a series of N iterations
        """
        f_corr = (f_err - f_xy) / (f_err - 1)
        res = x0 * f_corr + x1 * (1 - f_corr) 
        return res

    def set_r_dim(self):
        """
        Defining the ratios at which each frame consecutively changes in size; defined using the complete simulation range
        """
        
        _, _, _, start_height  = self.get_trajectory_point(0)
        _, _, _, end_height = self.get_trajectory_point(-1)

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


    def set_trajectory_change_at_frame_numbers(self):
        """ 
        Calculate at which iterations exactly we will be at the start of a new trajectory and so change the trajectory index.
        """
    
        self.trajectory_change_at_frame_numbers = []
    
        # fac_temp = the series of r_dim^i
        fac_temp = 1 

        j = 0
        for i in range(self.nr_frames):
            # fac is the inverse zoom here, and with respect to the start of simulation, not per trajectory as done later: 
            # start width or height multiplied with it will retrieve the width or height at zoom level xx
            fac_temp*= self.r_dim

            # if we have just crossed the zoomlevel of the current trajectory target, move to the next trajectory target
            if fac_temp <= 1/(self.trajectory_vector[j][-1]):
                # indicate at which framenumber we will change the trajectory
                self.trajectory_change_at_frame_numbers.append(i)
                j += 1
            
            # if we are going to the last trajectory now, just break and add the last manually
            if j == (len(self.trajectory_vector)-1):
                break   

        self.trajectory_change_at_frame_numbers.append(self.nr_frames)
       

    def next_trajectory_xy(self):
        """
        Generator for returning the next points and factor for interpolation and smooth trajectory

        @yield
        - traj_start_x, traj_start_y; x and y coordinates of the previous trajectory changepoint, for instance, the initial coords
        - traj_end_x, traj_end_y; x and y coordinates of the next trajectory changepoint
        - r_xy: the ratio used to converge the xy location from the one frame to the next
        - f_xy: uncorrected interpolation factor between the start and end of a trajectory (for the x and y location)
        - f_err:  constant, equal to 1-r_xy^N, or
            the error wrt to interpolation factor 0 after a series of N iterations
        """
        
        # xy_smoothing_power: let the location converge faster than the screensize (which is calculated with just r_dim); this looks smoother.
        # i.e. r_xy < r_dim so we converge faster to 0
        r_xy = self.r_dim**self.xy_smoothing_power

        for i in range(len(self.trajectory_vector)-1):
            traj_start_x, traj_start_y, _, _  = self.get_trajectory_point(i)
            traj_end_x, traj_end_y, _, _  = self.get_trajectory_point(i+1)

            # trajectory_change_at_frame_numbers: frame number at which we switch to the next trajecory target x, y and zoom
            traj_nr_frames = self.trajectory_change_at_frame_numbers[i+1] - self.trajectory_change_at_frame_numbers[i]

            # Below: with the smoothing_power applied (the effect of r_xy is stronger than globally needed), 
            # we need something that compensates for that in the interpolation, 
            # such that the endpoint in x and y correspond again compared to using the 'normal' r_dim at a trajectory switch.
            # Normally, we interpolate between x0 and x1 *of a trajectory* using an inverse interpolation factor fac in [1,0].
            # Remember that we defined traj_nr_frames using r_dim, so 1 and 1*r_dim^traj_nr_frames should exactly match this [1,0]
            # since r_xy is based on r_dim, the same holds for r_xy 
            # (Note that even without the smoothing_power we would require a correction!).

            # Lets represent the mismatch at the end of the (sub)series using 'f_err', this would mean we have f_uncorrected in [1, f_err].
            # where f_err = 1 * r_xy^N - 0 = r_xy^N. 
            # At the interpolation, we will normalize f_xy to the [0,1] range and use: 
            # f_corr = f_err - f_xy / (f_err - 1)

            rN = r_xy**(traj_nr_frames)
            f_err = rN/(1-rN)

            # reset the base interpolation factor for each new trajectory.
            f_xy = 1

            # NOTE A series of r^ni does not necessarily end up exactly at the zoom level of a new trajectory (very probably not)!
            # This is however fine, zoom will be close enough, while we keep the zoom rate steady, which is more important

            yield traj_start_x, traj_start_y, traj_end_x, traj_end_y, r_xy, f_xy, f_err