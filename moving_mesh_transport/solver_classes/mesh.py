import numpy as np
from numba import int64, float64
import numba
from numba.experimental import jitclass
import math
from .functions import problem_identifier 
#from functions import problem_identifier 
from .mesh_functions import set_func, _interp1d
#from mesh_functions import set_func, _interp1d
# import quadpy
import numpy.polynomial as nply
from .functions import converging_time_function, converging_r
from scipy.special import roots_legendre
from .mesh_functions import boundary_source_init_func_outside
#from mesh_functions import boundary_source_init_func_outside
from .functions import quadrature
import numba as nb
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))


#################################################################################################
data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('tfinal', float64),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("moving", int64),
        ("move_type", int64[:]),
        ("edges", float64[:]),
        ("edges0", float64[:]),
        ("Dedges", float64[:]),
        ("N_space", int64),
        ('middlebin', int64),
        ('sidebin', int64),
        ('speed', float64),
        ('Dedges_const', float64[:]),
        ('source_type', int64[:]),
        ('thick', int64), 
        ('move_func', int64),
        ('debugging', int64),
        ('wave_loc_array', float64[:,:,:]),
        ('delta_t', float64),
        ('tactual', float64),
        ('told', float64),
        ('index_old', int64),
        ('right_speed', float64),
        ('left_speed', float64),
        ('test_dimensional_rhs', int64),
        ('move_factor', float64),
        ('T_wave_speed', float64),
        ('pad', float64),
        ('follower_speed', float64),
        ('leader_speed', float64),
        ('span_speed', float64),
        ('thick_quad', float64[:]),
        ('middlebin', int64),
        ('sidebin', int64),
        ('leader_pad', float64),
        ('packet_leader_speed', float64),
        ('thick_quad_edge', float64[:]),
        ('t0', float64),
        ('edges0_2', float64[:]),
        ('c1s', float64[:]),
        ('finite_domain', int64),
        ('domain_width', float64),
        ('mesh_stopped', int64),
        ('vnaught', float64),
        ('boundary_on', int64[:]),
        ('vv0', float64),
        ('t0', float64),
        ('geometry', nb.typeof(params_default)),
        ('l', float64),
        ('c2s', float64[:]),
        ('sigma_func', nb.typeof(params_default)),
        ('shift', float64)
        # ('problem_type', int64)
        ]
#################################################################################################


# Really need a better mesh function 
@jitclass(data)
class mesh_class(object):
    def __init__(self, N_space, x0, tfinal, moving, move_type, source_type, edge_v,
     thick, move_factor, wave_loc_array, pad, leader_pad, thick_quad, thick_quad_edge, 
     finite_domain, domain_width, fake_sedov_v, boundary_on, t0, geometry, sigma_func):
        
        self.debugging = True
        self.test_dimensional_rhs = False
        self.sigma_func = sigma_func
        self.pad = pad
        self.tfinal = tfinal
        self.N_space = N_space
        self.x0 = x0
        self.moving = moving
        self.move_type = np.array(list(move_type), dtype = np.int64)
        self.edges = np.zeros(N_space+1)
        self.edges0 = np.zeros(N_space+1)
        self.Dedges = np.zeros(N_space+1)
        self.l = move_factor
        self.N_space = N_space
        self.speed = edge_v
        self.geometry = geometry

        self.move_factor = move_factor
        if self.test_dimensional_rhs == True:
            self.speed = 299.98

        # print('mesh edge velocity: ', edge_v)
        self.source_type = np.array(list(source_type), dtype = np.int64)

        if self.move_type[0] == True:
            self.move_func = 0 # simple linear
        elif self.move_type[1] == True:
            self.move_func = 1 # thick square source move
            # print('thick square source edge estimation mesh')
        elif self.move_type[2] == True:
            self.move_func = 2 # sqrt t static
        elif self.move_type[3] == True:
            self.move_func = 3
        
        # self.problem_type = problem_identifier(self.source_typem, self.x0)
        self.thick = thick
        self.wave_loc_array = wave_loc_array
        # self.smooth_wave_loc_array
        # for count, element in enumerate(self.wave_loc_array[0, 3, :]):
        #     if element < self.x0:
        #         self.wave_loc_array[0, 3, count] = self.x0 + 1e-8
        self.thick_quad = thick_quad
        self.thick_quad_edge = thick_quad_edge

        # print(self.wave_loc_array[0,2,-1], 'wave final location')

        self.sidebin = int(self.N_space/4)
        self.middlebin = int(self.N_space/2)
        
        self.tactual = -1.
        self.told = 0.0
        self.index_old = 0
        self.T_wave_speed = 0.0
        self.follower_speed = 0.0
        self.leader_speed = 0.0
        self.span_speed = 0.0
        self.leader_pad = leader_pad
        self.t0 = t0
        # print(self.t0, 't0')
        self.finite_domain = finite_domain
        if self.finite_domain == True:
            print('finite domain')
        self.domain_width = domain_width
        
        self.boundary_on = np.array(list(boundary_on), dtype = np.int64)

        self.mesh_stopped = False
        self.vnaught = fake_sedov_v

        if fake_sedov_v != 0 and np.all(self.source_type == 0):
            self.speed = fake_sedov_v
            # print('speed is ', self.speed)
        self.shift = 0.0
        self.initialize_mesh()
    
    def check_tangle(self, t):
        if ((self.edges[1:] - self.edges[0:-1]) <=1e-14).any():
            print("###############################################################################")
            print(self.edges)
            for ix, xx in enumerate(self.edges[:-1]):
                if xx > self.edges[ix+1]:
                    print(ix, 'index')
                    print(xx, self.edges[ix+1], 'tangled edges')
                    print('t=', t)
                    print("###############################################################################")
            raise ValueError('The mesh is tanlged. ')


    def move(self, t):

        # print(self.edges)pr
        """
        Called each time the rhs moves the mesh. Changes edges and Dedges
        """
        # print(self.edges)
        # if self.moving == True:
        """
        This mode moves all of the edges at a constant speed,
        linearly increasing from 0 to the wavespeed
        """
        self.told = t
        self.check_tangle(t)
        if self.moving == True:
            # if self.source_type[1] == 1 or self.source_type[2] == 1:
                # if t > 10.0:
                #     self.Dedges = self.edges/self.edges[-1] * self.speed

            if self.source_type[0] == 1 or self.source_type[0]==2:

                self.edges = self.edges0 + self.Dedges_const*t



            elif self.source_type[1] == 1:
                if self.finite_domain == True and t == self.tfinal:

                # print(self.edges0[-1] + t * self.speed, "###### final edge ######")
                
                # if self.edges0[-1] + t * self.speed > 5 and self.finite_domain == True:
                    self.edges = np.linspace(-self.domain_width/2, self.domain_width/2, self.N_space+1)
                    

                elif (self.finite_domain == True) and (self.edges[-1] >= self.domain_width/2 or abs(self.edges[-1]-self.domain_width/2)<=1e-2) and self.geometry['slab']==True:
                    self.edges = self.edges
                    self.Dedges = self.Dedges_const*0
                else:
                    # print(self.edges0, 'edges0')

                    self.edges = self.edges0 + self.Dedges_const*t
                    self.Dedges = self.Dedges_const

                    # print(self.Dedges_const*t, 'dedges times t')

           

            elif (self.source_type[2] == 1):
            #or self.source_type[1] == 1):
                # self.finite_domain = True # what is the deal with this?
                if (self.finite_domain == True) and (self.edges[-1] >= self.domain_width/2 or abs(self.edges[-1]-self.domain_width/2)<=1e-2):
                        #assert(0)          # For testing if this if statements evaluates to true
                        self.edges = self.edges
                        self.Dedges = self.Dedges_const*0
                        print('stopping')
                    # if t == self.tfinal:
                    #     self.edges = np.linspace(-5, 5, self.N_space+1)
                    #     self.Dedges = self.Dedges_const*0
                    # else:
                else:
                    if self.move_func == 0:
                            if t >= self.t0:
                                self.move_middle_edges(t)
                                tnew1 = t - self.t0 

                            ### uncomment this to go back to the old mesh

                            # self.edges = self.edges0_2 + self.Dedges_const * (t-self.t0)
                            # self.Dedges = self.Dedges_const


                            ### uncomment this for constant vel. 

                            # self.edges = self.edges0_2 + self.Dedges * (t-self.t0)

                            ### uncomment this for acceleration case

                                self.edges = 0.5 * self.c1s * (tnew1) ** 2 + self.Dedges_const * tnew1 + self.edges0_2
                                self.Dedges = self.c1s * tnew1 + self.Dedges_const


                            elif (t < self.t0):
                            
                                self.edges = self.edges0 + self.Dedges*t


                            # self.Dedges = self.Dedges_const
                            

                    # else:

                    #         self.edges = self.edges0 + self.Dedges*t


                    elif self.move_func == 1: 
                            """
                            This mode has the wavefront tracking edges moving at a constant speed
                            and interior edges tracking the diffusive wave
                            """
                            # self.thick_square_moving_func(t)
                            self.thick_square_moving_func_2(t)
                    

                    elif self.move_func == 2:
                            self.square_source_static_func_sqrt_t(t)


                    else:
                            print("no move function selected")
                            assert(0)
            elif np.all(self.source_type == 0):
                # print(self.Dedges)
                if self.move_type[0] ==1:
                    self.edges = self.edges0 + self.Dedges*t
                elif self.move_type[1] == 1:
                    # assert(0)
                    self.converging_move2(t)
                    # assert 0  dx = 

            # if self.debugging == True:
            #     for itest in range(self.edges.size()):
            #         if self.edges[itest] != np.sort(self.edges)[itest]:
            #             print("crossed edges")
            #             assert(0)
    def smooth_wave_loc_array(self):
        for ix in range(0,self.wave_loc_array[0,3,:].size-1):
            if self.wave_loc_array[0,3,ix] < self.wave_loc_array[0,3,ix +1]:
                self.wave_loc_array[0,3,ix] = self.wave_loc_array[0,3,ix +1]

    def move_middle_edges(self,t):
        middlebin = int(self.N_space/2)
        sidebin = int(middlebin/2)
        if self.Dedges[sidebin] == 0:
            self.edges0_2 = self.edges0 + self.Dedges_const * self.t0
            # final_pos = self.edges0[-1] + self.Dedges[-1] * self.tfinal
            final_pos = self.pad
            # final_pos = self.x0 + self.pad
            if self.geometry['slab'] == True:
                final_array = np.linspace(-final_pos, final_pos, self.N_space + 1)
            elif self.geometry['sphere'] == True:
                final_array = np.linspace(0, final_pos, self.N_space + 1)
            # print(final_array, 'final array')

            # constant velocity
            # new_Dedges = (final_array - self.edges0_2) / (self.tfinal-self.t0)
            # self.Dedges = new_Dedges
            # self.Dedges[sidebin:sidebin+middlebin+1] = 0    




            #### constant acceleration ###

            # print(self.Dedges_const, 'const edges')
            # print(self.tfinal, 'tfinal')
            # print(self.edges0_2, 'second edges0')
            tnew = self.tfinal - self.t0
            # print(self.t0, 't0 in move middle' )
            self.c1s = 2 * (self.Dedges_const * (self.t0) -self.tfinal * self.Dedges_const - self.edges0_2 + final_array) / ((self.t0-self.tfinal)**2)       

    

    def thick_wave_loc_and_deriv_finder(self, t):
        
        interpolated_wave_locations = _interp1d(np.ones(self.wave_loc_array[0,3,:].size)*t, self.wave_loc_array[0,0,:], self.wave_loc_array[0,3,:], np.zeros(self.wave_loc_array[0,3,:].size))
        # interpolated_wave_locations = np.interp(t, self.wave_loc_array[0,0,:], self.wave_loc_array[0,3,:] )

        # derivative = (interpolated_wave_locations[1] - interpolated_wave_locations[0])/delta_t_2

        edges = np.copy(self.edges)
        if t == 0 or  interpolated_wave_locations[0] < self.x0:
            edges = self.edges0
        else:
            edges[-1] = interpolated_wave_locations[0] + self.leader_pad
            if edges[-1] < self.edges0[-1]:
                edges[-1] = self.edges0[-1]
            edges[-2] = interpolated_wave_locations[0] + self.pad
            if edges[-2] < self.edges0[-2]:
                edges[-2] = self.edges0[-2]
            edges[-3] = interpolated_wave_locations[0] 
            if edges[-3] < self.edges0[-3]:
                edges[-3] = self.edges0[-3]
            edges[-4] = interpolated_wave_locations[0]  - self.pad
            if edges[-4] < self.edges0[-4]:
                edges[-4] = self.edges0[-4]

        
            edges[self.sidebin + self.middlebin: self.N_space-3] = np.linspace(self.x0, edges[-4], self.sidebin - 2)[:-1] 
            edges[0:self.sidebin] =  - np.flip(np.copy(edges[self.sidebin+self.middlebin+1:]))
        return edges

    def thick_square_moving_func_2(self, t):
        delta_t = 1e-7
        self.edges = self.thick_wave_loc_and_deriv_finder(t)
        edges_new = self.thick_wave_loc_and_deriv_finder(t + delta_t)

        self.Dedges = (edges_new - self.edges) / delta_t

        if self.edges[-3] < self.edges[-4]:
            print('crossed')

    def recalculate_wavespeed(self, t):
        sidebin = int(self.N_space/4)
        T_index = -2
        index = np.searchsorted(self.wave_loc_array[0,0,:], t)
        # pad = self.edges[int(self.N_space/2)+1] - self.edges[int(self.N_space/2)
        # print(abs(self.edges[-2]-self.wave_loc_array[0,3,index+1]), 'T wave from mesh edge')
        if self.debugging == True:
            if index >0:
                if not (self.wave_loc_array[0,0,index-1] <= t <= self.wave_loc_array[0,0,index+1]):
                    print('indexing error')
        if index != self.index_old:
            # print('calculating wavespeeds') s
            T_wave_location = self.wave_loc_array[0,3,index+1]
            # print(self.pad, 'pad')
            # print(index, 'index')
            # print(T_wave_location, 'T wave loc')
            self.delta_t = self.wave_loc_array[0,0,index+1] - t
            self.right_speed = (self.wave_loc_array[0,2,index+1]  - self.edges[-1])/self.delta_t
            self.T_wave_speed = (T_wave_location - self.edges[-2])/self.delta_t
            # print(T_wave_location, 't edge is moving to')
            self.leader_speed = (T_wave_location + self.leader_pad - self.edges[-1])/self.delta_t
            self.packet_leader_speed = (T_wave_location + self.pad - self.edges[-2])/self.delta_t
            # print(T_wave_location + self.pad, 'leader edge is moving to')
            self.follower_speed = (T_wave_location - self.pad - self.edges[-3])/self.delta_t

            last_follower_edge_loc = self.edges[-3] + self.Dedges_const[-3] * self.follower_speed * self.delta_t
            dx_span = (last_follower_edge_loc - self.x0) / (sidebin/2)  
            self.span_speed = (last_follower_edge_loc - dx_span - self.edges[-int(sidebin-2)])/self.delta_t
        
        self.index_old = index
        # print(self.edges)

        # print(self.delta_t, 'delta t')
        # print(self.leader_speed, 'leader')
        # print(self.T_wave_speed, "T speed")
        # print(self.follower_speed, 'follower')
        # # print(self.span_speed, 'span')
        # print(self.T_wave_speed, 't wave s')
        # print(self.leader_speed, 'leader s')
        # print(index, 'index')
        if self.T_wave_speed > self.leader_speed:
            print("speed problem")
            print(self.pad, 'pad')

      
    
        # if abs(self.edges[T_index] + self.Dedges_const[T_index] * self.T_wave_speed * self.delta_t - (self.edges[T_index -1] + self.Dedges_const[T_index-1] * self.T_wave_speed * self.delta_t)) <= 1e-12:
        #     #catching up
        #     print('catching up')
        #     self.T_wave_speed = (self.edges[(T_index)-1] - 0.0005 - self.edges[(T_index)])/self.delta_t

        
        if self.debugging == True:
            if abs(t - self.wave_loc_array[0,0,index+1]) < 1e-5:
                print('checking location')
                print(self.wave_loc_array[0,3,index+1] - self.edges[-2], 'T wave difference')
                print(self.wave_loc_array[0,2,index+1]  - self.edges[-1], 'right edge difference')
        

        if self.right_speed < 0.0:
            self.right_speed = 0.0
        if self.T_wave_speed < 0.0:
            # print('negative t speed')
            self.T_wave_speed = 0.0
        if self.follower_speed < 0.0:
            self.follower_speed = 0.0
        if self.leader_speed < 0.0:
            self.leader_speed = 0.0
        if self.span_speed < 0.0:
            self.span_speed = 0.0
        
        
        # print(self.edges[-2], "|", self.wave_loc_array[0,3,index+1])


    

            

    def thick_gaussian_static_init_func(self):
        # if abs(self.wave_loc_array[0, 2, -1]) > 5:
        if self.move_func == 1:
            right_edge = self.wave_loc_array[0,3,-1] + self.pad
        elif self.move_func == 0:
            right_edge = self.x0
        print(self.move_func, 'move_func')
        print(right_edge, 'right edge')
        # else:
            # right_edge = self.x0 + self.tfinal
        
        # if right_edge < self.x0:
            # right_edge = self.x0 + self.tfinal

        self.edges = np.linspace(-right_edge, right_edge, self.N_space + 1)
        self.Dedges = self.edges * 0


    def simple_thick_square_init_func(self):
        # does not accomodate moving mesh edges
        
        # wave_edge = self.wave_loc_array[0,2,index+1]
        wave_edge = self.wave_loc_array[0,2,-1] + self.pad

        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-8
        left = np.linspace(-wave_edge, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, wave_edge, sidebin + 1)
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed * 0 
 

    def square_source_static_func_sqrt_t(self, t):
        # only to be used to estimate the wavespeed
        move_factor = self.move_factor

        if t > 1e-10:
            sqrt_t = math.sqrt(t)
        else:
            sqrt_t = math.sqrt(1e-10)

        # move the interior edges
        self.Dedges = self.Dedges_const * move_factor * 0.5 / sqrt_t
        self.edges = self.edges0 + self.Dedges_const * move_factor * sqrt_t

    

        # move the wavefront edges
        # Commented code below moves the exterior edges at constant speed. Problematic because other edges pass them
        # self.Dedges[0] = self.Dedges_const[0]
        # self.Dedges[-1] = self.Dedges_const[-1]
        # self.edges[0] = self.edges0[0] + self.Dedges[0]*t
        # self.edges[-1] = self.edges0[-1] + self.Dedges[-1]*t
        # self.Dedges[0] = self.Dedges_const[0] * move_factor * 0.5 / sqrt_t
        # self.edges[0] = self.edges0[0] + self.Dedges_const[0] * move_factor * sqrt_t
        # print(self.edges0[0], 'x0')
        # print(self.Dedges_const[0]*move_factor*sqrt_t, 'f')
        # self.Dedges[-1] = self.Dedges_const[-1] * move_factor * 0.5 / sqrt_t
        # self.edges[-1] = self.edges0[-1] + self.Dedges_const[-1] * move_factor * sqrt_t

    ####### Initialization functions ########


    def simple_moving_init_func(self):
            if self.geometry['slab'] == True:
                self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
            elif self.geometry['sphere'] == True:
                self.edges = np.linspace(0, self.x0, self.N_space+1)
                self.edges0 = self.edges
            self.Dedges = self.edges/self.edges[-1] * self.speed
            self.Dedges_const = self.Dedges
            if self.source_type[0] == 2:
                self.edges += 0.01

    def shell_source(self):
        if self.shift == 0:
            print('shell source function')
            dx = 1e-5
            N_inside = int(self.N_space/2 + 1)
            edges_inside = np.linspace(0, self.x0, N_inside+1)
            N_outside = int(self.N_space + 1 - N_inside )
            edges_outside = np.linspace(self.x0, self.x0 + dx, N_outside)
            self.edges = np.concatenate((edges_inside, edges_outside[1:]))
            self.edges0 = self.edges
            assert(self.edges.size == self.N_space + 1)
            self.Dedges = np.zeros(self.N_space + 1)

            self.Dedges[N_inside + 1:] = (self.edges[N_inside + 1:] - self.x0)/(self.edges[-1] - self.x0) * self.speed
            self.Dedges_const = self.Dedges
            print(self.Dedges_const, 'dedges')
        else:
            x02 = self.x0 + self.shift
            x03 = self.x0 - self.shift
            self.edges = np.linspace(-x03 - self.tfinal * self.speed, x02 + self.tfinal * self.speed, self.N_space + 1 )
            closest_left = np.argmin(np.abs(self.edges + x03))
            closest_right = np.argmin(np.abs(self.edges - x02))
            self.edges[closest_left] = x03
            self.edges[closest_right] = x02
            self.edges = np.sort(self.edges)
            self.Dedges = self.edges * 0
            self.Dedges_const = self.Dedges
            print(self.edges, 'edges0')

        

    # def thick_square_moving_func(self, t):
    #     middlebin = int(self.N_space/2)
    #     sidebin = int(self.N_space/4)
    #     self.recalculate_wavespeed(t)
    #     little_delta_t = t-self.told

    #     # self.Dedges[0:sidebin/2] = self.Dedges_const[0:sidebin/2] * self.right_speed
    #     # self.Dedges[0:int(sidebin/4 + 1)] = self.Dedges_const[0:int(sidebin/4 + 1)] * self.leader_speed
    #     # self.Dedges[int(sidebin/4 + 1)] = self.Dedges_const[int(sidebin/4 + 1)] * self.T_wave_speed
    #     # self.Dedges[int(sidebin/4 + 2):int(sidebin/2 + 1)] = self.Dedges_const[int(sidebin/4 + 2):int(sidebin/2 + 1)] * self.follower_speed
    #     # self.Dedges[int(sidebin/2 + 2):sidebin] = self.Dedges_const[int(sidebin/2 + 2):sidebin] * self.span_speed

    #     self.Dedges[0] = self.Dedges_const[0] * self.leader_speed
    #     self.Dedges[1] = self.Dedges_const[1] * self.T_wave_speed
    #     self.Dedges[2] = self.Dedges_const[2] * self.follower_speed
    #     self.Dedges[3:sidebin] = self.Dedges_const[3:sidebin] * self.span_speed

    #     self.Dedges[middlebin+sidebin + 1:] =  - np.flip(np.copy(self.Dedges[0:sidebin]))

    #     # self.Dedges[1:-1] =  self.Dedges_const[1:-1] * self.T_wave_speed 
    #     # self.Dedges[0] =  self.Dedges_const[0] * self.right_speed 
    #     # self.Dedges[-1] =  self.Dedges_const[-1] * self.right_speed 


    #     # self.Dedges = self.Dedges_const * self.right_speed
    #     # self.edges = self.edges + self.Dedges * delta_t
    #     # self.told = t
    #     # print(self.edges[-1]-self.edges[-2], 'thin zone')

    #     self.edges = self.edges + self.Dedges * little_delta_t
    #     self.told = t

    def thin_square_init_func(self):
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-12
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)

        
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)

        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed
        self.edges0 = self.edges


    def thin_square_init_func_legendre(self):
        print('calling mesh with legendre spacing')
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-2
        # left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        # right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        left_old = self.thick_quad_edge
        right_old = self.thick_quad_edge
        right = (right_old*(self.x0-self.x0-dx)-self.x0-dx-self.x0)/-2
        left = (left_old*(-self.x0-dx+self.x0)+self.x0+dx+self.x0)/-2

        # if self.N_space == 32 and self.move_func == 2:
        #     middle = np.array([-0.99057548, -0.95067552, -0.88023915, -0.781514  , -0.65767116,
        #                 -0.51269054, -0.35123176, -0.17848418,  0.        ,  0.17848418,
        #                 0.35123176,  0.51269054,  0.65767116,  0.781514  ,  0.88023915,
        #                 0.95067552,  0.99057548]) 
        # else:
        # if self.move_func == 2:
        middle = self.x0 * self.thick_quad

            # left = roots_legendre(siebin+1)[0]
            # right = roots_legendre(siebin+1)[0]
            # right =(right*(self.x0-self.x0-dx)-self.x0-dx-self.x0)/-2
            # left =(left*(-self.x0-dx+self.x0)+self.x0+dx+self.x0)/-2
            # print(left, right)

        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed 
        self.Dedges_const = np.copy(self.Dedges)
        self.edges0 = self.edges


    def simple_thick_square_init_func_2(self):
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-14
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed * 0
    
    def thick_square_moving_init_func(self):
        # if self.N_space ==2 or self.N_space == 4 or self.N_space == 8:
        #     print(f"don't run this problem with {self.N_space} spaces")
        #     assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        # dx_min = 1e-4
        # wave_source_separation = self.wave_loc_array[0,3,:] - self.x0
        # wave_sep_div = wave_source_separation / (sidebin-1)
        # index = 0
        # if wave_sep_div[index] < dx_min:
        #     index +=1
        # else:
        #     first = index
        # dx = self.wave_loc_array[0,3,index]
        dx = 1e-3
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        # middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        middle = 0.5 * self.thick_quad
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        # self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        # self.Dedges[sidebin:sidebin+middlebin] = 0       
        # self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)

        self.Dedges[0] = -1.0
        self.Dedges[1] = -1.0
        self.Dedges[2] = -1.0
        self.Dedges[3:sidebin] = -(self.edges[3:sidebin] + self.x0) / (self.edges[3] + self.x0)

        # self.Dedges[int(sidebin/2+2):sidebin] = (self.edges[int(sidebin/2+2):sidebin] + self.x0 )/(self.edges[-int(sidebin/2 + 2)] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] =  - np.flip(np.copy(self.Dedges[0:sidebin]))
        self.Dedges = self.Dedges * self.speed



    def thick_square_init_func(self):
        print("initializing thick square source")

        dx = 1e-5

        half = int(self.N_space/2)
        self.edges = np.zeros(self.N_space+1)
        self.Dedges = np.zeros(self.N_space+1) 
        
        self.edges[half] = 0 # place center edge
        self.Dedges[half]= 0

        # self.edges[0] = -self.x0 - 2*dx# place wavefront tracking edges
        # self.edges[-1] = self.x0 + 2*dx

        # self.Dedges[0] = -1 * self.speed
        # self.Dedges[-1] = 1 * self.speed

        number_of_interior_edges = int(self.N_space/2 - 1)
        # print(number_of_interior_edges, "interior")

        # don't use N=4 
        if number_of_interior_edges == 1: # deal with N=4 case 
            self.edges[number_of_interior_edges] = -self.x0
            self.edges[number_of_interior_edges + half] = self.x0
            self.Dedges[number_of_interior_edges] = -1.0 * self.speed
            self.Dedges[number_of_interior_edges + half] = 1.0 * self.speed 
        
        else:                               # set interior edges to track the wavefront

            # set one edge to travel back towards zero and one to be fixed at the source width
            # self.set_func([half-2, half+2], [-self.x0, self.x0], [0,0])
            # self.set_func([half-1, half+1], [-self.x0+dx, self.x0-dx], [self.speed, -self.speed])
            # # set edges to track the wave
            # left_xs = np.linspace(-self.x0-2*dx, -self.x0-dx, half-2)
            # right_xs = np.linspace(self.x0+dx, self.x0+2*dx, half-2)
            # speeds = np.linspace(half-2, 1, half-2)
            # speeds = speeds/(half-2) * self.speed
            # indices_left = np.linspace(0, half-2-1, half-2)
            # indices_right = np.linspace(half+3, self.N_space, half-2)

            # self.set_func(indices_left, left_xs, -speeds)
            # self.set_func(indices_right, right_xs, np.flip(speeds))


            indices_left = np.linspace(0, half-1, half)
            indices_right = np.linspace(half+1, self.N_space, half)
            xs_left = np.zeros(half)
            xs_right = np.zeros(half)
            speeds = np.zeros(half)
            xs_left[int(half/2)] = -self.x0
            # xs_right[int(half/2)] = self.x0
            speeds[int(half/2)] = 0.0 
            xs_left[0:int(half/2)] = np.linspace(-self.x0-2*dx, -self.x0-dx,int(half/2))
            xs_left[int(half/2)+1:] = np.linspace(-self.x0+dx, -self.x0+2*dx, int(half/2)-1)
            # xs_right[0:int(half/2)] = np.linspace(self.x0-2*dx, self.x0-dx, int(half/2))
            # xs_right[int(half/2)+1:] = np.linspace(self.x0+dx, self.x0+2*dx, int(half/2)-1)
            xs_right = -np.flip(xs_left)
            speeds[0:int(half/2)] = np.linspace(int(half/2), 1, int(half/2))/int(half/2)
            speeds[int(half/2)+1:] = -np.linspace(1,int(half/2), int(half/2) -1)/ int(half/2)
            speeds = speeds * self.speed
            # print("#   #   #   #   #   #   #   #   #   #   #   ")
            # print(speeds, "speeds")
            # print("#   #   #   #   #   #   #   #   #   #   #   ")
            self.set_func(indices_left, xs_left, -speeds)
            self.set_func(indices_right, xs_right, np.flip(speeds))

            # self.edges[0:half-1] = np.linspace(-self.x0-dx, -self.x0 + dx, number_of_interior_edges + 1)
            # self.edges[half+2:] = np.linspace(self.x0 - dx, self.x0 + dx, number_of_interior_edges + 1)
            # self.edges[half-1] = -self.x0

            # # self.Dedges[1:half-1] = - self.edges[1:half-1]/self.edges[1] * self.speed
            # # self.Dedges[half+2:-1] = self.edges[half+2:-1]/self.edges[-2] * self.speed 
            # self.Dedges[0:half] = -np.linspace(1,-1, number_of_interior_edges + 1)* self.speed   
            # self.Dedges[half+1:] = np.linspace(-1,1, number_of_interior_edges + 1)* self.speed   


            self.delta_t = self.wave_loc_array[0,0,1] - self.wave_loc_array[0,0,0]

            # print(self.delta_t, 'delta_t')

    def menis_init(self):
        c = 29.98
        if self.moving == False:
            dimensional_t = self.tfinal/29.98
        else:
            dimensional_t = 0.0
        menis_t = -29.6255 + dimensional_t
        rfront = 0.01 * (-menis_t) ** 0.679502 - 1e-8

        dx = 5e-6
        # rf = min(rfront, self.x0/50)
        # self.edges = np.linspace(0*min(rfront, self.x0 - self.x0/20), self.x0, self.N_space + 1)
        self.edges = np.concatenate((np.array([rfront - dx]) ,np.linspace(rfront, self.x0 * self.l, self.N_space)))
        print(self.edges, 'edges0')
        self.Dedges = self.edges * 0
        self.edges0 = self.edges
        self.Dedges_const = self.Dedges
        self.Dedges_const[1:] = - (self.edges[1:]-self.x0-self.edges[1])/self.edges[-1]
        self.Dedges_const[0] = 1.0

       

    
    def menis_init2(self):
        pad = self.x0/100
        third = int(2*(self.N_space + 1)/3)
        rest = int(self.N_space + 1 - third)
        dx = 5e-6
        c = 29.98
        if self.moving == False:
            dimensional_t = self.tfinal/29.98
        else:
            dimensional_t = self.tfinal/29.98
        menis_t = -29.6255 + dimensional_t
        rfront = 0.01 * (-menis_t) ** 0.679502 - dx
    
        self.edges = np.concatenate((np.linspace(0.0, rfront, rest+1)[:-1], np.linspace(self.x0-dx, self.x0, third )))
        # print(self.edges.size)
        assert(int(self.edges.size) == self.N_space + 1)
        # print(self.edges[third])
        # print(self.edges)
        # if abs(self.edges[rest]-rfront) > 1e-14:
        #     assert 0
        self.edges0 = self.edges
        v, a = self.converging_move_interpolate(self.edges[rest])
        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        # print(self.Dedges_const[rest:])
        # print(np.linspace(1, 0.0, third))
        self.Dedges_const[rest:] = np.linspace(1.0, 0.0, third) * v
        self.c1s[rest:] = np.linspace(1.0, 0.0, third) * a
        print(self.Dedges_const*self.tfinal + 0.5 * self.c1s * self.tfinal**2 + self.edges0, 'final edges')

        # print(self.Dedges_const - v, 'dedges - v')
        # print(self.c1s - a, 'c1s - a')
        menis_t = -29.6255 + self.tfinal / c
        rfront = 0.01 * (-menis_t) ** 0.679502 - 1e-8
        # self.edges[rest-1] = rfront -5e-4
        # self.edges[0:rest-1] = np.linspace(0.0, self.edges[rest-1], rest -1)
        # print(self.edges)
        self.Dedges = self.Dedges_const
        # assert 0
    # def menis_init4(self):
    #     dx2 = self.x0/200
    #     if self.moving == False:
    #             dimensional_t = self.tfinal/29.98/self.l
    #     else:
    #             dimensional_t = self.tfinal/29.98/self.l

    #     menis_t = -29.6255 + dimensional_t
    #     rfront = 0.01 * (-menis_t) ** 0.679502
    #     third = int((self.N_space + 1)/3)

    def menis_init4(self):
             if self.moving == False:
                self.edges = np.zeros(self.N_space+1)
                # self.edges[1:] = np.linspace(rfront * self.l, self.x0 * self.l, self.N_space)
                self.edges = np.linspace(0.0, self.x0, self.N_space+1)
                self.edges0 = self.edges

                self.Dedges = self.edges *0 

                self.Dedges_const = self.edges*0
                print(self.edges, 'edges')
             else:
                menis_t = converging_time_function(self.tfinal, self.sigma_func)
                # rfront = 0.01 * (-menis_t) ** 0.679502 
                rfront = converging_r(menis_t, self.sigma_func)
                third = int(1*(self.N_space + 1)/3)
                if third%2 == 0:
                    third += 1
                rest = int(self.N_space + 1 - 2*third)
                assert(2*third+ rest == self.N_space+1)
                center_edge = int(third/2) + 1 + third
                dx2 = self.x0/10
                min_spacing = 1e-6
                dx1 = min_spacing * third
                cluster = np.linspace(self.x0 - dx1 - dx2, self.x0 - dx1, third)
                right_edges = np.linspace(self.x0-dx1+dx2/third, self.x0, third)

                # outside_wave_edges2 =  np.abs((np.flip((np.logspace(0,1,rest+1)-10)/-9) * (self.x0-dx2-dx1-self.x0/100))[:-1])
                outside_wave_edges2 = np.linspace(0.0, self.x0-dx2-dx1, rest+1)[:-1]
                self.edges = np.concatenate((outside_wave_edges2, cluster, right_edges))
                self.edges0 = self.edges
                print(self.edges, 'edges0')
                v, a, j = self.converging_move_interpolate2(self.edges0[-center_edge])
                print(self.edges0[-center_edge], 'tracking edge')
                if v >0:
                    v = 0
                    a = 0
                    j = 0

                self.Dedges_const = self.edges * 0 
                self.c1s = self.edges * 0
                self.c2s = self.edges * 0

                self.Dedges_const[-2*third:-third] = np.ones(third) * v
                self.c1s[-2*third:-third] = np.ones(third) * a
                self.c2s[-2*third:-third] = np.ones(third) * j


                self.Dedges_const[-third:] = np.linspace(1, 0, third) * v
                self.c1s[-third:] = np.linspace(1, 0, third) * a
                self.c2s[-third:] = np.linspace(1, 0, third) * j
                print(self.Dedges_const, 'v0')

                # self.Dedges_const[]
                # self.c1s
                # self.c2s
                print(self.edges0[-center_edge] + self.Dedges_const[-center_edge] * self.tfinal + 0.5 * self.c1s[-center_edge] * self.tfinal ** 2 + self.c2s[-center_edge] * self.tfinal**3/3, 'final center edge')
                final_outside_edge = self.edges0[-2*third] + self.Dedges_const[-2*third] * self.tfinal + 0.5 * self.c1s[-2*third] * self.tfinal ** 2 + self.c2s[-2*third] * self.tfinal**3/3
                print(final_outside_edge, 'last edge of cluster at tfinal')
                print(self.edges0 + self.Dedges_const * self.tfinal + 0.5 * self.c1s * self.tfinal ** 2 + self.c2s * self.tfinal**3/3, 'final edges')
                # assert(0)
                final_rest_edges  =  np.linspace(0.0, final_outside_edge, rest+1)[:-1]
                self.Dedges_const[:rest] = (final_rest_edges - self.edges[:rest]) / self.tfinal
    def two_point_interpolation(self, x1, x2, t1, t2, x0):
        a = (2*(t1*x0 - t2*x0 + t2*x1 - t1*x2))/(t1*(t1 - t2)*t2)
        v = -((t1**2*x0 - t2**2*x0 + t2**2*x1 - t1**2*x2)/(t1*(t1 - t2)*t2))
        return v, a
    def menis_init7(self):
        menis_tf = converging_time_function(self.tfinal, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)
        half = int((self.N_space+1)/2)
        rest = self.N_space +1 -half
        min_space = self.x0/240
        dx = min_space * half
        tracking_edges = np.linspace(0.0, self.x0-dx,half)
        outside_edges = np.linspace(self.x0-dx, self.x0, rest+1)[1:]
        self.edges0 = np.concatenate((tracking_edges, outside_edges))
        self.edges = self.edges0
        final_tracking = np.linspace(0, rfrontf , half)
        final_outside = np.linspace(rfrontf, self.x0, rest+1)[1:]
        final_edges = np.concatenate((final_tracking, final_outside))
        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0
        self.Dedges_const = (final_edges - self.edges0)/self.tfinal
        print(self.edges0, 'initial')
        print(final_edges, 'final')

    def tracker_region_mesh(self, location_to_track):
        width = self.x0/5
        third = int((self.N_space+1)/3)
        rest = int(self.N_space+1 - 2*third)
        tracker_edges = np.linspace(location_to_track - width/2, location_to_track + width/2, rest)
        left_edges = np.linspace(location_to_track + width/2, self.x0, third+1)[1:]
        right_edges = np.linspace(0, location_to_track - width/2, third + 1)[:-1]
        edges = np.concatenate((right_edges, tracker_edges, left_edges))
        assert edges.size == self.N_space +1
        assert (0 <= edges).all()
        assert (edges <= self.x0).all()
        return edges
    
    def menis_init_6real(self):

        t1 = self.tfinal/3
        t2 = 2*self.tfinal/3
        t3 = self.tfinal


        menis_tf = converging_time_function(t3, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)

        menis_tm = converging_time_function(t1, self.sigma_func)
        rfrontm= converging_r(menis_tm, self.sigma_func)

        menis_t23 = converging_time_function(t2, self.sigma_func)
        rfront23= converging_r(menis_t23, self.sigma_func)

        edges23 = self.tracker_region_mesh(rfront23)

        xi = 8.9
        edges0 = self.tracker_region_mesh(xi)
        print(edges0, 'initial edges')
        self.edges0 = edges0
        self.edges = edges0
        edgesm = self.tracker_region_mesh(rfrontm)
        print(edgesm, 'middle edges')
        edgesf = self.tracker_region_mesh(rfrontf +0.005)
        print(edgesf, 'final edges')

        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0

        # self.Dedges_const, self.c1s = self.two_point_interpolation(edgesm, edgesf, self.tfinal/2, self.tfinal, self.edges0)

        self.Dedges_const, self.c1s, self.c2s =  self.three_point_interpolation(t1, t2, t3, edgesm, edges23, edgesf, self.edges0)

        





    def three_point_interpolation(self, t1, t2, t3, r1, r2, r3, x0):
        v = -((-(r3*t1**3*t2**2) + r3*t1**2*t2**3 + r2*t1**3*t3**2 - r1*t2**3*t3**2 - r2*t1**2*t3**3 + r1*t2**2*t3**3 + t1**3*t2**2*x0 - t1**2*t2**3*x0 - t1**3*t3**2*x0 + t2**3*t3**2*x0 + t1**2*t3**3*x0 - t2**2*t3**3*x0)/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3))
        a1 = (2*(-(r3*t1**3*t2) + r3*t1*t2**3 + r2*t1**3*t3 - r1*t2**3*t3 - r2*t1*t3**3 + r1*t2*t3**3 + t1**3*t2*x0 - t1*t2**3*x0 - t1**3*t3*x0 + t2**3*t3*x0 + t1*t3**3*x0 - t2*t3**3*x0))/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3)
        a2 = (-3*(-(r3*t1**2*t2) + r3*t1*t2**2 + r2*t1**2*t3 - r1*t2**2*t3 - r2*t1*t3**2 + r1*t2*t3**2 + t1**2*t2*x0 - t1*t2**2*x0 - t1**2*t3*x0 + t2**2*t3*x0 + t1*t3**2*x0 - t2*t3**2*x0))/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3)
        return v, a1, a2




    def menis_init_5real(self):
        menis_tf = converging_time_function(self.tfinal, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)
        print(rfrontf, 'rfront at tfinal')
        Nedges = int(self.N_space+1)
        half = int(Nedges/2)
        M = int(2*half-1)
        rest = int(Nedges-half)
        Mr = int(2*rest-1)
        dx = 1e-2
        # left = quadrature(int(2* half-1), 'gauss_legendre')[0][int((M-1)/2):] 
        # left = self.thick_quad
        # right = self.thick_quad_edge
        left = np.linspace(0,1, half + 1)[:-1]
        right = np.linspace(1,0,rest)
        print(right, 'right')
        print(left, 'left')
        # right = np.flip(quadrature(int(2* rest-1), 'gauss_legendre')[int((Mr-1)/2):]) 
        # inside = (2 * right-1 - self.x0/(self.x0-dx)) * (self.x0-dx) * right + self.x0
        inside = right * (-dx) + self.x0 
        inside[-1] = self.x0

        # edges0 = np.concatenate((np.linspace(0, self.x0-dx, rest+1)[:-1], np.linspace(self.x0-dx, self.x0, half)))
        edges0 = np.concatenate((left * (inside[0]), inside  ))
        print(edges0, 'initial edges')

        self.edges0 = edges0
        self.edges = edges0
        assert self.edges.size == self.N_space + 1
        # edgesf = np.concatenate((np.linspace(0, rfrontf, rest+1)[:-1], np.linspace(rfrontf, self.x0, half)))
        # insidef = (2 * right-1 - self.x0/rfrontf)* rfrontf * right + self.x0
        # pad = self.x0 / self.N_space /100
        pad = 0.2
        insidef = right * (rfrontf+pad-self.x0) + self.x0
        insidef[-1] = self.x0
        
        outsidef = left * (insidef[0])
        edgesf = np.concatenate((outsidef, insidef))
        print(edgesf, 'final edges')
        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0
        self.Dedges_const = (edgesf - self.edges0)/self.tfinal




    def menis_init_4real(self):
        menis_tf = converging_time_function(self.tfinal, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)
        dx = 1e-5
        spacing = rfrontf*2/3
        Nedges = int(self.N_space+1)
        half = int(Nedges/2)
        halfhalf = int(half/2)
        rest = int(Nedges-2*halfhalf)
        resthalf = int(rest/2)
        rest_rest = rest - resthalf
        outside1 = np.linspace(0, self.x0 - spacing, resthalf +1)[:-1]
        outside2 = np.linspace(self.x0-spacing, self.x0-spacing/2, halfhalf + 1 )[:-1]
        inside1 = np.linspace(self.x0-spacing/2, self.x0 -dx, halfhalf+ 1 )[:-1]
        inside2 = np.linspace(self.x0-dx, self.x0, rest_rest)

        initial_edges = np.concatenate((outside1, outside2, inside1, inside2))
        self.edges0 = initial_edges
        self.edges = self.edges0
        assert self.edges.size == self.N_space + 1
        print(initial_edges, 'initial edges')

        outside1 = np.linspace(0, rfrontf - spacing/2 , resthalf +1)[:-1]
        outside2 = np.linspace(rfrontf -spacing/2,rfrontf, halfhalf + 1 )[:-1]
        inside1 = np.linspace(rfrontf,rfrontf + spacing * 1/2, halfhalf + 1 )[:-1]
        inside2 = np.linspace(rfrontf + spacing * 1/2, self.x0, rest_rest)
        final_edges = np.concatenate((outside1, outside2, inside1, inside2))
        print(final_edges, 'final')


        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0
        self.Dedges_const = (final_edges - self.edges0)/self.tfinal

        




        










    def menis_init_final(self):
        menis_tf = converging_time_function(self.tfinal, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)
        print(menis_tf, 'menis t')
        print(rfrontf, 'rf')
        half = int((self.N_space+1)/2)
        rest = self.N_space +1 -half
        self.thick_quad_edge = np.linspace(0,1, rest+1)[:-1]
        self.thick_quad = np.linspace(0,1, half)

        min_space = self.x0/1000
        pad = min_space 
        dx = min_space * half
        edges_inside = self.x0 - np.flip(self.thick_quad) * dx
        edges_outside = self.thick_quad_edge * (self.x0 - dx)
        initial_edges = np.concatenate((edges_outside, edges_inside))
        self.edges0 = initial_edges
        self.edges = initial_edges
        assert self.edges.size == self.N_space + 1
        edges_inside = (2 * np.flip(self.thick_quad)-1 - self.x0/rfrontf)* rfrontf * np.flip(self.thick_quad) + self.x0
        edges_outside = self.thick_quad_edge * ( rfrontf)
        final_edges = np.concatenate((edges_outside, edges_inside))
        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0
        self.Dedges_const = (final_edges - self.edges0)/self.tfinal
        print(self.edges0, 'initial')
        print(final_edges, 'final')        

        menis_tm = converging_time_function(2*self.tfinal/3, self.sigma_func)
        rfrontm= converging_r(menis_tm, self.sigma_func)
        print(rfrontm, 'rf middle')

        edges_inside = (2 * np.flip(self.thick_quad)-1 - self.x0/rfrontm)* rfrontm * np.flip(self.thick_quad) + self.x0
        edges_outside = self.thick_quad_edge * (rfrontm)
        middle_edges = np.concatenate((edges_outside, edges_inside))
        print(middle_edges, 'middle edges')
        if (middle_edges != np.sort(middle_edges)).any():
            print('edges out of order')
            middle_edges = np.sort(middle_edges)

        # self.Dedges_const, self.c1s = self.two_point_interpolation(middle_edges, final_edges, 2*self.tfinal/3, self.tfinal, self.edges0)












    def menis_init6(self):
        menis_tf = converging_time_function(self.tfinal, self.sigma_func)
        rfrontf= converging_r(menis_tf, self.sigma_func)

        

        half = int((self.N_space+1)/2)
        rest = self.N_space +1 -half
        min_space = self.x0/240
        bumper = max(self.x0/self.N_space/2, self.x0/100)
        assert bumper < rfrontf
        L = 2*(rfrontf-2*bumper)
        dx = min_space * rest
        tracking_edges = np.linspace( self.x0 - L-dx, self.x0-dx,half-1)
        outside_edges = np.linspace(self.x0-dx, self.x0, rest+1)[1:]
        self.edges0 = np.concatenate((np.array([0.0, bumper]), tracking_edges, outside_edges))
        self.edges = self.edges0
        print(self.edges0, 'initial edges')
        final_tracking = np.linspace(rfrontf-L/2, rfrontf + L/2, half-1)
        final_outside = np.linspace(rfrontf+L/2, self.x0, rest+1)[1:]
        final_edges = np.concatenate( (np.array([0, bumper]), final_tracking, final_outside))

        menis_tm = converging_time_function(self.tfinal/2, self.sigma_func)
        rfrontm= converging_r(menis_tm, self.sigma_func)

        tracking_edges_middle = np.linspace( rfrontm- L/2, rfrontm + L/2,half-1)
        outside_edges_middle = np.linspace(rfrontm + L/2, self.x0, rest+1)[1:]
        middle_edges = np.concatenate( (np.array([0, bumper]), tracking_edges_middle, outside_edges_middle))


        # v, a, j = self.converging_move_interpolate2(self.edges0[rest]
        

        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0
        self.Dedges_const = (final_edges - self.edges0)/self.tfinal
        # self.Dedges_const, self.c1s = self.two_point_interpolation(middle_edges, final_edges, self.tfinal/2, self.tfinal, self.edges0)

        if (self.Dedges_const> 1e-12).any():
            self.Dedges_const = self.Dedges_const * 0
            self.c1s = self.c1s * 0
            print(self.edges0 - final_edges)
            assert(0)

        print(final_edges, 'final edges')
        print(self.Dedges_const, 'v')
        print(self.c1s, 'a')


        






    def menis_init5(self):

        menis_t = converging_time_function(self.tfinal, self.sigma_func)
            # rfront = 0.01 * (-menis_t) ** 0.679502 
        pad = self.x0/self.N_space/150
        rfront = converging_r(menis_t, self.sigma_func) - pad
        print(rfront, 'rfront')
        half = int((self.N_space+1)/2)
        rest = self.N_space +1 -half
        
        min_space = self.x0/150
        dx = min_space * half
        inside_edges = self.x0 - (np.abs((np.logspace(0,1,half)-10)/-9) )*dx     
        outside_edges =  (np.flip((np.abs((np.logspace(0,1,rest+1)-10)/-9) )) * (self.x0-dx))[:-1]
        self.edges = np.concatenate((outside_edges, inside_edges))
        self.edges0 = self.edges
        print(self.edges0, 'initial edges')
        # v, a, j = self.converging_move_interpolate2(self.edges0[rest])

        self.Dedges_const = self.edges * 0 
        self.c1s = self.edges * 0
        self.c2s = self.edges * 0

        menis_t = converging_time_function(self.tfinal/2, self.sigma_func)
        rfront1 = converging_r(menis_t, self.sigma_func) - pad
        inside_edges_mid = self.x0 - (np.abs((np.logspace(0,1,half)-10)/-9) )* (self.x0-rfront1)    
        outside_edges_mid =  (np.flip((np.abs((np.logspace(0,1,rest+1)-10)/-9) )) * (rfront1))[:-1]


        menis_t = converging_time_function(self.tfinal, self.sigma_func)
        rfront2 = converging_r(menis_t, self.sigma_func) - pad
        rfront_stopper = rest * min_space
        rfront22 = max(rfront_stopper, rfront2)
        print(rfront2, rfront22, '#####')
        rfront2 = rfront22


        inside_edges = self.x0 - (np.abs((np.logspace(0,1,half)-10)/-9) )* (self.x0-rfront2)    
        outside_edges =  (np.flip((np.abs((np.logspace(0,1,rest+1)-10)/-9) )) * (rfront2))[:-1]

        print(inside_edges, 'inside finial ')
        print(outside_edges, 'outside final')
        final_edges = np.concatenate((outside_edges, inside_edges))
        print(final_edges, 'final edges')
        
        self.Dedges_const[:rest] = (outside_edges - self.edges0[:rest])/self.tfinal
        self.Dedges_const[rest:] = (inside_edges  -self.edges0[rest:])/self.tfinal

        if (final_edges> self.x0).any() or (final_edges<0).any() or (self.Dedges_const >0).any():
            self.Dedges_const = self.Dedges_const * 0

        


        # r1 = np.concatenate((outside_edges_mid, inside_edges_mid))
        # r2 = np.concatenate(( outside_edges, inside_edges))
        # t1 = self.tfinal/2
        # t2 = self.tfinal
        # x0 = self.edges0
        # self.Dedges_const = -((-(r2*t1**2) + r1*t2**2 + t1**2*x0 - t2**2*x0)/(t1*(t1 - t2)*t2))
        # self.c1s = (2*(-(r2*t1) + r1*t2 + t1*x0 - t2*x0))/(t1*(t1 - t2)*t2)
        # print(self.Dedges_const, 'v0s')
        # print(self.c1s, 'c1s')



        print('mesh built')
        

        


    def menis_init3(self):

            
            third = int((self.N_space + 1)/3)
            rest = int(self.N_space + 1 - third)
            # dx = 5e-5
            # min_space = self.x0 / 1e
            min_space = self.x0/300
            dx = min_space * third
            pad = 100* dx
            c = 29.98
            if self.moving == False:
                dimensional_t = self.tfinal/29.98/self.l
            else:
                dimensional_t = self.tfinal/29.98/self.l
            # menis_t = -29.6255 + dimensional_t
            menis_t = converging_time_function(self.tfinal, self.sigma_func)
            # rfront = 0.01 * (-menis_t) ** 0.679502 
            rfront = converging_r(menis_t, self.sigma_func)
            if self.moving == False:
                self.edges = np.zeros(self.N_space+1)
                # self.edges[1:] = np.linspace(rfront * self.l, self.x0 * self.l, self.N_space)
                # inside = np.linspace(rfront, self.x0, 2*third)
                # outside = np.linspace(0.0, rfront, rest+1)[0:-1]
                # # self.edges = np.linspace(0.0, self.x0, self.N_space+1)
                # self.edges = np.concatenate((outside, inside))
                self.edges = np.linspace(0, self.x0, self.N_space + 1)
                # self.edges = np.concatenate((np.linspace(0, rfront, third +1 )[0:-1], np.linspace(rfront, self.x0, rest)))
                self.edges0 = self.edges

                self.Dedges = self.edges *0 
                self.Dedges_const = self.edges * 0 
                self.c1s = self.edges * 0
                self.c2s = self.edges * 0

                # self.Dedges_const = self.edges*0
            else:
                min_space2 = 5 * min_space
                # dx2 = min_space2 * third
                # dx2 = self.x0/500
                dx2 = self.x0/50
                # inside_wave_edges = self.x0 - (np.abs((np.logspace(0,1,third)-10)/-9) )*dx 
                inside_wave_edges = np.linspace(self.x0-dx, self.x0, third)
                # inside_wave_edges = np.

                outside_wave_edges2 =  np.abs((np.flip((np.logspace(0,1,rest+1)-10)/-9) * (self.x0-dx2-dx))[:-1])
                outside_wave_edges = (np.linspace(self.x0-dx2-dx, self.x0-dx, third + 1)[:-1])
                print('#')
                print(inside_wave_edges)
                print(outside_wave_edges)
                print(outside_wave_edges2)
                print('#')

                self.edges = np.concatenate((outside_wave_edges2, outside_wave_edges, inside_wave_edges))

                print(self.edges, 'edges 0 ')
                # print(self.edges.size)
                assert(int(self.edges.size) == self.N_space + 1)
                # print(self.edges[third])
                # print(self.edges)
                # if abs(self.edges[rest]-rfront) > 1e-14:
                #     assert 0
                self.edges0 = self.edges
                # v, a = self.converging_move_interpolate(self.edges[rest])
                # v, a, j = self.converging_move_interpolate2(self.edges0[-third])
                v, a, j = self.converging_move_interpolate2(self.edges0[-third])

  
                
                # menis_t = -29.6255 + self.tfinal / c /self.l

                # rfront = 0.01 * (-menis_t) ** 0.679502 
                print(rfront,'final shock front')

                self.Dedges_const = self.edges * 0 
                self.c1s = self.edges * 0
                self.c2s = self.edges * 0
                # print(self.Dedges_const[rest:])
                # print(np.linspace(1, 0.0, third))



                self.Dedges_const[-third:] = np.linspace(1.0, 0.0, third) * v

                self.Dedges_const[-2*third:-third] = np.ones(third) * v

                self.c1s[-third:] = np.linspace(1.0, 0.0, third) * a

                self.c1s[-2*third:-third] = np.ones(third) * a

                self.c2s[-third:] = np.linspace(1.0, 0.0, third) * j

                self.c2s[-2*third:-third] = np.ones(third) * j

                # print(1)
                # self.Dedges_const[:rest] = np.linspace(0,1, rest) * v
                final_rest_edges  =  (np.flip((np.logspace(0,1,rest+1)-10)/-9) * (rfront*self.l - dx -dx2))[:-1]
                self.Dedges_const[:rest] = (final_rest_edges - self.edges[:rest]) / self.tfinal

                # self.Dedges_const[1:rest] = np.ones(rest-1) * v
                # print(2)
                print(self.Dedges_const, 'v0')
                print(self.c1s, 'c1')
                print(self.c2s,'c2')

                if v > 0:
                    self.Dedges_const = self.Dedges_const * 0
                    self.c1s = self.c1s * 0
                    self.c2s = self.c2s * 0
                

                # print(3)
                # self.c1s[:rest] = np.linspace(0, 1, rest) * a
                # self.c1s[1:rest] = np.ones(rest-1) * a
                # print(4)
                # self.c1s[1] = 1.0 * a
                print(self.Dedges_const*self.tfinal + 0.5 * self.c1s * self.tfinal**2 + self.edges0 + self.c2s * self.tfinal**3/3, 'final edges')

                # print(self.Dedges_const - v, 'dedges - v')
                # print(self.c1s - a, 'c1s - a')
                
                # self.edges[rest-1] = rfront -5e-4
                # self.edges[0:rest-1] = np.linspace(0.0, self.edges[rest-1], rest -1)
                # print(self.edges)
                self.Dedges = self.Dedges_const
        # assert 0

    # def converging_move(self, t):
    #     c = 29.98
    #     dimensional_t = t/c/self.l 

    #     # menis_t = -29.6255 + dimensional
    #     # m_t
    #     menis_t = converging_time_function(t, self.sigma_func)
    #     # rfront = 0.01 * (-menis_t) ** 0.679502 - 1e-8
    #     rfront = converging_r(menis_t, self.sigma_func)
    #     rfront = rfront * self.l
    #     dr_dt = -0.00022665176784523018/(29.6255 - 0.0333555703802535*t)**0.32049799999999995
    #     dx = 5e-4
    #     self.edges[1:] = np.linspace(rfront*self.l, self.x0*self.l, self.N_space)
    #     self.Dedges[1:] = self.Dedges_const[1:] * dr_dt
    #     self.edges[0] = rfront*self.l - dx
    #     self.Dedges[0] = self.Dedges_const[0] * dr_dt

    def converging_move2(self, t):
        if self.moving == True:
        # self.Dedges_const = self.Dedges_const
            self.edges = self.Dedges_const*t + 0.5 * self.c1s * t**2 + self.edges0 + self.c2s * t**3 /3
            # print(self.Dedges_const)
            self.Dedges =  self.Dedges_const + self.c1s * t  + self.c2s * t**2
            # print('### ### ## ### ### ###')
            # print(self.Dedges_const)
            # print(self.c1s)
            # print(self.c2s)
            # print(self.edges, t)
            # dimensional_t = t/29.98/self.l
            # menis_t = -29.6255 + dimensional_t
            # rfront = self.l * 0.01 * (-menis_t) ** 0.679502 
            # third = int(2*(self.N_space + 1)/3)
            # rest = int(self.N_space + 1 - third)
        # if self.tfinal-t <= 1e-4:
        #     print(self.edges[rest] - rfront)
        # print(self.edges[rest])
        # print(rfront, 'rf')
        # if abs(self.edges[rest]-rfront) > 5e-3:
        #     print(self.edges[rest]-rfront, 'rf')
        #     print(self.edges[rest-1]-rfront, 'rf-1')
        #     print(self.edges[rest+1]-rfront, 'rf+1')
            # if self.edges[1] > rfront:
            #     assert 0


        # print(self.edges, 'edges')
        # print(self.Dedges, 'dedges')

    
    def converging_move_interpolate(self, x0):
        c = 29.98 
        tf = self.tfinal
        tm = self.tfinal/2
        
        dimensional_tm = tm/c 
        menis_tm = -29.6255 + dimensional_tm
        rm = 0.01 * (-menis_tm) ** 0.679502 

        dimensional_tf = tf/c
        menis_tf = -29.6255 + dimensional_tf
        rf=  0.01 * (-menis_tf) ** 0.679502 

        v0 = -((-(rm*tf**2) + rf*tm**2 + tf**2*x0 - tm**2*x0)/(tf*(tf - tm)*tm))
        a = (2*(-(rm*tf) + rf*tm + tf*x0 - tm*x0))/(tf*(tf - tm)*tm)
        print(v0, 'v0')
        print(a, 'a')
        # if v0 >0:
        #     v0 = 0
        #     a = 0

        return v0, a
    def converging_move_interpolate2(self, x0):
        c = 29.98 
        t3 = self.tfinal
        t1 = self.tfinal/3
        t2 = 2 * self.tfinal/3

        
        dimensional_t1 = t1/c 
        # menis_t1 = -29.6255 + dimensional_t1
        menis_t1 = converging_time_function(t1, self.sigma_func)
        # r1 = 0.01 * (-menis_t1) ** 0.679502 
        r1 = converging_r(menis_t1, self.sigma_func)

        dimensional_t2 = t2/c
        # menis_t2 = -29.6255 + dimensional_t2
        menis_t2 = converging_time_function(t2, self.sigma_func)
        # r2=  0.01 * (-menis_t2) ** 0.679502 
        r2 = converging_r(menis_t2, self.sigma_func)

        dimensional_t3 = t3/c
        # menis_t3 = -29.6255 + dimensional_t3
        menis_t3 = converging_time_function(t3, self.sigma_func)
        # r3=  0.01 * (-menis_t3) ** 0.679502 
        r3 = converging_r(menis_t3, self.sigma_func)
        print(r1, r2, r3, 'rs')

        v0 = -((-(r3*t1**3*t2**2) + r3*t1**2*t2**3 + r2*t1**3*t3**2 - r1*t2**3*t3**2 - r2*t1**2*t3**3 + r1*t2**2*t3**3 + t1**3*t2**2*x0 - t1**2*t2**3*x0 - t1**3*t3**2*x0 + t2**3*t3**2*x0 + t1**2*t3**3*x0 - t2**2*t3**3*x0)/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3))
        a = (2*(-(r3*t1**3*t2) + r3*t1*t2**3 + r2*t1**3*t3 - r1*t2**3*t3 - r2*t1*t3**3 + r1*t2*t3**3 + t1**3*t2*x0 - t1*t2**3*x0 - t1**3*t3*x0 + t2**3*t3*x0 + t1*t3**3*x0 - t2*t3**3*x0))/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3)
        j = (-3*(-(r3*t1**2*t2) + r3*t1*t2**2 + r2*t1**2*t3 - r1*t2**2*t3 - r2*t1*t3**2 + r1*t2*t3**2 + t1**2*t2*x0 - t1*t2**2*x0 - t1**2*t3*x0 + t2**2*t3*x0 + t1*t3**2*x0 - t2*t3**2*x0))/(t1*(t1 - t2)*t2*(t1 - t3)*(t2 - t3)*t3)
       
        print(v0, 'v0')

        print(a, 'a')
        print(j,'j')
        # if v0 >0:
        #     v0 = 0
        #     a = 0

        return v0, a,j 
    

    def boundary_source_init_func(self, v0):
        mid = int(self.N_space/2)
        self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
        self.Dedges = np.copy(self.edges)*0
        if self.moving == False:
            v0 = 0
        # self.Dedges[mid] = - self.fake_sedov_v0
        ### First attempt -- Even spacing
        final_shock_point = - self.tfinal * v0
        final_edges_left_of_shock = np.linspace(-self.x0, final_shock_point, int(self.N_space/2+1))
        final_edges_right_of_shock = np.linspace(final_shock_point, self.x0, int(self.N_space/2+1))
        final_edges = np.concatenate((final_edges_left_of_shock[:-1], final_edges_right_of_shock))
        self.Dedges = (final_edges - self.edges) / self.tfinal
        self.edges0 = self.edges

        ### Second -- squared  spacing

        # xsi = ((np.linspace(0,1, mid + 1))**2)
        # final_shock_point = - self.tfinal * v0
        # initial_edges_right = ((xsi*2-1)*(-self.x0)-self.x0)/(-2)
        # initial_edges_left = ((xsi*2-1)*(self.x0)+self.x0)/(-2)
        # self.edges = np.concatenate((np.flip(initial_edges_left), initial_edges_right[1:]))
        # print(self.edges)
        # final_shock_point = - self.tfinal * v0
        # # final_edges_left_of_shock = np.linspace(-self.x0, final_shock_point, int(self.N_space/2+1))
        # # final_edges_right_of_shock = np.linspace(final_shock_point, self.x0, int(self.N_space/2+1))
        # final_edges_right_of_shock = ((xsi*2-1)*(final_shock_point-self.x0)-final_shock_point-self.x0)/(-2)
        # final_edges_left_of_shock = np.flip(((xsi*2-1)*(final_shock_point+self.x0)+ self.x0-final_shock_point)/(-2))
        # print(final_edges_left_of_shock)
        # final_edges = np.concatenate((final_edges_left_of_shock[:-1], final_edges_right_of_shock))
        # print(final_edges, 'final edges')
        # self.Dedges = (final_edges - self.edges) / self.tfinal
        # self.edges0 = self.edges


        
    def initialize_mesh(self):

        """
        Initializes initial mesh edges and initial edge derivatives. This function determines
        how the mesh will move
        """
        print('initializing mesh')
        # if self.problem_type in ['plane_IC']:
        if self.source_type[0] == 1 or self.source_type[0] == 2:
            self.simple_moving_init_func()

        if self.thick == False:     # thick and thin sources have different moving functions
            # if self.problem_type in ['gaussian_IC', 'gaussian_source']:
            if self.source_type[3] == 1 or self.source_type[5] == 1:
                self.simple_moving_init_func()
            # elif self.problem_type in ['square_IC', 'square_source']:
            #elif self.source_type[1] == 1 or self.source_type[2] == 1:
            elif self.source_type[1] == 1:
                print('calling thin square init')
                if self.geometry['slab'] == True:
                    self.thin_square_init_func_legendre()
                else:
                    # self.simple_moving_init_func()
                    if self.moving == True:
                        self.shell_source()
                    else:
                        self.edges = np.zeros(self.N_space+1)
                        self.edges[1:] = np.linspace(self.x0, self.x0 + self.tfinal, self.N_space)
                        self.Dedges = self.edges * 0
                        self.Dedges_const = self.Dedges
                        self.edges0 = self.edges
            elif self.source_type[2] == 1:
                print('calling thin square init')
                if self.geometry['slab'] == True:
                    self.thin_square_init_func_legendre()
                else:
                    # assert 0
                    # self.edges = np.linspace(500, 520, self.N_space + 1)
                    # self.edges[self.N_space//3] = 509.5
                    # self.edges[(2*self.N_space)//3] = 510.5
                    # self.edges = np.sort(self.edges)
                    # self.Dedges = self.edges * 0
                    self.shell_source()    
             
            elif np.all(self.source_type == 0):
                if self.geometry['sphere'] == True:
                    if self.moving == False:
                        self.menis_init3()
                    else:
                        # self.menis_init_final()
                        # self.menis_init_4real()
                        # self.menis_init4()
                        self.menis_init_6real()

                    # print(self.Dedges_const, 'dedges const')
                else:
                    self.boundary_source_init_func(self.vnaught)
                # boundary_source_init_func_outside(self.vnaught, self.N_space, self.x0, self.tfinal) 
                
                print('calling boundary source func')
            # if self.


        elif self.thick == True:
            # if self.problem_type in ['gaussian_IC', 'gaussian_source']:
            if self.source_type[3] == 1 or self.source_type[5] == 1:
                if self.moving == True:
                    self.simple_moving_init_func()
                elif self.moving == False:
                    self.thick_gaussian_static_init_func()

            elif self.source_type[1] == 1 or self.source_type[2] == 1 or self.source_type[0]!= 0:
                if self.move_func == 0:
                    self.simple_moving_init_func()

                if self.moving == False:

                    if self.move_func == 1:

                        self.simple_thick_square_init_func()
                    elif self.move_func == 2:

                        if self.source_type[0] != 0:

                            self.simple_moving_init_func()
                        else:
                            self.thin_square_init_func()
                elif self.moving == True:
                    if self.move_func == 1:
                        self.thick_square_moving_init_func()
                    elif self.move_func == 2:
                            self.thin_square_init_func()

        # self.edges0 = self.edges
        # self.Dedges_const = self.Dedges
        # print(self.Dedges_const, 'dedges const')



        if self.moving == False:

            self.tactual = 0.0
            # static mesh -- puts the edges at the final positions that the moving mesh would occupy
            # sets derivatives to 0
            # self.moving = True
            if self.thick == True:
                self.delta_t = self.tfinal 
            self.move(self.tfinal)
            self.Dedges = self.Dedges*0
            self.moving = False
            # self.edges[-1] = self.x0 + self.tfinal * self.speed
            # self.edges[0] = -self.x0 + -self.tfinal * self.speed


            print(self.edges[-1], "final edges -- last edge") 




    


