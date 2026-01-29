import FVis3 as FVis
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cst


class SolarConvection2D:

    # Constants
    mu = 0.61                   # Mean molecular weight
    G = cst.G.value             # Gravitational constant
    gamma = 5/3                 # Degrees of freedom parameter
    k_B = cst.k_B.value         # Boltzmann's constant
    m_u = cst.u.value           # Atomic mass unit
    M_sun = cst.M_sun.value     # Solar mass
    R_sun = cst.R_sun.value     # Solar radius
    g = G*M_sun/R_sun**2        # Graviational acceleration
    P_phsph = 1.8e4             # Photosphere pressure
    T_phsph = 5778              # Photosphere temperature
    
    def __init__(self, Nx=300, Ny=100, width_box=12e6, height_box=4e6, p=0.01, 
                N_pert=1, nabla=2/5):
        # Grid dimensions
        self.Nx, self.Ny = Nx, Ny
        
        # Physical size of the simulated box
        self.width_box = width_box
        self.height_box = height_box
        
        # x- and y-axis
        self.x = np.linspace(0, width_box, Nx)
        self.y = np.linspace(0, height_box, Ny)
        
        # Width and height of each cell in the grid
        self.delta_x = width_box/(Nx-1)
        self.delta_y = height_box/(Ny-1)
        
        # Percentage upper limit for relative change in variables
        self.p = p
        
        # Number of perturbation in temp
        self.N_pert = N_pert
        
        # Temperature gradient
        self.nabla = nabla
        

    def initialise(self, perturb=True, random=False):
        """
        initialise temperature, pressure, density and internal energy
        """
        # Horizontal and vertical flow
        self.u = np.zeros((self.Ny, self.Nx))
        self.w = np.zeros((self.Ny, self.Nx))
        
        # Temperature
        self.T = np.zeros((self.Ny, self.Nx))
        for j in range(0, self.Ny):
            Hy = self.height_box-self.y[j]
            self.T[j,:] = self.T_phsph + (self.mu*self.m_u*self.g)/self.k_B * self.nabla*Hy
        
        # Pressure
        self.P = self.P_phsph*(self.T/self.T_phsph)**(1/self.nabla)
        
        if perturb:
            self.perturb_initial_temp(random)
            
        # Internal energy
        self.e = np.zeros((self.Ny, self.Nx))
        self.e[:] = self.P/(self.gamma - 1)
        
        # Density
        self.rho = np.zeros((self.Ny, self.Nx))
        self.rho[:] = self.mu*self.m_u*self.P/(self.k_B*self.T)

        
        

    def perturb_initial_temp(self, random):
        """
        Adds N 2D Gaussians to the initial temperature.
        If random==True, the placements and sizes of the Gaussians are chosen at random.
        """
        N = self.N_pert
        if random:
            x_mean = (np.random.rand(N)+1)*self.width_box/3
            y_mean = np.random.rand(N)*self.height_box
            sigma_x = np.random.rand(N)*5e5  
            sigma_y = np.random.rand(N)*5e5
        else:
            # Equally spaced Gaussians
            x_mean = np.linspace(0, self.width_box, N+2)[1:-1]
            y_mean = np.zeros(N)
            sigma_x = np.ones(N)*3e5  
            sigma_y = np.ones(N)*1e6
    
        x,y = np.meshgrid(self.x,self.y) 
        
        pertamp = 0.5*np.max(self.T) # Peturbation amplitude
        print(f'Perturbation amp: {pertamp:.3g} K')
        for i in range(N): 
            x0 = x_mean[i]
            y0 = y_mean[i]
            stdx = sigma_x[i]
            stdy = sigma_y[i]
            perturbation = np.exp(-0.5*((x-x0)**2/stdx**2+(y-y0)**2/stdy**2))
            if random:
                rand_num = np.random.rand(1)[0]
                sign = 1
                if rand_num > 0.5:
                    sign = -1
                self.T += sign*pertamp*perturbation
            else:
                self.T += pertamp*perturbation
                
                
  
    def timestep(self):
        """
        calculate time step
        """
        safety_dt = 1e-6    # To ensure no division by zero
        delta = np.nanmax(
            np.array([
                np.nanmax(np.abs(self.u/self.delta_x)),
                np.nanmax(np.abs(self.w/self.delta_y)),
                np.nanmax(np.abs(self.drhodt/self.rho)),
                np.nanmax(np.abs(self.dedt/self.e)),
                safety_dt
            ])   
        )
        dt = self.p/delta

        # Upper limit on time step
        if dt > 0.1:
            dt = 0.1

        return dt
        
        

    def boundary_conditions(self):

        """
        boundary conditions for energy, density and velocity
        """
        # No vertical velocity at lower and upper bound
        self.w[0,:] = 0
        self.w[-1,:] = 0
        
        # Zero horizontal velocity gradient at lower and upper bound
        self.u[0,:] = (4*self.u[1,:] - self.u[2,:])/3
        self.u[-1,:] = (4*self.u[-2,:] - self.u[-3,:])/3
        
        # Internal energy gradient in accordance with hydrostatic equilibrium
        self.e[0,:] = (4*self.e[1,:] - self.e[2,:])/ \
                      (3 - (2*self.delta_y*self.mu*self.m_u*self.g/(self.k_B*self.T[0,:])))
        self.e[-1,:] = (4*self.e[-2,:] - self.e[-3,:])/ \
                      (3 - (2*self.delta_y*self.mu*self.m_u*self.g/(self.k_B*self.T[-1,:])))

        # Density at vertical boundaries
        self.rho[0,:] = (self.gamma-1)*self.mu*self.m_u/(self.k_B*self.T[0,:])*self.e[0,:]
        self.rho[-1,:] = (self.gamma-1)*self.mu*self.m_u/(self.k_B*self.T[-1,:])*self.e[-1,:]
        
        
    def central_x(self, var):
        """
        central difference scheme in x-direction
        """
        right = np.roll(var, shift=-1, axis=1)
        left = np.roll(var, shift=1, axis=1)
        return (right-left)/(2*self.delta_x)

    def central_y(self, var):
        """
        central difference scheme in y-direction
        """
        up = np.roll(var, shift=-1, axis=0)
        down = np.roll(var, shift=1, axis=0)
        return (up-down)/(2*self.delta_y)
        
    def upwind_x(self, var, u):
        """
        upwind difference scheme in x-direction
        """
        u_neg = u < 0
        u_pos = u >= 0
        
        diff = np.zeros((self.Ny, self.Nx))
        negative = (np.roll(var, shift=-1, axis=1) - var)/self.delta_x
        positive = (var - np.roll(var, shift=1, axis=1))/self.delta_x
        diff[u_neg] = negative[u_neg]
        diff[u_pos] = positive[u_pos]

        return diff
        
    def upwind_y(self, var, w):
        """
        upwind difference scheme in y-direction
        """
        w_neg = w < 0
        w_pos = w >= 0
        
        diff = np.zeros((self.Ny, self.Nx))
        negative = (np.roll(var, shift=-1, axis=0) - var)/self.delta_y
        positive = (var - np.roll(var, shift=1, axis=0))/self.delta_y
        diff[w_neg] = negative[w_neg]
        diff[w_pos] = positive[w_pos]
        
        return diff

    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        u, w = self.u, self.w
        T = self.T
        P = self.P
        e = self.e
        rho = self.rho

        rhou = rho*u
        rhow = rho*w
        
        # Continuity equation
        self.drhodt = -(
            u*self.upwind_x(rho, u) + w*self.upwind_y(rho, w)
            + rho*(self.central_x(u) + self.central_y(w))
        )
        
        # Energy equation
        self.dedt = -(
            u*self.upwind_x(e, u) + w*self.upwind_y(e, w) 
            + (e+P)*(self.central_x(u) + self.central_y(w))
        )
        # Momentum eq (x)
        drhoudt = -(
            u*self.upwind_x(rhou, u) + rhou*self.upwind_x(u, u)
            + w*self.upwind_y(rhou, w) + rhou*self.central_y(w) 
            + self.central_x(P)
        )
        # Momentum equation (y)
        drhowdt = -(
            w*self.upwind_y(rhow, w) + rhow*self.upwind_y(w, w)
            + u*self.upwind_x(rhow, u) + rhow*self.central_x(u) 
            + self.central_y(P) + rho*self.g
        )
        
        # Find optimal timestep
        dt = self.timestep()
        
        # Forward Euler to advance the primary variables in time
        self.e[:] = e + self.dedt*dt
        self.rho[:] = rho + self.drhodt*dt
        self.u[:] = (rhou + drhoudt*dt)/self.rho
        self.w[:] = (rhow + drhowdt*dt)/self.rho
        
        # Set boundary conditions
        self.boundary_conditions()
        
        # Update P and T
        self.P[:] = (self.gamma - 1)*self.e 
        self.T[:] = self.P*self.mu*self.m_u/(self.rho*self.k_B)
    
        return dt
        

# Initiate fluid visualiser
vis = FVis.FluidVisualiser(fontsize=20)

####################################
#### EQUILIBRIUM (SANITY CHECK) ####
####################################
solver = SolarConvection2D()
solver.initialise(perturb=False)

vis.save_data(
    60, solver.hydro_solver, 
    rho=solver.rho, u=solver.u, w=solver.w, 
    e=solver.e, P=solver.P, T=solver.T,
    folder='equilibrium', sim_fps=10
)
vis.animate_2D('w', folder='equilibrium',
            showQuiver=True, save=False, cmap='YlOrRd_r',
            extent=[0, 12, 0, 4], units={'Lx':'Mm', 'Lz':'Mm'})
# vis.plot_avg('w', folder='equilibrium')



##############################
####         CASE 1       ####
#### LOW NABLA AND 3 PERT ####
##############################
# solver = SolarConvection2D(N_pert=3, nabla=2/5+0.01)
# solver.initialise(perturb=True, random=False)

# vis.save_data(
#     1000, solver.hydro_solver, 
#     rho=solver.rho, T=solver.T, u=solver.u, w=solver.w,
#     P=solver.P, e=solver.e,
#     folder='case1', sim_fps=0.5
# )
# vis.animate_2D('T', folder='case1',
#             showQuiver=True, save=True, cmap='YlOrRd_r',
#             extent=[0, 12, 0, 4], units={'Lx':'Mm', 'Lz':'Mm'})
# vis.plot_avg('v', folder='case1', title=r'Average speed for $\nabla^*=0.41$')
# vis.plot_avg('rho', folder='case1', relative=True, showTrendline=True, title=r'Variation of avg. density for $\nabla^*=0.41$')
# vis.animate_energyflux(folder='case1', title=r'Vertical energy flux $\nabla^*=0.41$', save=True, extent=[0, 12, 0, 4], units={'Lx':'Mm', 'Lz':'Mm'})



#################################
####          CASE 2         ####
#### MEDIUM NABLA AND 3 PERT ####
#################################
# solver = SolarConvection2D(N_pert=3, nabla=4)
# solver.initialise(perturb=True, random=False)

# vis.save_data(
#     1000, solver.hydro_solver, 
#     rho=solver.rho, T=solver.T, u=solver.u, w=solver.w,
#     P=solver.P, e=solver.e,
#     folder='4nabla3pert', sim_fps=0.5
# )
# vis.animate_2D('rho', folder='4nabla3pert',
#             showQuiver=True, save=True, cmap='Blues',
#             extent=[0, 12, 0, 4], units={'Lx':'Mm', 'Lz':'Mm'},
#             lognorm=True)
# vis.plot_avg('v', folder='4nabla3pert', title=r'Average speed for $\nabla^*=4$')
# vis.plot_avg('rho', folder='4nabla3pert', relative=True, showTrendline=True, title=r'Variation of avg. density for $\nabla^*=4$')



##################################
####          CASE 3          ####
#### RANDOM PERT & HIGH NABLA ####
##################################
# np.random.seed(6251)
# solver = SolarConvection2D(N_pert=3, nabla=8)
# solver.initialise(perturb=True, random=True)

# vis.save_data(
#     1000, solver.hydro_solver, 
#     rho=solver.rho, T=solver.T, u=solver.u, w=solver.w,
#     P=solver.P, e=solver.e,
#     folder='random', sim_fps=0.5
# )
# vis.animate_2D('T', folder='random',
#             showQuiver=False, save=True, cmap='YlOrRd_r',
#             extent=[0, 12, 0, 4], units={'Lx':'Mm', 'Lz':'Mm'})
# vis.plot_avg('v', folder='random', title=r'Average speed for $\nabla^*=8$')
# vis.plot_avg('rho', folder='random', relative=True, showTrendline=True, title=r'Variation of avg. density for $\nabla^*=8$')
