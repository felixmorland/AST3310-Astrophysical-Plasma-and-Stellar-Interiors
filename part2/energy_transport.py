import matplotlib.pyplot as plt
import numpy as np
from fusion import Core
import requests
from tqdm import tqdm
from scipy import constants as const
import scipy.interpolate as intpol
from cross_section import *
from matplotlib.patches import Patch
import warnings
import matplotlib.gridspec as gridspec
from tabulate import tabulate

class Star:
    def __init__(self, M0, R0, L0, T0, rho0, check=True):
        """
        M0: initial mass [kg]
        R0: initial radius [m]
        L0: initial luminosity [W]
        T0: initial temperature [K]
        rho0: initial density [kg/m^3]
        -------------------------------
        check: (bool) perform sanity check or not
        """
        # Set initial conditions
        self.initial_cond = [M0, R0, L0, T0, rho0]
        self.X = 0.7; self.Y = 0.29; self.Y3 = 1e-10; self.Z = 2e-7+1e-11
        
        # Mean atomic weight
        self.mu = 1/(2*self.X + 0.75*self.Y + self.Y3 + 0.5*self.Z)
        
        # Initial pressure
        self.P0 = self.pressure(T0, rho0)
        self.get_opacities(
            'https://www.uio.no/studier/emner/matnat/astro/AST3310/v25/projects/project2/opacity.txt'
        )
        self.get_epsilons(
            'https://www.uio.no/studier/emner/matnat/astro/AST3310/v25/projects/project2/epsilon.txt'
        )
        # Relevant constants
        self.alpha_lm = 1; self.delta = 1
        self.c_P = 5*const.k/(2*self.mu*const.m_u)
        
        if check:
            self.sanity_check()
        
    
    
    def sanity_check(self):
        """
        Performs sanity check.
        """
        # Kappa sanity check
        # expected_kappa = [[logT, logR, logkappa, kappa[SI]]]
        expected_kappa = np.array([
            [3.750, -6.00, -1.55, 2.84e-3],
            [3.755, -5.95, -1.51, 3.11e-3],
            [3.755, -5.80, -1.57, 2.68e-3],
            [3.755, -5.70, -1.61, 2.46e-3],
            [3.755, -5.55, -1.67, 2.12e-3],
            [3.770, -5.95, -1.33, 4.70e-3],
            [3.780, -5.95, -1.20, 6.25e-3],
            [3.795, -5.95, -1.02, 9.45e-3],
            [3.770, -5.80, -1.39, 4.05e-3],
            [3.775, -5.75, -1.35, 4.43e-3],
            [3.780, -5.70, -1.31, 4.94e-3],
            [3.795, -5.55, -1.16, 6.89e-3],
            [3.800, -5.50, -1.11, 7.69e-3]
        ])
        calculated_kappa = []
        for logT, logR, _, _ in expected_kappa:
            T = 10**logT
            rho = 1e3 * 10**logR * (T*1e-6)**3
            calculated_kappa.append(self.kappa(T,rho))
        calculated_kappa = np.array(calculated_kappa)
        
        rel_diff = np.abs((expected_kappa[:,3]-calculated_kappa)/expected_kappa[:,3])
        
        table_kappa = tabulate([[f'{expected_kappa[i][0]:.3f}', f'{expected_kappa[i][1]:.3f}',
                                   f'{expected_kappa[i][3]:.3g}', f'{calculated_kappa[i]:.3g}', f'{rel_diff[i]:.3f}'] for i in range(len(expected_kappa))],
                                 headers=['logT', 'logR', 'Calculated', 'Expected', 'Rel diff'], tablefmt="pretty")
        print('Kappa values\n'+'-'*12)
        print(table_kappa+'\n')
        if np.any(rel_diff>0.05):
            warnings.warn('Kappa check failed!\n')
        
        
        # Epsilon sanity check
        # expected_epsilon = [[logT, logR, logepsilon, epsilon[SI]]]
        expected_epsilon = np.array([
            [3.750, -6.00, -87.995, 1.012e-92],
            [3.755, -5.95, -87.623, 2.415e-92]
        ])
        calculated_epsilon = []
        for logT, logR, _, _ in expected_epsilon:
            T = 10**logT
            rho = 1e3 * 10**logR * (T*1e-6)**3
            calculated_epsilon.append(self.epsilon(T,rho))
        calculated_epsilon = np.array(calculated_epsilon)
        
        rel_diff = np.abs((expected_epsilon[:,3]-calculated_epsilon)/expected_epsilon[:,3])
        
        table_epsilon = tabulate([[f'{expected_epsilon[i][0]:.3f}', f'{expected_epsilon[i][1]:.3f}',
                                   f'{expected_epsilon[i][3]:.3g}', f'{calculated_epsilon[i]:.3g}', f'{rel_diff[i]:.3f}'] for i in range(len(expected_epsilon))],
                                 headers=['logT', 'logR', 'Calculated', 'Expected', 'Rel diff'], tablefmt="pretty")
        print('Epsilon values\n'+'-'*14)
        print(table_epsilon+'\n')
        if np.any(rel_diff>0.05):
            warnings.warn('Epsilon check failed!\n')
        
        
        # EXAMPLE 5.1
        mu_copy = self.mu
        c_P_copy = self.c_P
        self.mu = 0.6
        self.c_P = 5*const.k/(2*self.mu*const.m_u)
        
        # Given parameters
        T = 0.9e6
        rho = 55.9
        R = 0.84*6.96e8
        M = 0.99*1.989e30
        L = 3.846e26
        P = self.pressure(T,rho)
        kappa = 3.98
        
        # Calculate gradients
        nabla_stable = self.nabla_stable(M,P,L,T,kappa=kappa)
        nabla_ad = self.nabla_ad(P,T)
        nabla_star, xi = self.nabla_star(M,R,P,T,nabla_stable,nabla_ad,kappa=kappa,return_xi=True)
        nabla_p = nabla_star-xi**2
        
        # Pressure scale length
        H_P = P*R**2/(const.G*M*rho)
        
        # U coeff
        U = 64*const.Stefan_Boltzmann*T**3/(3*kappa*rho**2*self.c_P)*np.sqrt(H_P*R**2/(const.G*M)) 
        
        # Speed v of parcel
        v = np.sqrt(const.G*M/(H_P*R**2))*H_P/2*xi
        
        # Fluxes
        F_tot = L/(4*np.pi*R**2)
        F_rad = self.F_rad(M,T,P,kappa,R,nabla_star)
        F_con = F_tot - F_rad
        
        labels = [
            'nabla_stable', 'nabla_star', 'nabla_p', 'nabla_ad',
            'H_P', 'U', 'xi', 'v', 'F_con/F_tot', 'F_rad/F_tot'
        ]
        calculated = np.array([
            nabla_stable, nabla_star, nabla_p, nabla_ad, H_P, U, xi,
            v, F_con/F_tot, F_rad/F_tot 
        ])
        expected = np.array([
            3.26078, 0.400001, 0.4, 0.4, 32.4e6, 5.94e5, 1.173e-3,
            65.5, 0.88, 0.12
        ])
        rel_diff = np.abs((expected-calculated)/expected)
        
        table_51 = tabulate([[labels[i], f'{calculated[i]:.6g}', f'{expected[i]:.6g}', f'{rel_diff[i]:.3f}'] for i in range(len(rel_diff))], 
                headers=['Quantity', 'Calculated', 'Expected', 'Rel diff'], tablefmt="pretty")
        print(f'Example 5.1\n'+'-'*11)
        print(table_51+'\n')
        
        if np.any(rel_diff>0.05):
            warnings.warn('Example 5.1 failed!\n')
        
        # Reset mu and c_P
        self.mu = mu_copy  
        self.c_P = c_P_copy
        
        print(f'mu = {self.mu:.4f}')
        print(f'c_P = {self.c_P:.4g}')
        
           
    def scale_init_vals(self, multipliers):
        """
        Scales initial values with associated multipliers.
        multipliers = [xM0, xR0, xT0, xrho0, xP0, xL0]
        """
        xM0, xR0, xT0, xrho0, xP0, xL0 = multipliers
        M0, R0, L0, T0, rho0 = self.initial_cond
        self.initial_cond = [
                xM0*M0, xR0*R0, xL0*L0, xT0*T0, xrho0*rho0
            ]
        if xrho0 != 1:
            self.P0 = self.pressure(xT0*T0, xrho0*rho0)
        else:
            self.P0 = xP0*self.P0



    def get_opacities(self, url):
        """
        Reads opacity.txt and 2D interpolates the data.
        self.logkappa(logT, logR) is callable to get logkappa values.
        """
        txtfile = requests.get(url)
        text = txtfile.text.split('\n')
        logR = [float(val) for val in text[0].split()[1:]]
        
        data = np.array([
            [float(val) for val in line.split()] for line in text[2:]
        ])
        
        logT = data[:,0]
        logkappa_sample = data[:,1:]
        
        self.logkappa = intpol.RectBivariateSpline(logT, logR, logkappa_sample, kx=1, ky=1)     
 


    def get_epsilons(self, url):
        """
        Reads opacity.txt and 2D interpolates the data.
        self.logk(logT, logR) is callable to get logkappa values.
        """
        txtfile = requests.get(url)
        text = txtfile.text.split('\n')
        logR = [float(val) for val in text[0].split()[1:]]
        
        data = np.array([
            [float(val) for val in line.split()] for line in text[2:]
        ])
        
        logT = data[:,0]
        logepsilon_sample = data[:,1:]
        
        self.logepsilon = intpol.RectBivariateSpline(logT, logR, logepsilon_sample)
    
    
    
    def kappa(self, T, rho):
        """Takes T [K] and rho [kg m^-3] and returns kappa [m^2 kg^-1]"""
        logT = np.log10(T)
        logR = np.log10(1e-3*rho/(T/1e6)**3)
        return 10**(self.logkappa(logT, logR)[0,0]-1)
 
 
    
    def epsilon(self, T, rho):
        """Takes T[K] and rho[kg m^-3] and returns epsilon[W kg^-1]"""
        # From interpolation
        logT = np.log10(T)
        logR = np.log10(1e-3*rho/(T/1e6)**3)
        epsilon = 10**(self.logepsilon(logT, logR)[0,0]-4)
        
        # From fusion.py (project 1): didn't work :(
        # energy_generator = Core(T, rho, check=False) 
        # epsilon = np.sum(energy_generator.calc_chain_energies())
        return epsilon
 
    
    
    def density(self, P, T):
        """Calculates density rho[kg/m^3] given P[Pa] and T[K]"""
        sigma = const.Stefan_Boltzmann
        return self.mu*const.m_u/(const.k*T)*(P-(4/3)*sigma/const.c * T**4)
 
 
    
    def pressure(self, T, rho):
        """Calculates pressure P[Pa] given T[K] and rho[kg/m^3]"""
        sigma = const.Stefan_Boltzmann
        P_gas = rho*const.k*T/(self.mu*const.m_u)
        P_rad = (4/3)*sigma/const.c * T**4
        return P_gas + P_rad



    def nabla_ad(self, P, T):
        """Calculates adiabatic temp gradient."""
        delta = self.delta; c_P = self.c_P
        rho = self.density(P,T)
        return P*delta/(T*rho*c_P)
    
    def nabla_stable(self, m, P, L, T, kappa=None):
        """Calculates stable temp gradient."""
        sigma = const.Stefan_Boltzmann
        rho = self.density(P,T)
        if kappa == None:
            kappa = self.kappa(T,rho)
        return 3*kappa*P*L/(64*np.pi*const.G*m*sigma*T**4)
    

    def nabla_star(self, m, r, P, T,
                   nabla_stable, nabla_ad, kappa=None, return_xi=False):
        """
        Calculates star temperature gradient.
        Also returns xi if return_xi==True.
        """
        delta = self.delta
        sigma = const.Stefan_Boltzmann
        
        rho = self.density(P, T)
        if kappa == None:
            kappa = self.kappa(T, rho)
        c_P = self.c_P
        g = const.G*m/r**2
        
        U_lm2 = 64*sigma*T**3*g/(3*kappa*c_P*P**2)*np.sqrt(P/(rho*delta))
        coeffs = [1, U_lm2, 4*U_lm2**2, -U_lm2*(nabla_stable-nabla_ad)]
        roots = np.roots(coeffs)
        for xi in roots:
            if xi.imag == np.min(np.abs(roots.imag)):
                nabla_star = xi.real**2 +4*U_lm2*xi.real + nabla_ad
                real_xi = xi.real
        if return_xi:
             return nabla_star, real_xi
        else:
            return nabla_star

    def F_rad(self, m, T, P, kappa, r, nabla_star):
        """Calculates radiative flux."""
        sigma = const.Stefan_Boltzmann
        return 16*sigma*const.G*m*T**4/(3*kappa*P*r**2)*nabla_star
    
    def calc_fluxes(self):
        """Calculates F_con, F_rad, F_tot."""
        _, _, nabla_star = np.rollaxis(self.nablas, axis=1)
        F = np.zeros((len(self.m),3))
        
        for i in range(len(self.m)):
            r, P, L, T = self.r[i], self.P[i], self.L[i], self.T[i]
            rho = self.density(P,T)
            m = self.m[i]
            kappa = self.kappa(T,rho)
            
            F_tot = L/(4*np.pi*r**2)
            F_rad = self.F_rad(m,T,P,kappa,r,nabla_star[i])
            F_con = F_tot-F_rad
            
            F[i] = F_con, F_rad, F_tot  
        return F
    
    
    
    def integrate_pdes(self, p=3e-3, showpbar=True):
        """Integrates the PDE's for the stellar variables"""
        def derivatives(m, variables):
            """RHS of the PDE's"""
            r, P, L, T = variables
            
            rho = self.density(P,T)
            epsilon = self.epsilon(T,rho)
            kappa = self.kappa(T,rho)
            
            drdm = 1/(4*np.pi*rho*r**2)
            dPdm = -const.G*m/(4*np.pi*r**4)
            dLdm = epsilon
            
            nabla_stable = self.nabla_stable(m,P,L,T)
            nabla_ad = self.nabla_ad(P,T)
            nabla_star = self.nabla_star(m, r, P, T,
                                         nabla_stable, nabla_ad)

            if nabla_stable > nabla_ad:
                dTdm = nabla_star*(T/P)*dPdm
            else:
                sigma = const.Stefan_Boltzmann
                dTdm = -3*kappa*L/(256*np.pi**2 * sigma*r**4 * T**3)
            
            new_nablas = [nabla_ad, nabla_stable, nabla_star]
            
            return np.array([drdm, dPdm, dLdm, dTdm]), new_nablas
        
        
        # Initialize
        M0, R0, L0, T0, rho0 = self.initial_cond
        P0 = self.P0
        m_axis = np.array([M0])
        m = M0
        
        initial_variables = np.array([R0, P0, L0, T0])
        initial_nablas = self.nabla_ad(P0, T0), self.nabla_stable(M0,P0,L0,T0),\
                         self.nabla_star(M0,R0,P0,T0,self.nabla_stable(M0,P0,L0,T0),self.nabla_ad(P0, T0))
        
        variables = np.zeros((1,4))
        nablas = np.zeros((1,3))
            
        variables[0] = initial_variables
        nablas[0] = initial_nablas
        
        # Progress bar
        if showpbar:
            pbar = tqdm(desc='Integrating over m', total=1,
                        bar_format = "{desc}: {percentage:.0f}%|{bar}| {n:.2g}/{total:.2g} [{elapsed}<{remaining}")
        
        while m > 0:
            # Old values
            old_vars = variables[-1]
            # Get RHS of PDE's and new temp gradients
            dvardm, new_nablas = derivatives(m, old_vars)
            # Determine dm
            dm_candidates = [np.abs(p*old_vars[i]/dvardm[i]) for i in range(4)]   
            dm = -np.min(dm_candidates)
            # Advance one step in m
            new_vars = old_vars + dvardm*dm
            variables = np.vstack([variables, new_vars])
            nablas = np.vstack([nablas, new_nablas])
            
            m += dm
            m_axis = np.concatenate([m_axis, [m]])
            
            # Conditions for success
            cond1 = (m/M0 < 0.05)
            cond2 = (new_vars[0]/R0 < 0.05)
            cond3 = (new_vars[2]/L0 < 0.05)
            
            if showpbar:  
                pbar.update(-dm/M0)
            if cond1 and cond2 and cond3:
                break
            elif abs(dm) < 1e6:
                print('Conditions saturated! dm < 1e6')
                break
        
        self.r, self.P, self.L, self.T = np.rollaxis(variables, axis=1)
        self.rho = self.density(self.P, self.T)
        self.m = m_axis
        self.nablas = nablas
        self.F = self.calc_fluxes()
        
        if showpbar:
            print('Final values')
            print(f'm/M0 = {m/M0:.3f}')
            print(f'L/L0 = {self.L[-1]/L0:.3f}')
            print(f'r/R0 = {self.r[-1]/R0:.3f}')
            print(f'T = {self.T[-1]:.3g} K')
            print(f'P = {self.P[-1]:.3g} Pa')
            print(f'rho = {self.rho[-1]:.3g} kg/m^3')
            pbar.close()

        

    def plot_nablas(self):
        """Plot temp gradients"""
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        fig.suptitle(r'Temperature gradients of optimal star', fontsize=20)
        nabla_ad, nabla_stable, nabla_star = np.rollaxis(self.nablas, axis=1)
        
        for i, nabla_stable_i in enumerate(nabla_stable):
            if nabla_stable_i <= nabla_ad[i]:
                nabla_star[i] = nabla_stable_i

        ax.plot(self.r/self.initial_cond[1], nabla_stable, label=r'$\nabla_\mathrm{stable}$')
        ax.plot(self.r/self.initial_cond[1], nabla_star, linestyle='dashed', label=r'$\nabla^*$')
        ax.plot(self.r/self.initial_cond[1], nabla_ad, linestyle='-.', label=r'$\nabla_\mathrm{ad}$')
        
        ax.set_xlabel(r'$r/R_0$', fontsize=18)
        ax.set_ylabel(r'Temperature gradient $\nabla$', fontsize=18)
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend(fontsize=18)
        


    def plot_variables(self):
        """Plot stellar variables"""
        fig, axs = plt.subplots(2,3,figsize=(10,6))
        r_axis = self.r/self.initial_cond[1]
        axs[0,0].plot(r_axis, self.m/self.initial_cond[0], color='black')
        axs[0,0].set_title(r'Mass, $M_0=M_\odot$')
        axs[0,0].set_ylabel(r'$m/M_0$')
        
        axs[0,1].plot(r_axis, self.T, color='black')
        axs[0,1].set_title(r'Temperature, $T_0=5770$ K')
        axs[0,1].set_ylabel(r'$T$ [K]')
        
        axs[1,0].plot(r_axis, self.L/self.initial_cond[2], color='black')
        axs[1,0].set_title(r'Luminosity, $L_0=1.2L_\odot$')
        axs[1,0].set_ylabel(r'$L/L_0$')
        
        axs[1,1].plot(r_axis, self.rho/self.initial_cond[4], color='black')
        axs[1,1].set_title(r'Density, $\rho_0=3.55\times 10^{-6}\cdot \bar{\rho}_\odot$')
        axs[1,1].set_ylabel(r'$\rho/\rho_0$')
        axs[1,1].set_yscale('log')
        
        axs[0,2].plot(r_axis, self.P, color='black')
        axs[0,2].set_title(r'Pressure, $P_0=$'+f' {self.P0:.2g} Pa')
        axs[0,2].set_ylabel(r'$P$ [Pa]')
        axs[0,2].set_yscale('log')
        
        fig.delaxes(axs[1,2]) 
        
        for i in range(5):
            axs.flat[i].minorticks_on()
            axs.flat[i].grid(True, which='major')
            axs.flat[i].grid(True, which='minor', alpha=0.3)
            axs.flat[i].set_xlabel(r'$r/R_0$')
        
        fig.supxlabel(r'$R_0=1.3R_\odot$')
        plt.tight_layout()
        
            
            
            
    def plot_energy_prod(self):
        """Plot energy production"""
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        energies = np.zeros((len(self.T), 4))
        for i in tqdm(range(1,len(self.T)), desc='Epsilon(r/R0)'):
            star_core = Core(self.T[i], self.rho[i], check=False)
            pp1, pp2, pp3, cno = star_core.calc_chain_energies()
            energies[i] = pp1, pp2, pp3, cno
        
        r_axis = self.r/self.initial_cond[1]
        total_energy = np.sum(energies, axis=1)
        
        ax.plot(r_axis, energies[:,0]/total_energy, label=r'$\varepsilon_\mathrm{PPI}$')
        ax.plot(r_axis, energies[:,1]/total_energy, label=r'$\varepsilon_\mathrm{PPII}$')
        ax.plot(r_axis, energies[:,2]/total_energy, label=r'$\varepsilon_\mathrm{PPIII}$')
        ax.plot(r_axis, energies[:,3]/total_energy, label=r'$\varepsilon_\mathrm{CNO}$')
        ax.plot(r_axis, total_energy/np.max(total_energy), label=r'$\varepsilon_\mathrm{tot}/\mathrm{max}(\varepsilon_\mathrm{tot})$')
        ax.set_xlabel(r'$r/R_0$', fontsize=18)
        ax.set_ylabel(r'$\varepsilon/\varepsilon_\mathrm{tot}$', fontsize=18)
        ax.set_xscale('log')
        ax.grid(True, which='major')
        ax.grid(True, which='minor', alpha=0.2)
        
        fig.suptitle(r'Energy production from chains in optimal star', fontsize=20)
        fig.legend(loc='outside lower center', fancybox=True, shadow=True, ncol=3, fontsize=18)
        plt.tight_layout()



    def plot_fluxes(self):
        """Plot radiative, convective and total flux"""
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        
        F_con, F_rad, F_tot = np.rollaxis(self.F, axis=1)
        F_rad[F_con<0] = F_tot[F_con<0]
        F_con[F_con<0] = 0
        r_axis = self.r/self.initial_cond[1]
        
        ax.plot(r_axis, F_con/F_tot, label=r'$F_\mathrm{con}$')
        ax.plot(r_axis, F_rad/F_tot, label=r'$F_\mathrm{rad}$')
        ax.plot(r_axis, F_tot/np.max(F_tot), label=r'$F_\mathrm{tot}/\mathrm{max}(F_\mathrm{tot})$')
        
        ax.set_xlabel(r'$r/R_0$', fontsize=18)
        ax.set_ylabel(r'$F/F_\mathrm{tot}$', fontsize=18)
        ax.set_xscale('log')
        ax.grid(True, which='major')
        ax.grid(True, which='minor', alpha=0.2)
        
        fig.suptitle(r'Energy flux in optimal star', fontsize=20)
        fig.legend(loc='outside lower center', fancybox=True, shadow=True, ncol=3, fontsize=18)
        plt.tight_layout()
        


    def plot_star_zone(self, ax=None, showlegend=True):
        """Plot 1D version of cross section"""
        if ax == None:
            fig, ax = plt.subplots(1,1,figsize=(8,3))
        # Luminosity and convective flux
        L = self.L; F_C = self.F[:,0]
        core_limit = 0.995 * np.max(L)
        # Star zone: 3 con outside core, 2 rad outside core,
        #            1 rad inside core, 0 con inside core 
        star_zone = np.int32(np.where(L>core_limit,0.5,-0.5) * np.where(F_C>0,3,1) + 2)
        colors = ['blue','cyan','yellow','red']
        zone_proportions = np.zeros(4)
        R0 = self.initial_cond[1]
        
        # Find boundaries of zones
        star_zone_prev = -1
        zone_limits = []
        for k in range(0,len(self.r)):
            if star_zone[k]!=star_zone_prev: 
                star_zone_prev = star_zone[k]
                zone_limits.append([self.r[k], star_zone[k]])
        last_zone = zone_limits[-1][1]
        zone_limits.append([self.r[-1], last_zone])
        
        # Plot zones as rectangles
        for i in range(1, len(zone_limits)):
            r_min = zone_limits[i-1][0]
            r_max = zone_limits[i][0]
            zone = zone_limits[i-1][1]
            zone_proportions[zone] += abs(r_max-r_min)/R0
            ax.axvspan(r_min/R0, r_max/R0, facecolor=colors[zone], edgecolor='none')
        
        ax.set_xlabel(r'$r/R_0$', fontsize=16)
        ax.set_yticks([])
        ax.set_xlim(0,1)

        if showlegend:
            plt.subplots_adjust(bottom=0.45, top=0.77)
            fig.suptitle(r'Star zone of optimal star', fontsize=20)
            ax.set_title(r'$\chi_{R_0}=1.3$, $\chi_{\rho_0}=25$, $\chi_{L_0}=1.2$')
            handles = [Patch(facecolor=colors[i]) for i in range(4)]
            labels = [r'Convection inside core'+f' {100*zone_proportions[0]:.1f}'+r'\%', r'Radiation inside core'+f' {100*zone_proportions[1]:.1f}'+r'\%',
                    r'Radiation outside core'+f' {100*zone_proportions[2]:.1f}'+r'\%', r'Convection outside core'+f' {100*zone_proportions[3]:.1f}'+r'\%']
            fig.legend(handles, labels, loc='outside lower center', fancybox=True, shadow=True, ncol=2, fontsize=15)



    def test_init_vals(self):
        """Test the effect of scaling the initial values"""
        fig, axs = plt.subplots(3, 2, figsize=(6,8))
        titles = [r'$M_0$', r'$R_0$', r'$T_0$', r'$\rho_0$', r'$P_0$', r'$L_0$']
        test_multipliers = [0.1, 1, 10]
        lstyle = ['solid', 'dashed', 'dotted']
        colors = ['black', 'blue', 'red']
        
        init_vals_copy = self.initial_cond
        P0_copy = self.P0
        for i, x in enumerate(test_multipliers):
            for j in tqdm(range(6), desc=f'Multiplying parameters by {test_multipliers[i]}'):
                self.initial_cond = init_vals_copy
                self.P0 = P0_copy

                multipliers = 6*[1]
                multipliers[j] = x
                
                self.scale_init_vals(multipliers)
                try:
                    self.integrate_pdes(showpbar=False)
                except:
                    continue
                
                L = self.L; F_C = self.F[:,0]
                core_limit = 0.995 * np.max(L)
                star_zone = np.int32(np.where(L>core_limit,0.5,-0.5) * np.where(F_C>0,3,1) + 2)
                
                axs.flat[j].plot(self.r/self.initial_cond[1], star_zone,
                                 linestyle=lstyle[i], color=colors[i])
        
        # Reset initial condition
        self.initial_cond = init_vals_copy
        
        for i in range(6):
            axs.flat[i].set_title(titles[i], fontsize=18)
            axs.flat[i].set_xlabel(r'$r/R_0$', fontsize=16)
            axs.flat[i].set_ylabel(r'Star zone', fontsize=16)
            axs.flat[i].label_outer()
        
        lines = [plt.Line2D([0], [0], color=colors[i], linestyle=lstyle[i]) for i in range(len(test_multipliers))]
        labels = [r'$\chi_\mathrm{par}=$'+f' {test_multipliers[i]}' for i in range(len(test_multipliers))]
        fig.legend(lines, labels, loc='outside lower center', fancybox=True, shadow=True, ncol=3, fontsize=14)
        plt.tight_layout()
 
        
        
    def test_rho_P_scaling(self):
        """
        Test scaling rho and P by bigger factors.
        Shows the effect by plotting the 1D cross sections.
        """
        test_multipliers = [1, 50, 500, 750]
        
        fig = plt.figure(figsize=(6,8))
        outer = gridspec.GridSpec(2, 1, hspace=0.4)
        titles = [r'Scaling $\rho_0$ by $\chi$', r'Scaling $P_0$ by $\chi$']
                
        init_vals_copy = self.initial_cond
        P0_copy = self.P0
        
        for j in tqdm(range(2)):
            outer_ax = plt.Subplot(fig, outer[j])
            outer_ax.set_title(titles[j], fontsize=20)
            outer_ax.set_yticks([])
            fig.add_subplot(outer_ax)
            
            inner = gridspec.GridSpecFromSubplotSpec(len(test_multipliers), 1,
                    subplot_spec=outer[j], wspace=0.1, hspace=0.1)
            
            for i, x in enumerate(test_multipliers):
                ax = plt.Subplot(fig, inner[i])
                self.initial_cond = init_vals_copy
                self.P0 = P0_copy

                multipliers = 6*[1]
                multipliers[j+3] = x
                
                self.scale_init_vals(multipliers)
                try:
                    self.integrate_pdes(showpbar=False)
                except:
                    continue
                
                self.plot_star_zone(ax, showlegend=False)
                ax.set_ylabel(r'$\chi =$'+f' {x}', rotation=0, ha='right')
                ax.label_outer()
                fig.add_subplot(ax)
            
        outer.tight_layout(fig)
        outer.update(bottom=0.17)
        colors = ['blue','cyan','yellow','red']
        handles = [Patch(facecolor=colors[i]) for i in range(4)]
        labels = [r'Convection inside core', r'Radiation inside core',
                  r'Radiation outside core', r'Convection outside core']
        fig.legend(handles, labels, loc='outside lower center', fancybox=True, shadow=True, ncol=2, fontsize=15)
            

# Solar parameters
M_sun = 1.989e30
R_sun = 6.96e8
L_sun = 3.846e26
avg_rho_sun = 1.408e3

# Initial conditions
M0 = M_sun
R0 = R_sun
L0 = L_sun
T0 = 5770
rho0 = 1.42e-7*avg_rho_sun

# Create star
star = Star(M0, R0, L0, T0, rho0)

# Test scaling of initial values
# star.test_init_vals()
# star.test_rho_P_scaling()

# Optimal star
star.scale_init_vals([1,1.3,1,25,1,1.2])

# Integrate the
star.integrate_pdes(p=3e-3)

# Plot gradients
star.plot_nablas()

# Plot stellar variables
star.plot_variables()

# Plot cross section
cross_section(star.r, star.L, star.F[:,0])

# Plot 1D version of cross section
star.plot_star_zone()

# Plot energy production
star.plot_energy_prod()

# Plot fluxes
star.plot_fluxes()

plt.show()