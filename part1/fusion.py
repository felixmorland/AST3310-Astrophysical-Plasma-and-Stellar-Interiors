import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const
from tabulate import tabulate
from copy import deepcopy
from tqdm import tqdm



class Core:
    def __init__(self, temp:float, density:float, check=True):
        """
        temp: temperature of the stellar core.
        density: density of the stellar core.
        check: (bool) True if sanity check is needed.
        """
        # Set temperature T and mass density rho
        # Temperature should be given with units K
        # Density with units kg m^-3
        self.T = temp; self.rho = density
        
        # Mass fractions in the core
        # X: Hydrogen, Y: Helium 4, Y3: Helium 3
        self.X = 0.7; self.Y = 0.29; self.Y3 = 1e-10
    
        self.Z_Li7 = 1e-7      # Li7: Lithium 7
        self.Z_Be7 = 1e-7      # Be7: Beryllium 7
        self.Z_N14 = 1e-11     # N14: Nitrogen 14
        
        mu = const.m_u
        # Masses of the nuclei and electron
        self.masses = {
            'H': 1.00784*mu, 'He3': 3.0160293*mu, 'He4': 4.002603254*mu,
            'Be7': 7.01692871*mu, 'Be8': 8.00530510*mu,'Li7': 7.016003434*mu, 
            'N14': 14.003074004251*mu, 'e': 5.4858e-4*mu, 'D2': 2.01410177811*mu, 
            'B8': 8.0246073*mu, 'C12': 12*mu, 'N13': 13.00573861*mu, 'C13': 13.003355*mu, 
            'O15': 15.0030656*mu, 'N15': 15.000108898266*mu, 'e+': 5.485799090441e-4*mu
        }
        
        # Number densities
        n_e = density*(1 + self.X)/(2*mu)    # Electrons (assuming full ionization)
        n_H = density*self.X/(mu)            # Hydrogen
        n_He4 = density*self.Y/(4*mu)        # Helium 4
        n_He3 = density*self.Y3/(3*mu)       # Helium 3
        n_Li7 = density*self.Z_Li7/(7*mu)    # Lithium 7
        n_Be7 = density*self.Z_Be7/(7*mu)    # Beryllium 7
        n_N14 = density*self.Z_N14/(14*mu)   # Nitrogen 14
        
        # Number densities in dictionary
        self.num_densities = {
            'e':n_e, 'H':n_H, 'He4':n_He4, 'He3':n_He3, 'Li7':n_Li7,
            'Be7':n_Be7, 'N14':n_N14
        }
        
        # Proton numbers
        self.Z = {
            'H': 1, 'He3': 2, 'He4': 2, 'Li7': 3, 'Be7': 4, 'N14': 7
        }
        
        # Nuclear reactions in the core.
        # Corresponds to the indices of the differents steps in the PP chain
        # and the CNO as given on p.33 of the lecture notes.
        #
        # 'name':[[nucleus 1, nucleus 2], [products*], Q']
        # Q' given in MeV
        self.nuc_reactions = {
            'pp':[['H','H'], ['He3','e','nu_e'], 1.177+5.494],
            '33':[['He3','He3'], ['He4','H','H'], 12.86],
            '34':[['He3','He4'], ['Be7'], 1.586],
            'e7':[['Be7','e'], ['Li7','nu_e'], 0.049],
            "17'":[['Li7','H'], ['He4','He4'], 17.346],
            '17':[['Be7','H'], ['He4', 'He4', 'e', 'nu_e'], 0.137+11.27+0.092],
            'p14':[['N14','H'], [], 25.028]
        }
        # Total generated energy from mass diff 4H - He4
        Q_4H_He4 = 26.732
        self.varrho_nu = {
            'PPI': 2*0.265/Q_4H_He4*100, 'PPII': (0.265+0.815)/(2*Q_4H_He4)*100,
            'PPIII': (0.265+6.711)/(2*Q_4H_He4)*100, 'CNO': (0.707+0.997)/Q_4H_He4*100
        }
         
        # Reference values for the Q' energies
        self.Q_reference = {
            'pp': 1.177, 'pd': 5.494, '33': 12.860, '34': 1.586, 
            'e7': 0.049, "17'": 17.346, '17': 0.137, '8': 11.270,
            "8'": 0.092, 'p12': 1.944, '13': 1.513, 'p13': 7.551,
            'p14': 7.297, '15': 1.757, 'p15': 4.966
        }
        
        # Calculate the Q' values
        self.calc_Q()
        
        # Find reaction energies
        self.reaction_energies()
        
        # Perform sanity check
        if check:
            self.sanity_check()
        
  
        
    def __str__(self):
        """
        Printing the Core object will display information about the different
        reaction rates
        ------------------------------
        Returns: info (str)
        """
        core_information = '\nINFORMATION ABOUT THE CORE\n'+'-'*26+'\n'
        core_information += f'Temperature: {self.T:.2e} K\nDensity: {self.rho:.2e} kg m^-3\n'
        core_information += tabulate([[key, f'{value:.3e}'] for key, value in self.lmbdas.items()], 
                                    headers=['Reaction', 'Lambda [cm^3 s^-1]'], tablefmt="pretty") + '\n\n'

        core_information += tabulate([[key, f'{value:.3e}'] for key, value in self.rates.items()], 
                                    headers=['Reaction', 'Reaction rate [kg^-1 s^-1]'], tablefmt="pretty") + '\n\n'
        
        core_information += tabulate([[key, f'{value:.3e}'] for key, value in self.energies.items()], 
                                    headers=['Reaction', 'Energy production [J m^-3 s^-1]'], tablefmt="pretty") + '\n\n'
        
        core_information += tabulate([[key, f'{self.Q[key]:.3f}', f'{self.Q_reference[key]:.3f}',\
                                    f'{(self.Q[key]-self.Q_reference[key])/self.Q_reference[key]:.3f}'] for key in list(self.Q.keys())], 
                                    headers=['Reaction', "Calculated Q' [MeV]", "Reference Q' [MeV]", "Rel diff"],
                                    tablefmt="pretty") + '\n\n'
        
        core_information += tabulate([[key, f'{value:.3f}'] for key, value in self.varrho_nu.items()], 
                                    headers=['Branch/cycle', '% of energy lost to neutrinos'], tablefmt="pretty") + '\n\n'
        
        return core_information
    
    
    
    def __call__(self, *args, **kwargs):
        pass
    
    
    
    def update_attr(self, **kwargs):
        """
        Meant to be used for updating the temperature or density (or both) and 
        recalculate the reaction energies, reaction rates and lambda proportionality functions.
        """
        self.__dict__.update(kwargs)
        del self.lmbdas, self.rates
        self.reaction_energies()
            


    def calc_Q(self):
        """
        Calculates the Q' for each individual reaction using the mass
        difference: Q' = delta m * c^2 [MeV].
        """
        self.Q = {}
        m = self.masses
        J_to_MeV = 6.242e+12
        
        self.Q['pp'] = (2*m['H']-m['D2']-m['e+'])*(const.c**2)*J_to_MeV + 1.022 - 0.265
        self.Q['pd'] = (m['D2']+m['H']-m['He3'])*(const.c**2)*J_to_MeV
        self.Q['33'] = (2*m['He3']-m['He4']-2*m['H'])*(const.c**2)*J_to_MeV
        self.Q['34'] = (m['He3']+m['He4']-m['Be7'])*(const.c**2)*J_to_MeV
        self.Q['e7'] = (m['Be7']+m['e']-m['Li7'])*(const.c**2)*J_to_MeV - 0.815
        self.Q["17'"] = (m['Li7']+m['H']-2*m['He4'])*(const.c**2)*J_to_MeV
        self.Q['17'] = (m['Be7']+m['H']-m['B8'])*(const.c**2)*J_to_MeV 
        self.Q['8'] = (m['B8']-m['Be8']-m['e+'])*(const.c**2)*J_to_MeV + 1.022 - 6.711
        self.Q["8'"] = (m['Be8']-2*m['He4'])*(const.c**2)*J_to_MeV
        
        self.Q['p12'] = (m['C12']+m['H']-m['N13'])*(const.c**2)*J_to_MeV
        self.Q['13'] = (m['N13']-m['C13']-m['e+'])*(const.c**2)*J_to_MeV + 1.022 - 0.707
        self.Q['p13'] = (m['C13']+m['H']-m['N14'])*(const.c**2)*J_to_MeV
        self.Q['p14'] = (m['N14']+m['H']-m['O15'])*(const.c**2)*J_to_MeV
        self.Q['15'] = (m['O15']-m['N15']-m['e+'])*(const.c**2)*J_to_MeV + 1.022 - 0.997
        self.Q['p15'] = (m['N15']+m['H']-m['C12']-m['He4'])*(const.c**2)*J_to_MeV
    
    
            
    def prop_funcs(self):
        """
        Calculates the proportionality functions of the different
        nuclear reactions in the core of the star given the temperature.
        The expressions for the different lambdas come from p.32 in the lecture notes.
        The lambdas have units [cm^3 s^-1]
        ----------------------------------------------------------
        Returns; lmbdas (dict): dictionary containing the different lambdas.
        """
        T9 = self.T*1e-9            # Temperature in units 10^9 K
        T9a = T9/(1 + 4.95e-2*T9)   # T9* (a = asterisk)
        T9aa = T9/(1+0.759*T9)      # T9** (aa = double asterisk)
        
        N_A = const.N_A     # Avogadro's constant
        
        # Proton + Proton -> Deuterium
        lmbda_pp = 4.01e-15*T9**(-2/3)*np.exp(-3.38*T9**(-1/3))\
                    *(1 + 0.123*T9**(1/3) + 1.09*T9**(2/3) + 0.938*T9)/N_A
        
        # Helium 3 + Helium 3 -> Helium 4 + 2x Protons
        lmbda_33 = 6.04e10*T9**(-2/3)*np.exp(-12.276*T9**(-1/3))\
                    *(1 + 0.034*T9**(1/3) - 0.522*T9**(2/3) - 0.124*T9\
                    + 0.353*T9**(4/3) + 0.213*T9**(5/3))/N_A
        
        # Helium 3 + Helium 4 -> Beryllium 7
        lmbda_34 = 5.61e6*T9a**(5/6)*T9**(-3/2)*np.exp(-12.826*T9a**(-1/3))/N_A
        
        # Beryllium 7 + e- -> Lithium 7
        lmbda_e7 = (1.34e-10*T9**(-1/2)*(1 - 0.537*T9**(1/3) + 3.86*T9**(2/3)\
                    + 0.0027*T9**(-1)*np.exp(2.515e-3*T9**(-1))))/N_A
        # Account for upper limit of electron capture by Be7
        if self.T <= 1e6:
            lmbda_e7_max = 1.51e-7/self.num_densities['e']*1e6
            lmbda_e7 = np.min([lmbda_e7_max, lmbda_e7])
        
        
        # Lithium 7 + Proton -> 2x Helium 4
        # (p = prime)
        lmbda_17p = (1.096e9*T9**(-2/3)*np.exp(-8.472*T9**(-1/3)) - 4.83e8*T9aa**(5/6)\
                    *T9**(-3/2)*np.exp(-8.472*T9aa**(-1/3)) + 1.06e10*T9**(-3/2)\
                    *np.exp(-30.442*T9**(-1)))/N_A
        
        # Beryllium 7 + Proton -> Boron 8
        lmbda_17 = (3.11e5*T9**(-2/3)*np.exp(-10.262*T9**(-1/3))\
                    + 2.53e3*T9**(-3/2)*np.exp(-7.306*T9**(-1)))/N_A
        
        # Nitrogen 14 + Proton -> Oxygen 15
        lmbda_p14 = (4.9e7*T9**(-2/3)*np.exp(-15.228*T9**(-1/3) - 0.092*T9**2)\
                    *(1 + 0.027*T9**(1/3) - 0.778*T9**(2/3) - 0.149*T9 + 0.261*T9**(4/3)\
                    + 0.127*T9**(5/3)) + 2.37e3*T9**(-3/2)*np.exp(-3.011*T9**(-1))\
                    + 2.19e4*np.exp(-12.53*T9**(-1)))/N_A

        # The proportionality functions in dictionary for later use
        lmbdas = {
            'pp': lmbda_pp, '33': lmbda_33, '34': lmbda_34, 'e7': lmbda_e7,
            "17'": lmbda_17p, '17': lmbda_17, 'p14': lmbda_p14
        }
        
        # Save the lambdas
        self.lmbdas = lmbdas
    
    
    
    def calc_reaction_rate(self, reaction:str):
        """
        Calculates the reaction rate for the reaction given in the
        argument of the function.
        Reaction rates with units [kg^-1 s^-1]
        ----------------------------------------
        Returns: rate (float) the reaction rate
        """
        # Get lmbda for the reaction
        lmbda = self.lmbdas[reaction]
        
        # Number densities
        n = self.num_densities
        
        # Get the reactants
        reaction_info = self.nuc_reactions[reaction]
        elem1, elem2 = reaction_info[0]
        
        # Calculate reaction rate
        m3_per_cm3 = 1e-6   # Convert units of lambda to m^3 s^-1
        rate = n[elem1]*n[elem2]*lmbda*m3_per_cm3/(self.rho*(1+(elem1==elem2)))
        
        return rate



    def reaction_rates(self):
        """
        Uses the calc_reaction_rate method to calculate the 7 reaction
        rates and then adjusts the rates such that no step consumes more
        than what is produced in the previous step.
        """
        # Check if the lambdas are calculated
        if 'lmbdas' not in self.__dict__.keys():
            self.prop_funcs()
        
        # Reaction rate dictionary
        self.rates = {}
        for reaction in self.nuc_reactions.keys():
            self.rates[reaction] = self.calc_reaction_rate(reaction)
        
        r = deepcopy(self.rates)
        
        # Update r_33 and r_34
        if 2*r['33']+r['34']>r['pp']:
            a = r['33']/r['34']
            r['33'] = r['pp']*a/(2*a+1)
            r['34'] = r['pp']/(2*a+1)
        
        # Update r_e7 and r_17
        if r['e7']+r['17']>r['34']:
            b = r['e7']/r['17']
            r['e7'] = r['34']*b/(b+1)
            r['17'] = r['34']/(b+1)
        
        # r_17' must be less than or equal to r_e7
        r["17'"] =  np.min([r["17'"], r['e7']])
        
        # Save the updated reaction rates
        self.rates = r
 
 
    
    def calc_energy(self, reaction:str):
        """
        Calculates the energy released in the given reaction.
        -----------------------------------------------------
        Returns: epsilon (float) energy released per volume per second [J m^-3 s^-1]
        """
        # Get reaction rate
        rate = self.rates[reaction]
        
        elem1, elem2 = self.nuc_reactions[reaction][0]
        products = self.nuc_reactions[reaction][1]
        
        # Convert released energy to Joules
        J_per_MeV = 1.60218e-13
        Q = self.nuc_reactions[reaction][2]*J_per_MeV
        
        # Calculate the energy production
        epsilon = rate*Q*self.rho
        
        return epsilon
 
 
        
    def reaction_energies(self):
        """
        Calculate the energy production for each reaction using the
        calc_energy method, then save them in a dictionary.
        """
        # Make sure the reaction rates are calculated
        if 'rates' not in self.__dict__.keys():
            self.reaction_rates()
        
        # Energy dictionary
        self.energies = {}
        for reaction in self.nuc_reactions.keys():
            self.energies[reaction] = self.calc_energy(reaction)
 
 
    
    def gamow(self):
        """
        Produces a plot of the relevant Gamow peaks in the energy range [10^-17, 10^-13] J.
        """
        N = int(1e4)                    # Number of datapoints
        E = np.logspace(-17, -13, N)    # Energy axis
        
        # Constant inside exponential
        k = const.elementary_charge**2*np.pi/(const.epsilon_0*const.h)
        
        peaks = {}
        for reaction in self.nuc_reactions.keys():
            # e7 is not a fusion reaction, so skip.
            if reaction == 'e7':
                continue
            
            # Get the nuclei in the reaction
            elem1, elem2 = self.nuc_reactions[reaction][0]
            # Proton numbers
            Zi, Zk = self.Z[elem1], self.Z[elem2]
            # Masses
            mi, mk = self.masses[elem1], self.masses[elem2]
            # Reduced mass
            m = mi*mk/(mi+mk)
            
            # Calculate the distribution
            peaks[reaction] = np.exp(-E/(const.k*self.T))*np.exp(-np.sqrt(m/(2*E))*Zi*Zk*k)
        
        # Plotting
        for reaction in peaks.keys():
            plt.semilogx(E, peaks[reaction]/np.max(peaks[reaction]), label=reaction)

        plt.grid(True, which='both', alpha=0.5)
        plt.suptitle('Gamow peaks of relevant reactions', fontsize=14)
        plt.xlabel('Energy $E$ [J]', fontsize=14)
        plt.ylabel('Probability density $P(E)/\\text{max}[P(E)]$', fontsize=14)
        plt.legend()
        plt.show()
        


   
    def calc_chain_energies(self):
        """
        Calculates the energy released from each completed branch of the PP chains
        and the CNO cycle. Units [J m^-3 s^-1]
        """
        r = self.rates              # Reaction rates
        info = self.nuc_reactions   # Info about the nuclear reactions]
        
        # Energy production from PPI
        pp1_energy = r['33']*(info['33'][2]+2*info['pp'][2])
        
        # Energy production from PPII
        pp2_energy = r["17'"]*(info["17'"][2]+info['e7'][2]+info['34'][2]\
                    +info['33'][2]+info['pp'][2])
        
        # Energy production from PPIII
        pp3_energy = r['17']*(info['17'][2]+info['34'][2]\
                    +info['33'][2]+info['pp'][2])
        
        # Energy production from CNO cycle
        cno_energy = r['p14']*info['p14'][2]
        
        self.chain_energies = pp1_energy, pp2_energy, pp3_energy, cno_energy
        
        return pp1_energy, pp2_energy, pp3_energy, cno_energy


    
    def sanity_check(self):
        """
        Perform sanity check.
        """
        # Save original temperature and density
        T_copy = self.T; rho_copy = self.rho
        
        # First test
        self.update_attr(T=1.57e7, rho=1.62e5)
        expected1 = np.array([4.05e2, 8.69e-9, 4.87e-5, 1.50e-6, 5.30e-4, 1.64e-6, 9.18e-8])
        computed1 = np.array([float(value) for value in self.energies.values()])
        rel_diff1 = (computed1-expected1)/expected1
        error_message1 = f'Sanity check failed for T=1.57e7 K and rho=1.62e5 kg/m^3 !!'
        
        test1_table = tabulate([[list(self.energies.keys())[i], f'{expected1[i]:.2e}', f'{computed1[i]:.2e}',
                                f'{rel_diff1[i]:.3f}'] for i in range(7)], 
                                headers=['Reaction', 'Expected [J m^-3 s^-1]', 'Computed [J m^-3 s^-1]', 'Relative diff'],
                                tablefmt="pretty") + '\n'
            
        print('\nRESULTS FROM SANITY CHECK\n'+25*'-'+'\n')
        print('T=1.57e7 K and rho=1.62e5 kg/m^3\n')
        print(test1_table)
        
        # Second test
        self.update_attr(T=1e8, rho=1.62e5)
        expected2 = np.array([7.34e4, 1.10e0, 1.75e4, 1.23e-3, 4.35e-1, 1.27e5, 3.45e4])
        computed2= np.array([float(value) for value in self.energies.values()])
        rel_diff2 = (computed2-expected2)/expected2
        error_message2 = f'Sanity check failed for T=1.00e8 K and rho=1.62e5 kg/m^3 !!'
        
        test2_table = tabulate([[list(self.energies.keys())[i], f'{expected2[i]:.2e}', f'{computed2[i]:.2e}',
                                f'{rel_diff2[i]:.3f}'] for i in range(7)], 
                                headers=['Reaction', 'Expected [J m^-3 s^-1]', 'Computed [J m^-3 s^-1]', 'Relative diff'],
                                tablefmt="pretty") + '\n'
    
        print('T=1.00e8 K and rho=1.62e5 kg/m^3\n')
        print(test2_table)
        
        # AssertionError if sanity check fails
        assert np.all(np.abs(rel_diff1)<1e-2), error_message1
        assert np.all(np.abs(rel_diff2)<1e-2), error_message2
        
        # Reset temperature and density
        self.update_attr(T=T_copy, rho=rho_copy)

    

def plot_chain_energies(T_min, T_max, rho=1.62e5):
    """
    Produce a plot of the energy production of each PP branch, the
    CNO cycle, and the total energy production as a function of temperature,
    all normalized to the maximum of the total.
    """
    N = int(1e4)    # Number of datapoints
    # Temperature axis (log)
    temps = np.logspace(np.log10(T_min), np.log10(T_max), N)
    
    # Energy production arrays
    pp1 = np.zeros(N); pp2 = np.zeros(N); pp3 = np.zeros(N)
    cno = np.zeros(N)
    
    # Create instance of Core
    core = Core(T_min, rho, check=False)
    for i in tqdm(range(N), desc='Energy production'):
        # Update temperature
        core.update_attr(T=temps[i])
        # Calculate the energy production of each branch
        pp1[i], pp2[i], pp3[i], cno[i] = core.calc_chain_energies()
    
    # Total energy production
    total = pp1 + pp2 + pp3 + cno
    
    # Plotting
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    fig.suptitle(r'Relative energy production as function of $T$', fontsize=20)
    ax.set_title(r'$\rho=4.5\times 10^6 \mathrm{kg/m}^3$')
    ax.plot(temps, pp1/total, label=r'$\varepsilon_\mathrm{PPI}$')
    ax.plot(temps, pp2/total, label=r'$\varepsilon_\mathrm{PPII}$')
    ax.plot(temps, pp3/total, label=r'$\varepsilon_\mathrm{PPIII}$')
    ax.plot(temps, cno/total, label=r'$\varepsilon_\mathrm{CNO}$')
    ax.plot(temps, total/np.max(total), label=r'$\varepsilon_\mathrm{tot}/\mathrm{max}(\varepsilon_\mathrm{tot})$')
    ax.axvline(5e7, linestyle='dashed', color='black', label=r'$T=5\times 10^7$ K')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.5)
    ax.set_xlabel(r'Temperature $T$ [K]', fontsize=18)
    ax.set_ylabel(r'$\varepsilon/\varepsilon_\mathrm{tot}$', fontsize=18)
    
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #              box.width, box.height * 0.9])

    fig.legend(loc='outside lower center',
            fancybox=True, shadow=True, ncol=3, fontsize=16)
    plt.show()
    

# Produce figure 1 in the report
plot_chain_energies(1e4, 1e9)

# Instance of Core with the conditions of the Sun
core = Core(5e7, 4.5e6, check=True) # Set check=False if the sanity check isn't needed.

# Print info about the stellar core
print(core)

# Produce figure 2 in the report
core.gamow()




