import numpy as np
import matplotlib.pyplot as plt

def cross_section(R, L, F_C, sanity=False, savefig=False):
    """
    plot cross section of star
        :param R: radius, array
        :param L: luminosity, array
        :param F_C: convective flux, array
        :param sanity: boolean, True/False
        :param savefig: boolean, True/False
    """
    L_sun      = 3.846e26 
    R_sun      = 6.96e8         # m

    r_range    = 1.2*np.max(R/R_sun)  # normalized 
    core_limit = 0.995 * np.max(L)

    ## Get the different zones:
        # 3: convection outside
        # 2: radiation outside
        # 1: radiation inside
        # 0: convection inside
    star_zone = np.int32(np.where(L>core_limit,0.5,-0.5) * np.where(F_C>0,3,1) + 2)
    colors=['blue','cyan','yellow','red']
    
    ## Initialize the figure
    plt.figure(figsize=(6,6))
    fig = plt.gcf()
    ax  = plt.gca()
    ax.set_xlim(-r_range, r_range)
    ax.set_ylim(-r_range, r_range)
    ax.set_aspect('equal')

    star_zone_prev = -1
    for k in range(0,len(R)-1):
        if star_zone[k]!=star_zone_prev: # only plot a new *filled* circle if a new star_zone
            star_zone_prev = star_zone[k]
            circle = plt.Circle((0,0),R[k]/R_sun,fc=colors[star_zone[k]],fill=True,ec=None)
            ax.add_artist(circle)
    
    circle_white=plt.Circle((0,0),R[-1]/R_sun,fc='white',fill=True,lw=0)
    ax.add_artist(circle_white)

    # create legends
    circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[3], fill=True)
    circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[2], fill=True)
    circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[1], fill=True)
    circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color=colors[0], fill=True)

    ax.legend([circle_red,circle_yellow,circle_cyan,circle_blue],\
            ['Convection outside core','Radiation outside core','Radiation inside core','Convection inside core']\
            , fontsize=15)
    plt.xlabel(r'$r/R_\odot$', fontsize=16)
    plt.ylabel(r'$r/R_\odot$', fontsize=16)
    plt.title('Cross section of optimal star', fontsize=20)
    if savefig:
        if sanity:
            fig.savefig('sanity_cross_section_2D.png', dpi=300)
        else:
            fig.savefig('final_cross_section_2D.png', dpi=300)