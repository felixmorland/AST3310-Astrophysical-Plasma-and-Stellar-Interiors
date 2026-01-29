# Astrophysical Plasma and Stellar Interiors
The three parts of the project in AST3310 at the University of Oslo, simulating the physics of a star. All the scientific reports are available for reading.

### Part 1
Calculates the energy production in the core of a star, given its temperature. The script calculates the total power from the fusion reactions per unit mass $\epsilon$ assuming a homogenous and isotropic core. The considered reactions are the PP chains and the CNO cycle.

### Part 2
In the second part, I analyse the stratification of the stellar interior, dividing it into layers where the energy from the fusion reactions in the core are transported outwards by means of (i) radiative processes or (ii) convection. The code allows visualisation of the resulting stratification of a star given its mass $M$, radius $R$, surface temperature $T$, luminosity $L$, and its average matter density $\rho$.


### Part 3
Visualises the convection in the outer layer of the star and generates 2D animations of the convective motion using FVis. Solves the three hydrodynamic equations

$$\frac{\partial \rho}{\partial t} + \nabla\cdot (\rho\vec{u}) = 0$$

$$\frac{\partial \rho\vec{u}}{\partial t} +\nabla\cdot (\rho\vec{u}\otimes\vec{u}) = -\nabla P +\rho\vec{g}$$

$$\frac{\partial e}{\partial t}+\nabla\cdot(e\vec{u}) = -P\nabla \cdot \vec{u}$$

Some example animations are available inside ```part3/animations```.
