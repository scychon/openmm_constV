# -*- coding: UTF-8 -*-
# ============================================================================================
# MODULE DOCSTRING
# ============================================================================================

"""
Thermostated integrators for molecular simulation.

DESCRIPTION

This module provides various thermostated custom integrators for OpenMM.

EXAMPLES

COPYRIGHT

@author Chang Yun Son <cson@caltech.edu>

All code in this repository is released under the MIT License.

This program is free software: you can redistribute it and/or modify it under
the terms of the MIT License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the MIT License for more details.

You should have received a copy of the MIT License along with this program.

"""

# ============================================================================================
# GLOBAL IMPORTS
# ============================================================================================

import simtk.unit as unit
import simtk.openmm as mm

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

class CanonicalVelocityRescalingIntegrator(mm.CustomIntegrator):

    """Velocity Verlet integrator with stochastic velocity rescaling thermostat satisfying canonical ensemble

    References
    ----------
    G. Bussi, D. Donadio and M. Parrinello "Canonical sampling through velocity rescaling", Journal of Chemical Physics 126, 014101 (2007)
    http://dx.doi.org/10.1063/1.2408420

    Examples
    --------

    Create a velocity Verlet integrator with canonical velocity rescaling thermostat.
    Define the number of degree of freedom from system particles*3-numconstraints

    >>> timestep = 1.0 * unit.femtoseconds
    >>> collision_rate = 10.0 / unit.picoseconds
    >>> temperature = 298.0 * unit.kelvin
    >>> ndf = system.getNumParticles()*3 - system.getNumConstraints()
    >>> integrator = CanonicalVelocityRescalingIntegrator(temperature, collision_rate, timestep, ndf)

    CanonicalVelocityRescalingIntegrator can also be used to create a velocity Verlet integrator with Berendsen thermostat.
    Just set the useBerendsen attribute to be True.
    >>> integrator = CanonicalVelocityRescalingIntegrator(temperature, collision_rate, timestep, ndf, useBerendsen=True)

    Notes
    ------
    The velocity Verlet integrator is taken verbatim from Peter Eastman's example in the CustomIntegrator header file documentation.

    """

    def __init__(self, temperature=298 * unit.kelvin, collision_rate=10.0 / unit.picoseconds, timestep=1.0 * unit.femtoseconds, ndf=0, useBerendsen=False):
        """Construct a velocity Verlet integrator with canonical velocity rescaling thermostat.

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default=298*unit.kelvin
           The temperature of the fictitious bath.
        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default=10/unit.picoseconds
           The collision rate with fictitious bath particles.
        timestep : np.unit.Quantity compatible with femtoseconds, default=1*unit.femtoseconds
           The integration timestep.
        ndf : integer, default=0 (ignore constraints)
           The number of degrees of freedom to be used to calculate the temperature.
        useBerendsen : boolean, default=False
           Whether use Berendsen thermostat instead of the canonical velocity rescaling thermostat.

        """
        super(CanonicalVelocityRescalingIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable('kT', kB * temperature)  # thermal energy
        self.addGlobalVariable("p_collision", timestep * collision_rate)  # per-particle collision probability per timestep
        self.addGlobalVariable("scale_v", 1.0)  # scaling factor for velocity
        self.addPerDofVariable("x1", 0)  # for constraints
        self.addGlobalVariable("ndf", ndf)      # number of degrees of freedom
        self.addGlobalVariable("KE2", 0)      # number of degrees of freedom

        #
        # Velocity Verlet step
        #
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

        #
        # Scale velocities to keep temperature from Maxwell-Boltzmann distribution.
        #
        self.addComputeSum("KE2", "m*v*v")
        if(ndf==0):
            self.addComputeSum("ndf", 1)
        if(useBerendsen):
            self.addComputeGlobal("scale_v","sqrt(1+(kr-1)*p_collision);kr=ndf*kT/KE2")
        else:
            self.addComputeGlobal("scale_v","sqrt(1+(kr-1)*p_collision +2*sqrt(kr*p_collision/ndf)*gaussian);kr=ndf*kT/KE2")
        self.addComputePerDof("v", "v*scale_v")

