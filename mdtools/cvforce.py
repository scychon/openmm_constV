# -*- coding: UTF-8 -*-
# ============================================================================================
# MODULE DOCSTRING
# ============================================================================================

"""
Utilities to create various custom forces used with OpenMM.

DESCRIPTION

This module provides various custom forces for OpenMM.

EXAMPLES

COPYRIGHT

@author Chang Yun Son <cson@caltech.edu>

All code in this repository is released under the Caltech License.

This program is free software: you can redistribute it and/or modify it under
the terms of the Caltech License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the Caltech License for more details.

You should have received a copy of the Caltech License along with this program.

"""

# ============================================================================================
# GLOBAL IMPORTS
# ============================================================================================

import simtk.unit as unit
import simtk.openmm as mm
import numpy as np

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
ONE_4PI_EPS0 = 138.935456

# check whether customCVForce works correctly
pos = state.getPositions()
mu0 = 0*(nanometer*elementary_charge)
for i in range(system.getNumParticles()):
    (q, sig, eps) = nbondedForce.getParticleParameters(i)
    mu0 += q*pos[i][2]

print(ONE_4PI_EPS0* 2*np.pi*mu0._value**2 /volume._value)

class SlabCorrection(mm.CustomCVForce):

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

    def __init__(self, system, volume):
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
        nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
        cvdipole = mm.CustomExternalForce('q*z')
        cvdipole.addPerParticleParameter('q')
        for i in range(system.getNumParticles()):
            (q, sig, eps) = nbondedForce.getParticleParameters(i)
            idxres = cvdipole.addParticle(i, [q])

        cvforce = CustomCVForce("twopioverV*(muz)^2")
        cvforce.addCollectiveVariable("muz",cvdipole)
        cvforce.addGlobalParameter("twopioverV",ONE_4PI_EPS0*2*np.pi/volume)
        system.addForce(cvforce)

