from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np
import os
import itertools
from constvplugin import ConstVLangevinIntegrator

# Thermostat parameters
temperature = 300*kelvin
pressure = 1.0*atmospheres
inittimestep = 1.*femtosecond
timestep = 1.*femtosecond
freq = 1/picosecond
cutoff = 1.2*nanometer
prodtime = int(150*nanosecond/(1000*timestep))+1

strdir = ''
ffFile = 'spce.xml'
forcefield = ForceField(ffFile)

#####################
# Specific setup for box dimension and potential difference of the electrodes
#####################
zmin = 0                # location of left wall atoms (always put this to be zero)
zmax = 4.2183           # location of right wall atoms
constV = 1*volt         # potential difference between two parallel electrodes
constEfield = (constV*elementary_charge*AVOGADRO_CONSTANT_NA/(zmax*nanometer)).in_units_of(kilojoule_per_mole/nanometer)        # electric field calculated for the gap between the two electrode


#####################
# Integrators for non-polarizable system
#####################
integ_md = ConstVLangevinIntegrator(temperature, freq, inittimestep)


##########################
# Setup the simulation system topology and initial positions
# The z-axis dimension of the simulation box must be twice the gap between two parallel walls
# Note that this module only allows sampling in Canonical ensemble,
# thus you may need to perform NPT equilibration first without the surface polarization effect using regular integrators.
##########################

# pdb file excluding drude particles to create modeller - equivalent to nonpolarizable simulations
# (typically an output from CHARMM-GUI or other preparation package)
pdb = PDBFile('nacl_1m_eq.pdb')
positions = pdb.positions
newTopology = pdb.topology
boxvec = pdb.topology.getUnitCellDimensions()._value
if (boxvec[2]-zmax*2 != 0):
    newTopology.setUnitCellDimensions(Vec3(boxvec[0],boxvec[1],zmax*2)*nanometer)

## create system with real particles only
system = forcefield.createSystem(newTopology, nonbondedCutoff=cutoff, constraints=AllBonds)
nbforce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
nbforce.setNonbondedMethod(NonbondedForce.PME)

## add Image charge particles to the topology and positions
newChain = newTopology.addChain()
newResidue = newTopology.addResidue('IM', newChain)
imsig = 1*nanometer             # ignore vdW interaction for image particles
imeps = 0*kilojoule/mole        # ignore vdW interaction for image particles
nRealAtoms = system.getNumParticles()

## create external force object if the external electric field by the walls is not zero
## (either due to potential difference or charged wall)
if constEfield._value >0:
    constVForce = CustomExternalForce('efield*q*z')
    constVForce.addGlobalParameter('efield',constEfield._value)
    constVForce.addPerParticleParameter('q')

## add image particles for all real particles regardless of the charge
for i in range(nRealAtoms):
    (q, sig, eps) = nbforce.getParticleParameters(i)
    newAtom = newTopology.addAtom('IM', None, newResidue)
    pos = positions[i].value_in_unit(nanometer)
    if q!=-q:
        # position the image charge particles at the mirror location with respect to the left wall at z=0
        positions.append((pos[0],pos[1],-pos[2])*nanometer)
    else:
        # shift the position of image particle for non-charged particles to avoid zero distance divergence between the real and image particles
        # (This is required for the wall atoms)
        positions.append((pos[0],pos[1],-pos[2]+0.001)*nanometer)
    idxat = system.addParticle(0*dalton)            # image particle does not have mass
    idxat2 = nbforce.addParticle(-q,imsig,imeps)    # image particle has no vdW interaction
    # add charged particles into constVForce to apply the E_field by the walls
    # (either potential difference or charged wall)
    if constEfield._value >0:
        idxres = constVForce.addParticle(i, [q])

## add the E-field external force to the system
if constEfield._value >0:
    system.addForce(constVForce)

##########################
# Add exclusion for image particles
##########################
# nbforces between image particles following the exception in real particles
# keep the electrostatic part (q) but ignore the vdW interaction (imsig,imeps)
for i in range(nbforce.getNumExceptions()):
    (idx0,idx1,q,sig,eps) = nbforce.getExceptionParameters(i)
    idxforce = nbforce.addException(idx0+nRealAtoms,idx1+nRealAtoms,q,imsig,imeps)

## add exclusion for electrode atoms
## In this example only the wall atoms have zero mass.
## If other real particles have zero mass, you need to specify other routine to set the exclusion between the wall atoms
## This step is may be skipped since the interactions between the wall atoms would be a constant throughout the simulation and does not affect the force on real atoms. But excluding the wall interaction ensures legitimate value for the system potential energy

# first find all the wall atoms
grpBottom = []
grpTop = []
for i in range(nRealAtoms):
    if system.getParticleMass(i) == 0*dalton:
        if positions[i][2]==0*nanometer:
            grpBottom.append(i)
        else:
            grpTop.append(i)

# add exclusion between atoms in right electrode (at z=zmax)
for idx0,idx1 in itertools.combinations(grpTop,2):
    dpos = positions[idx0]-positions[idx1]
    dr = dpos - (np.round(np.asarray(dpos._value)/boxvec)*boxvec)*nanometer
    if np.linalg.norm(dr._value)*nanometer < .5*nanometer:
        idxforce = nbforce.addException(idx0,idx1,0*elementary_charge**2,imsig,imeps)

# add exclusion between atoms in left electrode (at z=0)
for idx0,idx1 in itertools.combinations(grpBottom,2):
    dpos = positions[idx0]-positions[idx1]
    dr = dpos - (np.round(np.asarray(dpos._value)/boxvec)*boxvec)*nanometer
    if np.linalg.norm(dr._value)*nanometer < .5*nanometer:
        idxforce = nbforce.addException(idx0,idx1,0*elementary_charge**2,imsig,imeps)


#########################

# set force group to report each energy components
for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)


#########################
# Create simulation object
#########################

os.environ["CUDA_VISIBLE_DEVICES"]="5"
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

simmd = Simulation(newTopology, system, integ_md, platform, properties)
simmd.context.setPositions(positions)
state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
positions = state.getPositions()
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
PDBFile.writeFile(newTopology,positions,open('begin.pdb','w+'))

#############################
# Actual simulation routine
#############################
print('Equilibrating...')

simmd.reporters = []
dcdfile = 'eq_nvt.dcd'
logfile = 'eq_nvt.log'
chkfile = 'eq_nvt.chk'

simmd.reporters.append(DCDReporter(dcdfile, 1000))
simmd.reporters.append(StateDataReporter(logfile, 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, density=True,speed=True))
simmd.reporters.append(CheckpointReporter(chkfile, 50000))
simmd.reporters[1].report(simmd,state)

print('Simulating...')

## log file to report the energy components
enerlog = open('eq_nvt_ener.log', 'w')
#write the header for the energy log file
enerlog.write('# Energy log file\n')
enerlog.write('# x1 : time (ps)\n')
for j in range(system.getNumForces()):
    f = system.getForce(j)
    enerlog.write('# x'+str(j+2) + ' : ' +str(type(f)) + ' (kJ/mol)\n')

for i in range(1,10001):
    simmd.step(1000)
    enerlog.write(str(i))
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        enerlog.write('  ' + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)))
    enerlog.write('\n')
    enerlog.flush()

enerlog.close()
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True, enforcePeriodicBox=True)
position = state.getPositions()
PDBFile.writeFile(simmd.topology, position, open(strdir+'eq_nvt.pdb', 'w'))

print('Done!')

exit()
