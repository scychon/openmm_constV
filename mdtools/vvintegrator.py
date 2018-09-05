import simtk.unit as unit
import simtk.openmm as mm
import random
import numpy as np

class VelocityVerletIntegrator(mm.CustomIntegrator):

    """Verlocity Verlet integrator.
    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.
    References
    ----------
    W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, J. Chem. Phys. 76, 637 (1982)
    Examples
    --------
    Create a velocity Verlet integrator.
    >>> timestep = 1.0 * unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)
    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator.
        Parameters
        ----------
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1*unit.femtoseconds
           The integration timestep.
        """

        super(VelocityVerletIntegrator, self).__init__(timestep)

        self.addPerDofVariable("x1", 0)

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

class neMDMC(object):

    def __init__(self, simmd, simmc, biasforce, temperature = 298.0 * unit.kelvin, tmc=20.0*unit.picosecond, tboost=5.0*unit.picosecond, nboost=10,numChain=0,numPELen=0,genRandomParticleIdxs=True,biasFirstIdx=-1,biasParticleIdxs=[]):
        self.simmd = simmd
        self.simmc = simmc
        self.stepsize = simmc.integrator.getStepSize()
        self.biasforce = biasforce
        self.tmc = tmc
        self.tboost = tboost
        self.nboost = nboost
        self.booststepsize = int(tboost/(nboost*self.stepsize))
        self.topstepsize = int((tmc-2*tboost)/self.stepsize)
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA
        self.naccept = 0
        self.ntrials = 0
        self.numchain = numChain
        self.numpelen = numPELen
        self.useAutoGen = genRandomParticleIdxs
        self.biasidx = biasFirstIdx
        self.biasptclidxs = biasParticleIdxs

    def getBiasFirstIdx(self):
        return self.biasidx

    def setBiasFirstIdx(self,idx):
        self.biasidx = idx

    def setBiasParticleIdxs(self,idxs):
        self.biasptclidxs = idxs

    def setTemperature(self,temperature):
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA

    def getTemperature(self):
        return self.RT/(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA)

    def setTimeMC(self,tmc):
        self.tmc = tmc
        self.topstepsize = int((self.tmc-2*self.tboost)/self.stepsize)

    def getTimeMC(self):
        return self.tmc

    def setTimeMCBoost(self,tboost):
        self.tboost = tboost
        self.booststepsize = int(tboost/(self.nboost*self.stepsize))
        self.topstepsize = int((self.tmc-2*self.tboost)/self.stepsize)

    def getTimeMCBoost(self):
        return self.tboost

    def setNumMCBoost(self,nboost):
        self.nboost = nboost
        self.booststepsize = int(self.tboost/(self.nboost*self.stepsize))

    def getNumMCBoost(self):
        return self.nboost

    def propagate(self):
        self.ntrials += 1
        state = self.simmd.context.getState(getEnergy=True,getVelocities=True,getPositions=True)
        oldE = state.getPotentialEnergy()+state.getKineticEnergy()
        self.simmc.context.setPositions(state.getPositions())
        fflip = random.randint(0,1)*2.-1
        self.simmc.context.setVelocities(fflip*state.getVelocities())
        if(self.useAutoGen):
            if(self.numchain == 0 or self.numpelen == 0):
                print("There are no polymers !")
                return False
            else:
                self.biasidx = self.numpelen * random.randint(0,self.numchain-1)
                self.biasptclidxs = range(self.biasidx,self.biasidx+self.numpelen)
                print('biasidx : '+str(self.biasidx))
        if(len(self.biasptclidxs)==0):
            print("There are no particles to be biased !")
            return False

        fdir = random.randint(0,1)*2.-1

        for i in range(self.nboost):
            if(i==0):
                print(str(fdir))
#                for j in range(len(self.biasptclidxs)):
                    #self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[])
            if(i>0):
#                self.simmc.context.setParameter('lambda',fdir*i/self.nboost)
                for ptcl in self.biasptclidxs:
                    self.biasforce.setParticleParameters(ptcl,ptcl,[fdir*i/self.nboost])
                self.biasforce.updateParametersInContext(self.simmc.context)
#            for j in range(len(self.biasptclidxs)):
#                self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir*i/self.nboost])
            self.simmc.step(self.booststepsize)

#        for j in range(len(self.biasptclidxs)):
#            self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir])
        for ptcl in self.biasptclidxs:
            self.biasforce.setParticleParameters(ptcl,ptcl,[fdir])
        self.biasforce.updateParametersInContext(self.simmc.context)
#        self.simmc.context.setParameter('lambda',fdir)
        self.simmc.step(self.topstepsize)

        for i in range(self.nboost-1,-1,-1):
            for ptcl in self.biasptclidxs:
                self.biasforce.setParticleParameters(ptcl,ptcl,[fdir*i/self.nboost])
#            for j in range(len(self.biasptclidxs)):
#                self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir*i/self.nboost])
            self.biasforce.updateParametersInContext(self.simmc.context)
#            self.simmc.context.setParameter('lambda',fdir*i/self.nboost)
            self.simmc.step(self.booststepsize)

        state = self.simmc.context.getState(getEnergy=True,getVelocities=True,getPositions=True)
        newE = state.getPotentialEnergy()+state.getKineticEnergy()
        if self.metropolis(newE,oldE):
            self.simmd.context.setPositions(state.getPositions())
            self.simmd.context.setVelocities(fflip*state.getVelocities())
            self.naccept += 1
            return True
        else:
            return False

    def getAcceptRatio(self):
        return self.naccept/self.ntrials

    def metropolis(self,pecomp,peref):
        if pecomp < peref:
            return True
        elif (random.uniform(0.0,1.0) < np.exp(-(pecomp - peref)/self.RT)):
            return True
        else:
            return False


class Barostat(object):

    def __init__(self, simeq, pressure = 1.0*unit.bar, temperature = 298.0 * unit.kelvin, barofreq = 25):
        self.simeq = simeq
        self.temperature = temperature
        self.barofreq = barofreq
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA
        self.naccept = 0
        self.ntrials = 0
        self.celldim = self.simeq.topology.getUnitCellDimensions()
        self.lenscale = self.celldim[2]*0.01
        self.pressure = pressure*self.celldim[0]*self.celldim[1]*unit.AVOGADRO_CONSTANT_NA
        self.numres = simeq.topology.getNumResidues()
        self.numRealRes = simeq.topology.getNumResidues()
        self.firstRealRes = 0
        self.resmass = np.zeros(self.numres)*unit.dalton
        self.resNumAtoms = np.zeros(self.numres,int)
        self.resFirstAtomIdxs = np.zeros(self.numres,int)
        self.firstidx = 0
        self.lastidx = simeq.topology.getNumAtoms()
        for at in simeq.topology.atoms():
            self.resmass[at.residue.index] += simeq.system.getParticleMass(at.index)
            self.resNumAtoms[at.residue.index] += 1
        idxAtom = 0
        for i in range(self.numres):
            self.resFirstAtomIdxs[i] = idxAtom
            # Exclude fixed residues from barostat acceptance criteria
            if self.resmass[i] == 0*unit.dalton:
                self.numRealRes -= 1
                if self.firstidx == idxAtom:
                    self.firstidx += self.resNumAtoms[i]
                    self.firstRealRes += 1
                elif self.lastidx == simeq.topology.getNumAtoms():
                    self.lastidx = idxAtom
            idxAtom += self.resNumAtoms[i]

    def getAcceptRatio(self):
        return self.naccept/self.ntrials

    def metropolis(self,pecomp):
        if pecomp < 0*self.RT:
            return True
        elif (random.uniform(0.0,1.0) < np.exp(-(pecomp)/self.RT)):
            return True
        else:
            return False

    def step(self,nstep):
        niter = int(nstep/self.barofreq)
        for i in range(niter):
            self.simeq.step(self.barofreq)
            self.ntrials += 1
            statebak = self.simeq.context.getState(getEnergy=True,getPositions=True)
            oldE = statebak.getPotentialEnergy()
            oldpos = statebak.getPositions()
            newpos0 = np.asarray(oldpos.value_in_unit(unit.nanometer))
            newpos = np.asarray(oldpos.value_in_unit(unit.nanometer))
            boxvec = statebak.getPeriodicBoxVectors()
            oldboxlen = boxvec[2][2]
            deltalen = self.lenscale*(random.uniform(0,1)*2.-1)
            newboxlen = oldboxlen+deltalen
            for i in range(self.numres):
                fidx = self.resFirstAtomIdxs[i]
                lidx = self.resFirstAtomIdxs[i]+self.resNumAtoms[i]
                if self.resmass[i] == 0*unit.dalton:
                    newpos[fidx:lidx,2] += 0.5*deltalen/unit.nanometer
                else:
                    newpos[fidx,2] *=newboxlen/oldboxlen
                    posdel = newpos[fidx,2] - newpos0[fidx,2]
                    newpos[fidx+1:lidx,2] += posdel
            self.simeq.context.setPositions(newpos)
            self.simeq.context.setPeriodicBoxVectors(boxvec[0],boxvec[1],mm.Vec3(0,0,newboxlen/unit.nanometer)*unit.nanometer)
            statenew = self.simeq.context.getState(getEnergy=True,getPositions=True)
            newE = statenew.getPotentialEnergy()
            w = newE-oldE + self.pressure*deltalen - self.numRealRes * self.RT*np.log(newboxlen/oldboxlen)
            if self.metropolis(w):
                self.naccept += 1
                print('newE: '+str(newE)+'oldE: '+str(oldE))
            else:
                self.simeq.context.setPositions(oldpos)
                self.simeq.context.setPeriodicBoxVectors(boxvec[0],boxvec[1],mm.Vec3(0,0,oldboxlen/unit.nanometer)*unit.nanometer)
            if self.ntrials >= 10:
                if (self.naccept < 0.25*self.ntrials) :
                    self.lenscale /= 1.1
                    self.ntrials = 0
                    self.naccept = 0
                elif self.naccept > 0.75*self.ntrials :
                    self.lenscale = min(self.lenscale*1.1, newboxlen*0.3)
                    self.ntrials = 0
                    self.naccept = 0

