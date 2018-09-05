from simtk.openmm import Vec3
from simtk.unit import nanometer
from simtk.openmm.app import element
from copy import deepcopy
import numpy as np
import itertools


class MaxMonIterExceedError(Exception):
    """Base class for exceptions in this module."""
    pass

class MaxPolyIterExceedError(Exception):
    """Base class for exceptions in this module."""
    pass

# Definition of functions to create the chain
class ljtools():
    def __init__(self, boxvec=(30,30,30)*nanometer,zmin=10,zmax=20,sigma=.24*nanometer,nmaxmoniter=30,nmaxpolyiter=30):
        self.boxvec = boxvec
        self.boxsize = boxvec.value_in_unit(nanometer)
        self.zmin = zmin
        self.zmax = zmax
        self.sigma = sigma
        self.nmaxmoniter = nmaxmoniter
        self.nmaxpolyiter = nmaxpolyiter

    def getChain(self,topol, chainidx):
        chain = next(itertools.islice(topol.chains(),int(chainidx),int(chainidx+1)))
        return chain

    def getAtomIdxs(self,topol):
        atoms = []
        for i,at in enumerate(topol.atoms()):
            atoms.append(at.index)
        return atoms

    def getDistance(self,newPosition, oldPosition):
        pos1 =  newPosition.value_in_unit(nanometer)
        pos0 =  oldPosition.value_in_unit(nanometer)
        dr = np.zeros(3)
        for i in range(3):
            dr[i] = pos1[i]-pos0[i]
            dr[i] = dr[i] - np.floor(dr[i]/self.boxsize[i]+0.5)*self.boxsize[i]
        dr = np.linalg.norm(dr)
        return dr*nanometer

    def randompos(self):
        pos = Vec3(np.random.rand()*self.boxsize[0], np.random.rand()*self.boxsize[1], np.random.rand()*(self.zmax-self.zmin) + self.zmin)
        return pos*nanometer

    def randomsurfpos(self,zval):
        pos = Vec3(np.random.rand()*self.boxsize[0], np.random.rand()*self.boxsize[1], zval)
        return pos*nanometer

    def checkOverlapSurf(self,newPosition, positions):
        for position in positions:
            dr = self.getDistance(newPosition, position)
            if (dr < self.sigma):
                return True
        return False

    def checkOverlap(self,newPosition, positions):
        zval = newPosition[2].value_in_unit(nanometer)
        if zval < (self.zmin + self.sigma.value_in_unit(nanometer)/2):
            return True
        if zval > (self.zmax - self.sigma.value_in_unit(nanometer)/2):
            return True
        for position in positions:
            dr = self.getDistance(newPosition, position)
            if (dr < self.sigma):
                return True
        return False

    def getNewPosition(self,positions):
        pos0 = self.randompos()
        while (self.checkOverlap(pos0,positions)):
            pos0 = self.randompos()
        return pos0

    def posNext(self,position,sigma):
        costh = np.random.rand()*2-1
        psi = np.random.rand()*2*np.pi
        sinth = np.sqrt(1 - costh**2)
        pos0 = position.value_in_unit(nanometer)
        r0 = sigma.value_in_unit(nanometer)
        pos = (pos0[0] + r0*sinth*np.cos(psi), pos0[1] + r0*sinth*np.sin(psi), pos0[2] + r0*costh)*nanometer
        return pos

    def getNewMonomer(self,newPositions, idx, positions):
        possum = deepcopy(positions)
        for pos in newPositions:
            possum.append(pos)
        pos1 = self.posNext(newPositions[idx],self.sigma)
        iiter = 0
        while(self.checkOverlap(pos1,possum)):
            iiter += 1
            if iiter > self.nmaxmoniter:
                raise MaxMonIterExceedError
            pos1 = self.posNext(newPositions[idx],self.sigma)
        return pos1

    def getNewLinearPolymer(self,positions,nmon):
        iiter = 0
        posbb = []*nanometer
        while True:
            try:
                newPositions = []*nanometer
                pos0 = self.getNewPosition(positions)
                newPositions.append(pos0)
                for i in range(nmon-1):
                    newPositions.append(self.getNewMonomer(newPositions,-1, positions))
                break
            except MaxMonIterExceedError:
                print('Polymer insertion failed. Retrying')
                iiter += 1
                if iiter > self.nmaxpolyiter:
                    raise MaxPolyIterExceedError
        return newPositions


    def getNewBranchedPolymer(self,positions,nmonbranch,nskipbranch):
        iiter = 0
        posbb = []*nanometer
        while True:
            try:
                newPositions = []*nanometer
                pos0 = self.getNewPosition(positions)
                newPositions.append(pos0)
                for i in range(nskipbranch):
                    for j in range(nmonbranch):
                        newPositions.append(self.getNewMonomer(newPositions,-1, positions))
                    newPositions.append(self.getNewMonomer(newPositions,-5, positions))
                newPositions.append(self.getNewMonomer(newPositions,-1, positions))
                break
            except MaxMonIterExceedError:
                print('Polymer insertion failed. Retrying')
                iiter += 1
                if iiter > self.nmaxpolyiter:
                    raise MaxPolyIterExceedError
        return newPositions

    def addLinearPolymer(self,topol,positions,nmon,resname='LJ',atomname='LJ',element=element.carbon):
        idxat=len(positions)
        try:
            newPositions = self.getNewLinearPolymer(positions, nmon)
        except MaxPolyIterExceedError:
            print('Maximum iteration is reached')
            raise MaxPolyIterExceedError
        newChain = topol.addChain()
        newAtoms = []
        for j in range(nmon):
            newResidue = topol.addResidue(resname, newChain, j)
            newAtom = topol.addAtom(atomname, element, newResidue, idxat+j)
            newAtoms.append(newAtom)
            positions.append(newPositions[j])
            if j > 0:
                topol.addBond(newAtoms[j-1], newAtoms[j])
        return topol,positions

    def addLinearPolymerTopol(self,topol,nmon,resname='LJ',atomname='LJ',element=element.carbon):
        idxat=topol.getNumAtoms()
        newChain = topol.addChain()
        newAtoms = []
        for j in range(nmon):
            newResidue = topol.addResidue(resname, newChain, j)
            newAtom = topol.addAtom(atomname, element, newResidue, idxat+j)
            newAtoms.append(newAtom)
            if j > 0:
                topol.addBond(newAtoms[j-1], newAtoms[j])
        return topol
