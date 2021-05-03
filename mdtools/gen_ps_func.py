from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
import simtk.openmm as mm
from simtk.unit import *
from copy import deepcopy
import numpy as np
import os
import random

sigma = .1529 * nanometer
boxvec = (10,10,10)*nanometer
nmaxmoniter = 20
nmaxpolyiter = 150

class MaxMonIterExceedError(Exception):
    """Base class for exceptions in this module."""
    pass

class MaxPolyIterExceedError(Exception):
    """Base class for exceptions in this module."""
    pass

class UnknownPSXError(Exception):
    """Base class for exceptions in this module."""
    pass

def getDistance(newPosition, oldPosition):
    pos1 =  newPosition.value_in_unit(nanometer)
    pos0 =  oldPosition.value_in_unit(nanometer)
    boxsize = boxvec.value_in_unit(nanometer)
    dr = np.zeros(3)
    for i in range(3):
        dr[i] = pos1[i]-pos0[i]
        dr[i] = dr[i] - np.floor(dr[i]/boxsize[i]+0.5)*boxsize[i]
    dr = np.linalg.norm(dr)
    return dr*nanometer

def randompos():
    boxsize = boxvec.value_in_unit(nanometer)
    pos = Vec3(np.random.rand()*boxsize[0], np.random.rand()*boxsize[1], np.random.rand()*boxsize[2])
    return pos*nanometer

# subroutine to rotate xpos by angle theta in yz plane
def rotX(pos, theta):
    pos0 = pos.value_in_unit(nanometer)
    costh = cos(theta*np.pi/180)
    sinth = sin(theta*np.pi/180)
    posnew = Vec3(pos0[0], pos0[1]*costh-pos0[2]*sinth, pos0[1]*sinth+pos0[2]*costh)*nanometer
    return posnew

def rotXall(posall, theta):
    posnew = deepcopy(posall)
    for i in range(len(posall)):
        posnew[i] = rotX(posall[i],theta)
    return posnew

# subroutine to rotate xpos by angle theta in xz plane
def rotY(pos, theta):
    pos0 = pos.value_in_unit(nanometer)
    costh = cos(theta*np.pi/180)
    sinth = sin(theta*np.pi/180)
    posnew = Vec3(pos0[0]*costh+pos0[2]*sinth, pos0[1], -pos0[0]*sinth+pos0[2]*costh)*nanometer
    return posnew

def rotYall(posall, theta):
    posnew = deepcopy(posall)
    for i in range(len(posall)):
        posnew[i] = rotY(posall[i],theta)
    return posnew

# subroutine to rotate xpos by angle theta in xy plane
def rotZ(pos, theta):
    pos0 = pos.value_in_unit(nanometer)
    costh = cos(theta*np.pi/180)
    sinth = sin(theta*np.pi/180)
    posnew = Vec3(pos0[0]*costh-pos0[1]*sinth, pos0[0]*sinth+pos0[1]*costh,pos0[2])*nanometer
    return posnew

def rotZall(posall, theta):
    posnew = deepcopy(posall)
    for i in range(len(posall)):
        posnew[i] = rotZ(posall[i],theta)
    return posnew

# default structural information of the benzene group
# CA carbon is at the origin, CD carbon is on z axis, and the ring is on xz plane
# param rCX defines the z position of CA atom, which has default value of 0.151 for CA-CT bond length
def posBenzene(rCX=0.151*nanometer):
    rCC = 0.140*nanometer;
    if (not type(rCX)==type(Quantity())):
        rCX = rCX*nanometer
    posCA = rCX*Vec3(0,0,1)
    posnew = []*nanometer
    posnew.append(posCA+Vec3(0,0,0)*nanometer)        #CA
    posnew.append(posCA+rCC*Vec3(cos(np.pi/6),0,sin(np.pi/6)))        #CB1
    posnew.append(posCA+rCC*Vec3(-cos(np.pi/6),0,sin(np.pi/6)))       #CB2
    posnew.append(posCA+rCC*Vec3(cos(np.pi/6),0,1+sin(np.pi/6)))      #CC1
    posnew.append(posCA+rCC*Vec3(-cos(np.pi/6),0,1+sin(np.pi/6)))     #CC2
    posnew.append(posCA+rCC*Vec3(0,0,2))        #CD
    return posnew

# default structural information of sulfobenzene group for PSS & PSTFSI (-S(=O)2X) where X is O3(PSS) or N(PSTFSI)
# O3 (=N in PSTFSI) is located in yz plane and the other two oxygen are located relatively
# param rCX defines the z position of CA atom, which has default value of 0.151 for CA-CT bond length
# param rSO3 defines the length of S-O3 or S-N bond, which has default value of 0.144 for SY-OY bond length
def posSulfoBenzene(rSO3=0.144*nanometer, rCX=0.151*nanometer):
    rCS = 0.177*nanometer
    rSO = 0.144*nanometer
    if (not type(rSO3)==type(Quantity())):
        rSO3 = rSO3*nanometer
    thCSO = np.pi*107/180        #CSO angle = 107
    costh = cos(thCSO-np.pi/2)   #since CS bond lies in z axis, angle relative to yz plane becomes 17
    sinth = sin(thCSO-np.pi/2)   #since CS bond lies in z axis, angle relative to yz plane becomes 17
    posnew = posBenzene(rCX)
    posOref = rSO*Vec3(0,costh,sinth)         #reference position of O3 which lies on yz plane
    posS = posnew[-1]+rCS*Vec3(0,0,1)         #SY
    posnew.append(posS)                       #SY
    posnew.append(posS+rotZ(posOref, 120))    #OY1
    posnew.append(posS+rotZ(posOref, -120))   #OY2
    posnew.append(posS+posOref*rSO3/rSO)      #OY3
    return posnew

# default structural information of any SP3 group such as CH3 (default), CF3, SO3 etc.
# C is located on z axis, H3 is located in yz plane
def posSP3(rCC=0.152*nanometer,rCH=0.108*nanometer,theta=109.5,rCH3=0.108*nanometer):
    thCCH = np.pi*theta/180     #XCF angle, default for SCF = 112
    if (not type(rCC)==type(Quantity())):
        rCC = rCC*nanometer
    if (not type(rCH)==type(Quantity())):
        rCH = rCH*nanometer
    if (not type(rCH3)==type(Quantity())):
        rCH3 = rCH3*nanometer
    posC = rCC*Vec3(0,0,1)
    costh = cos(thCCH-np.pi/2)   #since CX bond lies in z axis, subtract 90 for angle relative to yz plane
    sinth = sin(thCCH-np.pi/2)   #since CX bond lies in z axis, subtract 90 for angle relative to yz plane
    posnew = []*nanometer
    posHref = rCH*Vec3(0,costh,sinth)         #reference position of O3 which lies on yz plane
    posnew.append(posC)                       #C
    posnew.append(posC+rotZ(posHref, 120))    #H1
    posnew.append(posC+rotZ(posHref, -120))   #H2
    posnew.append(posC+posHref*rCH3/rCH)      #H3
    return posnew

def posadd(positions,pos0):
    posnew = deepcopy(positions)
    for i in range(len(posnew)):
        posnew[i] = positions[i]+pos0
    return posnew

# default structural information of triflate group for PSTFSI (-NS(=O)2CF3)
# O3 (=N in PSTFSI) is located in yz plane and the other two oxygen are located relatively
def posTf2nBenzene(rSN0=0.144*nanometer, rCX=0.151*nanometer):
    rCF = 0.131*nanometer
    rSO = 0.143*nanometer
    rCS = 0.185*nanometer
    rSN = 0.155*nanometer
    thCSO = np.pi*107/180        #CSO angle = 107
    thCSN = np.pi*103/180        #CSN angle = 103
    thSNS = np.pi*125/180        #SNS angle = 107
    costh = cos(thCSO-np.pi/2)   #since CS bond lies in z axis, angle relative to yz plane becomes 17
    sinth = sin(thCSO-np.pi/2)   #since CS bond lies in z axis, angle relative to yz plane becomes 17
    posnew = posSulfoBenzene(rSN0,rCX)
    posN = posnew[-1]
    posSO3 = rotZall(posSP3(rSN, rSO,103, rCS),60)     # align SO2C group, angle CSN=103
    for i in range(len(posSO3)):
        posnew.append(posN+posSO3[i])
    posC = posnew[-1]
    posCF3 = rotZall(rotXall(posSP3(rCS, rCF,112, rCS),-77),60)                   # align CF3 group, angle NCF=112
    for i in range(1,len(posCF3)):
        posnew.append(posC+posCF3[i]-posCF3[0])
    return posnew

# get new PS derivative monomer positions.
# the BB atoms are located in yz plane, and the styrene side chain is in upward position
# tacticity (R,S configuration) is determined by the rotation angle theta
# posref : reference position of previous CBB atom
# theta : rotation angle w.r.t. y axis
# ptype : identity of the monomer ('PS', 'PSS', 'PSTFSI')
def posPSXmon(posref,ptype='PS',theta=54.75):
    rCC = 0.1529*nanometer
    thCCC = np.pi*112.7/180     #CCC angle = 112.7
    costh = cos(thCCC/2)
    sinth = sin(thCCC/2)
    posCBB1 = rCC*Vec3(0,sinth,costh)
    posCBB2 = posCBB1 + rCC*Vec3(0,sinth,-costh)
    if ptype=='PS':
        posside = rotYall(posBenzene(),theta)
    elif ptype=='PSS':
        posside = rotYall(posSulfoBenzene(),theta)
    elif ptype=='PSTFSI':
        posside = rotYall(posTf2nBenzene(),theta)
    else:
        raise UnknownPSXError('Error : polymer type '+ptype+' is not a recognized PS derivative !')
    posnew = []*nanometer
    posnew.append(posCBB1)
    posnew.append(posCBB2)
    for i in range(len(posside)):
        posnew.append(posCBB1+posside[i])
    posnew = rotYall(posnew,np.random.randint(360))
    return posadd(posnew,posref)

def posPMBmon(posref,theta=54.75):
    rCC = 0.1529*nanometer
    thCCC = np.pi*112.7/180     #CCC angle = 112.7
    costh = cos(thCCC/2)
    sinth = sin(thCCC/2)
    pos1 = rCC*Vec3(0,sinth,costh)
    pos2 = pos1 + rCC*Vec3(0,sinth,-costh)
    pos3 = pos2 + rCC*Vec3(0,sinth,costh)
    pos31 = pos3 + rotY(rCC*Vec3(0,0,1),theta)
    pos4 = pos3 + rCC*Vec3(0,sinth,-costh)
    posnew = []*nanometer
    posnew.append(pos1)
    posnew.append(pos2)
    posnew.append(pos3)
    posnew.append(pos31)
    posnew.append(pos4)
    posnew = rotYall(rotZall(rotYall(posnew,np.random.randint(360)),np.random.randint(360)),np.random.randint(360))
    return posadd(posnew,posref)



def checkOverlap(newPosition, positions):
    for position in positions:
        dr = getDistance(newPosition, position)
        if (dr < sigma):
            return True
    return False

def checkOverlapAll(newPositions, positions):
    for position in positions:
      for posnew in newPositions:
        dr = getDistance(posnew, position)
        if (dr < sigma):
            return True
    return False


def getNewPosition(positions):
    pos0 = randompos()
    while (checkOverlap(pos0,positions)):
        pos0 = randompos()
    return pos0

def posNext(position,sigma):
    costh = np.random.rand()*2-1
    psi = np.random.rand()*2*np.pi
    sinth = sqrt(1 - costh**2)
    pos0 = position.value_in_unit(nanometer)
    r0 = sigma.value_in_unit(nanometer)
    pos = Vec3(pos0[0] + r0*sinth*cos(psi), pos0[1] + r0*sinth*sin(psi), pos0[2] + r0*costh)*nanometer
    return pos

def getNextPosition(newPositions, positions,idx=-1,iiter=0):
    possum = deepcopy(positions)
    if len(newPositions)==0:
        pos1 = getNewPosition(positions)
    else:
        for pos in newPositions:
            possum.append(pos)
        pos1 = posNext(newPositions[idx],sigma)
        while(checkOverlap(pos1,possum)):
            iiter += 1
            if iiter > nmaxmoniter:
                raise MaxMonIterExceedError
            pos1 = posNext(newPositions[idx],sigma)
    return pos1,iiter

def getNewMonomerMB(newPositions, positions):
    natom = 4
    possum = deepcopy(positions)
    for pos in newPositions:
        possum.append(pos)
    idx = np.random.randint(-1,high=1)
    idx = -1
    posmon = []*nanometer
    iiter = 0
    pos1,iiter = getNextPosition(newPositions,positions,idx,iiter)
    posmon.append(pos1)
    for i in range(natom-1):
        pos1,iiter = getNextPosition(posmon,possum,idx,iiter)
        if idx<0:
            posmon.append(pos1)
        else:
            posmon.insert(idx,pos1)
    pos1,iiter = getNextPosition(posmon,possum,natom-2,iiter)
    posmon.insert(-1,pos1)
    return idx, posmon

def getNewPMBpos(positions, numpelen):
    iiter = 0
    possum = deepcopy(positions)
    while True:
        try:
            newPositions = []*nanometer
            imoniter = 0
            for i in range(numpelen):
                idx, posmon = getNewMonomerMB(newPositions, positions)
                if idx==-1:
                    for j in range(len(posmon)):
                        newPositions.append(posmon[j])
                else:
                    for j in range(len(posmon)):
                        newPositions.insert(idx,posmon[j])
            break
        except MaxMonIterExceedError:
            print('Polymer insertion failed. Retrying')
            iiter += 1
            if iiter > nmaxpolyiter:
                raise MaxPolyIterExceedError
    return newPositions


def addNewPMBtop(top, pelen):
    newChain = top.addChain()
    atnames = ['C1','C2','C3','C31','C4']
    bdpairs = [(0,1),(1,2),(2,3),(2,4)]
    for i in range(pelen):
        newAtoms = []
        if i==0:
            resname = 'PMBB'
        elif i==pelen-1:
            resname = 'PMBE'
        else:
            resname = 'PMB'
        newResidue = top.addResidue(resname, newChain, i)
        for j in range(len(atnames)):
            newAtom = top.addAtom(atnames[j],element.carbon,newResidue)
            newAtoms.append(newAtom)
        if i>0:
            top.addBond(newAtoms[0],lastAtom)
        for (k,l) in bdpairs:
            top.addBond(newAtoms[l],newAtoms[k])
        lastAtom = newAtoms[-1]
    return top

def getNewPMBMonTop(top,newChain,lastAtom=[],bLastRes=False,resname='PMB'):
    atnames = ['C1','C2','C3','C31','C4']
    bdpairs = [(0,1),(1,2),(2,3),(2,4)]
    if lastAtom==[]:
        bAddExtBond = False
        resname = resname+'B'
        atnames[0] = 'CT'
    else:
        bAddExtBond = True
        if bLastRes:
            resname = resname+'E'
            atnames[-1] = 'CT'
    newResidue = top.addResidue(resname, newChain)
    newAtoms = []
    for j in range(len(atnames)):
        newAtom = top.addAtom(atnames[j],element.carbon,newResidue)
        newAtoms.append(newAtom)
    if bAddExtBond:
        top.addBond(newAtoms[0],lastAtom)
    for (k,l) in bdpairs:
        top.addBond(newAtoms[l],newAtoms[k])
    return top,newAtoms,newResidue

def getNewPSMonTop(top,newChain,lastAtom=[],bLastRes=False,resname='PS'):
    #newChain = top.addChain()
    atnames = ['CBB1','CBB2','CA','CB1','CB2','CC1','CC2','CD']
    bdpairs = [(0,1),(0,2),(2,3),(2,4),(3,5),(4,6),(5,7),(6,7)]
    idxfirst = 0
    if lastAtom==[]:
        bAddExtBond = False
        resname = resname+'B'
        atnames.insert(0,'CBBT')
        bdpairs.insert(0,(-1,0))
        idxfirst=1
    else:
        bAddExtBond = True
        if bLastRes:
            resname = resname+'E'
    newResidue = top.addResidue(resname, newChain)
    newAtoms = []
    for j in range(len(atnames)):
        newAtom = top.addAtom(atnames[j],element.carbon,newResidue)
        newAtoms.append(newAtom)
    if bAddExtBond:
        top.addBond(newAtoms[0],lastAtom)
    for (k,l) in bdpairs:
        top.addBond(newAtoms[idxfirst+l],newAtoms[idxfirst+k])
    return top,newAtoms,newResidue

def getNewPSSMonTop(top,newChain, lastAtom=[],bLastRes=False,resname='PSS'):
    atnames = ['S','O1','O2','O3']
    atelements = [element.sulfur, element.oxygen, element.oxygen, element.oxygen]
    bdpairs = [(0,1),(0,2),(0,3)]
    top,newAtoms,newResidue = getNewPSMonTop(top,newChain,lastAtom,bLastRes,resname)
    idxfirst = len(newAtoms)
    for j in range(len(atnames)):
        newAtom = top.addAtom(atnames[j],atelements[j],newResidue)
        newAtoms.append(newAtom)
    top.addBond(newAtoms[idxfirst],newAtoms[idxfirst-1])
    for (k,l) in bdpairs:
        top.addBond(newAtoms[idxfirst+l],newAtoms[idxfirst+k])
    return top,newAtoms,newResidue

def getNewPSTFSIMonTop(top,newChain,lastAtom=[],bLastRes=False,resname='PSTFSI'):
    atnames = ['S1','O1','O2','N','S2','O3','O4','C1','F1','F2','F3']
    atelements = [element.sulfur, element.oxygen, element.oxygen, element.nitrogen, element.sulfur, element.oxygen, element.oxygen, element.carbon, element.fluorine, element.fluorine, element.fluorine]
    bdpairs = [(0,1),(0,2),(0,3),(3,4),(4,5),(4,6),(4,7),(7,8),(7,9),(7,10)]
    top,newAtoms,newResidue = getNewPSMonTop(top,newChain,lastAtom,bLastRes,resname)
    idxfirst = len(newAtoms)
    for j in range(len(atnames)):
        newAtom = top.addAtom(atnames[j],atelements[j],newResidue)
        newAtoms.append(newAtom)
    top.addBond(newAtoms[idxfirst],newAtoms[idxfirst-1])
    for (k,l) in bdpairs:
        top.addBond(newAtoms[idxfirst+l],newAtoms[idxfirst+k])
    return top,newAtoms,newResidue

def getNewPSXTop(newTopology,numpelen,ptype='PS'):
    newChain = newTopology.addChain()
    newAtoms=[]
    bLastRes = False
    for i in range(numpelen):
        if i==0:
            lastAtom=[]
        elif i==1:
            lastAtom = newAtoms[2]
        elif (i==numpelen-1):
            lastAtom = newAtoms[1]
            bLastRes=True
        else:
            lastAtom = newAtoms[1]
        if ptype=='PS':
            top,newAtoms,newResidue = getNewPSMonTop(newTopology,newChain,lastAtom,bLastRes,ptype)
        elif ptype=='PSS':
            top,newAtoms,newResidue = getNewPSSMonTop(newTopology,newChain,lastAtom,bLastRes,ptype)
        elif ptype=='PSTFSI':
            top,newAtoms,newResidue = getNewPSTFSIMonTop(newTopology,newChain,lastAtom,bLastRes,ptype)
        else:
            raise UnknownPSXError('Error : polymer type '+ptype+' is not a recognized PS derivative !')
    return top,newAtoms,newResidue


def getNewPSXPMBTop(newTopology,numps,numpsx,numpmb,psxlist,psxtype='PSS'):
    newChain = newTopology.addChain()
    newAtoms=[]
    bLastRes = False
    #if psmixtype=='rand':
#    else:
#        psxlist = np.arange(numpsx)
    for i in range(numps+numpsx+numpmb):
        if i==0:                            #PSX
            lastAtom=[]
        elif i==1:                          #PSXB
            lastAtom = newAtoms[2]
        elif (i<=numps+numpsx):             #PSX
            lastAtom = newAtoms[1]
        else:                               #PMB
            lastAtom = newAtoms[-1]
        if (i==numps+numpsx+numpmb-1):
            bLastRes=True
        if i in psxlist:
            if psxtype=='PS':
                top,newAtoms,newResidue = getNewPSMonTop(newTopology,newChain,lastAtom,bLastRes,psxtype)
            elif psxtype=='PSS':
                top,newAtoms,newResidue = getNewPSSMonTop(newTopology,newChain,lastAtom,bLastRes,psxtype)
            elif psxtype=='PSTFSI':
                top,newAtoms,newResidue = getNewPSTFSIMonTop(newTopology,newChain,lastAtom,bLastRes,psxtype)
            else:
                raise UnknownPSXError('Error : polymer type '+ptype+' is not a recognized PS derivative !')
        elif (i<numps+numpsx):
            top,newAtoms,newResidue = getNewPSMonTop(newTopology,newChain,lastAtom,bLastRes,'PS')
        else:
            top,newAtoms,newResidue = getNewPMBMonTop(newTopology,newChain,lastAtom,bLastRes,'PMB')
    return top,newAtoms,newResidue



def getNewPSXpos(positions, numpelen,ptype='PS'):
    iiter = 0
    possum = deepcopy(positions)
    while True:
        try:
            newPositions = []*nanometer
            pos1 = getNewPosition(positions)
            newPositions.append(pos1)
            for i in range(numpelen):
                imoniter = 0
                theta = (-1)**np.random.randint(2)*54.75
                posmon = posPSXmon(pos1,ptype,theta)
                while(checkOverlapAll(posmon,possum)):
                    imoniter += 1
                    if imoniter > nmaxmoniter:
                        raise MaxMonIterExceedError
                    theta = (-1)**np.random.randint(2)*54.75
                    posmon = posPSXmon(pos1,ptype,theta)
                pos1 = posmon[1]
                for pos in posmon:
                    newPositions.append(pos)
            break
        except MaxMonIterExceedError:
            print('Polymer insertion failed. Retrying')
            iiter += 1
            if iiter > nmaxpolyiter:
                raise MaxPolyIterExceedError
    return newPositions


def getNewPSXPMBpos(positions, numps,numpsx,numpmb,psxlist=[],psxtype='PSS'):
    iiter = 0
    possum = deepcopy(positions)
    if psxlist==[]:
        psxlist = random.sample(range(numps+numpsx),numpsx)
    while True:
        try:
            newPositions = []*nanometer
            pos1 = getNewPosition(positions)
            newPositions.append(pos1)
            for i in range(numps+numpsx):
                imoniter = 0
                theta = (-1)**np.random.randint(2)*54.75
                if i in psxlist:
                    posmon = posPSXmon(pos1,psxtype,theta)
                else:
                    posmon = posPSXmon(pos1,'PS',theta)
                while(checkOverlapAll(posmon,possum)):
                    imoniter += 1
                    if imoniter > nmaxmoniter:
                        raise MaxMonIterExceedError
                    theta = (-1)**np.random.randint(2)*54.75
                    if i in psxlist:
                        posmon = posPSXmon(pos1,psxtype,theta)
                    else:
                        posmon = posPSXmon(pos1,'PS',theta)
                pos1 = posmon[1]
                for pos in posmon:
                    newPositions.append(pos)
            for i in range(numpmb):
                imoniter = 0
                theta = (-1)**np.random.randint(2)*54.75
                posmon = posPMBmon(pos1,theta)
                while(checkOverlapAll(posmon,newPositions)):
                    imoniter += 1
                    if imoniter > nmaxmoniter:
                        raise MaxMonIterExceedError
                    theta = (-1)**np.random.randint(2)*54.75
                    posmon = posPMBmon(pos1,theta)
                pos1 = posmon[-1]
                for pos in posmon:
                    newPositions.append(pos)
            if checkOverlapAll(newPositions,possum):
                raise MaxMonIterExceedError
            break
        except MaxMonIterExceedError:
            print('Polymer insertion failed. Retrying')
            iiter += 1
            if iiter > nmaxpolyiter:
                raise MaxPolyIterExceedError
    return newPositions, psxlist


def getNewMonomer(newPositions, positions):
    possum = deepcopy(positions)
    for pos in newPositions:
        possum.append(pos)
    idx = np.random.randint(-1,high=1)
    pos1 = posNext(newPositions[idx*2],sigma)
    iiter = 0
    while(checkOverlap(pos1,possum)):
        iiter += 1
        if iiter > nmaxmoniter:
            raise MaxMonIterExceedError
        idx = np.random.randint(-1,high=1)
        pos1 = posNext(newPositions[idx*2],sigma)
    pos2 = posNext(pos1,sigma)
    iiter = 0
    while(checkOverlap(pos2,possum)):
        iiter += 1
        if iiter > nmaxmoniter:
            raise MaxMonIterExceedError
        pos2 = posNext(pos1,sigma)
    return idx,pos1,pos2

def getNewMonomerIdx(newPositions, positions,monidx):
    possum = deepcopy(positions)
    for pos in newPositions:
        possum.append(pos)
    pos1 = posNext(newPositions[monidx*2],sigma)
    iiter = 0
    while(checkOverlap(pos1,possum)):
        iiter += 1
        if iiter > nmaxmoniter:
            raise MaxMonIterExceedError
        pos1 = posNext(newPositions[monidx*2],sigma)
    pos2 = posNext(pos1,sigma)
    iiter = 0
    while(checkOverlap(pos2,possum)):
        iiter += 1
        if iiter > nmaxmoniter:
            raise MaxMonIterExceedError
        pos2 = posNext(pos1,sigma)
    return pos1,pos2

def getNewPolymer(positions, numpelen):
    iiter = 0
    possum = deepcopy(positions)
    while True:
        try:
            newPositions = []*nanometer
            pos0 = getNewPosition(positions)
            pos1 = posNext(pos0,sigma)
            imoniter = 0
            while(checkOverlap(pos1,possum)):
                imoniter += 1
                if imoniter > nmaxmoniter:
                    raise MaxMonIterExceedError
                pos1 = posNext(pos0,sigma)
            newPositions.append(pos0)
            newPositions.append(pos1)
            for i in range(numpelen-1):
                idx, pos0,pos1 = getNewMonomer(newPositions, positions)
                if idx==0:
                    newPositions.insert(idx,pos1)
                    newPositions.insert(idx,pos0)
                else:
                    newPositions.append(pos0)
                    newPositions.append(pos1)
            break
        except MaxMonIterExceedError:
            print('Polymer insertion failed. Retrying')
            iiter += 1
            if iiter > nmaxpolyiter:
                raise MaxPolyIterExceedError
    return newPositions

def getNewBranchedPolymer(positions, numpelen, lenbranch, idxbranch):
    iiter = 0
    possum = deepcopy(positions)
    while True:
        try:
            newPositions = []*nanometer
            pos0 = getNewPosition(positions)
            pos1 = posNext(pos0,sigma)
            imoniter = 0
            while(checkOverlap(pos1,possum)):
                imoniter += 1
                if imoniter > nmaxmoniter:
                    raise MaxMonIterExceedError
                pos1 = posNext(pos0,sigma)
            newPositions.append(pos0)
            newPositions.append(pos1)
            idxbb=1
            idxprev = 1
            idxbbprev = 1
            k = -1
            for j in range(1,numpelen):
                idxprev = j-1
                if idxbb in idxbranch:
                    if k<0:
                        idxbbprev=idxprev
                    k += 1
                    if k == lenbranch:
                        idxbb+=1
                else:
                    idxbb += 1
                    if k==lenbranch:
                        k=-1
                        idxprev = idxbbprev
                pos0,pos1 = getNewMonomerIdx(newPositions, positions,idxprev)
                newPositions.append(pos0)
                newPositions.append(pos1)
            break
        except MaxMonIterExceedError:
            print('Polymer insertion failed. Retrying')
            iiter += 1
            if iiter > nmaxpolyiter:
                raise MaxPolyIterExceedError
    return newPositions

def genNewTopology(newTopology):
    for i in range(numchain):
        newChain = newTopology.addChain()
        newAtoms = []
        listCharged = random.sample(range(numpelen),int(numpelen*fcharge))
        for j in range(numpelen):
            if j in listCharged:
                peResname = penResname
                peAtomname = penAtomname
                peElement = penElement
            else:
                peResname = pe0Resname
                peAtomname = pe0Atomname
                peElement = pe0Element
            newResidue = newTopology.addResidue(peResname, newChain, j)
            newAtom = newTopology.addAtom(peAtomname[0], peElement[0], newResidue)
            newAtoms.append(newAtom)
            newAtom = newTopology.addAtom(peAtomname[1], peElement[1], newResidue)
            newAtoms.append(newAtom)
            newTopology.addBond(newAtoms[j*2+1], newAtoms[j*2])
            if j > 0:
                newTopology.addBond(newAtoms[(j-1)*2], newAtoms[j*2])
    return newTopology

###
# calculate radius of gyration for each polymers in the system
# parameters ::
# numchain : # of polymer chains
# numpeatoms : # of atoms in each polymer chain
# positios : position array from the simulation context
###
def RGtot(numchain, numatoms, positions):
    """
    Returns
    -------
    Gyration ratius in units of length (bondlength).
    """
    rgval = np.zeros(numchain)*nanometer
    for i in range(numchain):
        pos = positions[i*numatoms:(i+1)*numatoms].value_in_unit(nanometer)
        rgval[i] = np.sqrt(np.sum(np.var(pos,0)))*nanometer
    return rgval

