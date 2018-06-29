/* -------------------------------------------------------------------------- *
 *                               OpenMMConstV                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c)                                                     *
 *         2018 California Institute and Techology and the Authors.           *
 * Authors: Chang Yun Son                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaConstVKernels.h"
#include "CudaConstVKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaForceInfo.h"
#include "CudaIntegrationUtilities.h"
#include "SimTKOpenMMRealType.h"
#include "openmm/ConstVLangevinIntegrator.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <set>
#include <iostream>



using namespace OpenMM;
using namespace std;


CudaIntegrateConstVLangevinStepKernel::~CudaIntegrateConstVLangevinStepKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaIntegrateConstVLangevinStepKernel::initialize(const System& system, const ConstVLangevinIntegrator& integrator) {
    cu.getPlatformData().initializeContexts(system);
    cu.setAsCurrent();
    cu.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    map<string, string> defines;
    CUmodule module = cu.createModule(CudaConstVKernelSources::vectorOps+CudaConstVKernelSources::constVLangevin, defines, "");
    kernel1 = cu.getKernel(module, "integrateConstVLangevinPart1");
    kernel2 = cu.getKernel(module, "integrateConstVLangevinPart2");
    kernelImage = cu.getKernel(module, "updateImageParticlePositions");
    params = new CudaArray(cu, 3, cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float), "langevinParams");
    prevStepSize = -1.0;

    // Check image particles are properly set (same number as original, mass = 0)
    int numAtoms = system.getNumParticles();
    if (numAtoms%2 != 0)
        throw OpenMMException("Not even number of particles");
    for (int i = numAtoms/2; i < numAtoms; i++) {
        if (system.getParticleMass(i) != 0.0)
            throw OpenMMException("Image Particle has nonzero mass");
    }

    // Initialize the positions of the image particles
    int numRealAtoms = numAtoms/2;
    CUdeviceptr posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    void* argsImage[] = {&numRealAtoms, &cu.getPosq().getDevicePointer(), &posCorrection, &cu.getAtomIndexArray().getDevicePointer()};
    cu.executeKernel(kernelImage, argsImage, numRealAtoms, 128);
}

void CudaIntegrateConstVLangevinStepKernel::execute(ContextImpl& context, const ConstVLangevinIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    int numRealAtoms = numAtoms/2;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    cu.getIntegrationUtilities().setNextStepSize(stepSize);
    if (temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Calculate the integration parameters.

        double kT = BOLTZ*temperature;
        double vscale = exp(-stepSize*friction);
        double fscale = (friction == 0 ? stepSize : (1-vscale)/friction);
        double noisescale = sqrt(kT*(1-vscale*vscale));
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            vector<double> p(params->getSize());
            p[0] = vscale;
            p[1] = fscale;
            p[2] = noisescale;
            params->upload(p);
        }
        else {
            vector<float> p(params->getSize());
            p[0] = (float) vscale;
            p[1] = (float) fscale;
            p[2] = (float) noisescale;
            params->upload(p);
        }
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }

    // Call the first integration kernel.

    int randomIndex = integration.prepareRandomNumbers(cu.getPaddedNumAtoms());
    void* args1[] = {&numAtoms, &paddedNumAtoms, &cu.getVelm().getDevicePointer(), &cu.getForce().getDevicePointer(), &integration.getPosDelta().getDevicePointer(),
            &params->getDevicePointer(), &integration.getStepSize().getDevicePointer(), &integration.getRandom().getDevicePointer(), &randomIndex};
    cu.executeKernel(kernel1, args1, numAtoms, 128);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    CUdeviceptr posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    void* args2[] = {&numAtoms, &cu.getPosq().getDevicePointer(), &posCorrection, &integration.getPosDelta().getDevicePointer(),
            &cu.getVelm().getDevicePointer(), &integration.getStepSize().getDevicePointer()};
    cu.executeKernel(kernel2, args2, numAtoms, 128);
    integration.computeVirtualSites();

    // Call the image charge position update kernel.

    void* argsImage[] = {&numRealAtoms, &cu.getPosq().getDevicePointer(), &posCorrection, &cu.getAtomIndexArray().getDevicePointer()};
    cu.executeKernel(kernelImage, argsImage, numRealAtoms, 128);

    // Update the time and step count.

    cu.setTime(cu.getTime()+stepSize);
    cu.setStepCount(cu.getStepCount()+1);
    cu.reorderAtoms();
}

double CudaIntegrateConstVLangevinStepKernel::computeKineticEnergy(ContextImpl& context, const ConstVLangevinIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}

