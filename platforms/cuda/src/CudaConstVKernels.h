#ifndef CUDA_CONSTV_KERNELS_H_
#define CUDA_CONSTV_KERNELS_H_

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

#include "openmm/ConstVKernels.h"
#include "CudaContext.h"
#include "CudaArray.h"

namespace OpenMM {


/**
 * This kernel is invoked by ConstVLangevinIntegrator to take one time step.
 */
class CudaIntegrateConstVLangevinStepKernel : public IntegrateConstVLangevinStepKernel {
public:
    CudaIntegrateConstVLangevinStepKernel(std::string name, const Platform& platform, CudaContext& cu) : IntegrateConstVLangevinStepKernel(name, platform), cu(cu), params(NULL), invAtomOrder(NULL) {
    }
    ~CudaIntegrateConstVLangevinStepKernel();
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ConstVLangevinIntegrator this kernel will be used for
     */
    void initialize(const System& system, const ConstVLangevinIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the ConstVLangevinIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const ConstVLangevinIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the ConstVLangevinIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const ConstVLangevinIntegrator& integrator);
private:
    CudaContext& cu;
    double prevTemp, prevFriction, prevStepSize, zmax;
    CudaArray* params;
    CudaArray* invAtomOrder;
//    CudaArray* idxqptcls;
    CUfunction kernel1, kernel2, kernelImage, kernelReorder;
};


/**
 * This kernel is invoked by ConstVDrudeLangevinIntegrator to take one time step
 */
class CudaIntegrateConstVDrudeLangevinStepKernel : public IntegrateConstVDrudeLangevinStepKernel {
public:
    CudaIntegrateConstVDrudeLangevinStepKernel(std::string name, const Platform& platform, CudaContext& cu) :
            IntegrateConstVDrudeLangevinStepKernel(name, platform), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ConstVDrudeLangevinIntegrator this kernel will be used for
     * @param force      the DrudeForce to get particle parameters from
     */
    void initialize(const System& system, const ConstVDrudeLangevinIntegrator& integrator, const DrudeForce& force);
    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the ConstVDrudeLangevinIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const ConstVDrudeLangevinIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     * 
     * @param context     the context in which to execute this kernel
     * @param integrator  the ConstVDrudeLangevinIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const ConstVDrudeLangevinIntegrator& integrator);
private:
    CudaContext& cu;
    double prevStepSize, zmax;
    CudaArray normalParticles;
    CudaArray pairParticles;
    CudaArray invAtomOrder;
    CUfunction kernel1, kernel2, hardwallKernel, kernelImage, kernelReorder;
};


} // namespace OpenMM

#endif /*CUDA_CONSTV_KERNELS_H_*/
