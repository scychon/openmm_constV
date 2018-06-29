%module constvplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "OpenMMConstV.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/
%pythonappend OpenMM::ConstVLangevinIntegrator::getTemperature() const %{
   val=unit.Quantity(val, unit.kelvin)
%}

%pythonappend OpenMM::ConstVLangevinIntegrator::getFriction() const %{
   val=unit.Quantity(val, 1/unit.picosecond)
%}

namespace OpenMM {

class ConstVLangevinIntegrator : public Integrator {
public:
   ConstVLangevinIntegrator(double temperature, double frictionCoeff, double stepSize) ;

   double getTemperature() const ;
   void setTemperature(double temp) ;
   double getFriction() const ;
   void setFriction(double coeff) ;
   int getRandomNumberSeed() const ;
   void setRandomNumberSeed(int seed) ;
   virtual void step(int steps) ;
};

}
