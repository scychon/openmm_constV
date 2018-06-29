#ifndef OPENMM_CONSTVLANGEVIN_INTEGRATOR_PROXY_H_
#define OPENMM_CONSTVLANGEVIN_INTEGRATOR_PROXY_H_

#include "openmm/serialization/XmlSerializer.h"

namespace OpenMM {

class ConstVLangevinIntegratorProxy : public SerializationProxy {
public:
    ConstVLangevinIntegratorProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

}

#endif /*OPENMM_CONSTVLANGEVIN_INTEGRATOR_PROXY_H_*/
