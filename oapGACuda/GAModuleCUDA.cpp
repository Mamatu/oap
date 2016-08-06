/* 
 * File:   GeneticAlgoModule.cpp
 * Author: marcin
 * 
 * Created on 04 May 2013, 20:45
 */

#include "GAModuleCUDA.h"
#include "GAProcessCUDA.h"


namespace ga {

    GAModuleCUDA* GAModuleCUDA::Create() {
        return new GAModuleCUDA();
    }

    void GAModuleCUDA::Destroy(GAModuleCUDA* gaModuleCPU) {
        delete gaModuleCPU;
    }

    ga::GAProcess* GAModuleInstanceCUDA::newGAProcess(const char* name, const core::ExecutablesConfigurators& extensionInstances) {
        ga::GAProcessCUDA* process = new ga::GAProcessCUDA(this, extensionInstances);
        process->copy(*this);
        return process;
    }

    void GAModuleInstanceCUDA::deleteGAProcess(ga::GAProcess* process) {
        ga::GAProcessCUDA* ptr = dynamic_cast<ga::GAProcessCUDA*> (process);
        if (ptr) {
            delete ptr;
        }
    }

    GAModuleInstanceCUDA::GAModuleInstanceCUDA(GAModuleCUDA* gaModuleCPU) : ga::GAProcessConfigurator(gaModuleCPU, "GAModuleCpu") {
        setMutationTypeFunc.setParameters(this);
        setCrossoverTypeFunc.setParameters(this);
        setSelectionTypeFunc.setParameters(this);
        functionsContainer.add(&setMutationTypeFunc);
        functionsContainer.add(&setCrossoverTypeFunc);
        functionsContainer.add(&setSelectionTypeFunc);
        objectsContainer.add(&boundaryMutationObj);
        this->setFunctionsContainer(&functionsContainer);
        this->setObjectsContainer(&objectsContainer);
    }

    GAModuleCUDA::GAModuleCUDA() : ga::GAModule("GAModuleCpu") {
    }

    GAModuleCUDA::~GAModuleCUDA() {
    }

    ga::GAProcessConfigurator* GAModuleCUDA::newGAModuleInstance() {
        return new GAModuleInstanceCUDA(this);
    }

    void GAModuleCUDA::deleteGAModuleInstance(ga::GAProcessConfigurator* object) {
        ga::GAModuleInstanceCUDA* ptr = dynamic_cast<ga::GAModuleInstanceCUDA*> (object);
        if (ptr) {
            delete ptr;
        }
    }
}



void GetComponentsCount(uint& count) {
    count = 0;
}

void LoadComponents(core::ExecutableConfigurator** objects) {
}

void UnloadComponent(core::ExecutableConfigurator* object) {
}

void GetModulesCount(uint& count) {
    count = 1;
}

void LoadModules(core::OapModule** objects) {
    objects[0] = ga::GAModuleCUDA::Create();
}

void UnloadModule(core::OapModule* object) {
    ga::GAModuleCUDA* ptr = dynamic_cast<ga::GAModuleCUDA*> (object);
    if (ptr) {
        ga::GAModuleCUDA::Destroy(ptr);
    }
}
