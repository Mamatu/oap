/* 
 * File:   GeneticAlgoModule.h
 * Author: marcin
 *
 * Created on 04 May 2013, 20:45
 */

#ifndef GAMODULECPU_H
#define	GAMODULECPU_H

#include "GACore.h"
#include "Functions.h"
#include "types.h"
#include "WrapperInterfaces.h"
#include "GAProcessCUDA.h"

namespace ga {

    class GAModuleCUDA : public ga::GAModule {
    public:
        static ga::GAModuleCUDA* Create();
        static void Destroy(ga::GAModuleCUDA* gaModuleCUDA);
    protected:
        GAModuleCUDA();
        virtual ~GAModuleCUDA();

        ga::GAProcessConfigurator* newGAModuleInstance();
        void deleteGAModuleInstance(ga::GAProcessConfigurator* object);
    };

    class GAModuleInstanceCUDA : public ga::GAProcessConfigurator, public Parameters {
        SetSelectionType setSelectionTypeFunc;
        SetCrossoverType setCrossoverTypeFunc;
        SetMutationType setMutationTypeFunc;
        SetSeed setSeedFunc;
        BoundaryMutation boundaryMutationObj;
        synchronization::Barrier barrier;
        utils::FunctionsContainer functionsContainer;
        utils::ObjectsContainer objectsContainer;
        void deleteGeneration(GAData::Generation& generation);
        synchronization::Mutex rwmutex;
    protected:
        ga::GAProcess* newGAProcess(const char* name, const core::ExecutablesConfigurators& extensionInstances);
        void deleteGAProcess(ga::GAProcess* process);
    public:
        GAModuleInstanceCUDA(GAModuleCUDA* gaModuleCPU);
        virtual ~GAModuleInstanceCUDA();
    };
}

void GetModulesCount(uint& count);
void LoadModules(core::ProcessConfigurator** objects);
void UnloadModule(core::ProcessConfigurator* object);

void GetComponentsCount(uint& count);
void LoadComponents(core::ExecutableConfigurator** objects);
void UnloadComponent(core::ExecutableConfigurator* object);
#endif	/* GENETICALGOMODULE_H */

