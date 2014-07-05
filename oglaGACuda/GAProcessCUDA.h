
#ifndef GAPROCESSCPU_H
#define	GAPROCESSCPU_H

#include "GARatingCUDA.h"
#include "NumbersGenerator.h"
#include "Parameters.h"

namespace ga {

    class GAModuleInstanceCUDA;

    class GAProcessCUDA : public GAProcess, public Parameters {

        class GADataImpl : public ga::GAData {
        public:
            GADataImpl();
            virtual ~GADataImpl();
        };

        /**
         */
        class GARandomsGenerator {
            floatt  comp;
            gsl_rng* rng;
            bool wasSeed;
        public:
            GARandomsGenerator();
            virtual ~GARandomsGenerator();
            void setSeed(unsigned int long seed);
            void setSeed();
            floatt  generate();
        };

        GARandomsGenerator gaRandomsGenerator;
        uint concurrentExecutionsCount;

        utils::NumbersAsyncGenerator<floatt > randomGenerator;
        uint randomsFactor;
        uint buffersCount;
        bool prepapreExecution();
        void destroyExecution();
        static void* Execute(void* ptr);
        static floatt  Generate(void* userPtr);
        static float GetRandomsFloat(void* ptr);
        static double GetRandomsDobule(void* ptr);
        GARatingEntityCUDA* rating;
        floatt * ranksSums;
        uint cuEntity;
        bool stopped;
    protected:
        static void Callback_Impl(int event, void* ptr, void* userPtr);
        GADataImpl* gaData;
        uint gaDatasCount;
        core::ExecutablesConfigurators extensionInstances;
        void setGAData(const GAData& gaData);
        void deleteGAData();
        uint executeGA();
        const char* getReadableInfo(uint result);
        friend class GAModuleInstanceCUDA;
    public:
        GAProcessCUDA(const char* name, const core::ExecutablesConfigurators& apiComponents);
        ~GAProcessCUDA();

    };
};


#endif	/* POPULATIONINFO_H */

