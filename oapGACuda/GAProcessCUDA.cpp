#include <stdio.h>
#include <signal.h>
#include <fstream>
#include <gsl/gsl_rng.h>

#include "GAProcessCUDA.h"
#include "GAModuleCUDA.h"
#include "Functions.h"
#include "CudaExecution.h"

namespace ga {

    GAProcessCUDA::GADataImpl::GADataImpl() {
    }

    GAProcessCUDA::GADataImpl::~GADataImpl() {
    }

    float GAProcessCUDA::GetRandomsFloat(void* ptr) {
        gsl_rng* rng = (gsl_rng*) ptr;
        return (float) gsl_rng_uniform(rng);
    }

    double GAProcessCUDA::GetRandomsDobule(void* ptr) {
        gsl_rng* rng = (gsl_rng*) ptr;
        return gsl_rng_uniform(rng);
    }

    const char* namesArray[] = {"cpu", "GeneticAlgorithm"};

    void GAProcessCUDA::setGAData(const GAData& gaData) {
        if (this->gaData == NULL) {
            this->gaData = new GADataImpl();

            this->gaData->realGenomeSize = gaData.realGenomeSize;
            this->gaData->realGeneSize = gaData.realGeneSize;
            this->gaData->realUnitSize = gaData.realUnitSize;
            this->gaData->generationCounter = gaData.generationCounter;
            this->gaData->genomesCount = gaData.genomesCount;

            this->gaData->ranks = new floatt [gaData.genomesCount * gaData.realUnitSize];
            memcpy(this->gaData->ranks, gaData.ranks, gaData.genomesCount * sizeof (floatt ));
            this->gaData->selectedGenomes = new uint[this->gaData->genomesCount * 2];
            memcpy(this->gaData->selectedGenomes, gaData.selectedGenomes, this->gaData->genomesCount * 2);
        }
    }

    GAProcessCUDA::GAProcessCUDA(const char* name, const core::ExecutablesConfigurators& extensionsInstances) : ga::GAProcess(name, extensionsInstances) {
    }

    GAProcessCUDA::~GAProcessCUDA() {
    }

    const char* GAProcessCUDA::getReadableInfo(uint result) {
        return "";
    }

    void GAProcessCUDA::Callback_Impl(int event, void* ptr, void* userPtr) {
        if (event == GAProcessCUDA::EVENT_STOP) {
            ga::CudaOperatorsExecutor* executor = (ga::CudaOperatorsExecutor*) userPtr;
            executor->stop();
        }
    }

    uint GAProcessCUDA::executeGA() {
        ga::CudaOperatorsExecutor executor(this->getIOGAData());
        executor.copy(*this);
        while (this->isStopped() == false) {
            if (this->isStopped() == false && executor.needRandoms()) {
                utils::Iterable<floatt >* it = NULL;
                randomGenerator.generateNumbers(&executor, GAProcessCUDA::Generate, this);
            }
            executor.start();
        }
        return 0;
    }

    GAProcessCUDA::GARandomsGenerator::GARandomsGenerator() : comp(-1), wasSeed(false) {
        this->rng = gsl_rng_alloc(gsl_rng_mt19937_1999);
    }

    void GAProcessCUDA::GARandomsGenerator::setSeed(unsigned int long seed) {
        gsl_rng_set(this->rng, seed);
        wasSeed = true;
    }

    void GAProcessCUDA::GARandomsGenerator::setSeed() {
        if (!wasSeed) {
            this->setSeed(time(NULL));
        }
    }

    GAProcessCUDA::GARandomsGenerator::~GARandomsGenerator() {
        gsl_rng_free(rng);
    }

    floatt  GAProcessCUDA::GARandomsGenerator::generate() {
        floatt  random = gsl_rng_uniform(this->rng);
        if (comp < -1) {
            comp = random;
        } else if (comp == random) {
            gsl_rng_set(this->rng, time(NULL));
            random = gsl_rng_uniform(this->rng);
        }
        return random;
    }

    floatt  GAProcessCUDA::Generate(void* userPtr) {
        GAProcessCUDA* thiz = (GAProcessCUDA*) userPtr;
        return thiz->gaRandomsGenerator.generate();
    }
}