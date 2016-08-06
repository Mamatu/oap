#include "CudaExecution.h"
#include "cu_struct.h"
#include "Parameters.h"

namespace ga {
    class CudaOperatorsExecutor;
    class EventHandle;

    uint ga_CreateCuInternalGeneration(CuGeneration* dst, ga::GAData::Generation& src);

    typedef std::vector<CudaOperatorsExecutor*> CuContextVector;
    typedef std::vector<uint> EventHandles;

    CudaOperatorsExecutor::CudaOperatorsExecutor(ga::GAData* _gaData) : GAExecutableRandsImpl(NULL), cuData(NULL), randomsCount(0), _end(false), _needRandoms(true),
    neededRandomsNumber_mutation(0),
    neededRandomsNumber_selection(0),
    neededRandomsNumber_crossover(0), gaData(_gaData) {
        this->copyGADataToDevice(_gaData);
    }

    CudaOperatorsExecutor::~CudaOperatorsExecutor() {
    }

    bool CudaOperatorsExecutor::needRandoms() {
        return this->_needRandoms;
    }

    void CudaOperatorsExecutor::createBuffers() {
        uint randomsTotalCount = (this->neededRandomsNumber_crossover + this->neededRandomsNumber_mutation + this->neededRandomsNumber_selection);
        this->randomsTotalCount = randomsTotalCount;
        this->index = 0;
        cuMemAlloc((CUdeviceptr*) & this->cuRandom1, sizeof (CuRandoms));
        cuMemAlloc((CUdeviceptr*) & this->cuRandom1->randoms, sizeof (floatt ) * this->randomsTotalCount);
        cuMemAlloc((CUdeviceptr*) & this->cuRandom2, sizeof (CuRandoms));
        cuMemAlloc((CUdeviceptr*) & this->cuRandom2->randoms, sizeof (floatt ) * this->randomsTotalCount);
        ga::GAExecutableRandsImpl::createBuffers(randomsTotalCount, this->getRandomsFactor());
    }

    bool CudaOperatorsExecutor::end() {
        bool end = this->_end;
        this->copyRandomsToDevice(this->getLastFilledBuffer());
        return end;
    }

    void CudaOperatorsExecutor::copyRandomsToDevice(floatt * buffer) {
        if (this->getLastFilledBuffer() == this->getBuffer1()) {
            cuMemcpyHtoD((CUdeviceptr)this->cuRandom1->randoms, this->getBuffer1(), this->randomsTotalCount * sizeof (floatt ));
        } else {
            cuMemcpyHtoD((CUdeviceptr)this->cuRandom2->randoms, this->getBuffer2(), this->randomsTotalCount * sizeof (floatt ));
        }
    }

    int CudaOperatorsExecutor::copyGADataToDevice(ga::GAData* gaData) {
        CUdeviceptr cuCurrentGeneration;
        CUdeviceptr cuPreviousGeneration;
        cuMemAlloc(&cuPreviousGeneration, sizeof (CuGeneration));
        cuMemAlloc(&cuCurrentGeneration, sizeof (CuGeneration));

        int result1 = ga_CreateCuInternalGeneration((CuGeneration*) cuCurrentGeneration, gaData->currentGeneration);
        int result2 = ga_CreateCuInternalGeneration((CuGeneration*) cuPreviousGeneration, gaData->previousGeneration);

        if (result1 == 0 && result2 == 0) {
            CUdeviceptr cuData;
            CUdeviceptr cuSelectedGenomes;

            cuMemAlloc(&cuData, sizeof (CuData));
            cuMemAlloc(&cuSelectedGenomes, gaData->genomesCount * 2 * sizeof (uint));

            CuData* cuDataCasted = (CuData*) cuData;
            cuMemcpyDtoD((CUdeviceptr) cuDataCasted->currentGeneration, cuCurrentGeneration, sizeof (CUdeviceptr));
            cuMemcpyDtoD((CUdeviceptr) cuDataCasted->previousGeneration, cuPreviousGeneration, sizeof (CUdeviceptr));
            cuMemcpyDtoD((CUdeviceptr) cuDataCasted->selectedGenomes, cuSelectedGenomes, sizeof (CUdeviceptr));

            this->isGAData = true;
            this->createBuffers();
            return 0;
        }
        return 1;
    }

    int CudaOperatorsExecutor::copyRandomToDevice(CuRandoms* cuRandom) {
        this->cuRandom = cuRandom;
        return 0;
    }

    int CudaOperatorsExecutor::copyParametersToDevice(Parameters& parameters) {
        cuMemAlloc((CUdeviceptr*) & this->cuParameters, sizeof (this->cuParameters));
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->crossoverType, (void*) parameters.getCrossoverType(), sizeof (parameters.getCrossoverType()));
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->mutationType, (void*) parameters.getMutationType(), sizeof (parameters.getMutationType()));
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->selectionType, (void*) parameters.getSelectionType(), sizeof (parameters.getSelectionType()));
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->mutationsCount, (void*) parameters.getMutationsCount(), sizeof (parameters.getMutationsCount()));
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->rangesCount, (void*) parameters.getRangesCount(), sizeof (parameters.getRangesCount()));
        cuMemAlloc((CUdeviceptr*) & this->cuParameters->ranges, parameters.getRangesCount() * sizeof (floatt )*2);
        floatt * rangesTemp = new floatt [parameters.getRangesCount() * 2];
        for (uint fa = 0; fa < parameters.getRangesCount(); fa++) {
            floatt  min, max;
            parameters.getRandomsRange(min, max, fa);
            rangesTemp[fa * 2] = min;
            rangesTemp[fa * 2 + 1] = max;
        }
        cuMemcpyHtoD((CUdeviceptr) this->cuParameters->ranges, rangesTemp, sizeof (floatt ) * parameters.getRangesCount()*2);
        delete[] rangesTemp;
        return 0;
    }

    int CudaOperatorsExecutor::deleteCuParameters() {
        cuMemFree((CUdeviceptr)this->cuParameters->ranges);
        cuMemFree((CUdeviceptr)this->cuParameters);
        return 0;
    }

    int CudaOperatorsExecutor::deleteGeneration(CuGeneration* generation) {
        cuMemFree((CUdeviceptr) generation->genesSizes);
        cuMemFree((CUdeviceptr) generation->genomes);
        cuMemFree((CUdeviceptr) generation->genomesSizes);
        return 0;
    }

    int CudaOperatorsExecutor::deleteCuData() {
        if (this->cuData == NULL) {
            return 2;
        }
        this->deleteGeneration(this->cuData->currentGeneration);
        this->deleteGeneration(this->cuData->previousGeneration);

        cuMemFree((CUdeviceptr)this->cuData->previousGeneration);
        cuMemFree((CUdeviceptr)this->cuData->currentGeneration);

        cuMemFree((CUdeviceptr)this->cuData->ranks);
        cuMemFree((CUdeviceptr)this->cuData->selectedGenomes);

        cuMemFree((CUdeviceptr)this->cuData);
        this->isGAData = false;
        return 0;
    }

    void CudaOperatorsExecutor::setMutationType(char mutationType) {
        Parameters::setMutationType(mutationType);
        switch (mutationType) {
            case 0:
                break;

            default:
                this->neededRandomsNumber_mutation = this->getMutationsCount(); // &GAComponentCPU::InternalExecutable::executeBoundaryMutation;
                break;
        }
    }

    void CudaOperatorsExecutor::setSelectionType(char selectionType) {
        Parameters::setSelectionType(selectionType);
        switch (selectionType) {
            case 0:
                this->neededRandomsNumber_selection = 2 * gaData->genomesCount;
                break;
            default:
                this->neededRandomsNumber_selection = 0;
                break;
        }
    }

    void CudaOperatorsExecutor::setCrossoverType(char crossoverType) {
        Parameters::setCrossoverType(crossoverType);
        switch (crossoverType) {
            case 0:
                this->neededRandomsNumber_crossover = 2 * gaData->genomesCount;
                break;
            case 1:
                this->neededRandomsNumber_crossover = 4 * gaData->genomesCount;
                break;
            case 2:
                break;
            default:
                this->neededRandomsNumber_crossover = gaData->genomesCount * gaData->realGenomeSize;
                break;
        }
    }

    int ga_TransformPointers(ga::GAData::Generation* dst, CuGeneration * src) {
        dst->genomesSizes = (uint*) (src->genomesSizes);
        dst->genesSizes = (uint*) (src->genesSizes);
        dst->genomes = (char*) (src->genomes);
    }

    uint ga_CreateCuInternalGeneration(CuGeneration* dst, ga::GAData::Generation& src) {
        ga::GAData* gaData = src.getGAData();
        if (src.genomesSizes) {
            cuMemAlloc((CUdeviceptr*)&(dst->genomesSizes), gaData->getGenomesSizesCount());
            cuMemcpyHtoD((CUdeviceptr) dst->genomesSizes, src.genomesSizes, gaData->getGenesSizesCount());
        } else {
            cuMemFree((CUdeviceptr) dst->genomesSizes);
            return 1;
        }

        if (src.genesSizes) {
            cuMemAlloc((CUdeviceptr*)&(dst->genesSizes), gaData->getGenesSizesCount());
            cuMemcpyHtoD((CUdeviceptr) dst->genesSizes, src.genesSizes, gaData->getGenesSizesCount());
        } else {
            cuMemFree((CUdeviceptr) dst->genomesSizes);
            cuMemFree((CUdeviceptr) dst->genesSizes);
            return 2;
        }

        if (src.genomes) {
            cuMemAlloc((CUdeviceptr*)&(dst->genomes), gaData->getGenerationMemorySize());
            cuMemcpyHtoD((CUdeviceptr) dst->genomes, src.genomes, gaData->getGenerationMemorySize());
        } else {
            cuMemFree((CUdeviceptr) dst->genomesSizes);
            cuMemFree((CUdeviceptr) dst->genesSizes);
            cuMemFree((CUdeviceptr) dst->genomes);
            return 3;
        }
        return 0;
    }

    void CudaOperatorsExecutor::executeGA() {
        gaRatingExecutableCuda->execute();
        
    }

    void CudaOperatorsExecutor::start() {
        this->executeGA();
    }

    void CudaOperatorsExecutor::stop() {
    }
}

