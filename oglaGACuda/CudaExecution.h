#ifndef CU_GAAPI_H
#define CU_GAAPI_H

#include "types.h"
#include "GACore.h"
#include "Parameters.h"
#include "cu_types.h"
#include "cu_struct.h"
#include "GARatingCUDA.h"

namespace ga {

    class CudaOperatorsExecutor : public utils::CallbacksManager, public Parameters, public ga::GAExecutableRandsImpl {
        uint neededRandomsNumber_mutation;
        uint neededRandomsNumber_selection;
        uint neededRandomsNumber_crossover;
        uint randomsCount;
        uint randomsTotalCount;

        bool _needRandoms;
        bool _end;

        ga::GAData* gaData;
        CuData* cuData;
        CuParameters* cuParameters;
        CuRandoms* cuRandom;
        CuRandoms* cuRandom1;
        CuRandoms* cuRandom2;
        uint index;
        bool isGAData;
        int deleteGeneration(CuGeneration* generation);
        void copyRandomsToDevice(floatt * buffer);

        int copyGADataToDevice(ga::GAData* gaData);
        int copyRandomToDevice(CuRandoms* cuRandom);
        int copyParametersToDevice(Parameters& parameters);

        void createBuffers();
        void executeGA();
        GARatingExecutorCUDA* gaRatingExecutableCuda;
    public:
        CudaOperatorsExecutor(ga::GAData* gaData);
        virtual ~CudaOperatorsExecutor();

        void setMutationType(char type);
        void setCrossoverType(char type);
        void setSelectionType(char type);

        bool end();
        void add(floatt  t);
        bool needRandoms();
        Parameters parameters;

        void start();
        void stop();

        int deleteCuData();
        int deleteCuParameters();
    };
}

#endif
