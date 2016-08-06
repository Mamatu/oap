#ifndef CU_MUTATION_H
#define CU_MUTATION_H

#include "cu_types.h"
#include "cu_utils.h"

__device__
void executeBoundaryMutation(CuData* cuData) {
    for (uint fa = 0; fa < cuData->cuParameters->mutationsCount; fa++) {
        uint genomeIndex = random_getUint(cuData, 0, cuData->genomesCount);
        uint genomeOffset = cuData->getGeneMemorySize() * genomeIndex;
        floatt * ptr = (cuData->currentGeneration->genomes + genomeOffset);
        for (uint fb = 0; cuData->cuParameters->rangesCount &&
                fb + genomeOffset < cuData->getGeneMemorySize(); fb++) {
            floatt  min = cuData->cuParameters->ranges[fb * 2];
            floatt  max = cuData->cuParameters->ranges[fb * 2 + 1];
            floatt  value = random_get(cuData, min, max);
            *(ptr + fb) = value;
        }
    }
}

#endif
