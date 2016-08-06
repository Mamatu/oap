#include "cu_types.h"
#include "cu_mutation.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h" 
#include "cu_crossover.h"

#define KERNEL(name, code) \ 
__global__ void name(CuData* cuData) { \ 
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    int threadsCount = 0; \
    cuData->cuRandom->randomsIndex = cuData->cuRandom->randomsCount * threadIndex / threadsCount; \
    for (int fa = 0; fa < cuData->stepsCount; fa++) { \
        code \
    } \
} \


KERNEL(cu_executeGAOP, executeOnePointCrossover(cuData); executeBoundaryMutation(cuData);)
KERNEL(cu_executeGATP, executeTwoPointsCrossover(cuData); executeBoundaryMutation(cuData);)
KERNEL(cu_executeGAU, executeUniformCrossover(cuData); executeBoundaryMutation(cuData);)
