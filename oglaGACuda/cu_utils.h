#ifndef CU_UTILS_H
#define	CU_UTILS_H


#include "cu_struct.h"

/****************************************************************************************************/
__device__ __forceinline__
void cu_memcpy(const void* dst, void* src, uint size) {
    
}
/****************************************************************************************************/

__device__ __forceinline__
floatt * genetic_getGenome(floatt * genomes, uint realGenomeSize, uint realGeneSize, uint genomeIndex) {
    return (genomes + genomeIndex * realGeneSize * realGenomeSize);
}

__device__ __forceinline__
floatt * genetic_getGenome(CuGeneration* generation, CuData* cuData, uint geneIndex) {
    return genetic_getGenome(generation->genomes, cuData->realGenomeSize, cuData->realGeneSize, geneIndex);
}

/****************************************************************************************************/

__device__ __forceinline__
floatt * genetic_getGene(floatt * genome, uint realGeneSize, uint geneIndex) {
    return (genome + geneIndex * realGeneSize);
}

__device__ __forceinline__
floatt * genetic_getGene(floatt * genome, CuData* cuData, uint geneIndex) {
    return genetic_getGene(genome, cuData->realGeneSize, geneIndex);
}

/****************************************************************************************************/

__device__ __forceinline__
floatt * genetic_getGene(floatt * genomes, uint realGenomeSize, uint realGeneSize, uint genomeIndex, uint geneIndex) {
    return genetic_getGenome(genomes, realGenomeSize, realGeneSize, genomeIndex) + realGeneSize * geneIndex;
}

__device__ __forceinline__
floatt * genetic_getGene(CuGeneration* generation, CuData* cuData, uint genomeIndex, uint geneIndex) {
    return genetic_getGene(generation->genomes, cuData->realGenomeSize, cuData->realGeneSize, genomeIndex, geneIndex);
}

/****************************************************************************************************/

__device__ __forceinline__
int random_getInt(floatt * randoms, uint index, int min, int max) {
    return (int) randoms[index] * (max - min) + min;
}

__device__ __forceinline__
int random_getInt(CuData* cuData, int min, int max) {
    return (int) cuData->cuRandom->randoms[cuData->cuRandom->randomsIndex++] * (max - min) + min;
}

__device__ __forceinline__
uint random_getUint(floatt * randoms, uint index, uint min, uint max) {
    return (uint) randoms[index] * (max - min) + min;
}

__device__ __forceinline__
uint random_getUint(CuData* cuData, uint min, uint max) {
//    return (uint) (cuData->cuRandom->randoms[cuData->cuRandom->randomsIndex++] * (max - min) + min);
}

__device__ __forceinline__
floatt  random_get(floatt * randoms, uint index, floatt  min, floatt  max) {
    return (floatt ) randoms[index] * (max - min) + min;
}

__device__ __forceinline__
floatt  random_get(CuData* cuData, floatt  min, floatt  max) {
    return (floatt ) cuData->cuRandom->randoms[cuData->cuRandom->randomsIndex++] * (max - min) + min;
}

#endif
