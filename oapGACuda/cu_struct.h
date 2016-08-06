/* 
 * File:   cu_struct.h
 * Author: mmatula
 *
 * Created on June 10, 2013, 9:20 PM
 */

#ifndef CU_STRUCT_H
#define	CU_STRUCT_H

#include "cu_types.h"

struct CuGeneration {
    uint* genomesSizes;
    uint* genesSizes;
    floatt * genomes;
};

struct CuRandoms {
    floatt * randoms;
    uint randomsIndex;
    uint randomsCount;
};

struct CuParameters {
    char mutationType;
    char selectionType;
    char crossoverType;
    uint mutationsCount;
    uint rangesCount;
    floatt * ranges;
};

struct CuData {
    floatt * ranks;
    uint* selectedGenomes;
    uint selectedGenomesCount;
    CuGeneration* previousGeneration;
    CuGeneration* currentGeneration;

    uint realGenomeSize;
    uint realGeneSize;
    uint realUnitSize;

    uint genomesCount;
    uint realGenomeSizeInBytes;
    uint genomesSizeInBytes;

    CuRandoms* cuRandom;
    CuRandoms* cuRandom1;
    CuRandoms* cuRandom2;
    CuParameters* cuParameters;
    
    uint stepsCount;
    
    __device__ __forceinline__ uint getGenomeMemorySize() {
        return realGenomeSize * realGeneSize * realUnitSize;
    }

    __device__ __forceinline__ uint getGeneMemorySize() {
        return realGeneSize * realUnitSize;
    }

    __device__ __forceinline__ uint getGenomesSizesCount() {
        return genomesCount;
    }

    __device__ __forceinline__ uint getGenesSizesCount() {
        return realGenomeSize * genomesCount* realUnitSize;
    }

    __device__ __forceinline__ uint getRanksMemorySize() {
        return genomesCount * realUnitSize;
    }

    __device__ __forceinline__ uint getSelectedMemorySize() {
        return genomesCount * sizeof (uint);
    }

    __device__ __forceinline__ uint getGenerationMemorySize() {
        return realGenomeSize * realGeneSize * genomesCount * realUnitSize;
    }
};


#endif	/* CU_STRUCT_H */

