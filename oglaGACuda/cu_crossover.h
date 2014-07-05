#ifndef CU_CROSSOVER_H
#define CU_CROSSOVER_H

#include "cu_types.h"
#include "cu_utils.h"

__device__
void executeOnePointCrossover(CuData* cuData) {
    for (uint fa = 0; fa < cuData->selectedGenomesCount; fa += 2) {
        uint genomeIndex1 = cuData->selectedGenomes[fa];
        uint genomeIndex2 = cuData->selectedGenomes[fa + 1];

        uint genomeSize1 = cuData->previousGeneration->genomesSizes[genomeIndex1];
        uint genomeSize2 = cuData->previousGeneration->genomesSizes[genomeIndex2];

        uint geneIndex1 = random_getUint(cuData, 0, genomeSize1);
        uint geneIndex2 = random_getUint(cuData, 0, genomeSize2);

        uint diff1 = (genomeSize1 - geneIndex1);
        uint diff2 = (genomeSize2 - geneIndex2);

        uint size11 = geneIndex1 * cuData->getGeneMemorySize();
        uint size12 = diff1 * cuData->getGeneMemorySize();
        uint size21 = geneIndex2 * cuData->getGeneMemorySize();
        uint size22 = diff2 * cuData->getGeneMemorySize();

        floatt * pgenome11 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex1);
        floatt * pgenome21 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex2);

        floatt * pgenome12 = genetic_getGene(pgenome11, cuData, geneIndex1);
        floatt * pgenome22 = genetic_getGene(pgenome21, cuData, geneIndex2);

        floatt * cgenome1 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex1);
        floatt * cgenome2 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex2);

        cu_memcpy(cgenome1, pgenome11, size11);
        cu_memcpy(cgenome1 + size11, pgenome12, size22);

        cu_memcpy(cgenome2, pgenome21, size21);
        cu_memcpy(cgenome2 + size21, pgenome11, size12);

        cuData->previousGeneration->genomesSizes[fa] = geneIndex1 + (diff2);
        cuData->previousGeneration->genomesSizes[fa + 1] = geneIndex2 + (diff1);
    }
}       

__device__
void executeTwoPointsCrossover(CuData* cuData) {
    for (uint fa = 0; fa < cuData->selectedGenomesCount; fa += 2) {
        uint genomeIndex1 = cuData->selectedGenomes[fa];
        uint genomeIndex2 = cuData->selectedGenomes[fa + 1];

        uint genomeSize1 = cuData->previousGeneration->genomesSizes[genomeIndex1];
        uint genomeSize2 = cuData->previousGeneration->genomesSizes[genomeIndex2];

        uint geneIndex1a = random_getUint(cuData, 0, genomeSize1);
        uint geneIndex1b = random_getUint(cuData, 0, genomeSize1);
        uint geneIndex2a = random_getUint(cuData, 0, genomeSize2);
        uint geneIndex2b = random_getUint(cuData, 0, genomeSize2);

        if (geneIndex1a > geneIndex1b) {
            uint temp = geneIndex1a;
            geneIndex1a = geneIndex1b;
            geneIndex1b = temp;
        }

        if (geneIndex2a > geneIndex2b) {
            uint temp = geneIndex2a;
            geneIndex2a = geneIndex2b;
            geneIndex2b = temp;
        }

        uint diff1b1a = geneIndex1b - geneIndex1a;
        uint diff11b = genomeSize1 - geneIndex1b;
        uint size11 = geneIndex1a * cuData->getGeneMemorySize();
        uint size12 = (diff1b1a) * cuData->getGeneMemorySize();
        uint size13 = (diff11b) * cuData->getGeneMemorySize();

        uint diff2b2a = geneIndex2b - geneIndex2a;
        uint diff22b = genomeSize2 - geneIndex2b;
        uint size21 = geneIndex2a * cuData->getGeneMemorySize();
        uint size22 = (diff2b2a) * cuData->getGeneMemorySize();
        uint size23 = (diff22b) * cuData->getGeneMemorySize();

        floatt * pgenome1 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex1);
        floatt * pgenome2 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex2);

        floatt * pgene1a = genetic_getGene(pgenome1, cuData, geneIndex1a);
        floatt * pgene1b = genetic_getGene(pgenome1, cuData, geneIndex1b);

        floatt * pgene2a = genetic_getGene(pgenome2, cuData, geneIndex2a);
        floatt * pgene2b = genetic_getGene(pgenome2, cuData, geneIndex2b);

        floatt * cgenome1 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex1);
        floatt * cgenome2 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex2);

        cu_memcpy(cgenome1, pgenome1, size11);
        cu_memcpy(cgenome1 + size11, pgene2a, size22);
        cu_memcpy(cgenome1 + (size11 + size22), pgene1b, size13);

        cu_memcpy(cgenome2, pgenome2, size21);
        cu_memcpy(cgenome2 + size21, pgene1a, size12);
        cu_memcpy(cgenome2 + (size21 + size12), pgene2b, size23);

        cuData->previousGeneration->genomesSizes[fa] = geneIndex1a + (diff2b2a) + diff11b;
        cuData->previousGeneration->genomesSizes[fa + 1] = geneIndex2a + (diff1b1a) + diff22b;
    }
}

__device__
void executeUniformCrossover(CuData* cuData) {
    for (uint fa = 0; fa < cuData->selectedGenomesCount; fa += 2) {
        uint genomeIndex1 = cuData->selectedGenomes[fa];
        uint genomeIndex2 = cuData->selectedGenomes[fa + 1];

        floatt * cgenome1 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex1);
        floatt * cgenome2 = genetic_getGenome(cuData->currentGeneration, cuData, genomeIndex2);

        floatt * pgenome1 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex1);
        floatt * pgenome2 = genetic_getGenome(cuData->previousGeneration, cuData, genomeIndex2);

        uint genomeSize1 = cuData->previousGeneration->genomesSizes[genomeIndex1];
        cuData->currentGeneration->genomesSizes[fa] = genomeSize1;
        uint genomeSize2 = cuData->previousGeneration->genomesSizes[genomeIndex2];
        cuData->currentGeneration->genomesSizes[fa + 1] = genomeSize2;

        for (uint fb = 0; fb < genomeSize1 && fb < genomeSize2; fb++) {
            floatt  randomFactor = random_get(cuData, 0, 1);
            if (randomFactor < 0.5f) {
                cu_memcpy(cgenome1 + (fb * cuData->getGeneMemorySize()), pgenome1 + (fb * cuData->getGeneMemorySize()), cuData->getGeneMemorySize());
                cu_memcpy(cgenome2 + (fb * cuData->getGeneMemorySize()), pgenome2 + (fb * cuData->getGeneMemorySize()), cuData->getGeneMemorySize());
            } else {
                cu_memcpy(cgenome2 + (fb * cuData->getGeneMemorySize()), pgenome1 + (fb * cuData->getGeneMemorySize()), cuData->getGeneMemorySize());
                cu_memcpy(cgenome1 + (fb * cuData->getGeneMemorySize()), pgenome2 + (fb * cuData->getGeneMemorySize()), cuData->getGeneMemorySize());
            }
        }
    }
}

#endif

