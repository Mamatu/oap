/* 
 * File:   GATypes.cpp
 * Author: mmatula
 * 
 * Created on July 3, 2013, 8:06 PM
 */

#include "GATypes.h"
namespace ga {

    GAData::Generation::Generation() : genomes(NULL), genesSizes(NULL), genomesSizes(NULL) {
    }

    GAData::Generation::~Generation() {
    }

    GAData::GAData() :
    selectedGenomes(NULL), realGeneSize(0),
    realGenomeSize(0), realUnitSize(GA_UNIT_TYPE_32_BITS), generationCounter(0), genomesCount(0) {
        this->currentGeneration.gaData = this;
        this->previousGeneration.gaData = this;
    }

    GAData::~GAData() {
    }
}