/* 
 * File:   Parameters.cpp
 * Author: mmatula
 * 
 * Created on September 27, 2013, 9:58 PM
 */

#include "Parameters.h"
#include "Parameters.h"

Parameters::Parameters() : ranges(NULL), mutationType(0), mutationsCount(0), selectionType(0), crossoverType(0), threadsCount(0) {
}

Parameters::~Parameters() {
    if (ranges) {
        delete[] ranges;
    }
}

#define SET(var)  mutex.lock(); this->var = var; mutex.unlock();
#define GET(type, var) mutex.lock(); type var = this->var; mutex.unlock(); return var;

void Parameters::setThreadsCount(uint threadsCount) {
    SET(threadsCount);
}

void Parameters::setSeed(long long int seed) {
    SET(seed);
}

void Parameters::setMutationType(uint mutationType) {
    SET(mutationType);
}

void Parameters::setSelectionType(uint selectionType) {
    SET(selectionType);
}

void Parameters::setCrossoverType(uint crossoverType) {
    SET(crossoverType);
}

uint Parameters::getThreadsCount() {
    GET(uint, threadsCount);
}

long long int Parameters::getSeed() {
    GET(long long int, seed);
}

uint Parameters::getMutationType() {
    GET(uint, mutationType);
}

uint Parameters::getSelectionType() {
    GET(uint, selectionType);
}

uint Parameters::getCrossoverType() {
    GET(uint, crossoverType);
}

void Parameters::setRandomsFactor(uint randomsFactor) {
    SET(randomsFactor);
}

uint Parameters::getRandomsFactor() {
    GET(uint, randomsFactor);
}

void Parameters::setMutationsCount(uint count) {
    this->mutationsCount = count;
}

void Parameters::setRangesCount(uint count) {
    if (ranges && this->rangesCount != count) {
        delete[] ranges;
        ranges = new floatt [count * 2];
    }
    this->rangesCount = count;
}

void Parameters::setRandomsRange(floatt  min, floatt  max, uint index) {
    this->ranges[index * 2] = min;
    this->ranges[index * 2 + 1] = max;
}

uint Parameters::getMutationsCount() {
    GET(uint, mutationsCount);
}

uint Parameters::getRangesCount() {
    GET(uint, rangesCount);
}

void Parameters::getRandomsRange(floatt & min, floatt & max, uint index) {
    min = this->ranges[index * 2];
    max = this->ranges[index * 2 + 1];
}

void Parameters::copy(Parameters& parameters) {
    this->mutex.lock();
    parameters.mutex.lock();
    this->setCrossoverType(parameters.getCrossoverType());
    this->setMutationType(parameters.getMutationType());
    this->setSelectionType(parameters.getSelectionType());
    this->setMutationsCount(parameters.getMutationsCount());
    this->setRangesCount(parameters.getRangesCount());
    this->setSeed(parameters.getSeed());
    this->setThreadsCount(parameters.getThreadsCount());
    this->setRandomsFactor(parameters.getRandomsFactor());
    if (this->ranges) {
        delete[] this->ranges;
    }
    this->ranges = new floatt [this->rangesCount * 2];
    for (uint fa = 0; fa<this->rangesCount; fa++) {
        floatt  min, max;
        parameters.getRandomsRange(min, max, fa);
        this->setRandomsRange(min, max, fa);
    }
    this->mutex.unlock();
    parameters.mutex.unlock();
}
