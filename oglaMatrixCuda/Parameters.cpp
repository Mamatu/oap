/* 
 * File:   Parameters.cpp
 * Author: mmatula
 * 
 * Created on September 30, 2013, 9:28 PM
 */

#include "Parameters.h"

Parameters::Parameters(bool isThreadsCount, bool isSerieLimit) {
    this->serieLimit.isAvailable = isSerieLimit;
    this->serieLimit.value = 10;
    this->threadsCount.isAvailable = isThreadsCount;
    this->threadsCount.value = 1;
}

Parameters::Parameters(const Parameters& orig) {
}

Parameters::~Parameters() {
}

bool Parameters::isParamThreadsCount() const {
    return this->threadsCount.isAvailable;
}

void Parameters::setThreadsCount(uintt count) {
    this->threadsCount.value = count;
}

uintt Parameters::getThreadsCount()const {
    return this->threadsCount.value;
}

bool Parameters::isParamSerieLimit() const {
    return this->serieLimit.isAvailable;
}

void Parameters::setSerieLimit(uintt limit) {
    this->serieLimit.value = limit;
}

uintt Parameters::getSerieLimit()const {
    return this->serieLimit.value;
}

