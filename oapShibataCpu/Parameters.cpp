/* 
 * File:   Parameters.cpp
 * Author: mmatula
 * 
 * Created on September 30, 2013, 9:28 PM
 */

#include "Parameters.h"

namespace shibataCpu {

    Parameters::Parameters() {
        this->serieLimit.set(100);
        this->threadsCount.set(1);
        this->trotterNumber.set(1);
        this->spinsCount.value = 0;
    }

    Parameters::~Parameters() {
    }

    Parameters::Parameters(const Parameters& orig) {
        this->quantumsCount = orig.quantumsCount;
        this->serieLimit = orig.serieLimit;
        this->spinsCount = orig.spinsCount;
        this->threadsCount = orig.threadsCount;
        this->trotterNumber = orig.trotterNumber;
    }

    bool Parameters::isThreadsCount() const {
        return this->threadsCount.isAvailable;
    }

    void Parameters::setThreadsCount(int count) {
        this->threadsCount.set(count);
    }

    int Parameters::getThreadsCount()const {
        return this->threadsCount.value;
    }

    bool Parameters::isSerieLimit() const {
        return this->serieLimit.isAvailable;
    }

    void Parameters::setSerieLimit(int limit) {
        this->serieLimit.set(limit);
    }

    int Parameters::getSerieLimit()const {
        return this->serieLimit.value;
    }

    bool Parameters::isQuantumsCount() const {
        return this->quantumsCount.isAvailable;
    }

    void Parameters::setQuantumsCount(int qunatumsCount) {
        this->quantumsCount.set(qunatumsCount);
    }

    int Parameters::getQunatumsCount() const {
        return this->quantumsCount.value;
    }

    bool Parameters::isSpinsCount() const {
        return this->spinsCount.isAvailable;
    }

    void Parameters::setSpinsCount(int spinsCount) {
        this->spinsCount.set(spinsCount);
    }

    int Parameters::getSpinsCount() const {
        return this->spinsCount.value;
    }

    bool Parameters::isTrotterNumber() const {
        return this->trotterNumber.isAvailable;
    }

    void Parameters::setTrotterNumber(int trotterNumber) {
        this->trotterNumber.set(trotterNumber);
    }

    int Parameters::getTrotterNumber() const {
        return this->trotterNumber.value;
    }

    bool Parameters::areAllAvailable() const {
        return this->quantumsCount.isAvailable &&
                this->serieLimit.isAvailable &&
                this->spinsCount.isAvailable &&
                this->threadsCount.isAvailable;
    }
}
