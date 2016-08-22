/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
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
