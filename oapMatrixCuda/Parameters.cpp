/*
 * Copyright 2016 - 2021 Marcin Matula
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
