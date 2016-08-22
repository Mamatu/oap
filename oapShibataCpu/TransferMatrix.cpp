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




#include "TransferMatrix.h"

namespace shibataCpu {

    void TransferMatrix::PrepareHamiltonian(math::Matrix* dst, math::Matrix* src,
            Orientation orientation) {
        if (orientation == ORIENTATION_REAL_DIRECTION) {
            char* tempSpins = NULL;
            int spinsCount = src->rows;
            int qc = 2; //this->parameters.getQunatumsCount();
            tempSpins = ArrayTools::create<char>(spinsCount, 0);
            ArrayTools::clear<char>(tempSpins, spinsCount, 0);
            do {
                floatt rvalue = host::GetReValue(src, tempSpins[0] + qc * tempSpins[1], tempSpins[2] + qc * tempSpins[3]);
                floatt ivalue = host::GetImValue(src, tempSpins[0] + qc * tempSpins[1], tempSpins[2] + qc * tempSpins[3]);
                host::SetReValue(dst, tempSpins[2] + qc * tempSpins[0], tempSpins[3] + qc * tempSpins[1], rvalue);
                host::SetImValue(dst, tempSpins[2] + qc * tempSpins[0], tempSpins[3] + qc * tempSpins[1], ivalue);
            } while (ArrayTools::increment<char>(tempSpins, 0, qc, 1, 0, spinsCount) < spinsCount);
            delete[] tempSpins;
        }
    }

    void TransferMatrix::PrepareExpHamiltonian(math::Matrix* matrix,
            math::Matrix* src, uintt M) {
    }

    TransferMatrix::TransferMatrix() : transferMatrix(NULL),
    expHamiltonian1(NULL),
    expHamiltonian2(NULL) {
        m_reoutputEntries = NULL;
        m_imoutputEntries = NULL;
    }

    TransferMatrix::TransferMatrix(const TransferMatrix& orig) {
    }

    TransferMatrix::~TransferMatrix() {
    }

    math::Status TransferMatrix::start() {
        return execute();
    }

    void TransferMatrix::setExpHamiltonian1(math::Matrix* matrix) {
        this->expHamiltonian1 = matrix;
    }

    void TransferMatrix::setExpHamiltonian2(math::Matrix* matrix) {
        this->expHamiltonian2 = matrix;
    }

    void TransferMatrix::setOutputMatrix(math::Matrix* output) {
        this->transferMatrix = output;
    }

    void TransferMatrix::setReOutputEntries(floatt* outputEntries) {
        m_reoutputEntries = outputEntries;
    }

    void TransferMatrix::setImOutputEntries(floatt* outputEntries) {
        m_imoutputEntries = outputEntries;
    }

    void TransferMatrix::setEntries(uintt* entries, uintt count) {
        m_entries = entries;
        m_entriesCount = count;
    }

    void TransferMatrix::setSpinsCount(int spinsCount) {
        this->parameters.setSpinsCount(spinsCount);
    }

    void TransferMatrix::setSerieLimit(int serieLimit) {
        this->parameters.setSerieLimit(serieLimit);
    }

    void TransferMatrix::setQuantumsCount(int quantumsCount) {
        this->parameters.setQuantumsCount(quantumsCount);
    }

    void TransferMatrix::setTrotterNumber(int trotterNumber) {
        this->parameters.setTrotterNumber(trotterNumber);
    }

    TransferMatrixObject::TransferMatrixObject(const char* name, TransferMatrix* transferMatrixPtr) : utils::OapObject(name) {
    }

    TransferMatrixObject::~TransferMatrixObject() {
    }
}
