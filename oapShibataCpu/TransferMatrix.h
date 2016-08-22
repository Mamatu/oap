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




#ifndef OAP_TRANSFER_MATRIX_H
#define	OAP_TRANSFER_MATRIX_H


#include "Matrix.h"
#include "MathOperationsCpu.h"
#include "Parameters.h"
#include "ThreadUtils.h"
#include "TreePointerCreator.h"

typedef std::vector<std::pair<int, int> > IntegersPairs;


namespace shibataCpu {

    enum Orientation {
        ORIENTATION_REAL_DIRECTION,
        ORIENTATION_TROTTER_DIRECTION
    };

    class TransferMatrix {
    public:
        static void PrepareHamiltonian(math::Matrix* dst, math::Matrix* src,
                Orientation hamiltonianOrientation);
        static void PrepareExpHamiltonian(math::Matrix* matrix,
                math::Matrix* src, uintt M);

        /**
         * 
         * @param matrix
         * @param hamiltonianOrientation - defines orientation of matrix
         */
        void setExpHamiltonian1(math::Matrix* matrix);
        /**
         * 
         * @param matrix
         * @param hamiltonianOrientation - defines orientation of matrix
         */
        void setExpHamiltonian2(math::Matrix* matrix);
        void setQuantums(int* qunatums);
        void setSpinsCount(int spinsCount);
        void setSerieLimit(int serieLimit);
        void setQuantumsCount(int quantumsCount);
        void setTrotterNumber(int trotterNumber);

        /**
         * Set pointer on transfer matrix.
         * @param output
         */
        void setOutputMatrix(math::Matrix* output);

        void setReOutputEntries(floatt* outputEntries);
        void setImOutputEntries(floatt* outputEntries);
        void setEntries(uintt* entries, uintt count);

        TransferMatrix();
        virtual ~TransferMatrix();
        math::Status start();
    protected:
        virtual math::Status execute() = 0;
        virtual int getSpinsCount() = 0;
        virtual int getVirtualTime() = 0;
        virtual math::Status onExecute() = 0;
        math::Matrix* transferMatrix;
        math::Matrix* expHamiltonian1;
        math::Matrix* expHamiltonian2;
        floatt* m_reoutputEntries;
        floatt* m_imoutputEntries;
        uintt* m_entries;
        uintt m_entriesCount;
        shibataCpu::Parameters parameters;
    private:
        TransferMatrix(const TransferMatrix& orig);
    };

    class TransferMatrixObject : public utils::OapObject {
    public:
        TransferMatrixObject(const char* name, TransferMatrix* transferMatrixPtr);
        virtual ~TransferMatrixObject();
    };
}
#endif	/* TRANSFERMATRIX_H */
