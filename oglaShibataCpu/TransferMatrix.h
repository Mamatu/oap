/* 
 * File:   TransferMatrix.h
 * Author: mmatula
 *
 * Created on January 14, 2014, 8:09 PM
 */

#ifndef OGLA_TRANSFER_MATRIX_H
#define	OGLA_TRANSFER_MATRIX_H


#include "Matrix.h"
#include "MathOperationsCpu.h"
#include "Parameters.h"
#include "ThreadUtils.h"
#include "TreePointerCreator.h"

typedef std::vector<std::pair<int, int> > IntegersPairs;


namespace shibata {

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
        shibata::Parameters parameters;
    private:
        TransferMatrix(const TransferMatrix& orig);
    };

    class TransferMatrixObject : public utils::OglaObject {
    public:
        TransferMatrixObject(const char* name, TransferMatrix* transferMatrixPtr);
        virtual ~TransferMatrixObject();
    };
}
#endif	/* TRANSFERMATRIX_H */

