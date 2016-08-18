
#ifndef OAP_REAL_TRANSFER_MATRIX_CPU_H
#define	OAP_REAL_TRANSFER_MATRIX_CPU_H

#include "TransferMatrixCpu.h"

namespace shibataCpu {

class RealTransferMatrix : public shibataCpu::cpu::TransferMatrix {
public:
    RealTransferMatrix();
    virtual ~RealTransferMatrix();
    void setThreadsCount(int threadsCount);
protected:
    HostTreePointerCreator m_treePointerCreator;
    intt getThreadsCount();
    bool mustBeDestroyed[4];
    int getSpinsChainsCount();
    int getSpinsCount();
    int getVirtualTime();
    math::Status onExecute();
    void onThreadCreation(uintt index,
        shibataCpu::cpu::Data* data);
private:

    class TMData {
        void dealloc();
    public:
        TreePointerCreator* m_treePointerCreator;
        TMData(TreePointerCreator* treePointerCreator);
        virtual ~TMData();
        void alloc(uintt m, uintt N, uintt quantumsCount,
            math::Matrix* expHamiltonian1, math::Matrix* expHamiltonian2);
        uintt M;
        intt* pows;
        intt* tmColumns;
        intt* tmRows;
        char* tmRowsBits;
        uintt upIndeciesCount;
        uintt downIndeciesCount;
        uintt* upIndecies;
        uintt* downIndecies;
        TreePointer** treePointers;
    };

    std::vector<TMData*> tmDatas;
    floatt* m_outputsEntries;
    uintt* m_entries;
    uintt m_size;
    HostMatrixUtils funcs;
    HostMatrixPrinter hmp;
    inline int getValueIndex(int index1, int columns, int index2);
    char* tempSpins;
    HostMatrixAllocator hmm;
    static void ExecuteTM(shibataCpu::cpu::Data* dataPtr, void* ptr);
    RealTransferMatrix(const RealTransferMatrix& orig);
};
}

#endif	/* VIRTUALTRANSFERMATRIX_H */

