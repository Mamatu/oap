/* 
 * File:   oglaCudaUtils.cpp
 * Author: mmatula
 * 
 * Created on March 3, 2015, 11:15 PM
 */

#include "oglaCudaStub.h"

#include "Matrix.h"
#include "MatrixEx.h"

OglaCudaStub::OglaCudaStub() {
}

OglaCudaStub::OglaCudaStub(const OglaCudaStub& orig) {
}

OglaCudaStub::~OglaCudaStub() {
}

void OglaCudaStub::executeKernelSync(OglaCudaStub::KernelStub* cudaStub) {
    for (uintt blockIdxX = 0; blockIdxX < cudaStub->gridDim.x; ++blockIdxX) {
        for (uintt blockIdxY = 0; blockIdxY < cudaStub->gridDim.y; ++blockIdxY) {
            for (uintt threadIdxX = 0; threadIdxX < cudaStub->blockDim.x; ++threadIdxX) {
                for (uintt threadIdxY = 0; threadIdxY < cudaStub->blockDim.y; ++threadIdxY) {
                    cudaStub->threadIdx.x = threadIdxX;
                    cudaStub->threadIdx.y = threadIdxY;
                    cudaStub->blockIdx.x = blockIdxX;
                    cudaStub->blockIdx.y = blockIdxY;
                    cudaStub->execute();
                }
            }
        }
    }
}

void OglaCudaStub::KernelStub::calculateDims(uintt columns, uintt rows) {
    uintt blocks[2];
    uintt threads[2];
    utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
    setDims(blocks, threads);
}

void OglaCudaStub::KernelStub::setDims(const Dim3& blockDim, const Dim3& gridDim) {
    this->blockDim = blockDim;
    this->gridDim = gridDim;
}
