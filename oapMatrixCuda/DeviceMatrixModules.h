#ifndef OAP_DEVICE_MATRIX_UTILS_H
#define OAP_DEVICE_MATRIX_UTILS_H
#include <stdio.h>
#include <cuda.h>
#include <map>
#include "HostMatrixModules.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "ThreadUtils.h"
#include "CudaUtils.h"

namespace device {

/**
 * @brief NewDeviceMatrix
 * @param columns
 * @param rows
 * @return
 */
math::Matrix* NewDeviceMatrix(uintt columns, uintt rows, floatt revalue = 0.f,
                              floatt imvalue = 0.f);

/**
 * @brief NewDeviceMatrix
 * @param hostMatrix
 * @return
 */
math::Matrix* NewDeviceMatrixHostRef(const math::Matrix* hostMatrix);

math::Matrix* NewDeviceMatrix(const math::Matrix* deviceMatrix);

math::Matrix* NewDeviceMatrix(const std::string& matrixStr);


uintt GetColumns(const math::Matrix* dMatrix);

uintt GetRows(const math::Matrix* dMatrix);

/**
 * @brief NewDeviceMatrixCopy
 * @param hostMatrix
 * @return
 */
math::Matrix* NewDeviceMatrixCopy(const math::Matrix* hostMatrix);

/**
 * @brief NewDeviceMatrix
 * @param hostMatrix
 * @param columns
 * @param rows
 * @return
 */
math::Matrix* NewDeviceMatrix(const math::Matrix* hostMatrix, uintt columns,
                              uintt rows);

/**
 * @brief NewDeviceMatrix
 * @param allocRe
 * @param allocIm
 * @param columns
 * @param rows
 * @return
 */
math::Matrix* NewDeviceMatrix(bool allocRe, bool allocIm, uintt columns,
                              uintt rows);

/**
 * @brief NewHostMatrixCopyOfDeviceMatrix
 * @param matrix
 * @return
 */
math::Matrix* NewHostMatrixCopyOfDeviceMatrix(const math::Matrix* matrix);

/**
 * @brief DeleteDeviceMatrix
 * @param deviceMatrix
 */
void DeleteDeviceMatrix(math::Matrix* deviceMatrix);

/**
 *
 * @param dst
 * @param src
 */
void CopyDeviceMatrixToHostMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 *
 * @param dst
 * @param src
 */
void CopyHostMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 * @brief CopyDeviceMatrixToDeviceMatrix
 * @param dst
 * @param src
 */
void CopyDeviceMatrixToDeviceMatrix(math::Matrix* dst, const math::Matrix* src);

/**
 *
 * @param dst
 * @param src
 */
void CopyHostArraysToDeviceMatrix(math::Matrix* dst, const floatt* rearray,
                                  const floatt* imarray);

/**
 * @brief NewDeviceMatrixEx
 * @param count
 * @return
 */
MatrixEx** NewDeviceMatrixEx(uintt count);

/**
 * @brief DeleteDeviceMatrixEx
 * @param matrixEx
 */
void DeleteDeviceMatrixEx(MatrixEx** matrixEx);

/**
 * @brief SetMatrixEx
 * @param deviceMatrixEx
 * @param buffer
 * @param count
 */
void SetMatrixEx(MatrixEx** deviceMatrixEx, const uintt* buffer, uintt count);

/**
 * @brief NewDeviceMatrixEx
 * @return
 */
MatrixEx* NewDeviceMatrixEx();

/**
 * @brief DeleteDeviceMatrixEx
 * @param matrixEx
 */
void DeleteDeviceMatrixEx(MatrixEx* matrixEx);

/**
 * @brief SetMatrixEx
 * @param deviceMatrixEx
 * @param hostMatrixEx
 */
void SetMatrixEx(MatrixEx* deviceMatrixEx, const MatrixEx* hostMatrixEx);

/**
 * @brief PrintMatrix
 * @param text
 * @param matrix
 */
void PrintMatrix(const std::string& text, const math::Matrix* matrix, floatt zeroLimit = 0);

/**
 * @brief PrintMatrix
 * @param matrix
 */
void PrintMatrix(const math::Matrix* matrix);
}

#endif /* MATRIXMEM_H */
