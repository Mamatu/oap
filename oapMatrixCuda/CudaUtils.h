/*
 * File:   DeviceUtils.h
 * Author: mmatula
 *
 * Created on November 22, 2014, 7:58 PM
 */

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <stdio.h>
#include <cuda.h>
#include <string>
#include "Matrix.h"
#include "MatrixEx.h"

namespace CudaUtils {

/**
 *
 * @param v
 * @return
 */
template <typename T>
T* AllocDeviceObj(const T& v = 0);

/**
 *
 * @param valuePtr
 */
template <typename T>
void FreeDeviceObj(T* valuePtr);

/**
 *
 * @param size
 * @return
 */
void* AllocDeviceMem(uintt size);

/**
 *
 * @param size
 * @param src
 * @return
 */
void* AllocDeviceMem(uintt size, const void* src);

/**
 *
 * @param devicePtr
 */
void FreeDeviceMem(void* devicePtr);

void FreeDeviceMem(CUdeviceptr ptr);

/**
 *
 * @param dst
 * @param src
 * @param size
 */
void CopyHostToDevice(void* dst, const void* src, uintt size);

/**
 *
 * @param dst
 * @param src
 * @param size
 */
void CopyDeviceToHost(void* dst, const void* src, uintt size);

/**
 *
 * @param dst
 * @param src
 * @param size
 */
void CopyDeviceToDevice(void* dst, const void* src, uintt size);

/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetReValuesAddress(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetImValuesAddress(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetColumnsAddress(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetRowsAddress(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
floatt* GetReValues(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
floatt* GetImValues(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
uintt GetColumns(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
uintt GetRows(const math::Matrix* matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetReValuesAddress(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetImValuesAddress(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetColumnsAddress(CUdeviceptr matrix);
CUdeviceptr GetRealColumnsAddress(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
CUdeviceptr GetRowsAddress(CUdeviceptr matrix);
CUdeviceptr GetRealRowsAddress(CUdeviceptr matrix);

CUdeviceptr GetBColumnAddress(const MatrixEx* matrixEx);
CUdeviceptr GetEColumnAddress(const MatrixEx* matrixEx);

CUdeviceptr GetBRowAddress(const MatrixEx* matrixEx);
CUdeviceptr GetERowAddress(const MatrixEx* matrixEx);

/**
 *
 * @param matrix
 * @return
 */
floatt* GetReValues(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
floatt* GetImValues(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
uintt GetColumns(CUdeviceptr matrix);
/**
 *
 * @param matrix
 * @return
 */
uintt GetRows(CUdeviceptr matrix);

uintt GetColumns(const MatrixEx* matrix);
uintt GetRows(const MatrixEx* matrix);

/**
 *
 * @param allocRe
 * @param allocIm
 * @param columns
 * @param rows
 * @param revalue
 * @param imvalue
 * @return
 */
CUdeviceptr AllocMatrix(bool allocRe, bool allocIm, uintt columns, uintt rows,
                        floatt revalue = 0, floatt imvalue = 0);
/**
 *
 * @param devicePtrMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
CUdeviceptr AllocReMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value);
/**
 *
 * @param devicePtrMatrix
 * @param columns
 * @param rows
 * @param value
 * @return
 */
CUdeviceptr AllocImMatrix(CUdeviceptr devicePtrMatrix, uintt columns,
                          uintt rows, floatt value);
/**
 *
 * @param devicePtrMatrix
 * @return
 */
CUdeviceptr SetReMatrixToNull(CUdeviceptr devicePtrMatrix);
/**
 *
 * @param devicePtrMatrix
 * @return
 */
CUdeviceptr SetImMatrixToNull(CUdeviceptr devicePtrMatrix);
/**
 *
 * @param devicePtrMatrix
 * @param columns
 * @param rows
 */
void SetVariables(CUdeviceptr devicePtrMatrix, uintt columns, uintt rows);

void SetReValue(math::Matrix* m, uintt index, floatt value);
floatt GetReValue(math::Matrix* m, uintt index);

void SetImValue(math::Matrix* m, uintt index, floatt value);
floatt GetImValue(math::Matrix* m, uintt index);

floatt GetReDiagonal(math::Matrix* m, uintt index);
floatt GetImDiagonal(math::Matrix* m, uintt index);

void SetZeroMatrix(math::Matrix* matrix, bool re = true, bool im = true);
void SetZeroRow(math::Matrix* matrix, uintt index, bool re = true,
                bool im = true);

void GetMatrixStr(std::string& output, const math::Matrix* matrix,
                  floatt zeroLimit = 0, bool repeats = true, bool pipe = true,
                  bool endl = true);
void PrintMatrix(FILE* stream, const math::Matrix* matrix, floatt zeroLimit = 0,
                 bool repeats = true, bool pipe = true, bool endl = true);
void PrintMatrix(const math::Matrix* matrix, floatt zeroLimit = 0,
                 bool repeats = true, bool pipe = true, bool endl = true);
void PrintMatrix(const std::string& output, const math::Matrix* matrix,
                 floatt zeroLimit = 0, bool repeats = true, bool pipe = true,
                 bool endl = true);
}

template <typename T>
T* CudaUtils::AllocDeviceObj(const T& v) {
  T* valuePtr = NULL;
  void* ptr = CudaUtils::AllocDeviceMem(sizeof(T));
  valuePtr = reinterpret_cast<T*>(ptr);
  CudaUtils::CopyHostToDevice(valuePtr, &v, sizeof(T));
  return valuePtr;
}

template <typename T>
void CudaUtils::FreeDeviceObj(T* valuePtr) {
  CudaUtils::FreeDeviceMem(valuePtr);
}

#endif /* DEVICEUTILS_H */
