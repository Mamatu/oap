/* 
 * File:   DeviceUtils.h
 * Author: mmatula
 *
 * Created on November 22, 2014, 7:58 PM
 */

#ifndef CUDAUTILS_H
#define	CUDAUTILS_H

namespace CudaUtils {

    /**
     * 
     * @param v
     * @return 
     */
    template<typename T>T* NewDeviceValue(T v = 0);

    /**
     * 
     * @param valuePtr
     */
    template<typename T>void DeleteDeviceValue(T* valuePtr);

    /**
     * 
     * @param size
     * @return 
     */
    void* NewDevice(intt size);

    /**
     * 
     * @param size
     * @param src
     * @return 
     */
    void* NewDevice(intt size, const void* src);

    /**
     * 
     * @param devicePtr
     */
    void DeleteDevice(void* devicePtr);

    void DeleteDevice(CUdeviceptr ptr); 
    
    /**
     * 
     * @param dst
     * @param src
     * @param size
     */
    void CopyHostToDevice(void* dst, const void* src, intt size);

    /**
     * 
     * @param dst
     * @param src
     * @param size
     */
    void CopyDeviceToHost(void* dst, const void* src, intt size);

    /**
     * 
     * @param dst
     * @param src
     * @param size
     */
    void CopyDeviceToDevice(void* dst, const void* src, intt size);

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
    uintt GetDeviceColumns(const math::Matrix* matrix);
    /**
     * 
     * @param matrix
     * @return 
     */
    uintt GetDeviceRows(const math::Matrix* matrix);
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
    /**
     * 
     * @param matrix
     * @return 
     */
    CUdeviceptr GetRowsAddress(CUdeviceptr matrix);
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
    intt GetDeviceColumns(CUdeviceptr matrix);
    /**
     * 
     * @param matrix
     * @return 
     */
    intt GetDeviceRows(CUdeviceptr matrix);
    /**
     * 
     * @return 
     */
    CUdeviceptr AllocMatrix();
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
    CUdeviceptr AllocMatrix(bool allocRe, bool allocIm, intt columns,
            intt rows, floatt revalue = 0, floatt imvalue = 0);
    /**
     * 
     * @param devicePtrMatrix
     * @param columns
     * @param rows
     * @param value
     * @return 
     */
    CUdeviceptr AllocReMatrix(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows, floatt value);
    /**
     * 
     * @param devicePtrMatrix
     * @param columns
     * @param rows
     * @param value
     * @return 
     */
    CUdeviceptr AllocImMatrix(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows, floatt value);
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
    void SetVariables(CUdeviceptr devicePtrMatrix,
            intt columns, intt rows);
}

template<typename T>T* CudaUtils::NewDeviceValue(T v) {
    T* valuePtr = NULL;
    void* ptr = CudaUtils::NewDevice(sizeof (T));
    valuePtr = reinterpret_cast<T*> (ptr);
    CudaUtils::CopyHostToDevice(valuePtr, &v, sizeof (T));
    return valuePtr;
}

template<typename T>void CudaUtils::DeleteDeviceValue(T* valuePtr) {
    CudaUtils::DeleteDevice(valuePtr);
}

#endif	/* DEVICEUTILS_H */

