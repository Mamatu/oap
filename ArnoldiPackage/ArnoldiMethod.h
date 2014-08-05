#ifndef ARNOLDIMETHOD_H
#define	ARNOLDIMETHOD_H

#include "Types.h"

/**
 * 
 */
Status AR_Create(Handle** handle, utils::Callback_f callback = NULL);

/**
 * 
 * @param matrix
 */
Status AR_SetDeviceMatrix(Handle* handle, math::Matrix* matrix);

/**
 * 
 * @param matrix
 */
Status AR_SetDeviceEigenvaluesBuffer(Handle* handle,
        floatt* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status AR_SetDeviceEigenvactorsBuffer(Handle* handle,
        floatt* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status AR_SetHostEigenvaluesBuffer(Handle* handle,
        math::Matrix* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status AR_SetHostEigenvactorsBuffer(Handle* handle,
        math::Matrix* outputs, uint size);

/**
 * 
 * @param handle
 * @return 
 */
Status AR_Start(Handle* handle);

/**
 * 
 * @param handle
 * @return 
 */
Status AR_Stop(Handle* handle);

/**
 * 
 * @param handle
 * @param buffer
 * @return 
 */
Status AR_GetState(Handle* handle, const Buffer* buffer);

/**
 * 
 * @param handle
 * @param buffer
 * @return 
 */
Status AR_SetState(Handle* handle, const Buffer* buffer);

/**
 * 
 */
Status AR_Destroy(Handle* handle);

#endif	/* ARNOLDIMETHOD_H */

