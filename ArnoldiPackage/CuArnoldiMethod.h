#ifndef CU_ARNOLDIMETHOD_H
#define	CU_ARNOLDIMETHOD_H

#include "Types.h"

/**
 * 
 */
Status CUAR_Create(Handle** handle, utils::Callback_f callback = NULL);

/**
 * 
 * @param matrix
 */
Status CUAR_SetDeviceMatrix(Handle* handle, math::Matrix* matrix);

/**
 * 
 * @param matrix
 */
Status CUAR_SetDeviceEigenvaluesBuffer(Handle* handle,
        floatt* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status CUAR_SetDeviceEigenvactorsBuffer(Handle* handle,
        floatt* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status CUAR_SetHostEigenvaluesBuffer(Handle* handle,
        math::Matrix* outputs, uint size);

/**
 * 
 * @param handle
 * @param outputs
 * @param size
 * @return 
 */
Status CUAR_SetHostEigenvactorsBuffer(Handle* handle,
        math::Matrix* outputs, uint size);

/**
 * 
 * @param handle
 * @return 
 */
Status CUAR_Start(Handle* handle);

/**
 * 
 * @param handle
 * @return 
 */
Status CUAR_Stop(Handle* handle);

/**
 * 
 * @param handle
 * @param buffer
 * @return 
 */
Status CUAR_GetState(Handle* handle, const Buffer* buffer);

/**
 * 
 * @param handle
 * @param buffer
 * @return 
 */
Status CUAR_SetState(Handle* handle, const Buffer* buffer);

/**
 * 
 */
Status CUAR_Destroy(Handle* handle);



#endif	/* ARNOLDIMETHOD_H */

