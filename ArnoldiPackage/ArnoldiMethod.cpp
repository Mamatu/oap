/* 
 * File:   ArnoldiMethod.cpp
 * Author: mmatula
 * 
 * Created on August 5, 2014, 10:15 PM
 */

#include "CuArnoldiMethod.h"

Status AR_Create(Handle** handle, utils::Callback_f callback = NULL) {
}

Status AR_SetDeviceMatrix(Handle* handle, math::Matrix* matrix) {
}

Status AR_SetDeviceEigenvaluesBuffer(Handle* handle,
        floatt* outputs, uint size) {
}

Status AR_SetDeviceEigenvactorsBuffer(Handle* handle,
        floatt* outputs, uint size) {
}

Status AR_SetHostEigenvaluesBuffer(Handle* handle,
        math::Matrix* outputs, uint size) {
}

Status AR_SetHostEigenvactorsBuffer(Handle* handle,
        math::Matrix* outputs, uint size) {
}

Status AR_Start(Handle* handle) {
}

Status AR_Stop(Handle* handle) {
}

Status AR_GetState(Handle* handle, const Buffer* buffer) {
}

Status AR_SetState(Handle* handle, const Buffer* buffer) {
}

Status AR_Destroy(Handle* handle) {
}
