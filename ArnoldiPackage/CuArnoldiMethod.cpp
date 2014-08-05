#include "CuArnoldiMethod.h"
#include "CuArnoldiMethodImpl.h"

class Parameters {
public:

    Parameters() :
    m_arnodliMethod(NULL),
    m_arnodliMethodCallback(NULL),
    m_callback(NULL) {
    }
    cpu::ArnoldiMethod* m_arnodliMethod;
    cpu::ArnoldiMethodCallback* m_arnodliMethodCallback;
    utils::Callback_f m_callback;
    math::Matrix* m_deviceMatrix;
};

Buffer::Buffer() : data(NULL), size(0) {
}

Status CUAR_Create(Handle** handle, utils::Callback_f callback) {
}

Status CUAR_SetDeviceMatrix(Handle* handle, math::Matrix* matrix) {
}

Status CUAR_SetDeviceEigenvaluesBuffer(Handle* handle,
        floatt* outputs, uint size) {
}

Status CUAR_SetDeviceEigenvactorsBuffer(Handle* handle,
        floatt* outputs, uint size) {
}

Status CUAR_SetHostEigenvaluesBuffer(Handle* handle,
        math::Matrix* outputs, uint size) {
}

Status CUAR_SetHostEigenvactorsBuffer(Handle* handle,
        math::Matrix* outputs, uint size) {
}

Status CUAR_Start(Handle* handle) {
}

Status CUAR_Stop(Handle* handle) {
}

Status CUAR_GetState(Handle* handle, const Buffer* buffer) {
}

Status CUAR_SetState(Handle* handle, const Buffer* buffer) {
}

Status CUAR_Destroy(Handle* handle) {
}
