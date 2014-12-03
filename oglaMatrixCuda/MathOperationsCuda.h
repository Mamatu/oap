/* 
 * File:   MatrixOperationsCPU.h
 * Author: mmatula
 *
 * Created on September 24, 2013, 9:33 PM
 */

#ifndef OGLA_MATRIXOPERATIONSCUDA_H
#define	OGLA_MATRIXOPERATIONSCUDA_H
#include "Parameters.h"
#include "Math.h"
#include "ThreadUtils.h"
#include "KernelsOperations.h"
#include "MathOperations.h"
namespace math {

    class KernelType : public ::cuda::DeviceInfo {
    protected:
        ::cuda::KernelMatrix m_kernel;
    public:
        void setDevice(CUdevice cuDecive);
        void setDeviceInfo(const ::cuda::DeviceInfo& deviceInfo);
        CUdevice getDevice() const;
    };

    class AdditionOperationCuda : public math::IAdditionOperation,
    public KernelType {
        KernelsOperations m_kernelOperation;
    public:
        math::Status beforeExecution();
        void execute();
        AdditionOperationCuda();
        ~AdditionOperationCuda();
    };

    class SubstracionOperationCuda : public math::ISubstracionOperation,
    public KernelType {
        KernelsOperations m_kernelOperations;
    public:
        math::Status beforeExecution();
        void execute();
        SubstracionOperationCuda();
        ~SubstracionOperationCuda();
    };

    class DotProductOperationCuda : public math::IDotProductOperation,
    public KernelType {
        KernelsOperations m_kernelOperation;
    public:
        math::Status beforeExecution();
        void execute();
        DotProductOperationCuda();
        ~DotProductOperationCuda();
    };

    class MultiplicationConstOperationCuda : public math::IMultiplicationConstOperation,
    public KernelType {
        KernelsOperations m_kernelOperation;
    public:
        math::Status beforeExecution();
        void execute();
        MultiplicationConstOperationCuda();
        ~MultiplicationConstOperationCuda();
    };

    class ExpOperationCuda : public math::IExpOperation, public KernelType {
        KernelsOperations m_kernelOperation;
        int serieLimit;
    protected:
        void setParamIn(void* inPtr, int index);
        DotProductOperationCuda multiplicationOperation;
        MultiplicationConstOperationCuda multiplicationConstOperation;
        AdditionOperationCuda additionOperation;
    public:
        void setSerieLimit(int serieLimit);
        void execute();
        ExpOperationCuda();
        ~ExpOperationCuda();
    };

    class DiagonalizationOperationCuda : public math::IDiagonalizationOperation,
    public KernelType {
        KernelsOperations m_kernelOperation;
    public:
        void execute();
        DiagonalizationOperationCuda();
        ~DiagonalizationOperationCuda();
    };

    class MathOperationsCuda : public utils::Module, public ::cuda::DeviceInfo {
        AdditionOperationCuda additionOperation;
        SubstracionOperationCuda substracionOperation;
        DotProductOperationCuda dotProductOperation;
        MultiplicationConstOperationCuda multiplicationConstOperation;
        void registerDeviceInfo(::cuda::DeviceInfo* deviceInfo);
        typedef std::vector<DeviceInfo*> DevicesInfos;
        DevicesInfos devicesInfos;
    public:
        void setDevice(CUdevice cuDecive);
        void setDeviceInfo(const ::cuda::DeviceInfo& deviceInfo);
        CUdevice getDevice() const;
        MathOperationsCuda();
        math::Status add(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status substract(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status multiply(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
        math::Status multiply(math::Matrix* output, math::Matrix* matrix, floatt* value);
    };
}

#endif	/* MATRIXOPERATIONSCPU_H */