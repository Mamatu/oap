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
    namespace cuda {

        class KernelType : public ::cuda::DeviceInfo {
        protected:
            ::cuda::KernelMatrix m_kernel;
        public:
            void setDevice(CUdevice cuDecive);
            void setDeviceInfo(const ::cuda::DeviceInfo& deviceInfo);
            CUdevice getDevice() const;
        };

        class AdditionOperation : public math::IAdditionOperation,
        public KernelType {
            KernelsOperations m_kernelOperation;
        public:
            math::Status beforeExecution();
            void execute();
            AdditionOperation();
            ~AdditionOperation();
        };

        class SubstracionOperation : public math::ISubstracionOperation,
        public KernelType {
            KernelsOperations m_kernelOperations;
        public:
            math::Status beforeExecution();
            void execute();
            SubstracionOperation();
            ~SubstracionOperation();
        };

        class DotProductOperation : public math::IDotProductOperation,
        public KernelType {
            KernelsOperations m_kernelOperation;
        public:
            math::Status beforeExecution();
            void execute();
            DotProductOperation();
            ~DotProductOperation();
        };

        class MultiplicationConstOperation : public math::IMultiplicationConstOperation,
        public KernelType {
            KernelsOperations m_kernelOperation;
        public:
            math::Status beforeExecution();
            void execute();
            MultiplicationConstOperation();
            ~MultiplicationConstOperation();
        };

        class ExpOperation : public math::IExpOperation, public KernelType {
            KernelsOperations m_kernelOperation;
            int serieLimit;
        protected:
            void setParamIn(void* inPtr, int index);
            DotProductOperation multiplicationOperation;
            MultiplicationConstOperation multiplicationConstOperation;
            AdditionOperation additionOperation;
        public:
            void setSerieLimit(int serieLimit);
            void execute();
            ExpOperation();
            ~ExpOperation();
        };

        class DiagonalizationOperation : public math::IDiagonalizationOperation,
        public KernelType {
            KernelsOperations m_kernelOperation;
        public:
            void execute();
            DiagonalizationOperation();
            ~DiagonalizationOperation();
        };

        class MathOperations;

        class MathOperations : public utils::Module, public ::cuda::DeviceInfo {
            AdditionOperation additionOperation;
            SubstracionOperation substracionOperation;
            DotProductOperation dotProductOperation;
            MultiplicationConstOperation multiplicationConstOperation;
            void registerDeviceInfo(::cuda::DeviceInfo* deviceInfo);
            typedef std::vector<DeviceInfo*> DevicesInfos;
            DevicesInfos devicesInfos;
        public:
            void setDevice(CUdevice cuDecive);
            void setDeviceInfo(const ::cuda::DeviceInfo& deviceInfo);
            CUdevice getDevice() const;
            MathOperations();
            math::Status add(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
            math::Status substract(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
            math::Status multiply(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
            math::Status dotProduct(math::Matrix* output, math::Matrix* matrix1, math::Matrix* matrix2);
            math::Status multiply(math::Matrix* output, math::Matrix* matrix, floatt* value);
        };
    }
}

#endif	/* MATRIXOPERATIONSCPU_H */