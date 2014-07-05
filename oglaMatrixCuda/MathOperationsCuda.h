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
        
        class IraMethod : public math::IIraMethod {
            math::Matrix* w;
            math::Matrix* f;
            math::Matrix* f1;
            math::Matrix* vh;
            math::Matrix* h;
            math::Matrix* s;
            math::Matrix* vs;
            math::Matrix* V;
            math::Matrix* V2;
            math::Matrix* V1;
            math::Matrix* transposeV;
            math::Matrix* H;
            math::Matrix* HC;
            math::Matrix* H1;
            math::Matrix* H2;
            math::Matrix* A2;
            math::Matrix* I;
            math::Matrix* A;
            math::Matrix* A1;
            math::Matrix* v;
            math::Matrix* Q1T;
            math::Matrix* Q1;
            math::Matrix* Q2;
            math::Matrix* R1;
            math::Matrix* R2;
            math::Matrix* HO;
            math::Matrix* HO1;
            math::Matrix* Q;
            math::Matrix* QJ;
            std::vector<floatt> unwanted;
            std::vector<floatt> wanted;
            uintt m_k;
            floatt m_rho;
            uintt m_wantedCount;
            MathOperations& m_mathOperations;
            floatt getDiagonal(math::Matrix* matrix, intt index);
            bool continueProcedure(math::Matrix* A, std::vector<floatt>& wanted);
            bool executeArnoldiFactorization(bool init = true, intt initj = 0);
            void selection(math::Matrix* H, std::vector<floatt>& unwanted,
                    std::vector<floatt>& wanted, int i);
        public:
            IraMethod(MathOperations* mathOperations);
            IraMethod(MatrixModule* matrixModule,
                    MatrixStructureUtils* matrixStructureUtils,
                    MathOperations* mathOperations);
            ~IraMethod();
            void setHSize(uintt k);
            void setRho(floatt rho);
            void execute();
        };

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