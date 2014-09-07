/* 
 * File:   MatrixOperationsCUDA.cpp
 * Author: mmatula
 * 
 * Created on November 24, 2013, 3:34 PM
 */

#include <vector>

#include "MathOperationsCuda.h"
#include "KernelsOperations.h"
#include "DeviceMatrixModules.h"
#include "DeviceMatrixStructure.h"

namespace math {

    namespace cuda {

        void KernelType::setDevice(CUdevice cuDecive) {
            this->m_kernel.setDevice(cuDecive);
        }

        void KernelType::setDeviceInfo(const DeviceInfo& deviceInfo) {
            this->m_kernel.setDeviceInfo(deviceInfo);
        }

        CUdevice KernelType::getDevice() const {
            return this->m_kernel.getDevice();
        }

        AdditionOperation::AdditionOperation() :
        math::IAdditionOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        AdditionOperation::~AdditionOperation() {
        }

        math::Status AdditionOperation::beforeExecution() {
            MatrixStructure* output = this->m_outputStructure;
            uintt columns = m_matrixStructureUtils->getSubColumns(output);
            uintt rows = m_matrixStructureUtils->getSubRows(output);
            m_kernel.setMatrixSizes(columns, rows);
            return IAdditionOperation::beforeExecution();
        }

        math::Status SubstracionOperation::beforeExecution() {
            MatrixStructure* output = this->m_outputStructure;
            uintt columns = m_matrixStructureUtils->getSubColumns(output);
            uintt rows = m_matrixStructureUtils->getSubRows(output);
            m_kernel.setMatrixSizes(columns, rows);
            return ISubstracionOperation::beforeExecution();
        }

        math::Status DotProductOperation::beforeExecution() {
            MatrixStructure* output = this->m_outputStructure;
            uintt columns = m_matrixStructureUtils->getSubColumns(output);
            uintt rows = m_matrixStructureUtils->getSubRows(output);
            m_kernel.setMatrixSizes(columns, rows);
            return IDotProductOperation::beforeExecution();
        }

        math::Status MultiplicationConstOperation::beforeExecution() {
            MatrixStructure* output = this->m_outputStructure;
            uintt columns = m_matrixStructureUtils->getSubColumns(output);
            uintt rows = m_matrixStructureUtils->getSubRows(output);
            m_kernel.setMatrixSizes(columns, rows);
            return IMultiplicationConstOperation::beforeExecution();
        }

        void AdditionOperation::execute() {
            MatrixStructure* output = this->m_outputStructure;
            MatrixStructure* matrix1 = this->m_matrixStructure1;
            MatrixStructure* matrix2 = this->m_matrixStructure2;
            DeviceMatrixUtils dmu;
            if (this->m_executionPathRe == AdditionOperation::EXECUTION_NORMAL &&
                    this->m_executionPathIm == AdditionOperation::EXECUTION_NORMAL) {
                m_kernelOperation.addDeviceMatrices(output, matrix1, matrix2, m_kernel);
            } else if (this->m_executionPathRe == AdditionOperation::EXECUTION_NORMAL) {
                m_kernelOperation.addDeviceReMatrices(output, matrix1, matrix2,
                        m_kernel);
            } else if (this->m_executionPathIm == AdditionOperation::EXECUTION_NORMAL) {
                m_kernelOperation.addDeviceImMatrices(output, matrix1, matrix2, m_kernel);
            }
        }

        SubstracionOperation::SubstracionOperation() :
        math::ISubstracionOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        SubstracionOperation::~SubstracionOperation() {

        }

        void SubstracionOperation::execute() {
            MatrixStructure* output = this->m_outputStructure;
            MatrixStructure* matrix1 = this->m_matrixStructure1;
            MatrixStructure* matrix2 = this->m_matrixStructure2;
            if (this->m_executionPathRe == SubstracionOperation::EXECUTION_NORMAL &&
                    this->m_executionPathIm == SubstracionOperation::EXECUTION_NORMAL) {
                m_kernelOperations.substractDeviceMatrices(output, matrix1, matrix2, m_kernel);
            } else if (this->m_executionPathRe == SubstracionOperation::EXECUTION_NORMAL) {
                m_kernelOperations.substractDeviceReMatrices(output, matrix1, matrix2, m_kernel);
            } else if (this->m_executionPathIm == SubstracionOperation::EXECUTION_NORMAL) {
                m_kernelOperations.substractDeviceImMatrices(output, matrix1, matrix2, m_kernel);
            }
        }

        DotProductOperation::DotProductOperation() :
        math::IDotProductOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        DotProductOperation::~DotProductOperation() {
        }

        void DotProductOperation::execute() {
            MatrixStructure* output = this->m_outputStructure;
            MatrixStructure* matrix1 = this->m_matrixStructure1;
            MatrixStructure* matrix2 = this->m_matrixStructure2;
            if (this->m_executionPathRe == DotProductOperation::EXECUTION_NORMAL &&
                    this->m_executionPathIm == DotProductOperation::EXECUTION_NORMAL) {
                m_kernelOperation.dotProductDeviceMatrices(output, matrix1, matrix2, m_kernel);
            } else if (this->m_executionPathRe == DotProductOperation::EXECUTION_NORMAL) {
                m_kernelOperation.dotProductDeviceReMatrices(output, matrix1, matrix2, m_kernel);
            } else if (this->m_executionPathIm == DotProductOperation::EXECUTION_NORMAL) {
                m_kernelOperation.dotProductDeviceImMatrices(output, matrix1, matrix2, m_kernel);
            }
        }

        MultiplicationConstOperation::MultiplicationConstOperation() :
        math::IMultiplicationConstOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        MultiplicationConstOperation::~MultiplicationConstOperation() {

        }

        void MultiplicationConstOperation::execute() {
            MatrixStructure* output = this->m_outputStructure;
            MatrixStructure* matrix = this->m_matrixStructure;
            floatt* value = this->m_revalue;
            if (this->m_executionPathRe == MultiplicationConstOperation::EXECUTION_NORMAL &&
                    this->m_executionPathIm == MultiplicationConstOperation::EXECUTION_NORMAL) {
                m_kernelOperation.multiplyConstantDeviceMatrix(output, matrix, value, m_kernel);
            } else if (this->m_executionPathRe == MultiplicationConstOperation::EXECUTION_NORMAL) {
                m_kernelOperation.multiplyConstantDeviceReMatrix(output, matrix, value, m_kernel);
            } else if (this->m_executionPathIm == MultiplicationConstOperation::EXECUTION_NORMAL) {
                m_kernelOperation.multiplyConstantDeviceImMatrix(output, matrix, value, m_kernel);
            }
        }

        ExpOperation::ExpOperation() :
        math::IExpOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        ExpOperation::~ExpOperation() {
        }

        void ExpOperation::setSerieLimit(int serieLimit) {
            this->serieLimit = serieLimit;
        }

        void ExpOperation::execute() {
        }

        DiagonalizationOperation::DiagonalizationOperation() :
        math::IDiagonalizationOperation(DeviceMatrixModules::GetInstance(),
        DeviceMatrixStructureUtils::GetInstance()) {
        }

        DiagonalizationOperation::~DiagonalizationOperation() {

        }

        void DiagonalizationOperation::execute() {
        }

        MathOperations::MathOperations() {
            registerDeviceInfo(&additionOperation);
            registerDeviceInfo(&substracionOperation);
            registerDeviceInfo(&dotProductOperation);
            registerDeviceInfo(&multiplicationConstOperation);
        }

        void MathOperations::registerDeviceInfo(DeviceInfo* deviceInfo) {
            this->devicesInfos.push_back(deviceInfo);
        }

        void MathOperations::setDevice(CUdevice cuDecive) {
            for (int fa = 0; fa<this->devicesInfos.size(); fa++) {
                this->devicesInfos[fa]->setDevice(cuDecive);
            }
        }

        void MathOperations::setDeviceInfo(const DeviceInfo& deviceInfo) {
            for (int fa = 0; fa<this->devicesInfos.size(); fa++) {
                this->devicesInfos[fa]->setDeviceInfo(deviceInfo);
            }
        }

        CUdevice MathOperations::getDevice() const {
            return this->devicesInfos[0]->getDevice();
        }

        math::Status MathOperations::add(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2) {
            this->additionOperation.setOutputMatrix(output);
            this->additionOperation.setMatrix1(matrix1);
            this->additionOperation.setMatrix2(matrix2);
            math::Status status = this->additionOperation.start();
            return status;
        }

        math::Status MathOperations::substract(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2) {
            this->substracionOperation.setOutputMatrix(output);
            this->substracionOperation.setMatrix1(matrix1);
            this->substracionOperation.setMatrix2(matrix2);
            math::Status status = this->substracionOperation.start();
            return status;
        }

        math::Status MathOperations::multiply(math::Matrix* output,
                math::Matrix* matrix1, math::Matrix* matrix2) {
            this->dotProductOperation.setOutputMatrix(output);
            this->dotProductOperation.setMatrix1(matrix1);
            this->dotProductOperation.setMatrix2(matrix2);
            math::Status status = this->dotProductOperation.start();
            return status;
        }

        math::Status MathOperations::multiply(math::Matrix* output,
                math::Matrix* matrix, floatt* value) {
            this->multiplicationConstOperation.setOutputMatrix(output);
            this->multiplicationConstOperation.setMatrix(matrix);
            this->multiplicationConstOperation.setReValue(value);
            math::Status status = this->multiplicationConstOperation.start();
            return status;
        }
    }
}