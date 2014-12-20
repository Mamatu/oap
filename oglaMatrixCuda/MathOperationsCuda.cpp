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

namespace math {

    void KernelType::setDevice(CUdevice cuDecive) {
        this->m_kernel.setDevice(cuDecive);
    }

    void KernelType::setDeviceInfo(const DeviceInfo& deviceInfo) {
        this->m_kernel.setDeviceInfo(deviceInfo);
    }

    CUdevice KernelType::getDevice() const {
        return this->m_kernel.getDevice();
    }

    AdditionOperationCuda::AdditionOperationCuda() :
    math::IAdditionOperation(DeviceMatrixModules::GetInstance()) {
    }

    AdditionOperationCuda::~AdditionOperationCuda() {
    }

    math::Status AdditionOperationCuda::beforeExecution() {
        Matrix* output = this->m_output;
        uintt columns = m_module->getMatrixUtils()->getColumns(output);
        uintt rows = m_module->getMatrixUtils()->getRows(output);
        m_kernel.setMatrixSizes(columns, rows);
        return IAdditionOperation::beforeExecution();
    }

    math::Status SubstracionOperationCuda::beforeExecution() {
        Matrix* output = this->m_output;
        uintt columns = m_module->getMatrixUtils()->getColumns(output);
        uintt rows = m_module->getMatrixUtils()->getRows(output);
        m_kernel.setMatrixSizes(columns, rows);
        return ISubstracionOperation::beforeExecution();
    }

    math::Status DotProductOperationCuda::beforeExecution() {
        Matrix* output = this->m_output;
        uintt columns = m_module->getMatrixUtils()->getColumns(output);
        uintt rows = m_module->getMatrixUtils()->getRows(output);
        m_kernel.setMatrixSizes(columns, rows);
        return IDotProductOperation::beforeExecution();
    }

    math::Status MultiplicationConstOperationCuda::beforeExecution() {
        Matrix* output = this->m_output;
        uintt columns = m_module->getMatrixUtils()->getColumns(output);
        uintt rows = m_module->getMatrixUtils()->getRows(output);
        m_kernel.setMatrixSizes(columns, rows);
        return IMultiplicationConstOperation::beforeExecution();
    }

    void AdditionOperationCuda::execute() {
        Matrix* output = this->m_output;
        Matrix* matrix1 = this->m_matrix1;
        Matrix* matrix2 = this->m_matrix2;
        DeviceMatrixUtils dmu;
        if (this->m_executionPathRe == AdditionOperationCuda::EXECUTION_NORMAL &&
                this->m_executionPathIm == AdditionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.addDeviceMatrices(output, matrix1, matrix2, m_kernel);
        } else if (this->m_executionPathRe == AdditionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.addDeviceReMatrices(output, matrix1, matrix2,
                    m_kernel);
        } else if (this->m_executionPathIm == AdditionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.addDeviceImMatrices(output, matrix1, matrix2, m_kernel);
        }
    }

    SubstracionOperationCuda::SubstracionOperationCuda() :
    math::ISubstracionOperation(DeviceMatrixModules::GetInstance()) {
    }

    SubstracionOperationCuda::~SubstracionOperationCuda() {

    }

    void SubstracionOperationCuda::execute() {
        Matrix* output = this->m_output;
        Matrix* matrix1 = this->m_matrix1;
        Matrix* matrix2 = this->m_matrix2;
        if (this->m_executionPathRe == SubstracionOperationCuda::EXECUTION_NORMAL &&
                this->m_executionPathIm == SubstracionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperations.substractDeviceMatrices(output, matrix1, matrix2, m_kernel);
        } else if (this->m_executionPathRe == SubstracionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperations.substractDeviceReMatrices(output, matrix1, matrix2, m_kernel);
        } else if (this->m_executionPathIm == SubstracionOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperations.substractDeviceImMatrices(output, matrix1, matrix2, m_kernel);
        }
    }

    DotProductOperationCuda::DotProductOperationCuda() :
    math::IDotProductOperation(DeviceMatrixModules::GetInstance()) {
    }

    DotProductOperationCuda::~DotProductOperationCuda() {
    }

    void DotProductOperationCuda::execute() {
        Matrix* output = this->m_output;
        Matrix* matrix1 = this->m_matrix1;
        Matrix* matrix2 = this->m_matrix2;
        if (this->m_executionPathRe == DotProductOperationCuda::EXECUTION_NORMAL &&
                this->m_executionPathIm == DotProductOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.dotProductDeviceMatrices(output, matrix1, matrix2, m_kernel);
        } else if (this->m_executionPathRe == DotProductOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.dotProductDeviceReMatrices(output, matrix1, matrix2, m_kernel);
        } else if (this->m_executionPathIm == DotProductOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.dotProductDeviceImMatrices(output, matrix1, matrix2, m_kernel);
        }
    }

    MultiplicationConstOperationCuda::MultiplicationConstOperationCuda() :
    math::IMultiplicationConstOperation(DeviceMatrixModules::GetInstance()) {
    }

    MultiplicationConstOperationCuda::~MultiplicationConstOperationCuda() {

    }

    void MultiplicationConstOperationCuda::execute() {
        Matrix* output = this->m_output;
        Matrix* matrix = this->m_matrix;
        floatt* value = this->m_revalue;
        if (this->m_executionPathRe == MultiplicationConstOperationCuda::EXECUTION_NORMAL &&
                this->m_executionPathIm == MultiplicationConstOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.multiplyConstantDeviceMatrix(output, matrix, value, m_kernel);
        } else if (this->m_executionPathRe == MultiplicationConstOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.multiplyConstantDeviceReMatrix(output, matrix, value, m_kernel);
        } else if (this->m_executionPathIm == MultiplicationConstOperationCuda::EXECUTION_NORMAL) {
            m_kernelOperation.multiplyConstantDeviceImMatrix(output, matrix, value, m_kernel);
        }
    }

    ExpOperationCuda::ExpOperationCuda() :
    math::IExpOperation(DeviceMatrixModules::GetInstance()) {
    }

    ExpOperationCuda::~ExpOperationCuda() {
    }

    void ExpOperationCuda::setSerieLimit(int serieLimit) {
        this->serieLimit = serieLimit;
    }

    void ExpOperationCuda::execute() {
    }

    DiagonalizationOperationCuda::DiagonalizationOperationCuda() :
    math::IDiagonalizationOperation(DeviceMatrixModules::GetInstance()) {
    }

    DiagonalizationOperationCuda::~DiagonalizationOperationCuda() {

    }

    void DiagonalizationOperationCuda::execute() {
    }

    MathOperationsCuda::MathOperationsCuda() {
        registerDeviceInfo(&additionOperation);
        registerDeviceInfo(&substracionOperation);
        registerDeviceInfo(&dotProductOperation);
        registerDeviceInfo(&multiplicationConstOperation);
    }

    void MathOperationsCuda::registerDeviceInfo(DeviceInfo* deviceInfo) {
        this->devicesInfos.push_back(deviceInfo);
    }

    void MathOperationsCuda::setDevice(CUdevice cuDecive) {
        for (int fa = 0; fa<this->devicesInfos.size(); fa++) {
            this->devicesInfos[fa]->setDevice(cuDecive);
        }
    }

    void MathOperationsCuda::setDeviceInfo(const DeviceInfo& deviceInfo) {
        for (int fa = 0; fa<this->devicesInfos.size(); fa++) {
            this->devicesInfos[fa]->setDeviceInfo(deviceInfo);
        }
    }

    CUdevice MathOperationsCuda::getDevice() const {
        return this->devicesInfos[0]->getDevice();
    }

    math::Status MathOperationsCuda::add(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2) {
        this->additionOperation.setOutputMatrix(output);
        this->additionOperation.setMatrix1(matrix1);
        this->additionOperation.setMatrix2(matrix2);
        math::Status status = this->additionOperation.start();
        return status;
    }

    math::Status MathOperationsCuda::substract(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2) {
        this->substracionOperation.setOutputMatrix(output);
        this->substracionOperation.setMatrix1(matrix1);
        this->substracionOperation.setMatrix2(matrix2);
        math::Status status = this->substracionOperation.start();
        return status;
    }

    math::Status MathOperationsCuda::multiply(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2) {
        return dotProduct(output, matrix1, matrix2);
    }

    math::Status MathOperationsCuda::dotProduct(math::Matrix* output,
            math::Matrix* matrix1, math::Matrix* matrix2) {
        this->dotProductOperation.setOutputMatrix(output);
        this->dotProductOperation.setMatrix1(matrix1);
        this->dotProductOperation.setMatrix2(matrix2);
        math::Status status = this->dotProductOperation.start();
        return status;
    }

    math::Status MathOperationsCuda::multiply(math::Matrix* output,
            math::Matrix* matrix, floatt* value) {
        this->multiplicationConstOperation.setOutputMatrix(output);
        this->multiplicationConstOperation.setMatrix(matrix);
        this->multiplicationConstOperation.setReValue(value);
        math::Status status = this->multiplicationConstOperation.start();
        return status;
    }
}
