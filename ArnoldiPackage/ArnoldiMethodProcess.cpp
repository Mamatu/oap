/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#include "ArnoldiMethodProcess.h"
#include "ArnoldiMethodHostImpl.h"
#include <math.h>


namespace api {

ArnoldiPackage::ArnoldiPackage(ArnoldiPackage::Type type) :
    m_type(type),
    m_method(NULL),
    m_operationsCpu(NULL),
    m_matrix(NULL),
    m_rho(sqrt(2.)),
    m_hDimension(0) {
}

ArnoldiPackage::ArnoldiPackage(const ArnoldiPackage& orig) {
    // not implemented
}

ArnoldiPackage::~ArnoldiPackage() {
    delete m_method;
    delete m_operationsCpu;
}

math::Status ArnoldiPackage::start() {
    math::Status status = math::STATUS_ERROR;
    if (NULL == m_method) {
        m_method = newArnoldiMethod();
    } else {
        debugError("Method is not supoorted. Method = %p", m_method);
    }
    if (NULL != m_matrix) {
        m_state = STATE_STARTED;
        m_method->setMatrix(m_matrix);
        m_method->setReOutputValues(m_reoutputs.m_outputs,
            m_reoutputs.m_count);
        m_method->setImOutputValues(m_imoutputs.m_outputs,
            m_imoutputs.m_count);
        m_method->setHSize(m_hDimension);
        m_method->setRho(m_rho);
        status = m_method->start();
    } else {
        debugError("Not defined argument! Matrix = %p", m_matrix);
    }
    return status;
}

math::Status ArnoldiPackage::stop() {
    m_state = STATE_STOPED;
    if (NULL != m_method) {

        delete m_method;
        m_method = NULL;
    }
    return math::STATUS_NOT_SUPPORTED;
}

math::Status ArnoldiPackage::setMatrix(math::Matrix* matrix) {
    m_matrix = matrix;
    return math::STATUS_OK;
}

math::Status ArnoldiPackage::setEigenvaluesBuffer(floatt* reoutputs,
    floatt* imoutputs,
    size_t count) {
    m_reoutputs.m_outputs = reoutputs;
    m_reoutputs.m_count = count;
    m_imoutputs.m_outputs = imoutputs;
    m_imoutputs.m_count = count;
    return math::STATUS_OK;
}

math::Status ArnoldiPackage::setHDimension(uintt dimension) {
    m_hDimension = dimension;
    return math::STATUS_OK;
}

math::Status ArnoldiPackage::setEigenvectorsBuffer(math::Matrix* outputs,
    size_t count) {
    m_outputsVector.m_outputs = outputs;
    m_outputsVector.m_count = count;
    return math::STATUS_OK;
}

math::IArnoldiMethod* ArnoldiPackage::newArnoldiMethod() {
    switch (m_type) {
        case ARNOLDI_CPU:
            m_operationsCpu = new math::MathOperationsCpu();
            return new math::ArnoldiMethodCpu(m_operationsCpu);
        case ARNOLDI_GPU:
            return NULL;
        case ARNOLDI_CALLBACK_CPU:
            m_operationsCpu = new math::MathOperationsCpu();
            return new math::ArnoldiMethodCallbackCpu(m_operationsCpu, 1);
    }
}

void ArnoldiPackage::setRho(floatt rho) {
    m_rho = rho;
}
}
