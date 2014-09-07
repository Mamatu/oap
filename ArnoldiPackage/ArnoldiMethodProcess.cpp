/* 
 * File:   ArnoldiMethodProcess.cpp
 * Author: mmatula
 * 
 * Created on August 16, 2014, 7:18 PM
 */

#include "ArnoldiMethodProcess.h"
#include "MathOperationsCpu.h"
#include "ArnoldiMethodImpl.h"
#include <math.h>

namespace api {

    ArnoldiPackage::ArnoldiPackage(ArnoldiPackage::Type type) :
    m_type(type),
    m_method(NULL),
    m_matrix(NULL),
    m_rho(sqrt(2.)),
    m_hDimension(0) {
    }

    ArnoldiPackage::ArnoldiPackage(const ArnoldiPackage& orig) {
    }

    ArnoldiPackage::~ArnoldiPackage() {
    }

    math::Status ArnoldiPackage::getState(const Buffer* buffer) {
        if (m_state == STATE_STOPED) {

        }
        return math::STATUS_OK;
    }

    math::Status ArnoldiPackage::setState(const Buffer* buffer) {
        if (m_state == STATE_STOPED) {

        }
        return math::STATUS_OK;
    }

    math::Status ArnoldiPackage::start() {
        math::Status status = math::STATUS_ERROR;
        if (NULL == m_method) {
            m_method = newArnoldiMethod();
        }
        if (NULL != m_matrix) {
            m_state = STATE_STARTED;
            m_method->setMatrix(m_matrix);
            m_method->setReOutputValues(m_reoutputs.m_outputs, m_reoutputs.m_count);
            m_method->setImOutputValues(m_imoutputs.m_outputs, m_imoutputs.m_count);
            m_method->setHSize(m_hDimension);
            m_method->setRho(m_rho);
            status = m_method->start();
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

    math::IArnoldiMethod* ArnoldiPackage::newArnoldiMethod() const {
        if (m_type == ARNOLDI_CPU) {
            math::MathOperationsCpu* operations =
                    new math::MathOperationsCpu();
            return new math::ArnoldiMethodCpu(operations);
        }
    }

    void ArnoldiPackage::setRho(floatt rho) {
        m_rho = rho;
    }
}
