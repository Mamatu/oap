/* 
 * File:   ArnoldiMethodProcess.h
 * Author: mmatula
 *
 * Created on August 16, 2014, 7:18 PM
 */

#ifndef ARNOLDIMETHODPROCESS_H
#define	ARNOLDIMETHODPROCESS_H

#include "Types.h"
#include "InternalTypes.h"
#include "MathOperations.h"
#include "IArnoldiMethod.h"
#include "MathOperationsCpu.h"

namespace api {

class ArnoldiPackage {
public:

    enum Type {
        ARNOLDI_CPU,
        ARNOLDI_CALLBACK_CPU,
        ARNOLDI_GPU,
        ARNOLDI_CALLBACK_GPU,
    };

private:
    State m_state;
    /*Not supported yet.*/
    ArnoldiPackage(const ArnoldiPackage& orig);

    template<typename T> class Outputs {
    public:

        Outputs() {
            m_outputs = NULL;
            m_count = 0;
        }

        T* m_outputs;
        size_t m_count;
    };

    Outputs<floatt> m_reoutputs;
    Outputs<floatt> m_imoutputs;
    Outputs<math::Matrix> m_outputsVector;
    Type m_type;
    math::IArnoldiMethod* m_method;
    math::MathOperationsCpu* m_operationsCpu;
    math::Matrix* m_matrix;
    floatt m_rho;
    uintt m_hDimension;
private:
    math::IArnoldiMethod* newArnoldiMethod();

public:
    ArnoldiPackage(Type type);

    virtual ~ArnoldiPackage();

    void setRho(floatt rho);

    math::Status setMatrix(math::Matrix* matrix);

    math::Status setEigenvaluesBuffer(
        floatt* reoutputs,
        floatt* imoutputs,
        size_t count);

    math::Status setHDimension(uintt dimension);

    math::Status setEigenvectorsBuffer(math::Matrix* outputs, size_t size);

    math::Status start();
    math::Status stop();
};

}

#endif	/* ARNOLDIMETHODPROCESS_H */
