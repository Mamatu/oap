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



#ifndef ARNOLDIMETHODPROCESS_H
#define	ARNOLDIMETHODPROCESS_H

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
