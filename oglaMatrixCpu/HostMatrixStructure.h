#ifndef OGLA_MATRIX_STRUCTURE_H
#define	OGLA_MATRIX_STRUCTURE_H

#include <string.h>
#include "Matrix.h"
#include "MatrixModules.h"
#include "MatrixStructureUtils.h"


#define IS_DEFINED(sub) (sub[0] != -1 && sub[1] != -1)

class HostMatrixStructureUtils : public MatrixStructureUtils {
    static HostMatrixStructureUtils* m_hostMatrixStructureUtils;
protected:
    HostMatrixStructureUtils(MatrixModule* matrixModule);
    virtual ~HostMatrixStructureUtils();
    void setMatrixToStructure(MatrixStructure* matrixStructure, math::Matrix* matrix);
public:
    MatrixStructure* newMatrixStructure();
    void deleteMatrixStructure(MatrixStructure* matrixStructure);
    bool isValid(MatrixStructure* matrixStructure);
    void setSubColumns(MatrixStructure* matrixStructure, uintt columns);
    void setBeginColumn(MatrixStructure* matrixStructure, uintt beginColumn);
    void setSubRows(MatrixStructure* matrixStructure, uintt rows);
    void setBeginRow(MatrixStructure* matrixStructure, uintt beginRow);
    math::Matrix* getMatrix(MatrixStructure* matrixStructure);
    static HostMatrixStructureUtils* GetInstance(MatrixModule* matrixModule);

    virtual uintt getBeginColumn(MatrixStructure* matrixStructure) const;
    virtual uintt getBeginRow(MatrixStructure* matrixStructure) const;
    virtual uintt getSubColumns(MatrixStructure* matrixStructure) const;
    virtual uintt getSubRows(MatrixStructure* matrixStructure) const;
};

#endif	/* INTERNALTYPES_H */

