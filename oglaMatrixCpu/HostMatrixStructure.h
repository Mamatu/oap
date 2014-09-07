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
    HostMatrixStructureUtils();
    virtual ~HostMatrixStructureUtils();
protected:
    void setMatrixToStructure(MatrixStructure* matrixStructure, math::Matrix* matrix);
public:
    static HostMatrixStructureUtils* GetInstance();

    virtual MatrixStructure* newMatrixStructure();
    virtual void deleteMatrixStructure(MatrixStructure* matrixStructure);

    virtual void setBeginColumn(MatrixStructure* matrixStructure, uintt beginColumn);
    virtual void setBeginRow(MatrixStructure* matrixStructure, uintt beginRow);

    virtual math::Matrix* getMatrix(MatrixStructure* matrixStructure);

    virtual uintt getBeginColumn(MatrixStructure* matrixStructure) const;
    virtual uintt getBeginRow(MatrixStructure* matrixStructure) const;
    virtual uintt getSubColumns(MatrixStructure* matrixStructure) const;
    virtual uintt getSubRows(MatrixStructure* matrixStructure) const;

    virtual bool isValid(MatrixStructure* matrixStructure);
    void setSubColumns(MatrixStructure* matrixStructure, uintt columns);
    void setSubRows(MatrixStructure* matrixStructure, uintt rows);
};

#endif	/* INTERNALTYPES_H */

