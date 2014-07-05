/* 
 * File:   MatrixStructureUtils.h
 * Author: mmatula
 *
 * Created on May 24, 2014, 5:35 PM
 */

#ifndef MATRIXSTRUCTUREUTILS_H
#define	MATRIXSTRUCTUREUTILS_H

#include "MatrixStructure.h"
#include "MatrixModules.h"

class MatrixStructureUtils {
    MatrixModule* m_matrixModule;
protected:
    MatrixStructureUtils(MatrixModule* matrixModule);
    virtual ~MatrixStructureUtils();
    virtual void setMatrixToStructure(MatrixStructure* matrixStructure, math::Matrix* matrix) = 0;
public:
    virtual MatrixStructure* newMatrixStructure() = 0;
    virtual void deleteMatrixStructure(MatrixStructure* matrixStructure) = 0;
    void setMatrix(MatrixStructure* matrixStructure, math::Matrix* matrix);
    virtual math::Matrix* getMatrix(MatrixStructure* matrixStructure) = 0;
    void setSubColumns(MatrixStructure* matrixStructure,
            uintt begin, uintt end);
    void setSubColumns(MatrixStructure* matrixStructure, uintt range[2]);
    virtual void setSubColumns(MatrixStructure* matrixStructure, uintt columns) = 0;
    virtual void setBeginColumn(MatrixStructure* matrixStructure, uintt beginColumn) = 0;
    void setSubRows(MatrixStructure* matrixStructure,
            uintt begin, uintt end);
    void setSubRows(MatrixStructure* matrixStructure, uintt range[2]);
    virtual void setSubRows(MatrixStructure* matrixStructure, uintt rows) = 0;
    virtual void setBeginRow(MatrixStructure* matrixStructure, uintt beginRow) = 0;
    virtual bool isValid(MatrixStructure* matrixStructure) = 0;
    void setSub(MatrixStructure* matrixStructure, math::Matrix* matrix,
            uintt subcolumns[2], uintt subrows[2]);
    void setSub(MatrixStructure* matrixStructure, uintt subcolumns[2], uintt subrows[2]);

    virtual uintt getBeginColumn(MatrixStructure* matrixStructure) const = 0;
    virtual uintt getBeginRow(MatrixStructure* matrixStructure) const = 0;
    virtual uintt getSubColumns(MatrixStructure* matrixStructure) const = 0;
    virtual uintt getSubRows(MatrixStructure* matrixStructure) const = 0;
};

#endif	/* MATRIXSTRUCTUREUTILS_H */

