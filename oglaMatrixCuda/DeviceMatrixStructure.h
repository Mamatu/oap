/* 
 * File:   DeviceMatrixStructure.h
 * Author: mmatula
 *
 * Created on May 21, 2014, 8:03 PM
 */

#ifndef DEVICEMATRIXSTRUCTURE_H
#define	DEVICEMATRIXSTRUCTURE_H

#include "MatrixStructureUtils.h"
#include "DeviceMatrixModules.h"

class DeviceMatrixStructureUtils : public MatrixStructureUtils {
    static DeviceMatrixStructureUtils* m_deviceMatrixStructureUtils;
    DeviceMatrixUtils dmu;
protected:
    DeviceMatrixStructureUtils();

    virtual ~DeviceMatrixStructureUtils();
    virtual void setMatrixToStructure(MatrixStructure* matrixStructure, math::Matrix* matrix);
public:
    virtual MatrixStructure* newMatrixStructure();
    virtual void deleteMatrixStructure(MatrixStructure* matrixStructure);
    virtual math::Matrix* getMatrix(MatrixStructure* matrixStructure);
    virtual void setSubColumns(MatrixStructure* matrixStructure, uintt columns);
    virtual void setBeginColumn(MatrixStructure* matrixStructure, uintt beginColumn);
    virtual void setSubRows(MatrixStructure* matrixStructure, uintt rows);
    virtual void setBeginRow(MatrixStructure* matrixStructure, uintt beginRow);
    virtual bool isValid(MatrixStructure* matrixStructure);
    static DeviceMatrixStructureUtils* GetInstance();

    virtual uintt getBeginColumn(MatrixStructure* matrixStructure) const;
    virtual uintt getBeginRow(MatrixStructure* matrixStructure) const;
    virtual uintt getSubColumns(MatrixStructure* matrixStructure) const;
    virtual uintt getSubRows(MatrixStructure* matrixStructure) const;
    
    static MatrixStructure* CreateMatrixStructure(math::Matrix* matrix);
};

#endif	/* DEVICEMATRIXSTRUCTURE_H */

