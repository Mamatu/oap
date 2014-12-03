/* 
 * File:   DeviceMatrixStructureFuncs.cpp
 * Author: mmatula
 * 
 * Created on May 21, 2014, 8:13 PM
 */

#include "DeviceMatrixStructure.h"
#include "DeviceMatrixModules.h"

DeviceMatrixStructureUtils* DeviceMatrixStructureUtils::m_deviceMatrixStructureUtils = NULL;

DeviceMatrixStructureUtils::DeviceMatrixStructureUtils() :
MatrixStructureUtils(DeviceMatrixModules::GetInstance()) {
}

DeviceMatrixStructureUtils::~DeviceMatrixStructureUtils() {
}

MatrixStructure* DeviceMatrixStructureUtils::newMatrixStructure() {
    MatrixStructure temp;
    temp.m_beginColumn = 0;
    temp.m_beginRow = 0;
    temp.m_subcolumns = 0;
    temp.m_subrows = 0;
    temp.m_matrix = NULL;
    MatrixStructure* deviceMatrixStructure = 
            static_cast<MatrixStructure*> (CudaUtils::NewDevice(sizeof (MatrixStructure), &temp));
    return deviceMatrixStructure;
}

void DeviceMatrixStructureUtils::deleteMatrixStructure(MatrixStructure* matrixStructure) {
    CudaUtils::DeleteDevice(matrixStructure);
}

void DeviceMatrixStructureUtils::setMatrixToStructure(MatrixStructure* matrixStructure,
        math::Matrix* matrix) {
    CudaUtils::CopyHostToDevice(&matrixStructure->m_matrix, &matrix, sizeof (matrix));
    uintt rows = dmu.getRows(matrix);
    uintt columns = dmu.getColumns(matrix);
    this->setSubColumns(matrixStructure, columns);
    this->setSubRows(matrixStructure, rows);
    this->setBeginColumn(matrixStructure, 0);
    this->setBeginRow(matrixStructure, 0);
    matrix = this->getMatrix(matrixStructure);
    dmu.printInfo(matrix);
}

math::Matrix* DeviceMatrixStructureUtils::getMatrix(MatrixStructure* matrixStructure) {
    math::Matrix* matrix = NULL;
    CudaUtils::CopyDeviceToHost(&matrix, &matrixStructure->m_matrix, sizeof (matrix));
    return matrix;
}

void DeviceMatrixStructureUtils::setSubColumns(MatrixStructure* matrixStructure, uintt columns) {
    CudaUtils::CopyHostToDevice(&matrixStructure->m_subcolumns, &columns,
            sizeof (uintt));
}

void DeviceMatrixStructureUtils::setBeginColumn(MatrixStructure* matrixStructure, uintt beginColumn) {
    CudaUtils::CopyHostToDevice(&matrixStructure->m_beginColumn, &beginColumn,
            sizeof (uintt));
}

void DeviceMatrixStructureUtils::setSubRows(MatrixStructure* matrixStructure, uintt rows) {
    CudaUtils::CopyHostToDevice(&matrixStructure->m_subrows, &rows,
            sizeof (uintt));
}

void DeviceMatrixStructureUtils::setBeginRow(MatrixStructure* matrixStructure, uintt beginRow) {
    CudaUtils::CopyHostToDevice(&matrixStructure->m_beginRow, &beginRow,
            sizeof (uintt));
}

bool DeviceMatrixStructureUtils::isValid(MatrixStructure* matrixStructure) {
    return true;
}

DeviceMatrixStructureUtils* DeviceMatrixStructureUtils::GetInstance() {
    if (m_deviceMatrixStructureUtils == NULL) {
        m_deviceMatrixStructureUtils = new DeviceMatrixStructureUtils();
    }
    return m_deviceMatrixStructureUtils;
}

uintt DeviceMatrixStructureUtils::getBeginColumn(MatrixStructure* matrixStructure) const {
    uintt beginColumn = 0;
    CudaUtils::CopyDeviceToHost(&beginColumn, &matrixStructure->m_beginColumn,
            sizeof (beginColumn));
    return beginColumn;
}

uintt DeviceMatrixStructureUtils::getBeginRow(MatrixStructure* matrixStructure) const {
    uintt beginRow = 0;
    CudaUtils::CopyDeviceToHost(&beginRow, &matrixStructure->m_beginRow,
            sizeof (beginRow));
    return beginRow;
}

uintt DeviceMatrixStructureUtils::getSubColumns(MatrixStructure* matrixStructure) const {
    uintt columns = 0;
    CudaUtils::CopyDeviceToHost(&columns, &matrixStructure->m_subcolumns,
            sizeof (columns));
    return columns;
}

uintt DeviceMatrixStructureUtils::getSubRows(MatrixStructure* matrixStructure) const {
    uintt rows = 0;
    CudaUtils::CopyDeviceToHost(&rows, &matrixStructure->m_subrows,
            sizeof (rows));
    return rows;
}

MatrixStructure* DeviceMatrixStructureUtils::CreateMatrixStructure(math::Matrix* matrix) {
    DeviceMatrixStructureUtils* dmsu = DeviceMatrixStructureUtils::GetInstance();
    MatrixStructure* ms = dmsu->newMatrixStructure();
    dmsu->setMatrix(ms, matrix);
    return ms;
}
