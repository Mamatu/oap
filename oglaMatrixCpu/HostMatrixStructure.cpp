/* 
 * File:   HostMatrixStructure.cpp
 * Author: mmatula
 * 
 * Created on May 24, 2014, 8:08 AM
 */

#include "HostMatrixStructure.h"
#include "HostMatrixModules.h"

HostMatrixStructureUtils::HostMatrixStructureUtils() :
MatrixStructureUtils(HostMatrixModules::GetInstance()) {
}

math::Matrix* HostMatrixStructureUtils::getMatrix(MatrixStructure* matrixStructure) {
    return matrixStructure->m_matrix;
}

HostMatrixStructureUtils::~HostMatrixStructureUtils() {
}

void HostMatrixStructureUtils::setSubColumns(MatrixStructure* matrixStructure,
        uintt columns) {
    matrixStructure->m_subcolumns = columns;
}

void HostMatrixStructureUtils::setBeginColumn(MatrixStructure* matrixStructure,
        uintt beginColumn) {
    matrixStructure->m_beginColumn = beginColumn;
}

void HostMatrixStructureUtils::setSubRows(MatrixStructure* matrixStructure,
        uintt rows) {
    matrixStructure->m_subrows = rows;
}

void HostMatrixStructureUtils::setBeginRow(MatrixStructure* matrixStructure,
        uintt beginRow) {
    matrixStructure->m_beginRow = beginRow;
}

void HostMatrixStructureUtils::setMatrixToStructure(MatrixStructure* matrixStructure,
        math::Matrix* matrix) {
    matrixStructure->m_matrix = matrix;
    matrixStructure->m_subcolumns = matrix->columns;
    matrixStructure->m_subrows = matrix->rows;
    matrixStructure->m_beginColumn = 0;
    matrixStructure->m_beginRow = 0;
}

HostMatrixStructureUtils* HostMatrixStructureUtils::m_hostMatrixStructureUtils = NULL;

HostMatrixStructureUtils* HostMatrixStructureUtils::GetInstance() {
    if (NULL == m_hostMatrixStructureUtils) {
        m_hostMatrixStructureUtils = new HostMatrixStructureUtils();
    }
    return m_hostMatrixStructureUtils;
}

bool HostMatrixStructureUtils::isValid(MatrixStructure* matrixStructure) {
    return true;
}

MatrixStructure* HostMatrixStructureUtils::newMatrixStructure() {
    MatrixStructure* matrixStructure = new MatrixStructure;
    matrixStructure->m_beginColumn = 0;
    matrixStructure->m_subcolumns = 0;
    matrixStructure->m_beginRow = 0;
    matrixStructure->m_subrows = 0;
    matrixStructure->m_matrix = NULL;
    return matrixStructure;
}

void HostMatrixStructureUtils::deleteMatrixStructure(MatrixStructure* matrixStructure) {
    delete matrixStructure;
}

uintt HostMatrixStructureUtils::getBeginColumn(MatrixStructure* matrixStructure) const {
    return matrixStructure->m_beginColumn;
}

uintt HostMatrixStructureUtils::getBeginRow(MatrixStructure* matrixStructure) const {
    return matrixStructure->m_beginRow;
}

uintt HostMatrixStructureUtils::getSubColumns(MatrixStructure* matrixStructure) const {
    return matrixStructure->m_subcolumns;
}

uintt HostMatrixStructureUtils::getSubRows(MatrixStructure* matrixStructure) const {
    return matrixStructure->m_subrows;
}