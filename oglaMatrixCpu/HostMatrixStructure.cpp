/* 
 * File:   HostMatrixStructure.cpp
 * Author: mmatula
 * 
 * Created on May 24, 2014, 8:08 AM
 */

#include "HostMatrixStructure.h"

HostMatrixStructureUtils::HostMatrixStructureUtils(MatrixModule* matrixModule) :
MatrixStructureUtils(matrixModule) {
}

math::Matrix* HostMatrixStructureUtils::getMatrix(MatrixStructure* matrixStructure) {
    return matrixStructure->m_matrix;
}

HostMatrixStructureUtils::~HostMatrixStructureUtils() {
}

HostMatrixStructureUtils* HostMatrixStructureUtils::m_hostMatrixStructureUtils = NULL;

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

HostMatrixStructureUtils* HostMatrixStructureUtils::GetInstance(MatrixModule* matrixModule) {
    if (m_hostMatrixStructureUtils == NULL) {
        m_hostMatrixStructureUtils = new HostMatrixStructureUtils(matrixModule);
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