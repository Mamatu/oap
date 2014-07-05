#include "MatrixStructureUtils.h"

MatrixStructureUtils::MatrixStructureUtils(MatrixModule* _matrixModule) :
m_matrixModule(_matrixModule) {
}

MatrixStructureUtils::~MatrixStructureUtils() {
}

void MatrixStructureUtils::setSubColumns(MatrixStructure* matrixStructure,
        uintt range[2]) {
    setSubColumns(matrixStructure, range[0], range[1]);
}

void MatrixStructureUtils::setSubColumns(MatrixStructure* matrixStructure,
        uintt begin, uintt end) {
    math::Matrix* matrix = getMatrix(matrixStructure);
    uintt columns = m_matrixModule->getMatrixUtils()->getColumns(matrix);
    if (begin < 0 || begin > columns) {
        begin = 0;
    }
    if (end < 0 || end > columns) {
        end = columns;
    }
    if (begin == 0 && end == 0) {
        begin = 0;
        end = columns;
    }
    if (begin > end) {
        uintt b = end;
        end = b;
        begin = end;
    }
    setBeginColumn(matrixStructure, begin);
    setSubColumns(matrixStructure, end - begin);
}

void MatrixStructureUtils::setSubRows(MatrixStructure* matrixStructure,
        uintt range[2]) {
    setSubRows(matrixStructure, range[0], range[1]);
}

void MatrixStructureUtils::setSubRows(MatrixStructure* matrixStructure,
        uintt begin, uintt end) {
    math::Matrix* matrix = getMatrix(matrixStructure);
    uintt rows = m_matrixModule->getMatrixUtils()->getRows(matrix);
    if (begin < 0 || begin > rows) {
        begin = 0;
    }
    if (end < 0 || end > rows) {
        end = rows;
    }
    if (begin > end) {
        uintt b = end;
        end = b;
        begin = end;
    }
    if (begin == 0 && end == 0) {
        begin = 0;
        end = rows;
    }
    if (begin > end) {
        uintt b = end;
        end = b;
        begin = end;
    }
    setBeginRow(matrixStructure, begin);
    setSubRows(matrixStructure, end - begin);
}

void MatrixStructureUtils::setSub(MatrixStructure* matrixStructure, math::Matrix* matrix,
        uintt subcolumns[2], uintt subrows[2]) {
    setMatrix(matrixStructure, matrix);
    setSubColumns(matrixStructure, subcolumns);
    setSubRows(matrixStructure, subrows);
}

void MatrixStructureUtils::setSub(MatrixStructure* matrixStructure, uintt _subcolumns[2],
        uintt _subrows[2]) {
    setSubColumns(matrixStructure, _subcolumns);
    setSubRows(matrixStructure, _subrows);
}

void MatrixStructureUtils::setMatrix(MatrixStructure* matrixStructure, math::Matrix* matrix) {
    setMatrixToStructure(matrixStructure, matrix);
}