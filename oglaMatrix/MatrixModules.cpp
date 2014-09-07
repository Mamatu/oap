/* 
 * File:   MatrixModule.cpp
 * Author: mmatula
 * 
 * Created on March 22, 2014, 12:27 PM
 */

#include "MatrixModules.h"
#include <stdio.h>

MatrixAllocator::MatrixAllocator(MatrixModule* matrixModule) :
m_matrixModule(matrixModule) {
}

MatrixAllocator::~MatrixAllocator() {
}

MatrixCopier::MatrixCopier(MatrixModule* matrixModule) :
m_matrixModule(matrixModule) {
}

MatrixCopier::~MatrixCopier() {
}

MatrixUtils::MatrixUtils(MatrixModule* matrixModule) :
m_matrixModule(matrixModule) {
}

MatrixUtils::~MatrixUtils() {
}

MatrixModule::MatrixModule() {
}

MatrixModule::~MatrixModule() {
}

MatrixPrinter::MatrixPrinter(MatrixModule* matrixModule) :
m_matrixModule(matrixModule) {
}

MatrixPrinter::~MatrixPrinter() {
}

bool MatrixUtils::isMatrix(const math::Matrix* matrix) const {
    return this->isReMatrix(matrix) && this->isImMatrix(matrix);
}

void MatrixPrinter::printMatrix(FILE* stream, const math::Matrix* matrix) {
    if (m_matrixModule->getMatrixUtils()->isMatrix(matrix)) {
        std::string matrixStr;
        getMatrixStr(matrixStr, matrix);
        fprintf(stream, "%s\n", matrixStr.c_str());
    } else if (m_matrixModule->getMatrixUtils()->isReMatrix(matrix)) {
        printReMatrix(matrix);
    } else if (m_matrixModule->getMatrixUtils()->isImMatrix(matrix)) {
        printImMatrix(matrix);
    }
}

void MatrixPrinter::printMatrix(const std::string& text, const math::Matrix* matrix) {
    if (m_matrixModule->getMatrixUtils()->isMatrix(matrix)) {
        std::string matrixStr;
        getMatrixStr(matrixStr, matrix);
        fprintf(stdout, "%s %s\n", text.c_str(), matrixStr.c_str());
    } else if (m_matrixModule->getMatrixUtils()->isReMatrix(matrix)) {
        printReMatrix(text, matrix);
    } else if (m_matrixModule->getMatrixUtils()->isImMatrix(matrix)) {
        printImMatrix(text, matrix);
    }
}

void MatrixPrinter::printReMatrix(FILE* stream, const math::Matrix* matrix) {
    std::string matrixStr;
    getReMatrixStr(matrixStr, matrix);
    fprintf(stream, "%s\n", matrixStr.c_str());
}

void MatrixPrinter::printReMatrix(const math::Matrix* matrix) {
    std::string matrixStr;
    getReMatrixStr(matrixStr, matrix);
    fprintf(stdout, "%s\n", matrixStr.c_str());
}

void MatrixPrinter::printReMatrix(const std::string& text, const math::Matrix* matrix) {
    std::string matrixStr;
    getReMatrixStr(matrixStr, matrix);
    fprintf(stdout, "%s", text.c_str());
    fprintf(stdout, "%s\n", matrixStr.c_str());
}

void MatrixPrinter::printImMatrix(FILE* stream, const math::Matrix* matrix) {
    std::string matrixStr;
    getImMatrixStr(matrixStr, matrix);
    fprintf(stream, "%s\n", matrixStr.c_str());
}

void MatrixPrinter::printImMatrix(const math::Matrix* matrix) {
    std::string matrixStr;
    getImMatrixStr(matrixStr, matrix);
    fprintf(stdout, "%s\n", matrixStr.c_str());
}

void MatrixPrinter::printImMatrix(const std::string& text, const math::Matrix* matrix) {
    std::string matrixStr;
    getImMatrixStr(matrixStr, matrix);
    fprintf(stdout, "%s", text.c_str());
    fprintf(stdout, "%s\n", matrixStr.c_str());
}

math::Matrix* MatrixAllocator::newReValue(floatt value) {
    return newReMatrix(1, 1, value);
}

math::Matrix* MatrixAllocator::newImValue(floatt value) {
    return newImMatrix(1, 1, value);
}

math::Matrix* MatrixAllocator::newValue(floatt value) {
    return newMatrix(1, 1, value);
}

math::Matrix* MatrixModule::newMatrix(math::Matrix* matrix,
        intt columns,
        intt rows) {
    math::Matrix* output = NULL;
    if (this->getMatrixUtils()->isMatrix(matrix)) {
        output = this->getMatrixAllocator()->newMatrix(columns, rows);
        this->getMatrixUtils()->setZeroMatrix(output);
    } else if (this->getMatrixUtils()->isReMatrix(matrix)) {
        output = this->getMatrixAllocator()->newReMatrix(columns, rows);
        this->getMatrixUtils()->setZeroReMatrix(output);
    } else if (this->getMatrixUtils()->isImMatrix(matrix)) {
        output = this->getMatrixAllocator()->newImMatrix(columns, rows);
        this->getMatrixUtils()->setZeroImMatrix(output);
    }
    return output;
}

math::Matrix* MatrixModule::newMatrix(math::Matrix* matrix) {
    math::Matrix* output = NULL;
    uintt columns = this->getMatrixUtils()->getColumns(matrix);
    uintt rows = this->getMatrixUtils()->getRows(matrix);
    if (this->getMatrixUtils()->isMatrix(matrix)) {
        output = this->getMatrixAllocator()->newMatrix(columns, rows);
        this->getMatrixCopier()->copyMatrixToMatrix(output, matrix);
    } else if (this->getMatrixUtils()->isReMatrix(matrix)) {
        output = this->getMatrixAllocator()->newReMatrix(columns, rows);
        this->getMatrixCopier()->copyReMatrixToReMatrix(output, matrix);
    } else if (this->getMatrixUtils()->isImMatrix(matrix)) {
        output = this->getMatrixAllocator()->newImMatrix(columns, rows);
        this->getMatrixCopier()->copyImMatrixToImMatrix(output, matrix);
    }
    return output;
}

void MatrixModule::deleteMatrix(math::Matrix* matrix) {
    this->getMatrixAllocator()->deleteMatrix(matrix);
}

void MatrixUtils::setZeroMatrix(math::Matrix* matrix) {
    this->setZeroReMatrix(matrix);
    this->setZeroImMatrix(matrix);
}

void MatrixUtils::setIdentityMatrix(math::Matrix* matrix) {
    setDiagonalReMatrix(matrix, 1);
    setZeroImMatrix(matrix);
}

void MatrixUtils::setIdentityReMatrix(math::Matrix* matrix) {
    setDiagonalReMatrix(matrix, 1);
}

void MatrixUtils::setIdentityImMatrix(math::Matrix* matrix) {
    setDiagonalImMatrix(matrix, 1);
}

void MatrixUtils::setDiagonalMatrix(math::Matrix* matrix, floatt value) {
    setDiagonalReMatrix(matrix, value);
    setDiagonalImMatrix(matrix, value);
}

void MatrixUtils::setDiagonalMatrix(math::Matrix* matrix, floatt revalue, floatt imvalue) {
    setDiagonalReMatrix(matrix, revalue);
    setDiagonalImMatrix(matrix, imvalue);
}