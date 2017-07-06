/*
 * Copyright 2016, 2017 Marcin Matula
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



#ifndef OAP_MATRIX_MODULE_H
#define	OAP_MATRIX_MODULE_H
#include "Module.h"
#include "Matrix.h"
#include "Math.h"

class MatrixModule;

class MatrixAllocator : public utils::Module {
    MatrixModule* m_matrixModule;
public:
    MatrixAllocator(MatrixModule* matrixModule);
    virtual ~MatrixAllocator();
    virtual math::Matrix* newReMatrix(uintt columns, uintt rows, floatt value = 0) = 0;
    virtual math::Matrix* newImMatrix(uintt columns, uintt rows, floatt value = 0) = 0;
    virtual math::Matrix* newMatrix(uintt columns, uintt rows, floatt value = 0) = 0;
    virtual math::Matrix* newReValue(floatt value = 0);
    virtual math::Matrix* newImValue(floatt value = 0);
    virtual math::Matrix* newValue(floatt value = 0);
    virtual bool isMatrix(math::Matrix* matrix) = 0;
    virtual math::Matrix* newMatrixFromAsciiFile(const char* path) = 0;
    virtual math::Matrix* newMatrixFromBinaryFile(const char* path) = 0;
    virtual void deleteMatrix(math::Matrix* matrix) = 0;
};

class MatrixCopier : public utils::Module {
    MatrixModule* m_matrixModule;
public:
    MatrixCopier(MatrixModule* matrixModule);
    virtual ~MatrixCopier();
    virtual void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void copy(floatt* dst, const floatt* src, uintt length) = 0;

    virtual void setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) = 0;
    virtual void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) = 0;
    virtual void setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length) = 0;
    virtual void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length) = 0;

    virtual void getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) = 0;
    virtual void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) = 0;
    virtual void getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column) = 0;
    virtual void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row) = 0;

    virtual void setVector(math::Matrix* matrix, uintt column,
            math::Matrix* vector, uintt rows) = 0;
    virtual void getVector(math::Matrix* vector, uintt rows,
            math::Matrix* matrix, uintt column) = 0;

};

class MatrixUtils : public utils::Module {
    MatrixModule* m_matrixModule;
public:
    MatrixUtils(MatrixModule* matrixModule);
    virtual ~MatrixUtils();
    void setIdentityMatrix(math::Matrix* matrix);
    void setIdentityReMatrix(math::Matrix* matrix);
    void setIdentityImMatrix(math::Matrix* matrix);
    void setDiagonalMatrix(math::Matrix* matrix, floatt value);
    void setDiagonalMatrix(math::Matrix* matrix, floatt revalue,floatt imvalue);
    virtual void setDiagonalReMatrix(math::Matrix* matrix, floatt value) = 0;
    virtual void setDiagonalImMatrix(math::Matrix* matrix, floatt value) = 0;
    void setZeroMatrix(math::Matrix* matrix);
    virtual void setZeroReMatrix(math::Matrix* matrix) = 0;
    virtual void setZeroImMatrix(math::Matrix* matrix) = 0;
    virtual uintt getColumns(const math::Matrix* matrix) const = 0;
    virtual uintt getRows(const math::Matrix* matrix) const = 0;
    bool isMatrix(const math::Matrix* matrix) const;
    virtual bool isReMatrix(const math::Matrix* matrix) const = 0;
    virtual bool isImMatrix(const math::Matrix* matrix) const = 0;
};

class MatrixPrinter : public utils::Module {
    MatrixModule* m_matrixModule;
public:
    MatrixPrinter(MatrixModule* matrixModule);
    ~MatrixPrinter();
    virtual void getMatrixStr(std::string& str, const math::Matrix* matrix) = 0;
    virtual void getReMatrixStr(std::string& str, const math::Matrix* matrix) = 0;
    virtual void getImMatrixStr(std::string& str, const math::Matrix* matrix) = 0;
    virtual void printMatrix(FILE* stream, const math::Matrix* matrix);
    virtual void printMatrix(const std::string& text, const math::Matrix* matrix);
    virtual void printReMatrix(FILE* stream, const math::Matrix* matrix);
    virtual void printReMatrix(const math::Matrix* matrix);
    virtual void printReMatrix(const std::string& text, const math::Matrix* matrix);
    virtual void printImMatrix(FILE* stream, const math::Matrix* matrix);
    virtual void printImMatrix(const math::Matrix* matrix);
    virtual void printImMatrix(const std::string& text, const math::Matrix* matrix);
};

class MatrixModule : public utils::Module {
public:
    MatrixModule();
    virtual ~MatrixModule();
    virtual MatrixAllocator* getMatrixAllocator() = 0;
    virtual MatrixCopier* getMatrixCopier() = 0;
    virtual MatrixUtils* getMatrixUtils() = 0;
    virtual MatrixPrinter* getMatrixPrinter() = 0;
    math::Matrix* newMatrix(math::Matrix* matrix);
    math::Matrix* newMatrix(math::Matrix* matrix, uintt columns, uintt rows);
    void deleteMatrix(math::Matrix* matrix);
};


#endif	/* MATRIXMODULE_H */
