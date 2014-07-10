/* 
 * File:   MatrixAllocator.cpp
 * Author: mmatula
 * 
 * Created on December 13, 2013, 9:16 PM
 */

#include "HostMatrixModules.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <sstream>
#include <linux/fs.h>
#include "ArrayTools.h"
#include "Writer.h"

#define ReIsNotNULL(m) m->reValues != NULL 
#define ImIsNotNULL(m) m->imValues != NULL 

HostMatrixAllocator::HostMatrices HostMatrixAllocator::hostMatrices;
synchronization::Mutex HostMatrixAllocator::mutex;

void _memset(floatt* s, floatt value, int n) {
    for (int fa = 0; fa < n; fa++) {
        memcpy(s + fa, &value, sizeof (floatt));
    }
}

math::Matrix* HostMatrixAllocator::createHostMatrix(math::Matrix* matrix,
        intt columns, intt rows, floatt* values, floatt** valuesPtr) {
    *valuesPtr = values;
    matrix->columns = columns;
    matrix->rows = rows;
    HostMatrixAllocator::mutex.lock();
    HostMatrixAllocator::hostMatrices.push_back(matrix);
    HostMatrixAllocator::mutex.unlock();
    return matrix;
}

HostMatrixCopier::HostMatrixCopier() :
MatrixCopier(&HostMatrixModules::GetInstance()) {
}

HostMatrixCopier::~HostMatrixCopier() {
}

inline void HostMatrixAllocator::initMatrix(math::Matrix* matrix) {
    matrix->columns = 0;
    matrix->rows = 0;
    matrix->imValues = NULL;
    matrix->reValues = NULL;
}

math::Matrix* HostMatrixAllocator::createHostReMatrix(intt columns, intt rows, floatt* values) {
    math::Matrix* matrix = new math::Matrix();
    initMatrix(matrix);
    matrix->columns = columns;
    matrix->rows = rows;
    return createHostMatrix(matrix, columns, rows, values, &matrix->reValues);
}

math::Matrix* HostMatrixAllocator::createHostImMatrix(intt columns, intt rows, floatt* values) {
    math::Matrix* matrix = new math::Matrix();
    initMatrix(matrix);
    return createHostMatrix(matrix, columns, rows, values, &matrix->imValues);
}

void HostMatrixUtils::fillRePart(math::Matrix* output, floatt value) {
    math::Memset(output->reValues, value, output->columns * output->rows);
}

void HostMatrixUtils::fillImPart(math::Matrix* output, floatt value) {
    math::Memset(output->imValues, value, output->columns * output->rows);
}

void HostMatrixUtils::fillMatrix(math::Matrix* output, floatt value) {
    if (output->reValues) {
        fillRePart(output, value);
    }
    if (output->imValues) {
        fillImPart(output, value);
    }
}

math::Matrix* HostMatrixAllocator::newMatrix(intt columns, intt rows, floatt value) {
    math::Matrix* output = new math::Matrix();
    intt length = columns*rows;
    output->columns = columns;
    output->rows = rows;
    output->reValues = new floatt[length];
    output->imValues = new floatt[length];
    hmu.fillRePart(output, value);
    if (output->imValues) {
        hmu.fillImPart(output, value);
    }
    return output;
}

math::Matrix* HostMatrixAllocator::newReMatrix(intt columns, intt rows, floatt value) {
    math::Matrix* output = new math::Matrix();
    intt length = columns*rows;
    output->reValues = new floatt[length];
    output->imValues = NULL;
    output->columns = columns;
    output->rows = rows;
    hmu.fillRePart(output, value);
    return output;
}

math::Matrix* HostMatrixAllocator::newImMatrix(intt columns, intt rows,
        floatt value) {
    math::Matrix* output = new math::Matrix();
    intt length = columns*rows;
    output->columns = columns;
    output->rows = rows;
    output->reValues = NULL;
    output->imValues = new floatt[length];
    hmu.fillImPart(output, value);
    return output;
}

void HostMatrixAllocator::deleteMatrix(math::Matrix* matrix) {
    if (matrix->reValues != NULL) {
        delete[] matrix->reValues;
    }
    if (matrix->imValues != NULL) {
        delete[] matrix->imValues;
    }
    delete matrix;
}

void HostMatrixCopier::copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src) {
    const intt length1 = dst->columns * dst->rows;
    const intt length2 = src->columns * src->rows;
    const intt length = length1 < length2 ? length1 : length2;
    intt bytesLength = length * sizeof (floatt);
    if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
        memcpy(dst->reValues, src->reValues, bytesLength);
    }
    if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
        memcpy(dst->imValues, src->imValues, bytesLength);
    }
}

void HostMatrixCopier::copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src) {
    const intt length1 = dst->columns * dst->rows;
    const intt length2 = src->columns * src->rows;
    const intt length = length1 < length2 ? length1 : length2;
    if (ReIsNotNULL(dst) && ReIsNotNULL(src)) {
        memcpy(dst->reValues, src->reValues, length * sizeof (floatt));
    } else {
    }
}

void HostMatrixCopier::copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src) {
    const intt length1 = dst->columns * dst->rows;
    const intt length2 = src->columns * src->rows;
    const intt length = length1 < length2 ? length1 : length2;
    if (ImIsNotNULL(dst) && ImIsNotNULL(src)) {
        memcpy(dst->imValues, src->imValues, length * sizeof (floatt));
    } else {
    }
}

void HostMatrixCopier::copy(floatt* dst, const floatt* src, intt length) {
    memcpy(dst, src, length * sizeof (floatt));
}

HostMatrixUtils::HostMatrixUtils() :
MatrixUtils(&HostMatrixModules::GetInstance()) {
}

HostMatrixUtils::~HostMatrixUtils() {
}

HostMatrixAllocator::HostMatrixAllocator() :
MatrixAllocator(&HostMatrixModules::GetInstance()) {
}

HostMatrixAllocator::~HostMatrixAllocator() {
}

HostMatrixPrinter::HostMatrixPrinter() :
MatrixPrinter(&HostMatrixModules::GetInstance()) {
}

HostMatrixPrinter::~HostMatrixPrinter() {
}

bool HostMatrixAllocator::isMatrix(math::Matrix* matrix) {
    HostMatrices::iterator it = std::find(hostMatrices.begin(), hostMatrices.end(), matrix);
    return (it != hostMatrices.end());
}
#ifdef MATLAB

void HostMatrixPrinter::getMatrixStr(std::string& str, const math::Matrix* matrix) {
    str = "";
    if (matrix == NULL) {
        return;
    }
    std::stringstream sstream;
    str += "{";
    for (int fb = 0; fb < matrix->rows; fb++) {
        for (int fa = 0; fa < matrix->columns; fa++) {
            if (fa == 0) {
                str += "{";
            }
            sstream << matrix->reValues[fb * matrix->columns + fa];
            str += sstream.str();
            sstream.str("");
            sstream << matrix->imValues[fb * matrix->columns + fa];
            str += "+" + sstream.str() + "i";
            sstream.str("");
            if (fa != matrix->columns - 1) {
                str += ",";
            }
            if (fa == matrix->columns - 1 /*&& fb != matrix->rows - 1*/) {
                str += "}";
            }
        }
    }
    str += "}";
}
#endif

void HostMatrixPrinter::getMatrixStr(std::string& str, const math::Matrix* matrix) {
    str = "";
    if (matrix == NULL) {
        return;
    }
    std::stringstream sstream;
    str += "[";
    for (int fb = 0; fb < matrix->rows; fb++) {
        for (int fa = 0; fa < matrix->columns; fa++) {
            sstream << matrix->reValues[fb * matrix->columns + fa];
            str += "(" + sstream.str();
            sstream.str("");
            sstream << matrix->imValues[fb * matrix->columns + fa];
            str += "," + sstream.str() + "i)";
            sstream.str("");
            if (fa != matrix->columns - 1) {
                str += ",";
            }
            if (fa == matrix->columns - 1 && fb != matrix->rows - 1) {
                str += "\n";
            }
        }
    }
    str += "]";
}

void HostMatrixPrinter::getReMatrixStr(std::string& text, const math::Matrix* matrix) {
    text = "";
    if (matrix == NULL) {
        return;
    }
    std::stringstream sstream;
    text += "\n[";
    char buffer[128];
    memset(buffer, 0, 128 * sizeof (char));
    for (int fb = 0; fb < matrix->rows; fb++) {
        for (int fa = 0; fa < matrix->columns; fa++) {
            sprintf(buffer, "%lf", matrix->reValues[fb * matrix->columns + fa]);
            text += buffer;
            memset(buffer, 0, 128 * sizeof (char));
            if (fa != matrix->columns - 1) {
                text += ",";
            }
            if (fa == matrix->columns - 1 && fb != matrix->rows - 1) {
                text += "\n";
            }
        }
    }
    text += "]";
}

void HostMatrixPrinter::getImMatrixStr(std::string& str, const math::Matrix* matrix) {
    str = "";
    if (matrix == NULL) {
        return;
    }
    std::stringstream sstream;
    str += "[";
    for (int fb = 0; fb < matrix->rows; fb++) {
        for (int fa = 0; fa < matrix->columns; fa++) {
            sstream << matrix->imValues[fb * matrix->columns + fa];
            str += sstream.str();
            sstream.str("");
            if (fa != matrix->columns - 1) {
                str += ",";
            }
            if (fa == matrix->columns - 1 && fb != matrix->rows - 1) {
                str += "\n";
            }
        }
    }
    str += "]";
}

void HostMatrixUtils::getReValues(floatt* dst, math::Matrix* matrix, intt index, intt length) {
    if (matrix->reValues) {
        memcpy(dst, &matrix->reValues[index], length);
    }
}

void HostMatrixUtils::getImValues(floatt* dst, math::Matrix* matrix, intt index, intt length) {
    if (matrix->imValues) {
        memcpy(dst, &matrix->imValues[index], length);
    }
}

void HostMatrixUtils::setReValues(math::Matrix* matrix, floatt* src, intt index, intt length) {
    if (matrix->reValues) {
        memcpy(&matrix->reValues[index], src, length);
    }
}

void HostMatrixUtils::setImValues(math::Matrix* matrix, floatt* src, intt index, intt length) {
    if (matrix->imValues) {
        memcpy(&matrix->imValues[index], src, length);
    }
}

void HostMatrixUtils::setZeroReMatrix(math::Matrix* matrix) {
    if (matrix->reValues) {
        fillRePart(matrix, 0);
    }
}

void HostMatrixUtils::setZeroImMatrix(math::Matrix* matrix) {
    if (matrix->imValues) {
        fillImPart(matrix, 0);
    }
}

void HostMatrixUtils::setDiagonalReMatrix(math::Matrix* matrix, floatt value) {
    if (matrix->reValues) {
        fillRePart(matrix, 0);
        for (int fa = 0; fa < matrix->columns; fa++) {
            matrix->reValues[fa * matrix->columns + fa] = value;
        }
    }
}

void HostMatrixUtils::setDiagonalImMatrix(math::Matrix* matrix, floatt value) {
    if (matrix->imValues) {
        fillImPart(matrix, 0);
        for (int fa = 0; fa < matrix->columns; fa++) {
            matrix->imValues[fa * matrix->columns + fa] = value;
        }
    }
}

void HostMatrixCopier::setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {
    if (matrix->reValues) {
        memcpy(&matrix->reValues[row * matrix->columns], vector, length * sizeof (floatt));
    }
}

void HostMatrixCopier::setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {
    if (matrix->reValues) {
        for (intt fa = 0; fa < length; fa++) {
            matrix->reValues[column + matrix->columns * fa] = vector[fa];
        }
    }
}

void HostMatrixCopier::setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {
    if (matrix->imValues) {
        memcpy(&matrix->imValues[row * matrix->columns], vector, length * sizeof (floatt));
    }
}

void HostMatrixCopier::setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {
    if (matrix->imValues) {
        for (intt fa = 0; fa < length; fa++) {
            matrix->imValues[column + matrix->columns * fa] = vector[fa];
        }
    }
}

void HostMatrixCopier::getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {
    if (matrix->reValues) {
        memcpy(vector, &matrix->reValues[row * matrix->columns], length * sizeof (floatt));
    }
}

void HostMatrixCopier::getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {
    if (matrix->reValues) {
        for (intt fa = 0; fa < length; fa++) {
            vector[fa] = matrix->reValues[column + matrix->columns * fa];
        }
    }
}

void HostMatrixCopier::getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {
    if (matrix->imValues) {
        memcpy(vector, &matrix->imValues[row * matrix->columns], length * sizeof (floatt));
    }
}

void HostMatrixCopier::setVector(math::Matrix* matrix, intt column,
        math::Matrix* vector, uintt rows) {
    setReVector(matrix, column, vector->reValues, rows);
    setImVector(matrix, column, vector->imValues, rows);
}

void HostMatrixCopier::getVector(math::Matrix* vector, uintt rows,
        math::Matrix* matrix, intt column) {
    getReVector(vector->reValues, rows, matrix, column);
    getImVector(vector->imValues, rows, matrix, column);
}

void HostMatrixCopier::getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {
    if (matrix->imValues) {
        for (intt fa = 0; fa < length; fa++) {
            vector[fa] = matrix->imValues[column + matrix->columns * fa];
        }
    }
}

math::Matrix* HostMatrixAllocator::newMatrixFromAsciiFile(const char* path) {
    FILE* file = fopen(path, "r");
    math::Matrix* matrix = NULL;
    int stackCounter = 0;
    if (file) {
        bool is = false;
        floatt* values = NULL;
        int valuesSize = 0;
        intt columns = 0;
        intt rows = 0;
        std::string buffer = "";
        fseek(file, 0, SEEK_END);
        long int size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char* text = new char[size];
        fread(text, size * sizeof (char), 1, file);
        for (int fa = 0; fa < size; fa++) {
            char sign = text[fa];
            if (sign == '[') {
                stackCounter++;
            } else if (sign == ']') {
                columns++;
                is = true;
                stackCounter--;
            } else if (sign == ',') {
                if (is == false) {
                    rows++;
                }
                floatt value = atof(buffer.c_str());
                buffer.clear();
                ArrayTools::add(&values, valuesSize, value);
            } else {
                buffer += sign;
            }
            if (stackCounter == 0) {
                break;
            }
        }
        columns--;
        rows++;
        matrix = createHostReMatrix(columns, rows, values);
        delete[] text;
    }
    return matrix;
}

math::Matrix* HostMatrixAllocator::newMatrixFromBinaryFile(const char* path) {
    FILE* file = fopen(path, "rb");
    math::Matrix* matrix = NULL;
    if (file) {
        int size = 0;
        intt columns = 0;
        intt rows = 0;
        fread(&size, sizeof (int), 1, file);
        fread(&columns, sizeof (int), 1, file);
        fread(&rows, sizeof (int), 1, file);
        matrix = this->newReMatrix(columns, rows);
        fread(matrix->reValues, sizeof (floatt) * columns*rows, 1, file);
        fclose(file);
    }
    return matrix;
}

intt HostMatrixUtils::getColumns(const math::Matrix* matrix) const {
    return matrix->columns;
}

intt HostMatrixUtils::getRows(const math::Matrix* matrix) const {
    return matrix->rows;
}

bool HostMatrixUtils::isMatrix(const math::Matrix* matrix) const {
    return matrix != NULL;
}

bool HostMatrixUtils::isReMatrix(const math::Matrix* matrix) const {
    return matrix != NULL && matrix->reValues != NULL;
}

bool HostMatrixUtils::isImMatrix(const math::Matrix* matrix) const {
    return matrix != NULL && matrix->imValues != NULL;
}

HostMatrixModules::HostMatrixModules() {
}

HostMatrixModules::~HostMatrixModules() {
}

HostMatrixAllocator* HostMatrixModules::getMatrixAllocator() {
    return &hma;
}

HostMatrixCopier* HostMatrixModules::getMatrixCopier() {
    return &hmc;
}

HostMatrixUtils* HostMatrixModules::getMatrixUtils() {
    return &hmu;
}

HostMatrixPrinter* HostMatrixModules::getMatrixPrinter() {
    return &hmp;
}

HostMatrixModules HostMatrixModules::hostMatrixModule;

HostMatrixModules& HostMatrixModules::GetInstance() {
    return HostMatrixModules::hostMatrixModule;
}


namespace host {

    math::Matrix* NewMatrixCopy(const math::Matrix* matrix) {
        math::Matrix* output = NULL;
        if (matrix->reValues && matrix->imValues) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(matrix->columns, matrix->rows);
            HostMatrixModules::GetInstance().getMatrixCopier()->copyMatrixToMatrix(output, matrix);
        } else if (matrix->reValues) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(matrix->columns, matrix->rows);
            HostMatrixModules::GetInstance().getMatrixCopier()->copyReMatrixToReMatrix(output, matrix);
        } else if (matrix->imValues) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(matrix->columns, matrix->rows);
            HostMatrixModules::GetInstance().getMatrixCopier()->copyImMatrixToImMatrix(output, matrix);
        }
        return output;
    }

    math::Matrix* NewMatrix(math::Matrix* matrix, floatt value) {
        math::Matrix* output = NULL;
        if (matrix->reValues != NULL && matrix->imValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(matrix->columns, matrix->rows, value);
        } else if (matrix->reValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(matrix->columns, matrix->rows, value);
        } else if (matrix->imValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(matrix->columns, matrix->rows, value);
        } else {
            return NULL;
        }
    }

    math::Matrix* NewMatrix(math::Matrix* matrix, intt columns, intt rows, floatt value) {
        math::Matrix* output = NULL;
        if (matrix->reValues != NULL && matrix->imValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(columns, rows, value);
        } else if (matrix->reValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(columns, rows, value);
        } else if (matrix->imValues != NULL) {
            output = HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(columns, rows, value);
        } else {
            return NULL;
        }
    }

    math::Matrix* NewMatrixCopy(intt columns, intt rows,
            floatt* reArray, floatt* imArray) {
        math::Matrix* output = HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(columns, rows);
        HostMatrixModules::GetInstance().getMatrixCopier()->copy(output->reValues, reArray, columns * rows);
        HostMatrixModules::GetInstance().getMatrixCopier()->copy(output->imValues, imArray, columns * rows);
        return output;
    }

    math::Matrix* NewReMatrixCopy(intt columns, intt rows, floatt* array) {
        math::Matrix* output = HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(columns, rows);
        HostMatrixModules::GetInstance().getMatrixCopier()->copy(output->reValues, array, columns * rows);
        return output;
    }

    math::Matrix* NewImMatrixCopy(intt columns, intt rows, floatt* array) {
        math::Matrix* output = HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(columns, rows);
        HostMatrixModules::GetInstance().getMatrixCopier()->copy(output->reValues, array, columns * rows);
        return output;
    }

    math::Matrix* NewMatrix(intt columns, intt rows, floatt value) {
        return HostMatrixModules::GetInstance().getMatrixAllocator()->newMatrix(columns, rows, value);
    }

    math::Matrix* NewReMatrix(intt columns, intt rows, floatt value) {
        return HostMatrixModules::GetInstance().getMatrixAllocator()->newReMatrix(columns, rows, value);
    }

    math::Matrix* NewImMatrix(intt columns, intt rows, floatt value) {
        return HostMatrixModules::GetInstance().getMatrixAllocator()->newImMatrix(columns, rows, value);
    }

    void DeleteMatrix(math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixAllocator()->deleteMatrix(matrix);
    }

    floatt GetReValue(const math::Matrix* matrix, intt column, intt row) {
        if (matrix->reValues == NULL) {
            return 0;
        }
        return matrix->reValues[row * matrix->columns + column];
    }

    floatt GetImValue(const math::Matrix* matrix, intt column, intt row) {
        if (matrix->imValues == NULL) {
            return 0;
        }
        return matrix->imValues[row * matrix->columns + column];
    }

    void SetReValue(const math::Matrix* matrix, intt column, intt row, floatt value) {
        if (matrix->reValues) {
            matrix->reValues[row * matrix->columns + column] = value;
        }
    }

    void SetImValue(const math::Matrix* matrix, intt column, intt row, floatt value) {
        if (matrix->imValues) {
            matrix->imValues[row * matrix->columns + column] = value;
        }
    }

    void PrintMatrix(const std::string& text, const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printMatrix(text, matrix);
    }

    void PrintReMatrix(FILE* stream, const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printReMatrix(stream, matrix);
    }

    void PrintReMatrix(const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printReMatrix(matrix);
    }

    void PrintReMatrix(const std::string& text, const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printReMatrix(text, matrix);
    }

    void PrintImMatrix(FILE* stream, const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printImMatrix(stream, matrix);
    }

    void PrintImMatrix(const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printImMatrix(matrix);
    }

    void PrintImMatrix(const std::string& text, const math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixPrinter()->printImMatrix(text, matrix);
    }

    void Copy(math::Matrix* dst, const math::Matrix* src, const SubMatrix& subMatrix,
            intt column, intt row) {
        HostMatrixCopier* copier = HostMatrixModules::GetInstance().getMatrixCopier();
        intt rows = dst->rows;
        intt columns2 = subMatrix.m_columns;
        for (intt fa = 0; fa < rows; fa++) {
            intt fa1 = fa + subMatrix.m_brow;
            if (fa < row) {
                copier->copy(dst->reValues + fa * dst->columns,
                        src->reValues + (fa1) * columns2,
                        column);
                copier->copy(dst->reValues + column + fa * dst->columns,
                        src->reValues + (1 + column) + fa * columns2,
                        (columns2 - column));
            } else if (fa >= row) {
                copier->copy(dst->reValues + fa * dst->columns,
                        &src->reValues[(fa1 + 1) * columns2],
                        column);

                copier->copy(dst->reValues + column + fa * dst->columns,
                        &src->reValues[(fa1 + 1) * columns2 + column + 1],
                        (columns2 - column));
            }
        }
    }

    void Copy(math::Matrix* dst, const math::Matrix* src, intt column, intt row) {
        HostMatrixCopier* copier = HostMatrixModules::GetInstance().getMatrixCopier();
        uintt rows = src->rows;
        uintt columns = src->columns;
        for (uintt fa = 0; fa < rows; fa++) {
            if (fa < row) {
                copier->copy(&dst->reValues [fa * dst->columns],
                        &src->reValues[ fa * columns],
                        column);
                if (column < src->columns - 1) {
                    copier->copy(&dst->reValues [column + fa * dst->columns],
                            &src->reValues[(1 + column) + fa * columns],
                            (src->columns - (column + 1)));
                }
            } else if (fa > row) {
                copier->copy(&dst->reValues [ (fa - 1) * dst->columns],
                        &src->reValues[fa * columns],
                        column);
                if (column < src->columns - 1) {
                    copier->copy(&dst->reValues[column + (fa - 1) * dst->columns],
                            &src->reValues[fa * columns + (column + 1)],
                            (src->columns - (column + 1)));
                }
            }
        }
    }

    void CopyMatrix(math::Matrix* dst, const math::Matrix* src) {

        HostMatrixModules::GetInstance().getMatrixCopier()->copyMatrixToMatrix(dst, src);
    }

    void CopyRe(math::Matrix* dst, const math::Matrix* src) {

        HostMatrixModules::GetInstance().getMatrixCopier()->copyReMatrixToReMatrix(dst, src);
    }

    void CopyIm(math::Matrix* dst, const math::Matrix* src) {

        HostMatrixModules::GetInstance().getMatrixCopier()->copyImMatrixToImMatrix(dst, src);
    }

    void SetReVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {

        HostMatrixModules::GetInstance().getMatrixCopier()->setReVector(matrix, column, vector, length);
    }

    void SetTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {

        HostMatrixModules::GetInstance().getMatrixCopier()->setTransposeReVector(matrix, row, vector, length);
    }

    void SetImVector(math::Matrix* matrix, intt column, floatt* vector, intt length) {

        HostMatrixModules::GetInstance().getMatrixCopier()->setImVector(matrix, column, vector, length);
    }

    void SetTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length) {

        HostMatrixModules::GetInstance().getMatrixCopier()->setTransposeImVector(matrix, row, vector, length);
    }

    void SetReVector(math::Matrix* matrix, intt column, floatt* vector) {

        SetReVector(matrix, column, vector, matrix->rows);
    }

    void SetTransposeReVector(math::Matrix* matrix, intt row, floatt* vector) {

        SetTransposeReVector(matrix, row, vector, matrix->columns);
    }

    void SetImVector(math::Matrix* matrix, intt column, floatt* vector) {

        SetImVector(matrix, column, vector, matrix->rows);
    }

    void SetTransposeImVector(math::Matrix* matrix, intt row, floatt* vector) {

        SetTransposeImVector(matrix, row, vector, matrix->columns);
    }

    void GetReMatrixStr(std::string& text, const math::Matrix* matrix) {

        HostMatrixModules::GetInstance().getMatrixPrinter()->getReMatrixStr(text, matrix);
    }

    void GetImMatrixStr(std::string& text, const math::Matrix* matrix) {

        HostMatrixModules::GetInstance().getMatrixPrinter()->getImMatrixStr(text, matrix);
    }

    void GetReVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {

        HostMatrixModules::GetInstance().getMatrixCopier()->getReVector(vector, length, matrix, column);
    }

    void GetTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {

        HostMatrixModules::GetInstance().getMatrixCopier()->getTransposeReVector(vector, length, matrix, row);
    }

    void GetImVector(floatt* vector, intt length, math::Matrix* matrix, intt column) {

        HostMatrixModules::GetInstance().getMatrixCopier()->getImVector(vector, length, matrix, column);
    }

    void GetTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row) {

        HostMatrixModules::GetInstance().getMatrixCopier()->getTransposeImVector(vector, length, matrix, row);
    }

    void GetReVector(floatt* vector, math::Matrix* matrix, intt column) {

        GetReVector(vector, matrix->rows, matrix, column);
    }

    void GetTransposeReVector(floatt* vector, math::Matrix* matrix, intt row) {

        GetTransposeReVector(vector, matrix->columns, matrix, row);
    }

    void GetImVector(floatt* vector, math::Matrix* matrix, intt column) {

        GetImVector(vector, matrix->rows, matrix, column);
    }

    void GetTransposeImVector(floatt* vector, math::Matrix* matrix, intt row) {

        GetTransposeReVector(vector, matrix->columns, matrix, row);
    }

    floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1) {
        floatt diff = matrix->reValues[0] - matrix1->reValues[0];
        for (intt fa = 0; fa < matrix->columns; fa++) {
            for (intt fb = 0; fb < matrix->rows; fb++) {
                intt index = fa + fb * matrix->columns;
                floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
                if (diff1 < 0) {
                    diff1 = -diff1;
                }
                if (diff > diff1) {
                    diff = diff1;
                }
            }
        }
        return diff;
    }

    floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1) {
        floatt diff = matrix->reValues[0] - matrix1->reValues[0];
        for (intt fa = 0; fa < matrix->columns; fa++) {
            for (intt fb = 0; fb < matrix->rows; fb++) {
                intt index = fa + fb * matrix->columns;
                floatt diff1 = matrix->reValues[index] - matrix1->reValues[index];
                if (diff1 < 0) {
                    diff1 = -diff1;
                }
                if (diff < diff1) {
                    diff = diff1;
                }
            }
        }
        return diff;
    }

    void SetIdentity(math::Matrix* matrix) {
        HostMatrixModules::GetInstance().getMatrixUtils()->setIdentityMatrix(matrix);
    }

    void SetReZero(math::Matrix* matrix) {
        if (matrix->reValues) {
            memset(matrix->reValues, 0, matrix->columns * matrix->rows * sizeof (floatt));
        }
    }

    void SetImZero(math::Matrix* matrix) {
        if (matrix->imValues) {
            memset(matrix->imValues, 0, matrix->columns * matrix->rows * sizeof (floatt));
        }
    }

    bool IsEquals(math::Matrix* transferMatrix2, math::Matrix* transferMatrix1,
            floatt diff) {
        for (intt fa = 0; fa < transferMatrix2->columns; fa++) {
            for (intt fb = 0; fb < transferMatrix2->rows; fb++) {
                floatt p = transferMatrix2->reValues[fa + transferMatrix2->columns * fb] -
                        transferMatrix1->reValues[fa + transferMatrix1->columns * fb];
                if (p < -diff || p > diff) {
                    return false;
                }
            }
        }
        return true;
    }

    floatt GetTrace(math::Matrix* matrix) {
        floatt o = 1.;
        for (uintt fa = 0; fa < matrix->columns; ++fa) {
            floatt v = matrix->reValues[fa * matrix->columns + fa];
            if (-MATH_VALUE_LIMIT < v && v < MATH_VALUE_LIMIT) {
                v = 0;
            }
            o = o * v;
        }
        return o;
    }

    void SetReDiagonals(math::Matrix* matrix, floatt a) {
        HostMatrixModules::GetInstance().getMatrixUtils()->setDiagonalReMatrix(matrix, a);
    }

    char* load(const char* path, uintt& _size) {
        FILE* file = fopen(path, "r");
        fseek(file, 0, SEEK_END);
        long int size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char* buffer = new char[size + 1];
        buffer[size] = 0;
        fread(buffer, size, 1, file);
        fclose(file);
        _size = size;
        return buffer;
    }

    void loadFloats(floatt* values, uintt count,
            char* data, unsigned int size, char separator, uintt skip) {
        char* ptr = &data[0];
        uintt index = 0;
        uintt index1 = 0;
        bool c = false;
        if (skip == index1) {
            c = true;
        }
        for (uint fa = 0; fa < size; ++fa) {
            if (data[fa] == separator) {
                char* ptr1 = &data[fa];
                if (c) {
                    std::string s(ptr, ptr1 - ptr);
                    floatt f = atof(s.c_str());
                    values[index] = f;
                }
                ptr = &data[fa + 1];
                index++;
                if (index == count) {
                    index = 0;
                    index1++;
                    if (skip == index1) {
                        c = true;
                    } else if (skip + 1 == index1) {
                        return;
                    }
                }
            }
        }
    }

    math::Matrix* LoadMatrix(uintt columns, uintt rows,
            const char* repath, const char* impath) {
        math::Matrix* matrix = NewMatrix(columns, rows, 0);
        LoadMatrix(matrix, repath, impath);
        return matrix;
    }

    void LoadMatrix(math::Matrix* matrix,
            const char* repath, const char* impath) {
        LoadMatrix(matrix, repath, impath, 0);
    }

    void LoadMatrix(math::Matrix* matrix,
            const char* repath, const char* impath, uintt skipCount) {
        if (NULL != matrix) {
            uintt length = matrix->columns * matrix->rows;
            uintt size;
            char* b = NULL;
            if (NULL != repath) {
                b = load(repath, size);
                loadFloats(matrix->reValues, length, b, size, ',', skipCount);
                delete[] b;
            }
            if (NULL != impath) {
                b = load(impath, size);
                loadFloats(matrix->imValues, length, b, size, ',', skipCount);
                delete[] b;
            }
        }
    }

};
