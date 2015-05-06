#ifndef OGLA_HOST_MATRIX_UTILS_H
#define	OGLA_HOST_MATRIX_UTILS_H
#include "MatrixModules.h"
#include "Matrix.h"
#include <stdio.h>
#include "ThreadUtils.h"

class HostMatrixUtils : public MatrixUtils {
public:
    HostMatrixUtils();
    ~HostMatrixUtils();
    void getReValues(floatt* dst, math::Matrix* matrix, uintt index, uintt length);
    void getImValues(floatt* dst, math::Matrix* matrix, uintt index, uintt length);
    void setReValues(math::Matrix* matrix, floatt* src, uintt index, uintt length);
    void setImValues(math::Matrix* matrix, floatt* src, uintt index, uintt length);
    void setDiagonalReMatrix(math::Matrix* matrix, floatt a);
    void setDiagonalImMatrix(math::Matrix* matrix, floatt a);
    void setZeroReMatrix(math::Matrix* matrix);
    void setZeroImMatrix(math::Matrix* matrix);
    uintt getColumns(const math::Matrix* matrix) const;
    uintt getRows(const math::Matrix* matrix) const;
    bool isMatrix(const math::Matrix* matrix) const;
    bool isReMatrix(const math::Matrix* matrix) const;
    bool isImMatrix(const math::Matrix* matrix) const;
};

class HostMatrixCopier : public MatrixCopier {
    HostMatrixUtils hmu;
public:
    HostMatrixCopier();
    virtual ~HostMatrixCopier();
    void copyMatrixToMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyReMatrixToReMatrix(math::Matrix* dst, const math::Matrix* src);
    void copyImMatrixToImMatrix(math::Matrix* dst, const math::Matrix* src);
    /**
     * Copy floatts where length is number of floatts (not bytes!).
     * @param dst
     * @param src
     * @param length number of numbers to copy
     */
    void copy(floatt* dst, const floatt* src, uintt length);


    void setReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void setImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void setTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);

    void getReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void getImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void getTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void setVector(math::Matrix* matrix, uintt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, uintt column);
};

class HostMatrixAllocator : public MatrixAllocator {
    HostMatrixUtils hmu;
    HostMatrixCopier hmc;
    typedef std::vector<math::Matrix*> HostMatrices;
    static HostMatrices hostMatrices;
    static utils::sync::Mutex mutex;
    static math::Matrix* createHostMatrix(math::Matrix* matrix, uintt columns, uintt rows, floatt* values, floatt** valuesPtr);
    static void initMatrix(math::Matrix* matrix);
    static math::Matrix* createHostReMatrix(uintt columns, uintt rows, floatt* values);
    static math::Matrix* createHostImMatrix(uintt columns, uintt rows, floatt* values);
public:
    HostMatrixAllocator();
    ~HostMatrixAllocator();
    math::Matrix* newReMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* newImMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* newMatrix(uintt columns, uintt rows, floatt value = 0);
    bool isMatrix(math::Matrix* matrix);
    math::Matrix* newMatrixFromAsciiFile(const char* path);
    math::Matrix* newMatrixFromBinaryFile(const char* path);
    void deleteMatrix(math::Matrix* matrix);
};

class HostMatrixPrinter : public MatrixPrinter {
public:
    HostMatrixPrinter();
    ~HostMatrixPrinter();
    void getMatrixStr(std::string& str, const math::Matrix* matrix);
    void getReMatrixStr(std::string& str, const math::Matrix* matrix);
    void getImMatrixStr(std::string& str, const math::Matrix* matrix);
};

class HostMatrixModules : public MatrixModule {
    HostMatrixAllocator* m_hma;
    HostMatrixCopier* m_hmc;
    HostMatrixUtils* m_hmu;
    HostMatrixPrinter* m_hmp;
    static HostMatrixModules* hostMatrixModule;
protected:
    HostMatrixModules();
    virtual ~HostMatrixModules();
public:
    static HostMatrixModules* GetInstance();
    HostMatrixAllocator* getMatrixAllocator();
    HostMatrixCopier* getMatrixCopier();
    HostMatrixUtils* getMatrixUtils();
    HostMatrixPrinter* getMatrixPrinter();
};


namespace host {

    class SubMatrix {
    public:

        SubMatrix() : m_bcolum(0), m_brow(0),
        m_columns(0), m_rows(0) {
        }

        SubMatrix(uintt bcolumn, uintt brow, uintt columns, uintt rows) :
        m_bcolum(bcolumn), m_brow(brow),
        m_columns(columns), m_rows(rows) {
        }

        SubMatrix(math::Matrix* matrix) {
            m_bcolum = 0;
            m_brow = 0;
            m_columns = matrix->columns;
            m_rows = matrix->rows;
        }

        uintt m_bcolum;
        uintt m_brow;
        uintt m_columns;
        uintt m_rows;
    };
    math::Matrix* NewMatrixCopy(const math::Matrix* matrix);
    math::Matrix* NewMatrixCopy(uintt columns, uintt rows,
            floatt* reArray, floatt* imArray);
    math::Matrix* NewReMatrixCopy(uintt columns, uintt rows, floatt* reArray);
    math::Matrix* NewImMatrixCopy(uintt columns, uintt rows, floatt* imArray);
    math::Matrix* NewMatrix(math::Matrix* matrix, floatt value);
    math::Matrix* NewMatrix(math::Matrix* matrix, uintt columns, uintt rows, floatt value);
    math::Matrix* NewMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* NewMatrix(bool isre, bool isim, uintt columns, uintt rows, floatt value = 0);
    math::Matrix* NewReMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* NewImMatrix(uintt columns, uintt rows, floatt value = 0);
    math::Matrix* NewMatrix(const std::string& text);
    void CopyMatrix(math::Matrix* dst, const math::Matrix* src);
    /**
     * Copy data to dst matrix which has one column and row less than
     * src matrix. Row and column which will be omitted are added as params..
     * @param dst
     * @param src
     * @param column index of column which will be omitted
     * @param row index of row which will be omitted
     */
    void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row);
    void Copy(math::Matrix* dst, const math::Matrix* src, const SubMatrix& subMatrix,
            uintt column, uintt row);
    void CopyRe(math::Matrix* dst, const math::Matrix* src);
    void CopyIm(math::Matrix* dst, const math::Matrix* src);

    void DeleteMatrix(math::Matrix* matrix);
    floatt GetReValue(const math::Matrix* matrix, uintt column, uintt row);
    void SetReValue(const math::Matrix* matrix, uintt column, uintt row, floatt value);
    floatt GetImValue(const math::Matrix* matrix, uintt column, uintt row);
    void SetImValue(const math::Matrix* matrix, uintt column, uintt row, floatt value);
    void GetReMatrixStr(std::string& text, const math::Matrix* matrix);
    void GetImMatrixStr(std::string& text, const math::Matrix* matrix);
    void PrintReMatrix(FILE* stream, const math::Matrix* matrix);
    void PrintReMatrix(const math::Matrix* matrix);
    void PrintReMatrix(const std::string& text, const math::Matrix* matrix);
    void PrintMatrix(const std::string& text, const math::Matrix* matrix);
    void PrintMatrix(const math::Matrix* matrix);
    void PrintImMatrix(FILE* stream, const math::Matrix* matrix);
    void PrintImMatrix(const math::Matrix* matrix);
    void PrintImMatrix(const std::string& text, const math::Matrix* matrix);
    void SetReVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void SetImVector(math::Matrix* matrix, uintt column, floatt* vector, uintt length);
    void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector, uintt length);
    void SetReVector(math::Matrix* matrix, uintt column, floatt* vector);
    void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector);
    void SetImVector(math::Matrix* matrix, uintt column, floatt* vector);
    void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector);
    void GetReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void GetTransposeReVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void GetImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt column);
    void GetTransposeImVector(floatt* vector, uintt length, math::Matrix* matrix, uintt row);
    void GetReVector(floatt* vector, math::Matrix* matrix, uintt column);
    void GetTransposeReVector(floatt* vector, math::Matrix* matrix, uintt row);
    void GetImVector(floatt* vector, math::Matrix* matrix, uintt column);
    void GetTransposeImVector(floatt* vector, math::Matrix* matrix, uintt row);
    void SetIdentity(math::Matrix* matrix);
    void SetDiagonalReMatrix(math::Matrix* matrix, floatt a);
    void SetIdentityMatrix(math::Matrix* matrix);
    floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1);
    floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1);
    floatt GetTrace(math::Matrix* matrix);
    void SetReZero(math::Matrix* matrix);
    void SetImZero(math::Matrix* matrix);
    void SetZero(math::Matrix* matrix);
    bool IsEquals(math::Matrix* matrix, math::Matrix* matrix1, floatt diff = 0.1);

    math::Matrix* LoadMatrix(uintt columns, uintt rows,
            const char* repath, const char* impath);

    void LoadMatrix(math::Matrix* matrix,
            const char* repath, const char* impath);

    void LoadMatrix(math::Matrix* matrix,
            const char* repath, const char* impath, uintt skipCount);

    void SetSubs(math::Matrix* matrix, uintt subcolumns, uintt subrows);
    void SetSubColumns(math::Matrix* matrix, uintt subcolumns);
    void SetSubRows(math::Matrix* matrix, uintt subrows);
    void SetSubsSafe(math::Matrix* matrix, uintt subcolumns, uintt subrows);
    void SetSubColumnsSafe(math::Matrix* matrix, uintt subcolumns);
    void SetSubRowsSafe(math::Matrix* matrix, uintt subrows);
    
};


#endif	/* MATRIXALLOCATOR_H */

