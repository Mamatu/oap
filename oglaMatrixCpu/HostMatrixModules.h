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
    void getReValues(floatt* dst, math::Matrix* matrix, intt index, intt length);
    void getImValues(floatt* dst, math::Matrix* matrix, intt index, intt length);
    void setReValues(math::Matrix* matrix, floatt* src, intt index, intt length);
    void setImValues(math::Matrix* matrix, floatt* src, intt index, intt length);
    void setDiagonalReMatrix(math::Matrix* matrix, floatt a);
    void setDiagonalImMatrix(math::Matrix* matrix, floatt a);
    void setZeroReMatrix(math::Matrix* matrix);
    void setZeroImMatrix(math::Matrix* matrix);
    intt getColumns(const math::Matrix* matrix) const;
    intt getRows(const math::Matrix* matrix) const;
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
    void copy(floatt* dst, const floatt* src, intt length);


    void setReVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void setImVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void setTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length);

    void getReVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void getImVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void getTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void setVector(math::Matrix* matrix, intt column, math::Matrix* vector, uintt rows);
    void getVector(math::Matrix* vector, uintt rows, math::Matrix* matrix, intt column);
};

class HostMatrixAllocator : public MatrixAllocator {
    HostMatrixUtils hmu;
    HostMatrixCopier hmc;
    typedef std::vector<math::Matrix*> HostMatrices;
    static HostMatrices hostMatrices;
    static synchronization::Mutex mutex;
    static math::Matrix* createHostMatrix(math::Matrix* matrix, intt columns, intt rows, floatt* values, floatt** valuesPtr);
    static void initMatrix(math::Matrix* matrix);
    static math::Matrix* createHostReMatrix(intt columns, intt rows, floatt* values);
    static math::Matrix* createHostImMatrix(intt columns, intt rows, floatt* values);
public:
    HostMatrixAllocator();
    ~HostMatrixAllocator();
    math::Matrix* newReMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* newImMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* newMatrix(intt columns, intt rows, floatt value = 0);
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
    HostMatrixAllocator hma;
    HostMatrixCopier hmc;
    HostMatrixUtils hmu;
    HostMatrixPrinter hmp;
    static HostMatrixModules hostMatrixModule;
protected:
    HostMatrixModules();
    virtual ~HostMatrixModules();
public:
    static HostMatrixModules& GetInstance();
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

        SubMatrix(intt bcolumn, intt brow, intt columns, intt rows) :
        m_bcolum(bcolumn), m_brow(brow),
        m_columns(columns), m_rows(rows) {
        }

        SubMatrix(math::Matrix* matrix) {
            m_bcolum = 0;
            m_brow = 0;
            m_columns = matrix->columns;
            m_rows = matrix->rows;
        }

        intt m_bcolum;
        intt m_brow;
        intt m_columns;
        intt m_rows;
    };
    math::Matrix* NewMatrixCopy(const math::Matrix* matrix);
    math::Matrix* NewMatrixCopy(intt columns, intt rows,
            floatt* reArray, floatt* imArray);
    math::Matrix* NewReMatrixCopy(intt columns, intt rows, floatt* reArray);
    math::Matrix* NewImMatrixCopy(intt columns, intt rows, floatt* imArray);
    math::Matrix* NewMatrixCopy(intt columns, intt rows);
    math::Matrix* NewMatrix(math::Matrix* matrix, floatt value);
    math::Matrix* NewMatrix(math::Matrix* matrix, intt columns, intt rows, floatt value);
    math::Matrix* NewMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* NewReMatrix(intt columns, intt rows, floatt value = 0);
    math::Matrix* NewImMatrix(intt columns, intt rows, floatt value = 0);
    void CopyMatrix(math::Matrix* dst, const math::Matrix* src);
    /**
     * Copy data to dst matrix which has one column and row less than
     * src matrix. Row and column which will be omitted are added as params..
     * @param dst
     * @param src
     * @param column index of column which will be omitted
     * @param row index of row which will be omitted
     */
    void Copy(math::Matrix* dst, const math::Matrix* src, intt column, intt row);
    void Copy(math::Matrix* dst, const math::Matrix* src, const SubMatrix& subMatrix,
            intt column, intt row);
    void CopyRe(math::Matrix* dst, const math::Matrix* src);
    void CopyIm(math::Matrix* dst, const math::Matrix* src);

    void DeleteMatrix(math::Matrix* matrix);
    floatt GetReValue(const math::Matrix* matrix, intt column, intt row);
    void SetReValue(const math::Matrix* matrix, intt column, intt row, floatt value);
    floatt GetImValue(const math::Matrix* matrix, intt column, intt row);
    void SetImValue(const math::Matrix* matrix, intt column, intt row, floatt value);
    void GetReMatrixStr(std::string& text, const math::Matrix* matrix);
    void GetImMatrixStr(std::string& text, const math::Matrix* matrix);
    void PrintReMatrix(FILE* stream, const math::Matrix* matrix);
    void PrintReMatrix(const math::Matrix* matrix);
    void PrintReMatrix(const std::string& text, const math::Matrix* matrix);
    void PrintMatrix(const std::string& text, const math::Matrix* matrix);
    void PrintImMatrix(FILE* stream, const math::Matrix* matrix);
    void PrintImMatrix(const math::Matrix* matrix);
    void PrintImMatrix(const std::string& text, const math::Matrix* matrix);
    void SetReVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void SetTransposeReVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void SetImVector(math::Matrix* matrix, intt column, floatt* vector, intt length);
    void SetTransposeImVector(math::Matrix* matrix, intt row, floatt* vector, intt length);
    void SetReVector(math::Matrix* matrix, intt column, floatt* vector);
    void SetTransposeReVector(math::Matrix* matrix, intt row, floatt* vector);
    void SetImVector(math::Matrix* matrix, intt column, floatt* vector);
    void SetTransposeImVector(math::Matrix* matrix, intt row, floatt* vector);
    void GetReVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void GetTransposeReVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void GetImVector(floatt* vector, intt length, math::Matrix* matrix, intt column);
    void GetTransposeImVector(floatt* vector, intt length, math::Matrix* matrix, intt row);
    void GetReVector(floatt* vector, math::Matrix* matrix, intt column);
    void GetTransposeReVector(floatt* vector, math::Matrix* matrix, intt row);
    void GetImVector(floatt* vector, math::Matrix* matrix, intt column);
    void GetTransposeImVector(floatt* vector, math::Matrix* matrix, intt row);
    void SetIdentity(math::Matrix* matrix);
    void SetReDiagonals(math::Matrix* matrix, floatt a);
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
};


#endif	/* MATRIXALLOCATOR_H */

