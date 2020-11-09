/*
 * Copyright 2016 - 2021 Marcin Matula
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

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdio.h>
#include <sstream>
#include <vector>

#include <linux/fs.h>

#include "oapHostMatrixUtils.h"

#include "oapHostMatrixUPtr.h"

#include "MatrixParser.h"
#include "ReferencesCounter.h"

#include "GenericCoreApi.h"

#include "MatricesList.h"
#include "oapMemoryCounter.h"

#define ReIsNotNULL(m) gReValues (m) != nullptr
#define ImIsNotNULL(m) gImValues (m) != nullptr

#ifdef DEBUG

std::ostream& operator<<(std::ostream& output, const math::Matrix*& matrix)
{
  return output << matrix << ", [" << gColumns (matrix) << ", " << gRows (matrix)
         << "]";
}

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::Matrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset (values, value, length);
}

inline void fillRePart(math::Matrix* output, floatt value)
{
  fillWithValue (gReValues (output), value, output->dim.columns * output->dim.rows);
}

inline void fillImPart(math::Matrix* output, floatt value)
{
  fillWithValue (gImValues (output), value, output->dim.columns * output->dim.rows);
}

namespace oap
{
namespace host
{

namespace
{
#if 0
std::map<floatt*, uintt> g_usedMemoryCounter;

void registerMemory (const oap::Memory& memory)
{
  if (memory.ptr != nullptr)
  {
    auto it = g_usedMemoryCounter.find (memory.ptr);
    if (it == g_usedMemoryCounter.end())
    {
      g_usedMemoryCounter[memory.ptr] = 0;
    }
    g_usedMemoryCounter[memory.ptr]++;
  }
}

template<typename Callback>
void unregisterMemory (const oap::Memory& memory, Callback&& callback)
{
  if (memory.ptr != nullptr)
  {
    auto it = g_usedMemoryCounter.find(memory.ptr);
    it->second--;
    if (it->second == 0)
    {
      callback (memory);
      g_usedMemoryCounter.erase (it);
    }
  }
}
#endif
MatricesList g_matricesList ("MATRICES_HOST");

math::Matrix* allocMatrix (const math::Matrix& ref)
{
  math::Matrix* output = NEW_MATRIX();

  memcpy (output, &ref, sizeof(math::Matrix));

  g_matricesList.add (output, CreateMatrixInfo (output));

  return output;
}

math::Matrix* allocMatrix_AllocMemory (bool isre, bool isim, uintt columns, uintt rows, floatt revalue = 0., floatt imvalue = 0.)
{
  math::Matrix ref;
  ref.dim = {columns, rows};

  if (isre)
  {
    ref.re = oap::host::NewMemory ({columns, rows});
    ref.reReg = {{0, 0}, {columns, rows}};
    fillRePart (&ref, revalue);
  }
  else
  {
    ref.re = {nullptr, {0, 0}};
    ref.reReg = {{0, 0}, {0, 0}};
  }

  if (isim)
  {
    ref.im = oap::host::NewMemory ({columns, rows});
    ref.imReg = {{0, 0}, {columns, rows}};
    fillImPart (&ref, imvalue);
  }
  else
  {
    ref.im = {nullptr, {0, 0}};
    ref.imReg = {{0, 0}, {0, 0}};
  }

  return allocMatrix (ref);
}

}

math::Matrix* NewMatrixRef (const math::Matrix* matrix)
{
  math::Matrix* output = nullptr;
  if (gReValues (matrix) != nullptr && gImValues (matrix) != nullptr)
  {
    output = NewMatrix(gColumns (matrix), gRows (matrix));
  }
  else if (gReValues (matrix) != nullptr)
  {
    output = NewReMatrix(gColumns (matrix), gRows (matrix));
  }
  else if (gImValues (matrix) != nullptr)
  {
    output = NewImMatrix(gColumns (matrix), gRows (matrix));
  }
  return output;
}

math::Matrix* NewMatrix(const math::Matrix* matrix, uintt columns, uintt rows)
{
  math::Matrix* output = nullptr;
  if (gReValues (matrix) != nullptr && gImValues (matrix) != nullptr)
  {
    output = NewMatrix(columns, rows);
  }
  else if (gReValues (matrix) != nullptr)
  {
    output = NewReMatrix(columns, rows);
  }
  else if (gImValues (matrix) != nullptr)
  {
    output = NewImMatrix(columns, rows);
  }
  return output;
}

math::Matrix* NewMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray, const floatt* imarray)
{
  math::Matrix* matrix = NewMatrix (columns, rows);
  oap::host::CopyArrayToMatrix (matrix, rearray, imarray);
  return matrix;
}

math::Matrix* NewReMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray)
{
  math::Matrix* matrix = NewReMatrix (columns, rows);
  oap::host::CopyArrayToReMatrix (matrix, rearray);
  return matrix;
}

math::Matrix* NewImMatrixCopyOfArray (uintt columns, uintt rows, const floatt* imarray)
{
  math::Matrix* matrix = NewImMatrix (columns, rows);
  oap::host::CopyArrayToImMatrix (matrix, imarray);
  return matrix;
}

math::Matrix* NewMatrix(const math::MatrixInfo& matrixInfo)
{
  return NewMatrix(matrixInfo.isRe, matrixInfo.isIm, matrixInfo.columns (), matrixInfo.rows ());
}

math::Matrix* NewMatrix (bool isre, bool isim, uintt columns, uintt rows)
{
  if (isre && isim)
  {
    return oap::host::NewMatrix(columns, rows);
  }
  else if (isre)
  {
    return oap::host::NewReMatrix(columns, rows);
  }
  else if (isim)
  {
    return oap::host::NewImMatrix(columns, rows);
  }
  return nullptr;
}

math::Matrix* NewMatrixWithValue (const math::MatrixInfo& minfo, floatt value)
{
  return oap::host::NewMatrixWithValue (minfo.isRe, minfo.isIm, minfo.columns (), minfo.rows (), value);
}

math::Matrix* NewMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt value)
{
  if (isre && isim)
  {
    return oap::host::NewMatrixWithValue (columns, rows, value);
  }
  else if (isre)
  {
    return oap::host::NewReMatrixWithValue (columns, rows, value);
  }
  else if (isim)
  {
    return oap::host::NewImMatrixWithValue (columns, rows, value);
  }
  return nullptr;
}

math::Matrix* NewMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (true, true, columns, rows);
}

math::Matrix* NewReMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (true, false, columns, rows);
}

math::Matrix* NewImMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (false, true, columns, rows);
}

math::Matrix* NewMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (true, true, columns, rows, value, value);
}

math::Matrix* NewReMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (true, false, columns, rows, value, value);
}

math::Matrix* NewImMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (false, true, columns, rows, value, value);
}

math::Matrix* NewMatrix (const std::string& text)
{
  matrixUtils::Parser parser(text);

  uintt columns = 0;
  uintt rows = 0;

  bool iscolumns = parser.getColumns (columns);
  bool isrows = parser.getRows (rows);

  bool isre = false;
  bool isim = false;

  oap::Memory memRe = {nullptr, {0, 0}};
  oap::Memory memIm = {nullptr, {0, 0}};

  oap::MemoryRegion reReg = {{0, 0}, {0, 0}};
  oap::MemoryRegion imReg = {{0, 0}, {0, 0}};

  try
  {
    if (matrixUtils::HasArray (text, 1))
    {
      memRe = matrixUtils::CreateArrayDefaultAlloc (text, 1);
      reReg = {{0, 0}, memRe.dims};
      columns = columns == 0 ? reReg.dims.width : columns;
      rows = rows == 0 ? reReg.dims.height : rows;
      isre = true;
    }

    if (matrixUtils::HasArray (text, 2))
    {
      memIm = matrixUtils::CreateArrayDefaultAlloc (text, 2);
      imReg = {{0, 0}, memIm.dims};
      columns = columns == 0 ? imReg.dims.width : columns;
      rows = rows == 0 ? imReg.dims.height : rows;
      isim = true;
    }
  }
  catch (const matrixUtils::Parser::ParsingException& pe)
  {
    logError ("%s", pe.what ());
    abort ();
  }

  math::Matrix ref;
  ref.dim = {columns, rows};
  ref.re = memRe;
  ref.reReg = reReg;
  ref.im = memIm;
  ref.imReg = imReg;

  oapAssert (!isre || !isim || reReg.dims == imReg.dims);
  logAssert(memRe.ptr != nullptr || memIm.ptr != nullptr);

  return allocMatrix (ref);
}

void DeleteMatrix(const math::Matrix* matrix)
{
  if (nullptr == matrix)
  {
    return;
  }

  auto minfo = g_matricesList.remove (matrix);

  oap::host::DeleteMemory (matrix->re);
  oap::host::DeleteMemory (matrix->im);

  DELETE_MATRIX(matrix);

  if (minfo.isInitialized ())
  {
    logTrace ("Deallocate: host matrix = %p %s", matrix, minfo.toString().c_str());
  }
}

floatt GetReValue(const math::Matrix* matrix, uintt column, uintt row)
{
  if (gReValues (matrix) == nullptr)
  {
    return 0;
  }
  return gReValues (matrix)[row * gColumns (matrix) + column];
}

floatt GetImValue(const math::Matrix* matrix, uintt column, uintt row)
{
  if (gImValues (matrix) == nullptr)
  {
    return 0;
  }
  return gImValues (matrix)[row * gColumns (matrix) + column];
}

void SetReValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (gReValues (matrix))
  {
    gReValues (matrix)[row * gColumns (matrix) + column] = value;
  }
}

void SetImValue(const math::Matrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (gImValues (matrix))
  {
    gImValues (matrix)[row * gColumns (matrix) + column] = value;
  }
}

std::string GetMatrixStr (const math::Matrix* matrix)
{
  std::string output;
  oap::generic::printMatrix (output, matrix, matrixUtils::PrintArgs(), oap::host::GetMatrixInfo);
  return output;
}

math::Matrix GetRefHostMatrix (const math::Matrix* matrix)
{
  return *matrix;
}

void PrintMatrix(FILE* stream, const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  std::string output;
  oap::generic::printMatrix (output, matrix, matrixUtils::PrintArgs(), oap::host::GetMatrixInfo);
  fprintf(stream, "%s", output.c_str());
}

void PrintMatrix(FILE* stream, const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(stream, args, matrix);
}

void PrintMatrix(const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  PrintMatrix(stdout, args, matrix);
}

void PrintMatrix(const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(args, matrix);
}

bool PrintMatrixToFile(const std::string& path, const matrixUtils::PrintArgs& args, const math::Matrix* matrix)
{
  FILE* file = fopen(path.c_str(), "w");

  if (file == nullptr)
  {
    return false;
  }

  PrintMatrix (file, args, matrix);

  fclose(file);
  return true;
}

bool PrintMatrixToFile(const std::string& path, const math::Matrix* matrix, const matrixUtils::PrintArgs& args)
{
  return PrintMatrixToFile(path, args, matrix);
}

void Copy(math::Matrix* dst, const math::Matrix* src, const MatrixEx& subMatrix, uintt column, uintt row)
{
  uintt rows = dst->re.dims.width;
  uintt columns2 = subMatrix.columns;
  for (uintt fa = 0; fa < rows; fa++)
  {
    uintt fa1 = fa + subMatrix.row;
    if (fa < row)
    {
      Copy(gReValues (dst) + fa * gColumns (dst),
           gReValues (src) + (fa1)*columns2, column);
      Copy(gReValues (dst) + column + fa * gColumns (dst),
           gReValues (src) + (1 + column) + fa * columns2,
           (columns2 - column));
    }
    else if (fa >= row)
    {
      Copy(gReValues (dst) + fa * gColumns (dst),
           &gReValues (src)[(fa1 + 1) * columns2], column);

      Copy(gReValues (dst) + column + fa * gColumns (dst),
           &gReValues (src)[(fa1 + 1) * columns2 + column + 1],
           (columns2 - column));
    }
  }
}

void Copy(math::Matrix* dst, const math::Matrix* src, uintt column, uintt row)
{
  uintt rows = gRows (src);
  uintt columns = gColumns (src);
  for (uintt fa = 0; fa < rows; fa++)
  {
    if (fa < row)
    {
      Copy(&gReValues (dst)[fa * gColumns (dst)],
           &gReValues (src)[fa * columns], column);
      if (column < gColumns (src) - 1)
      {
        Copy(&gReValues (dst)[column + fa * gColumns (dst)],
             &gReValues (src)[(1 + column) + fa * columns],
             (gColumns (src) - (column + 1)));
      }
    }
    else if (fa > row)
    {
      Copy(&gReValues (dst)[(fa - 1) * gColumns (dst)],
           &gReValues (src)[fa * columns], column);
      if (column < gColumns (src) - 1)
      {
        Copy(&gReValues (dst)[column + (fa - 1) * gColumns (dst)],
             &gReValues (src)[fa * columns + (column + 1)],
             (gColumns (src) - (column + 1)));
      }
    }
  }
}

void CopyMatrix (math::Matrix* dst, const math::Matrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype(oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> mmApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, memcpy, mmApi, mmApi);
#endif
  if (dst->re.ptr && src->re.ptr) {oap::host::CopyHostToHost (dst->re, {0, 0}, src->re, {{0, 0}, src->re.dims});}
  if (dst->im.ptr && src->im.ptr) {oap::host::CopyHostToHost (dst->im, {0, 0}, src->im, {{0, 0}, src->im.dims});}
}

void CopyMatrixRegion (math::Matrix* dst, const oap::MemoryLoc& dstLoc, const math::Matrix* src, const oap::MemoryRegion& srcReg)
{
  if (dst->re.ptr && src->re.ptr) {oap::host::CopyHostToHost (dst->re, GetReMatrixMemoryLoc (dst, &dstLoc), src->re, GetReMatrixMemoryRegion (src, &srcReg));}
  if (dst->im.ptr && src->im.ptr) {oap::host::CopyHostToHost (dst->im, GetImMatrixMemoryLoc (dst, &dstLoc), src->im, GetImMatrixMemoryRegion (src, &srcReg));}
}

void CopyRe(math::Matrix* dst, const math::Matrix* src)
{
  const uintt length1 = gColumns (dst) * gRows (dst);
  const uintt length2 = gColumns (src) * gRows (src);
  const uintt length = length1 < length2 ? length1 : length2;
  if (ReIsNotNULL(dst) && ReIsNotNULL(src))
  {
    memcpy(gReValues (dst), gReValues (src), length * sizeof(floatt));
  }
}

void CopyIm(math::Matrix* dst, const math::Matrix* src)
{
  const uintt length1 = gColumns (dst) * gRows (dst);
  const uintt length2 = gColumns (src) * gRows (src);
  const uintt length = length1 < length2 ? length1 : length2;
  if (ImIsNotNULL(dst) && ImIsNotNULL(src))
  {
    memcpy(gImValues (dst), gImValues (src), length * sizeof(floatt));
  }
}

math::Matrix* NewMatrixCopy(const math::Matrix* matrix)
{
  math::Matrix* output = oap::host::NewMatrixRef (matrix);
  oap::host::CopyMatrix(output, matrix);
  return output;
}

void SetVector(math::Matrix* matrix, uintt column, math::Matrix* vector)
{
  SetVector(matrix, column, gReValues (vector), gImValues (vector), gRows (vector));
}

void SetVector(math::Matrix* matrix, uintt column, floatt* revector,
               floatt* imvector, uintt length)
{
  if (revector != nullptr)
  {
    SetReVector(matrix, column, revector, length);
  }

  if (imvector != nullptr)
  {
    SetImVector(matrix, column, imvector, length);
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length)
{
  if (gReValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      gReValues (matrix)[column + gColumns (matrix) * fa] = vector[fa];
    }
  }
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (gReValues (matrix))
  {
    memcpy(&gReValues (matrix)[row * gColumns (matrix)], vector,
           length * sizeof(floatt));
  }
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector,
                 uintt length)
{
  if (gImValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      gImValues (matrix)[column + gColumns (matrix) * fa] = vector[fa];
    }
  }
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (gImValues (matrix))
  {
    memcpy(&gImValues (matrix)[row * gColumns (matrix)], vector,
           length * sizeof(floatt));
  }
}

void SetReVector(math::Matrix* matrix, uintt column, floatt* vector)
{
  SetReVector(matrix, column, vector, gRows (matrix));
}

void SetTransposeReVector(math::Matrix* matrix, uintt row, floatt* vector)
{
  SetTransposeReVector(matrix, row, vector, gColumns (matrix));
}

void SetImVector(math::Matrix* matrix, uintt column, floatt* vector)
{
  SetImVector(matrix, column, vector, gRows (matrix));
}

void SetTransposeImVector(math::Matrix* matrix, uintt row, floatt* vector)
{
  SetTransposeImVector(matrix, row, vector, gColumns (matrix));
}

void GetMatrixStr(std::string& text, const math::Matrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void ToString (std::string& str, const math::Matrix* matrix)
{
  if (matrix == nullptr)
  {
    str = "nullptr";
    return;
  }

  matrixUtils::PrintArgs args;
  args.prepareSection (matrix);

  oap::generic::printMatrix (str, matrix, args, oap::host::GetMatrixInfo);
}

void GetReMatrixStr(std::string& text, const math::Matrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void GetImMatrixStr(std::string& str, const math::Matrix* matrix)
{
  str = "";
  if (matrix == nullptr)
  {
    return;
  }
  std::stringstream sstream;
  str += "[";
  for (int fb = 0; fb < gRows (matrix); fb++)
  {
    for (int fa = 0; fa < gColumns (matrix); fa++)
    {
      sstream << gImValues (matrix)[fb * gColumns (matrix) + fa];
      str += sstream.str();
      sstream.str("");
      if (fa != gColumns (matrix) - 1)
      {
        str += ",";
      }
      if (fa == gColumns (matrix) - 1 && fb != gRows (matrix) - 1)
      {
        str += "\n";
      }
    }
  }
  str += "]";
}

void GetVector(math::Matrix* vector, math::Matrix* matrix, uintt column)
{
  GetVector(gReValues (vector), gImValues (vector), gRows (vector), matrix, column);
}

void GetVector(floatt* revector, floatt* imvector, uint length, math::Matrix* matrix, uint column)
{
  if (revector != nullptr)
  {
    GetReVector(revector, length, matrix, column);
  }

  if (imvector != nullptr)
  {
    GetImVector(imvector, length, matrix, column);
  }
}

void GetTransposeVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  if (gReValues (vector) != nullptr)
  {
    GetTransposeReVector(vector, matrix, column);
  }
  if (gImValues (vector) != nullptr)
  {
    GetTransposeImVector(vector, matrix, column);
  }
}

void GetTransposeReVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  GetTransposeReVector(gReValues (vector), matrix, column);
}

void GetTransposeImVector(math::Matrix* vector, math::Matrix* matrix, uint column)
{
  GetTransposeImVector(gImValues (vector), matrix, column);
}

void GetReVector(floatt* vector, uint length, math::Matrix* matrix, uint column)
{
  if (gReValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = gReValues (matrix)[column + gColumns (matrix) * fa];
    }
  }
}

void GetTransposeReVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (gReValues (matrix))
  {
    memcpy(vector, &gReValues (matrix)[row * gColumns (matrix)],
           length * sizeof(floatt));
  }
}

void GetImVector(floatt* vector, uint length, math::Matrix* matrix, uint column)
{
  if (gImValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = gImValues (matrix)[column + gColumns (matrix) * fa];
    }
  }
}

void GetTransposeImVector(floatt* vector, uint length, math::Matrix* matrix, uint row)
{
  if (gImValues (matrix))
  {
    memcpy(vector, &gImValues (matrix)[row * gColumns (matrix)], length * sizeof(floatt));
  }
}

void GetReVector(floatt* vector, math::Matrix* matrix, uint column)
{
  GetReVector(vector, gRows (matrix), matrix, column);
}

void GetTransposeReVector(floatt* vector, math::Matrix* matrix, uint row)
{
  GetTransposeReVector(vector, gColumns (matrix), matrix, row);
}

void GetImVector(floatt* vector, math::Matrix* matrix, uint column)
{
  GetImVector(vector, gRows (matrix), matrix, column);
}

void GetTransposeImVector(floatt* vector, math::Matrix* matrix, uint row)
{
  GetTransposeReVector(vector, gColumns (matrix), matrix, row);
}

floatt SmallestDiff(math::Matrix* matrix, math::Matrix* matrix1)
{
  floatt diff = gReValues (matrix)[0] - gReValues (matrix1)[0];
  for (uintt fa = 0; fa < gColumns (matrix); fa++)
  {
    for (uintt fb = 0; fb < gRows (matrix); fb++)
    {
      uintt index = fa + fb * gColumns (matrix);
      floatt diff1 = gReValues (matrix)[index] - gReValues (matrix1)[index];
      if (diff1 < 0)
      {
        diff1 = -diff1;
      }
      if (diff > diff1)
      {
        diff = diff1;
      }
    }
  }
  return diff;
}

floatt LargestDiff(math::Matrix* matrix, math::Matrix* matrix1)
{
  floatt diff = gReValues (matrix)[0] - gReValues (matrix1)[0];
  for (uintt fa = 0; fa < gColumns (matrix); fa++)
  {
    for (uintt fb = 0; fb < gRows (matrix); fb++)
    {
      uintt index = fa + fb * gColumns (matrix);
      floatt diff1 = gReValues (matrix)[index] - gReValues (matrix1)[index];
      if (diff1 < 0)
      {
        diff1 = -diff1;
      }
      if (diff < diff1)
      {
        diff = diff1;
      }
    }
  }
  return diff;
}

void SetIdentity(math::Matrix* matrix)
{
  oap::host::SetDiagonalReMatrix(matrix, 1);
  oap::host::SetImZero(matrix);
}

void SetReZero(math::Matrix* matrix)
{
  if (gReValues (matrix))
  {
    memset(gReValues (matrix), 0,
           gColumns (matrix) * gRows (matrix) * sizeof(floatt));
  }
}

void SetImZero(math::Matrix* matrix)
{
  if (gImValues (matrix))
  {
    memset(gImValues (matrix), 0,
           gColumns (matrix) * gRows (matrix) * sizeof(floatt));
  }
}

void SetZero(math::Matrix* matrix)
{
  SetReZero(matrix);
  SetImZero(matrix);
}

void SetIdentityMatrix(math::Matrix* matrix)
{
  SetDiagonalReMatrix(matrix, 1);
  SetImZero(matrix);
}

bool IsEquals(math::Matrix* transferMatrix2, math::Matrix* transferMatrix1,
              floatt diff)
{
  for (uintt fa = 0; fa < gColumns (transferMatrix2); fa++)
  {
    for (uintt fb = 0; fb < gRows (transferMatrix2); fb++)
    {
      floatt p = gReValues (transferMatrix2)[fa + gColumns (transferMatrix2) * fb] -
                 gReValues (transferMatrix1)[fa + gColumns (transferMatrix1) * fb];
      if (p < -diff || p > diff)
      {
        return false;
      }
    }
  }
  return true;
}

floatt GetTrace(math::Matrix* matrix)
{
  floatt o = 1.;
  for (uintt fa = 0; fa < gColumns (matrix); ++fa)
  {
    floatt v = gReValues (matrix)[fa * gColumns (matrix) + fa];
    if (-MATH_VALUE_LIMIT < v && v < MATH_VALUE_LIMIT)
    {
      v = 0;
    }
    o = o * v;
  }
  return o;
}

void SetDiagonalMatrix(math::Matrix* matrix, floatt a)
{
  SetDiagonalReMatrix(matrix, a);
  SetDiagonalImMatrix(matrix, a);
}

void SetDiagonalReMatrix(math::Matrix* matrix, floatt a)
{
  if (gReValues (matrix))
  {
    fillRePart(matrix, 0);
    for (int fa = 0; fa < gColumns (matrix); fa++)
    {
      gReValues (matrix)[fa * gColumns (matrix) + fa] = a;
    }
  }
}

void SetDiagonalImMatrix(math::Matrix* matrix, floatt a)
{
  if (gImValues (matrix))
  {
    fillImPart(matrix, 0);
    for (int fa = 0; fa < gColumns (matrix); fa++)
    {
      gImValues (matrix)[fa * gColumns (matrix) + fa] = a;
    }
  }
}

math::MatrixInfo CreateMatrixInfo(const math::Matrix* matrix)
{
  return math::MatrixInfo (gReValues (matrix) != nullptr,
                           gImValues (matrix) != nullptr,
                           gColumns (matrix),
                           gRows (matrix));
}

math::MatrixInfo GetMatrixInfo (const math::Matrix* matrix)
{
  return g_matricesList.getUserData (matrix);
}

math::Matrix* ReadMatrix (const std::string& path)
{
  utils::ByteBuffer buffer (path);
  math::Matrix* matrix = oap::host::LoadMatrix (buffer);

  return matrix;
}

math::Matrix* ReadRowVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::Matrix* subMatrix = oap::host::NewSubMatrix (matrix, 0, index, gColumns (matrix), 1);
  return subMatrix;
}

math::Matrix* ReadColumnVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::Matrix* subMatrix = oap::host::NewSubMatrix (matrix, index, 0, 1, gRows (matrix));
  return subMatrix;
}

void CopyReBuffer (math::Matrix* houtput, math::Matrix* hinput)
{
  size_t sOutput = gColumns (houtput) * gRows (houtput);
  size_t sInput = gColumns (hinput) * gRows (hinput);

  debugExceptionMsg(sOutput == sInput, "Buffers have different sizes.");

  memcpy (gReValues (houtput), gReValues (hinput), sOutput * sizeof (floatt));
}

bool WriteMatrix (const std::string& path, const math::Matrix* matrix)
{
  utils::ByteBuffer buffer;
  oap::host::SaveMatrix (matrix, buffer);
  try
  {
    buffer.fwrite (path);
  }
  catch (const std::runtime_error& error)
  {
    debugError ("Write to file error: %s", error.what());
    return false;
  }
  return true;
}

void copySubMatrix (math::Matrix* dst, const math::Matrix* src, uintt cindex, uintt rindex)
{
  if (ReIsNotNULL(dst) && ReIsNotNULL(src))
  {
    for (uintt fa = 0; fa < gRows (dst); ++fa)
    {
      CopyBuffer(GetRePtr(dst, 0, fa), GetRePtr(src, cindex, fa + rindex), gColumns (dst));
    }
  }
  if (ImIsNotNULL(dst) && ImIsNotNULL(src))
  {
    for (uintt fa = 0; fa < gRows (dst); ++fa)
    {
      CopyBuffer(GetImPtr(dst, 0, fa), GetImPtr(src, cindex, fa + rindex), gColumns (dst));
    }
  }
}

void CopySubMatrix(math::Matrix* dst, const math::Matrix* src, uintt cindex, uintt rindex)
{
  copySubMatrix (dst, src, cindex, rindex);
}

inline uintt calculate (uintt matrixd, uintt dindex, uintt dlength)
{
  return dindex + dlength < matrixd ? dlength : matrixd - dindex;
}

math::Matrix* NewSubMatrix (const math::Matrix* orig, uintt cindex, uintt rindex, uintt clength, uintt rlength)
{
  clength = calculate (gColumns (orig), cindex, clength);
  rlength = calculate (gRows (orig), rindex, rlength);

  math::Matrix* submatrix = oap::host::NewMatrix (orig, clength, rlength);
  copySubMatrix (submatrix, orig, cindex, rindex);
  return submatrix;
}

math::Matrix* GetSubMatrix (const math::Matrix* orig, uintt cindex, uintt rindex, math::Matrix* matrix)
{
  uintt clength = calculate (gColumns (orig), cindex, gColumns (matrix));
  uintt rlength = calculate (gRows (orig), rindex, gRows (matrix));

  if (gColumns (matrix) == clength && gRows (matrix) == rlength)
  {
    copySubMatrix (matrix, orig, cindex, rindex);
    return matrix;
  }

  oap::host::DeleteMatrix (matrix);
  return NewSubMatrix (orig, cindex, rindex, clength, rlength);
}

void SaveMatrixInfo (const math::MatrixInfo& minfo, utils::ByteBuffer& buffer)
{
  buffer.push_back (minfo.isRe);
  buffer.push_back (minfo.isIm);
  buffer.push_back (minfo.columns ());
  buffer.push_back (minfo.rows ());
}

void SaveMatrix (const math::Matrix* matrix, utils::ByteBuffer& buffer)
{
  bool isMatrix = (matrix != nullptr);

  buffer.push_back (isMatrix);

  if (!isMatrix)
  {
    return;
  }

  auto minfo = oap::host::GetMatrixInfo (matrix);

  SaveMatrixInfo (minfo, buffer);

  if (minfo.isRe)
  {
    buffer.push_back (gReValues (matrix), minfo.length ());
  }

  if (minfo.isIm)
  {
    buffer.push_back (gImValues (matrix), minfo.length ());
  }
}

math::Matrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  bool isMatrix = buffer.read <bool>();

  if (!isMatrix)
  {
    return nullptr;
  }

  math::MatrixInfo minfo = LoadMatrixInfo (buffer);
  math::Matrix* matrix = NewMatrix (minfo);

  if (minfo.isRe)
  {
    buffer.read (gReValues (matrix), minfo.length ());
  }

  if (minfo.isIm)
  {
    buffer.read (gImValues (matrix), minfo.length ());
  }

  return matrix;
}

math::MatrixInfo LoadMatrixInfo (const utils::ByteBuffer& buffer)
{
  math::MatrixInfo minfo;

  minfo.isRe = buffer.read<decltype (minfo.isRe)> ();
  minfo.isIm = buffer.read<decltype (minfo.isIm)> ();
  minfo.m_matrixDim.columns = buffer.read<decltype (minfo.m_matrixDim.columns)> ();
  minfo.m_matrixDim.rows = buffer.read<decltype (minfo.m_matrixDim.rows)> ();

  return minfo;
}

void CopyArrayToMatrix (math::Matrix* matrix, const floatt* rebuffer, const floatt* imbuffer)
{
  if (rebuffer != nullptr)
  {
    CopyArrayToReMatrix (matrix, rebuffer);
  }
  if (imbuffer != nullptr)
  {
    CopyArrayToImMatrix (matrix, imbuffer);
  }
}

void CopyArrayToReMatrix (math::Matrix* matrix, const floatt* buffer)
{
  debugAssert (gReValues (matrix) != nullptr);
  memcpy (gReValues (matrix), buffer, gColumns (matrix) * gRows (matrix) * sizeof(floatt));
}

void CopyArrayToImMatrix (math::Matrix* matrix, const floatt* buffer)
{
  debugAssert (gImValues (matrix) != nullptr);
  memcpy (gImValues (matrix), buffer, gColumns (matrix) * gRows (matrix) * sizeof(floatt));
}

void CopyHostArrayToHostMatrix (math::Matrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToMatrix (matrix, rebuffer, imbuffer);
}

void CopyHostArrayToHostReMatrix (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToReMatrix (matrix, buffer);
}

void CopyHostArrayToHostImMatrix (math::Matrix* matrix, const floatt* buffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToImMatrix (matrix, buffer);
}

void SetSubs(math::Matrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumns (matrix, subcolumns);
  SetSubRows (matrix, subrows);
}

void SetSubColumns(math::Matrix* matrix, uintt subcolumns)
{
  if (subcolumns != MATH_UNDEFINED)
  {
    if (matrix->re.ptr != nullptr) { matrix->dim.columns = subcolumns; matrix->reReg.dims.width = subcolumns; }
    if (matrix->im.ptr != nullptr) { matrix->dim.columns = subcolumns; matrix->imReg.dims.width = subcolumns; }
  }
  else
  {
    if (matrix->re.ptr != nullptr) { matrix->dim.columns = matrix->re.dims.width; matrix->reReg.dims.width = matrix->im.dims.width; }
    if (matrix->im.ptr != nullptr) { matrix->dim.columns = matrix->im.dims.width; matrix->imReg.dims.width = matrix->im.dims.width; }
  }
}

void SetSubRows(math::Matrix* matrix, uintt subrows)
{
  if (subrows != MATH_UNDEFINED)
  {
    if (matrix->re.ptr != nullptr) { matrix->dim.rows = subrows; matrix->reReg.dims.height = subrows; }
    if (matrix->im.ptr != nullptr) { matrix->dim.rows = subrows; matrix->imReg.dims.height = subrows; }
  }
  else
  {
    if (matrix->re.ptr != nullptr) { matrix->dim.rows = matrix->re.dims.height; matrix->reReg.dims.height = matrix->re.dims.height; }
    if (matrix->im.ptr != nullptr) { matrix->dim.rows = matrix->im.dims.height; matrix->imReg.dims.height = matrix->im.dims.height; }
  }
}

void SetSubsSafe(math::Matrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumnsSafe(matrix, subcolumns);
  SetSubRowsSafe(matrix, subrows);
}

void SetSubColumnsSafe(math::Matrix* matrix, uintt subcolumns)
{
  if (subcolumns == MATH_UNDEFINED || gColumns (matrix) < subcolumns)
  {
    matrix->reReg = {{0, 0}, {0, 0}};
    matrix->imReg = {{0, 0}, {0, 0}};
  }
  else
  {
    oap::MemoryRegion reg = {{0, 0}, {gColumns (matrix), gRows (matrix)}};
    reg.dims.width = subcolumns;

    (matrix->reReg) = reg;
    (matrix->imReg) = reg;
  }
}

void SetSubRowsSafe(math::Matrix* matrix, uintt subrows)
{
  if (subrows == MATH_UNDEFINED || gRows (matrix) < subrows)
  {
    matrix->reReg = {{0, 0}, {0, 0}};
    matrix->imReg = {{0, 0}, {0, 0}};
  }
  else
  {
    oap::MemoryRegion reg = {{0, 0}, {gColumns (matrix), gRows (matrix)}};
    reg.dims.height = subrows;

    (matrix->reReg) = reg;
    (matrix->imReg) = reg;
  }
}

void SetMatrix(math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  SetReMatrix (matrix, matrix1, column, row);
  SetImMatrix (matrix, matrix1, column, row);
}

void SetReMatrix (math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  oap::generic::setMatrix (matrix, matrix1, column, row, [](math::Matrix* matrix) { return matrix->re; }, [](math::Matrix* matrix) { return matrix->reReg; }, memcpy);
}

void SetImMatrix (math::Matrix* matrix, math::Matrix* matrix1, uintt column, uintt row)
{
  oap::generic::setMatrix (matrix, matrix1, column, row, [](math::Matrix* matrix) { return matrix->im; }, [](math::Matrix* matrix) { return matrix->imReg; }, memcpy);
}

oap::ThreadsMapper CreateThreadsMapper (const std::vector<std::vector<math::Matrix*>>& matricesVec, oap::threads::ThreadsMapperAlgo algo)
{
  return createThreadsMapper (matricesVec, algo);
}

namespace
{

inline math::Matrix* allocReMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::Matrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re = oap::host::ReuseMemory (mem);
  hostRefMatrix.reReg = reg;
  hostRefMatrix.im = {nullptr, {0, 0}};
  hostRefMatrix.imReg = {{0, 0}, {0, 0}};

  return allocMatrix (hostRefMatrix);
}

inline math::Matrix* allocImMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::Matrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re = {nullptr, {0, 0}};
  hostRefMatrix.reReg = {{0, 0}, {0, 0}};
  hostRefMatrix.im = oap::host::ReuseMemory (mem);
  hostRefMatrix.imReg = reg;

  return allocMatrix (hostRefMatrix);
}

inline math::Matrix* allocRealMatrix_FromMemory (oap::Memory& remem, const oap::MemoryRegion& rereg, oap::Memory& immem, const oap::MemoryRegion& imreg)
{
  oapAssert (rereg.dims == imreg.dims);

  math::Matrix hostRefMatrix;

  hostRefMatrix.dim = {rereg.dims.width, rereg.dims.height};

  hostRefMatrix.re = oap::host::ReuseMemory (remem);
  hostRefMatrix.reReg = rereg;
  hostRefMatrix.im = oap::host::ReuseMemory (immem);
  hostRefMatrix.imReg = imreg;

  return allocMatrix (hostRefMatrix);
}

}

math::Matrix* NewMatrixFromMemory (uintt columns, uintt rows, oap::Memory& remem, const oap::MemoryLoc& reloc, oap::Memory& immem, const oap::MemoryLoc& imloc)
{
  return allocRealMatrix_FromMemory (remem, {reloc, {columns, rows}}, immem, {imloc, {columns, rows}});
}

math::Matrix* NewReMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocReMatrix_FromMemory (memory, {loc, {columns, rows}});
}

math::Matrix* NewImMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocImMatrix_FromMemory (memory, {loc, {columns, rows}});
}

void SetZeroRow (const math::Matrix* matrix, uintt index, bool re, bool im)
{
  if (re)
  {
    SetReZeroRow (matrix, index);
  }
  if (im)
  {
    SetImZeroRow (matrix, index);
  }
}

void SetReZeroRow (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.re.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.re, hm.reReg, {index, 0});
    oap::generic::copy (hm.re.ptr, hm.re.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, memcpy);
  }
}

void SetImZeroRow (const math::Matrix* matrix, uintt index)
{
  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.im.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.im, hm.imReg, {index, 0});
    oap::generic::copy (hm.im.ptr, hm.im.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, memcpy);
  }
}

void SetValueToMatrix (math::Matrix* matrix, floatt re, floatt im)
{
  SetValueToReMatrix (matrix, re);
  SetValueToImMatrix (matrix, im);
}

void SetValueToReMatrix (math::Matrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.re.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewReMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetReMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetReMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.re.ptr, hm.re.dims, loc, uptr->re.ptr, uptr->re.dims, reg, memcpy);
  }
}

void SetValueToImMatrix (math::Matrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::Matrix hm = GetRefHostMatrix (matrix);

  if (hm.im.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewImMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetImMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetImMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.im.ptr, hm.im.dims, loc, uptr->im.ptr, uptr->im.dims, reg, memcpy);
  }
}

void SetZeroMatrix (math::Matrix* matrix)
{
  SetValueToMatrix (matrix, 0, 0);
}

void SetZeroReMatrix (math::Matrix* matrix)
{
  SetValueToReMatrix (matrix, 0);
}

void SetZeroImMatrix (math::Matrix* matrix)
{
  SetValueToImMatrix (matrix, 0);
}

floatt GetReDiagonal (const math::Matrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::host::GetRefHostMatrix,
                                    [](const math::Matrix* matrix, const math::Matrix& ref){return matrix->re;},
                                    [](const math::Matrix* matrix, const math::Matrix& ref){return matrix->reReg;}, memcpy);
}

floatt GetImDiagonal (const math::Matrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::host::GetRefHostMatrix,
                                    [](const math::Matrix* matrix, const math::Matrix& ref){return matrix->im;},
                                    [](const math::Matrix* matrix, const math::Matrix& ref){return matrix->imReg;}, memcpy);
}

void CopyReMatrixToHostBuffer (floatt* buffer, uintt length, const math::Matrix* matrix)
{
  math::Matrix ref = oap::host::GetRefHostMatrix (matrix);
  oap::host::CopyHostToHostBuffer (buffer, length, ref.re, ref.reReg);
}

void CopyHostBufferToReMatrix (math::Matrix* matrix, const floatt* buffer, uintt length)
{
  math::Matrix ref = oap::host::GetRefHostMatrix (matrix);
  oap::host::CopyHostBufferToHost (ref.re, ref.reReg, buffer, length);
}

}
}
