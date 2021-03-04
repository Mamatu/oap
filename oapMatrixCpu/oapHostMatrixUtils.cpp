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

std::ostream& operator<<(std::ostream& output, const math::ComplexMatrix*& matrix)
{
  return output << matrix << ", [" << gColumns (matrix) << ", " << gRows (matrix)
         << "]";
}

#define NEW_MATRIX() new math::ComplexMatrix();

#define DELETE_MATRIX(matrix) delete matrix;

#else

#define NEW_MATRIX() new math::ComplexMatrix();

#define DELETE_MATRIX(matrix) delete matrix;

#endif

inline void fillWithValue (floatt* values, floatt value, uintt length)
{
  math::Memset (values, value, length);
}

inline void fillRePart(math::ComplexMatrix* output, floatt value)
{
  fillWithValue (gReValues (output), value, output->dim.columns * output->dim.rows);
}

inline void fillImPart(math::ComplexMatrix* output, floatt value)
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

math::ComplexMatrix* allocMatrix (const math::ComplexMatrix& ref)
{
  math::ComplexMatrix* output = NEW_MATRIX();

  memcpy (output, &ref, sizeof(math::ComplexMatrix));

  g_matricesList.add (output, CreateMatrixInfo (output));

  return output;
}

math::ComplexMatrix* allocMatrix_AllocMemory (bool isre, bool isim, uintt columns, uintt rows, floatt revalue = 0., floatt imvalue = 0.)
{
  math::ComplexMatrix ref;
  ref.dim = {columns, rows};

  if (isre)
  {
    ref.re.mem = oap::host::NewMemory ({columns, rows});
    ref.re.reg = {{0, 0}, {columns, rows}};
    fillRePart (&ref, revalue);
  }
  else
  {
    ref.re.mem = {nullptr, {0, 0}};
    ref.re.reg = {{0, 0}, {0, 0}};
  }

  if (isim)
  {
    ref.im.mem = oap::host::NewMemory ({columns, rows});
    ref.im.reg = {{0, 0}, {columns, rows}};
    fillImPart (&ref, imvalue);
  }
  else
  {
    ref.im.mem = {nullptr, {0, 0}};
    ref.im.reg = {{0, 0}, {0, 0}};
  }

  return allocMatrix (ref);
}

}

math::ComplexMatrix* NewMatrixRef (const math::ComplexMatrix* matrix)
{
  math::ComplexMatrix* output = nullptr;
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

math::ComplexMatrix* NewMatrix(const math::ComplexMatrix* matrix, uintt columns, uintt rows)
{
  math::ComplexMatrix* output = nullptr;
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

math::ComplexMatrix* NewMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray, const floatt* imarray)
{
  math::ComplexMatrix* matrix = NewMatrix (columns, rows);
  oap::host::CopyArrayToMatrix (matrix, rearray, imarray);
  return matrix;
}

math::ComplexMatrix* NewReMatrixCopyOfArray (uintt columns, uintt rows, const floatt* rearray)
{
  math::ComplexMatrix* matrix = NewReMatrix (columns, rows);
  oap::host::CopyArrayToReMatrix (matrix, rearray);
  return matrix;
}

math::ComplexMatrix* NewImMatrixCopyOfArray (uintt columns, uintt rows, const floatt* imarray)
{
  math::ComplexMatrix* matrix = NewImMatrix (columns, rows);
  oap::host::CopyArrayToImMatrix (matrix, imarray);
  return matrix;
}

math::ComplexMatrix* NewMatrix(const math::MatrixInfo& matrixInfo)
{
  return NewMatrix(matrixInfo.isRe, matrixInfo.isIm, matrixInfo.columns (), matrixInfo.rows ());
}

math::ComplexMatrix* NewMatrix (bool isre, bool isim, uintt columns, uintt rows)
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

math::ComplexMatrix* NewMatrixWithValue (const math::MatrixInfo& minfo, floatt value)
{
  return oap::host::NewMatrixWithValue (minfo.isRe, minfo.isIm, minfo.columns (), minfo.rows (), value);
}

math::ComplexMatrix* NewMatrixWithValue (bool isre, bool isim, uintt columns, uintt rows, floatt value)
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

math::ComplexMatrix* NewMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (true, true, columns, rows);
}

math::ComplexMatrix* NewReMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (true, false, columns, rows);
}

math::ComplexMatrix* NewImMatrix (uintt columns, uintt rows)
{
  return allocMatrix_AllocMemory (false, true, columns, rows);
}

math::ComplexMatrix* NewMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (true, true, columns, rows, value, value);
}

math::ComplexMatrix* NewReMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (true, false, columns, rows, value, value);
}

math::ComplexMatrix* NewImMatrixWithValue (uintt columns, uintt rows, floatt value)
{
  return allocMatrix_AllocMemory (false, true, columns, rows, value, value);
}

math::ComplexMatrix* NewMatrix (const std::string& text)
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

  math::ComplexMatrix ref;
  ref.dim = {columns, rows};
  ref.re.mem = memRe;
  ref.re.reg = reReg;
  ref.im.mem = memIm;
  ref.im.reg = imReg;

  oapAssert (!isre || !isim || reReg.dims == imReg.dims);
  logAssert(memRe.ptr != nullptr || memIm.ptr != nullptr);

  return allocMatrix (ref);
}

void DeleteMatrix(const math::ComplexMatrix* matrix)
{
  if (nullptr == matrix)
  {
    return;
  }

  auto minfo = g_matricesList.remove (matrix);

  oap::host::DeleteMemory (matrix->re.mem);
  oap::host::DeleteMemory (matrix->im.mem);

  DELETE_MATRIX(matrix);

  if (minfo.isInitialized ())
  {
    logTrace ("Deallocate: host matrix = %p %s", matrix, minfo.toString().c_str());
  }
}

floatt GetReValue(const math::ComplexMatrix* matrix, uintt column, uintt row)
{
  if (gReValues (matrix) == nullptr)
  {
    return 0;
  }
  return gReValues (matrix)[row * gColumns (matrix) + column];
}

floatt GetImValue(const math::ComplexMatrix* matrix, uintt column, uintt row)
{
  if (gImValues (matrix) == nullptr)
  {
    return 0;
  }
  return gImValues (matrix)[row * gColumns (matrix) + column];
}

void SetReValue(const math::ComplexMatrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (gReValues (matrix))
  {
    gReValues (matrix)[row * gColumns (matrix) + column] = value;
  }
}

void SetImValue(const math::ComplexMatrix* matrix, uintt column, uintt row,
                floatt value)
{
  if (gImValues (matrix))
  {
    gImValues (matrix)[row * gColumns (matrix) + column] = value;
  }
}

std::string GetMatrixStr (const math::ComplexMatrix* matrix)
{
  std::string output;
  oap::generic::printMatrix (output, matrix, matrixUtils::PrintArgs(), oap::host::GetMatrixInfo);
  return output;
}

math::ComplexMatrix GetRefHostMatrix (const math::ComplexMatrix* matrix)
{
  if (!g_matricesList.contains (matrix))
  {
    oapAssert ("Not in list" == nullptr);
  }
  return *matrix;
}

void PrintMatrix(FILE* stream, const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix)
{
  std::string output;
  oap::generic::printMatrix (output, matrix, matrixUtils::PrintArgs(), oap::host::GetMatrixInfo);
  fprintf(stream, "%s", output.c_str());
}

void PrintMatrix(FILE* stream, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(stream, args, matrix);
}

void PrintMatrix(const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix)
{
  PrintMatrix(stdout, args, matrix);
}

void PrintMatrix(const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args)
{
  PrintMatrix(args, matrix);
}

bool PrintMatrixToFile(const std::string& path, const matrixUtils::PrintArgs& args, const math::ComplexMatrix* matrix)
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

bool PrintMatrixToFile(const std::string& path, const math::ComplexMatrix* matrix, const matrixUtils::PrintArgs& args)
{
  return PrintMatrixToFile(path, args, matrix);
}

void Copy(math::ComplexMatrix* dst, const math::ComplexMatrix* src, const MatrixEx& subMatrix, uintt column, uintt row)
{
  uintt rows = dst->re.mem.dims.width;
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

void Copy(math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt column, uintt row)
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

void CopyMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
#if 0
  oap::generic::MatrixMemoryApi<decltype(oap::host::GetMatrixInfo), decltype (oap::host::ToHost)> mmApi (oap::host::GetMatrixInfo, oap::host::ToHost);
  oap::generic::copyMatrixToMatrix (dst, src, memcpy, mmApi, mmApi);
#endif
  if (dst->re.mem.ptr && src->re.mem.ptr) {oap::host::CopyHostToHost (dst->re.mem, {0, 0}, src->re.mem, {{0, 0}, src->re.mem.dims});}
  if (dst->im.mem.ptr && src->im.mem.ptr) {oap::host::CopyHostToHost (dst->im.mem, {0, 0}, src->im.mem, {{0, 0}, src->im.mem.dims});}
}

void CopyMatrixRegion (math::ComplexMatrix* dst, const oap::MemoryLoc& dstLoc, const math::ComplexMatrix* src, const oap::MemoryRegion& srcReg)
{
  if (dst->re.mem.ptr && src->re.mem.ptr) {oap::host::CopyHostToHost (dst->re.mem, GetReMatrixMemoryLoc (dst, &dstLoc), src->re.mem, GetReMatrixMemoryRegion (src, &srcReg));}
  if (dst->im.mem.ptr && src->im.mem.ptr) {oap::host::CopyHostToHost (dst->im.mem, GetImMatrixMemoryLoc (dst, &dstLoc), src->im.mem, GetImMatrixMemoryRegion (src, &srcReg));}
}

void CopyRe(math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  const uintt length1 = gColumns (dst) * gRows (dst);
  const uintt length2 = gColumns (src) * gRows (src);
  const uintt length = length1 < length2 ? length1 : length2;
  if (ReIsNotNULL(dst) && ReIsNotNULL(src))
  {
    memcpy(gReValues (dst), gReValues (src), length * sizeof(floatt));
  }
}

void CopyIm(math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  const uintt length1 = gColumns (dst) * gRows (dst);
  const uintt length2 = gColumns (src) * gRows (src);
  const uintt length = length1 < length2 ? length1 : length2;
  if (ImIsNotNULL(dst) && ImIsNotNULL(src))
  {
    memcpy(gImValues (dst), gImValues (src), length * sizeof(floatt));
  }
}

math::ComplexMatrix* NewMatrixCopy(const math::ComplexMatrix* matrix)
{
  math::ComplexMatrix* output = oap::host::NewMatrixRef (matrix);
  oap::host::CopyMatrix(output, matrix);
  return output;
}

void SetVector(math::ComplexMatrix* matrix, uintt column, math::ComplexMatrix* vector)
{
  SetVector(matrix, column, gReValues (vector), gImValues (vector), gRows (vector));
}

void SetVector(math::ComplexMatrix* matrix, uintt column, floatt* revector,
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

void SetReVector(math::ComplexMatrix* matrix, uintt column, floatt* vector,
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

void SetTransposeReVector(math::ComplexMatrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (gReValues (matrix))
  {
    memcpy(&gReValues (matrix)[row * gColumns (matrix)], vector,
           length * sizeof(floatt));
  }
}

void SetImVector(math::ComplexMatrix* matrix, uintt column, floatt* vector,
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

void SetTransposeImVector(math::ComplexMatrix* matrix, uintt row, floatt* vector,
                          uintt length)
{
  if (gImValues (matrix))
  {
    memcpy(&gImValues (matrix)[row * gColumns (matrix)], vector,
           length * sizeof(floatt));
  }
}

void SetReVector(math::ComplexMatrix* matrix, uintt column, floatt* vector)
{
  SetReVector(matrix, column, vector, gRows (matrix));
}

void SetTransposeReVector(math::ComplexMatrix* matrix, uintt row, floatt* vector)
{
  SetTransposeReVector(matrix, row, vector, gColumns (matrix));
}

void SetImVector(math::ComplexMatrix* matrix, uintt column, floatt* vector)
{
  SetImVector(matrix, column, vector, gRows (matrix));
}

void SetTransposeImVector(math::ComplexMatrix* matrix, uintt row, floatt* vector)
{
  SetTransposeImVector(matrix, row, vector, gColumns (matrix));
}

void GetMatrixStr(std::string& text, const math::ComplexMatrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void ToString (std::string& str, const math::ComplexMatrix* matrix)
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

void GetReMatrixStr(std::string& text, const math::ComplexMatrix* matrix)
{
  matrixUtils::PrintMatrix(text, matrix, matrixUtils::PrintArgs());
}

void GetImMatrixStr(std::string& str, const math::ComplexMatrix* matrix)
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

void GetVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uintt column)
{
  GetVector(gReValues (vector), gImValues (vector), gRows (vector), matrix, column);
}

void GetVector(floatt* revector, floatt* imvector, uint length, math::ComplexMatrix* matrix, uint column)
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

void GetTransposeVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column)
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

void GetTransposeReVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column)
{
  GetTransposeReVector(gReValues (vector), matrix, column);
}

void GetTransposeImVector(math::ComplexMatrix* vector, math::ComplexMatrix* matrix, uint column)
{
  GetTransposeImVector(gImValues (vector), matrix, column);
}

void GetReVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint column)
{
  if (gReValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = gReValues (matrix)[column + gColumns (matrix) * fa];
    }
  }
}

void GetTransposeReVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint row)
{
  if (gReValues (matrix))
  {
    memcpy(vector, &gReValues (matrix)[row * gColumns (matrix)],
           length * sizeof(floatt));
  }
}

void GetImVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint column)
{
  if (gImValues (matrix))
  {
    for (uintt fa = 0; fa < length; fa++)
    {
      vector[fa] = gImValues (matrix)[column + gColumns (matrix) * fa];
    }
  }
}

void GetTransposeImVector(floatt* vector, uint length, math::ComplexMatrix* matrix, uint row)
{
  if (gImValues (matrix))
  {
    memcpy(vector, &gImValues (matrix)[row * gColumns (matrix)], length * sizeof(floatt));
  }
}

void GetReVector(floatt* vector, math::ComplexMatrix* matrix, uint column)
{
  GetReVector(vector, gRows (matrix), matrix, column);
}

void GetTransposeReVector(floatt* vector, math::ComplexMatrix* matrix, uint row)
{
  GetTransposeReVector(vector, gColumns (matrix), matrix, row);
}

void GetImVector(floatt* vector, math::ComplexMatrix* matrix, uint column)
{
  GetImVector(vector, gRows (matrix), matrix, column);
}

void GetTransposeImVector(floatt* vector, math::ComplexMatrix* matrix, uint row)
{
  GetTransposeReVector(vector, gColumns (matrix), matrix, row);
}

floatt SmallestDiff(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1)
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

floatt LargestDiff(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1)
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

void SetIdentity(math::ComplexMatrix* matrix)
{
  oap::host::SetDiagonalReMatrix(matrix, 1);
  oap::host::SetImZero(matrix);
}

void SetReZero(math::ComplexMatrix* matrix)
{
  if (gReValues (matrix))
  {
    memset(gReValues (matrix), 0,
           gColumns (matrix) * gRows (matrix) * sizeof(floatt));
  }
}

void SetImZero(math::ComplexMatrix* matrix)
{
  if (gImValues (matrix))
  {
    memset(gImValues (matrix), 0,
           gColumns (matrix) * gRows (matrix) * sizeof(floatt));
  }
}

void SetZero(math::ComplexMatrix* matrix)
{
  SetReZero(matrix);
  SetImZero(matrix);
}

void SetIdentityMatrix(math::ComplexMatrix* matrix)
{
  SetDiagonalReMatrix(matrix, 1);
  SetImZero(matrix);
}

bool IsEquals(math::ComplexMatrix* transferMatrix2, math::ComplexMatrix* transferMatrix1,
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

floatt GetTrace(math::ComplexMatrix* matrix)
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

void SetDiagonalMatrix(math::ComplexMatrix* matrix, floatt a)
{
  SetDiagonalReMatrix(matrix, a);
  SetDiagonalImMatrix(matrix, a);
}

void SetDiagonalReMatrix(math::ComplexMatrix* matrix, floatt a)
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

void SetDiagonalImMatrix(math::ComplexMatrix* matrix, floatt a)
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

math::MatrixInfo CreateMatrixInfo(const math::ComplexMatrix* matrix)
{
  return math::MatrixInfo (gReValues (matrix) != nullptr,
                           gImValues (matrix) != nullptr,
                           gColumns (matrix),
                           gRows (matrix));
}

math::MatrixInfo GetMatrixInfo (const math::ComplexMatrix* matrix)
{
  return g_matricesList.getUserData (matrix);
}

math::ComplexMatrix* ReadMatrix (const std::string& path)
{
  utils::ByteBuffer buffer (path);
  math::ComplexMatrix* matrix = oap::host::LoadMatrix (buffer);

  return matrix;
}

math::ComplexMatrix* ReadRowVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::ComplexMatrix* subMatrix = oap::host::NewSubMatrix (matrix, 0, index, gColumns (matrix), 1);
  return subMatrix;
}

math::ComplexMatrix* ReadColumnVector (const std::string& path, size_t index)
{
  oap::HostMatrixUPtr matrix = ReadMatrix (path);
  math::ComplexMatrix* subMatrix = oap::host::NewSubMatrix (matrix, index, 0, 1, gRows (matrix));
  return subMatrix;
}

void CopyReBuffer (math::ComplexMatrix* houtput, math::ComplexMatrix* hinput)
{
  size_t sOutput = gColumns (houtput) * gRows (houtput);
  size_t sInput = gColumns (hinput) * gRows (hinput);

  debugExceptionMsg(sOutput == sInput, "Buffers have different sizes.");

  memcpy (gReValues (houtput), gReValues (hinput), sOutput * sizeof (floatt));
}

bool WriteMatrix (const std::string& path, const math::ComplexMatrix* matrix)
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

void copySubMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt cindex, uintt rindex)
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

void CopySubMatrix(math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt cindex, uintt rindex)
{
  copySubMatrix (dst, src, cindex, rindex);
}

inline uintt calculate (uintt matrixd, uintt dindex, uintt dlength)
{
  return dindex + dlength < matrixd ? dlength : matrixd - dindex;
}

math::ComplexMatrix* NewSubMatrix (const math::ComplexMatrix* orig, uintt cindex, uintt rindex, uintt clength, uintt rlength)
{
  clength = calculate (gColumns (orig), cindex, clength);
  rlength = calculate (gRows (orig), rindex, rlength);

  math::ComplexMatrix* submatrix = oap::host::NewMatrix (orig, clength, rlength);
  copySubMatrix (submatrix, orig, cindex, rindex);
  return submatrix;
}

math::ComplexMatrix* GetSubMatrix (const math::ComplexMatrix* orig, uintt cindex, uintt rindex, math::ComplexMatrix* matrix)
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

void SaveMatrix (const math::ComplexMatrix* matrix, utils::ByteBuffer& buffer)
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

math::ComplexMatrix* LoadMatrix (const utils::ByteBuffer& buffer)
{
  bool isMatrix = buffer.read <bool>();

  if (!isMatrix)
  {
    return nullptr;
  }

  math::MatrixInfo minfo = LoadMatrixInfo (buffer);
  math::ComplexMatrix* matrix = NewMatrix (minfo);

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

void CopyArrayToMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer)
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

void CopyArrayToReMatrix (math::ComplexMatrix* matrix, const floatt* buffer)
{
  debugAssert (gReValues (matrix) != nullptr);
  memcpy (gReValues (matrix), buffer, gColumns (matrix) * gRows (matrix) * sizeof(floatt));
}

void CopyArrayToImMatrix (math::ComplexMatrix* matrix, const floatt* buffer)
{
  debugAssert (gImValues (matrix) != nullptr);
  memcpy (gImValues (matrix), buffer, gColumns (matrix) * gRows (matrix) * sizeof(floatt));
}

void CopyHostArrayToHostMatrix (math::ComplexMatrix* matrix, const floatt* rebuffer, const floatt* imbuffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToMatrix (matrix, rebuffer, imbuffer);
}

void CopyHostArrayToHostReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToReMatrix (matrix, buffer);
}

void CopyHostArrayToHostImMatrix (math::ComplexMatrix* matrix, const floatt* buffer, size_t length)
{
  debugAssert (gColumns (matrix) * gRows (matrix) == length);
  CopyArrayToImMatrix (matrix, buffer);
}

void SetSubs(math::ComplexMatrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumns (matrix, subcolumns);
  SetSubRows (matrix, subrows);
}

void SetSubColumns(math::ComplexMatrix* matrix, uintt subcolumns)
{
  if (subcolumns != MATH_UNDEFINED)
  {
    if (matrix->re.mem.ptr != nullptr) { matrix->dim.columns = subcolumns; matrix->re.reg.dims.width = subcolumns; }
    if (matrix->im.mem.ptr != nullptr) { matrix->dim.columns = subcolumns; matrix->im.reg.dims.width = subcolumns; }
  }
  else
  {
    if (matrix->re.mem.ptr != nullptr) { matrix->dim.columns = matrix->re.mem.dims.width; matrix->re.reg.dims.width = matrix->im.mem.dims.width; }
    if (matrix->im.mem.ptr != nullptr) { matrix->dim.columns = matrix->im.mem.dims.width; matrix->im.reg.dims.width = matrix->im.mem.dims.width; }
  }
}

void SetSubRows(math::ComplexMatrix* matrix, uintt subrows)
{
  if (subrows != MATH_UNDEFINED)
  {
    if (matrix->re.mem.ptr != nullptr) { matrix->dim.rows = subrows; matrix->re.reg.dims.height = subrows; }
    if (matrix->im.mem.ptr != nullptr) { matrix->dim.rows = subrows; matrix->im.reg.dims.height = subrows; }
  }
  else
  {
    if (matrix->re.mem.ptr != nullptr) { matrix->dim.rows = matrix->re.mem.dims.height; matrix->re.reg.dims.height = matrix->re.mem.dims.height; }
    if (matrix->im.mem.ptr != nullptr) { matrix->dim.rows = matrix->im.mem.dims.height; matrix->im.reg.dims.height = matrix->im.mem.dims.height; }
  }
}

void SetSubsSafe(math::ComplexMatrix* matrix, uintt subcolumns, uintt subrows)
{
  SetSubColumnsSafe(matrix, subcolumns);
  SetSubRowsSafe(matrix, subrows);
}

void SetSubColumnsSafe(math::ComplexMatrix* matrix, uintt subcolumns)
{
  if (subcolumns == MATH_UNDEFINED || gColumns (matrix) < subcolumns)
  {
    matrix->re.reg = {{0, 0}, {0, 0}};
    matrix->im.reg = {{0, 0}, {0, 0}};
  }
  else
  {
    oap::MemoryRegion reg = {{0, 0}, {gColumns (matrix), gRows (matrix)}};
    reg.dims.width = subcolumns;

    (matrix->re.reg) = reg;
    (matrix->im.reg) = reg;
  }
}

void SetSubRowsSafe(math::ComplexMatrix* matrix, uintt subrows)
{
  if (subrows == MATH_UNDEFINED || gRows (matrix) < subrows)
  {
    matrix->re.reg = {{0, 0}, {0, 0}};
    matrix->im.reg = {{0, 0}, {0, 0}};
  }
  else
  {
    oap::MemoryRegion reg = {{0, 0}, {gColumns (matrix), gRows (matrix)}};
    reg.dims.height = subrows;

    (matrix->re.reg) = reg;
    (matrix->im.reg) = reg;
  }
}

void SetMatrix(math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  SetReMatrix (matrix, matrix1, column, row);
  SetImMatrix (matrix, matrix1, column, row);
}

void SetReMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  oap::generic::setMatrix (matrix, matrix1, column, row, [](math::ComplexMatrix* matrix) { return matrix->re.mem; }, [](math::ComplexMatrix* matrix) { return matrix->re.reg; }, memcpy);
}

void SetImMatrix (math::ComplexMatrix* matrix, math::ComplexMatrix* matrix1, uintt column, uintt row)
{
  oap::generic::setMatrix (matrix, matrix1, column, row, [](math::ComplexMatrix* matrix) { return matrix->im.mem; }, [](math::ComplexMatrix* matrix) { return matrix->im.reg; }, memcpy);
}

oap::ThreadsMapper CreateThreadsMapper (const std::vector<std::vector<math::ComplexMatrix*>>& matricesVec, oap::threads::ThreadsMapperAlgo algo)
{
  return createThreadsMapper (matricesVec, algo);
}

namespace
{

inline math::ComplexMatrix* allocReMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::ComplexMatrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re.mem = oap::host::ReuseMemory (mem);
  hostRefMatrix.re.reg = reg;
  hostRefMatrix.im.mem = {nullptr, {0, 0}};
  hostRefMatrix.im.reg = {{0, 0}, {0, 0}};

  return allocMatrix (hostRefMatrix);
}

inline math::ComplexMatrix* allocImMatrix_FromMemory (oap::Memory& mem, const oap::MemoryRegion& reg)
{
  math::ComplexMatrix hostRefMatrix;
  hostRefMatrix.dim = {reg.dims.width, reg.dims.height};

  hostRefMatrix.re.mem = {nullptr, {0, 0}};
  hostRefMatrix.re.reg = {{0, 0}, {0, 0}};
  hostRefMatrix.im.mem = oap::host::ReuseMemory (mem);
  hostRefMatrix.im.reg = reg;

  return allocMatrix (hostRefMatrix);
}

inline math::ComplexMatrix* allocRealMatrix_FromMemory (oap::Memory& remem, const oap::MemoryRegion& rereg, oap::Memory& immem, const oap::MemoryRegion& imreg)
{
  oapAssert (rereg.dims == imreg.dims);

  math::ComplexMatrix hostRefMatrix;

  hostRefMatrix.dim = {rereg.dims.width, rereg.dims.height};

  hostRefMatrix.re.mem = oap::host::ReuseMemory (remem);
  hostRefMatrix.re.reg = rereg;
  hostRefMatrix.im.mem = oap::host::ReuseMemory (immem);
  hostRefMatrix.im.reg = imreg;

  return allocMatrix (hostRefMatrix);
}

}

math::ComplexMatrix* NewMatrixFromMemory (uintt columns, uintt rows, oap::Memory& remem, const oap::MemoryLoc& reloc, oap::Memory& immem, const oap::MemoryLoc& imloc)
{
  return allocRealMatrix_FromMemory (remem, {reloc, {columns, rows}}, immem, {imloc, {columns, rows}});
}

math::ComplexMatrix* NewReMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocReMatrix_FromMemory (memory, {loc, {columns, rows}});
}

math::ComplexMatrix* NewImMatrixFromMemory (uintt columns, uintt rows, oap::Memory& memory, const oap::MemoryLoc& loc)
{
  return allocImMatrix_FromMemory (memory, {loc, {columns, rows}});
}

void SetZeroRow (const math::ComplexMatrix* matrix, uintt index, bool re, bool im)
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

void SetReZeroRow (const math::ComplexMatrix* matrix, uintt index)
{
  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.re.mem.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.re.mem, hm.re.reg, {index, 0});
    oap::generic::copy (hm.re.mem.ptr, hm.re.mem.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, memcpy);
  }
}

void SetImZeroRow (const math::ComplexMatrix* matrix, uintt index)
{
  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.im.mem.ptr)
  {
    uintt columns = gColumns (&hm);
    std::vector<floatt> row(columns, 0.);
    oap::MemoryLoc loc = oap::common::ConvertRegionLocToMemoryLoc (hm.im.mem, hm.im.reg, {index, 0});
    oap::generic::copy (hm.im.mem.ptr, hm.im.mem.dims, loc, row.data(), {1, columns}, {{0, 0}, {1, columns}}, memcpy);
  }
}

void SetValueToMatrix (math::ComplexMatrix* matrix, floatt re, floatt im)
{
  SetValueToReMatrix (matrix, re);
  SetValueToImMatrix (matrix, im);
}

void SetValueToReMatrix (math::ComplexMatrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.re.mem.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewReMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetReMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetReMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.re.mem.ptr, hm.re.mem.dims, loc, uptr->re.mem.ptr, uptr->re.mem.dims, reg, memcpy);
  }
}

void SetValueToImMatrix (math::ComplexMatrix* matrix, floatt v)
{
  using namespace oap::utils;

  math::ComplexMatrix hm = GetRefHostMatrix (matrix);

  if (hm.im.mem.ptr)
  {
    auto minfo = GetMatrixInfo (matrix);
    oap::HostMatrixUPtr uptr = oap::host::NewImMatrixWithValue (minfo.columns(), minfo.rows(), v);

    oap::MemoryLoc loc = GetImMatrixMemoryLoc (&hm);
    oap::MemoryRegion reg = GetImMatrixMemoryRegion (uptr);
    oap::generic::copy (hm.im.mem.ptr, hm.im.mem.dims, loc, uptr->im.mem.ptr, uptr->im.mem.dims, reg, memcpy);
  }
}

void SetZeroMatrix (math::ComplexMatrix* matrix)
{
  SetValueToMatrix (matrix, 0, 0);
}

void SetZeroReMatrix (math::ComplexMatrix* matrix)
{
  SetValueToReMatrix (matrix, 0);
}

void SetZeroImMatrix (math::ComplexMatrix* matrix)
{
  SetValueToImMatrix (matrix, 0);
}

floatt GetReDiagonal (const math::ComplexMatrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::host::GetRefHostMatrix,
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return matrix->re.mem;},
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return matrix->re.reg;}, memcpy);
}

floatt GetImDiagonal (const math::ComplexMatrix* matrix, uintt index)
{
  return oap::generic::getDiagonal (matrix, index, oap::host::GetRefHostMatrix,
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return matrix->im.mem;},
                                    [](const math::ComplexMatrix* matrix, const math::ComplexMatrix& ref){return matrix->im.reg;}, memcpy);
}

void CopyReMatrixToHostBuffer (floatt* buffer, uintt length, const math::ComplexMatrix* matrix)
{
  math::ComplexMatrix ref = oap::host::GetRefHostMatrix (matrix);
  oap::host::CopyHostToHostBuffer (buffer, length, ref.re.mem, ref.re.reg);
}

void CopyHostBufferToReMatrix (math::ComplexMatrix* matrix, const floatt* buffer, uintt length)
{
  math::ComplexMatrix ref = oap::host::GetRefHostMatrix (matrix);
  oap::host::CopyHostBufferToHost (ref.re.mem, ref.re.reg, buffer, length);
}

std::string to_carraystr(const math::ComplexMatrix* matrix)
{
  std::string str;
  if (matrix == nullptr)
  {
    str = "nullptr";
    return str;
  }

  matrixUtils::PrintArgs args;
  args.prepareSection (matrix);
  args.section.separator = ",";
  args.leftBracket = "{";
  args.rightBracket = "}";
  args.repeats = true;

  oap::generic::printMatrix (str, matrix, args, oap::host::GetMatrixInfo);
  return str;
}

std::string to_carraystr(const std::vector<math::ComplexMatrix*>& matrices)
{
  std::vector<std::string> strs;
  for (const math::ComplexMatrix* matrix : matrices)
  {
    strs.push_back (oap::host::to_carraystr (matrix));
  }
  std::stringstream sstream;
  sstream << "{";
  for (uintt idx = 0; idx < strs.size(); ++idx)
  {
    std::string str = strs[idx];
    sstream << str;
    if (idx < strs.size() - 1)
    {
      sstream << ", ";
    }
  }
  sstream << "}";
  return sstream.str();
}

std::vector<math::ComplexMatrix*> NewMatrices (const std::vector<math::MatrixInfo>& minfos)
{
  std::vector<math::ComplexMatrix*> matrices;
  for (const auto& minfo : minfos)
  {
    math::ComplexMatrix* matrix = oap::host::NewHostMatrixFromMatrixInfo (minfo);
    matrices.push_back (matrix);
  }
  return matrices;
}

std::vector<math::ComplexMatrix*> NewMatrices (const math::MatrixInfo& minfo, uintt count)
{
  std::vector<math::MatrixInfo> minfos (count, minfo);
  return NewMatrices (minfos);
}

std::vector<math::ComplexMatrix*> NewMatricesCopyOfArray (const std::vector<math::MatrixInfo>& minfos, const std::vector<std::vector<floatt>>& arrays)
{
  std::vector<math::ComplexMatrix*> matrices;
  oapAssert (arrays.size() == minfos.size());
  for (uintt idx = 0; idx < minfos.size(); ++idx)
  {
    auto& minfo = minfos[idx];
    if (minfo.isRe && minfo.isIm)
    {
      math::ComplexMatrix* matrix = NewMatrixCopyOfArray (minfo.columns(), minfo.rows(), arrays[idx].data(), arrays[idx].data());
      matrices.push_back (matrix);
    }
    else if(minfo.isRe)
    {
      math::ComplexMatrix* matrix = NewReMatrixCopyOfArray (minfo.columns(), minfo.rows(), arrays[idx].data());
      matrices.push_back (matrix);
    }
    else if(minfo.isIm)
    {
      math::ComplexMatrix* matrix = NewImMatrixCopyOfArray (minfo.columns(), minfo.rows(), arrays[idx].data());
      matrices.push_back (matrix);
    }
    else
    {
      oapAssert ("not supported" == nullptr);
    }
  }
  return matrices;
}

std::vector<math::ComplexMatrix*> NewMatricesCopyOfArray(const math::MatrixInfo& minfo, const std::vector<std::vector<floatt>>& arrays)
{
  std::vector<math::MatrixInfo> minfos (arrays.size(), minfo);
  return NewMatricesCopyOfArray (minfos, arrays);  
}

math::ComplexMatrix* NewSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& dim, const math::ComplexMatrix* matrix)
{
  auto minfo = oap::host::GetMatrixInfo (matrix);

  oapAssert (loc.x < minfo.columns());
  oapAssert (loc.y < minfo.rows());
  oapAssert (loc.x + dim.columns <= minfo.columns());
  oapAssert (loc.y + dim.rows <= minfo.rows());

  math::ComplexMatrix refmatrix = oap::host::GetRefHostMatrix (matrix);
  math::ComplexMatrix* output = nullptr;

  if (minfo.isRe && minfo.isIm)
  {
    oap::MemoryLoc reloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.re.mem, refmatrix.re.reg, {loc.x, loc.y});
    oap::MemoryLoc imloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.im.mem, refmatrix.im.reg, {loc.x, loc.y});

    output = oap::host::NewMatrixFromMemory (dim.columns, dim.rows, refmatrix.re.mem, reloc, refmatrix.im.mem, imloc);
  }
  else if (minfo.isRe)
  {
    oap::MemoryLoc reloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.re.mem, refmatrix.re.reg, {loc.x, loc.y});
    output = oap::host::NewReMatrixFromMemory (dim.columns, dim.rows, refmatrix.re.mem, reloc);
  }
  else if (minfo.isIm)
  {
    oap::MemoryLoc imloc = oap::common::ConvertRegionLocToMemoryLoc (refmatrix.im.mem, refmatrix.im.reg, {loc.x, loc.y});
    output = oap::host::NewImMatrixFromMemory (dim.columns, dim.rows, refmatrix.im.mem, imloc);
  }

  return output;
}

math::ComplexMatrix* NewSharedSubMatrix (const math::MatrixDim& dim, const math::ComplexMatrix* matrix)
{
  return NewSharedSubMatrix ({0,0}, dim, matrix);
}

}
}
