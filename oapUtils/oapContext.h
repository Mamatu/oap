/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef OAP_CONTEXT_H
#define OAP_CONTEXT_H

#include "Logger.h"

#include "MatrixInfo.h"
#include "Matrix.h"

#include <functional>
#include <map>
#include <set>


namespace oap { namespace generic {

class Context
{
  private:

    using Allocator = std::function<math::Matrix*(const math::MatrixInfo& minfo)>;
    using Deallocator = std::function<void(const math::Matrix*)>;
    
    struct MemType
    {
      std::string name;
      Allocator allocator;
      Deallocator deallocator;
    };

    struct MapEntry
    {
      math::Matrix* matrix;
      bool isUsed;
      std::string memTypeName;
    };

    std::multimap<math::MatrixInfo, MapEntry> m_matrices;
    std::map<std::string, MemType> m_memTypes;

  public:

    virtual ~Context ()
    {
      clear ();
    }

    class Getter final
    {
      private:
        Context& m_context;
        std::set<math::Matrix*> m_matrices;

        Getter() = delete;

        Getter (Context& context) : m_context (context)
        {}

      public:

        inline math::Matrix* useMatrix (const math::MatrixInfo& minfo, const std::string& memTypeName)
        {
          math::Matrix* matrix = m_context.useMatrix (minfo, memTypeName);
          m_matrices.insert (matrix);
          return  matrix;
        }

        inline math::Matrix* useMatrix (bool isRe, bool isIm, uintt columns, uintt rows, const std::string& memTypeName)
        {
          return useMatrix (math::MatrixInfo (isRe, isIm, columns, rows), memTypeName);
        }

        ~Getter ()
        {
          m_context.unuseMatrices (m_matrices);
          m_matrices.clear();
        }

        friend class Context;
    };

    inline void registerMemType (const std::string& typeName, Allocator&& allocator, Deallocator&& deallocator)
    {
      MemType memType = {typeName, allocator, deallocator};
      m_memTypes[typeName] = memType; 
    }

    Getter getter ()
    {
      return Getter (*this);
    }

    inline math::Matrix* useMatrix (const math::MatrixInfo& minfo, const std::string& memTypeName)
    {
      math::Matrix* matrix = nullptr;
      auto range = m_matrices.equal_range (minfo);

      for (auto it = range.first; it != range.second; ++it)
      {
        auto& entry = it->second;
        const bool isused = entry.isUsed;
        const std::string& ememTypeName = entry.memTypeName;
        if (!isused && ememTypeName == memTypeName)
        {
          matrix = entry.matrix;
          entry.isUsed = true;
        }
      }

      if (matrix == nullptr)
      {
        auto it = m_memTypes.find (memTypeName);

        debugAssertMsg (it != m_memTypes.end(), "Not registered memTypeName: %s", memTypeName.c_str());

        const MemType& memType = it->second;

        matrix = memType.allocator (minfo);

        MapEntry mapEntry = {matrix, true, memTypeName};

        m_matrices.insert (std::make_pair (minfo, mapEntry));
      }

      return matrix;
    }

    inline void unuseAllMatrices ()
    {
      for (auto& kv : m_matrices)
      {
        kv.second.isUsed = false;
      }
    }

    template<typename Matrices>
    inline void unuseMatrices (const Matrices& matrices)
    {
      for (auto& kv : m_matrices)
      {
        if (matrices.find (kv.second.matrix) != matrices.end())
        {
          kv.second.isUsed = false;
        }
      }
    }

    inline void clear ()
    {
      for (auto& kv : m_matrices)
      {
        const MemType& memType = m_memTypes[kv.second.memTypeName];
        memType.deallocator (kv.second.matrix);
      }
      m_matrices.clear();
      m_memTypes.clear();
    }
};

}}

#endif
