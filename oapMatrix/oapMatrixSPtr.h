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

#ifndef OAP_MATRIXSPTR_H
#define OAP_MATRIXSPTR_H

#include <memory>
#include <list>
#include <type_traits>

#include "Math.h"
#include "Matrix.h"

#include "oapSmartPointerUtils.h"

namespace oap
{
namespace stdlib
{
using MatrixSharedPtr = ::std::shared_ptr<math::Matrix>;

using MatrixUniquePtr = ::std::unique_ptr<math::Matrix, deleters::MatrixDeleter<math::Matrix>>;

using ComplexMatrixSharedPtr = ::std::shared_ptr<math::ComplexMatrix>;

using ComplexMatrixUniquePtr = ::std::unique_ptr<math::ComplexMatrix, deleters::MatrixDeleter<math::ComplexMatrix>>;
}

namespace
{
template<typename Dst, typename Src>
class NormalCopy
{
  public:
    bool copy (Dst& dst, const Src& src) const
    {
      dst = src;
      return true;
    }
};

template<typename Dst, typename Src>
class NotCopy
{
  public:
    bool copy (Dst& dst, const Src& src) const
    {
      static_assert("unique_ptr(const unique_ptr&) = delete" != nullptr, "unique_ptr(const unique_ptr&) = delete");
      return false;
    }
};
}

template<typename MatrixT, typename DeleterT>
class SMDeleterWrapper final
{
  public:
    SMDeleterWrapper (DeleterT&& deleter, bool deallocate) : m_deleter (std::forward<DeleterT>(deleter)), m_bDeallocate (deallocate)
    {}

    SMDeleterWrapper (const SMDeleterWrapper&) = default;
    SMDeleterWrapper (SMDeleterWrapper&&) = default;
    SMDeleterWrapper& operator= (const SMDeleterWrapper&) = default;
    SMDeleterWrapper& operator= (SMDeleterWrapper&&) = default;

    ~SMDeleterWrapper () = default;

    void operator()(MatrixT* matrix)
    {
      if (m_bDeallocate)
      {
        m_deleter (matrix);
      }
    }

    DeleterT getDeleterCopy () const
    {
      return m_deleter;
    }

    void enable (bool b)
    {
      m_bDeallocate = b;
    }

    bool willBeDeallocated () const
    {
      return m_bDeallocate;
    }

  private:
    DeleterT m_deleter;
    bool m_bDeallocate;
};

template<typename MatrixSPtrT, typename MatrixT, typename DeleterT>
class MatrixSPtrWrapper : public MatrixSPtrT
{
  public:
    using CorePtrType = MatrixSPtrT;

    MatrixSPtrWrapper (MatrixT* ptr, DeleterT&& deleter, bool deallocate = true) : MatrixSPtrT (ptr, SMDeleterWrapper<MatrixT, DeleterT>(std::forward<DeleterT>(deleter), deallocate))
    {}

    MatrixSPtrWrapper (MatrixSPtrWrapper&& orig) = default;
    MatrixSPtrWrapper (const MatrixSPtrWrapper& orig) = default;
    MatrixSPtrWrapper& operator=(const MatrixSPtrWrapper& orig) = default;
    MatrixSPtrWrapper& operator=(MatrixSPtrWrapper&& orig) = default;

    virtual ~MatrixSPtrWrapper () = default;

    void reset (MatrixT* matrix, bool deallocate = true)
    {
      resetIt (matrix);
      getDeleterWrapper().enable (deallocate);
    }

    bool isDeallocated () const
    {
      return getDeleterWrapper().willBeDeallocated();
    }
  protected:
    virtual void resetIt (MatrixT* matrix) = 0;
    virtual SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() = 0;
    virtual const SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const = 0;
};

template<typename MatrixT, typename DeleterT>
class MMDeleterWrapper final
{
  public:
    MMDeleterWrapper (DeleterT&& deleter, size_t count, bool deallocate) : m_deleter (std::forward<DeleterT>(deleter)), m_count(count), m_bDeallocate (deallocate)
    {}

    MMDeleterWrapper (const MMDeleterWrapper&) = default;
    MMDeleterWrapper (MMDeleterWrapper&&) = default;
    MMDeleterWrapper& operator= (const MMDeleterWrapper&) = default;
    MMDeleterWrapper& operator= (MMDeleterWrapper&&) = default;

    ~MMDeleterWrapper () = default;

    void operator()(MatrixT** matrices)
    {
      if (m_bDeallocate)
      {
        m_deleter (matrices, m_count);
      }
      delete[] matrices;
    }

    void setCount (size_t count)
    {
      m_count = count;
    }

    void enable (bool b)
    {
      m_bDeallocate = b;
    }

    bool willBeDeallocated () const
    {
      return m_bDeallocate;
    }

    DeleterT getDeleterCopy() const
    {
      return m_deleter;
    }

  private:
    DeleterT m_deleter;
    size_t m_count;
    bool m_bDeallocate;
};

template<typename MatricesSPtrT, typename MatrixT, typename DeleterT>
class MatricesSPtrWrapper : public MatricesSPtrT
{
  public:
    using CorePtrType = MatricesSPtrT;
    using ElementType = typename MatricesSPtrT::element_type;
    using Deleter = MMDeleterWrapper<MatrixT, DeleterT>;
    using BasePtr = MatricesSPtrT;

    MatricesSPtrWrapper (MatrixT** ptr, size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrT (make(ptr, count), MMDeleterWrapper<MatrixT, DeleterT>(std::forward<DeleterT>(deleter), count, deallocate))
    {}

    MatricesSPtrWrapper (std::initializer_list<MatrixT*> list, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper (list.size(), std::forward<DeleterT>(deleter), deallocate)
    {
      for (auto it = list.begin(); it != list.end(); ++it)
      {
        auto idx = std::distance (list.begin(), it);
        this->get()[idx] = *it;
      }
    }

    MatricesSPtrWrapper (size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrT (new MatrixT*[count], MMDeleterWrapper<MatrixT, DeleterT>(std::forward<DeleterT>(deleter), count, deallocate))
    {}

    MatricesSPtrWrapper (MatricesSPtrWrapper&& orig) = default;
    MatricesSPtrWrapper (const MatricesSPtrWrapper& orig) = default;
    MatricesSPtrWrapper& operator=(const MatricesSPtrWrapper& orig) = default;
    MatricesSPtrWrapper& operator=(MatricesSPtrWrapper&& orig) = default;

    virtual ~MatricesSPtrWrapper () = default;

    static MatrixT** make (MatrixT** array, size_t count)
    {
      MatrixT** matrices = new MatrixT*[count];
      memcpy(matrices, array, count * sizeof (MatrixT*));
      return matrices;
    }

    template<typename T, typename Container>
    static T make (const Container& container, bool deallocate = true) {
      Container copy;
      std::copy(container.begin(), container.end(), std::back_inserter(copy));
      return T (copy.data(), container.size(), deallocate);
    }

    template<typename T>
    static T make (const std::list<MatrixT*>& list, bool deallocate = true) {
      std::vector<MatrixT*> copy;
      std::copy (list.begin(), list.end(), std::back_inserter(copy));
      return T (copy.data(), list.size(), deallocate);
    }

    MatrixT*& operator[](std::size_t idx)
    {
      return (this->get())[idx];
    }

    operator MatrixT**() const
    {
      return this->get();
    }

    template<typename Container>
    void reset (const Container& container, bool deallocate = true)
    {
      Container copy;
      std::copy (container.begin(), container.end(), std::back_inserter(copy));
      
      reset (copy.data(), copy.size(), deallocate);
    }

    void reset (std::initializer_list<MatrixT*> list, bool deallocate = true)
    {
      std::vector<MatrixT*> copy;
      std::copy (list.begin(), list.end(), std::back_inserter (copy));
      reset (copy.data(), list.size(), deallocate);
    }

    void reset (MatrixT** matrices, size_t count, bool deallocate = true)
    {
      MatrixT** copy = new MatrixT*[count];
      memcpy (copy, matrices, count * sizeof (MatrixT*));

      resetIt (copy);
      getDeleterWrapper().setCount (count);
      getDeleterWrapper().enable (deallocate);
    }

    bool isDeallocated () const
    {
      return getDeleterWrapper().willBeDeallocated();
    }

    size_t count () const
    {
      return getDeleterWrapper().count ();
    }

  protected:
    virtual void resetIt (MatrixT** matrices) = 0;
    virtual MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() = 0;
    virtual const MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const = 0;
};

template<typename MatrixT>
using SharedMatrixBase = std::shared_ptr<MatrixT>;

template<typename MatrixT>
using SharedMatricesBase = std::shared_ptr<MatrixT*>;

template<typename MatrixT, typename DeleterT>
class MatrixSharedWrapper : public MatrixSPtrWrapper<SharedMatrixBase<MatrixT>, MatrixT, DeleterT>
{
  public:
    MatrixSharedWrapper (MatrixT* ptr, DeleterT&& deleter, bool deallocate = true) : MatrixSPtrWrapper<SharedMatrixBase<MatrixT>, MatrixT, DeleterT> (ptr, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatrixSharedWrapper (const MatrixSharedWrapper& orig) = default;
    MatrixSharedWrapper (MatrixSharedWrapper&& orig) = default;
 
    MatrixSharedWrapper& operator=(const MatrixSharedWrapper& orig) = default;
    MatrixSharedWrapper& operator=(MatrixSharedWrapper&& orig) = default;

    virtual ~MatrixSharedWrapper () = default;

  protected:
    virtual void resetIt (MatrixT* matrix) override
    {
      SMDeleterWrapper<MatrixT, DeleterT> deleterWrapper = getDeleterWrapper();
      SharedMatrixBase<MatrixT>::reset (matrix, deleterWrapper);
    }

    virtual SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() override
    {
      auto* internalDeleter = std::get_deleter<SMDeleterWrapper<MatrixT, DeleterT>, MatrixT>(*this);
      return *internalDeleter;
    }

    virtual const SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const override
    {
      auto* internalDeleter = std::get_deleter<SMDeleterWrapper<MatrixT, DeleterT>, MatrixT>(*this);
      return *internalDeleter;
    }
};

template<typename MatrixT, typename DeleterT>
class MatricesSharedWrapper : public MatricesSPtrWrapper<std::shared_ptr<MatrixT*>, MatrixT, DeleterT>
{
  public:
    MatricesSharedWrapper (MatrixT** ptr, size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<SharedMatricesBase<MatrixT>, MatrixT, DeleterT> (ptr, count, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatricesSharedWrapper (std::initializer_list<MatrixT*> list, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<SharedMatricesBase<MatrixT>, MatrixT, DeleterT> (list, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatricesSharedWrapper (size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<SharedMatricesBase<MatrixT>, MatrixT, DeleterT> (count, std::forward<DeleterT>(deleter), deallocate)
    {}
 
    MatricesSharedWrapper (const MatricesSharedWrapper& orig) = default;
    MatricesSharedWrapper (MatricesSharedWrapper&& orig) = default;
 
    MatricesSharedWrapper& operator=(const MatricesSharedWrapper& orig) = default;
    MatricesSharedWrapper& operator=(MatricesSharedWrapper&& orig) = default;

  protected:
    virtual void resetIt (MatrixT** matrices) override
    {
      MMDeleterWrapper<MatrixT, DeleterT> deleterWrapper = getDeleterWrapper();
      std::shared_ptr<MatrixT*>::reset (matrices, deleterWrapper);
    }

    virtual MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() override
    {
      auto* interalDeleter = std::get_deleter<MMDeleterWrapper<MatrixT, DeleterT>, MatrixT*>(*this);
      return *interalDeleter;
    }

    virtual const MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const override
    {
      auto* interalDeleter = std::get_deleter<MMDeleterWrapper<MatrixT, DeleterT>, MatrixT*>(*this);
      return *interalDeleter;
    }
};

template<typename MatrixT, typename DeleterT>
using UniqueMatrixBase = std::unique_ptr<MatrixT, SMDeleterWrapper<MatrixT, DeleterT>>;

template<typename MatrixT, typename DeleterT>
using UniqueMatricesBase = std::unique_ptr<MatrixT*, MMDeleterWrapper<MatrixT, DeleterT>>;

template<typename MatrixT, typename DeleterT>
class MatrixUniqueWrapper : public MatrixSPtrWrapper<UniqueMatrixBase<MatrixT,DeleterT>, MatrixT, DeleterT>
{
  public:
    MatrixUniqueWrapper (MatrixT* ptr, DeleterT&& deleter, bool deallocate = true) : MatrixSPtrWrapper<UniqueMatrixBase<MatrixT, DeleterT>, MatrixT, DeleterT> (ptr, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatrixUniqueWrapper (const MatrixUniqueWrapper& orig) = default;
    MatrixUniqueWrapper (MatrixUniqueWrapper&& orig) = default;
 
    MatrixUniqueWrapper& operator=(const MatrixUniqueWrapper& orig) = default;
    MatrixUniqueWrapper& operator=(MatrixUniqueWrapper&& orig) = default;

    virtual ~MatrixUniqueWrapper () = default;

  protected:
    virtual void resetIt (MatrixT* matrix) override
    {
      UniqueMatrixBase<MatrixT, DeleterT>::reset (matrix);
    }

    virtual SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() override
    {
      auto& internalDeleter = this->get_deleter();
      return internalDeleter;
    }

    virtual const SMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const override
    {
      auto& internalDeleter = this->get_deleter();
      return internalDeleter;
    }
};

template<typename MatrixT, typename DeleterT>
class MatricesUniqueWrapper : public MatricesSPtrWrapper<UniqueMatricesBase<MatrixT,DeleterT>, MatrixT, DeleterT>
{
  public:
    MatricesUniqueWrapper (MatrixT** ptr, size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<UniqueMatricesBase<MatrixT, DeleterT>, MatrixT, DeleterT> (ptr, count, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatricesUniqueWrapper (std::initializer_list<MatrixT*> list, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<UniqueMatricesBase<MatrixT, DeleterT>, MatrixT, DeleterT> (list, std::forward<DeleterT>(deleter), deallocate)
    {}
  
    MatricesUniqueWrapper (size_t count, DeleterT&& deleter, bool deallocate = true) : MatricesSPtrWrapper<UniqueMatricesBase<MatrixT, DeleterT>, MatrixT, DeleterT> (count, std::forward<DeleterT>(deleter), deallocate)
    {}

    MatricesUniqueWrapper (const MatricesUniqueWrapper& orig) = default;
    MatricesUniqueWrapper (MatricesUniqueWrapper&& orig) = default;
 
    MatricesUniqueWrapper& operator=(const MatricesUniqueWrapper& orig) = default;
    MatricesUniqueWrapper& operator=(MatricesUniqueWrapper&& orig) = default;

  protected:
    virtual void resetIt(MatrixT** matrices) override
    {
      UniqueMatricesBase<MatrixT,DeleterT>::reset (matrices);
    }

    virtual MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() override
    {
      auto& internalDeleter = this->get_deleter();
      return internalDeleter;
    }

    virtual const MMDeleterWrapper<MatrixT, DeleterT>& getDeleterWrapper() const override
    {
      auto& internalDeleter = this->get_deleter();
      return internalDeleter;
    }
};

using ComplexMatricesDeleter = std::function<void(oap::math::ComplexMatrix**, size_t count)>;

using ComplexMatricesSharedPtr = MatricesSharedWrapper<oap::math::ComplexMatrix, ComplexMatricesDeleter>;
using ComplexMatricesUniquePtr = MatricesUniqueWrapper<oap::math::ComplexMatrix, ComplexMatricesDeleter>;

using MatricesDeleter = std::function<void(oap::math::Matrix**, size_t count)>;

using MatricesSharedPtr = MatricesSharedWrapper<oap::math::Matrix, MatricesDeleter>;
using MatricesUniquePtr = MatricesUniqueWrapper<oap::math::Matrix, MatricesDeleter>;

using ComplexMatrixDeleter = std::function<void(oap::math::ComplexMatrix*)>;
using ComplexMatrixSharedPtr = MatrixSharedWrapper<oap::math::ComplexMatrix, ComplexMatrixDeleter>;
using ComplexMatrixUniquePtr = MatrixUniqueWrapper<oap::math::ComplexMatrix, ComplexMatrixDeleter>;

using MatrixDeleter = std::function<void(oap::math::Matrix*)>;
using MatrixSharedPtr = MatrixSharedWrapper<oap::math::Matrix, MatrixDeleter>;
using MatrixUniquePtr = MatrixUniqueWrapper<oap::math::Matrix, MatrixDeleter>;

}

#endif
