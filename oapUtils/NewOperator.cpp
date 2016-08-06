#include "NewOperator.h"
#include <vector>
#include <map>
/*
typedef std::pair<std::vector<uintt>, std::vector<uintt> > StackPointers;

class Operators {
 public:
  static std::map<void*, StackPointers> m_pointers;
};

std::map<void*, StackPointers> Operators::m_pointers;

void* operator new(std::size_t count) throw std::bad_alloc {
  void* ptr = ::operator new(count);
  return ptr;
}

void* operator new[](std::size_t count) throw std::bad_alloc {
  void* ptr = ::operator new[](count);
  return ptr;
}

void operator delete(void* ptr) throw std::bad_alloc { ::operator delete(ptr); }

void operator delete[](void* ptr) throw std::bad_alloc {
  ::operator delete[](ptr);
}
*/
