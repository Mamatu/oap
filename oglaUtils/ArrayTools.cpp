#include "ArrayTools.h"
#include "Types.h"
#include <string.h>

#define memcpy(dst,src,size)\
if(size > 0){\
memcpy(dst,src,size);\
}

namespace ArrayTools {


    /*

    template<typename T> void add(T t, T** ts, uint& size) {
   if ((*ts) == NULL || ts == NULL || size == 0) {
       (*ts) = new T[1];
       size = 1;
       (*ts)[0] = t;
       return ts;
   }

   T* nts = new T[size + 1];
   memcpy(nts, (*ts), size);
   nts[size] = t;
   size++;
   delete[] (*ts);
   (*ts) = nts;
    }

    template<typename T> void add(T** dst, uint& dst_size, const T* src, uint src_size) {
   if (dst == NULL) {
       return;
   }
   if (*dst != NULL) {
       delete[] (*dst);
   }
   T* nts = new T[dst_size + src_size];
   memcpy(nts, (*dst), dst_size);
   memcpy(nts + dst_size, src, src_size);
   dst_size += src_size;
   (*dst) = nts;
    }

    template<typename T> void extend(T** dst, uint& dst_size, T element, uint src_size) {
   if (dst == NULL) {
       return;
   }
   if (*dst != NULL) {

       T* nts = new T[dst_size + src_size];
       memcpy(nts, *dst, dst_size);

       for (uint fa = 0; fa < src_size; fa++) {
           nts[fa + dst_size] = element;
       }
       delete[] (*dst);
       (*dst) = nts;
   }
    }

    template<typename T> void extend(T** dst, uint& dst_size, uint src_size) {
   if (dst == NULL) {
       return;
   }
   if (*dst != NULL) {

       T* nts = new T[dst_size + src_size];
       memcpy(nts, *dst, dst_size);

       delete[] (*dst);
       (*dst) = nts;
   }
    }

    template<typename T> void extend(T*& dst, uint& dst_size, T element, uint src_size) {
   extend(&dst, dst_size, element, src_size);
    }

    template<typename T> void extend(T*& dst, uint& dst_size, uint src_size) {
   extend(&dst, dst_size, src_size);
    }

 

    template<typename T> void remove(T t, T** ts, uint& size) {
   if (ts != NULL && size > 0) {
       int index = indexOf(t, ts, size);
       if (index != -1) {
           T* nts = new T[size - 1];
           memcpy(nts, ts, index);
           memcpy(nts + index, ts + index + 1, size - 1);
           size--;
           delete[] ts;
           return nts;
       }
   }
   return NULL;
    }

    template<typename T> void add(T t, T*& ts, uint& size) {
   add(t, &ts, size);
    }

    template<typename T> void add(T*& dst, uint& dst_size, T* src, uint src_size) {
   add(&dst, dst_size, src, src_size);
    }

  

    template<typename T> void remove(T t, T*& ts, uint& size) {
   remove(t, &ts, size);
    }

    template<typename T> int indexOf(T t, T* ts, uint size) {
   for (uint fa = 0; fa < size; fa++) {
       if (ts[fa] == t) {
           return fa;
       }
   }
   return -1;
    }

    template<typename T> int indexOf(T* t, T* ts, uint size) {
   if (ts <= t && t <= ts + size) {
       return t - ts;
   }
   return -1;
    }

    template<typename T> int indexOf(T* t, T** ts, uint size) {
   for (uint fa = 0; fa < size; fa++) {
       if ((ts[fa]) == t) {
           return fa;
       }
   }
   return -1;
    }*/
}
