#include <string.h>
#include "Math.h"

namespace math {
    
    void Memset(floatt* array, floatt value, intt length) {
        for (uintt fa = 0; fa < length; fa++) {
            array[fa] = value;
        }
    }
}
