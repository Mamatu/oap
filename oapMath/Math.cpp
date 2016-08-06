#include <string.h>
#include "Math.h"

namespace math {

void Memset(floatt* array, floatt value, intt length) {
  for (uintt fa = 0; fa < length; ++fa) {
    array[fa] = value;
  }
}
}

bool operator==(const Complex& c1, const Complex& c2) {
  floatt limit = 0.2;
  bool isRe = c2.re - limit <= c1.re && c1.re <= c2.re + limit;
  bool isIm = c2.im - limit <= c1.im && c1.im <= c2.im + limit;
  return isRe && isIm;
}
