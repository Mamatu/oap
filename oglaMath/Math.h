/* 
 * File:   MathTypes.h
 * Author: mmatula
 *
 * Created on February 13, 2014, 11:30 PM
 */

#ifndef OGLA_MATH_TYPES_H
#define	OGLA_MATH_TYPES_H

#define NORMAL_TYPES
//#define EXTEDED_TYPES
//#define RICH_TYPES

#ifdef NORMAL_TYPES
#define NORMAL_FLOAT_TYPE
#define NORMAL_INT_TYPES
#endif

#ifdef EXTEDED_TYPES
#define EXTENDED_FLOAT_TYPE
#define EXTENDED_INT_TYPES
#endif

#ifdef RICH_TYPES
#define RICH_FLOAT_TYPE
#define EXTENDED_INT_TYPES
#endif

#ifdef NORMAL_FLOAT_TYPE
typedef float floatt;
#endif

#ifdef EXTENDED_FLOAT_TYPE
typedef double floatt;
#endif

#ifdef RICH_FLOAT_TYPE
typedef long double floatt;
#endif

#ifdef NORMAL_INT_TYPES
typedef int intt;
typedef unsigned int uintt;
#endif 

#ifdef EXTENDED_INT_TYPES
typedef long long intt;
typedef unsigned long long uintt;
#endif


#define MATH_VALUE_LIMIT 0.000000000000000005

namespace math {
    void Memset(floatt* array, floatt value, intt length);
}

struct Complex {

    Complex() {
        re = 0;
        im = 0;
    }
    floatt re;
    floatt im;
};

#endif	/* MATHTYPES_H */

