/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef QRTEST1
#define QRTEST1

// Generated from ArnoldiPackage procedure. Step 3.
namespace samples {
namespace qrtest1 {

const char* matrix =
    "[-3.25000 , 3.60555 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          3.60555 ,-2.32692 , 4.90230 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 4.90230 ,-1.54640 , 5.76068 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 5.76068 ,-0.91061 , 6.36171 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 6.36171 ,-0.41668 , 6.77074 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.77074 ,-0.06041 , 7.01617 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.01617 , 0.16061 , 7.10992 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.10992 , 0.24628 , 7.05417 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.05417 , 0.19482 , 6.84374 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.84374 , 0.00436 , 6.46680 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.46680 ,-0.32193 , 5.90731 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.90731 ,-0.75464 , 5.16807 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.16807 ,-1.09953 , 4.47483 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 4.47483 , 0.08886 , 5.40542 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.40542 , 4.49216 , 5.48863 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.48863 ,-2.36005 , 6.31732 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.31732 , 0.28511 , 5.75847 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.75847 ,-0.33049 , 5.66978 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.66978 ,-0.36293 , 5.60496 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.60496 ,-0.24422 , 5.63400 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.63400 , 0.35960 , 5.88692 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.88692 , 1.07043 , 5.96654 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.96654 , 0.80533 , 5.54403 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.54403 ,-0.21378 , 5.31925 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.31925 ,-0.76342 , 5.89886 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.89886 ,-0.22727 , 6.52897 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.52897 , 0.17239 , 5.77112 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.77112 ,-0.07570 , 5.48895 , 0.00000 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.48895 , 1.16170 , 6.34324 , 0.00000 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.34324 ,-0.19258 , 5.80947 , 0.00000,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.80947 ,-1.70667 , 5.58139,\
          0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.58139 , 0.00000]";

const char* qref =
    "[-0.66953 , 0.16546 , 0.37945 ,-0.05942 ,-0.27277 , 0.04356 , 0.21881 ,-0.04364 ,-0.18892 , 0.03943 , 0.17753 ,-0.01434 ,-0.17552 , 0.01248 , 0.13762 , 0.03839 ,-0.13174 ,-0.04046 , 0.11266 , 0.03969 ,-0.09718 ,-0.01847 , 0.09613 , 0.02077 ,-0.07586 ,-0.01982 , 0.07903 , 0.01994 ,-0.06622 ,-0.02227 , 0.05757 , 0.23812,\
        0.74278 , 0.14915 , 0.34203 ,-0.05356 ,-0.24587 , 0.03927 , 0.19723 ,-0.03933 ,-0.17029 , 0.03554 , 0.16002 ,-0.01292 ,-0.15822 , 0.01125 , 0.12405 , 0.03460 ,-0.11875 ,-0.03647 , 0.10155 , 0.03577 ,-0.08760 ,-0.01664 , 0.08665 , 0.01872 ,-0.06838 ,-0.01786 , 0.07124 , 0.01798 ,-0.05969 ,-0.02007 , 0.05189 , 0.21464,\
        0.00000 , 0.97487 ,-0.11673 , 0.01828 , 0.08391 ,-0.01340 ,-0.06731 , 0.01342 , 0.05812 ,-0.01213 ,-0.05461 , 0.00441 , 0.05400 ,-0.00384 ,-0.04234 ,-0.01181 , 0.04053 , 0.01245 ,-0.03466 ,-0.01221 , 0.02989 , 0.00568 ,-0.02957 ,-0.00639 , 0.02334 , 0.00610 ,-0.02431 ,-0.00614 , 0.02037 , 0.00685 ,-0.01771 ,-0.07325,\
        0.00000 , 0.00000 , 0.85171 , 0.05049 , 0.23176 ,-0.03701 ,-0.18591 , 0.03708 , 0.16052 ,-0.03350 ,-0.15084 , 0.01218 , 0.14914 ,-0.01060 ,-0.11693 ,-0.03262 , 0.11193 , 0.03438 ,-0.09572 ,-0.03372 , 0.08257 , 0.01569 ,-0.08168 ,-0.01765 , 0.06446 , 0.01684 ,-0.06715 ,-0.01695 , 0.05626 , 0.01892 ,-0.04892 ,-0.20232,\
        0.00000 , 0.00000 , 0.00000 , 0.99535 ,-0.04281 , 0.00684 , 0.03434 ,-0.00685 ,-0.02965 , 0.00619 , 0.02786 ,-0.00225 ,-0.02755 , 0.00196 , 0.02160 , 0.00602 ,-0.02068 ,-0.00635 , 0.01768 , 0.00623 ,-0.01525 ,-0.00290 , 0.01509 , 0.00326 ,-0.01191 ,-0.00311 , 0.01240 , 0.00313 ,-0.01039 ,-0.00350 , 0.00904 , 0.03737,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.89586 , 0.03520 , 0.17680 ,-0.03526 ,-0.15265 , 0.03186 , 0.14344 ,-0.01159 ,-0.14182 , 0.01008 , 0.11119 , 0.03102 ,-0.10644 ,-0.03269 , 0.09103 , 0.03207 ,-0.07852 ,-0.01492 , 0.07767 , 0.01678 ,-0.06130 ,-0.01601 , 0.06386 , 0.01611 ,-0.05351 ,-0.01799 , 0.04652 , 0.19240,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99686 ,-0.03162 , 0.00631 , 0.02730 ,-0.00570 ,-0.02565 , 0.00207 , 0.02536 ,-0.00180 ,-0.01989 ,-0.00555 , 0.01904 , 0.00585 ,-0.01628 ,-0.00573 , 0.01404 , 0.00267 ,-0.01389 ,-0.00300 , 0.01096 , 0.00286 ,-0.01142 ,-0.00288 , 0.00957 , 0.00322 ,-0.00832 ,-0.03441,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.91689 , 0.03465 , 0.15002 ,-0.03131 ,-0.14097 , 0.01139 , 0.13938 ,-0.00991 ,-0.10928 ,-0.03048 , 0.10461 , 0.03213 ,-0.08946 ,-0.03151 , 0.07717 , 0.01466 ,-0.07634 ,-0.01649 , 0.06024 , 0.01574 ,-0.06276 ,-0.01584 , 0.05258 , 0.01768 ,-0.04572 ,-0.18908,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99622 ,-0.03275 , 0.00684 , 0.03078 ,-0.00249 ,-0.03043 , 0.00216 , 0.02386 , 0.00665 ,-0.02284 ,-0.00701 , 0.01953 , 0.00688 ,-0.01685 ,-0.00320 , 0.01667 , 0.00360 ,-0.01315 ,-0.00344 , 0.01370 , 0.00346 ,-0.01148 ,-0.00386 , 0.00998 , 0.04128,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.92610 , 0.03208 , 0.14443 ,-0.01166 ,-0.14280 , 0.01015 , 0.11196 , 0.03123 ,-0.10718 ,-0.03292 , 0.09165 , 0.03229 ,-0.07906 ,-0.01502 , 0.07821 , 0.01690 ,-0.06172 ,-0.01612 , 0.06430 , 0.01623 ,-0.05387 ,-0.01812 , 0.04684 , 0.19372,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99638 ,-0.03267 , 0.00264 , 0.03230 ,-0.00230 ,-0.02532 ,-0.00706 , 0.02424 , 0.00745 ,-0.02073 ,-0.00730 , 0.01788 , 0.00340 ,-0.01769 ,-0.00382 , 0.01396 , 0.00365 ,-0.01454 ,-0.00367 , 0.01219 , 0.00410 ,-0.01059 ,-0.04382,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.92325 , 0.01291 , 0.15808 ,-0.01124 ,-0.12394 ,-0.03457 , 0.11865 , 0.03644 ,-0.10146 ,-0.03574 , 0.08752 , 0.01663 ,-0.08658 ,-0.01871 , 0.06833 , 0.01785 ,-0.07118 ,-0.01796 , 0.05964 , 0.02006 ,-0.05185 ,-0.21446,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99944 ,-0.01384 , 0.00098 , 0.01085 , 0.00303 ,-0.01039 ,-0.00319 , 0.00888 , 0.00313 ,-0.00766 ,-0.00146 , 0.00758 , 0.00164 ,-0.00598 ,-0.00156 , 0.00623 , 0.00157 ,-0.00522 ,-0.00176 , 0.00454 , 0.01877,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.91133 , 0.01322 , 0.14581 , 0.04067 ,-0.13958 ,-0.04287 , 0.11936 , 0.04205 ,-0.10296 ,-0.01956 , 0.10186 , 0.02201 ,-0.08038 ,-0.02100 , 0.08374 , 0.02113 ,-0.07016 ,-0.02360 , 0.06100 , 0.25230,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99948 ,-0.01138 ,-0.00317 , 0.01089 , 0.00335 ,-0.00931 ,-0.00328 , 0.00803 , 0.00153 ,-0.00795 ,-0.00172 , 0.00627 , 0.00164 ,-0.00653 ,-0.00165 , 0.00547 , 0.00184 ,-0.00476 ,-0.01969,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.93511 ,-0.03746 , 0.12855 , 0.03948 ,-0.10993 ,-0.03873 , 0.09483 , 0.01802 ,-0.09381 ,-0.02027 , 0.07403 , 0.01934 ,-0.07712 ,-0.01946 , 0.06462 , 0.02173 ,-0.05618 ,-0.23236,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99440 , 0.03856 , 0.01184 ,-0.03298 ,-0.01162 , 0.02844 , 0.00541 ,-0.02814 ,-0.00608 , 0.02221 , 0.00580 ,-0.02313 ,-0.00584 , 0.01938 , 0.00652 ,-0.01685 ,-0.06970,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.93108 ,-0.04390 , 0.12223 , 0.04306 ,-0.10544 ,-0.02003 , 0.10430 , 0.02254 ,-0.08231 ,-0.02150 , 0.08575 , 0.02164 ,-0.07185 ,-0.02416 , 0.06246 , 0.25836,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99273 , 0.04062 , 0.01431 ,-0.03504 ,-0.00666 , 0.03466 , 0.00749 ,-0.02735 ,-0.00714 , 0.02849 , 0.00719 ,-0.02387 ,-0.00803 , 0.02076 , 0.08585,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.94132 ,-0.04263 , 0.10439 , 0.01984 ,-0.10327 ,-0.02231 , 0.08149 , 0.02129 ,-0.08490 ,-0.02142 , 0.07113 , 0.02392 ,-0.06184 ,-0.25579,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99199 , 0.03938 , 0.00748 ,-0.03896 ,-0.00842 , 0.03074 , 0.00803 ,-0.03203 ,-0.00808 , 0.02684 , 0.00902 ,-0.02333 ,-0.09650,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.95015 ,-0.01944 , 0.10121 , 0.02187 ,-0.07987 ,-0.02086 , 0.08320 , 0.02100 ,-0.06972 ,-0.02345 , 0.06061 , 0.25069,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99805 , 0.02028 , 0.00438 ,-0.01600 ,-0.00418 , 0.01667 , 0.00421 ,-0.01397 ,-0.00470 , 0.01214 , 0.05023,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.94563 ,-0.02417 , 0.08828 , 0.02306 ,-0.09197 ,-0.02321 , 0.07706 , 0.02592 ,-0.06699 ,-0.27709,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99723 , 0.02023 , 0.00528 ,-0.02107 ,-0.00532 , 0.01766 , 0.00594 ,-0.01535 ,-0.06349,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.96225 ,-0.02011 , 0.08020 , 0.02024 ,-0.06720 ,-0.02260 , 0.05842 , 0.24165,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99727 , 0.02183 , 0.00551 ,-0.01829 ,-0.00615 , 0.01590 , 0.06578,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.95535 ,-0.02306 , 0.07657 , 0.02575 ,-0.06657 ,-0.27535,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99695 , 0.02029 , 0.00682 ,-0.01764 ,-0.07295,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.96562 ,-0.02353 , 0.06084 , 0.25163,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.99589 , 0.02128 , 0.08800,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.97200 ,-0.23500]";

const char* rref =
    "[4.85412 ,-4.14243 , 3.64134 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 5.02865 ,-0.77639 , 5.61593 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 6.76367 ,-1.44802 , 5.41833 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 6.39144 ,-0.09356 , 6.73924 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.55783 ,-0.34397 , 6.28549 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.03829 , 0.40707 , 7.08758 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.75438 , 0.00101 , 6.46791 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.08090 , 0.43851 , 6.81790 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 7.38985 ,-0.22012 , 5.98890 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.49030 ,-0.11331 , 5.88592 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.39841 ,-0.88970 , 4.77140 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.17099 ,-1.03217 , 4.47230 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 4.91023 , 0.01906 , 4.92611 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.40821 , 4.56129 , 5.48580 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.86951 ,-2.26935 , 5.90738 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.35291 , 0.04688 , 5.72621 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.18470 ,-0.08566 , 5.27904 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.71129 ,-0.60921 , 5.56423 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.95435 ,-0.00224 , 5.30340 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.67949 , 0.11654 , 5.83977 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.19576 , 1.24890 , 5.66913 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.97817 , 0.68777 , 5.53324 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.86280 ,-0.08972 , 5.03004 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.33400 ,-0.88988 , 5.88255 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.13030 ,-0.09937 , 6.28248 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.54686 , 0.04063 , 5.75535 , 0.00000 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.04085 , 0.05367 , 5.24387 , 0.00000 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.50575 , 1.03157 , 6.32389 , 0.00000 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 6.56906 ,-0.05726 , 5.60976 , 0.00000,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.83343 ,-1.83638 , 5.55847,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 5.74220 , 0.11875,\
        0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.00000 , 0.49116]";
}
}
#endif
