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



#ifndef MATRIX1
#define MATRIX1

const char* matrix1Str =
    "  (columns=16, rows=16) [-12.7197, 3.6577, 1.02183e-05, -1.23725e-08, "
    "1.22547e-10, -2.13381e-10, 2.39094e-11, 2.21756e-12, 4.48291e-15, "
    "-2.58683e-14, 8.62322e-15, -1.07807e-13, 1.16873e-13, 1.82634e-11, "
    "-6.52886e-11, 1.03901e-08 | 3.6577, 12.7196, -0.000962584, 1.16551e-06, "
    "-1.15429e-08, 2.01002e-08, 7.0744e-11, 6.5016e-12, 1.15047e-14, "
    "2.08317e-14, -2.64363e-14, 2.27488e-13, -2.38072e-13, -2.73002e-11, "
    "9.75254e-11, 1.11926e-08 | 1.02183e-05, -0.000962584, -11.6683, "
    "0.000957369, -1.75983e-08, -2.20456e-08, -3.79038e-08, -3.48401e-09, "
    "-7.56415e-13, -9.70135e-15, -1.47895e-14, 9.46865e-12, -9.86608e-12, "
    "-3.76564e-10, 1.24899e-09, -5.94154e-08 | -1.23725e-08, 1.16551e-06, "
    "0.000957369, 10.9335, 0.000608623, 0.000762429, 2.702e-05, 2.48361e-06, "
    "5.40699e-10, 1.28956e-14, -1.86345e-14, 2.72342e-10, -2.84965e-10, "
    "-3.64756e-09, 1.15398e-08, 4.84052e-07 | 1.22533e-10, -1.15429e-08, "
    "-1.75983e-08, 0.000608623, 10.3506, 0.000999129, 0.000922771, "
    "8.48185e-05, 1.84657e-08, 4.18973e-13, -1.70268e-13, 2.04113e-09, "
    "-2.14097e-09, -1.4567e-08, 4.40794e-08, 1.30919e-06 | -2.13374e-10, "
    "2.01003e-08, -2.20456e-08, 0.000762429, 0.000999129, -10.0667, "
    "-0.000131322, -1.20707e-05, -2.6279e-09, -7.32539e-14, -3.80239e-14, "
    "7.73891e-10, -8.12755e-10, -6.10966e-09, 1.80609e-08, -3.23588e-07 | "
    "2.39159e-11, 7.07242e-11, -3.79038e-08, 2.702e-05, 0.000922771, "
    "-0.000131322, -8.50242, 0.000990557, -8.55548e-05, -2.1503e-09, "
    "2.22834e-10, 8.50311e-08, -8.94377e-08, -1.06741e-07, 2.62238e-07, "
    "-1.59534e-06 | 2.19829e-12, 6.5006e-12, -3.48401e-09, 2.48361e-06, "
    "8.48185e-05, -1.20707e-05, 0.000990557, 8.39596, 0.000907605, "
    "2.28115e-08, -2.37026e-09, 6.65747e-08, -7.00172e-08, -4.41793e-08, "
    "1.06624e-07, 8.95042e-07 | 3.51997e-16, 1.49443e-15, -7.5886e-13, "
    "5.40702e-10, 1.84657e-08, -2.62791e-09, -8.55548e-05, 0.000907605, "
    "-7.12051, 0.000965376, -6.59436e-06, 7.29445e-06, -7.67388e-06, "
    "-1.42501e-06, 2.6203e-06, -5.53873e-06 | -7.78711e-17, 6.78064e-17, "
    "6.01494e-16, 7.48823e-15, 4.31863e-13, -6.28296e-14, -2.15031e-09, "
    "2.28115e-08, 0.000965376, 6.13923, 0.000968079, 5.64373e-05, "
    "-5.93733e-05, -1.26436e-06, 1.59804e-06, 2.71399e-06 | 3.84098e-16, "
    "-4.23831e-16, -3.51674e-15, -3.46131e-14, -1.67956e-13, -4.22151e-14, "
    "2.22828e-10, -2.37025e-09, -6.59436e-06, 0.000968079, -5.72781, "
    "0.000672712, -0.000707717, -1.59735e-05, 1.58624e-05, -1.17412e-05 | "
    "-1.13198e-13, 2.33403e-13, 9.47181e-12, 2.72342e-10, 2.04114e-09, "
    "7.73881e-10, 8.50311e-08, 6.65747e-08, 7.29445e-06, 5.64373e-05, "
    "0.000672712, -4.03725, -0.0290168, 0.000588068, 0.000288654, 2.8249e-05 | "
    "1.14288e-13, -2.35175e-13, -9.86505e-12, -2.84958e-10, -2.14098e-09, "
    "-8.12774e-10, -8.94377e-08, -7.00172e-08, -7.67388e-06, -5.93733e-05, "
    "-0.000707717, -0.0290168, 3.95387, 0.000148118, 7.56936e-05, -1.39995e-05 "
    "| 1.82747e-11, -2.732e-11, -3.76575e-10, -3.64756e-09, -1.4567e-08, "
    "-6.10965e-09, -1.06741e-07, -4.41793e-08, -1.42501e-06, -1.26436e-06, "
    "-1.59735e-05, 0.000588068, 0.000148118, -2.16251, -0.000945986, "
    "-0.000149331 | -6.53051e-11, 9.75458e-11, 1.24899e-09, 1.15398e-08, "
    "4.40794e-08, 1.80609e-08, 2.62238e-07, 1.06624e-07, 2.6203e-06, "
    "1.59804e-06, 1.58624e-05, 0.000288654, 7.56936e-05, -0.000945986, "
    "1.84544, 0.000194614 | 1.03901e-08, 1.11926e-08, -5.94154e-08, "
    "4.84052e-07, 1.30919e-06, -3.23588e-07, -1.59534e-06, 8.95042e-07, "
    "-5.53873e-06, 2.71399e-06, -1.17412e-05, 2.8249e-05, -1.39995e-05, "
    "-0.000149331, 0.000194614, -0.193069] (length=256) [0 <repeats 256 "
    "times>] (length=256)";

#endif  // MATRIX1
