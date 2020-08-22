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

#ifndef GRAPHICUTILS_H
#define GRAPHICUTILS_H

#include <cstddef>
#include <stdio.h>
#include <utility>

#include "ImageSection.h"

namespace oap
{

template <typename T2DArray, typename T>
ImageSection GetOptWidth (T2DArray bitmap2d, size_t width, size_t height, size_t colorsCount);

template <typename T2DArray, typename T>
ImageSection GetOptHeight (T2DArray bitmap2d, size_t width, size_t height, size_t colorsCount);
};

#include "GraphicsUtilsImpl.h"

#endif  // GRAPHICUTILS_H
