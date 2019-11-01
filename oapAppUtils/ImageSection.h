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

#ifndef OAP_IMAGE_SECTION_H
#define OAP_IMAGE_SECTION_H

#include <cstddef>
#include <stdio.h>
#include <utility>

namespace oap
{

struct ImageSection
{
  ImageSection() : section (std::make_pair (0, 0)) {}

  ImageSection(size_t length) : section (std::make_pair (0, length)) {}

  ImageSection(size_t length, size_t point) : section (std::make_pair(point, length)) {}

  std::pair <size_t, size_t> section;

  void setPoint (size_t point)
  {
    section.first = point;
  }

  size_t getPoint () const
  {
    return section.first;
  }

  void setLength (size_t length)
  {
    section.second = length;
  }

  size_t getLength () const
  {
    return section.second;
  }

  void setp (size_t point)
  {
    section.first = point;
  }

  size_t getp () const
  {
    return section.first;
  }

  void setl (size_t length)
  {
    section.second = length;
  }

  size_t getl () const
  {
    return section.second;
  }
};

};

#endif
