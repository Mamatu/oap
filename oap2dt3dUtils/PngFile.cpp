/*
 * Copyright 2016 Marcin Matula
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

#include "PngFile.h"
#include "Exceptions.h"

namespace oap {

PngFile::PngFile() : m_fp(NULL), m_bitmap(NULL) {}

PngFile::~PngFile() { close(); }

bool PngFile::read(void* buffer, size_t repeat, size_t size) {
  fread(buffer, repeat, size, m_fp);
}

void PngFile::loadBitmap() {
  int width, height;
  png_byte color_type;
  png_byte bit_depth;

  m_png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  m_info_ptr = png_create_info_struct(m_png_ptr);

  png_init_io(m_png_ptr, m_fp);

  png_set_sig_bytes(m_png_ptr, 0);

  png_read_info(m_png_ptr, m_info_ptr);

  width = getWidth();
  height = getHeight();

  color_type = png_get_color_type(m_png_ptr, m_info_ptr);
  bit_depth = png_get_bit_depth(m_png_ptr, m_info_ptr);

  int number_of_passes = png_set_interlace_handling(m_png_ptr);
  png_read_update_info(m_png_ptr, m_info_ptr);

  m_bitmap = new png_bytep[height];
  for (unsigned int fa = 0; fa < height; fa++) {
    m_bitmap[fa] = new png_byte[png_get_rowbytes(m_png_ptr, m_info_ptr) /
                                sizeof(png_byte)];
  }
  png_read_image(m_png_ptr, m_bitmap);
}

void PngFile::freeBitmap() {
  if (m_bitmap == NULL) {
    return;
  }

  int height = getHeight();

  for (unsigned int fa = 0; fa < height; fa++) {
    delete[] m_bitmap[fa];
  }
  delete[] m_bitmap;

  png_destroy_info_struct(m_png_ptr, &m_info_ptr);

  png_destroy_read_struct(&m_png_ptr, NULL, NULL);

  m_png_ptr = NULL;
  m_info_ptr = NULL;
}

void PngFile::close() {
  if (m_fp != NULL) {
    fclose(m_fp);
    m_fp = NULL;
  }
}

unsigned int PngFile::getWidth() const {
  return png_get_image_width(m_png_ptr, m_info_ptr);
}

unsigned int PngFile::getHeight() const {
  return png_get_image_height(m_png_ptr, m_info_ptr);
}

bool PngFile::openInternal(const char* path) {
  m_fp = fopen(path, "rb");
  return m_fp != NULL;
}

bool PngFile::isPngInternal() const {
  const size_t header_size = 8;
  unsigned char header[header_size];
  return png_sig_cmp(header, 0, header_size);
}

Pixel PngFile::getPixelInternal(unsigned int x, unsigned int y) const {
  png_byte* row = m_bitmap[y];
  png_byte* ptr = &(row[x * 4]);
  Pixel pixel(ptr[0], ptr[1], ptr[2]);
  return pixel;
}
}
