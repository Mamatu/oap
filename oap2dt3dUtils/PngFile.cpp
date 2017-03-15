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

size_t g_colorsCount = 3;

PngFile::PngFile(const std::string& path)
    : oap::Image(path),
      m_fp(NULL),
      m_bitmap2d(NULL),
      m_bitmap1d(NULL),
      m_pixels(NULL) {}

PngFile::~PngFile() { close(); }

bool PngFile::read(void* buffer, size_t repeat, size_t size) {
  fread(buffer, repeat, size, m_fp);
}

void PngFile::close() {
  if (m_fp != NULL) {
    fclose(m_fp);
    m_fp = NULL;
  }
}

oap::OptSize PngFile::getWidth() const {
  return oap::OptSize(png_get_image_width(m_png_ptr, m_info_ptr));
}

oap::OptSize PngFile::getHeight() const {
  return oap::OptSize(png_get_image_height(m_png_ptr, m_info_ptr));
}

void PngFile::forceOutputWidth(const oap::OptSize& optWidth) {
  m_optWidth = optWidth;
}

void PngFile::forceOutputHeight(const oap::OptSize& optHeight) {
  m_optHeight = optHeight;
}

oap::OptSize PngFile::getOutputWidth() const {
  if (m_optWidth.optSize != 0) {
    return m_optWidth;
  }
  return getWidth();
}

oap::OptSize PngFile::getOutputHeight() const {
  if (m_optHeight.optSize != 0) {
    return m_optHeight;
  }
  return getHeight();
}

std::string PngFile::getSufix() const { return "png"; }

void PngFile::loadBitmapProtected() {
  png_byte color_type;
  png_byte bit_depth;

  m_png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  m_info_ptr = png_create_info_struct(m_png_ptr);

  png_init_io(m_png_ptr, m_fp);

  png_set_sig_bytes(m_png_ptr, 0);

  png_read_info(m_png_ptr, m_info_ptr);

  size_t width = getWidth().optSize;
  size_t height = getHeight().optSize;

  color_type = png_get_color_type(m_png_ptr, m_info_ptr);
  bit_depth = png_get_bit_depth(m_png_ptr, m_info_ptr);

  int number_of_passes = png_set_interlace_handling(m_png_ptr);
  png_read_update_info(m_png_ptr, m_info_ptr);

  m_bitmap2d = new png_bytep[height];

  for (unsigned int fa = 0; fa < height; ++fa) {
    const size_t localWidth =
        png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);
    m_bitmap2d[fa] = new png_byte[localWidth];
  }

  png_read_image(m_png_ptr, m_bitmap2d);

  calculateOutputSizes(width, height);

  createBitmap1dFrom2d(&m_bitmap1d, m_bitmap2d);

  createPixelsVectorFrom1d(m_bitmap1d);

  destroyBitmap2d();

  destroyBitmap1d();
}

void PngFile::freeBitmapProtected() {
  destroyBitmap2d();

  destroyBitmap1d();

  png_destroy_info_struct(m_png_ptr, &m_info_ptr);

  png_destroy_read_struct(&m_png_ptr, NULL, NULL);

  delete[] m_pixels;

  m_pixels = NULL;
  m_png_ptr = NULL;
  m_info_ptr = NULL;
}

void PngFile::getPixelsVectorProtected(pixel_t* pixels) const {
  const size_t length = getLength();
  memcpy(pixels, m_pixels, sizeof(pixel_t) * length);
}

bool PngFile::openProtected(const std::string& path) {
  m_fp = fopen(path.c_str(), "rb");
  return m_fp != NULL;
}

bool PngFile::isCorrectFormat() const {
  const size_t header_size = 8;
  unsigned char header[header_size];
  return png_sig_cmp(header, 0, header_size);
}

pixel_t PngFile::getPixelProtected(unsigned int x, unsigned int y) const {
  const size_t width = getWidth().optSize;
  return m_pixels[y * width + x];
}

void PngFile::calculateOutputSizes(size_t width, size_t height) {
  if (m_optWidth.optSize == 0) {
    oap::OptSize owidth = oap::GetOptWidth<png_bytep*, png_byte>(
        m_bitmap2d, width, height, g_colorsCount);

    m_optWidth = owidth;
  }

  if (m_optHeight.optSize == 0) {
    oap::OptSize oheight = oap::GetOptHeight<png_bytep*, png_byte>(
        m_bitmap2d, width, height, g_colorsCount);

    m_optHeight = oheight;
  }
}

void PngFile::createBitmap1dFrom2d(png_byte** bitmap1d, png_bytep* bitmap2d) {
  const size_t beginC = m_optWidth.begin;
  const size_t beginR = m_optHeight.begin;

  size_t width = m_optWidth.optSize * g_colorsCount;
  size_t height = m_optHeight.optSize;

  (*bitmap1d) = new png_byte[width * height];

  for (size_t fa = beginR; fa < height; ++fa) {
    memcpy(&(*bitmap1d)[fa * width * sizeof(png_byte)], &(bitmap2d[fa][beginC]),
           width * sizeof(png_byte));
  }
}

void PngFile::createPixelsVectorFrom1d(png_byte* bitmap1d) {
  size_t width = getOutputWidth().optSize;
  size_t height = getOutputHeight().optSize;

  m_pixels = new pixel_t[width * height];

  for (size_t fa = 0; fa < height; ++fa) {
    for (size_t fb = 0; fb < width; ++fb) {
      const size_t index = fa * width * g_colorsCount + fb * g_colorsCount;
      const size_t index1 = fa * width + fb;
      const png_byte r = bitmap1d[index];
      const png_byte g = bitmap1d[index + 1];
      const png_byte b = bitmap1d[index + 2];
      m_pixels[index1] = convertRgbToPixel(r, g, b);
    }
  }
}

void PngFile::destroyBitmap2d() {
  if (m_bitmap2d != NULL) {
    int height = getHeight().optSize;

    for (unsigned int fa = 0; fa < height; ++fa) {
      delete[] m_bitmap2d[fa];
    }

    delete[] m_bitmap2d;

    m_bitmap2d = NULL;
  }
}

void PngFile::destroyBitmap1d() {
  if (m_bitmap1d != NULL) {
    delete[] m_bitmap1d;
    m_bitmap1d = NULL;
  }
}
}
