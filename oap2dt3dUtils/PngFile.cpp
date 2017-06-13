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
#include "DebugLogs.h"

namespace oap {

PngFile::PngFile(const std::string& path, bool truncateImage)
    : oap::Image(path),
      m_fp(NULL),
      m_png_ptr(NULL),
      m_bitmap2d(NULL),
      m_bitmap1d(NULL),
      m_pixels(NULL),
      m_destroyTmp(true),
      m_truncateImage(truncateImage) {}

PngFile::~PngFile() {
  freeBitmapProtected();
  closeProtected();
}

bool PngFile::read(void* buffer, size_t repeat, size_t size) {
  fread(buffer, repeat, size, m_fp);
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

bool PngFile::save(const std::string& path) {
  if (this->isLoaded() == true) {
    return false;
  }

  FILE* fp = fopen(path.c_str(), "wb");

  if (fp == NULL) {
    return false;
  }

  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (png == NULL) {
    fclose(fp);
    return false;
  }

  png_infop info = png_create_info_struct(png);

  if (info == NULL) {
    fclose(fp);
    return false;
  }

  png_init_io(png, fp);

  this->setAutomaticDestroyTmpData(false);

  this->open();
  this->loadBitmap();
  this->close();

  OptSize outputWidth = this->getOutputWidth();
  OptSize outputHeight = this->getOutputHeight();

  png_set_IHDR(png, info, outputWidth.optSize, outputHeight.optSize, 8,
               PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  png_bytep* row_pointers = NULL;

  bool rowPointerToDestroy = false;

  if (outputWidth.optSize == this->getWidth().optSize &&
      outputHeight.optSize == this->getHeight().optSize &&
      outputWidth.begin == 0 && outputHeight.begin == 0) {
    row_pointers = this->m_bitmap2d;
  } else {
    rowPointerToDestroy = true;
    row_pointers = this->copyBitmap(outputWidth, outputHeight);
  }

  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  fclose(fp);

  this->freeBitmap();

  png_destroy_write_struct(&png, &info);

  if (rowPointerToDestroy) {
    destroyBitmap2d(row_pointers, outputHeight.optSize);
  }

  return true;
}

bool PngFile::save(const std::string& prefix, const std::string& path) {
  std::string filename = this->getFileName();

  filename = prefix + filename;

  std::string filepath = path + "/" + filename;

  return save(filepath);
}

void PngFile::closeProtected() {
  if (m_fp != NULL) {
    fclose(m_fp);
    m_fp = NULL;
  }
}

void PngFile::loadBitmapProtected() {
  if (m_png_ptr == NULL) {
    png_byte color_type;
    png_byte bit_depth;

    m_png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    m_info_ptr = png_create_info_struct(m_png_ptr);

    png_init_io(m_png_ptr, m_fp);

    png_set_sig_bytes(m_png_ptr, 0);

    png_read_info(m_png_ptr, m_info_ptr);

    color_type = png_get_color_type(m_png_ptr, m_info_ptr);
    bit_depth = png_get_bit_depth(m_png_ptr, m_info_ptr);

    int number_of_passes = png_set_interlace_handling(m_png_ptr);
    png_read_update_info(m_png_ptr, m_info_ptr);
  }

  loadBitmapBuffers();

  if (m_destroyTmp) {
    destroyTmpData();
  }
}

void PngFile::loadBitmapBuffers() {
  size_t width = 0;
  size_t height = 0;

  if (m_png_ptr != NULL && m_info_ptr != NULL) {
    calculateColorsCount();
  }

  if (m_bitmap2d == NULL && m_png_ptr != NULL && m_info_ptr != NULL) {
    width = getWidth().optSize;
    height = getHeight().optSize;

    const size_t width1 =
        png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

    debugAssert(width1 / m_colorsCount == width);

    m_bitmap2d = createBitmap2D(width, height, m_colorsCount);

    png_read_image(m_png_ptr, m_bitmap2d);
  }

  if (m_bitmap1d == NULL && m_bitmap2d != NULL) {
    calculateOutputSizes(width, height);

    convertRawdataToBitmap1D();
  }
}

void PngFile::convertRawdataToBitmap1D() {
  destroyBitmap1d();

  destroyPixels();

  m_bitmap1d = createBitmap1dFrom2d(m_bitmap2d, getOutputWidth(),
                                    getOutputHeight(), m_colorsCount);

  m_pixels = createPixelsVectorFrom1d(m_bitmap1d, getOutputWidth().optSize,
                                      getOutputHeight().optSize);
}

void PngFile::destroyTmpData() {
  destroyBitmap2d(m_bitmap2d, getHeight().optSize);

  m_bitmap2d = NULL;

  destroyBitmap1d();
}

void PngFile::setAutomaticDestroyTmpData(bool destroyTmp) {
  m_destroyTmp = destroyTmp;
}

void PngFile::freeBitmapProtected() {
  destroyTmpData();

  if (m_png_ptr != NULL && m_info_ptr != NULL) {
    png_destroy_info_struct(m_png_ptr, &m_info_ptr);
  }

  if (m_png_ptr != NULL) {
    png_destroy_read_struct(&m_png_ptr, NULL, NULL);
  }

  destroyPixels();

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

png_bytep* PngFile::copyBitmap(const OptSize& width, const OptSize& height) {
  const png_bytep* origin = this->m_bitmap2d;

  png_bytep* copy = new png_bytep[height.optSize];

  const size_t localWidth =
      png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

  const size_t imageWidth = getWidth().optSize + getWidth().begin;

  size_t factor = localWidth / imageWidth;

  for (size_t fa = 0; fa < height.optSize; ++fa) {
    copy[fa] = new png_byte[width.optSize * factor];
    memcpy(copy[fa], &origin[height.begin + fa][width.begin * factor],
           width.optSize * factor * sizeof(png_byte));
  }
  return copy;
}

png_bytep* PngFile::createBitmap2D(size_t width, size_t height,
                                   size_t colorsCount) const {
  png_bytep* bitmap2d = new png_bytep[height];

  for (unsigned int fa = 0; fa < height; ++fa) {
    bitmap2d[fa] = new png_byte[width * colorsCount];
  }

  return bitmap2d;
}

void PngFile::destroyBitmap2d(png_bytep* bitmap2d, size_t height) const {
  if (bitmap2d != NULL) {
    for (unsigned int fa = 0; fa < height; ++fa) {
      delete[] bitmap2d[fa];
    }

    delete[] bitmap2d;
  }
}

png_byte* PngFile::createBitmap1dFrom2d(png_bytep* bitmap2d,
                                        const OptSize& optWidth,
                                        const OptSize& optHeight,
                                        size_t colorsCount) {
  const size_t beginC = optWidth.begin;
  const size_t beginR = optHeight.begin;

  size_t width = optWidth.optSize * colorsCount;
  size_t height = optHeight.optSize;

  png_byte* bitmap1d = new png_byte[width * height];

  for (size_t fa = beginR; fa < height; ++fa) {
    memcpy(&(bitmap1d[fa * width]), &(bitmap2d[fa][beginC]),
           width * sizeof(png_byte));
  }

  return bitmap1d;
}

oap::pixel_t* PngFile::createPixelsVectorFrom1d(png_byte* bitmap1d,
                                                size_t width, size_t height) {
  oap::pixel_t* pixels = new pixel_t[width * height];

  for (size_t fa = 0; fa < height; ++fa) {
    for (size_t fb = 0; fb < width; ++fb) {
      const size_t index = fa * width * m_colorsCount + fb * m_colorsCount;
      const size_t index1 = fa * width + fb;
      const png_byte r = bitmap1d[index + 0];
      const png_byte g = bitmap1d[index + 1];
      const png_byte b = bitmap1d[index + 2];
      pixels[index1] = convertRgbToPixel(r, g, b);
    }
  }
  return pixels;
}

void PngFile::calculateColorsCount() {
  const size_t localWidth =
      png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

  const size_t imageWidth = getWidth().optSize + getWidth().begin;

  m_colorsCount = localWidth / imageWidth;
}

void PngFile::calculateOutputSizes(size_t width, size_t height) {
  if (m_truncateImage == false) {
    m_optWidth = getWidth();
    m_optHeight = getHeight();
  } else {
    if (m_optWidth.optSize == 0) {
      oap::OptSize owidth = oap::GetOptWidth<png_bytep*, png_byte>(
          m_bitmap2d, width, height, m_colorsCount);

      m_optWidth = owidth;
    }

    if (m_optHeight.optSize == 0) {
      oap::OptSize oheight = oap::GetOptHeight<png_bytep*, png_byte>(
          m_bitmap2d, width, height, m_colorsCount);

      m_optHeight = oheight;
    }
  }
}

void PngFile::destroyBitmap1d() {
  destroyBuffer(m_bitmap1d);
  m_bitmap1d = NULL;
}

void PngFile::destroyPixels() {
  destroyBuffer(m_pixels);
  m_pixels = NULL;
}
}
