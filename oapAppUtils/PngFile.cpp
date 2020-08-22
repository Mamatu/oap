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

#include "PngFile.h"
#include "Exceptions.h"
#include "Logger.h"

#include <functional>
#include <map>

namespace oap
{

PngFile::PngFile(const std::string& path, bool truncateImage)
  : oap::Image(path),
    m_fp(nullptr),
    m_png_ptr(nullptr),
    m_bitmap2d(nullptr),
    m_bitmap1d(nullptr),
    m_pixels(nullptr),
    m_destroyPngBitmaps(true),
    m_truncateImage(truncateImage) {}

PngFile::~PngFile()
{
  PngFile::freeBitmap ();
  PngFile::close ();
}

bool PngFile::read(void* buffer, size_t count, size_t size)
{
  size_t result = fread(buffer, size, count, m_fp);
  if (result != count && feof(m_fp))
  {
    return true;
  }
  return false;
}

oap::ImageSection PngFile::getWidth() const
{
  return oap::ImageSection (png_get_image_width (m_png_ptr, m_info_ptr));
}

oap::ImageSection PngFile::getHeight() const
{
  return oap::ImageSection (png_get_image_height (m_png_ptr, m_info_ptr));
}

void PngFile::forceOutputWidth(const oap::ImageSection& optWidth)
{
  m_optWidth = optWidth;
}

void PngFile::forceOutputHeight(const oap::ImageSection& optHeight)
{
  m_optHeight = optHeight;
}

oap::ImageSection PngFile::getOutputWidth() const
{
  if (m_optWidth.getl() != 0)
  {
    return m_optWidth;
  }
  return getWidth();
}

oap::ImageSection PngFile::getOutputHeight() const
{
  if (m_optHeight.getl() != 0)
  {
    return m_optHeight;
  }
  return getHeight();
}

std::string PngFile::getSufix() const
{
  return "png";
}

void PngFile::closeProtected()
{
  if (m_fp != nullptr)
  {
    fclose(m_fp);
    m_fp = nullptr;
  }
}

void PngFile::loadBitmapProtected()
{
  if (m_png_ptr == nullptr)
  {
    m_png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    debugAssert (m_png_ptr != nullptr);

    m_info_ptr = png_create_info_struct (m_png_ptr);
    debugAssert (m_info_ptr != nullptr);

    png_init_io(m_png_ptr, m_fp);

    png_set_sig_bytes(m_png_ptr, 0);

    png_read_info(m_png_ptr, m_info_ptr);

    m_pngInfo.color_type = png_get_color_type(m_png_ptr, m_info_ptr);
    m_pngInfo.bit_depth = png_get_bit_depth(m_png_ptr, m_info_ptr);

    int number_of_passes = png_set_interlace_handling(m_png_ptr);
    png_read_update_info(m_png_ptr, m_info_ptr);
  }

  createPngBitmaps();

  if (m_destroyPngBitmaps)
  {
    destroyPngBitmaps ();
  }

  m_destroyPngBitmaps = true;
}

void PngFile::createPngBitmaps()
{
  size_t width = 0;
  size_t height = 0;

  if (m_png_ptr != nullptr && m_info_ptr != nullptr)
  {
    calculateColorsCount();
  }

  if (m_bitmap2d == nullptr && m_png_ptr != nullptr && m_info_ptr != nullptr)
  {
    width = getWidth().getl();
    height = getHeight().getl();

    const size_t width1 =
      png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

    debugAssert(width1 / m_colorsCount == width);

    m_bitmap2d = createBitmap2D(width, height, m_colorsCount);

    png_read_image(m_png_ptr, m_bitmap2d);
  }

  if (m_bitmap1d == nullptr && m_bitmap2d != nullptr)
  {
    calculateOutputSizes(width, height);

    convertRawdataToBitmap1D();
  }
}

void PngFile::convertRawdataToBitmap1D()
{
  destroyBitmap1d();

  destroyPixels();

  m_bitmap1d = createBitmap1dFrom2d(m_bitmap2d, getOutputWidth(),
                                    getOutputHeight(), m_colorsCount);

  m_pixels = createPixelsVectorFrom1d(m_bitmap1d, getOutputWidth().getl(), getOutputHeight().getl(), m_colorsCount);
}

void PngFile::destroyPngBitmaps()
{
  destroyBitmap2d (m_bitmap2d, getHeight().getl());

  m_bitmap2d = nullptr;

  destroyBitmap1d();
}

void PngFile::setDestroyPngBitmapsAfterLoad(bool destroy)
{
  m_destroyPngBitmaps = destroy;
}

void PngFile::freeBitmapProtected()
{
  destroyPngBitmaps();

  if (m_png_ptr != nullptr && m_info_ptr != nullptr)
  {
    png_destroy_info_struct(m_png_ptr, &m_info_ptr);
  }

  if (m_png_ptr != nullptr)
  {
    png_destroy_read_struct(&m_png_ptr, nullptr, nullptr);
  }

  destroyPixels();

  m_png_ptr = nullptr;
  m_info_ptr = nullptr;
}

void PngFile::getPixelsVectorProtected(pixel_t* pixels) const
{
  const size_t length = getLength();
  memcpy(pixels, m_pixels, sizeof(pixel_t) * length);
}

bool PngFile::openProtected(const std::string& path)
{
  m_fp = fopen(path.c_str(), "rb");
  return m_fp != nullptr;
}

bool PngFile::isCorrectFormat() const
{
  const size_t header_size = 8;
  unsigned char header[header_size];
  return png_sig_cmp(header, 0, header_size);
}

pixel_t PngFile::getPixelProtected(unsigned int x, unsigned int y) const
{
  const size_t width = getOutputWidth().getl();
  return m_pixels[y * width + x];
}

void PngFile::onSave (const std::string& path)
{
  this->setDestroyPngBitmapsAfterLoad (false);

  bool bitmapToSaveExists = (m_bitmap2d != nullptr);

  if (false == bitmapToSaveExists)
  {
    // if m_bitmap2d doesn't exist it should be reloaded
    this->freeBitmap();
    this->close();
  }
}

bool PngFile::saveProtected (const std::string& path)
{
  FILE* fp = fopen(path.c_str(), "wb");

  if (fp == nullptr)
  {
    return false;
  }

  png_structp png =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (png == nullptr)
  {
    fclose(fp);
    return false;
  }

  png_infop info = png_create_info_struct(png);

  if (info == nullptr)
  {
    fclose(fp);
    return false;
  }

  png_init_io(png, fp);

  ImageSection outputWidth = this->getOutputWidth();
  ImageSection outputHeight = this->getOutputHeight();

  png_set_IHDR(png, info, outputWidth.getl(), outputHeight.getl(), m_pngInfo.bit_depth,
               m_pngInfo.color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  png_bytep* row_pointers = nullptr;

  bool rowPointerToDestroy = false;

  const bool isNotTruncated = outputWidth.getl() == this->getWidth().getl()
  && outputHeight.getl() == this->getHeight().getl() 
  && outputWidth.getp() == 0 && outputHeight.getp() == 0;

  if (isNotTruncated)
  {
    row_pointers = this->m_bitmap2d;
  }
  else
  {
    rowPointerToDestroy = true;
    row_pointers = this->copyBitmap (outputWidth, outputHeight);
  }

  png_write_image(png, row_pointers);
  png_write_end(png, nullptr);

  fclose(fp);

  this->freeBitmap();

  png_destroy_write_struct(&png, &info);

  if (rowPointerToDestroy)
  {
    destroyBitmap2d(row_pointers, outputHeight.getl());
  }

  return true;
}

png_bytep* PngFile::copyBitmap(const ImageSection& width, const ImageSection& height)
{
  const png_bytep* origin = this->m_bitmap2d;

  png_bytep* copy = new png_bytep[height.getl()];

  const size_t localWidth =
    png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

  const size_t imageWidth = getWidth().getl() + getWidth().getp();

  size_t factor = localWidth / imageWidth;

  for (size_t fa = 0; fa < height.getl(); ++fa)
  {
    copy[fa] = new png_byte[width.getl() * factor];
    memcpy(copy[fa], &origin[height.getp() + fa][width.getp() * factor],
           width.getl() * factor * sizeof(png_byte));
  }
  return copy;
}

png_bytep* PngFile::createBitmap2D(size_t width, size_t height,
                                   size_t colorsCount) const
{
  png_bytep* bitmap2d = new png_bytep[height];

  for (unsigned int fa = 0; fa < height; ++fa)
  {
    bitmap2d[fa] = new png_byte[width * colorsCount];
  }

  return bitmap2d;
}

void PngFile::destroyBitmap2d(png_bytep* bitmap2d, size_t height) const
{
  if (bitmap2d != nullptr)
  {
    for (unsigned int fa = 0; fa < height; ++fa)
    {
      delete[] bitmap2d[fa];
    }

    delete[] bitmap2d;
  }
}

png_byte* PngFile::createBitmap1dFrom2d(png_bytep* bitmap2d,
                                        const ImageSection& optWidth,
                                        const ImageSection& optHeight,
                                        size_t colorsCount)
{
  const size_t beginC = optWidth.getp();
  const size_t beginR = optHeight.getp();

  size_t width = optWidth.getl();
  size_t widthOfBytes = optWidth.getl() * colorsCount;
  size_t height = optHeight.getl();

  png_byte* bitmap1d = new png_byte[widthOfBytes * height];

  const size_t lengthToCopy = widthOfBytes * sizeof(png_byte);

  for (size_t fa = beginR; fa < beginR + height; ++fa)
  {
    memcpy(&(bitmap1d[(fa - beginR) * lengthToCopy]),
           &(bitmap2d[fa][beginC * colorsCount]), lengthToCopy);
  }

  return bitmap1d;
}

oap::pixel_t* PngFile::createPixelsVectorFrom1d(png_byte* bitmap1d,
    size_t width, size_t height,
    size_t colorsCount)
{
  oap::pixel_t* pixels = new pixel_t[width * height];

  auto handleRGB = [&bitmap1d, &pixels] (const size_t index, const size_t index1)
  {
    const png_byte r = bitmap1d[index + 0];
    const png_byte g = bitmap1d[index + 1];
    const png_byte b = bitmap1d[index + 2];
    pixels[index1] = convertRgbToPixel(r, g, b);
  };

  auto handleRGBA = [&bitmap1d, &pixels] (const size_t index, const size_t index1)
  {
    const png_byte r = bitmap1d[index + 0];
    const png_byte g = bitmap1d[index + 1];
    const png_byte b = bitmap1d[index + 2];
    const png_byte a = bitmap1d[index + 3];
    pixels[index1] = convertRgbToPixel(r, g, b);
  };

  auto handleBlackWhite = [&bitmap1d, &pixels] (const size_t index, const size_t index1)
  {
    const png_byte p = bitmap1d[index];
    pixels[index1] = convertRgbToPixel(p, p, p);
  };

  std::map<size_t, std::function<void(const size_t, const size_t)>> coloursMap =
  {
    {1, handleBlackWhite},
    {3, handleRGB},
    {4, handleRGBA},
  };

  auto it = coloursMap.find (colorsCount);

  debugAssert (it != coloursMap.end ());

  for (size_t fa = 0; fa < height; ++fa)
  {
    for (size_t fb = 0; fb < width; ++fb)
    {
      const size_t index = fa * width * colorsCount + fb * colorsCount;
      const size_t index1 = fa * width + fb;
      it->second (index, index1);
    }
  }
  return pixels;
}

void PngFile::calculateColorsCount()
{
  const size_t localWidth =
    png_get_rowbytes(m_png_ptr, m_info_ptr) / sizeof(png_byte);

  const size_t imageWidth = getWidth().getl() + getWidth().getp();

  m_colorsCount = localWidth / imageWidth;
}

void PngFile::calculateOutputSizes(size_t width, size_t height)
{
  if (m_truncateImage == false)
  {
    m_optWidth = getWidth();
    m_optHeight = getHeight();
  }
  else
  {
    if (m_optWidth.getl() == 0)
    {
      oap::ImageSection owidth = oap::GetOptWidth<png_bytep*, png_byte>(
                              m_bitmap2d, width, height, m_colorsCount);

      m_optWidth = owidth;
    }

    if (m_optHeight.getl() == 0)
    {
      oap::ImageSection oheight = oap::GetOptHeight<png_bytep*, png_byte>(
                               m_bitmap2d, width, height, m_colorsCount);

      m_optHeight = oheight;
    }
  }
}

void PngFile::destroyBitmap1d()
{
  destroyBuffer(m_bitmap1d);
  m_bitmap1d = nullptr;
}

void PngFile::destroyPixels()
{
  destroyBuffer(m_pixels);
  m_pixels = nullptr;
}
}
