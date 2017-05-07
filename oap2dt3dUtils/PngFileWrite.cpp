
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

namespace oap {

bool PngFile::writeImageToFile(PngFile* pngFile, const std::string& path) {
  if (pngFile->isLoaded() == true) {
    return false;
  }
  pngFile->open();
  pngFile->loadBitmap();
  pngFile->close();

  OptSize width = pngFile->getWidth();
  OptSize height = pngFile->getHeight();

  pngFile->freeBitmap();

  return PngFile::writeImageToFile(pngFile, path, width.optSize, height.optSize);
}

bool PngFile::writeImageToFile(PngFile* pngFile, const std::string& path,
                               size_t width, size_t height) {
  if (pngFile->isLoaded() == true) {
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

  pngFile->setAutomaticDestroyTmpData(false);

  pngFile->open();
  pngFile->loadBitmap();
  pngFile->close();

  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  png_bytep* row_pointers = pngFile->m_bitmap2d;

  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  fclose(fp);

  pngFile->freeBitmap();

  png_destroy_write_struct(&png, &info);

  return true;
}
}
