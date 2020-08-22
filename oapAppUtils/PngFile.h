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

#ifndef PNGFILE_H
#define PNGFILE_H

#include <png.h>

#include <stdio.h>
#include "Image.h"

namespace oap
{
class PngFile : public Image
{
  public:
    PngFile(const std::string& path, bool truncateImage = true);

    virtual ~PngFile();

    virtual bool read(void* buffer, size_t repeat, size_t size) override;

    virtual oap::ImageSection getWidth() const override;

    virtual oap::ImageSection getHeight() const override;

    virtual void forceOutputWidth(const oap::ImageSection& optWidth) override;

    virtual void forceOutputHeight(const oap::ImageSection& optHeight) override;

    virtual oap::ImageSection getOutputWidth() const override;

    virtual oap::ImageSection getOutputHeight() const override;

    virtual std::string getSufix() const override;

  protected:
    virtual void closeProtected() override;

    virtual void loadBitmapProtected() override;

    void createPngBitmaps();

    void convertRawdataToBitmap1D();

    void destroyPngBitmaps();

    void setDestroyPngBitmapsAfterLoad (bool destroy);

    virtual void freeBitmapProtected() override;

    virtual void getPixelsVectorProtected(pixel_t* pixels) const override;

    virtual bool openProtected(const std::string& path) override;

    virtual bool isCorrectFormat() const override;

    virtual pixel_t getPixelProtected(unsigned int x, unsigned int y) const override;

    virtual void onSave(const std::string& path) override;
    virtual bool saveProtected(const std::string& path) override;

    png_bytep* copyBitmap(const ImageSection& width, const ImageSection& height);

    png_bytep* createBitmap2D(size_t width, size_t height,
                              size_t colorsCount) const;

    void destroyBitmap2d(png_bytep* bitmap2d, size_t height) const;

    png_byte* createBitmap1dFrom2d(png_bytep* bitmap2d, const ImageSection& optWidth,
                                   const ImageSection& optHeight, size_t colorsCount);

    oap::pixel_t* createPixelsVectorFrom1d(png_byte* bitmap1d,
                                           size_t width, size_t height,
                                           size_t colorsCount);

    template <typename T>
    void destroyBuffer(T* buffer)
    {
      if (buffer != NULL)
      {
        delete[] buffer;
      }
    }

  private:
    void calculateColorsCount();

    void calculateOutputSizes(size_t width, size_t height);

    void destroyBitmap1d();

    void destroyPixels();

    oap::ImageSection m_optWidth;
    oap::ImageSection m_optHeight;

    FILE* m_fp;
    png_structp m_png_ptr;
    png_infop m_info_ptr;
    png_bytep* m_bitmap2d;
    png_byte* m_bitmap1d;
    oap::pixel_t* m_pixels;

    bool m_destroyPngBitmaps;

    size_t m_colorsCount;
    bool m_truncateImage;

    struct PngInfo
    {
      png_byte color_type;
      png_byte bit_depth;
    };

    PngInfo m_pngInfo;
};
}

#endif  // PNGFILE_H
