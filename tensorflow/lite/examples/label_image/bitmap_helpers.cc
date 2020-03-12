/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <unistd.h>  // NOLINT(build/include_order)

#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace label_image {

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          exit(-1);
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input, int* width,
                              int* height, int* channels) {
  int begin, end;

  std::ifstream file(input, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input << " not found\n";
    exit(-1);
  }
  LOG(INFO) << "Input file: " << input << "\n";

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t size = end - begin;

  LOG(INFO) << "Input file size: " << size << "\n";

  std::vector<uint8_t> img_bytes(size);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), size);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;

  LOG(INFO) << "Input file width, height, channels: " << *width
            << ", " << *height << ", " << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

std::vector<uint8_t> read_rgb(const std::string& input, int width,
                              int height, int channels) {

  std::ifstream file(input, std::ios::ate | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input << " not found\n";
    exit(-1);
  }
  LOG(INFO) << "Input file: " << input << "\n";

  auto size = file.tellg();
  file.seekg(0);

  LOG(INFO) << "Input file size: " << size << "\n";

  LOG(INFO) << "Input file width, height, channels: " << width
            << ", " << height << ", " << channels << "\n";

  std::vector<uint8_t> in(height * width * channels);
  if (!file.read(reinterpret_cast<char*>(in.data()), in.size())) {
    LOG(FATAL) << "input file read failed\n";
    exit(-1);
  }

  const int row_size = width * channels;
  return decode_bmp(in.data(), row_size, width, height, channels, false);
}

std::vector<uint8_t> read_blob(const std::string& input, int width,
                               int height, int channels) {

  std::ifstream file(input, std::ios::ate | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input << " not found\n";
    exit(-1);
  }
  LOG(INFO) << "Input file: " << input << "\n";

  auto size = file.tellg();
  file.seekg(0);

  LOG(INFO) << "Input file size: " << size << "\n";

  LOG(INFO) << "Input file width, height, channels: " << width
            << ", " << height << ", " << channels << "\n";

  std::vector<uint8_t> output(height * width * channels);
  if (!file.read(reinterpret_cast<char*>(output.data()), output.size())) {
    LOG(FATAL) << "input file read failed\n";
    exit(-1);
  }

  return output;
}

}  // namespace label_image
}  // namespace tflite
