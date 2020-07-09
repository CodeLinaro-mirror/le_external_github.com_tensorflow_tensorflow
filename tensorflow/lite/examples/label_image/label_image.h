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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_

#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace label_image {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool old_accel = false;
  bool input_floating = false;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool dump_tensors = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  float threshold = 0.001f;
  string model_name = "./mobilenet_quant_v1_224.tflite";
  tflite::FlatBufferModel* model;
  std::vector<string> input_names = {"./grace_hopper.bmp"};
  string labels_file_name = "./labels.txt";
  string input_layer_type = "uint8_t";
  string accelerator_list = "0";
  int number_of_threads = -1;
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
  int num_models = 1;
  int preferences = 0;
  std::vector<int> preferences_list;
  float frequency = 1000;
  std::vector<float> frequency_list;
  int parser = 0;
  float throughput = 0;
};

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
