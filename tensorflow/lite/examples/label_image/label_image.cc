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

#include "tensorflow/lite/examples/label_image/label_image.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <numeric>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#define LOG(x) std::cerr

#define TIMER_START(NAME) \
  auto t_##NAME##_start = std::chrono::high_resolution_clock::now()
#define TIMER_STOP(NAME) \
  auto t_## NAME##_stop = std::chrono::high_resolution_clock::now()
#define TIMER_GET(NAME) \
  std::chrono::duration_cast<std::chrono::microseconds>( \
      t_##NAME##_stop - t_##NAME##_start).count()
#define TIMER_PRINT(NAME) \
  std::cout << #NAME << ": " << TIMER_GET(NAME) << " us\n"
#define TIMER_STOP_PRINT(NAME) \
  TIMER_STOP(NAME); TIMER_PRINT(NAME)

namespace tflite {
namespace label_image {

  // Synchronization variables between runs:
  // 1. Wait for all threads to load the model to start looping
  // 2. When one of them is done, terminate the rest
  // This is to make sure all the timings are concurrent
  int num_init = 0;
  std::condition_variable init_cv;
  std::mutex init_mutex;
  int num_finished = 0;
  std::condition_variable finished_cv;
  std::mutex finished_mutex;
  // int num_threads;
  std::mutex print_mutex;
  std::pair<double, double> get_mean_std(const std::vector<double>& times) {

  // mean
  int N = times.size();
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / N;

  // std deviation
  std::vector<double> diff(N);
  std::transform(times.begin(), times.end(), diff.begin(),
                 [mean](double a) { return a - mean; });
  auto sigma = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.);
  sigma = std::sqrt(sigma / N);

  return std::make_pair(mean, sigma);
}

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(Settings* s) {
#if defined(__ANDROID__)
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.inference_priority1 =
      s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  return evaluation::CreateGPUDelegate(&gpu_opts);
#else
  return evaluation::CreateGPUDelegate();
#endif
}

TfLiteDelegatePtrMap GetDelegates(Settings* s) {
  TfLiteDelegatePtrMap delegates;
  if (s->gl_backend) {
    auto delegate = CreateGPUDelegate(s);
    if (!delegate) {
      LOG(INFO) << "GPU acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("GPU", std::move(delegate));
    }
  }

  if (s->accel) {
    StatefulNnApiDelegate::Options options;
    options.execution_preference =
        static_cast<StatefulNnApiDelegate::Options::ExecutionPreference>(
        s->preferences);

    auto delegate = evaluation::CreateNNAPIDelegate(options);
    if (!delegate) {
      LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  }

  if (s->hexagon_delegate) {
    const std::string libhexagon_path("/data/local/tmp");
    auto delegate =
        evaluation::CreateHexagonDelegate(libhexagon_path, s->profiling);

    if (!delegate) {
      LOG(INFO) << "Hexagon acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("Hexagon", std::move(delegate));
    }
  }

  return delegates;
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

std::vector<string> parse_comma_separated(const std::string& str) {
  // Parse the models if more than one comma separated
  size_t start = 0, end = 0;
  std::vector<std::string> out;

  while (start < str.size()) {
    end = str.find(",", start);
    if (end == std::string::npos) {
      // get rest of string
      out.push_back(str.substr(start));
      break;
    } else {
      out.push_back(str.substr(start, end - start));
    }
    start = end + 1;
  }

  return out;
}

void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  LOG(INFO) << "Setting preferences for model " << s->model_name <<
      " to " << std::hex << s->preferences << std::dec << "\n";

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  s->model = model.get();
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    return;
  }

  interpreter->UseNNAPI(s->old_accel);
  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Error allocating tensors\n";
    exit(-1);
  }

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  LOG(INFO) << "Model has " << inputs.size() << " inputs and "
    << outputs.size() << " outputs\n";

  if (s->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  auto delegates_ = GetDelegates(s);
  for (const auto& delegate : delegates_) {
    if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
        kTfLiteOk) {
      LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
    } else {
      LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
    return;
  }

  if (s->verbose) PrintInterpreterState(interpreter.get());

  int image_width;
  int image_height;
  int image_channels;

  for (int i = 0; i < s->input_names.size(); ++i) {
    LOG(INFO) << "Processing input: " << i << "\n";
    const std::string input_name = s->input_names[i];
    int input = interpreter->inputs()[i];
    auto* input_tensor = interpreter->tensor(input);

    if (s->verbose) {
      LOG(INFO) << "input: " << input_tensor->name <<
      " id:" << input << "\n";
    }

    // get input dimension from the input tensor metadata
    // assuming one input only
    TfLiteIntArray* dims = input_tensor->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->size > 3 ? dims->data[3] : 1;

    std::vector<uint8_t> in;

    in = read_bmp(input_name, &image_width, &image_height, &image_channels);
    LOG(INFO) << "Input has bytes = " << in.size() << "\n";

    // get input dimension from the input tensor metadata
    // assuming one input only
    switch (input_tensor->type) {
      case kTfLiteFloat32:
        s->input_floating = true;
        resize<float>(input_tensor->data.f, in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, s);
        break;
      case kTfLiteUInt8:
        resize<uint8_t>(input_tensor->data.uint8, in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, s);
        break;
      default:
        LOG(FATAL) << "cannot handle input type "<<
                  input_tensor->type << " yet";
        return;
    }
  }

  auto profiler =
      absl::make_unique<profiling::Profiler>(s->max_profiling_buffer_entries);
  interpreter->SetProfiler(profiler.get());

  if (s->profiling) profiler->StartProfiling();
  if (s->loop_count > 1)
    for (int i = 0; i < s->number_of_warmup_runs; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

  if (s->num_models > 1) {
    // wait for rest to finish init
    std::unique_lock<std::mutex> lk(init_mutex);
    num_init++;
    if (num_init == s->num_models) {
      init_cv.notify_all();
    } else {
      init_cv.wait(lk, [&s] { return num_init == s->num_models; });
    }
  }

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  std::vector<double> times;
  times.reserve(s->loop_count);
  bool quit = false;

  // minimum time between invocations based on desired frequency
  float time_between_ms = 1000.f / s->frequency;

  for (int i = 0; i < s->loop_count && !quit; i++) {
    TIMER_START(run);
    auto status = interpreter->Invoke();
    TIMER_STOP(run);

    float time_ms = TIMER_GET(run) / 1000.;
    times.push_back(time_ms);

    if (status != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite for " << s->model_name << "\n";
    }

    {
      std::unique_lock<std::mutex> lk(print_mutex);
      LOG(INFO) << s->model_name << ": run = " << time_ms << " ms\n";
    }

    // other threads finished?
    if (s->num_models > 1) {
      std::unique_lock<std::mutex> lk(finished_mutex);
      if (num_finished > 0) {
        quit = true;
      }
    }
  }

  // signal for others to quit
  if (s->num_models > 1) {
    std::unique_lock<std::mutex> lk(finished_mutex);
    num_finished++;
  }

  auto mean_std = get_mean_std(times);
  {
    std::unique_lock<std::mutex> lk(print_mutex);
    LOG(INFO) << "Model: " << s->model_name <<
        "\tmean=" << mean_std.first << "ms" <<
        "\tstd=" << mean_std.second << "ms" <<
        " runs=" << times.size() <<
        " max freq=" << 1000 / mean_std.first <<
        "\n";
  }

  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked \n";
  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
            << " ms \n";

  if (s->profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for (int i = 0; i < profile_events.size(); i++) {
      auto subgraph_index = profile_events[i]->event_subgraph_index;
      auto op_index = profile_events[i]->event_metadata;
      const auto subgraph = interpreter->subgraph(subgraph_index);
      const auto node_and_registration =
          subgraph->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
                         registration);
    }
  }

  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                       s->number_of_results, threshold, &top_results, true);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, s->number_of_results, threshold,
                         &top_results, false);
      break;
    default:
      LOG(FATAL) << "cannot handle output type "
                 << interpreter->tensor(output)->type << " yet";
      return;
  }

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
    return;

  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
  }

}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not\n"
      << "--preferences, -x: [comma-separated] preferences for the choosen delegate in hex\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: use GL GPU Delegate on Android\n"
      << "--hexagon_delegate: use Hexagon Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: [comma-separated] tflite model(s)\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--frequency, -J: [comma-separated] desired frequency for model(s)\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"old_accelerated", required_argument, nullptr, 'd'},
        {"preferences", required_argument, nullptr, 'x'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"hexagon_delegate", required_argument, nullptr, 'j'},
        {"frequency", required_argument, nullptr, 'J'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:j:l:m:p:r:s:t:v:w:x:J:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'x': {
          auto pref_str = parse_comma_separated(std::string(optarg));
          if (pref_str.size() == 1) {
            s.preferences =
              strtol(optarg, nullptr, 16);  // NOLINT(runtime/deprecated_fn)
          } else {
           for (auto pref: pref_str) {
              s.preferences_list.push_back(strtol(pref.c_str(), nullptr, 16));
            }
          }
        }
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'd':
        s.old_accel =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.max_profiling_buffer_entries =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'g':
        s.gl_backend =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_names = parse_comma_separated(optarg);
        break;
      case 'j':
        s.hexagon_delegate = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'J': {
          auto freq_str = parse_comma_separated(std::string(optarg));
          if (freq_str.size() == 1) {
            s.frequency = strtod(optarg, nullptr);  // NOLINT(runtime/deprecated_fn)
          } else {
            for (auto freq: freq_str) {
              s.frequency_list.push_back(strtod(freq.c_str(), nullptr));
            }
          }
        }
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }


  // Parse the models if more than one comma separated
  auto models = parse_comma_separated(s.model_name);
  s.num_models = models.size();
  LOG(INFO) << "Found " << s.num_models << " models to run.\n";

  if (s.num_models == 1) {
    RunInference(&s);
  } else {
    // Loop over models and create threads
    std::vector<std::thread> threads(s.num_models);
    std::vector<Settings> ss(s.num_models, s);

    for (int i = 0; i < s.num_models; ++i) {
      ss[i].model_name = models[i];
      if (s.preferences_list.size() > i) {
        ss[i].preferences = s.preferences_list[i];
      }
      if (s.frequency_list.size() > i) {
        ss[i].frequency = s.frequency_list[i];
      }

      LOG(INFO) << "Thread " << i <<
          " with model " << ss[i].model_name <<
          " pref = " << std::hex << ss[i].preferences << std::dec <<
          " freq = " << ss[i].frequency <<
          "\n";

      threads[i] = std::thread(RunInference, &ss[i]);
    }

    for (int i = 0; i < s.num_models; ++i) {
      threads[i].join();
    }
  }

  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
