/* Copyright 2025 The OpenXLA Authors.

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

// A tool for computing GPU statistics from an XSpace protobuf.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {
const char* const kUsage = R"(
    This tool computes GPU statistics from an XSpace protobuf.

    Usage:

      bazel run compute_gpu_device_stats -- --input=path/to/xspace.pb

    Output:
      Device Time: 12345.67 us
      Device Memcpy Time: 1234.56 us
    )";
const char* const kDeviceName = "/device:GPU:0";

bool IsMemcpy(const tensorflow::profiler::XEvent& event,
              int64_t memcpy_details_id) {
  for (const auto& stat : event.stats()) {
    if (stat.metadata_id() == memcpy_details_id) {
      return true;
    }
  }
  return false;
}

void ProcessLineEvents(const tensorflow::profiler::XLine& line,
                       int64_t memcpy_details_id, int64_t* total_time_ps,
                       int64_t* memcpy_time_ps) {
  for (const auto& event : line.events()) {
    *total_time_ps += event.duration_ps();
    if (IsMemcpy(event, memcpy_details_id)) {
      *memcpy_time_ps += event.duration_ps();
    }
  }
}

absl::Status CalculateDeviceTimeAndMemcpy(
    const tensorflow::profiler::XSpace& xspace, const std::string& device_name,
    double* device_time_us, double* device_memcpy_time_us) {
  // Validate arguments
  if (device_time_us == nullptr || device_memcpy_time_us == nullptr) {
    return absl::InvalidArgumentError(
        "device_time_us and device_memcpy_time_us must not be null.");
  }

  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;

  // Iterate over planes to find the device
  for (const tensorflow::profiler::XPlane& plane : xspace.planes()) {
    if (plane.name() == device_name) {
      // Create a map for stat metadata
      absl::flat_hash_map<std::string, int64_t> stat_metadata_map;
      for (const auto& stat_metadata : plane.stat_metadata()) {
        stat_metadata_map[stat_metadata.second.name()] =
            stat_metadata.second.id();
      }

      // Determine the memcpy details ID
      int64_t memcpy_details_id = stat_metadata_map.contains("memcpy_details")
                                      ? stat_metadata_map["memcpy_details"]
                                      : -1;

      // Process each line in the plane
      for (const auto& line : plane.lines()) {
        ProcessLineEvents(line, memcpy_details_id, &total_time_ps,
                          &memcpy_time_ps);
      }
      break;  // Assuming only one plane matches device_name
    }
  }

  // Calculate the time in microseconds
  *device_time_us = static_cast<double>(total_time_ps) / 1e6;
  *device_memcpy_time_us = static_cast<double>(memcpy_time_ps) / 1e6;
  return absl::OkStatus();
}

absl::Status Run(const std::string& input) {
  LOG(INFO) << "Input file: " << input;

  // Read the XSpace protobuf
  tsl::Env* env = tsl::Env::Default();
  auto xspace_proto = std::make_unique<tensorflow::profiler::XSpace>();
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(env, input, xspace_proto.get()));

  LOG(INFO) << "Successfully parsed XSpace proto.";

  // Calculate device and memcpy times
  double device_time_us;
  double device_memcpy_time_us;
  const std::string device_name = kDeviceName;
  TF_RETURN_IF_ERROR(CalculateDeviceTimeAndMemcpy(
      *xspace_proto, device_name, &device_time_us, &device_memcpy_time_us));

  // Print the results
  std::cout << absl::StrFormat("Device Time: %.2f us\n", device_time_us)
            << absl::StrFormat("Device Memcpy Time: %.2f us\n",
                               device_memcpy_time_us);
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string input;
  std::vector<tsl::Flag> flag_list = {tsl::Flag("input", &input, "input file")};
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  absl::Status status = Run(input);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
