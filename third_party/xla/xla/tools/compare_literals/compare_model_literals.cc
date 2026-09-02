/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tools/compare_literals/compare_model_literals.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "json/json.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"

namespace xla::compare_literals {
namespace {

constexpr absl::string_view kLiteralPrefix = "literal_";
constexpr absl::string_view kDevicePrefix = "device_";

// Parses filename patterns such as:
// "output.hlo_0.task_0.device_7.literal_27.pb",
// "output.device_1.literal_3.pb", or "literal_5.pb".
std::optional<LiteralKey> ParseLiteralFilename(absl::string_view filename) {
  if (!absl::EndsWith(filename, ".pb")) {
    return std::nullopt;
  }
  const size_t lit_pos = filename.rfind(kLiteralPrefix);
  if (lit_pos == absl::string_view::npos) {
    return std::nullopt;
  }

  const absl::string_view after_lit =
      filename.substr(lit_pos + kLiteralPrefix.size());
  const size_t dot_pos = after_lit.find_first_of("._");
  const absl::string_view lit_num_str = (dot_pos == absl::string_view::npos)
                                            ? after_lit
                                            : after_lit.substr(0, dot_pos);
  LiteralKey parsed;
  if (!absl::SimpleAtoi(lit_num_str, &parsed.literal_id) ||
      parsed.literal_id < 0) {
    return std::nullopt;
  }

  const size_t dev_pos = filename.rfind(kDevicePrefix);
  if (dev_pos != absl::string_view::npos) {
    const absl::string_view after_dev =
        filename.substr(dev_pos + kDevicePrefix.size());
    const size_t dev_dot = after_dev.find_first_of("._");
    const absl::string_view dev_num_str = (dev_dot == absl::string_view::npos)
                                              ? after_dev
                                              : after_dev.substr(0, dev_dot);
    if (!absl::SimpleAtoi(dev_num_str, &parsed.device_id) ||
        parsed.device_id < 0) {
      return std::nullopt;
    }
  } else {
    parsed.device_id = 0;
  }
  return parsed;
}

void SetJsonDouble(Json::Value& parent, const char* key, double val) {
  if (std::isnan(val)) {
    parent[key] = "NaN";
  } else if (std::isinf(val)) {
    parent[key] = (val > 0) ? "Infinity" : "-Infinity";
  } else {
    parent[key] = val;
  }
}

struct ComparisonTask {
  int64_t literal_id = 0;
  int64_t device_id = 0;
  std::string golden_path;
  std::string test_path;
};

struct TaskOutcome {
  ComparisonTask task;
  absl::StatusOr<ComparisonResult> result;
};

}  // namespace

absl::StatusOr<ModelComparisonResult> CompareModelDirectories(
    absl::string_view golden_dir, absl::string_view test_dir,
    const ModelComparisonOptions& options) {
  tsl::Env* env = tsl::Env::Default();

  if (!env->FileExists(golden_dir).ok()) {
    return absl::NotFoundError(
        absl::StrCat("Golden directory does not exist: ", golden_dir));
  }
  if (!env->FileExists(test_dir).ok()) {
    return absl::NotFoundError(
        absl::StrCat("Test directory does not exist: ", test_dir));
  }

  std::vector<std::string> golden_files;
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(golden_dir), &golden_files))
      << "Failed to list golden directory: " << golden_dir;

  std::vector<std::string> test_files;
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(test_dir), &test_files))
      << "Failed to list test directory: " << test_dir;

  // Map key is LiteralKey (literal_id, device_id)
  std::map<LiteralKey, std::string> golden_map;
  for (const std::string& fname : golden_files) {
    const auto parsed = ParseLiteralFilename(fname);
    if (!parsed.has_value()) {
      continue;
    }
    if (!options.target_devices.empty() &&
        !absl::c_linear_search(options.target_devices, parsed->device_id)) {
      continue;
    }
    auto [it, inserted] =
        golden_map.try_emplace(*parsed, tsl::io::JoinPath(golden_dir, fname));
    if (!inserted) {
      LOG(WARNING) << "Duplicate literal file in golden_dir: literal "
                   << parsed->literal_id << ", device " << parsed->device_id
                   << " (" << it->second << " vs " << fname << ")";
    }
  }

  std::map<LiteralKey, std::string> test_map;
  for (const std::string& fname : test_files) {
    const auto parsed = ParseLiteralFilename(fname);
    if (!parsed.has_value()) {
      continue;
    }
    if (!options.target_devices.empty() &&
        !absl::c_linear_search(options.target_devices, parsed->device_id)) {
      continue;
    }
    auto [it, inserted] =
        test_map.try_emplace(*parsed, tsl::io::JoinPath(test_dir, fname));
    if (!inserted) {
      LOG(WARNING) << "Duplicate literal file in test_dir: literal "
                   << parsed->literal_id << ", device " << parsed->device_id
                   << " (" << it->second << " vs " << fname << ")";
    }
  }

  ModelComparisonResult model_result;
  model_result.golden_dir = std::string(golden_dir);
  model_result.test_dir = std::string(test_dir);

  std::vector<ComparisonTask> tasks;
  std::vector<int64_t> discovered_devices;

  for (const auto& [key, g_path] : golden_map) {
    const auto it = test_map.find(key);
    if (it == test_map.end()) {
      LOG(WARNING) << "Missing in test_dir: literal " << key.literal_id
                   << ", device " << key.device_id;
      model_result.missing_in_test.push_back(key);
      continue;
    }
    ComparisonTask task;
    task.literal_id = key.literal_id;
    task.device_id = key.device_id;
    task.golden_path = g_path;
    task.test_path = it->second;
    tasks.push_back(std::move(task));
    if (!absl::c_linear_search(discovered_devices, key.device_id)) {
      discovered_devices.push_back(key.device_id);
    }
  }

  absl::c_sort(discovered_devices);
  model_result.devices = std::move(discovered_devices);

  for (const auto& [key, unused_path] : test_map) {
    if (golden_map.find(key) == golden_map.end()) {
      LOG(WARNING) << "Missing in golden_dir: literal " << key.literal_id
                   << ", device " << key.device_id;
      model_result.missing_in_golden.push_back(key);
    }
  }

  if (tasks.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No matching literal files found between ", golden_dir,
                     " and ", test_dir));
  }

  std::vector<TaskOutcome> outcomes(tasks.size());
  for (size_t i = 0; i < tasks.size(); ++i) {
    outcomes[i].task = tasks[i];
  }

  const int num_threads = std::max(1, options.num_threads);
  {
    tsl::thread::ThreadPool thread_pool(env, "CompareModel", num_threads);
    absl::BlockingCounter counter(static_cast<int>(tasks.size()));

    for (size_t i = 0; i < tasks.size(); ++i) {
      thread_pool.Schedule([&outcomes, &options, i, &counter]() {
        outcomes[i].result = CompareLiteralFiles(outcomes[i].task.golden_path,
                                                 outcomes[i].task.test_path,
                                                 options.comparison_options);
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }

  // Group outcomes by literal_id
  std::map<int64_t, std::vector<TaskOutcome>> grouped_outcomes;
  for (auto& outcome : outcomes) {
    grouped_outcomes[outcome.task.literal_id].push_back(std::move(outcome));
  }

  for (auto& [lit_id, dev_outcomes] : grouped_outcomes) {
    ModelLiteralEntry entry;
    entry.literal_index = lit_id;
    entry.literal_name = absl::StrCat("literal_", lit_id);

    double sum_mean_rel = 0.0;
    int valid_devices = 0;

    for (const auto& outcome : dev_outcomes) {
      if (!outcome.result.ok()) {
        LOG(ERROR) << "Comparison failed for literal " << lit_id << " device "
                   << outcome.task.device_id << ": " << outcome.result.status();
        ++entry.failed_devices;
        entry.min_exact_match_pct = 0.0;
        LiteralDeviceComparison dev_res;
        dev_res.device_id = outcome.task.device_id;
        dev_res.comparison_ok = false;
        dev_res.error_message = outcome.result.status().ToString();
        entry.per_device_results.push_back(std::move(dev_res));
        continue;
      }

      const ComparisonResult& comp = *outcome.result;
      if (entry.shape_str.empty()) {
        entry.shape_str = comp.shape_str;
        entry.element_type = comp.element_type;
        entry.element_count = comp.total_elements;
      }

      const double match_pct =
          (comp.total_elements > 0)
              ? (100.0 * comp.exact_matches / comp.total_elements)
              : 100.0;

      double sugg_abs = 0.0;
      double sugg_rel = 0.0;
      if (comp.suggested_error_spec.has_value()) {
        sugg_abs = comp.suggested_error_spec->abs_bound;
        sugg_rel = comp.suggested_error_spec->rel_bound;
      }

      LiteralDeviceComparison dev_res;
      dev_res.device_id = outcome.task.device_id;
      dev_res.comparison_ok = true;
      dev_res.exact_matches = comp.exact_matches;
      dev_res.exact_match_pct = match_pct;
      dev_res.mismatches = comp.mismatches;
      dev_res.nan_mismatches = comp.nan_mismatches;
      dev_res.inf_mismatches = comp.inf_mismatches;
      dev_res.max_abs_error = comp.max_abs_error;
      dev_res.max_rel_error = comp.max_rel_error;
      dev_res.mean_rel_error = comp.histogram.mean_rel_error;
      dev_res.suggested_abs_error = sugg_abs;
      dev_res.suggested_rel_error = sugg_rel;
      entry.per_device_results.push_back(std::move(dev_res));

      entry.min_exact_match_pct =
          std::min(entry.min_exact_match_pct, match_pct);
      entry.total_nan_mismatches += comp.nan_mismatches;
      entry.total_inf_mismatches += comp.inf_mismatches;
      entry.max_abs_error = std::max(entry.max_abs_error, comp.max_abs_error);
      entry.max_rel_error = std::max(entry.max_rel_error, comp.max_rel_error);
      entry.suggested_abs_error = std::max(entry.suggested_abs_error, sugg_abs);
      entry.suggested_rel_error = std::max(entry.suggested_rel_error, sugg_rel);

      sum_mean_rel += comp.histogram.mean_rel_error;
      ++valid_devices;
    }

    entry.num_devices = entry.per_device_results.size();
    if (valid_devices == 0) {
      entry.min_exact_match_pct = 0.0;
    } else {
      entry.mean_rel_error = sum_mean_rel / valid_devices;
    }

    absl::c_sort(
        entry.per_device_results,
        [](const LiteralDeviceComparison& a, const LiteralDeviceComparison& b) {
          return a.device_id < b.device_id;
        });

    model_result.entries.push_back(std::move(entry));
  }

  absl::c_sort(model_result.entries,
               [](const ModelLiteralEntry& a, const ModelLiteralEntry& b) {
                 return a.literal_index < b.literal_index;
               });

  return model_result;
}

std::string ModelComparisonResult::ToJson() const {
  Json::Value root(Json::objectValue);
  root["golden_dir"] = golden_dir;
  root["test_dir"] = test_dir;
  root["total_literals"] = static_cast<Json::UInt64>(entries.size());

  root["devices"] = Json::Value(Json::arrayValue);
  for (int64_t d : devices) {
    root["devices"].append(static_cast<Json::Int64>(d));
  }

  if (!missing_in_test.empty()) {
    root["missing_in_test"] = Json::Value(Json::arrayValue);
    for (const auto& item : missing_in_test) {
      Json::Value m(Json::objectValue);
      m["literal"] = static_cast<Json::Int64>(item.literal_id);
      m["device"] = static_cast<Json::Int64>(item.device_id);
      root["missing_in_test"].append(std::move(m));
    }
  }

  if (!missing_in_golden.empty()) {
    root["missing_in_golden"] = Json::Value(Json::arrayValue);
    for (const auto& item : missing_in_golden) {
      Json::Value m(Json::objectValue);
      m["literal"] = static_cast<Json::Int64>(item.literal_id);
      m["device"] = static_cast<Json::Int64>(item.device_id);
      root["missing_in_golden"].append(std::move(m));
    }
  }

  root["literals"] = Json::Value(Json::arrayValue);
  for (const auto& e : entries) {
    Json::Value lit(Json::objectValue);
    lit["index"] = static_cast<Json::Int64>(e.literal_index);
    lit["name"] = e.literal_name;
    lit["shape"] = e.shape_str;
    lit["element_type"] = e.element_type;
    lit["element_count"] = static_cast<Json::Int64>(e.element_count);

    Json::Value agg(Json::objectValue);
    agg["num_devices"] = static_cast<Json::Int64>(e.num_devices);
    agg["failed_devices"] = static_cast<Json::Int64>(e.failed_devices);
    agg["min_exact_match_pct"] = e.min_exact_match_pct;
    SetJsonDouble(agg, "max_abs_error", e.max_abs_error);
    SetJsonDouble(agg, "max_rel_error", e.max_rel_error);
    SetJsonDouble(agg, "mean_rel_error", e.mean_rel_error);
    SetJsonDouble(agg, "suggested_abs_error", e.suggested_abs_error);
    SetJsonDouble(agg, "suggested_rel_error", e.suggested_rel_error);
    agg["nan_mismatches"] = static_cast<Json::Int64>(e.total_nan_mismatches);
    agg["inf_mismatches"] = static_cast<Json::Int64>(e.total_inf_mismatches);
    lit["aggregate"] = std::move(agg);

    lit["devices"] = Json::Value(Json::arrayValue);
    for (const auto& dev : e.per_device_results) {
      Json::Value d(Json::objectValue);
      d["device"] = static_cast<Json::Int64>(dev.device_id);
      d["comparison_ok"] = dev.comparison_ok;
      if (!dev.comparison_ok) {
        d["error_message"] = dev.error_message;
      }
      d["exact_matches"] = static_cast<Json::Int64>(dev.exact_matches);
      d["exact_match_pct"] = dev.exact_match_pct;
      d["mismatches"] = static_cast<Json::Int64>(dev.mismatches);
      SetJsonDouble(d, "max_abs_error", dev.max_abs_error);
      SetJsonDouble(d, "max_rel_error", dev.max_rel_error);
      SetJsonDouble(d, "mean_rel_error", dev.mean_rel_error);
      SetJsonDouble(d, "suggested_abs_error", dev.suggested_abs_error);
      SetJsonDouble(d, "suggested_rel_error", dev.suggested_rel_error);
      d["nan_mismatches"] = static_cast<Json::Int64>(dev.nan_mismatches);
      d["inf_mismatches"] = static_cast<Json::Int64>(dev.inf_mismatches);
      lit["devices"].append(std::move(d));
    }

    root["literals"].append(std::move(lit));
  }

  Json::StreamWriterBuilder builder;
  builder["indentation"] = "  ";
  return Json::writeString(builder, root);
}

std::string ModelComparisonResult::ToTsv() const {
  std::string tsv =
      "literal\tshape\ttype\telements\tdevices\tfailed_devices\tmin_exact_pct\t"
      "max_abs_err\tmax_rel_err\tmean_rel_err\tsugg_abs_err\tsugg_rel_err\t"
      "nan_count\tinf_count\n";

  for (const auto& e : entries) {
    absl::StrAppendFormat(
        &tsv,
        "%v\t%s\t%s\t%v\t%v\t%v\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%v\t%v\n",
        e.literal_index, e.shape_str, e.element_type, e.element_count,
        e.num_devices, e.failed_devices, e.min_exact_match_pct, e.max_abs_error,
        e.max_rel_error, e.mean_rel_error, e.suggested_abs_error,
        e.suggested_rel_error, e.total_nan_mismatches, e.total_inf_mismatches);
  }
  return tsv;
}

std::string ModelComparisonResult::ToDeviceTsv() const {
  std::string tsv =
      "literal\tdevice\tcomparison_ok\terror_message\tshape\ttype\telements\t"
      "exact_matches\texact_pct\tmax_abs_err\tmax_rel_err\tmean_rel_err\t"
      "sugg_abs_err\tsugg_rel_err\tnan_count\tinf_count\n";

  for (const auto& e : entries) {
    for (const auto& dev : e.per_device_results) {
      std::string sanitized_err = dev.error_message;
      absl::c_replace(sanitized_err, '\t', ' ');
      absl::c_replace(sanitized_err, '\n', ' ');
      absl::StrAppendFormat(
          &tsv,
          "%v\t%v\t%s\t%s\t%s\t%s\t%v\t%v\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e\t%."
          "6e\t%v\t%v\n",
          e.literal_index, dev.device_id, dev.comparison_ok ? "true" : "false",
          sanitized_err.empty() ? "-" : sanitized_err, e.shape_str,
          e.element_type, e.element_count, dev.exact_matches,
          dev.exact_match_pct, dev.max_abs_error, dev.max_rel_error,
          dev.mean_rel_error, dev.suggested_abs_error, dev.suggested_rel_error,
          dev.nan_mismatches, dev.inf_mismatches);
    }
  }
  return tsv;
}

std::string ModelComparisonResult::SummaryToString() const {
  int64_t exact_literals = 0;
  int64_t differing_literals = 0;
  int64_t nan_inf_literals = 0;
  int64_t total_failed_devices = 0;
  double worst_abs_err = 0.0;
  double worst_rel_err = 0.0;
  int64_t worst_abs_literal = -1;
  int64_t worst_rel_literal = -1;

  for (const auto& e : entries) {
    total_failed_devices += e.failed_devices;
    if (e.total_nan_mismatches > 0 || e.total_inf_mismatches > 0) {
      ++nan_inf_literals;
    }
    const bool is_exact =
        e.num_devices > 0 && e.failed_devices == 0 &&
        absl::c_all_of(e.per_device_results,
                       [](const LiteralDeviceComparison& d) {
                         return d.comparison_ok && d.mismatches == 0 &&
                                d.nan_mismatches == 0 && d.inf_mismatches == 0;
                       });
    if (is_exact) {
      ++exact_literals;
    } else {
      ++differing_literals;
    }
    if (e.max_abs_error > worst_abs_err) {
      worst_abs_err = e.max_abs_error;
      worst_abs_literal = e.literal_index;
    }
    if (e.max_rel_error > worst_rel_err) {
      worst_rel_err = e.max_rel_error;
      worst_rel_literal = e.literal_index;
    }
  }

  std::string summary;
  absl::StrAppend(&summary, "Model Comparison Summary:\n");
  absl::StrAppend(&summary, "  Golden Dir: ", golden_dir, "\n");
  absl::StrAppend(&summary, "  Test Dir:   ", test_dir, "\n");
  absl::StrAppendFormat(&summary,
                        "  Total Literals: %zu across %zu device(s)\n",
                        entries.size(), devices.size());
  absl::StrAppendFormat(
      &summary, "  Exact Match Literals: %v (%.2f%%)\n", exact_literals,
      entries.empty() ? 0.0 : 100.0 * exact_literals / entries.size());
  absl::StrAppendFormat(
      &summary, "  Differing Literals:   %v (%.2f%%)\n", differing_literals,
      entries.empty() ? 0.0 : 100.0 * differing_literals / entries.size());
  if (total_failed_devices > 0) {
    absl::StrAppendFormat(&summary, "  Failed Device Comparisons:  %v\n",
                          total_failed_devices);
  }
  if (nan_inf_literals > 0) {
    absl::StrAppendFormat(&summary, "  NaN/Inf Mismatch Literals: %v\n",
                          nan_inf_literals);
  }
  if (!missing_in_test.empty()) {
    absl::StrAppendFormat(
        &summary, "  Missing in Test Dir:        %zu literal/device pair(s)\n",
        missing_in_test.size());
  }
  if (!missing_in_golden.empty()) {
    absl::StrAppendFormat(
        &summary, "  Missing in Golden Dir:      %zu literal/device pair(s)\n",
        missing_in_golden.size());
  }
  if (worst_abs_literal >= 0) {
    absl::StrAppendFormat(&summary,
                          "  Worst Absolute Error: %.6e (literal_%v)\n",
                          worst_abs_err, worst_abs_literal);
  }
  if (worst_rel_literal >= 0) {
    absl::StrAppendFormat(&summary,
                          "  Worst Relative Error: %.6e (literal_%v)\n",
                          worst_rel_err, worst_rel_literal);
  }
  return summary;
}

absl::Status WriteModelComparisonOutputs(const ModelComparisonResult& result,
                                         absl::string_view json_path,
                                         absl::string_view tsv_path,
                                         absl::string_view device_tsv_path) {
  tsl::Env* env = tsl::Env::Default();

  auto create_parent_dir = [env](absl::string_view file_path) -> absl::Status {
    if (file_path.empty()) {
      return absl::OkStatus();
    }
    const absl::string_view dir = tsl::io::Dirname(file_path);
    if (!dir.empty() && !env->FileExists(dir).ok()) {
      ABSL_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir))
          << "Failed to create directory: " << dir;
    }
    return absl::OkStatus();
  };

  if (!json_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(json_path));
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, json_path, result.ToJson()))
        << "Failed to write JSON output to: " << json_path;
  }
  if (!tsv_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(tsv_path));
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, tsv_path, result.ToTsv()))
        << "Failed to write TSV output to: " << tsv_path;
  }
  if (!device_tsv_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(device_tsv_path));
    ABSL_RETURN_IF_ERROR(
        tsl::WriteStringToFile(env, device_tsv_path, result.ToDeviceTsv()))
        << "Failed to write Device TSV output to: " << device_tsv_path;
  }
  return absl::OkStatus();
}

}  // namespace xla::compare_literals
