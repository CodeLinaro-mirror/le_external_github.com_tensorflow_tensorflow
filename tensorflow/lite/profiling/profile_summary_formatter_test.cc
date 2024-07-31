/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/profile_summary_formatter.h"

#include <fstream>
#include <ios>
#include <map>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/core/util/stat_summarizer_options.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/profiling/proto/profiling_info.pb.h"

namespace tflite {
namespace profiling {

namespace {

bool CompareOpProfilingStat(const OpProfilingStat& op_profiling_stat_1,
                            const OpProfilingStat& op_profiling_stat_2) {
  if (op_profiling_stat_1.first() != op_profiling_stat_2.first()) {
    return false;
  }
  if (op_profiling_stat_1.last() != op_profiling_stat_2.last()) {
    return false;
  }
  if (op_profiling_stat_1.avg() != op_profiling_stat_2.avg()) {
    return false;
  }
  if (op_profiling_stat_1.stddev() != op_profiling_stat_2.stddev()) {
    return false;
  }
  if (op_profiling_stat_1.variance() != op_profiling_stat_2.variance()) {
    return false;
  }
  if (op_profiling_stat_1.min() != op_profiling_stat_2.min()) {
    return false;
  }
  if (op_profiling_stat_1.max() != op_profiling_stat_2.max()) {
    return false;
  }
  if (op_profiling_stat_1.sum() != op_profiling_stat_2.sum()) {
    return false;
  }
  if (op_profiling_stat_1.count() != op_profiling_stat_2.count()) {
    return false;
  }
  return true;
}

bool CompareOpProfileData(const OpProfileData& op_profile_data_1,
                          const OpProfileData& op_profile_data_2) {
  if (op_profile_data_1.node_type() != op_profile_data_2.node_type()) {
    return false;
  }
  if (op_profile_data_1.times_called() != op_profile_data_2.times_called()) {
    return false;
  }
  if (op_profile_data_1.name() != op_profile_data_2.name()) {
    return false;
  }
  if (op_profile_data_1.run_order() != op_profile_data_2.run_order()) {
    return false;
  }
  if (!CompareOpProfilingStat(op_profile_data_1.inference_microseconds(),
                              op_profile_data_2.inference_microseconds())) {
    return false;
  }
  if (!CompareOpProfilingStat(op_profile_data_1.mem_kb(),
                              op_profile_data_2.mem_kb())) {
    return false;
  }
  return true;
}

bool CompareSubGraphProfilingData(
    const SubGraphProfilingData& subgraph_profiling_data_1,
    const SubGraphProfilingData& subgraph_profiling_data_2) {
  if (subgraph_profiling_data_1.subgraph_name() !=
      subgraph_profiling_data_2.subgraph_name()) {
    return false;
  }
  if (subgraph_profiling_data_1.per_op_profiles().size() !=
      subgraph_profiling_data_2.per_op_profiles().size()) {
    return false;
  }
  for (int i = 0; i < subgraph_profiling_data_1.per_op_profiles().size(); ++i) {
    if (!CompareOpProfileData(subgraph_profiling_data_1.per_op_profiles(i),
                              subgraph_profiling_data_2.per_op_profiles(i))) {
      return false;
    }
  }
  return true;
}

bool CompareDelegateProfilingData(
    const DelegateProfilingData& delegate_profiling_data_1,
    const DelegateProfilingData& delegate_profiling_data_2) {
  if (delegate_profiling_data_1.delegate_name() !=
      delegate_profiling_data_2.delegate_name()) {
    return false;
  }
  if (delegate_profiling_data_1.per_op_profiles().size() !=
      delegate_profiling_data_2.per_op_profiles().size()) {
    return false;
  }
  for (int i = 0; i < delegate_profiling_data_1.per_op_profiles().size(); ++i) {
    if (!CompareOpProfileData(delegate_profiling_data_1.per_op_profiles(i),
                              delegate_profiling_data_2.per_op_profiles(i))) {
      return false;
    }
  }
  return true;
}

bool CompareModelProfilingData(
    const ModelProfilingData& model_profiling_data_1,
    const ModelProfilingData& model_profiling_data_2) {
  if (model_profiling_data_1.subgraph_profiles().size() !=
      model_profiling_data_2.subgraph_profiles().size()) {
    return false;
  }
  for (int i = 0; i < model_profiling_data_1.subgraph_profiles().size(); ++i) {
    if (!CompareSubGraphProfilingData(
            model_profiling_data_1.subgraph_profiles(i),
            model_profiling_data_2.subgraph_profiles(i))) {
      return false;
    }
  }
  if (model_profiling_data_1.delegate_profiles().size() !=
      model_profiling_data_2.delegate_profiles().size()) {
    return false;
  }
  for (int i = 0; i < model_profiling_data_1.delegate_profiles().size(); ++i) {
    if (!CompareDelegateProfilingData(
            model_profiling_data_1.delegate_profiles(i),
            model_profiling_data_2.delegate_profiles(i))) {
      return false;
    }
  }
  return true;
}

TEST(SummaryWriterTest, SummaryOptionStdOut) {
  ProfileSummaryDefaultFormatter writer;
  tensorflow::StatSummarizerOptions options = writer.GetStatSummarizerOptions();
  EXPECT_EQ(options.show_summary, false);
  EXPECT_EQ(options.show_memory, false);
  EXPECT_EQ(options.format_as_csv, false);
}

TEST(SummaryWriterTest, SummaryOptionCSV) {
  ProfileSummaryCSVFormatter writer;
  tensorflow::StatSummarizerOptions options = writer.GetStatSummarizerOptions();
  EXPECT_EQ(options.show_summary, false);
  EXPECT_EQ(options.show_memory, false);
  EXPECT_EQ(options.format_as_csv, true);
}
TEST(SummaryWriterTest, EmptyOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::string output = writer.GetOutputString(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()), {});
  EXPECT_EQ(output.size(), 0);
}

TEST(SummaryWriterTest, EmptyShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::string output = writer.GetShortSummary(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()), {});
  EXPECT_EQ(output.size(), 0);
}

TEST(SummaryWriterTest, SingleSubgraphOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetOutputString(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()), {});
  ASSERT_TRUE(absl::StrContains(output, "Run Order"));
  ASSERT_TRUE(absl::StrContains(output, "Top by Computation Time"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Memory Use"));
  ASSERT_TRUE(absl::StrContains(output, "Summary by node type"));
  ASSERT_TRUE(absl::StrContains(output, "nodes observed"));
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, SingleSubgraphShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetShortSummary(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()),
      {{0, "Primary graph"}});
  ASSERT_TRUE(!absl::StrContains(output, "Run Order"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Computation Time"));
  ASSERT_TRUE(!absl::StrContains(output, "Top by Memory Use"));
  ASSERT_TRUE(!absl::StrContains(output, "Summary by node type"));
  ASSERT_TRUE(absl::StrContains(output, "nodes observed"));
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, MultiSubgraphOutputString) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  stats_calculator_map[1] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetOutputString(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()),
      {{0, "Primary graph"}, {1, "Subgraph 1"}});
  ASSERT_TRUE(absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, MultiSubgraphOutputStringForProto) {
  ProfileSummaryProtoFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string kernel_name_1 = "Kernel 1";
  std::string kernel_name_2 = "Kernel 2";
  std::string kernel_name_3 = "Kernel 3";

  std::string op_name_1 = "Convolution";
  std::string op_name_2 = "Reshape";
  std::string op_name_3 = "Convolution";
  stats_calculator_map[0]->AddNodeStats(kernel_name_1, op_name_1, 1, 10, 10000);
  stats_calculator_map[0]->AddNodeStats(kernel_name_1, op_name_1, 1, 20, 20000);
  stats_calculator_map[0]->AddNodeStats(kernel_name_2, op_name_2, 2, 15, 10000);
  stats_calculator_map[0]->UpdateRunTotalUs(25);
  stats_calculator_map[1] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  stats_calculator_map[1]->AddNodeStats(kernel_name_3, op_name_3, 3, 10, 10000);
  stats_calculator_map[1]->UpdateRunTotalUs(10);

  std::string output = writer.GetOutputString(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()),
      {{0, "Primary graph"}, {1, "Subgraph 1"}});
  ModelProfilingData model_profiling_data;
  model_profiling_data.ParseFromString(output);
  ASSERT_TRUE(absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
  ASSERT_EQ(model_profiling_data.subgraph_profiles().size(), 2);
  ASSERT_EQ(model_profiling_data.subgraph_profiles(0).subgraph_name(),
            "Primary graph");
  ASSERT_EQ(model_profiling_data.subgraph_profiles(0).per_op_profiles().size(),
            2);

  OpProfileData op_profile_data_1;
  op_profile_data_1.set_node_type(op_name_1);
  OpProfilingStat* inference_microseconds_stat_1 =
      op_profile_data_1.mutable_inference_microseconds();
  inference_microseconds_stat_1->set_first(10);
  inference_microseconds_stat_1->set_last(20);
  inference_microseconds_stat_1->set_max(20);
  inference_microseconds_stat_1->set_min(10);
  inference_microseconds_stat_1->set_avg(15);
  inference_microseconds_stat_1->set_stddev(5);
  inference_microseconds_stat_1->set_variance(25);
  inference_microseconds_stat_1->set_sum(30);
  inference_microseconds_stat_1->set_count(2);
  OpProfilingStat* memory_stat_1 = op_profile_data_1.mutable_mem_kb();
  memory_stat_1->set_first(10);
  memory_stat_1->set_last(20);
  memory_stat_1->set_max(20);
  memory_stat_1->set_min(10);
  memory_stat_1->set_avg(15);
  memory_stat_1->set_stddev(5);
  memory_stat_1->set_variance(25);
  memory_stat_1->set_sum(30);
  memory_stat_1->set_count(2);
  op_profile_data_1.set_name(kernel_name_1);
  op_profile_data_1.set_run_order(1);
  op_profile_data_1.set_times_called(2);
  EXPECT_TRUE(CompareOpProfileData(
      model_profiling_data.subgraph_profiles(0).per_op_profiles(0),
      op_profile_data_1));

  OpProfileData op_profile_data_2;
  op_profile_data_2.set_node_type(op_name_2);
  OpProfilingStat* inference_microseconds_stat_2 =
      op_profile_data_2.mutable_inference_microseconds();
  inference_microseconds_stat_2->set_first(15);
  inference_microseconds_stat_2->set_last(15);
  inference_microseconds_stat_2->set_max(15);
  inference_microseconds_stat_2->set_min(15);
  inference_microseconds_stat_2->set_avg(15);
  inference_microseconds_stat_2->set_stddev(0);
  inference_microseconds_stat_2->set_variance(0);
  inference_microseconds_stat_2->set_sum(15);
  inference_microseconds_stat_2->set_count(1);
  OpProfilingStat* memory_stat_2 = op_profile_data_2.mutable_mem_kb();
  memory_stat_2->set_first(10);
  memory_stat_2->set_last(10);
  memory_stat_2->set_max(10);
  memory_stat_2->set_min(10);
  memory_stat_2->set_avg(10);
  memory_stat_2->set_stddev(0);
  memory_stat_2->set_variance(0);
  memory_stat_2->set_sum(10);
  memory_stat_2->set_count(1);
  op_profile_data_2.set_times_called(1);
  op_profile_data_2.set_name(kernel_name_2);
  op_profile_data_2.set_run_order(2);

  EXPECT_TRUE(CompareOpProfileData(
      model_profiling_data.subgraph_profiles(0).per_op_profiles(1),
      op_profile_data_2));

  ASSERT_EQ(model_profiling_data.subgraph_profiles(1).subgraph_name(),
            "Subgraph 1");
  ASSERT_EQ(model_profiling_data.subgraph_profiles(1).per_op_profiles().size(),
            1);

  OpProfileData op_profile_data_3;
  op_profile_data_3.set_node_type(op_name_3);
  OpProfilingStat* inference_microseconds_stat_3 =
      op_profile_data_3.mutable_inference_microseconds();
  inference_microseconds_stat_3->set_first(10);
  inference_microseconds_stat_3->set_last(10);
  inference_microseconds_stat_3->set_max(10);
  inference_microseconds_stat_3->set_min(10);
  inference_microseconds_stat_3->set_avg(10);
  inference_microseconds_stat_3->set_stddev(0);
  inference_microseconds_stat_3->set_variance(0);
  inference_microseconds_stat_3->set_sum(10);
  inference_microseconds_stat_3->set_count(1);
  OpProfilingStat* memory_stat_3 = op_profile_data_3.mutable_mem_kb();
  memory_stat_3->set_first(10);
  memory_stat_3->set_last(10);
  memory_stat_3->set_max(10);
  memory_stat_3->set_min(10);
  memory_stat_3->set_avg(10);
  memory_stat_3->set_stddev(0);
  memory_stat_3->set_variance(0);
  memory_stat_3->set_sum(10);
  memory_stat_3->set_count(1);
  op_profile_data_3.set_times_called(1);
  op_profile_data_3.set_name(kernel_name_3);
  op_profile_data_3.set_run_order(3);
  EXPECT_TRUE(CompareOpProfileData(
      model_profiling_data.subgraph_profiles(1).per_op_profiles(0),
      op_profile_data_3));
}

TEST(SummaryWriterTest, MultiSubgraphHandleOutputForProto) {
  ProfileSummaryProtoFormatter writer;

  ModelProfilingData model_profiling_data_run;
  SubGraphProfilingData* subgraph_profiling_data =
      model_profiling_data_run.add_subgraph_profiles();
  subgraph_profiling_data->set_subgraph_name("Primary graph");
  OpProfileData* op_profile_data_1 =
      subgraph_profiling_data->add_per_op_profiles();
  op_profile_data_1->set_node_type("Convolution");
  OpProfilingStat* inference_stat_1 =
      op_profile_data_1->mutable_inference_microseconds();
  inference_stat_1->set_first(10);
  inference_stat_1->set_avg(10);
  OpProfilingStat* mem_stat_1 = op_profile_data_1->mutable_mem_kb();
  mem_stat_1->set_first(10);
  mem_stat_1->set_avg(10);
  op_profile_data_1->set_times_called(1);
  op_profile_data_1->set_name("Kernel 1");
  op_profile_data_1->set_run_order(1);
  OpProfileData* op_profile_data_2 =
      subgraph_profiling_data->add_per_op_profiles();
  op_profile_data_2->set_node_type("Reshape");
  OpProfilingStat* inference_stat_2 =
      op_profile_data_2->mutable_inference_microseconds();
  inference_stat_2->set_first(15);
  inference_stat_2->set_avg(15);
  OpProfilingStat* mem_stat_2 = op_profile_data_2->mutable_mem_kb();
  mem_stat_2->set_first(10);
  mem_stat_2->set_avg(10);
  op_profile_data_2->set_times_called(1);
  op_profile_data_2->set_name("Kernel 2");
  op_profile_data_2->set_run_order(2);
  SubGraphProfilingData* subgraph_profiling_data_1 =
      model_profiling_data_run.add_subgraph_profiles();
  subgraph_profiling_data_1->set_subgraph_name("Subgraph 1");
  OpProfileData* op_profile_data_3 =
      subgraph_profiling_data_1->add_per_op_profiles();
  op_profile_data_3->set_node_type("Convolution");
  OpProfilingStat* inference_stat_3 =
      op_profile_data_3->mutable_inference_microseconds();
  inference_stat_3->set_first(10);
  inference_stat_3->set_avg(10);
  OpProfilingStat* mem_stat_3 = op_profile_data_3->mutable_mem_kb();
  mem_stat_3->set_first(10);
  mem_stat_3->set_avg(10);
  op_profile_data_3->set_times_called(1);
  op_profile_data_3->set_name("Kernel 3");
  op_profile_data_3->set_run_order(3);
  DelegateProfilingData* delegate_profiling_data =
      model_profiling_data_run.add_delegate_profiles();
  OpProfileData* op_profile_data_4 =
      delegate_profiling_data->add_per_op_profiles();
  op_profile_data_4->set_node_type("Convolution");
  OpProfilingStat* inference_stat_4 =
      op_profile_data_4->mutable_inference_microseconds();
  inference_stat_4->set_first(10);
  inference_stat_4->set_avg(10);
  OpProfilingStat* mem_stat_4 = op_profile_data_4->mutable_mem_kb();
  mem_stat_4->set_first(10);
  mem_stat_4->set_avg(10);
  op_profile_data_4->set_times_called(1);
  op_profile_data_4->set_name("Kernel 4");
  op_profile_data_4->set_run_order(4);

  ModelProfilingData model_profiling_data_init;
  SubGraphProfilingData* subgraph_profiling_data_init =
      model_profiling_data_init.add_subgraph_profiles();
  subgraph_profiling_data_init->set_subgraph_name("Primary graph");
  OpProfileData* op_profile_data_init_1 =
      subgraph_profiling_data_init->add_per_op_profiles();
  op_profile_data_init_1->set_node_type("Convolution");
  OpProfilingStat* inference_stat_init_1 =
      op_profile_data_init_1->mutable_inference_microseconds();
  inference_stat_init_1->set_first(10);
  inference_stat_init_1->set_avg(10);
  op_profile_data_init_1->set_times_called(1);
  OpProfilingStat* mem_stat_init_1 = op_profile_data_init_1->mutable_mem_kb();
  mem_stat_init_1->set_first(10);
  mem_stat_init_1->set_avg(10);
  op_profile_data_init_1->set_name("ModifyGraphWithDelegate");
  op_profile_data_init_1->set_run_order(1);

#ifdef __ANDROID__
  std::string file_name = "/data/local/tmp/test_file.proto";
#else
  std::string file_name = "/tmp/test_file.proto";
#endif

  writer.HandleOutput(model_profiling_data_init.SerializeAsString(),
                      model_profiling_data_run.SerializeAsString(), file_name);

  std::ifstream file(file_name, std::ios::binary);

  ASSERT_TRUE(file.good());

  BenchmarkProfilingData benchmark_profiling_data;
  benchmark_profiling_data.ParseFromIstream(&file);
  file.close();

  ASSERT_TRUE(benchmark_profiling_data.model_name().empty());
  EXPECT_TRUE(CompareModelProfilingData(benchmark_profiling_data.init_profile(),
                                        model_profiling_data_init));
  EXPECT_TRUE(CompareModelProfilingData(
      benchmark_profiling_data.runtime_profile(), model_profiling_data_run));
}

TEST(SummaryWriterTest, MultiSubgraphShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>
      stats_calculator_map;
  stats_calculator_map[0] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  stats_calculator_map[1] = std::make_unique<tensorflow::StatsCalculator>(
      writer.GetStatSummarizerOptions());
  std::string output = writer.GetShortSummary(
      stats_calculator_map,
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions()),
      {{0, "Primary graph"}, {1, "Subgraph 1"}});
  ASSERT_TRUE(absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(!absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, DelegationOutputString) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetOutputString(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator, {});
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(absl::StrContains(output, "Delegate internal"));
}

TEST(SummaryWriterTest, DelegationShortSummary) {
  ProfileSummaryDefaultFormatter writer;
  auto delegate_stats_calculator =
      tensorflow::StatsCalculator(writer.GetStatSummarizerOptions());
  delegate_stats_calculator.UpdateRunTotalUs(1);
  std::string output = writer.GetShortSummary(
      std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>(),
      delegate_stats_calculator, {});
  ASSERT_TRUE(!absl::StrContains(output, "Primary graph"));
  ASSERT_TRUE(!absl::StrContains(output, "Subgraph"));
  ASSERT_TRUE(absl::StrContains(output, "Delegate internal"));
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
