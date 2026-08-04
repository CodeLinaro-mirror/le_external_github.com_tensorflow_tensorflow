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

#include "xla/util/split_proto/proto_field_size_utils.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message_lite.h"
#include "google/protobuf/wire_format.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

constexpr absl::string_view kSeparator =
    "========================================"
    "========================================\n";
constexpr size_t kOneMib = 1024 * 1024;
constexpr int kMaxDrillDownDepth = 3;
constexpr int kMaxSubfieldsPerField = 5;

struct SubfieldEntry {
  std::string name;
  const tsl::protobuf::FieldDescriptor* field = nullptr;
  size_t byte_size = 0;
};

struct RootFieldInfo {
  const tsl::protobuf::FieldDescriptor* descriptor = nullptr;
  size_t byte_size = 0;
  std::vector<SubfieldEntry> subfields;
};

void CollectSubfields(const tsl::protobuf::Message& msg,
                      absl::string_view prefix, int depth, size_t threshold,
                      std::vector<SubfieldEntry>& subfields) {
  const tsl::protobuf::Reflection* reflection = msg.GetReflection();
  if (reflection == nullptr) {
    return;
  }

  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(msg, &fields);

  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    size_t field_size =
        tsl::protobuf::internal::WireFormat::FieldByteSize(field, msg);
    if (field_size < threshold) {
      continue;
    }

    std::string full_name;
    if (prefix.empty()) {
      full_name = field->name();
    } else {
      full_name = absl::StrCat(prefix, ".", field->name());
    }

    if (field->cpp_type() == tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE &&
        depth < kMaxDrillDownDepth) {
      size_t prev_size = subfields.size();
      if (!field->is_repeated()) {
        const tsl::protobuf::Message& child_msg =
            reflection->GetMessage(msg, field);
        CollectSubfields(child_msg, full_name, depth + 1, threshold, subfields);
      } else {
        int field_count = reflection->FieldSize(msg, field);
        for (int i = 0; i < field_count; ++i) {
          const tsl::protobuf::Message& child_msg =
              reflection->GetRepeatedMessage(msg, field, i);
          CollectSubfields(child_msg, full_name, depth + 1, threshold,
                           subfields);
        }
      }
      if (subfields.size() > prev_size) {
        continue;
      }
    }

    auto it = absl::c_find_if(
        subfields, [&](const SubfieldEntry& e) { return e.name == full_name; });
    if (it != subfields.end()) {
      it->byte_size += field_size;
    } else {
      SubfieldEntry entry;
      entry.name = full_name;
      entry.field = field;
      entry.byte_size = field_size;
      subfields.push_back(std::move(entry));
    }
  }
}

}  // namespace

std::string GetTopKProtoFieldSizes(const tsl::protobuf::MessageLite& message,
                                   int top_k) {
  const auto* full_msg =
      tsl::protobuf::DynamicCastMessage<tsl::protobuf::Message>(&message);
  if (full_msg == nullptr) {
    const size_t total_bytes = message.ByteSizeLong();
    return absl::StrCat(
        kSeparator,
        absl::StrFormat(
            "Proto [%s] (Reflection not supported for Lite proto)\n",
            message.GetTypeName()),
        absl::StrFormat("Total ByteSize: %.2f MiB (%zu bytes)\n",
                        static_cast<double>(total_bytes) / (1024.0 * 1024.0),
                        total_bytes),
        kSeparator);
  }

  size_t total_byte_size = full_msg->ByteSizeLong();
  const tsl::protobuf::Reflection* reflection = full_msg->GetReflection();
  if (reflection == nullptr) {
    return absl::StrCat(
        kSeparator,
        absl::StrFormat("Proto [%s] (Reflection unavailable)\n",
                        message.GetTypeName()),
        absl::StrFormat(
            "Total ByteSize: %.2f MiB (%zu bytes)\n",
            static_cast<double>(total_byte_size) / (1024.0 * 1024.0),
            total_byte_size),
        kSeparator);
  }

  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(*full_msg, &fields);

  std::vector<RootFieldInfo> root_fields;
  root_fields.reserve(fields.size());

  size_t drill_down_threshold =
      std::max(static_cast<size_t>(0.01 * total_byte_size), kOneMib);

  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    size_t field_size =
        tsl::protobuf::internal::WireFormat::FieldByteSize(field, *full_msg);
    RootFieldInfo info;
    info.descriptor = field;
    info.byte_size = field_size;

    if (field->cpp_type() == tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE &&
        field_size >= drill_down_threshold) {
      if (!field->is_repeated()) {
        const tsl::protobuf::Message& sub_msg =
            reflection->GetMessage(*full_msg, field);
        CollectSubfields(sub_msg, /*prefix=*/"", /*depth=*/1,
                         drill_down_threshold, info.subfields);
      } else {
        int field_count = reflection->FieldSize(*full_msg, field);
        for (int i = 0; i < field_count; ++i) {
          const tsl::protobuf::Message& sub_msg =
              reflection->GetRepeatedMessage(*full_msg, field, i);
          CollectSubfields(sub_msg, /*prefix=*/"", /*depth=*/1,
                           drill_down_threshold, info.subfields);
        }
      }
      absl::c_stable_sort(info.subfields,
                          [](const SubfieldEntry& a, const SubfieldEntry& b) {
                            if (a.byte_size != b.byte_size) {
                              return a.byte_size > b.byte_size;
                            }
                            return a.field->number() < b.field->number();
                          });
    }

    root_fields.push_back(std::move(info));
  }

  int num_to_display =
      std::min(static_cast<int>(root_fields.size()), std::max(0, top_k));

  absl::c_partial_sort(root_fields, root_fields.begin() + num_to_display,
                       [](const RootFieldInfo& a, const RootFieldInfo& b) {
                         if (a.byte_size != b.byte_size) {
                           return a.byte_size > b.byte_size;
                         }
                         return a.descriptor->number() < b.descriptor->number();
                       });

  std::string output;
  absl::StrAppend(&output, kSeparator);
  absl::StrAppendFormat(&output, "Top %d largest fields in proto [%s]\n",
                        num_to_display, full_msg->GetTypeName());
  absl::StrAppendFormat(
      &output, "Total ByteSize: %.2f MiB (%zu bytes)\n",
      static_cast<double>(total_byte_size) / (1024.0 * 1024.0),
      total_byte_size);
  absl::StrAppend(&output, kSeparator);

  for (int i = 0; i < num_to_display; ++i) {
    const auto& root_field = root_fields[i];
    double mib = static_cast<double>(root_field.byte_size) / (1024.0 * 1024.0);
    double pct = total_byte_size > 0
                     ? (static_cast<double>(root_field.byte_size) /
                        static_cast<double>(total_byte_size)) *
                           100.0
                     : 0.0;
    absl::StrAppendFormat(
        &output, "  %d. %s (tag %d, type %s): %.2f MiB (%.2f%%)\n", i + 1,
        root_field.descriptor->name(), root_field.descriptor->number(),
        absl::AsciiStrToUpper(root_field.descriptor->type_name()), mib, pct);

    int sub_count = 0;
    for (const auto& sub : root_field.subfields) {
      double sub_mib = static_cast<double>(sub.byte_size) / (1024.0 * 1024.0);
      absl::StrAppendFormat(&output, "     -> %s (tag %d, type %s): %.2f MiB\n",
                            sub.name, sub.field->number(),
                            absl::AsciiStrToUpper(sub.field->type_name()),
                            sub_mib);
      if (++sub_count >= kMaxSubfieldsPerField) {
        break;
      }
    }
  }

  absl::StrAppend(&output, kSeparator);
  return output;
}

absl::Status AnnotateResourceExhaustedError(
    const absl::Status& status, const tsl::protobuf::MessageLite& record,
    int top_k) {
  if (!absl::IsResourceExhausted(status)) {
    return status;
  }
  std::string annotation = GetTopKProtoFieldSizes(record, top_k);
  absl::Status new_status(
      status.code(), status.message().empty()
                         ? annotation
                         : absl::StrCat(status.message(), "\n\n", annotation));
  status.ForEachPayload(
      [&](absl::string_view type_url, const absl::Cord& payload) {
        new_status.SetPayload(type_url, payload);
      });
  return new_status;
}

}  // namespace xla
