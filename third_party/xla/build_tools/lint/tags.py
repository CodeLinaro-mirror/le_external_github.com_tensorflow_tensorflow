# Copyright 2024 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Asserts all tags in XLA are documented.

`bazel query //xla/... --output=build` is read from stdin, and then we check
all tags are present in the `_TAGS_TO_DOCUMENTATION_MAP`. Ideally we would parse
using
https://github.com/bazelbuild/bazel/blob/master/src/main/protobuf/build.proto
but this is not possible due to XLA's old protobuf version. So we parse by hand
instead.
"""
import logging
import sys
from typing import Set

_TAGS_TO_DOCUMENTATION_MAP = {}


def get_tags_from_line(line: str) -> Set[str]:
  if line.strip().startswith("tags = "):
    tags_list = line[10:-3]
    if tags_list.strip():
      return {tag[1:-1] for tag in tags_list.split(", ")}

  return set()


def main():
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)

  tags = set.union(*(get_tags_from_line(line) for line in sys.stdin))

  logging.info(str(tags))

  for tag in tags:
    if tag not in _TAGS_TO_DOCUMENTATION_MAP:
      raise ValueError(
          f'Tag "{tag}" is undocumented! Please add it to'
          " build_tools/lint/tags.py."
      )

  unused_but_documented_tags = _TAGS_TO_DOCUMENTATION_MAP.keys() - tags

  if unused_but_documented_tags:
    logging.info(
        "The following tags are documented but unused. Do we expect they'll be"
        " used in the future?"
    )
    for tag in unused_but_documented_tags:
      logging.info(tag)


if __name__ == "__main__":
  main()
