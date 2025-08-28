# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# limitations under the License..
# ==============================================================================
"""Modify setup.py with TensorFlow and NVIDIA wheel versions."""

import argparse
import pathlib
import re


def _update_setup_with_tf_and_nvidia_wheel_versions(
    template_path: pathlib.Path,
    output_path: pathlib.Path,
    nvidia_wheel_versions_data: str,
    tf_version: str,
):
  """Updates a setup.py template with TensorFlow and NVIDIA wheel versions.

  This function reads a setup.py template file, replaces placeholder versions
  for TensorFlow and various NVIDIA-related wheels based on the provided
  data, and writes the result to an output file.

  Args:
    template_path: Path to the input setup.py.tpl template file.
    output_path: Path where the modified setup.py file will be written.
    nvidia_wheel_versions_data: A string containing NVIDIA wheel version data,
      with each line in the format "wheel_name version_spec".
    tf_version: The version string for the TensorFlow package.
  """
  nvidia_wheel_versions = {}
  # Regex to capture wheel name and its version constraint
  pattern = re.compile(r"^([a-z0-9_-]+)\s+(.*)$")

  for line in enumerate(nvidia_wheel_versions_data.splitlines()):
    match = pattern.match(line)
    if match:
      wheel_name = match.group(1).replace("-", "_")
      version_spec = match.group(2).strip()

      key = wheel_name.replace("_cu12", "") + "_version"
      nvidia_wheel_versions[key] = version_spec

  with open(template_path) as f:
    content = f.read()

  content = content.replace("_VERSION = '0.0.0'", f"_VERSION = '{tf_version}'")

  for version_name, version_value in nvidia_wheel_versions.items():
    content = content.replace(
        f"{version_name} = ''  # placeholder",
        f"{version_name} = '{version_value}'",
    )

  with open(output_path, "w") as f:
    f.write(content)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--template_file",
      type=pathlib.Path,
      required=True,
      help="Path to the setup.py.tpl template file",
  )
  parser.add_argument(
      "--output_file",
      type=pathlib.Path,
      required=True,
      help="Path to write the generated setup.py file",
  )
  parser.add_argument(
      "--nvidia_wheel_versions_data",
      default=None,
      required=True,
      help="NVIDIA wheel versions data",
  )
  parser.add_argument(
      "--tf_version",
      type=str,
      required=True,
      help="The TensorFlow package version string",
  )
  args = parser.parse_args()

  _update_setup_with_tf_and_nvidia_wheel_versions(
      args.template_file,
      args.output_file,
      args.nvidia_wheel_versions_file,
      args.tf_version,
  )
