# Copyright 2026 The OpenXLA Authors
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
# ==============================================================================
"""End-to-end Python test validating XLA float checker crash dump mode."""

import os
import subprocess
import tempfile
from absl import logging
from absl.testing import absltest

# copybara:comment_begin(oss-only)
def get_run_hlo_binary_path():
  runfiles_dir = (
      os.environ.get('BAZEL_RUNFILES_DIR') or os.environ.get('RUNFILES_DIR')
  )
  if not runfiles_dir:
    raise RuntimeError('Runfiles directory not found in environment.')
  binary_path = os.path.join(runfiles_dir, 'xla/xla/tools/run_hlo_module')
  if not os.path.exists(binary_path):
    raise FileNotFoundError(
        f'Could not find run_hlo_module binary at {binary_path}'
    )
  return binary_path
# copybara:comment_end


class FloatCheckDeviceE2eTest(absltest.TestCase):

  def test_nan_in_dump_mode_should_dump_and_reproduce(self):
    run_hlo_binary = get_run_hlo_binary_path()
    self.assertTrue(os.path.exists(run_hlo_binary))

    hlo_string = """
HloModule test_module

ENTRY main {
  p0 = f32[] parameter(0)
  p0_broadcast = f32[1024] broadcast(p0), dimensions={}
  zero = f32[] constant(0)
  zero_init = f32[1024] broadcast(zero), dimensions={}
  ROOT div = f32[1024] divide(zero_init, p0_broadcast)
}
"""

    inputs_pbtxt = """iterations {
  arguments {
    shape {
      element_type: F32
      layout {}
    }
    f32s: 0.0
  }
}
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
      hlo_path = os.path.join(tmp_dir, 'crashing_module.hlo')
      with open(hlo_path, 'w') as f:
        f.write(hlo_string)

      inputs_path = os.path.join(tmp_dir, 'inputs.pbtxt')
      with open(inputs_path, 'w') as f:
        f.write(inputs_pbtxt)

      # Run run_hlo_module under DETECTION_MODE_DUMP (dump) on GPU (CUDA).
      # We expect it to terminate with a non-zero exit code due to LOG(FATAL).
      run_args = [
          run_hlo_binary,
          '--platform=cuda',
          '--reference_platform=',
          '--input_format=hlo',
          hlo_path,
          f'--input_literals_file={inputs_path}',
          f'--xla_dump_to={tmp_dir}',
          '--xla_gpu_detect_nan=dump',
      ]

      with self.assertRaises(subprocess.CalledProcessError) as ctx:
        subprocess.run(run_args, check=True, capture_output=True)

      stderr_out = ctx.exception.stderr.decode('utf-8')
      logging.info('run_hlo_module stderr: %s', stderr_out)
      self.assertIn('Float check crash dump generated', stderr_out)
      self.assertIn(os.path.join(tmp_dir, 'crash_dump'), stderr_out)

      # Verify that crash dump snapshot artifact was created.
      crash_dump_dir = os.path.join(tmp_dir, 'crash_dump')
      self.assertTrue(os.path.isdir(crash_dump_dir))

      snapshots = [
          f for f in os.listdir(crash_dump_dir) if f.endswith('.snapshot.pb')
      ]
      self.assertLen(snapshots, 1)

      snapshot_path = os.path.join(crash_dump_dir, snapshots[0])
      self.assertGreater(os.path.getsize(snapshot_path), 0)

      # Verify the crash dump artifacts can be used to reproduce the crash.
      reproduce_args = [
          run_hlo_binary,
          '--platform=cuda',
          '--reference_platform=',
          '--input_format=pb',
          snapshot_path,
          f'--xla_dump_to={tmp_dir}',
          '--xla_gpu_detect_nan=fail',
      ]

      with self.assertRaises(subprocess.CalledProcessError) as ctx_repr:
        subprocess.run(reproduce_args, check=True, capture_output=True)

      repr_stderr = ctx_repr.exception.stderr.decode('utf-8')
      self.assertIn('Float check failed, aborting.', repr_stderr)


if __name__ == '__main__':
  absltest.main()
