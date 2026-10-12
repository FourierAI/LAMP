"""Small CPU tests for numerical-environment and resume receipts."""
from pathlib import Path
import copy
import json
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_checks import execution_environment, record_runtime


class RuntimeChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix='lamp_runtime_regression_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'runtime.json'
        self.runtime = dict(schema_version=2, environment=execution_environment('cpu'),
                            source_sha256={'main.py': 'a'}, config_sha256={'grid': 'b'}, bank_sha256='c',
                            batch_size=128, workers=0, dense_limit=12000, lu_tile=2048,
                            invocation={'stage': 'select', 'method': 'lamp'})

    def test_decoder_and_binary_are_fingerprinted(self):
        decoder = self.runtime['environment']['decoder']
        self.assertEqual(len(decoder['imaging_sha256']), 64)
        for key in ('jpeg', 'libjpeg_turbo', 'libjpeg_turbo_version'):
            self.assertIn(key, decoder)

    def test_stage_and_method_switches_are_recorded(self):
        record_runtime(self.path, self.runtime)
        later = copy.deepcopy(self.runtime)
        later['invocation'] = {'stage': 'adapt', 'method': 'both'}
        later['workers'] = 4
        record_runtime(self.path, later)
        record = json.loads(self.path.read_text())
        self.assertEqual(len(record['runs']), 2)
        self.assertEqual(record['runs'][0]['invocation'], self.runtime['invocation'])
        self.assertEqual(record['runs'][1]['invocation'], later['invocation'])
        self.assertEqual(record['invocation'], later['invocation'])

    def test_environment_mismatch_does_not_overwrite_receipt(self):
        record_runtime(self.path, self.runtime)
        before = self.path.read_bytes()
        for key in ('python', 'torch', 'torchvision', 'numpy', 'pillow', 'cuda_build', 'cudnn',
                    'device_name', 'matmul_tf32', 'cudnn_deterministic'):
            with self.subTest(field=key):
                altered = copy.deepcopy(self.runtime)
                altered['environment'][key] = 'changed'
                with self.assertRaisesRegex(RuntimeError, key):
                    record_runtime(self.path, altered)
                self.assertEqual(self.path.read_bytes(), before)
        for key in ('jpeg', 'libjpeg_turbo_version', 'imaging_sha256'):
            with self.subTest(decoder=key):
                altered = copy.deepcopy(self.runtime)
                altered['environment']['decoder'][key] = 'changed'
                with self.assertRaisesRegex(RuntimeError, 'decoder'):
                    record_runtime(self.path, altered)
                self.assertEqual(self.path.read_bytes(), before)

    def test_changed_inputs_and_solver_settings_are_rejected(self):
        record_runtime(self.path, self.runtime)
        for key in ('source_sha256', 'config_sha256', 'bank_sha256', 'batch_size', 'dense_limit', 'lu_tile'):
            with self.subTest(field=key):
                altered = copy.deepcopy(self.runtime)
                altered[key] = 'changed'
                with self.assertRaisesRegex(RuntimeError, key):
                    record_runtime(self.path, altered)

    def test_unverifiable_old_outputs_are_rejected(self):
        self.path.write_text(json.dumps({'pillow': '11.1.0'}))
        with self.assertRaisesRegex(RuntimeError, 'legacy runtime record'):
            record_runtime(self.path, self.runtime)
        self.path.unlink()
        (self.root / 'model.pt').write_bytes(b'cached model')
        with self.assertRaisesRegex(RuntimeError, 'no runtime record'):
            record_runtime(self.path, self.runtime)


if __name__ == '__main__':
    unittest.main()
