import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from q3 import reference


class ReferenceTests(unittest.TestCase):
    def test_actual_frozen_baselines_are_intact(self):
        self.assertEqual(len(reference.verify_reference()), 18)

    def test_export_copies_baseline_and_rejects_modified_destination(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'reference'
            source.mkdir()
            data = b'fixed outer training coefficients'
            (source / 'fit.json').write_bytes(data)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'retained_files': {'fit.json': hashlib.sha256(data).hexdigest()}}))
            out = root / 'export'
            out.mkdir()
            with patch.object(reference, 'REFERENCE', source), patch.object(reference, 'MANIFEST', manifest):
                # Explicitly pass the patched source when validating defaults.
                with patch.object(reference, 'verify_reference', wraps=reference.verify_reference) as verify:
                    reference.prepare_reference(out)
                    self.assertEqual((out / 'reference/fit.json').read_bytes(), data)
                    (out / 'reference/fit.json').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, '已改变'):
                        reference.prepare_reference(out)
                    self.assertEqual((source / 'fit.json').read_bytes(), data)

    def test_missing_frozen_file_is_an_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, '缺失'):
                reference.verify_reference(Path(tmp))
