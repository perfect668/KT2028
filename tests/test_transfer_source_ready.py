import tempfile
from pathlib import Path
import unittest
from kt2028.tcga_transfer import source_ready


class TransferReadinessTests(unittest.TestCase):
    def test_full_length_but_aria2_incomplete_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.czi'; p.write_bytes(b'complete-length')
            state=Path(str(p)+'.aria2'); state.write_bytes(b'not complete')
            self.assertFalse(source_ready(p,p.stat().st_size))
            state.unlink()
            self.assertTrue(source_ready(p,p.stat().st_size))

    def test_wrong_size_or_missing_file_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.czi'
            self.assertFalse(source_ready(p,10))
            p.write_bytes(b'short')
            self.assertFalse(source_ready(p,10))


if __name__=='__main__':unittest.main()
