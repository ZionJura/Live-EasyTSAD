"""Local process identities must agree between the Python and Node hosts."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tsad_harness.process_identity import process_identity


class ProcessIdentityTests(unittest.TestCase):
    def test_identity_for_live_and_exited_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            self.assertTrue(process_identity(child.pid))
        finally:
            child.terminate()
            child.wait()
        self.assertIsNone(process_identity(child.pid))

    @unittest.skipUnless(sys.platform == "darwin", "macOS compatibility")
    def test_node_and_python_report_the_same_identity(self):
        root = Path(__file__).resolve().parents[3]
        code = "import {identity} from './EasyTSAD/Harness/dsh-host/src/model-permits.mjs'; console.log(await identity(Number(process.argv[1])))"
        output = subprocess.check_output(["node", "--input-type=module", "-e", code, str(os.getpid())],
                                         cwd=root, text=True)
        self.assertEqual(output.strip(), process_identity(os.getpid()))
