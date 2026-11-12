import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))      # helpers of the main test-suite (tiny models)
sys.path.insert(0, ROOT)                             # recurrent_armt, carmt
