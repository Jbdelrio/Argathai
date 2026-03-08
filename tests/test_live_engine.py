import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.engine import LiveEngine


class TestLiveEngine(unittest.TestCase):
    def test_individual_strategy_start_stop(self):
        eng = LiveEngine()
        if not eng.strategies:
            self.skipTest('No strategies loaded')

        first = next(iter(eng.strategies.keys()))
        eng.connected = True  # simulate connected state for unit test

        self.assertTrue(eng.start_strategy(first))
        self.assertTrue(eng.strategy_runtime[first]['active'])

        eng.update_runtime()
        self.assertGreaterEqual(eng.strategy_runtime[first]['buffered_sec'], 0)

        self.assertTrue(eng.stop_strategy(first))
        self.assertFalse(eng.strategy_runtime[first]['active'])


if __name__ == '__main__':
    unittest.main()