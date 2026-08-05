from __future__ import annotations

import unittest

from scripts.evaluate_p0a14_math import numeric_vote, stable_seed


class P0A14RuntimeTest(unittest.TestCase):
    def test_numeric_majority(self) -> None:
        self.assertEqual(numeric_vote(["4", "5", "4"]), ("4", 0, False))

    def test_three_way_tie_uses_first_without_reference(self) -> None:
        self.assertEqual(numeric_vote(["7", "8", "9"]), ("7", 0, True))

    def test_empty_votes_are_deterministic(self) -> None:
        self.assertEqual(numeric_vote(["", "", ""]), ("", 0, True))

    def test_seed_is_stable(self) -> None:
        self.assertEqual(stable_seed("sample/1"), stable_seed("sample/1"))
        self.assertNotEqual(stable_seed("sample/1"), stable_seed("sample/2"))


if __name__ == "__main__":
    unittest.main()
