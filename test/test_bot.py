from typing import Tuple
from unittest import TestCase


def is_synchronized(neighbor_round: dict, my_round_y, my_round_nu_bar) -> Tuple[bool, bool]:
    y_bool: bool = True
    nu_bar_bool: bool = True
    for round_y, _ in neighbor_round.values():
        if my_round_y != round_y:  # or self.round_nu_bar != round_nu_bar:
            y_bool = False
            break
    for _, round_nu_bar in neighbor_round.values():
        if my_round_nu_bar != round_nu_bar:
            nu_bar_bool = False
            break
    return y_bool, nu_bar_bool


class TestBot(TestCase):
    def test_is_synchronized(self):
        inputs = [
            ({'1': (1, 1), '2': (0, 0), '3': (1, 1)}, 1, 1),
            ({'1': (1, 1), '2': (1, 1), '3': (1, 1)}, 1, 1),
            ({'1': (1, 0), '2': (1, 1), '3': (1, 1)}, 1, 1),
            ({'1': (0, 1), '2': (1, 1), '3': (1, 1)}, 1, 1),
        ]

        outputs = [
            (False, False),
            (True, True),
            (True, False),
            (False, True)
        ]

        for i in range(len(inputs)):
            self.assertEqual(is_synchronized(*inputs[i]), outputs[i])
