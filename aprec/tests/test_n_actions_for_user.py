import unittest
REFERENCE_1_ACTION =\
"""Action(uid=0, item=0, ts=2)
Action(uid=1, item=3, ts=0)
Action(uid=2, item=0, ts=0)
Action(uid=3, item=2, ts=3)"""


REFERENCE_2_ACTION =\
"""Action(uid=0, item=0, ts=2)
Action(uid=0, item=2, ts=4)
Action(uid=1, item=0, ts=0)
Action(uid=1, item=3, ts=0)
Action(uid=2, item=0, ts=0)
Action(uid=2, item=2, ts=2)
Action(uid=3, item=0, ts=4)
Action(uid=3, item=2, ts=3)"""

def sorted_actions_str(actions):
    return "\n".join(sorted([str(action) for action in actions]))

class TestNActionsForUser(unittest.TestCase):
    def test_n_actions_for_user(self):
        from aprec.tests.generate_actions import generate_actions
        from aprec.evaluation.n_actions_for_user import n_actions_for_user


        actions = generate_actions(10)
        actions_1 = n_actions_for_user(actions, 1)
        actions_2 = n_actions_for_user(actions, 2)
        self.assertEqual(sorted_actions_str(actions_1), REFERENCE_1_ACTION)
        self.assertEqual(sorted_actions_str(actions_2), REFERENCE_2_ACTION)
        

if __name__ == "__main__":
    unittest.main()
