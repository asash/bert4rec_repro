from collections import defaultdict
import numpy as np

class ActionsSplitter(object):
    def __call__(self, actions):
        raise NotImplementedError

class TemporalGlobal(ActionsSplitter):
    def __init__(self, fractions):
        self.fractions = fractions
        fractions_sum = sum(fractions)
        self.fractions_real = [fraction / fractions_sum for fraction in fractions]

    def __call__(self, actions):
        """split actions into n lists by timestamp in chronological order"""


        actions_list = sorted([action for action in actions],
                               key = lambda action: action.timestamp)
        cummulative = 0.0
        num_actions = len(actions_list)
        result = []
        for fraction in self.fractions_real[:-1]:
            left_border = int(cummulative * num_actions)
            right_border = int((cummulative + fraction) * num_actions)
            result.append(actions_list[left_border:right_border])
            cummulative += fraction

        left_border = int(cummulative * num_actions)
        right_border = int(num_actions)
        result.append(actions_list[left_border:right_border])
        return result


def get_control_users(actions):
    result = set()
    for action in actions:
        if 'is_control' in action.data and action.data['is_control']:
            result.add(action.user_id)
    return result


def get_single_action_users(users):
    result = set()
    for user in users:
        if len(users[user]) == 1:
            result.add(user)
    return result

class LeaveOneOut(ActionsSplitter):
    def __init__(self, max_test_users=4096, random_seed = 31337, remove_single_action=True, recently_interacted_hours = None):
        self.max_test_users=max_test_users
        self.random_seed = random_seed
        self.remove_single_actions =remove_single_action
        self.recently_interacted_hours = recently_interacted_hours

    def __call__(self, actions):
        sorted_actions = sorted(actions, key=lambda x: x.timestamp)
        latest_action_time = sorted_actions[-1].timestamp
        users = defaultdict(list)
        eligible_users = set()
        for action in sorted_actions:
            if self.recently_interacted_hours is not None:
                if (latest_action_time - action.timestamp) * 3600 < self.recently_interacted_hours:
                    eligible_users.add(action.user_id)
            users[action.user_id].append(action)
        train = []
        test = []
        control_users = get_control_users(actions)
        if self.recently_interacted_hours is None:
            eligible_users = users.keys()

        if self.remove_single_actions:
            single_action_users = get_single_action_users(users)
            valid_user_selection = list(eligible_users - control_users - single_action_users)
        else:
            valid_user_selection = list(eligible_users - control_users)
        valid_user_selection.sort()
        np.random.seed(self.random_seed)
        test_user_ids = set(np.random.choice(valid_user_selection, self.max_test_users, replace=False))
        for user_id in users:
            if user_id in test_user_ids:
                train += users[user_id][:-1]
                test.append(users[user_id][-1])
            else:
                train += users[user_id]
        return sorted(train, key=lambda x: x.timestamp), sorted(test, key=lambda x: x.timestamp)

class RandomSplit(ActionsSplitter):
    def __init__(self, test_fraction = 0.3, max_test_users=4096, random_seed = 31337):
        self.test_fraction = test_fraction
        self.max_test_users = max_test_users
        self.random_seed = random_seed

    def __call__(self, actions):
        sorted_actions = sorted(actions, key=lambda x: x.timestamp)
        users = defaultdict(list)
        for action in sorted_actions:
            users[action.user_id].append(action)
        train = []
        test = []
        control_users = get_control_users(actions)
        valid_user_selection = users.keys() - control_users
        np.random.seed(self.random_seed)
        test_user_ids = set(np.random.choice(list(valid_user_selection), self.max_test_users, replace=False))
        for user_id in users:
            if user_id in test_user_ids:
                num_user_actions = len(users[user_id])
                num_test_actions = int(max(num_user_actions * self.test_fraction, 1))
                test_action_indices = set(np.random.choice(range(num_test_actions), num_test_actions, replace=False))
                for action_id in range(num_user_actions):
                    if action_id in test_action_indices:
                        test.append(users[user_id][action_id])
                    else:
                        train.append(users[user_id][action_id])
            else:
                train += users[user_id]
        return sorted(train, key=lambda x: x.timestamp), sorted(test, key=lambda x: x.timestamp)