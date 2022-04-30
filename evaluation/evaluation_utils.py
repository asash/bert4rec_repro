from collections import defaultdict


def group_by_user(actions):
    result = defaultdict(list)
    for action in actions:
        result[action.user_id].append(action)
    return result