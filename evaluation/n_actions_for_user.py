from collections import defaultdict
def n_actions_for_user(actions, n):
    """leave only n first actions for particular user"""
    user_actions = defaultdict(lambda: [])
    for action in actions:
        user_actions[action.user_id].append(action)

    result = []
    for user_id in user_actions:
        result += sorted(user_actions[user_id], key = lambda action: action.timestamp)[:n]

    return result
        
        
