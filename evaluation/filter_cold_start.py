def filter_cold_start(train_actions, test_actions):
    train_user_ids = set()
    cold_start_set = set()
    non_cold_start = set()
    result = []

    for action in train_actions:
        train_user_ids.add(action.user_id)

    for action in test_actions:
        if action.user_id in train_user_ids:
            non_cold_start.add(action.user_id)
            result.append(action)
        else:
            cold_start_set.add(action.user_id)
    print("number of cold start users filtered: {}".format(len(cold_start_set)))
    print("number of users in test set: {}".format(len(non_cold_start)))
    return result

