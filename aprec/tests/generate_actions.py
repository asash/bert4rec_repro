def generate_actions(n):
    from math import sin, cos
    from aprec.api.action import Action
    max_users = n / 3
    max_timestamp = n / 2
    result = []
    actions_set = set() 
    i = 0
    while len(result) < n:
        user_id = int((sin(i) + 1)/2 * max_users) 
        item_id = int((cos(i) + 1)/2 * max_users) 
        timestamp = int((sin(i) ** 2) * max_timestamp)  
        if ((user_id, item_id) not in actions_set):
            actions_set.add((user_id, item_id))
            result.append(Action(user_id, item_id, timestamp))
        i += 1
    return result
