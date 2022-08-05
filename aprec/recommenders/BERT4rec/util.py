from __future__ import print_function
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        user_train[user] = User[user][:-1]
        user_test[user] = [User[user][-1]]
    return [user_train, user_test, usernum, itemnum]

