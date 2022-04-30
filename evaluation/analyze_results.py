import sys
import json
import os
from copy import deepcopy
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False) 

experiment_file = sys.argv[1]
data = json.load(open(experiment_file))

commit_filename = os.path.join(os.path.dirname(experiment_file), "commit")
if os.path.isfile(commit_filename):
    with open(commit_filename) as commit_file:
        print(commit_file.read())

i = 0
for split_fraction in data:
    print("="*40)
    i += 1
    doc = deepcopy(split_fraction)
    recommenders = doc['recommenders']
    del(doc['recommenders'])
    print("experiment_{}".format(i))
    print (pd.DataFrame([doc]).T)
    print("\n")

    experiment_docs = []
    for recommender_name in recommenders:
        recommender = recommenders[recommender_name]
        recommender['name'] = recommender_name
        del(recommender['model_metadata'])
        experiment_docs.append(recommender)

    df = pd.DataFrame(experiment_docs)
    df = df.sort_values("ndcg@40")
    df = df.set_index('name')
    print(df)
        


