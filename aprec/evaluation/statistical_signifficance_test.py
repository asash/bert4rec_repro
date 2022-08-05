import gzip
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from scipy.stats import ttest_ind

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def process(arguments):
    metrics = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(arguments.predictions_dir):
        if filename.endswith(".json.gz"):
            in_file = gzip.open(os.path.join(arguments.predictions_dir, filename))
            recommender_name = ".".join(filename.split(".")[:-2])
        elif filename.endswith(".json"):
            in_file = open(os.path.join(filename, filename))
            recommender_name = ".".join(filename.split(".")[:-1])
        else:
            continue
        for line in in_file:
            user_doc = json.loads(line)
            for metric in user_doc["metrics"]:
                metrics[metric][recommender_name].append(user_doc["metrics"][metric])
    result = defaultdict(lambda: defaultdict(dict))
    for metric in metrics:
       for recommender_name_1 in metrics[metric]:
           rec_1_sample = metrics[metric][recommender_name_1]
           for recommender_name_2 in metrics[metric]:
                rec_2_sample = metrics[metric][recommender_name_2]
                t, p_value = ttest_ind(rec_1_sample, rec_2_sample)
                result[recommender_name_1][metric][recommender_name_2] = p_value
    with open(arguments.output_file, 'w') as output:
        output.write(json.dumps(result, indent=4))







def main():
    arguments = get_arguments()
    process(arguments)


if __name__ == "__main__":
    main()
