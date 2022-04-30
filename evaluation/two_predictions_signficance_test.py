import os
import sys
import gzip
import json
import pandas as pd
from scipy.stats import ttest_ind
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--first', type=str, required=True)
parser.add_argument('--second', type=str, required=True)
parser.add_argument("--metrics", type=str,required=False, default=None)
args = parser.parse_args()

predictions_file_1 = args.first 
predictions_file_2 = args.second
first_name = os.path.basename(predictions_file_1).rstrip(".json.gz")
second_name = os.path.basename(predictions_file_2).rstrip(".json.gz")

def get_metrics(doc):
    result = doc['metrics']
    if 'sampled_metrics' in doc:
        for key in doc['sampled_metrics']:
            result[f"sampled_{key}"] = doc['sampled_metrics'][key]
    return result

def read_data(filename):
    result = []
    with gzip.open(filename) as input:
        for line in input:
            doc = json.loads(line)
            metrics = get_metrics(doc)
            result.append(metrics)
    return pd.DataFrame(result)

df1 = read_data(predictions_file_1)
df2 = read_data(predictions_file_2)

overlap_columns = set(df1.columns).intersection(set(df2.columns))

if args.metrics is not None:
    overlap_columns = overlap_columns.intersection(set(args.metrics.split(",")))


docs = []

for column_name in overlap_columns:
    df1_series = df1[column_name]
    df2_series = df2[column_name]

    mean1 = df1_series.mean()
    mean2 = df2_series.mean()
    doc = {}
    doc["metric_name"] = column_name
    doc[first_name] = mean1
    doc[second_name] = mean2
    doc["difference"] = mean2 - mean1
    doc["difference_pct"] = (mean2 - mean1) * 100 / mean1
    t, pval = ttest_ind(df1_series, df2_series) 
    doc["p_value"] = pval 
    doc["p_value_bonferoni"] = pval * len(overlap_columns)
    docs.append(doc)

result = pd.DataFrame(docs)
result['significant_0.05'] = result["p_value_bonferoni"] < 0.05
result['significant_0.01'] = result["p_value_bonferoni"] < 0.01
result['significant_0.001'] = result["p_value_bonferoni"] < 0.001
result['significant_0.0001'] = result["p_value_bonferoni"] < 0.0001

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):  # more options can be specified also
    print(result)

