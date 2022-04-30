config=$1
N=$2

if [ "$CHECK_COMMIT_STATUS" != "false" ]; then
    if [ -n "$(git status --porcelain)" ]; then
      echo "There are changes in the repo. Please commit the code in order to maintain traceability of the experiments";
      exit 1
    fi
fi

config_filename=$(basename -- "$config")
config_id="${config_filename%.*}"



date=`date +%Y_%m_%dT%H_%M_%S`
experiment_id="${config_id}_${date}"
root_dir=./results/$experiment_id
experiment_stdout=$root_dir/stdout
experiment_stderr=$root_dir/stderr
experiment_commit=$root_dir/commit

mkdir $root_dir

latest_experiment_link=./results/latest_experiment

rm -f $latest_experiment_link
ln -s `pwd`/$root_dir $latest_experiment_link

echo experement resutls are saved at $root_dir

cp $config $root_dir

for i in `seq 1 $N`;
do 
    experiment_result=$root_dir/experiment_${i}.json
    echo "experiment_stdout: ${experiment_stdout}"
    echo "experiment_stderr: ${experiment_stderr}"
    echo "experiment_result: ${experiment_result}"
    echo "experiment_commit: ${experiment_commit}"
    git log -1 > $experiment_commit
	unbuffer python3 run_experiment.py $config $experiment_result > $experiment_stdout 2> $experiment_stderr;
done;
