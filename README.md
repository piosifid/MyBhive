# b-hive

b-hive is a framework for Machine Learning tasks on flat root files. It aims to provide
full pipelines form dataset export to evaluation with
state-of-the-art taggers, like ParticleNet and ParticleTransformer.
Some functionalities are still work in progress and new changes will be discussed in a 
dedicated round-table every Friday. For questions, please contact the [b-hive-contact](404)

For the old BTV frameworks, see [DeepJet](https://github.com/DL4Jets/DeepJet) and [DeepJetCore](https://github.com/DL4Jets/DeepJetCore).

## Join the [Mattermost-Channel!](https://mattermost.web.cern.ch/cms-exp/pl/ft4fewfa4b86t8z8nsza7taq1a)
## Setup

The codebase is python based, using tools like [awkward](https://awkward-array.org/doc/main/), [uproot](https://uproot.readthedocs.io/en/latest/)
for reading in the root-files and [numpy](https://numpy.org), [PyTorch](https://pytorch.org)
for the training.
The different tasks are stitched together with the workflow management system [law](https://law.readthedocs.io/en/latest/).

### installation

Clone the repository via gitlab (either ssh or http):

```bash
# clone the repository
git clone ssh://git@gitlab.cern.ch:7999/cms-btv/b-hive.git
cd b-hive
```

Next, install the needed python environment via [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) (or conda):

```bash
mamba env create -n b_hive -f env.yml
mamba activate b_hive
```

Next, some environment variables need to be configured. These should be put inside
the `local_setup.sh`, which is ignored by git, but used by the `setup.sh`. The variable
`$DATA_PATH` specifies, where the outputs will be written to. Thus, chose a directory, where you
have sufficient space!

```bash
# example local_setup.sh
export DATA_PATH=/net/scratch/myUserName/b-hive/output/
```

Now source the setup-script to set up the environment everything:

```bash
# initialize needed environments 
source setup.sh
# init law tasks (needed to be executed only once)
law index
```
## Tutorial

### List tasks

To get familiar with the structure, it is advisable to list the available tasks:

```bash
# list all tasks
law index --verbose
"""
output
"""
indexing tasks in 8 module(s)
loading module 'tasks.base', done
loading module 'tasks.dataset', done
loading module 'tasks.plotting', done
loading module 'tasks.training', done
loading module 'tasks.inference', done
loading module 'tasks.working_point', done
loading module 'tasks.input_plotter', done
loading module 'workflows.DeepFlavour', done

module 'workflows.DeepFlavour', 4 task(s):
    - DeepJetRunHLT
    - DeepJetRun
    - DeepJetRunHLTPhase2
    - ParticleNetRunHLT

module 'tasks.dataset', 1 task(s):
    - DatasetConstructorTask

module 'tasks.training', 1 task(s):
    - TrainingTask

module 'tasks.inference', 1 task(s):
    - InferenceTask

module 'tasks.plotting', 1 task(s):
    - ROCCurveTask

module 'tasks.working_point', 1 task(s):
    - WorkingPointTask

module 'tasks.input_plotter', 2 task(s):
    - InputHistogrammerTask
    - HistogramPlotterTask

written 11 task(s) to index file '/eos/user/p/pkashko/b_hive_fix/b-hive/.law/index'

```

This shows us all taks that are at hand:

- DatasetConstructorTask
    - exports a dataset of root file to the training format
- TrainingTask 
    -  trains a model
- InferenceTask
    - runs the prediction on a dataset
- ROCCurveTask 
    - computes and plots the ROC on a prediction
- WorkingPointTask
    - calculates working points on a prediction
- InputHistogrammerTask
    - creates histograms of input features
- HistogramPlotterTask
    - plot histograms of input features
- ~~DeepJetRun~~ 
    - full pipeline

all these need to be in files that are specified in the `law.cfg`. Task that are in the `tasks` directory, are basic tasks, whereas the `workflows` directory specifies full training pipelines,
with the necessary parameters set.

### DatasetConstructorTask

To learn, what parameters a tasks accepts, you can either have a look into the code, or
type:
```bash
law run DatasetConstructorTask <TAB><TAB>
```
so please press double-tab to list all acceptable parameters
```bash
law run DatasetConstructorTask
--chunk-size
--config
--debug
--help
--log-file
--print-deps
--print-status
--scheduler-host
--filelist
--verbose
--coffea-worker
--dataset-version
--fetch-output
--local-scheduler
--log-level
--print-output
--remove-output
--scheduler-port
--workers
```

or you can try put
```bash
law run DatasetConstructorTask --help
```
to receive an extensive list. Some of these parameters are by law, whereas others are defined in
b-hive.

Let's explain some parameters:
 - --config 
   - this specifies the config that should be used, described in [Config](#config)
 - --dataset-version
   - this specifies the version-tag that should be used, for example `v_01`, `test_new_config_1`, ...
- --filelist
   - .txt file with input root files. This should be a file with absolut paths of the root files
   to read in. Streming with xrootd is possible since these files are read in by coffea.
- --coffea-worker
   - number of workers that should be used to parallelize the conversion. On a large machine, this can be easily set to 32 or 64
- --chunk-size
   - This specifies the number of events/jets/datapoints that will be written to the target files.
   With the number of features, you can roughly estimate the resulting filesize. A reasonable size would be 100000.
- --debug
   - activate debug mode, where not the full fileset is processed but only one per process. This is good for checking if the export is working without processing the full fileset.

Many of these paramters, like `chunk-size`, `coffea-worker` have reasonable defaults set, while
the `filelist` must explicitly be set (without, what would we read in anyway?!).
Some of the parameters are *significant* whereas some others are *insiginificant*. This refers to
if a parameter changes the output or doesn't. For example the `--coffea-worker` parameter only changes
the parallelization but not what is done, whereas the `config` might change, what config is read in.

### Task-Tree

Law is used for datapipeline management. It parses arguments and checks if a task in the pipeline
has already run or if it needs to be run prior before the specified task can be executed.
This is useful, if one wants to produce for example a ROC curve or working points on a specified
dataset but the training has not run - so it is run automatically.

To see the dependencies, one runs the command the task with `--print-status -1`, which prints *all* prior tasks. 
The number specifies the depth, whereas -1 denotes everything
```bash
(b-hive) NiclasEich@vispa-portal2(structured_arrays):~/b_tagging/b-hive$ law run ROCCurveTask --dataset-version run3_test_strucuted_array_24 --DatasetConstructorTask-coffea-worker 64 --DatasetConstructorTask-test-filelist /net/scratch/cms/data/btv/2023_08_22/test_files.txt --DatasetConstructorTask-training-filelist /net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt --config hlt_run3 --TrainingTask-n-threads 20 --model-name DeepJetHLT --batch-size 10000 --epochs 2 --training-version test_structured_array_16 --DatasetConstructorTask-chunk-size 1000000 --print-status -1
/home/NiclasEich/b_tagging/b-hive/tasks/training.py:14: UserWarning: Anomaly Detection has been enabled. This mode will increase the runtime and should only be enabled for debugging.
  torch.autograd.detect_anomaly(True)
print task status with max_depth -1 and target_depth 0

0 > ROCCurveTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000)
│     LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/ROCCurveTask/run3_test_strucuted_array_24/.../loss.pdf)
│       absent
│
├──1 > TrainingTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000, loss_weighting=False)
│  │     training_metrics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../training_metrics.npz)
│  │       absent
│  │     validation_metrics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../validation_metrics.npz)
│  │       absent
│  │     model: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../model_1.pt)
│  │       absent
│  │     best_model: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../best_model.pt)
│  │       absent
│  │
│  └──2 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_file
│         s.txt, chunk_size=1000000)
│           file_list: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/processed_files.txt)
│             existent
│           histogram_training: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/histogram_training.npy)
│             existent
│           histogram_test: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/histogram_test.npy)
│             existent
│
├──1 > InferenceTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000)
│  │     output_root: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../output.root)
│  │       absent
│  │     prediction: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../prediction.npy)
│  │       absent
│  │     process: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../process.npy)
│  │       absent
│  │     truth: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../truth.npy)
│  │       absent
│  │     kinematics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../kinematics.npy)
│  │       absent
│  │
│  ├──2 > TrainingTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000, loss_weighting=False)
│  │  │     outputs already checked
│  │  │
│  │  └──3 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_f
│  │         iles.txt, chunk_size=1000000)
│  │           outputs already checked
│  │
│  └──2 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_file
│         s.txt, chunk_size=1000000)
│           outputs already checked
│
└──1 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_files.t
       xt, chunk_size=1000000)
         outputs already checked

```


### Config

To receive a modular setup and create the possibility for multiple groups, to collaborate, different configs are
created in the `config` directory.
These denote the different root-branches that are specified to be exported, together with $p_T$ and $\eta$ bins, and other 
configuration possibilities.

All branches that are specified in the config are exported. The different models than

### Running a Training

Test files are saved in `/eos/cms/store/group/phys_btag/b-hive/test_files/data`. You can download these and then run a training.
The HLT test-files should be used with the `hlt_run3` config, ~~whereas the offline ones with the `offline_run3` config~~TODO!.

Download the files via:
```bash
scp -r user@lxplus.cern.ch:/eos/cms/store/group/phys_btag/b-hive/test_files/data/hlt local_path/hlt_test
```

these should give you these three files:
```
out_11.root  out_1.root  out_2000_test.root
```

now we need to create two filelist with a `training_files.txt` and `test_files.txt`.
This can be either done manually, or with a simple bash command:

```bash
# find all files - works also in nested dirs
find ~+ -type f -name "*.root" > all_files.txt

# put the first 2 files into training
head -n 2 all_files.txt > training_files.txt
# put last 1 into test
tail -n 1 all_files.txt > test_files.txt

# Let's check what we have created here
echo "Training files:"
cat training_files.txt
echo "Test files:"
cat test_files.txt 
```

Now we can run a Dataset-Construction with:
```bash
law run DatasetConstructorTask --dataset-version tutorial_01 --filelist YOUR_PATH/hlt_test/training_files.txt  --coffea-worker 10 --config hlt_run3
```

This should run without error. To check the outputs, we can run:

```bash
law run DatasetConstructorTask --dataset-version tutorial_01 --config hlt_run3 --print-status 0
Warning: No CUDA device available. Running on cpu...
print task status with max_depth 0 and target_depth 0

0 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=tutorial_01, chunk_size=1000000)
      file_list: LocalFileTarget(fs=local_fs, path=/DatasetConstructorTask/hlt_run3/tutorial_01/processed_files.txt)
        existent
      histogram_training: LocalFileTarget(fs=local_fs, path=YOUR_PATH/DatasetConstructorTask/hlt_run3/tutorial_01/histogram_training.npy)
        existent
      histogram_test: LocalFileTarget(fs=local_fs, path=YOUR_PATH/DatasetConstructorTask/hlt_run3/tutorial_01/histogram_test.npy)
        existent
las run Dataset
```

great, so we see that we have created the files. You can also call a `ls -l YOUR_PATH/DatasetConstructorTask/hlt_run3/tutorial_01` to see
all the files that are in there.

Next we can run a training via:
```bash
law run TrainingTask --training-version tutorial_training_01 --dataset-version tutorial_01 --config hlt_run3 --model-name DeepJetHLT --epochs 1
```

I hope you are as excited as me to see the accuracy improve!

Let's plot the final results:

```bash
law run ROCCurveTask --training-version tutorial_training_01 --dataset-version tutorial_01 --test-filelist YOUR_PATH/hlt_test/test_files.txt --test-dataset-version tutorial_test_files_01 --config hlt_run3 --model-name DeepJetHLT --epochs 1
```

Now if you check prior, which tasks already ran, you will realise that we have not run a prediciton!
This is however not a problem, since law recognizes this and rund the task itself.

Now have a looko at the results and start training your tagger on a real dataset!


## Collaborating

This describes how to add new models and configs TODO!

Classifiers are based on the parent class `utils/models/base_model.py`. Each model `utils/models` is a child class with its own specification regarding `forward()` function. Other methods could be directly overwritten in the child class, but don't change `base_model.py`.
