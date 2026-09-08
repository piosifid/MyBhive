# Using b-hive to custom train state-of-the-art (for b-tagging Z' and beyond)

## Step 1: Select the MC samples from DAS and preprocess them with DeepNtuplizer, in order to create model-ready training/test datasets

### 1.1: Select the MC samples from DAS

For instance, for the custom training specific to the BSM Z' search with two b-jets in the final state, we use the following training samples:

1. [dataset=/ZprimeTobb_M*_TuneCP5_13p6TeV_pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v15-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FZprimeTobb_M*_TuneCP5_13p6TeV_pythia8%2FRun3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v15-v2%2FMINIAODSIM)
2. [dataset=/QCD_PT-\*to\*0_TuneCP5_13p6TeV_pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FQCD_PT-*to*0_TuneCP5_13p6TeV_pythia8%2FRun3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2%2FMINIAODSIM)
3. [dataset=/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FTTto4Q_TuneCP5_13p6TeV_powheg-pythia8%2FRun3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2%2FMINIAODSIM)
4. [dataset=/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FTTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8%2FRun3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2%2FMINIAODSIM)
5. [dataset=/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FTTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8%2FRun3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2%2FMINIAODSIM)

and the following test samples:

1. [dataset=/ZprimeTobb_M*_TuneCP5_13p6TeV_pythia8/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FZprimeTobb_M*_TuneCP5_13p6TeV_pythia8%2FRun3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2%2FMINIAODSIM)
2. [dataset=/QCD_PT-*_TuneCP5_13p6TeV_pythia8/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5_ext1-v2/MINIAODSIM](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FQCD_PT-*_TuneCP5_13p6TeV_pythia8%2FRun3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5_ext1-v2%2FMINIAODSIM)

### 1.2: Preprocess the samples with DeepNtuplizer

After selecting the DAS samples, we preprocess them with [DeepNtuplizer](https://gitlab.cern.ch/cms-btv/DeepNTuples), in order to "flatten" them, where "flattening" means transforming the samples from events containing jets to plain jet samples (we no longer keep the concept of events, we don't know which event each jet belongs to, etc.). This is necessary, since the input of the state-of-the-art flavour tagging models (PNet, ParT, UParT) is (typically) a single jet, in its particle cloud representation.

For the custom training specific to the BSM Z' search, two "flat" training datasets:

1. a full dataset that keeps all jets from all events \
Location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Mon_115056_highPT`
2. a dijet dataset, the subset of the full dataset that only keeps the two leading jets from each event and applies selection criteria (mjj > 1.6TeV and |Δη| < 1.1) \
Location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Mon_235233_highPT`

and one "flat" test dataset:

1. a dijet dataset, similar to the training dijet dataset (only Z' and QCD samples + only two leading jets + selection criteria), but using the DAS samples dedicated for testing/evaluation (`Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2`) \
Location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Tue_151515_highPT`

were created.

(The `DeepNtuplizer` source code was slightly tweaked, in order to incorporate filtering with the dijet criteria.)

## Step 2: Download and set up the b-hive source code (instructions provided in the b-hive main README)

The current project is an old fork of the [b-hive repo](https://gitlab.cern.ch/cms-btv/b-hive). You can download the most recent one, in order to have the most updated version, with improved models, features, etc. The b-hive main README contains information and instructions on how to set up the framework, including the installation and activation of a conda envorinoment, in order to install and use the required python packages.

## Step 3: Use the scripts provided under this condor folder, in order to use b-hive modular tasks for custom trainings, evalutations, etc.

The current `condor` folder contains scripts, both bash (`.sh`) and condor submission files (`.sub`), used to submit jobs via condor, in order to:

1. construct the datasets, both for training and for testing (`DatasetConstructorTask`)
2. run the trainings (`TrainingTask`)
3. evaluate the custom trained models, via their ROC curves (`ROCCurveTask`)

If you decide to install the most recent version of b-hive, you can copy this `condor` folder as is, under your own `b-hive` folder (else you can use this project directly).

### 3.1: Use the "flat" training and test datasets (DeepNtuplizer output), in order to create the final b-hive datasets (`DatasetConstructorTask`)

`DatasetConstructorTask` uses:

1. a `.yml` configuration file (here `config/part_run3_lt.yml` was used), in order to determine the set of features, pT and eta ranges, ground truths, etc., to be used for the creation of the dataset
2. a `.txt` file, in order to determine the set of samples (DeepNtuplizer output) to be used for the creation of the dataset

The `.txt` file contains a new line separated list of absolute paths to the "flat" `.root` files.

Five training datasets:

1. a full enhanced dataset, containing all jet from all events from all DAS training samples (Z', QCD and tt) \
`.txt` location: `filelists/full/train_val_samples_enhanced.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/DatasetConstructorTask/part_run3_lt/train_enhanced`
2. a full dataset, the subset of the full enhanced dataset that only contains Z' and QCD samples (no tt) \
`.txt` location: `filelists/full/train_val_samples.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/DatasetConstructorTask/part_run3_lt/train`
3. a dijet dataset, the subset of the full dataset (only Z' and QCD samples) that also incorporates the dijet selection criteria (only two leading jets + mjj > 1.6TeV and |Δη| < 1.1) \
`.txt` location: `filelists/dijet/train_val_samples.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/DatasetConstructorTask/part_run3_lt/train`
4. a full balanced dataset, the subset of the full dataset (only Z' and QCD samples) that only keeps ~1.5% of the QCD samples (equally picked across pT and η) \
`.txt` location: `filelists/full/train_val_samples_balanced.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/DatasetConstructorTask/part_run3_lt/train_balanced`
5. a dijet balanced dataset, the subset of the full dataset (only Z' and QCD samples + only two leading jets + mjj > 1.6TeV and |Δη| < 1.1) that only keeps ~1.5% of the QCD samples (equally picked across pT and η) \
`.txt` location: `filelists/dijet/train_val_samples_balanced.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/DatasetConstructorTask/part_run3_lt/train_balanced`

and five test datasets:

1. a summer22 dataset, containing the Z' and QCD test samples (`Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2`) filtered with the dijet criteria (only two leading jets + mjj > 1.6TeV and |Δη| < 1.1) \
`.txt` location: `filelists/test/all_samples.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test/DatasetConstructorTask/part_run3_lt/summer22`
2. a summer22 937 dataset, the subset of the summer22 dataset that only keeps jets with pT < 937GeV \
`.txt` location: `filelists/test/all_samples.txt` \
(the pT cut is applied via the `.yml` configuration file) \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test/DatasetConstructorTask/part_run3_lt/summer22_937`
3. a summer22 937 2000 dataset, the subset of the summer22 dataset that only keeps jets with pT between 937GeV and 2000GeV \
`.txt` location: `filelists/test/all_samples.txt` \
(the pT cut is applied via the `.yml` configuration file) \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test/DatasetConstructorTask/part_run3_lt/summer22_937_2000`
4. a summer22 2000 8000 dataset, the subset of the summer22 dataset that only keeps jets with pT between 2000GeV and 8000GeV \
`.txt` location: `filelists/test/all_samples.txt` \
(the pT cut is applied via the `.yml` configuration file) \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test/DatasetConstructorTask/part_run3_lt/summer22_2000_8000`
5. a summer22 signal dataset, the subset of the summer22 dataset that only keeps Z' jets \
`.txt` location: `filelists/test/signal_samples.txt` \
dataset location: `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test/DatasetConstructorTask/part_run3_lt/summer22_signal`

were created.

The `.txt` files were created with the `create_file_lists.py` python script.

The condor submission file used to submit the dataset construction task to condor is `submit_dataset.sub`. It executes the `run_dataset.sh` bash script, which runs the `run_ParT_dataset.sh`, after setting up b-hive, activating the conda environment and setting the output path.

For instance, you should set

```
export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full"
```

before submitting the job to create any of the three full training datasets (full enhanced, full and full balanced).

### 3.2: Use the (five) training datasets to custom train the state-of-the-art models, PNet, ParT and UParT (`TrainingTask`)

The condor submission file used to submit the custom training task (`TrainingTask`) to condor is `submit_training.sub`. It executes the `run_training.sh` bash script, which runs the `run_ParT_training.sh`, after setting up b-hive, activating the conda environment and setting the output path.

Again, the output path should be set as

```
export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full"
```

when training on one of the three full datasets (full enhanced, full and full balanced), while it should be set as

```
export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet"
```

when training on one of the two dijet datasets (dijet and dijet balanced).

Training three models (PNet, ParT and UParT) on five different datasets results into fifteen custom trained models, which are stored under:

1. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/TrainingTask/part_run3_lt/train_enhanced/train_enhanced`
2. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/TrainingTask/part_run3_lt/train/train`
3. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/TrainingTask/part_run3_lt/train_balanced/train_balanced`

for the custom trainings on the full enhanced, full and full balanced training datasets respectively, and under:

4. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/TrainingTask/part_run3_lt/train/train`
5. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/TrainingTask/part_run3_lt/train_balanced/train_balanced`

for the custom trainings on the dijet and dijet balanced training datasets respectively.

For instance, the output of the custom training of PNet on the dijet balanced dataset with 200 epochs is stored under

```
/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/TrainingTask/part_run3_lt/train_balanced/train_balanced/ParticleNet_InPro/epochs_200/nominal/
```

with a model stored for each epoch, along with training and validation metrics for each epoch (the best model is the one with the best validation metrics and stored under `best_model.pt`).

### 3.2: Use the (five) test datasets to evaluate the (fifteen) custom trained models, via their ROC curves (`ROCCurveTask`)

Each of the fifteen custom trained models was evaluated on each of the five `summer22*` test datasets (subsets of `summer22`). The condor submission file used to submit the task that calculates the corresponding ROC curve (`ROCCurveTask`) to condor is `submit_roc.sub`. It executes the `run_roc.sh` bash script, which runs the `run_ParT_roc.sh`, after setting up b-hive, activating the conda environment and setting the output path (again `DATA_PATH` should be set depending on whether the custom trained model was trained on a full or dijet dataset).

The `ROCCurveTask` automatically runs the corresponding `InferenceTask` (if it hasn't been independently ran already).

The ROC curve outputs are stored under:

1. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/ROCCurveTask/part_run3_lt/train_enhanced/summer22*/train_enhanced/`
2. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/ROCCurveTask/part_run3_lt/train/summer22*/train/`
3. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/ROCCurveTask/part_run3_lt/train_balanced/summer22*/train_balanced/`

for the evaluation of the custom trained models that were trained on the full enhanced, full and full balanced training datasets respectively, and under:

4. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/ROCCurveTask/part_run3_lt/train/summer22*/train/`
5. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/ROCCurveTask/part_run3_lt/train_balanced/summer22*/train_balanced/`

for the evaluation of the custom trained models that were trained on the dijet and dijet balanced training datasets respectively (the `summer22*` in the paths corresponds to `summer22_937`, `summer22_937_2000`, `summer22_2000_8000` and `summer22_signal`).

The corresponding paths for the output of the `InferenceTask` are stored under:

1. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full/InferenceTask`
2. `/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet/InferenceTask`