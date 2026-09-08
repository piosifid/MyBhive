#!/bin/bash
test_version="test_modjet"

# set so script aborts when something fails
set -e

if [ -z "$LXUSERNAME" ]; then
    echo "LXUSERNAME is not set. Please set it in your environment."
    exit 1
fi
if [ -z "$TESTDIRECTORY" ]; then
    echo "TESTDIRECTORY is not set. Please set it in your environment."
    exit 1
fi
mkdir -p $TESTDIRECTORY/data

# download data
if [ ! -f $TESTDIRECTORY/data/ntuple_merged_0.root ]; then
    scp -r $LXUSERNAME@lxplus.cern.ch:/eos/cms/store/group/phys_btag/ParticleEdge/big_dataset/merged_TT/ntuple_merged_0.root $TESTDIRECTORY/data/ntuple_merged_0.root
fi
echo "begin teardown of old tests"
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version --remove-output -1
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack pgd --remove-output -1
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack pgd --remove-output -1
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack jetfool --remove-output -1
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack optimizer --remove-output -1
#
# create test filelist
echo "$TESTDIRECTORY/data/ntuple_merged_0.root" > $TESTDIRECTORY/data/filelist.txt
law run DatasetConstructorTask --dataset-version $test_version --filelist $TESTDIRECTORY/data/filelist.txt --coffea-worker 4 --config mod_offline_run3
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack pgd
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack jetfool
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config mod_offline_run3 --model-name MoDJet --epochs 1 --test-dataset-version $test_version  --test-attack-magnitude 1.0 --test-attack optimizer
