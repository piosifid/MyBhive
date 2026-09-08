# Tests

The folder provides six shell scripts that together form the required regression-test suite. Each script both illustrates a typical b-hive workflow and verifies that the corresponding functionality is intact. Successful completion is indicated by the terminal banner “GREAT SUCCESS”. Before committing any modification, run the master script `run_all_tests.sh` or run the individual tests listed below..

1) `test_offline.sh`
   This is the biggest test for various combinations of hyperparameters for offline analysis with different models. There are 4 tests inside.
   - Test 1:  part_run3 + ParticleNet_InPro
   - Test 2:  part_fp16_run3 + DeepJet     
   - Test 3:  part_run3 + DeepJetTransformer
   - Test 4:  offline_run3 + ParticleTransformer
   - Test 5:  UParT_v0_run3 + UParT_v0
   - Test 6:  part_run3 + ParticleTransformer2
   
2) `test_hlt.sh`
   Validates models tailored to the HLT environment, where charged- and neutral-particle features are processed jointly. There are 4 tests inside.
   - Test 1:  hlt_run3 + DeepJetHLT
   - Test 2:  hlt_run3 + DeepJetTransformerHLT
   - Test 3:  hlt_run3 + UParticleNet_InProHLT
   - Test 4:  hlt_run3 + GlobalParticleTransformerHLT
  
3) `test_LT.sh`
   Confirms the correct handling of models that use lost-track (LT) candidate information. There are 2 tests inside.
   - Test 1:  part_run3_lt + ParticleTransformer2_LT
   - Test 2:  part_run3_lt_fp16 + ParticleTransformer2_LT_tau
  
4) `test_MoD.sh`
   Demonstrates the creation of derived features directly within the configuration file and verifies the associated model:
   - Test 1:  mod_offline_run3 + MoDJet

5) `test_numpy.sh`
   Ensures backward compatibility with earlier versions that stored data as NumPy structured arrays:
   - Test 1:  part_run3_npy + ParticleTransformer2
   - 
6) `test_paired.sh`
   This is the test of PairedJet tagger created for tagging 2 jets together:
   - Test 1:  PAIReD_ParT_cls + LZ4PAIReDTagger
 