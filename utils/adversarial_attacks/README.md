# Adversarial Attacks
Adversarial attacks are a way to modify the inputs based on the targeted network with the goal of degrading the performance during inference. But by performing the training on these modified inputs the robustness against an attack can be increased.
The main reason why we should be concerned with adversarial attacks is that the models are trained on simulated data. Since the real CMS detector is not always exactly alignment or parts can fail, there are always small deviations between simulation and reality. For example, a more recent study in the jet tagging context can be found [here](https://arxiv.org/abs/2203.13890).

b-hive is able to perform such attacks during training and/or inference. The [projected gradient decent (PGD)](https://arxiv.org/abs/1607.02533v4) attack is currently already implemented under `/b-hive/utils/adversarial_attacks.py` and can be used with the DeepJet model. For usage with other models only small modifications are necessary.

To apply and finetune an attack additional command line arguments are used:
- `attack` specifies which attack should be used. By default the "nominal attack" (= no attack) is used. The default is `nominal`.
- `attack_magnitude` specifies the global magnitude of the attack. The default is `0`.
- `attack_iterations` specifies how many iterations of the attack should be applied, if applicable. The default is `1`.
- `attack_individual_factors` specifies wether to use a scaled attack magnitude by input feature. It has to be provided by the user, but is not yet implemented. Currently the per-inputs attack maginutes are set to 1. The default is `True`.
- `attack_reduce` specifies wether to not change default and integer values. The default is `True`.
- `attack_restrict_impact` gives an additional restriction on the attack magnitude as a relative fraction of the original values. Negative numbers turn it off. The default is `-1.0`.

The argument `attack` is always part of the store path, while `attack_magnitude` and `attack_iterations` are only added when an attack is applied. A training with the PGD attack applied can be started like this:
```
law run TrainingTask --training-version tutorial_training_01 --dataset-version tutorial_01 --config hlt_run3 --model-name DeepJet --epochs 1 --attack pgd --attack-magnitude 0.2
```
.
