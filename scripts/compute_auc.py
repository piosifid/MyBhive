import numpy as np, os
from sklearn.metrics import roc_auc_score

DATA_PATH = os.environ['DATA_PATH'].rstrip('/')

paths = {
    'baseline': 'InferenceTask/part_run3_lt/dijet_full_wide/dijet_test_wide/particleNet_dijet_wide/ParticleNet_InPro/epochs_40/nominal/test_attack_nominal',
    'flatpt':   'InferenceTask/part_run3_lt/dijet_full_plus_flatpt_wide/dijet_test_wide/particleNet_dijet_flatpt_wide/ParticleNet_InPro/epochs_40/nominal/test_attack_nominal',
}

for label, rel in paths.items():
    base = os.path.join(DATA_PATH, rel)
    pred = np.load(os.path.join(base, 'prediction.npy'))
    truth = np.load(os.path.join(base, 'truth.npy'))

    b_score = pred[:, 0]
    is_b = (truth == 0).astype(int)

    auc = roc_auc_score(is_b, b_score)
    print(f'{label}: b-vs-rest AUC = {auc:.4f}   (n={len(truth):,}, n_b={is_b.sum():,})')
