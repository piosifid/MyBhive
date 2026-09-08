import numpy as np
import os
from numpy.lib import recfunctions

base = "/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/hlt_run3/run3_deploy_01/"

flavs = ["isB","isBB","isGBB","isLeptonicB","isLeptonicB_C","isC","isCC","isGCC","isUD","isS","isG"]

histogram = np.load(os.path.join(base, "histogram_test.npy"))
data = np.load(os.path.join(base, "validation_0.npz"))


weight = data["weight"]
truths = data["truth"]
unt = recfunctions.structured_to_unstructured(truths)

for i, f in enumerate(flavs):
   print("{0}:\t\t#events: {3:2.1f}M\tweights: {1:1.2f}+-{2:1.2f}".format(f, np.mean(weight[ t:= (np.argmax(unt, axis=1) == i)]), np.std(weight[t]) , np.sum(histogram, axis=(1,2))[i]/1e6))