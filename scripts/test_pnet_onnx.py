import numpy as np
import onnx
import onnxruntime as ort
from utils.models.models import BTaggingModels
from torch.utils.data import DataLoader
from utils.config.config_loader import ConfigLoader

device = "cpu"
config = ConfigLoader.load_config("hlt_run3_pnet")
training_files = [
    "/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/hlt_run3_pnet/test_pnet_14/train_0.npz"
]

model = BTaggingModels("ParticleNet").to(device)
datasetClass = model.datasetClass

training_data = datasetClass(
    training_files,
    model=model,
    data_type="validation",
    weighted_sampling=False,
    device=device,
    histogram_training=None,
    bins_pt=config["bins_pt"],
    bins_eta=config["bins_eta"],
    verbose=True,
)

training_dataloader = DataLoader(
    training_data,
    batch_size=1000,
    drop_last=True,
    pin_memory=True,
    num_workers=1,
)

# now we can load the onnx model
# just check that it is not broken
# onnx_model = onnx.load("your_model.onnx")
# onnx.checker.check_model(onnx_model)
# # now run inference
# ort_sess = ort.InferenceSession("your_model.onnx")
results = []
for (
    global_features,
    cpf_features,
    vtx_features,
    cpf_points,
    vtx_points,
    truth,
    weight,
    process,
) in training_dataloader:
    # now here you need to adapt how the inputs are fed in
    print(global_features)
    # outputs = ort_sess.run(
    #     None,
    #     {
    #         "input": global_features.numpy(),
    #     },
    # )
    # results.append(outputs)
# Print Result
...
