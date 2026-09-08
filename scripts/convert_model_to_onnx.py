import argparse

import numpy as np
import onnx
import torch
import torch.nn as nn

from utils.models.deepjet import DeepJet

batch_size = 2  # needed for cmssw!


class DeepJetCMSSW(nn.Module):
    def __init__(self, model, **kwargs):
        super(DeepJetCMSSW, self).__init__(**kwargs)
        self.deepjet_model = model

    def forward(self, global_vars, cpf, npf, vtx):
        flat = torch.cat(
            (
                global_vars,
                torch.flatten(cpf, start_dim=1),
                torch.flatten(npf, start_dim=1),
                torch.flatten(vtx, start_dim=1),
            ),
            dim=1,
        )
        flat = flat.reshape((global_vars.shape[0], 613, 1))
        out = self.deepjet_model(flat)
        return torch.softmax(out, dim=1)


def load_model(model_path: str):
    """
    Loads the DeepJet model and returns it.
    """
    print(f"loading model {model_path}")

    model = DeepJet().to("cpu")
    best_model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(best_model["model_state_dict"])
    model = DeepJetCMSSW(model)
    model.eval()
    return model


def save_to_onnx(model, output_path: str):
    """
    Exports the DeepJet model in the ONNX format.
    """
    print(f"saving to {output_path}")
    input_shapes = {
        "input_0": (batch_size, 15),
        "input_1": (batch_size, 25, 16),
        "input_2": (batch_size, 25, 6),
        "input_3": (batch_size, 4, 12),
    }
    inputs = tuple(
        torch.ones(value, dtype=torch.float32, device="cpu") for value in input_shapes.values()
    )
    torch.onnx.export(
        model,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        output_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=15,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        output_names=["ID_pred"],
        dynamic_axes={
            "input_0": {0: "N"},
            "input_1": {0: "N"},
            "input_2": {0: "N"},
            "input_3": {0: "N"},
            "ID_pred": {0: "N"},
        },
        input_names=list(input_shapes.keys()),  # the model's input names
    )

    # Test the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def main(model_path, output_path):
    model = load_model(model_path)
    save_to_onnx(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Path to model to convert", required=True)

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file where the exported model is stored",
        required=True,
    )
    args = parser.parse_args()
    main(args.model, args.output)
