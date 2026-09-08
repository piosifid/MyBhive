from torch import nn
from utils.loss.CrossEntropyLogCosh import CrossEntropyLogCosh, CrossEntropyLogCoshHLT

def LossFunctionLoader(
    loss_function_name: str,
    class_weights=None,
    **kwargs,
) -> object:
    """
    Loads the loss function based on the specified name and parameters.

    Args:
        loss_function_name (str): The name of the loss function to use.
        class_weights (Tensor): A manual rescaling weight given to each class.
        **kwargs: Additional keyword arguments for the loss function.

    Returns:
        torch.nn.L1Loss: An instance of the specified loss function.

    Raises:
        NotImplementedError: If the loss function name is not recognized.
    """
    match loss_function_name:
        case "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        case "CrossEntropyLogCosh":
            return CrossEntropyLogCosh(weight=class_weights, reduction="mean", quantiles = [-1, 0.16, 0.84])
        case "CrossEntropyLogCoshHLT":
            return CrossEntropyLogCoshHLT(weight=class_weights, reduction="mean")
        case _:
            raise NotImplementedError(f"The loss function '{loss_function_name}' is not implemented.")