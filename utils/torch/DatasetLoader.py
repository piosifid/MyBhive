from utils.torch.l1tDataset import L1TDataset
from utils.torch.LZ4Dataset import LZ4Dataset
from utils.torch.PAIReDDataset import PAIReDDataset
from utils.torch.LZ4PAIReDDataset import LZ4PAIReDDataset
from utils.torch.NumpyDataset import NumpyDataset

class DatasetName:
    L1TDataset       = "L1TDataset"
    LZ4Dataset       = "LZ4Dataset"
    PAIReDDataset    = "PAIReDDataset"
    LZ4PAIReDDataset = "LZ4PAIReDDataset"
    NumpyDataset     = "NumpyDataset"

    
def DatasetLoader(dataset_name: str = ""):
    match dataset_name:
        case DatasetName.L1TDataset:
            return L1TDataset
        case DatasetName.LZ4Dataset:
            return LZ4Dataset
        case DatasetName.PAIReDDataset:
            return PAIReDDataset
        case DatasetName.LZ4PAIReDDataset:
            return LZ4PAIReDDataset
        case DatasetName.NumpyDataset:
            return NumpyDataset
        case _:
            raise NotImplementedError