from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing
from utils.coffea_processors.lz4_processor import LZ4Processing 
from utils.coffea_processors.l1_processor import L1PFCandidateAndVertexProcessing 
from utils.coffea_processors.custom_selections import PairedTaggerProcessor
from utils.coffea_processors.lz4_custom_selections import LZ4_PairedTaggerProcessor

class ProcessorClasses:
    PFCandidateAndVertexProcessing = "PFCandidateAndVertexProcessing" 
    LZ4Processing = "LZ4Processing" 
    L1PFCandidateAndVertexProcessing = "L1PFCandidateAndVertexProcessing"
    PairedTaggerProcessor = "PairedTaggerProcessor"
    LZ4_PairedTaggerProcessor = "LZ4_PairedTaggerProcessor"


def ProcessorLoader(model: str = "", *args, **kwargs):
    match model:
        case ProcessorClasses.PFCandidateAndVertexProcessing:
            return PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.LZ4Processing:
            return LZ4Processing(*args, **kwargs)
        case ProcessorClasses.L1PFCandidateAndVertexProcessing:
            return L1PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.PairedTaggerProcessor:
            return PairedTaggerProcessor(*args, **kwargs)
        case ProcessorClasses.LZ4_PairedTaggerProcessor:
            return LZ4_PairedTaggerProcessor(*args, **kwargs)
        case _:
            return PFCandidateAndVertexProcessing(*args, **kwargs) 

