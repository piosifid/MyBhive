class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    MoDJet = "MoDJet"
    ParticleTransformer = "ParticleTransformer"
    ParticleTransformer2 = "ParticleTransformer2"
    ParticleTransformer2_LT = "ParticleTransformer2_LT"
    ParticleTransformer2_LT_tau_all = "ParticleTransformer2_LT_tau_all"
    ParticleTransformer2_LT_tau = "ParticleTransformer2_LT_tau"
    ParticleTransformer2_LT_tau_Big = "ParticleTransformer2_LT_tau_Big"
    ParticleTransformer2_LT_tau_Huge = "ParticleTransformer2_LT_tau_Huge"
    ParticleTransformerHLT = "ParticleTransformerHLT"
    GlobalParticleTransformerHLT = "GlobalParticleTransformerHLT"
    UParT_v0 = "UParT_v0"
    FP16ParticleTransformer = "FP16ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"
    DeepJetTransformerHLT = "DeepJetTransformerHLT"
    UDeepJetTransformerHLT = "UDeepJetTransformerHLT"
    GlobalDeepJetTransformerHLT = "GlobalDeepJetTransformerHLT"
    ParticleNet = "ParticleNet"
    ParticleNet_InPro = "ParticleNet_InPro"
    ParticleNet_InProHLT = "ParticleNet_InProHLT"
    UParticleNet_InProHLT = "UParticleNet_InProHLT"
    ParticleNetHION = "ParticleNetHION"
    DeepJetTransformer = "DeepJetTransformer"
    L1TKerasDeepSet = "L1TKerasDeepSet"
    L1TTorchBase = "L1TTorchBase"
    MoDJet = "MoDJet"
    #PAIReDTagger = "PAIReDTagger"
    LZ4PAIReDTagger = "LZ4PAIReDTagger"
    ParticleNet = "ParticleNet"
    DeepJet_PAIReD = "DeepJet_PAIReD"
    #ParT_cls = "ParT_cls"

    
def BTaggingModels(model: str = "", *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            from utils.models.deepjet import DeepJet
            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            from utils.models.deepjet import DeepJetHLT
            return DeepJetHLT(*args, **kwargs)
        case ModelName.UParT_v0:
            from utils.models.UParT_v0 import UParT_v0
            return UParT_v0(*args, **kwargs)
        case ModelName.MoDJet:
            from utils.models.deepjet import MoDJet 
            return MoDJet(*args, **kwargs)
        case ModelName.ParticleTransformer:
            from utils.models.particletransformer import ParticleTransformer
            return ParticleTransformer(*args, **kwargs)
        case ModelName.ParticleTransformer2:
            from utils.models.particletransformer2 import ParticleTransformer2
            return ParticleTransformer2(*args, **kwargs)
        case ModelName.ParticleTransformer2_LT:
            from utils.models.particletransformer2_LT import ParticleTransformer2
            return ParticleTransformer2(*args, **kwargs)
        case ModelName.ParticleTransformer2_LT_tau:
            from utils.models.particletransformer2_LT import ParticleTransformer2_tau
            return ParticleTransformer2_tau(*args, **kwargs)
        case ModelName.ParticleTransformer2_LT_tau_all:
            from utils.models.particletransformer2_LT import ParticleTransformer2_tau_all
            return ParticleTransformer2_tau_all(*args, **kwargs)
        case ModelName.ParticleTransformer2_LT_tau_Big:
            from utils.models.particletransformer2_LT import ParticleTransformer2_tau
            return ParticleTransformer2_tau(*args, num_enc=6, embed_dim=192, **kwargs)
        case ModelName.ParticleTransformer2_LT_tau_Huge:
            from utils.models.particletransformer2_LT import ParticleTransformer2_tau
            return ParticleTransformer2_tau(*args, num_enc=12, embed_dim=192, **kwargs)
        case ModelName.ParticleTransformerHLT:
            from utils.models.particletransformer import ParticleTransformerHLT
            return ParticleTransformerHLT(*args, **kwargs)
        case ModelName.GlobalParticleTransformerHLT:
            from utils.models.particletransformer import GlobalParticleTransformerHLT
            return GlobalParticleTransformerHLT(*args, **kwargs)
        case ModelName.FP16ParticleTransformer:
            from utils.models.fp16particletransformer import FP16ParticleTransformer
            return FP16ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            from utils.models.deepjettransformer import DeepJetTransformer
            return DeepJetTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformerHLT:
            from utils.models.deepjettransformer import  DeepJetTransformerHLT
            return DeepJetTransformerHLT(*args, **kwargs)
        case ModelName.UDeepJetTransformerHLT:
            from utils.models.deepjettransformer import  UDeepJetTransformerHLT
            return UDeepJetTransformerHLT(*args, **kwargs)
        case ModelName.GlobalDeepJetTransformerHLT:
            from utils.models.deepjettransformer import  GlobalDeepJetTransformerHLT
            return GlobalDeepJetTransformerHLT(*args, **kwargs)
        case ModelName.ParticleNet_InPro:
            from utils.models.particlenet_InPro import ParticleNetTagger as ParticleNet_InPro
            return ParticleNet_InPro(*args, **kwargs)
        case ModelName.ParticleNet:
            from utils.models.particlenet_base import ParticleNetTagger
            return ParticleNetTagger(*args, **kwargs)
        case ModelName.ParticleNet_InProHLT:
            from utils.models.particlenet_InPro import ParticleNetTaggerHLT as ParticleNet_InProHLT
            return ParticleNet_InProHLT(*args, **kwargs)
        case ModelName.UParticleNet_InProHLT:
            from utils.models.particlenet_InPro import UParticleNetTaggerHLT as UParticleNet_InProHLT
            return UParticleNet_InProHLT(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            from utils.models.l1t_kerasDeepset import L1TKerasDeepSet
            return L1TKerasDeepSet(*args, **kwargs)
        case ModelName.L1TTorchBase:
            from utils.models.l1t_base import L1TTorchBase
            return L1TTorchBase(*args, **kwargs)
        case ModelName.LZ4PAIReDTagger:
            from utils.models.LZ4PAIReDTagger import LZ4PAIReDTagger
            return LZ4PAIReDTagger(*args, **kwargs)
        case ModelName.DeepJet_PAIReD:
            from utils.models.deepjet import DeepJet_PAIReD   
            return DeepJet_PAIReD(*args, **kwargs)
        case _:
            raise NotImplementedError(f"The model '{model}' is not implemented.")
'''
case ModelName.PAIReDTagger:
    from utils.models.PAIReDTagger import PAIReDTagger
    return PAIReDTagger(*args, **kwargs)
case ModelName.ParT_cls:
    from utils.models.ParT_Paired import ParT_cls   
    return ParT_cls(**kwargs)       
''' 