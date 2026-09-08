import torch
from utils.optimizing.Cosine_LR import CosineAnnealingWarmupRestarts

def SchedulerLoader(
    scheduler_name,
    learning_rate,
    lr_decay_factor,
    optimizer,
    nepochs,
    dataloader = None
):
    
    match scheduler_name:
        
        case 'batch_cosine_warmup':
            nsteps = dataloader.nits_expected * nepochs
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer, 
                first_cycle_steps=nsteps, 
                max_lr = learning_rate, 
                min_lr = learning_rate * lr_decay_factor, 
                warmup_steps = int(nsteps*(1/nepochs))
            )
            batch_lr = True
        
        case "epoch_lin_decay":
            lr_epochs = max(1, int(nepochs * 0.3))
            lr_rate = lr_decay_factor ** (1.0 / lr_epochs)
            mil = list(range(nepochs - lr_epochs, nepochs))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones = mil, 
                gamma = lr_rate
            )
            batch_lr = False
        
        case "batch_lin_decay":
            nsteps = dataloader.nits_expected * nepochs
            lr_epochs = max(1, int(nsteps * 0.3))
            lr_rate = lr_decay_factor ** (1.0 / lr_epochs)
            mil = list(range(nsteps - lr_epochs, nsteps))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones = mil, 
                gamma = lr_rate
            )
            batch_lr = True
        
        case _:
            raise NotImplementedError
    
    return scheduler, batch_lr