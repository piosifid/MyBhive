from .radam import RAdam
from .lookahead import Lookahead


class Ranger(object):
    def __init__(self, params,
           lr=1e-3,
           betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # RAdam options
           alpha=0.5, k=6,  # LookAhead options
           ):
        self.alpha = 0.5
        self.k = 6
        self.radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def __call__(self):
        return Lookahead(self.radam, self.alpha, self.k)

def Ranger_PAIReD(params,
           lr=1e-3,  # lr
           betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # RAdam options
           alpha=0.5, k=6,  # LookAhead options
           ):
    radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(radam, alpha, k)
