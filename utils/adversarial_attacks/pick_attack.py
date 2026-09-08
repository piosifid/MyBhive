from utils.adversarial_attacks.attacks import Attacks


class AttackName:
    jetfool = "jetfool"
    nominal = "nominal"
    optimizer = "optimizer"
    pgd = "pgd"


def pick_attack(attack: str = None, *args, **kwargs):
    match attack:
        case AttackName.jetfool:
            return Attacks(*args, **kwargs).jetfool
        case AttackName.nominal:
            return Attacks(*args, **kwargs).nominal
        case AttackName.optimizer:
            return Attacks(*args, **kwargs).optimizer
        case AttackName.pgd:
            return Attacks(*args, **kwargs).pgd
        case _:
            raise NotImplementedError
