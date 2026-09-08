from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        device=None,
        nepochs=0,
        learning_rate=0.001,
        resume_epochs=0,
        **kwargs,
    ):
        pass

    @abstractmethod
    def validate_model(
        self,
        dataloader,
        loss_fn,
        device="cpu",
        verbose=True,
    ):
        pass

    @abstractmethod
    def predict_model(self, dataloader, device=None):
        pass

    @abstractmethod
    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        pass
