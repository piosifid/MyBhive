import numpy as np
from utils.plotting.termplot import terminal_roc, _term_roc
from utils.plotting.roc import plot_roc
from utils.torch import L1TDataset
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import keras
from qkeras import *

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class L1TKerasDeepSet(keras.Model):
    classes = {
        "b": ["label_b"],
        "uds": ["label_uds"],
        "g": ["label_g"],
        # "tau": ["label_tau"],
    }

    n_cpf = 16
    datasetClass = L1TDataset

    global_features = [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_energy",
        "jet_npfcand",
    ]

    cpf_candidates = [
        "jet_pfcand_pt_rel",
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
        "jet_pfcand_charge",
        "jet_pfcand_id",
        "jet_pfcand_track_vx",
        "jet_pfcand_track_vy",
        "jet_pfcand_track_vz",
    ]

    custom_objects = {
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization,
    }
    optimizerClass = keras.optimizers.Adam

    def __init__(
        self,
        for_inference=False,
        learning_rate=0.001,
        **kwargs,
    ):
        super(L1TKerasDeepSet, self).__init__(**kwargs)

        # REGL = tf.keras.regularizers.l2(0.0001)
        dense_kwargs = dict(
            # kernel_initializer = tf.keras.initializers.glorot_normal(),
            # kernel_regularizer = REGL,
            # bias_regularizer = REGL,
            # kernel_constraint = tf.keras.constraints.max_norm(5),
            # kernel_quantizer = qbits,
            # bias_quantizer = qbits,
            # dropout=0.1,
        )
        self.inputs = keras.Input(shape=(16, 8), name="inputs")
        # self.bn =  keras.layers.BatchNormalization(name='BatchNorm')(self.inputs)
        # self.x1 = keras.layers.Dense(16, activation="relu", **dense_kwargs)(self.bn)
        # self.x2 = keras.layers.Dense(16, activation="relu", **dense_kwargs)(self.x1)
        # self.x3 = keras.layers.Dense(16, activation="relu", **dense_kwargs)(self.x2)
        # self.g = keras.layers.GlobalAveragePooling1D(name='avgpool')(self.x3)
        # self.x4 = keras.layers.Dense(16, activation="relu", **dense_kwargs)(self.g)
        # self.x5 = keras.layers.Dense(16, activation="relu", **dense_kwargs)(self.x4)
        # self.outputs = keras.layers.Dense(len(self.classes.items()), name="predictions")(self.x5)

        nbits = 8
        integ = 0
        # Set QKeras quantizer and activation
        if nbits == 1:
            qbits = "binary(alpha=1)"
        elif nbits == 2:
            qbits = "ternary(alpha=1)"
        else:
            qbits = "quantized_bits({},0,alpha=1)".format(nbits)
        qact = "quantized_relu({},0)".format(nbits)
        nnodes_phi = 16
        nnodes_rho = 16

        dense_kwargs = dict(
            # kernel_initializer = tf.keras.initializers.glorot_normal(),
            # kernel_regularizer = REGL,
            # bias_regularizer = REGL,
            # kernel_constraint = tf.keras.constraints.max_norm(5),
            kernel_quantizer=qbits,
            bias_quantizer=qbits,
            # dropout=0.1,
        )
        self.bn = QBatchNormalization(
            name="qBatchnorm", beta_quantizer=qbits, gamma_quantizer=qbits
        )(self.inputs)
        self.x1 = QDense(nnodes_phi, name="qDense_phi1", **dense_kwargs)(self.bn)
        self.a1 = QActivation(qact, name="qActivation_phi1")(self.x1)
        self.x2 = QDense(nnodes_phi, name="qDense_phi2", **dense_kwargs)(self.a1)
        self.a2 = QActivation(qact, name="qActivation_phi2")(self.x2)
        self.x3 = QDense(nnodes_phi, name="qDense_phi3", **dense_kwargs)(self.a2)
        self.a3 = QActivation(qact, name="qActivation_phi3")(self.x3)
        self.g = keras.layers.GlobalAveragePooling1D(name="avgpool")(self.a3)
        self.x4 = QDense(nnodes_rho, name="qDense_rho1", **dense_kwargs)(self.g)
        self.a4 = QActivation(qact, name="qActivation_rho1")(self.x4)
        self.x5 = QDense(nnodes_rho, name="qDense_rho2", **dense_kwargs)(self.a4)
        self.a5 = QActivation(qact, name="qActivation_rho2")(self.x5)
        self.outputs = QDense(
            len(self.classes.items()), name="qDense_rho3", **dense_kwargs
        )(self.a5)

        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        self.optimizer = self.optimizerClass(learning_rate=learning_rate)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["categorical_accuracy"],
        )
        self.model.summary()

    def forward(
        self,
        cpf_features,
        training=False,
    ):
        output = self.model(cpf_features, training=training)
        return output

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        device,
        optimizer=None,
        nepochs=0,
        **kwargs,
    ):
        best_loss_val = np.inf
        train_loss = []
        validation_loss = []
        train_acc = []
        validation_acc = []

        print("Initial ROC")

        if validation_data:
            _, _ = self.validate_model(
                validation_data, self.loss_fn, device, directory=directory, epoch=0
            )
        for t in range(nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_train, acc_train = self.update(
                training_data,
                self.loss_fn,
                optimizer,
                epoch=t,
                device=device,
                directory=directory,
            )
            train_loss+=loss_train
            train_acc.append(acc_train)

            if validation_data:
                loss_val, acc_val = self.validate_model(
                    validation_data, self.loss_fn, device, directory=directory, epoch=t
                )
                validation_acc.append(acc_val)
                validation_loss += loss_val

            self.model.save("{}/model_{}.keras".format(directory, t))
            self.model.save("{}/model_{}.tf".format(directory, t), save_format="tf")
            self.model.save("{}/model_{}.h5".format(directory, t), save_format="h5")

            if np.mean(loss_val) < best_loss_val:
                best_loss_val = np.mean(loss_val)
                self.model.save("{}/best_model.keras".format(directory))
                self.model.save("{}/best_model.tf".format(directory), save_format="tf")
                self.model.save("{}/best_model.h5".format(directory), save_format="h5")

        return train_loss, validation_loss, train_acc, validation_acc 

    def predict_model(self, dataloader, device):
        # self.eval()
        kinematics = []
        truths = []
        processes = []
        predictions = []
        # append jet_pt and jet_eta
        # this should be done differenlty in the future... avoid array slicing with magic numbers!
        for (
            global_features,
            cpf_features,
            truth,
            weight,
            process,
        ) in dataloader:
            cpf_features = cpf_features.numpy(force=True)
            pred = self.forward(cpf_features, training=False)
            kinematics.append(global_features[..., :2].numpy())
            truths.append(truth.numpy())
            processes.append(process.numpy())
            predictions.append(pred.numpy())

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths).astype(dtype=np.int)
        processes = np.concatenate(processes).astype(dtype=np.int)
        return predictions, truths, kinematics, processes

    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        directory=None,
        epoch=None,
        device="cpu",
        verbose=True,
    ):
        losses = []
        accuracy = 0.0
        train_acc_metric = keras.metrics.CategoricalAccuracy()

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                truth,
                weight,
                process,
            ) in dataloader:

                with tf.GradientTape() as tape:
                    cpf_features = cpf_features.numpy(force=True)
                    pred = self.forward(cpf_features, training=True)

                    truth_ = keras.utils.to_categorical(
                        truth, num_classes=len(self.classes.items())
                    )
                    loss = loss_fn(truth, pred)
                    loss = tf.nn.compute_average_loss(loss)
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


                losses.append(tf.squeeze(loss).numpy())
                train_acc_metric.update_state(truth_, pred)
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        # accuracy /= N

        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(accuracy):.4f}")
        return losses, float(accuracy)

    def validate_model(
        self,
        dataloader,
        loss_fn,
        device="cpu",
        directory=None,
        epoch=None,
        verbose=True,
    ):
        losses = []
        accuracy = 0.0
        train_acc_metric = keras.metrics.CategoricalAccuracy()

        predictions = np.empty((0, len(self.classes.items())))
        truths = np.empty((0))
        processes = np.empty((0))

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                truth,
                weight,
                process,
            ) in dataloader:
                cpf_features = cpf_features.numpy(force=True)
                pred = self.forward(cpf_features, training=False)

                truth_ = keras.utils.to_categorical(
                    truth, num_classes=len(self.classes.items())
                )
                loss = loss_fn(truth, pred)
                loss = tf.nn.compute_average_loss(loss)
                loss = tf.squeeze(loss).numpy()
                losses.append(tf.squeeze(loss).numpy())
                train_acc_metric.update_state(truth_, pred)
                # accuracy_ = train_acc_metric.result()
                predictions = np.append(predictions, pred.numpy(), axis=0)
                truths = np.append(truths, truth.numpy(), axis=0)
                processes = np.append(processes, process.numpy(), axis=0)
                N += global_features.size(dim=0)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        # accuracy /= N
        rocs = self.calculate_roc_list(predictions, truths)
        accuracy = train_acc_metric.result()
        _term_roc(*rocs[0], *rocs[1], *rocs[3], *rocs[4], "bvsall")
        fpr, tpr, _ = roc_curve(rocs[1][0], rocs[0][0])
        plot_roc(
            [[fpr, tpr, 0.0]],
            *rocs[3],
            "test",
            x_label=rocs[4],
            colors="orange",
            output_path=f"{directory}/val_roc_{epoch}.jpg",
        )
        # if verbose:
        #     terminal_roc(predictions, truths, title="Validation ROC")

        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(accuracy):.4f}")
        return losses, float(accuracy)

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = truth == 0
        others = truth != 0
        summed_jets = b_jets + others

        b_pred = predictions[:, 0]
        other_pred = predictions[:, 1:].sum(axis=1)

        bvsall = np.where(
            (b_pred + other_pred) > 0, (b_pred) / (b_pred + other_pred), -1
        )

        no_veto = np.ones(b_pred.shape, dtype=bool)

        labels = ["bvsall"]
        discs = [bvsall]
        vetos = [no_veto]
        truths = [b_jets]
        xlabels = [
            "b-identification",
        ]
        ylabels = ["mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels
