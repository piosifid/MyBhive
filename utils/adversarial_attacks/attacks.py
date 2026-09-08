import numpy as np
import torch


torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("medium")

def make_tensor(value, device, pin_memory_cpu=True):
    try:
        if device != 'cpu' and pin_memory_cpu:
            return torch.tensor(value, pin_memory=True, device=device)
        else:
            return torch.tensor(value, device=device)
    except RuntimeError as e:
        if 'pin_memory' in str(e):
            # Fallback: silently ignore pin_memory if unsupported
            return torch.tensor(value, device=device)
        else:
            raise


class Attacks:
    def __init__(
        self,
        number_classes,
        device=torch.device("cpu"),
        input_keys=[],
        integer_positions=None,
        default_values=None,
        epsilon=0.0,
        epsilon_factors=False,
        iterations=1,
        reduce=True,
        restrict_impact=-1,
        overshoot=0.02,
        epsilon_dir="",
        store_path="",
        uncertainty=1.0,
        **kwargs,
    ):
        super(Attacks, self).__init__(**kwargs)
        self.device = device
        self.work_dir = store_path
        
        self.torch_zero = make_tensor(0.0, self.device)
        self.torch_one = make_tensor(1.0, self.device)
        self.torch_inf = make_tensor(float("inf"), self.device)
        self.number_classes = make_tensor(number_classes, self.device)
        self.epsilon = make_tensor(epsilon, self.device)
            
        self.input_keys = input_keys
        
        if epsilon_factors:
            self.epsilons_per_feature = [
                torch.from_numpy(
                    np.array(
                        [
                            np.load(f"{epsilon_dir}input_category_{i}.npy")[key]
                            for key in self.input_keys[i]
                        ]
                    ).reshape(-1)
                ).to(self.device, non_blocking=True)
                for i in range(len(self.input_keys))
            ]
        else:
            self.epsilons_per_feature = [
                self.torch_one for i in range(len(self.input_keys))
            ]
        self.iterations = iterations
        self.reduce = reduce
        self.restrict_impact = restrict_impact
        self.integers = [
            integer.to(self.device, non_blocking=True) for integer in integer_positions
        ]
        self.defaults = [
            default.to(self.device, non_blocking=True) for default in default_values
        ]
        self.overshoot = make_tensor(overshoot, self.device)
        self.uncertainty = make_tensor(uncertainty, self.device)

    def nominal(self, inputs, truth, model, criterion, device):
        return *inputs, truth

    def create_default_integer_mask(self, inputs):
        masks = []
        for input, integer, default in zip(inputs, self.integers, self.defaults):
            mask = input == default
            mask[..., integer] = True
            masks.append(mask)
        return masks

    def do_not_change(self, inputs, adversarial_vectors):
        if self.reduce == False:
            return adversarial_vectors

        elif self.reduce == True:
            masks = self.create_default_integer_mask(inputs)

            for index, (mask, adversarial_vector) in enumerate(
                zip(masks, adversarial_vectors)
            ):
                adversarial_vectors[index] = torch.where(
                    mask, self.torch_zero, adversarial_vector
                )
            return adversarial_vectors

    def already_fooled(self, adversarial_vectors, nominal_labels, adversarial_labels):
        initial_mask = nominal_labels == adversarial_labels

        for index, adversarial_vector in enumerate(adversarial_vectors):
            shape = list(adversarial_vector.shape)
            shape[0] = -1
            mask = initial_mask.clone().reshape(-1, *[1 for i in range(len(shape) - 1)])
            mask = mask.expand(shape)
            adversarial_vectors[index] = torch.where(
                mask, adversarial_vector, self.torch_zero
            )
        return adversarial_vectors

    def limit_relative_change(self, inputs, adversarial_inputs):
        with torch.no_grad():
            if self.restrict_impact > 0:
                for index, (input, adversarial_input) in enumerate(
                    zip(inputs, adversarial_inputs)
                ):
                    adversarial_inputs[index] = torch.clamp(
                        adversarial_input,
                        min=input - self.restrict_impact * torch.abs(input),
                        max=input + self.restrict_impact * torch.abs(input),
                    )
        return adversarial_inputs

    def optimizer_constraints(
        self, nominal_inputs, adversarial_inputs, uncertainty, model, batch_size
    ):
        scalers = []
        with torch.no_grad():
            for nominal in nominal_inputs:
                scaler = torch.max(torch.abs(nominal), 0)[0]
                scaler = torch.where(scaler == 0, 1, scaler)
                scalers.append(scaler)

        summation = 0
        for nominal, adversarial, scaler in zip(
            nominal_inputs, adversarial_inputs, scalers
        ):
            summation += torch.sum(
                ((nominal - adversarial) / (scaler * uncertainty)) ** 2
            )
        summation /= (
            model.feature_edges[-1].clone().detach().to(self.device, non_blocking=True) * batch_size
        )

        return summation

    def pgd(self, inputs, truth, model, criterion, device):
        alphas = []
        for e in self.epsilons_per_feature:
            alphas.append(self.epsilon * e / self.iterations)
        print(f"len(alphas)={len(alphas)}")
        print(f"torch.isnan(alphas).any()={torch.isnan(torch.Tensor(alphas)).any()}")
        print(f"torch.isnan(alphas).all()={torch.isnan(torch.Tensor(alphas)).all()}")
        
        
        adversarial_inputs = []
        for input in inputs:
            adversarial_inputs.append(
                input.detach()
                .clone()
                .to(self.device, non_blocking=True)
                .requires_grad_(True)
            )

        for i in range(self.iterations):
            prediction = model.forward(tuple(adversarial_inputs))
            if (isinstance(truth, tuple)):
                loss, _, _ = criterion(prediction[0], truth[0], prediction[1][:,:-1], truth[1], prediction[1][:,-1], truth[2], device)
                truth = truth[0]
            else:
                loss = criterion(prediction, truth).mean()

            model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                gradients = []
                for input in adversarial_inputs:
                    if input.grad is not None:
                        gradients.append(input.grad.detach().sign())
                    else:
                        gradients.append(torch.zeros_like(input).to(self.device))
                print(f"len(gradients)={len(gradients)}")
                print(f"[torch.isnan(g).any() for g in gradients]={[torch.isnan(g).any() for g in gradients]}")
                # print(f"torch.isnan(gradients).any()={torch.isnan(torch.Tensor(gradients)).any()}")
                # print(f"torch.isnan(gradients).all()={torch.isnan(torch.Tensor(gradients)).all()}")
                deltas = []
                for input, alpha, gradient in zip(
                    adversarial_inputs, alphas, gradients
                ):
                    deltas.append(
                        torch.clamp(
                            (input + alpha * gradient) - input,
                            min=-alpha * self.iterations,
                            max=alpha * self.iterations,
                        )
                    )

                print(f"len(deltas)={len(deltas)}")
                print(f"[torch.isnan(d).any() for d in deltas]={[torch.isnan(d).any() for d in deltas]}")
                # print(f"torch.isnan(deltas).any()={torch.isnan(torch.Tensor(deltas)).any()}")
                # print(f"torch.isnan(deltas).all()={torch.isnan(torch.Tensor(deltas)).all()}")
                deltas = self.do_not_change(inputs, deltas)
                print(f"len(deltas)={len(deltas)}")
                print(f"[torch.isnan(d).any() for d in deltas]={[torch.isnan(d).any() for d in deltas]}")
                # print(f"torch.isnan(deltas).any()={torch.isnan(torch.Tensor(deltas)).any()}")
                # print(f"torch.isnan(deltas).all()={torch.isnan(torch.Tensor(deltas)).all()}")

                for index, delta in enumerate(deltas):
                    adversarial_inputs[index] += delta
        adversarial_inputs = self.limit_relative_change(inputs, adversarial_inputs)
        # print(f"adversarial_inputs={adversarial_inputs}")
        return *[input.detach() for input in adversarial_inputs], truth

    # Code adapted from https://github.com/LTS4/DeepFool, based on https://arxiv.org/pdf/1511.04599.pdf
    def jetfool(self, inputs, truth, model, criterion):
        batch_size = make_tensor(inputs[0].shape[0], self.device)
        prediction = model.forward(inputs)
        I = torch.argsort(prediction, dim=1, descending=True)
        label = I[:, 0]

        adversarial_inputs = []
        ws = []
        r_tots = []
        for input in inputs:
            adversarial_inputs.append(
                input.detach()
                .clone()
                .to(self.device, non_blocking=True)
                .requires_grad_(True)
            )
            shape = input.shape
            ws.append(torch.zeros(shape, device=self.device))
            r_tots.append(torch.zeros(shape, device=self.device))

        loop = 0

        fs = model.forward(adversarial_inputs)

        k_i = label

        while torch.any(k_i == label).item() and loop < self.iterations:
            perturbations = [self.torch_inf for i in range(len(inputs))]

            fs_sorted = torch.gather(fs, 1, I)
            fs_sorted[:, 0].backward(
                gradient=torch.ones(batch_size, device=self.device),
                retain_graph=True,
            )

            original_gradients = []
            for input in adversarial_inputs:
                if input.grad is not None:
                    original_gradients.append(input.grad.clone())
                else:
                    original_gradients.append(torch.zeros_like(input).to(self.device))

            for c in range(1, self.number_classes):
                for input in adversarial_inputs:
                    if input.grad is not None:
                        input.grad.zero_()

                fs_sorted[:, c].backward(
                    gradient=torch.ones(batch_size, device=self.device),
                    retain_graph=True,
                )

                adversarial_gradients = []
                for input in adversarial_inputs:
                    if input.grad is not None:
                        adversarial_gradients.append(input.grad.clone())
                    else:
                        adversarial_gradients.append(torch.zeros_like(input).to(self.device))

                w_cs = []
                for original_gradient, adversarial_gradient in zip(
                    original_gradients, adversarial_gradients
                ):
                    w_cs.append((adversarial_gradient - original_gradient).detach())

                f_c = (fs_sorted[:, c] - fs_sorted[:, 0]).detach()

                new_perturbations = []
                for w_c in w_cs:
                    new_perturbation = torch.abs(f_c) / torch.linalg.norm(
                        w_c.reshape(batch_size, -1), axis=1
                    )
                    new_perturbations.append(
                        torch.where(
                            new_perturbation == self.torch_inf,
                            self.torch_zero,
                            new_perturbation,
                        )
                    )

                for index, (perturbation, new_perturbation, w_c, w) in enumerate(
                    zip(perturbations, new_perturbations, w_cs, ws)
                ):
                    compare = new_perturbation < perturbation
                    perturbations[index] = torch.where(
                        compare, new_perturbation, perturbation
                    )
                    w[compare, ...] = w_c[compare, ...]
                    ws[index] = w

            r_is = []
            for w, perturbation in zip(ws, perturbations):
                w_dimension = w.ndim
                shape = [-1]
                for i in range(w_dimension - 1):
                    shape.append(1)

                r_is.append(
                    torch.nan_to_num(
                        w
                        * (
                            (perturbation + 1e-4)
                            / (torch.linalg.norm(w.reshape(batch_size, -1), axis=1))
                        ).reshape(*shape),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )

            r_is = self.already_fooled(r_is, label, k_i)

            for index, r_i in enumerate(r_is):
                r_tots[index] += r_i

            r_tots = self.do_not_change(inputs, r_tots)

            for index, (nominal_input, r_tot) in enumerate(zip(inputs, r_tots)):
                adversarial_inputs[index] = (
                    nominal_input + (1 + self.overshoot) * r_tot * self.epsilon
                ).requires_grad_(True)

            fs = model.forward(adversarial_inputs)
            k_i = torch.argmax(fs, axis=1)

            loop += 1

        for index, r_tot in enumerate(r_tots):
            r_tots[index] *= 1 + self.overshoot

        adversarial_inputs = self.limit_relative_change(inputs, adversarial_inputs)

        return *[input.detach() for input in adversarial_inputs], truth

    def optimizer(self, inputs, truth, model, criterion):
        softmax = torch.nn.Softmax(dim=-1)
        masks = self.create_default_integer_mask(inputs)
        min_loss = self.torch_inf
        batch_size = make_tensor(inputs[0].shape[0], self.device)

        goal_score = (
            1
            / (self.number_classes - 1)
            * torch.ones(
                batch_size,
                self.number_classes,
                device=self.device,
                requires_grad=False,
            )
        )
        for i in range(batch_size):
            goal_score[i, truth[i]] = 0

        adversarial_inputs = []
        for input in inputs:
            adversarial_inputs.append(
                input.detach()
                .clone()
                .to(self.device, non_blocking=True)
                .requires_grad_(True)
            )

        minimizer = torch.optim.Adam(params=adversarial_inputs, lr=self.epsilon)

        objective_score = []
        for i in range(self.iterations):
            adversarial_prediction = softmax(model.forward(adversarial_inputs))
            constraints = self.optimizer_constraints(
                inputs,
                adversarial_inputs,
                self.uncertainty,
                model,
                batch_size,            
            )
            objective = constraints - torch.mean(
                (goal_score - adversarial_prediction) ** 2
            )

            minimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=False)
            minimizer.step()

            with torch.no_grad():
                for index, (mask, input) in enumerate(zip(masks, inputs)):
                    adversarial_inputs[index][mask] = input[mask]

            objective_score.append(objective.item())

            if objective_score[-1] < min_loss:
                min_loss = objective_score[-1]
                best_fit = [
                    adversarial_input.detach().clone()
                    for adversarial_input in adversarial_inputs
                ]

        best_fit = self.limit_relative_change(inputs, best_fit)
        return *[input.detach() for input in best_fit], truth
