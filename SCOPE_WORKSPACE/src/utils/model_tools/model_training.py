import os
import random
import time
import torch
from torch.utils import data
from itertools import chain
from tqdm import tqdm
# from data_loader import TensorDataset, AggDataset
from contextlib import contextmanager
import numpy as np
from copy import deepcopy
from src.utils.data_load_tools import DL_Dataset, ML_Dataset, RL_Dataset
from src.utils.mini_tools import get_model_functions
import wandb
RESULTS_FOLDER = "res"

# Das PSP steckt als Deep learning Modell mit in DL
class ModelTrainer():
    # Debug: merkt sich über alle Trainer hinweg, ob die Trainingsdaten schon ausgegeben wurden
    train_data_printed = False

    def __init__(self, args, data_train, data_infer, weights_train, weights_infer, model_params, data_train_ps=None, data_infer_ps=None):
        self.args = args
        self.data_train = data_train
        self.data_infer = data_infer
        self.data_train_ps = data_train_ps
        self.data_infer_ps = data_infer_ps
        self.weights_train = weights_train
        self.weights_infer = weights_infer
        self.model_params = model_params

        # Set the save paths
        self.savepath_checkpoint = self.model_params["model_savepath_checkpoint"] + str(self.args.train_size) + str(self.args.delta) + str(self.model_params["target"]) + "_" + str(self.model_params["model_specific"]) + "_" + str(self.model_params["seed"]) + "_" + str(self.model_params["stage"]) + ".pt"

        # Initialize the model functions based on the target and model specific parameters
        # hier werden Funktonen übergeben, wie forward, loss, fit_and_get_loss, predict, etc. Diese Funktionen sind in model_functions.py definiert und werden hier für das Training verwendet.
        self.model_functions = get_model_functions(model_params=self.model_params)

        random.seed(self.model_params["seed"])
        torch.manual_seed(self.model_params["seed"])
        torch.cuda.manual_seed_all(self.model_params["seed"])
        np.random.seed(self.model_params["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # Trainieren der Modelle
    def train(self, tuning=False):
        # if self.args.wandb:
        #     wandb.login(key="")

        #     wandb.init(
        #         project="CLIPPS",      # e.g. "treatment-effect-nn"
        #         config=self.model_params,         # logs your hyperparameters
        #         name=self.model_params["method"],           # optional, for clarity
        #     )

        if self.model_params["model_category"] == "dl":
            return self.train_dl()
        elif self.model_params["model_category"] == "ml":
            self.train_ml(tuning=tuning)
        elif self.model_params["model_category"] == "rl":
            self.train_kmeans_q(tuning=tuning)
        # elif self.model_params["model_category"] == "cluster":
        #     self.train_kmeans()
        # elif self.model_params["model_category"] == "rl":
        #     self.train_q_learner(tuning=tuning)
        else:
            raise ValueError("Unknown model category: {}".format(self.model_params["model_category"]))
        
    def get_model(self):
        model_to_return = deepcopy(self.best_model)
        if self.model_params["model_category"] == "dl":
            if isinstance(model_to_return, list):
                # T-learner: multiple models
                for model in model_to_return:
                    model.eval()
                return [m.state_dict() for m in model_to_return]
            else:
                # S-learner: single model
                model_to_return.eval()
                return model_to_return.state_dict()
        else:
            return model_to_return
        
    # Debug: gibt alles aus, was als Trainingsdaten in das DL_Dataset geht (vor dem Aufteilen in Batches)
    def print_train_data(self, n_cases=2):
        entries = {
            "x_case": self.data_train["X_case"],
            "x_event": self.data_train["X_event"],
            "prefix_len": self.data_train["prefix_len"],
            "t": self.data_train["T"],
            "y": self.data_train["Y"],
            "weights": self.weights_train,
            "x_suffix": self.data_train.get("X_suffix"),
            "suffix_len": self.data_train.get("suffix_len"),
        }
        print("\n=== Trainingsdaten vor dem DataLoader ===")
        for name, value in entries.items():
            if value is None:
                print(f"\n{name}: None")
                continue
            # Große Tensoren kürzt PyTorch mit "..." ab
            print(f"\n{name}: Form {tuple(value.shape)}, dtype {value.dtype}")
            print(value)

        # PSP: die ersten Fälle zusätzlich als lesbare Tabellen mit Spaltennamen
        if hasattr(self.model_functions, "debug_print_batch") and entries["x_suffix"] is not None:
            self.model_functions.debug_print_batch(**entries, n_cases=n_cases)
        print("\n=== Ende Trainingsdaten ===\n")

    # Training rein für Deep Learning. Bestes Modell übergeben
    # Hier muss das PSP übergeben werden können.
    # Wir sind in dem Schritt nach dem Preprocessing. jetzt wird traniert.
    def train_dl(self):
        """
        Train either a single NN (S-learner) or multiple NNs (T-learner).
        Works with get_loss() returning a single loss or a list of per-arm losses.
        Returns train_attenuated_losses, infer_losses, infer_attenuated_losses as in the PSP trainer
        (infer = validation; the first two are only filled for the PSP).
        """

        # Sehr hilfreiche übersicht
        # Debug: die ersten Fälle der Trainingsdaten als Tabellen (debug_print_batch gibt es nur beim PSP)
        # if self.model_params["model_specific"] == "psp":
        #     self.model_functions.debug_print_batch(
        #         x_case=self.data_train["X_case"],
        #         x_event=self.data_train["X_event"],
        #         t=self.data_train["T"],
        #         prefix_len=self.data_train["prefix_len"],
        #         y=self.data_train["Y"],
        #         weights=self.weights_train,
        #         x_suffix=self.data_train.get("X_suffix"),
        #         suffix_len=self.data_train.get("suffix_len"),
        #         n_cases=2,
        #     )

        # === Data loaders ===
        # DATEN LADEN UND VORBEREITEN ZUM TRAINING
        self.data_loader_train = data.DataLoader(
            DL_Dataset(
                x_case=self.data_train["X_case"],
                x_event=self.data_train["X_event"],
                prefix_len=self.data_train["prefix_len"],
                t=self.data_train["T"],
                y=self.data_train["Y"],
                weights=self.weights_train,
                x_suffix=self.data_train.get("X_suffix"), # Für PSP Training hinzugefügt
                suffix_len=self.data_train.get("suffix_len"), # Für PSP Training hinzugefügt
            ),
            shuffle=True,
            batch_size=self.model_params["batch_size"],
        )

        self.data_loader_infer = data.DataLoader(
            DL_Dataset(
                x_case=self.data_infer["X_case"],
                x_event=self.data_infer["X_event"],
                prefix_len=self.data_infer["prefix_len"],
                t=self.data_infer["T"],
                y=self.data_infer["Y"],
                weights=self.weights_infer,
                x_suffix=self.data_infer.get("X_suffix"),
                suffix_len=self.data_infer.get("suffix_len"),
            ),
            shuffle=False,
            # WICHTIG
            # Hier wird es auf 128er inputs transformiert
            batch_size=self.model_params["batch_size"],
        )

        # === Optimizer(s) ===
        # Optimieren der Parameter und Merken der beste Parameter, um das beste modell zu erhalten
        if isinstance(self.model_functions.model, list):
            # T-learner: one optimizer per model
            self.optims = [
                torch.optim.Adam(
                    model_i.parameters(),
                    lr=self.model_params["lr"],
                    weight_decay=self.model_params["weight_decay"],
                )
                for model_i in self.model_functions.model
            ]
        else:
            # S-learner: single optimizer
            self.optim = torch.optim.Adam(
                self.model_functions.model.parameters(),
                lr=self.model_params["lr"],
                weight_decay=self.model_params["weight_decay"],
            )

        # === Scheduler (nur PSP) ===
        # Wie im PSP-Trainer: Learning Rate senken, wenn sich der Val-Loss nicht mehr verbessert.
        # Ein Schritt pro Evaluation, die Patience zählt also in Evaluationen (alle eval_every Epochen).
        self.scheduler = None
        if self.model_params["model_specific"] == "psp" and self.model_params.get("psp_use_scheduler", False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode="min",
                factor=self.model_params.get("psp_scheduler_factor", 0.1),
                patience=self.model_params.get("psp_scheduler_patience", 15),
                min_lr=self.model_params.get("psp_scheduler_min_lr", 1e-8),
            )

        # === Tensorboard writer (nur PSP) ===
        # Wie im PSP-Trainer. Jedes Training bekommt einen eigenen Ordner, benannt wie der Checkpoint plus Zeitstempel.
        is_psp = self.model_params["model_specific"] == "psp"
        self.writer = None
        if is_psp and self.model_params.get("psp_use_tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter
            run_name = os.path.basename(self.savepath_checkpoint)[:-len(".pt")] + "_" + time.strftime("%Y%m%d-%H%M%S")
            self.tensorboard_path = os.path.join(self.model_params["psp_tensorboard_dir"], run_name)
            self.writer = SummaryWriter(log_dir=self.tensorboard_path)
            print(f"    Tensorboard: {self.tensorboard_path}")
        # Nur PSP: zusätzlich den aktuellen Stand alle n Epochen speichern (0 = aus). Das beste Modell liegt immer im Checkpoint.
        save_model_n_th_epoch = self.model_params.get("psp_save_model_n_th_epoch", 0) if is_psp else 0
        self.saving_path = self.savepath_checkpoint[:-len(".pt")] + "_last.pt"

        # === Tracking variables ===
        self.losses_train = []
        self.losses_infer = []
        # Nur PSP: mittlerer Trainings-Loss je Epoche wie im PSP-Trainer
        self.train_attenuated_losses = []
        # Verläufe je Evaluation wie im PSP-Trainer (infer = Validierung):
        # infer_losses = Standard-Loss ohne Unsicherheit (nur PSP), infer_attenuated_losses = der Val-Loss
        self.infer_losses = []
        self.infer_attenuated_losses = self.losses_infer
        self.best_loss_infer = float("inf")
        self.best_model = None
        # Nur PSP: Losses je Attribut wie im PSP-Trainer, aufsummiert über die Batches
        epoch_cat_loss = {}
        epoch_num_loss = {}

        epoch_loss = 0.0
        num_train_batches = 0
        best_epoch = 0

        # Ausgabe während des Trainings: einmal zum Start, dann bei jeder Evaluation eine Zeile
        n_infer = len(self.data_loader_infer.dataset)
        print(f"    Training {self.model_params['model_specific']} (Stage {self.model_params['stage']}, Ziel {self.model_params['target']}): "
              f"{len(self.data_loader_train.dataset)} Train / {n_infer} Val, {self.model_params['num_epochs']} Epochen, Evaluation alle {self.model_params['eval_every']}")

        for epoch in tqdm(range(self.model_params["num_epochs"]), disable=True):
            if is_psp and self.model_params.get("psp_use_scheduled_sampling", False):
                # Reduce Teacher forcing ratio dynamically (scheduled sampling), wie im PSP-Trainer
                self.model_functions.teacher_forcing_ratio = max(
                    self.model_params.get("psp_min_teacher_forcing_value", 0.0),
                    self.model_params.get("psp_max_teacher_forcing_value", 1.0) - epoch / (self.model_params["num_epochs"] * 0.5))

            # === TRAINING ===
            n_batches_before = len(self.losses_train)
            for x_case, x_event, prefix_len, t, y, weights, x_suffix, suffix_len in self.data_loader_train:
                # Nur das PSP bekommt die Suffixe, die anderen Modelle kennen diese Argumente nicht
                suffix_kwargs = {"x_suffix": x_suffix, "suffix_len": suffix_len} if self.model_params["model_specific"] == "psp" else {}
                # Nur das PSP gibt zusätzlich die Losses je Attribut zurück
                parts_kwargs = {"return_parts": True} if self.model_params["model_specific"] == "psp" else {}
                if isinstance(self.model_functions.model, list):
                    # T-learner: compute per-arm losses
                    # Welches Netz genau trainiert wird, wird über get_model_functions(model_params) entschieden
                    # Hier muss dann das PSP wählbar gemacht werden über die paramter
                    # Steckt also in model_functions
                    # ------------------WICHTIG--------------------------
                    losses = self.model_functions.get_loss(
                        x_case=x_case, x_event=x_event, prefix_len=prefix_len,
                        t=t, y=y, weights=weights, set_eval=False, **suffix_kwargs
                    )
                    for arm_idx, (model_i, optim_i) in enumerate(zip(self.model_functions.model, self.optims)):
                        loss_i = losses[arm_idx] if arm_idx < len(losses) else None
                        if loss_i is None:
                            continue
                        optim_i.zero_grad()
                        loss_i.backward()
                        torch.nn.utils.clip_grad_norm_(model_i.parameters(), self.model_params["grad_norm"])
                        optim_i.step()
                        self.losses_train.append(loss_i.item())
                        epoch_loss += loss_i.item()
                        num_train_batches += 1
                else:
                    # S-learner: standard single-model training
                    self.optim.zero_grad()
                    # Welches Netz genau trainiert wird, wird über get_model_functions(model_params) entschieden
                    # mode_functions entprcht hier dann dem PSP. Es wird hier also get_loss des PSP aufgerufen
                    # Hier muss dann das PSP wählbar gemacht werden über die paramter
                    # Steckt also in model_functions
                    loss = self.model_functions.get_loss(
                        x_case=x_case, x_event=x_event, prefix_len=prefix_len,
                        t=t, y=y, weights=weights, set_eval=False, **suffix_kwargs, **parts_kwargs
                    )
                    if isinstance(loss, tuple):
                        # PSP: Losses je Attribut aufsummieren
                        loss, (cat_losses_dict, num_losses_dict) = loss
                        for feature_name in cat_losses_dict.keys():
                            epoch_cat_loss[feature_name] = epoch_cat_loss.get(feature_name, 0.0) + cat_losses_dict[feature_name].item()
                        for feature_name in num_losses_dict.keys():
                            epoch_num_loss[feature_name] = epoch_num_loss.get(feature_name, 0.0) + num_losses_dict[feature_name].item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_functions.model.parameters(), self.model_params["grad_norm"])
                    self.optim.step()
                    self.losses_train.append(loss.item())
                    epoch_loss += loss.item()
                    num_train_batches += 1

            if self.model_params["model_specific"] == "psp":
                # Prints per Epoch (wie im PSP-Trainer). Ohne psp_use_scheduled_sampling bleibt die Teacher forcing ratio 1.0.
                teacher_forcing_ratio = self.model_functions.teacher_forcing_ratio
                batch_losses = self.losses_train[n_batches_before:]
                epoch_loss_train = sum(batch_losses) / max(1, len(batch_losses))
                current_lr = self.optim.param_groups[0]['lr']
                print(f"      Epoch [{epoch+1}/{self.model_params['num_epochs']}], Learning Rate: {current_lr}, Teacher forcing ratio: {teacher_forcing_ratio}")
                print(f"      Training: Avg Attenuated Training Loss: {epoch_loss_train:.4f}")
                self.train_attenuated_losses.append(epoch_loss_train)

                if self.writer is not None:
                    # Tensorboard writer:
                    # Hyperparameters
                    self.writer.add_scalars("Hyperparameter", {'Learning Rate': current_lr, 'Teacher Forcing Ratio': teacher_forcing_ratio}, epoch+1)
                    # Total losses (die Infer-Werte kommen bei jeder Evaluation dazu)
                    self.writer.add_scalars("Total Losses", {'Training Total': epoch_loss_train}, epoch+1)

            # === EVALUATION ===
            if epoch % self.model_params["eval_every"] == 0:
                with torch.no_grad():
                    loss_infer = self.evaluate_dl(data_type="infer")
                    self.losses_infer.append(loss_infer)
                    # Nur PSP: Standard-Loss ohne Unsicherheit (Cross-Entropy + MSE), siehe get_loss
                    loss_infer_std = self.last_eval_feature_losses.get("total", {}).get("std")
                    if loss_infer_std is not None:
                        self.infer_losses.append(loss_infer_std)

                    avg_train_loss = epoch_loss / max(1, num_train_batches)
                    # Take the mean losses over all batches
                    for feature_name in epoch_cat_loss.keys():
                        epoch_cat_loss[feature_name] = epoch_cat_loss[feature_name] / max(1, num_train_batches)
                    for feature_name in epoch_num_loss.keys():
                        epoch_num_loss[feature_name] = epoch_num_loss[feature_name] / max(1, num_train_batches)

                    # if self.args.wandb:
                    #     wandb.log({
                    #         "epoch": epoch,
                    #         "train_loss": avg_train_loss,
                    #         "val_loss": loss_infer,
                    #     })

                    is_best = loss_infer < self.best_loss_infer
                    if is_best:
                        # print('best validation loss:', loss_infer)
                        self.best_loss_infer = loss_infer
                        self.best_model = deepcopy(self.model_functions.model)
                        best_epoch = epoch
                        torch.save(self.best_model, self.savepath_checkpoint)

                    self.print_progress(epoch=epoch, loss_train=avg_train_loss, loss_infer=loss_infer, is_best=is_best,
                                        epoch_cat_loss=epoch_cat_loss, epoch_num_loss=epoch_num_loss)
                    if self.writer is not None:
                        self.write_tensorboard(epoch=epoch, loss_infer_std=loss_infer_std, loss_infer=loss_infer,
                                               epoch_cat_loss=epoch_cat_loss, epoch_num_loss=epoch_num_loss)
                    if self.scheduler is not None:
                        # Adjust the learning rate if necessary (wie im PSP-Trainer mit dem Standard-Loss)
                        print(f"      Validation Loss for Scheduler: {loss_infer_std:.4f}")
                        self.scheduler.step(loss_infer_std)
                    # Der Trainings-Loss der nächsten Zeile gilt nur für die Epochen seit dieser Evaluation
                    epoch_cat_loss = {}
                    epoch_num_loss = {}

                    epoch_loss = 0.0
                    num_train_batches = 0

            if save_model_n_th_epoch > 0 and (epoch + 1) % save_model_n_th_epoch == 0:
                print("      saving model")
                torch.save(self.model_functions.model, self.saving_path)

            # === EARLY STOPPING ===
            if (
                self.model_params["early_stop"]
                and self.model_params["patience"] is not None
                and len(self.losses_infer) > 0
                and (epoch - best_epoch) >= self.model_params["patience"]
            ):
                print("Early stopping triggered.")
                break

        if is_psp:
            print("    Training complete.")
            if save_model_n_th_epoch > 0:
                # Letzter Stand des Trainings
                torch.save(self.model_functions.model, self.saving_path)
                print(f"    Model saved to path: {self.saving_path}")
            # Das beste Modell wurde während des Trainings im Checkpoint gespeichert
            print(f"    Best model saved to path: {self.savepath_checkpoint}")
            if self.writer is not None:
                self.writer.close()

        print(f"    Bestes Modell: Epoche {best_epoch}, Val-Loss {self.best_loss_infer:.4f}")

        # === LOAD BEST MODEL ===
        # Bestes Model wird gesetzt
        if self.model_params["early_stop"] and self.best_model is not None:
            print("Loading best-val-loss model (early stopping checkpoint).")
            self.model_functions.model = self.best_model

        # if self.args.wandb:
        #     wandb.finish()

        # === SUFFIX-AUSWERTUNG (nur PSP) ===
        # Wie die Auswertung des PSP: Suffixe für jeden Prefix der Validierungsdaten ziehen und mit den echten vergleichen
        if is_psp and self.model_params.get("psp_evaluate_suffixes", False):
            from src.utils.model_tools.psp_evaluation import PSPSuffixEvaluation
            suffix_evaluation = PSPSuffixEvaluation(
                model=self.best_model if self.best_model is not None else self.model_functions.model,
                model_params=self.model_params,
                samples_per_case=self.model_params.get("psp_evaluation_samples_per_case", 100),
                growing_num_values=self.model_params.get("psp_growing_num_values", ["elapsed_time"]),
            )
            self.suffix_evaluation = suffix_evaluation.evaluate(self.data_infer)
            suffix_evaluation.print_summary(self.suffix_evaluation)

        return self.train_attenuated_losses, self.infer_losses, self.infer_attenuated_losses

    def write_tensorboard(self, epoch, loss_infer_std, loss_infer, epoch_cat_loss, epoch_num_loss):
        # PSP: Losses einer Evaluation wie im PSP-Trainer in Tensorboard schreiben (infer = Validierung)
        infer = self.last_eval_feature_losses

        # Total losses
        self.writer.add_scalars(
            "Total Losses",
            {
                'Standard Infer Total': loss_infer_std,
                'Uncertainty Infer Total': loss_infer
            },
            epoch+1)

        # Categorical losses (die Aktivität hat keinen Loss mit Unsicherheit)
        for feature_name in epoch_cat_loss.keys():
            safe_feature_name = feature_name.replace(':', '_').replace('/', '_')
            self.writer.add_scalars(
                "Categorical Feature Losses",
                {
                    f'Training {safe_feature_name}': epoch_cat_loss[feature_name],
                    f'Standard Infer {safe_feature_name}': infer["cat_std"][feature_name]
                },
                epoch + 1)
        # Numerical losses
        for feature_name in epoch_num_loss.keys():
            safe_feature_name = feature_name.replace(':', '_').replace('/', '_')
            self.writer.add_scalars(
                "Numerical Feature Losses",
                {
                    f'Training {safe_feature_name}': epoch_num_loss[feature_name],
                    f'Standard Infer {safe_feature_name}': infer["num_std"][feature_name],
                    f'Uncertainty Infer {safe_feature_name}': infer["num_unc"][feature_name]
                },
                epoch + 1)
        # GradNorm
        if getattr(self.model_functions, "use_gradnorm", False):
            write_weights = self.model_functions.gn_weights.tolist()
            for i, feature_name in enumerate(self.model_functions.gn_task_names):
                safe_feature_name = feature_name.replace(':', '_').replace('/', '_')
                self.writer.add_scalars(
                    "Gradnorm values",
                    {
                        f'Gradnorm Weight {safe_feature_name}': write_weights[i]
                    },
                    epoch+1)

    def evaluate_dl(self, data_type):
        """
        Evaluate model(s) on the given dataset type ("train" or "infer").
        Supports both S-learner (single model) and T-learner (list of models).
        Returns: scalar mean loss across all samples.
        """
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "infer":
            data_loader = self.data_loader_infer
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        total_loss = 0.0
        total_samples = 0
        # Nur PSP: Teil-Losses und Kennzahlen, gewichtet mit der Batchgröße
        total_parts = {}

        with torch.no_grad():
            for x_case, x_event, prefix_len, t, y, weights, x_suffix, suffix_len in data_loader:
                suffix_kwargs = {"x_suffix": x_suffix, "suffix_len": suffix_len, "return_parts": True} if self.model_params["model_specific"] == "psp" else {}
                batch_loss = self.model_functions.get_loss(
                    x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, y=y, weights=weights, set_eval=True, **suffix_kwargs
                )
                if isinstance(batch_loss, tuple):
                    batch_loss, batch_parts = batch_loss
                    for key, value in batch_parts.items():
                        total_parts[key] = total_parts.get(key, 0.0) + value * x_case.size(0)

                if isinstance(batch_loss, list):
                    # T-learner: list of per-arm losses
                    batch_total = 0.0
                    batch_count = 0
                    for arm_idx, loss_i in enumerate(batch_loss):
                        if loss_i is None:
                            continue
                        # Count how many samples belong to this arm in the batch
                        mask = (t == arm_idx).flatten()
                        n_i = mask.sum().item()
                        batch_total += loss_i.item() * n_i
                        batch_count += n_i
                    if batch_count > 0:
                        total_loss += batch_total
                        total_samples += batch_count
                else:
                    # S-learner: single loss tensor
                    batch_size = x_case.size(0)
                    total_loss += batch_loss.item() * batch_size
                    total_samples += batch_size

        # SWITCH BACK TO TRAIN MODE
        if isinstance(self.model_functions.model, list):
            [m.train() for m in self.model_functions.model]
        else:
            self.model_functions.model.train()

        averaged_parts = {key: value / total_samples for key, value in total_parts.items()} if total_samples > 0 else {}
        self.last_eval_parts = {key: value for key, value in averaged_parts.items() if "/" not in key}
        # Nur PSP: Losses je Attribut, {"cat_std": {...}, "num_std": {...}, "num_unc": {...}, "total": {"std": ...}}
        self.last_eval_feature_losses = {}
        for key, value in averaged_parts.items():
            if "/" in key:
                group, feature_name = key.split("/", 1)
                self.last_eval_feature_losses.setdefault(group, {})[feature_name] = value
        return total_loss / total_samples if total_samples > 0 else float("inf")

    def print_progress(self, epoch, loss_train, loss_infer, is_best, epoch_cat_loss=None, epoch_num_loss=None):
        # Eine Zeile pro Evaluation. Der Train-Loss enthält ggf. Regularisierer, der Val-Loss nicht.
        line = f"      Epoche {epoch:4d} | Train-Loss {loss_train:9.4f} | Val-Loss {loss_infer:9.4f}"
        if getattr(self, "scheduler", None) is not None:
            # PSP mit Scheduler: Learning Rate, mit der bis zu dieser Evaluation trainiert wurde
            line += f" | LR {self.optim.param_groups[0]['lr']:.1e}"
        parts = getattr(self, "last_eval_parts", {})
        if parts:
            # PSP (Val, Teacher Forcing): Teil-Losses und lesbare Kennzahlen, Outcome in der skalierten Einheit von Y
            line += f" | Aktivität CE {parts['activity_ce']:.4f} Acc {parts['activity_acc']:.3f}"
            for key in parts:
                # Weitere kategoriale Event-Attribute (z.B. process_type)
                if key.endswith("_ce") and key != "activity_ce":
                    line += f" | {key[:-3]} CE {parts[key]:.4f} Acc {parts[key[:-3] + '_acc']:.3f}"
            if "event_nll" in parts:
                line += f" | Attribute NLL {parts['event_nll']:.4f}"
            line += f" | Outcome NLL {parts['outcome_nll']:.4f} MAE {parts['outcome_mae']:.4f}"
        if is_best:
            line += " *"
        print(line)
        if epoch_cat_loss or epoch_num_loss:
            # PSP (Train, Teacher Forcing): Loss je Attribut, kategorial als CE, numerisch und Outcome als NLL
            feature_losses = {**(epoch_cat_loss or {}), **(epoch_num_loss or {})}
            print("                  | Train je Attribut: " + " | ".join(f"{name} {value:.4f}" for name, value in feature_losses.items()))
        if getattr(self.model_functions, "use_gradnorm", False):
            # PSP mit GradNorm: aktuelles Loss-Gewicht je Attribut
            gn_weights = zip(self.model_functions.gn_task_names, self.model_functions.gn_weights.tolist())
            print("                  | GradNorm-Gewichte: " + " | ".join(f"{name} {value:.4f}" for name, value in gn_weights))

    def train_ml(self, tuning=False):
        # Initialize data loaders NOTE: we don't use data_infer here, as it is contained in the data_train (we use cross-validation)
        self.ml_dataset = ML_Dataset(x_case=self.data_train["X_case"], x_event=self.data_train["X_event"],
                                     prefix_len=self.data_train["prefix_len"], t=self.data_train["T"],
                                     y=self.data_train["Y"], weights=self.weights_train)
        
        self.best_loss_infer = self.model_functions.fit_and_get_loss(x_case=self.ml_dataset.x_case,
                                                                     x_event=self.ml_dataset.x_event,
                                                                     prefix_len=self.ml_dataset.prefix_len,
                                                                     t=self.ml_dataset.t,
                                                                     y=self.ml_dataset.y,
                                                                     weights=self.ml_dataset.weights,
                                                                     tuning=tuning,
                                                                     dataset_ps=self.data_train_ps)
        
        # save the model
        self.best_model = deepcopy(self.model_functions.model)

    def train_kmeans_q(self, tuning=False):
        self.kmeans_q_dataset = RL_Dataset(x_case=self.data_train,
                                             x_event=None,  # RL models do not use x_event
                                             prefix_len=None,  # RL models do not use prefix_len
                                             t=None,  # RL models do not use t
                                             y=None,  # RL models do not use y
                                             weights=None)

        self.best_loss_infer = self.model_functions.fit_and_get_loss(x_case=self.kmeans_q_dataset.x_case,
                                                                     x_event=self.kmeans_q_dataset.x_event,
                                                                     prefix_len=self.kmeans_q_dataset.prefix_len,
                                                                     t=self.kmeans_q_dataset.t,
                                                                     y=self.kmeans_q_dataset.y,
                                                                     weights=self.kmeans_q_dataset.weights,
                                                                     tuning=tuning)

        self.best_model = deepcopy(self.model_functions.model)