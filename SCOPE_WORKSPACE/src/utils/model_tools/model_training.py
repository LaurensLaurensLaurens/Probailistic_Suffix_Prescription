import random
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

class ModelTrainer():
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
        self.model_functions = get_model_functions(model_params=self.model_params)

        random.seed(self.model_params["seed"])
        torch.manual_seed(self.model_params["seed"])
        torch.cuda.manual_seed_all(self.model_params["seed"])
        np.random.seed(self.model_params["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def train(self, tuning=False):
        # if self.args.wandb:
        #     wandb.login(key="")

        #     wandb.init(
        #         project="CLIPPS",      # e.g. "treatment-effect-nn"
        #         config=self.model_params,         # logs your hyperparameters
        #         name=self.model_params["method"],           # optional, for clarity
        #     )

        if self.model_params["model_category"] == "dl":
            self.train_dl()
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
        
    def train_dl(self):
        """
        Train either a single NN (S-learner) or multiple NNs (T-learner).
        Works with get_loss() returning a single loss or a list of per-arm losses.
        """
        # === Data loaders ===
        self.data_loader_train = data.DataLoader(
            DL_Dataset(
                x_case=self.data_train["X_case"],
                x_event=self.data_train["X_event"],
                prefix_len=self.data_train["prefix_len"],
                t=self.data_train["T"],
                y=self.data_train["Y"],
                weights=self.weights_train,
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
            ),
            shuffle=False,
            batch_size=self.model_params["batch_size"],
        )

        # === Optimizer(s) ===
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

        # === Tracking variables ===
        self.losses_train = []
        self.losses_infer = []
        self.best_loss_infer = float("inf")
        self.best_model = None
        epoch_train_loss = 0.0
        num_train_batches = 0
        best_epoch = 0

        for epoch in tqdm(range(self.model_params["num_epochs"]), disable=True):
            # === TRAINING ===
            for x_case, x_event, prefix_len, t, y, weights in self.data_loader_train:
                if isinstance(self.model_functions.model, list):
                    # T-learner: compute per-arm losses
                    losses = self.model_functions.get_loss(
                        x_case=x_case, x_event=x_event, prefix_len=prefix_len,
                        t=t, y=y, weights=weights, set_eval=False
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
                        epoch_train_loss += loss_i.item()
                        num_train_batches += 1
                else:
                    # S-learner: standard single-model training
                    self.optim.zero_grad()
                    loss = self.model_functions.get_loss(
                        x_case=x_case, x_event=x_event, prefix_len=prefix_len,
                        t=t, y=y, weights=weights, set_eval=False
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_functions.model.parameters(), self.model_params["grad_norm"])
                    self.optim.step()
                    self.losses_train.append(loss.item())
                    epoch_train_loss += loss.item()
                    num_train_batches += 1

            # === EVALUATION ===
            if epoch % self.model_params["eval_every"] == 0:
                with torch.no_grad():
                    loss_infer = self.evaluate_dl(data_type="infer")
                    self.losses_infer.append(loss_infer)

                    avg_train_loss = epoch_train_loss / max(1, num_train_batches)

                    # if self.args.wandb:
                    #     wandb.log({
                    #         "epoch": epoch,
                    #         "train_loss": avg_train_loss,
                    #         "val_loss": loss_infer,
                    #     })

                    if loss_infer < self.best_loss_infer:
                        # print('best validation loss:', loss_infer)
                        self.best_loss_infer = loss_infer
                        self.best_model = deepcopy(self.model_functions.model)
                        best_epoch = epoch
                        torch.save(self.best_model, self.savepath_checkpoint)

            # === EARLY STOPPING ===
            if (
                self.model_params["early_stop"]
                and self.model_params["patience"] is not None
                and len(self.losses_infer) > 0
                and (epoch - best_epoch) >= self.model_params["patience"]
            ):
                print("Early stopping triggered.")
                break

        # === LOAD BEST MODEL ===
        if self.model_params["early_stop"] and self.best_model is not None:
            print("Loading best-val-loss model (early stopping checkpoint).")
            self.model_functions.model = self.best_model

        # if self.args.wandb:
        #     wandb.finish()

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

        with torch.no_grad():
            for x_case, x_event, prefix_len, t, y, weights in data_loader:
                batch_loss = self.model_functions.get_loss(
                    x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, y=y, weights=weights, set_eval=True
                )

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

        return total_loss / total_samples if total_samples > 0 else float("inf")

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