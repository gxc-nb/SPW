import os
import numpy as np
# import wandb   # keep wandb import commented for now
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


class RewardModel:
    def __init__(self, config, obs_act_1, obs_act_2, attention_1, attention_2, labels, dimension):
        self.env = config.env
        self.config = config
        self.dimension = dimension
        self.device = config.device
        self.obs_act_1 = obs_act_1
        self.obs_act_2 = obs_act_2
        self.attention_1 = attention_1
        self.attention_2 = attention_2
        self.labels = labels
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.activation = config.activation
        self.segment_size = config.segment_size
        self.lr = config.lr
        self.hidden_sizes = config.hidden_sizes
        self.loss = None
        self.model_type = config.model_type
        if self.model_type == "BT":
            self.loss = self.BT_loss
        elif self.model_type == "linear_BT":
            self.loss = self.linear_BT_loss
        self.ensemble_num = config.ensemble_num
        self.ensemble_method = config.ensemble_method
        self.paramlist = []
        self.optimizer = []
        self.lr_scheduler = []
        self.net = None
        self.ensemble_model = None
        self.spw_tau = config.spw_tau
        self.rd_gamma = config.rd_gamma

    def save_test_dataset(self, test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels):
        """Store test dataset for later evaluation"""
        self.test_obs_act_1 = torch.from_numpy(test_obs_act_1).float().to(self.device)
        self.test_obs_act_2 = torch.from_numpy(test_obs_act_2).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels).float().to(self.device)
        self.test_binary_labels = torch.from_numpy(test_binary_labels).float().to(self.device)

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=2):
        """Build a simple MLP for reward modeling"""
        net = []
        for _ in range(n_layers):
            net.append(nn.Linear(in_dim, H))
            net.append(nn.LeakyReLU())
            in_dim = H
        net.append(nn.Linear(H, out_dim))

        if self.activation == "tanh":
            net.append(nn.Tanh())
        elif self.activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif self.activation == "relu":
            net.append(nn.ReLU())
        elif self.activation == "leaky_relu":
            net.append(nn.LeakyReLU())
        elif self.activation == "gelu":
            net.append(nn.GELU())

        return nn.Sequential(*net)

    def construct_ensemble(self):
        """Build an ensemble of reward networks"""
        ensemble_model = []
        for _ in range(self.ensemble_num):
            ensemble_model.append(
                self.model_net(in_dim=self.dimension, out_dim=1, H=self.hidden_sizes).to(self.device)
            )
        return ensemble_model

    def single_model_forward(self, obs_act):
        return self.net(obs_act)

    def ensemble_model_forward(self, obs_act):
        """Forward pass through all ensemble members and combine predictions"""
        pred = []
        for i in range(self.ensemble_num):
            pred.append(self.ensemble_model[i](obs_act))
        pred = torch.stack(pred, dim=1)
        if self.ensemble_method == "mean":
            return torch.mean(pred, dim=1)
        elif self.ensemble_method == "min":
            return torch.min(pred, dim=1).values
        elif self.ensemble_method == "uwo":
            return torch.mean(pred, dim=1) - 5 * torch.std(pred, dim=1)

    def BT_loss(self, pred_hat, label):
        logprobs = F.log_softmax(pred_hat, dim=1)
        return -(label * logprobs).sum()

    def linear_BT_loss(self, pred_hat, label):
        pred_hat += self.segment_size + 1e-5
        pred_prob = pred_hat / torch.sum(pred_hat, dim=1, keepdim=True)
        loss = -torch.sum(label * torch.log(pred_prob), dim=1)
        return torch.sum(loss)

    def save_model(self, path):
        """Save ensemble model members"""
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, f"reward_{member}.pt")
            torch.save(self.ensemble_model[member].state_dict(), member_path)

    def load_model(self, path):
        """Load ensemble model members"""
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, f"reward_{member}.pt")
            self.ensemble_model[member].load_state_dict(torch.load(member_path))

    def get_reward(self, dataset):
        """Compute rewards for dataset using ensemble"""
        obs = dataset["observations"]
        act = dataset["actions"]
        obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000: (i + 1) * 10000]
                pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                dataset["rewards"][i * 10000: (i + 1) * 10000] = pred_batch.cpu().numpy()
        return dataset["rewards"]

    def eval(self, obs_act_1, obs_act_2, labels, binary_labels, name, epoch):
        """Evaluate ensemble on validation or test set"""
        eval_acc = 0
        eval_loss = 0
        for member in range(self.ensemble_num):
            self.ensemble_model[member].eval()
        with torch.no_grad():
            for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                obs_act_1_batch = obs_act_1[batch * self.batch_size: (batch + 1) * self.batch_size]
                obs_act_2_batch = obs_act_2[batch * self.batch_size: (batch + 1) * self.batch_size]
                labels_batch = labels[batch * self.batch_size: (batch + 1) * self.batch_size]
                binary_labels_batch = binary_labels[batch * self.batch_size: (batch + 1) * self.batch_size]

                pred_1 = self.ensemble_model_forward(obs_act_1_batch)
                pred_2 = self.ensemble_model_forward(obs_act_2_batch)

                pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)

                pred_labels = torch.argmax(pred_hat, dim=-1)
                eval_acc += torch.sum(pred_labels == torch.argmax(binary_labels_batch, dim=-1)).item()
                eval_loss += self.loss(pred_hat, labels_batch).item()

        eval_loss /= obs_act_1.shape[0]
        eval_acc /= float(obs_act_1.shape[0])
        # wandb.log({name + "/loss": eval_loss, name + "/acc": eval_acc}, step=epoch)

    def train_model(self, mode="MR"):
        """Train the reward model"""
        if self.ensemble_model is None:
            self.ensemble_model = self.construct_ensemble()

        self.optimizer = []
        self.lr_scheduler = []

        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(optim.Adam(self.ensemble_model[member].parameters(), lr=self.lr))
            self.lr_scheduler.append(
                optim.lr_scheduler.StepLR(
                    self.optimizer[member],
                    step_size=10 if self.epochs <= 500 else 1000,
                    gamma=0.9,
                )
            )

        self.obs_act_1 = torch.from_numpy(self.obs_act_1).float().to(self.device)
        self.obs_act_2 = torch.from_numpy(self.obs_act_2).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).float().to(self.device)
        self.attention_1 = torch.from_numpy(self.attention_1).float().to(self.device)
        self.attention_2 = torch.from_numpy(self.attention_2).float().to(self.device)

        print("Start training with mode:", mode)

        for epoch in tqdm.tqdm(range(self.epochs)):
            train_loss = 0
            train_redistribution_loss = 0

            for member in range(self.ensemble_num):
                total_loss = 0
                self.optimizer[member].zero_grad()
                self.net = self.ensemble_model[member]

                idx = np.random.permutation(self.obs_act_1.shape[0])
                obs_act_1 = self.obs_act_1[idx]
                obs_act_2 = self.obs_act_2[idx]
                labels = self.labels[idx]
                attention_1 = self.attention_1[idx]
                attention_2 = self.attention_2[idx]

                for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                    obs_act_1_batch = obs_act_1[batch * self.batch_size: (batch + 1) * self.batch_size]
                    obs_act_2_batch = obs_act_2[batch * self.batch_size: (batch + 1) * self.batch_size]
                    labels_batch = labels[batch * self.batch_size: (batch + 1) * self.batch_size]
                    attention_1_batch = attention_1[batch * self.batch_size: (batch + 1) * self.batch_size]
                    attention_2_batch = attention_2[batch * self.batch_size: (batch + 1) * self.batch_size]

                    pred_1 = self.single_model_forward(obs_act_1_batch)
                    pred_2 = self.single_model_forward(obs_act_2_batch)

                    if mode in ["MR", "BC-P", "R-P", "D-REX"]:
                        pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                        pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                        pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)
                        total_loss = self.loss(pred_hat, labels_batch) / labels_batch.shape[0]

                    elif mode == "SPW":
                        weights_1 = F.softmax(attention_1_batch.squeeze(-1) / self.spw_tau, dim=1).unsqueeze(-1)
                        weights_2 = F.softmax(attention_2_batch.squeeze(-1) / self.spw_tau, dim=1).unsqueeze(-1)
                        weighted_sum_1 = torch.sum(pred_1 * weights_1, dim=1)
                        weighted_sum_2 = torch.sum(pred_2 * weights_2, dim=1)
                        pred_hat = torch.cat([weighted_sum_1, weighted_sum_2], dim=-1)
                        total_loss = self.loss(pred_hat, labels_batch) / labels_batch.shape[0]

                    elif mode == "RD":
                        weights_1 = F.softmax(attention_1_batch.squeeze(-1), dim=1).unsqueeze(-1)
                        weights_2 = F.softmax(attention_2_batch.squeeze(-1), dim=1).unsqueeze(-1)
                        pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                        pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                        pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)
                        preference_loss = self.loss(pred_hat, labels_batch) / labels_batch.shape[0]
                        target_pred_1 = pred_seg_sum_1.unsqueeze(1) * weights_1
                        target_pred_2 = pred_seg_sum_2.unsqueeze(1) * weights_2
                        redistribution_loss = (
                            F.mse_loss(pred_1, target_pred_1) +
                            F.mse_loss(pred_2, target_pred_2)
                        ) / 2.0
                        total_loss = preference_loss + self.rd_gamma * redistribution_loss
                        train_redistribution_loss += self.rd_gamma * redistribution_loss.item()

                    total_loss.backward()
                    self.optimizer[member].step()
                    train_loss += total_loss.item()

                self.lr_scheduler[member].step()

            train_loss /= self.ensemble_num

            if epoch % 20 == 0:
                self.eval(self.obs_act_1, self.obs_act_2, self.labels, self.labels, "train_eval", epoch)
                self.eval(self.test_obs_act_1, self.test_obs_act_2, self.test_labels, self.test_binary_labels, "test_eval", epoch)

    def pretrain_with_attention(self, dataset, config):
        """Pretrain reward model using attention as supervision"""
        print("Start pretraining with attention...")
        self.ensemble_model = self.construct_ensemble()
        self.optimizer = []
        self.lr_scheduler = []

        pretrain_lr = self.lr
        best_loss = float('inf')
        best_models = None
        patience = 5
        no_improve = 0

        obs_act = np.concatenate((dataset["observations"], dataset["actions"]), axis=-1)
        attention = dataset["attention"]

        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(optim.Adam(self.ensemble_model[member].parameters(), lr=pretrain_lr))
            self.lr_scheduler.append(optim.lr_scheduler.StepLR(self.optimizer[member], step_size=1, gamma=0.9))

        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        attention = torch.from_numpy(attention).float().to(self.device)

        for epoch in tqdm.tqdm(range(20)):
            train_loss = 0
            for member in range(self.ensemble_num):
                self.net = self.ensemble_model[member]
                epoch_loss = 0

                idx = np.random.permutation(obs_act.shape[0])
                obs_act_shuffled = obs_act[idx]
                attention_shuffled = attention[idx]

                for batch in range((obs_act.shape[0] - 1) // self.batch_size + 1):
                    self.optimizer[member].zero_grad()
                    obs_act_batch = obs_act_shuffled[batch * self.batch_size: (batch + 1) * self.batch_size]
                    attention_batch = attention_shuffled[batch * self.batch_size: (batch + 1) * self.batch_size]

                    pred = self.single_model_forward(obs_act_batch)
                    pred = pred.view(-1)
                    attention_batch = attention_batch.view(-1)
                    loss = F.mse_loss(pred, attention_batch)

                    loss.backward()
                    self.optimizer[member].step()
                    epoch_loss += loss.item()

                train_loss += epoch_loss
                self.lr_scheduler[member].step()

            train_loss /= self.ensemble_num

            self.eval(self.test_obs_act_1, self.test_obs_act_2, self.test_labels, self.test_binary_labels,
                      "pretraining_test_eval", epoch)

            # Early stopping check
            if train_loss < best_loss:
                best_loss = train_loss
                best_models = [model.state_dict() for model in self.ensemble_model]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_models is not None:
            for member in range(self.ensemble_num):
                self.ensemble_model[member].load_state_dict(best_models[member])

        print(f"Pretraining finished! Best loss: {best_loss:.6f}")
