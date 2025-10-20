
# train_utils/train_RNAGraph_graph_classification_weighted.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter
import warnings


class ClassWeightCalculator:
    def __init__(self, strategy='balanced', beta=0.999, smooth_eps=1e-9):
        self.strategy = strategy
        self.beta = beta
        self.smooth_eps = smooth_eps
        self.weights = None

    def calculate_weights(self, labels, n_classes=None):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        labels = np.array(labels)

        if n_classes is None:
            n_classes = len(np.unique(labels))

        class_counts = np.bincount(labels, minlength=n_classes)
        total_samples = len(labels)

        if self.strategy == 'balanced':
            weights = self._balanced_weights(class_counts, total_samples, n_classes)
        elif self.strategy == 'inverse':
            weights = self._inverse_weights(class_counts, total_samples)
        elif self.strategy == 'focal':
            weights = self._focal_weights(class_counts, total_samples)
        elif self.strategy == 'effective':
            weights = self._effective_weights(class_counts, total_samples, n_classes)
        else:
            warnings.warn(f"Unknown strategy: {self.strategy}, using balanced weights")
            weights = self._balanced_weights(class_counts, total_samples, n_classes)

        self.weights = torch.tensor(weights, dtype=torch.float32)
        return self.weights

    def _balanced_weights(self, class_counts, total_samples, n_classes):
        weights = total_samples / (n_classes * (class_counts + self.smooth_eps))
        return weights / np.sum(weights) * n_classes

    def _inverse_weights(self, class_counts, total_samples):
        weights = total_samples / (class_counts + self.smooth_eps)
        return weights / np.sum(weights) * len(class_counts)

    def _focal_weights(self, class_counts, total_samples):
        class_freq = class_counts / total_samples
        weights = 1 / (class_freq + self.smooth_eps)
        return weights / np.sum(weights) * len(class_counts)

    def _effective_weights(self, class_counts, total_samples, n_classes):
        effective_num = 1.0 - np.power(self.beta, class_counts)
        weights = (1.0 - self.beta) / (effective_num + self.smooth_eps)
        return weights / np.sum(weights) * n_classes


class WeightedLossFunction:
    def __init__(self, class_weights=None, alpha=0.25, gamma=2.0, reduction='mean'):
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def weighted_binary_cross_entropy(self, logits, targets, pos_weight=None):
        if pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=pos_weight, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction=self.reduction
            )

    def weighted_cross_entropy(self, logits, targets):
        if self.class_weights is not None:
            return F.cross_entropy(
                logits, targets, weight=self.class_weights, reduction=self.reduction
            )
        else:
            return F.cross_entropy(logits, targets, reduction=self.reduction)

    def focal_loss(self, logits, targets):
        if logits.shape[-1] == 1:
            return self._binary_focal_loss(logits, targets)
        else:
            return self._multiclass_focal_loss(logits, targets)

    def _binary_focal_loss(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

    def _multiclass_focal_loss(self, logits, targets):
        log_softmax = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_softmax, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def calculate_dataset_statistics(data_loader):
    all_labels = []
    for _, batch_labels in data_loader:
        all_labels.extend(batch_labels.cpu().numpy())

    labels = np.array(all_labels)
    n_classes = len(np.unique(labels))
    class_counts = np.bincount(labels, minlength=n_classes)
    total_samples = len(labels)

    imbalance_ratio = class_counts.max() / (class_counts.min() + 1e-9)

    stats = {
        'n_classes': n_classes,
        'class_counts': class_counts,
        'total_samples': total_samples,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': class_counts / total_samples
    }

    return stats


def train_epoch_sparse_weighted(model, optimizer, device, data_loader, epoch,
                                loss_strategy='weighted', class_weights=None,
                                alpha=0.25, gamma=2.0):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0

    all_logits = []
    all_labels = []

    if hasattr(model, 'n_classes'):
        n_classes = model.n_classes
    else:
        with torch.no_grad():
            try:
                sample_input = next(iter(data_loader))[0][0].to(device)
                sample_output = model(sample_input.unsqueeze(0))
                n_classes = sample_output.size(1) if sample_output.dim() > 1 else 1
            except:
                n_classes = 1

    loss_manager = WeightedLossFunction(
        class_weights=class_weights.to(device) if class_weights is not None else None,
        alpha=alpha,
        gamma=gamma
    )

    if n_classes == 1:
        pos_count = 0
        total_count = 0
        for _, batch_labels in data_loader:
            pos_count += batch_labels.sum().item()
            total_count += len(batch_labels)
        pos_weight = torch.tensor([(total_count - pos_count) / (pos_count + 1e-9)], dtype=torch.float32).to(device)
    else:
        pos_weight = None

    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        nb_data += batch_labels.size(0)

        optimizer.zero_grad()

        logits = model(batch_graphs)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN or Inf detected at iteration {iter}. Skipping batch.")
            continue

        if loss_strategy == 'focal':
            loss = loss_manager.focal_loss(logits, batch_labels)
        elif loss_strategy == 'weighted':
            if n_classes == 1:
                loss = loss_manager.weighted_binary_cross_entropy(logits, batch_labels, pos_weight)
            else:
                loss = loss_manager.weighted_cross_entropy(logits, batch_labels)
        else:
            if n_classes == 1:
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels.float())
            else:
                loss = F.cross_entropy(logits, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.detach().item()

        if n_classes == 1:
            preds = torch.sigmoid(logits).round().detach().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        acc = accuracy_score(batch_labels.cpu().numpy(), preds)
        epoch_train_acc += acc

        all_logits.append(logits.detach().cpu())
        all_labels.append(batch_labels.cpu())

    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    try:
        if n_classes == 1:
            epoch_train_auc = roc_auc_score(
                all_labels.numpy(),
                torch.sigmoid(all_logits).numpy().squeeze()
            )
        else:
            if n_classes == 2:
                probs = F.softmax(all_logits, dim=1)[:, 1].numpy()
                epoch_train_auc = roc_auc_score(all_labels.numpy(), probs)
            else:
                epoch_train_auc = roc_auc_score(
                    all_labels.numpy(),
                    F.softmax(all_logits, dim=1).numpy(),
                    multi_class='ovr'
                )
    except:
        epoch_train_auc = 0.0

    return epoch_loss, epoch_train_acc, epoch_train_auc, optimizer


//def evaluate_network_sparse_weighted(model, device, data_loader, epoch, class_weights=None):
//    pass


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    return train_epoch_sparse_weighted(model, optimizer, device, data_loader, epoch,
                                       loss_strategy='standard')


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    all_logits = []
    all_labels = []

    if hasattr(model, 'n_classes'):
        n_classes = model.n_classes
    else:
        with torch.no_grad():
            try:
                sample_input = next(iter(data_loader))[0][0].to(device)
                sample_output = model(sample_input.unsqueeze(0))
                n_classes = sample_output.size(1) if sample_output.dim() > 1 else 1
            except:
                n_classes = 1

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_graphs)

            if n_classes == 1:
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels.float())
            else:
                loss = F.cross_entropy(logits, batch_labels)

            epoch_test_loss += loss.detach().item()

            if n_classes == 1:
                preds = torch.sigmoid(logits).round().detach().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            acc = accuracy_score(batch_labels.cpu().numpy(), preds)
            epoch_test_acc += acc

            all_logits.append(logits.detach().cpu())
            all_labels.append(batch_labels.cpu())

    epoch_test_loss /= (iter + 1)
    epoch_test_acc /= (iter + 1)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    try:
        if n_classes == 1:
            epoch_test_auc = roc_auc_score(
                all_labels.numpy(),
                torch.sigmoid(all_logits).numpy().squeeze()
            )
        else:
            if n_classes == 2:
                probs = F.softmax(all_logits, dim=1)[:, 1].numpy()
                epoch_test_auc = roc_auc_score(all_labels.numpy(), probs)
            else:
                epoch_test_auc = roc_auc_score(
                    all_labels.numpy(),
                    F.softmax(all_logits, dim=1).numpy(),
                    multi_class='ovr'
                )
    except:
        epoch_test_auc = 0.0

    return epoch_test_loss, epoch_test_acc, epoch_test_auc
