#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import os
import sys
import time
import random
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from thop import profile as thop_profile

from data_loader import read_train_data, read_test_data
from mynet075 import mynet
from sht import n_sh_from_order
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def timestamp_tag() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def setup_logger(log_file: str = "log/output.log") -> logging.Logger:
    os.makedirs(Path(log_file).parent, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def set_seed(seed: int = 300, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def infer_split_tag_from_h5(h5_path: str) -> str:
    name = Path(h5_path).name.lower()
    if 'all' in name:
        return 'ALL'
    m = re.search(r'(\d+)\s*ft', name)
    if m:
        return f"{m.group(1)}ft"
    return 'UNK'

def build_save_path(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"mynet_{tag}.pth")

class Config:
    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 8,
        epochs: int = 50,
        lr: float = 1e-3,
        save_path: str  = "None",
        device_num: int = 0,
        train_h5: str = '../data/9ft_train.h5',
        test_h5: str = '../data/9ft_test.h5',
        run_for: str = 'auto',
        num_classes: int = 7,


        use_sht: bool = True,
        L: int = 4,
        sht_representation: str = 'power',

    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.device_num = device_num
        self.train_h5 = train_h5
        self.test_h5 = test_h5
        self.run_for = run_for
        self.num_classes = num_classes


        self.use_sht = use_sht
        self.L = L
        self.sht_representation = sht_representation

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, use_cuda: bool = True):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device if use_cuda else None)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_dataloader, optimizer, epoch, writer, device, loss_fn, alpha=0.5, logger=None) -> float:
    model.train()
    correct = 0
    cls_loss_sum = 0.0

    for data, target in train_dataloader:
        target = target.view(-1).long()
        data, target = data.to(device), target.to(device)

        data, target_a, target_b, lam = mixup_data(
            data, target, alpha=alpha, use_cuda=torch.cuda.is_available()
        )

        optimizer.zero_grad()
        logits = model(data)
        loss_batch = mixup_criterion(loss_fn, logits, target_a, target_b, lam)
        loss_batch.backward()
        optimizer.step()

        cls_loss_sum += loss_batch.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += (lam * pred.eq(target_a.view_as(pred)).sum().item()
                    + (1 - lam) * pred.eq(target_b.view_as(pred)).sum().item())

    cls_loss_avg = cls_loss_sum / max(1, len(train_dataloader))
    acc = 100.0 * correct / max(1, len(train_dataloader.dataset))
    if logger: logger.info(f"Train Epoch {epoch}: loss={cls_loss_avg:.6f}, acc={acc:.2f}%")

    writer.add_scalar('train/acc', acc, epoch)
    writer.add_scalar('train/loss', cls_loss_avg, epoch)

    os.makedirs('log', exist_ok=True)
    with open('log/train_log.txt', 'a', encoding='utf-8') as f:
        f.write(f'epoch: {epoch} Accuracy: {acc:.2f} Loss: {cls_loss_avg:.6f}\n')

    return cls_loss_avg

@torch.no_grad()
def evaluate_epoch(model, loss_fn, val_dataloader, epoch, writer, device, logger=None) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total_batches = len(val_dataloader)
    empty_batches = 0

    for data, target in val_dataloader:
        if len(target) == 0:
            empty_batches += 1
            continue

        target = target.view(-1).long()
        data, target = data.to(device), target.to(device)
        logits = model(data)
        val_loss += loss_fn(logits, target).item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    if empty_batches > 0 and logger:
        logger.warning(f"Found {empty_batches}/{total_batches} empty batches in validation!")

    val_loss /= max(1, len(val_dataloader))
    acc = 100.0 * correct / max(1, len(val_dataloader.dataset))

    if logger: logger.info(f'Val: loss={val_loss:.6f}, acc={acc:.2f}%')

    writer.add_scalar('val/acc', acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)

    os.makedirs('log', exist_ok=True)
    with open('log/val_log.txt', 'a', encoding='utf-8') as f:
        f.write(f'epoch: {epoch} Accuracy: {acc:.2f} Loss: {val_loss:.6f}\n')

    return val_loss, acc

def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader,
                       optimizer, epochs, writer, save_path, device, logger=None):
    os.makedirs(Path(save_path).parent, exist_ok=True)

    train_losses = []
    val_losses = []
    current_min_val_loss = float('inf')

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, epoch, writer, device, loss_function, alpha=0.5, logger=logger)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate_epoch(model, loss_function, val_dataloader, epoch, writer, device, logger=logger)
        val_losses.append(val_loss)

        if val_loss < current_min_val_loss:
            if logger: logger.info(f"↓ New best val loss: {current_min_val_loss:.6f} -> {val_loss:.6f} (model saved)")
            current_min_val_loss = val_loss
            torch.save(model, save_path)
        else:
            if logger: logger.info("Val loss not improved.")

        if logger:
            logger.info(f"time/epoch: {time.time() - t_start:.3f}s")
            logger.info("-" * 100)

    if logger: logger.info(f"Avg time/epoch: {(time.time() - t0) / max(1, epochs):.3f}s")
    return train_losses, val_losses

@torch.no_grad()
def test(model, test_dataloader, device=None, logger=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0
    feats_list = []
    labels_list = []
    pred_list = []
    real_list = []
    t_sum = 0.0
    iters = 0

    for data, target in test_dataloader:
        t0 = time.time()
        target = target.long()
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(data)
        pred   = logits.argmax(dim=1)

        correct += (pred == target).sum().item()
        total   += target.size(0)

        feats_list.append(logits.detach().cpu().numpy())
        labels_list.append(target.detach().cpu().numpy())
        pred_list.extend(pred.detach().cpu().tolist())
        real_list.extend(target.detach().cpu().tolist())

        t_sum += (time.time() - t0)
        iters += 1

    avg_time = t_sum / max(iters, 1)
    acc = correct / max(total, 1)

    if logger:
        logger.info(f"Avg infer time/batch: {avg_time:.6f}s")
        logger.info(f"[TEST] acc={acc*100:.2f}% ({correct}/{total})")

    all_features = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, ), dtype=np.float32)
    all_labels   = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0, ), dtype=np.int64)

    return np.array(pred_list), np.array(real_list), all_features, all_labels, acc

def run_train_then_test_once(conf: Config, train_h5: str, test_h5: str, device: torch.device) -> float:
    split_tag = infer_split_tag_from_h5(train_h5)
    save_path = conf.save_path if conf.save_path != "None" else build_save_path('model_weight', split_tag)

    tstamp = timestamp_tag()
    tb_dir = f'logs/{split_tag}_{tstamp}'
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    logger = setup_logger(log_file=f'log/{split_tag}_{tstamp}.log')

    logger.info(f"==== [{split_tag}] RUN START ====")
    logger.info(f"train_h5={train_h5} | test_h5={test_h5}")
    logger.info(f"save_path={save_path}")


    X_tr, X_va, Y_tr, Y_va = read_train_data(file_path=train_h5,
                                             use_sht=conf.use_sht,
                                             L=conf.L,
                                             sht_representation=conf.sht_representation)
    X_te, Y_te = read_test_data(file_path=test_h5,
                                use_sht=conf.use_sht,
                                L=conf.L,
                                sht_representation=conf.sht_representation)


    if conf.use_sht:
        M = n_sh_from_order(conf.L)
        in_ch = (2 * M) if conf.sht_representation == 'complex' else M
    else:
        in_ch = 1

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(Y_tr, dtype=torch.long).view(-1))
    val_ds   = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                             torch.tensor(Y_va, dtype=torch.long).view(-1))
    test_ds  = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                             torch.tensor(Y_te, dtype=torch.long).view(-1))

    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=conf.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=conf.test_batch_size, shuffle=False)

    model = mynet(num_classes=conf.num_classes, in_ch=in_ch)
    if torch.cuda.is_available():
        model = model.to(device)

    try:
        sample = torch.tensor(X_tr[:1], dtype=torch.float32).to(device)
        flops, params = thop_profile(model, inputs=(sample,), verbose=False)
        logger.info(f'[THOP] FLOPs={flops}  Params={params}')
    except Exception as e:
        logger.info(f'[THOP warn] {e}')
    try:
        if isinstance(X_tr, np.ndarray):
            if X_tr.ndim == 4:
                _, C_in, H, W = X_tr.shape
                summary(model, input_size=(C_in, H, W))
            elif X_tr.ndim == 3:
                _, H, W = X_tr.shape
                summary(model, input_size=(1, H, W))
    except Exception:
        pass

    loss_fn = nn.CrossEntropyLoss().to(device) if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0.0)

    t0 = time.time()
    train_losses, val_losses = train_and_evaluate(
        model, loss_function=loss_fn,
        train_dataloader=train_loader, val_dataloader=val_loader,
        optimizer=optim, epochs=conf.epochs, writer=writer, save_path=save_path,
        device=device, logger=logger
    )
    if logger: logger.info(f"total training time: {time.time() - t0:.3f}s")
    writer.flush()

    try:
        best_model = torch.load(save_path, map_location=device)
        best_model = best_model.to(device).eval()
    except Exception:
        logger.info("[warn] Failed to load best model; use last epoch model.")
        best_model = model

    pred, real, features, labels, acc = test(best_model, test_loader, device=device, logger=logger)

    writer.add_scalar('test/acc', acc * 100.0)
    writer.close()

    del model
    del best_model
    torch.cuda.empty_cache()

    logger.info(f"==== [{split_tag}] RUN END (test_acc={acc*100:.2f}%) ====")
    return acc

if __name__ == '__main__':
    conf = Config()
    set_seed(300, deterministic=True)

    device = torch.device(f"cuda:{conf.device_num}" if torch.cuda.is_available() else "cpu")

    if conf.run_for in ['train', 'test']:
        selected_h5 = conf.train_h5 if conf.run_for == 'train' else conf.test_h5
        split_tag = infer_split_tag_from_h5(selected_h5)
        if conf.save_path == "None":
            conf.save_path = build_save_path('model_weight', split_tag)

        tstamp = timestamp_tag()
        tb_dir = f'logs/{split_tag}_{tstamp}'
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        logger = setup_logger(log_file=f'log/{split_tag}_{tstamp}.log')

        if conf.run_for == 'train':
            X_tr, X_va, Y_tr, Y_va = read_train_data(file_path=conf.train_h5,
                                                     use_sht=conf.use_sht,
                                                     L=conf.L,
                                                     sht_representation=conf.sht_representation)
            if conf.use_sht:
                M = n_sh_from_order(conf.L)
                in_ch = (2 * M) if conf.sht_representation == 'complex' else M
            else:
                in_ch = 1

            train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                     torch.tensor(Y_tr, dtype=torch.long).view(-1))
            val_ds   = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                                     torch.tensor(Y_va, dtype=torch.long).view(-1))
            train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=conf.batch_size, shuffle=True)

            model = mynet(num_classes=conf.num_classes, in_ch=in_ch).to(device)

            try:
                sample = torch.tensor(X_tr[:1], dtype=torch.float32).to(device)
                flops, params = thop_profile(model, inputs=(sample,), verbose=False)
                logger.info(f'[THOP] FLOPs={flops}  Params={params}')
            except Exception as e:
                logger.info(f'[THOP warn] {e}')
            try:
                if isinstance(X_tr, np.ndarray):
                    if X_tr.ndim == 4:
                        _, C_in, H, W = X_tr.shape
                        summary(model, input_size=(C_in, H, W))
                    elif X_tr.ndim == 3:
                        _, H, W = X_tr.shape
                        summary(model, input_size=(1, H, W))
            except Exception:
                pass

            loss_fn = nn.CrossEntropyLoss().to(device)
            optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0.0)

            best = float('inf')
            for epoch in range(1, conf.epochs + 1):
                _ = train_epoch(model, train_loader, optim, epoch, writer, device, loss_fn, alpha=0.5, logger=logger)
                val_loss, val_acc = evaluate_epoch(model, loss_fn, val_loader, epoch, writer, device, logger)
                if val_loss < best:
                    best = val_loss
                    os.makedirs(Path(conf.save_path).parent, exist_ok=True)
                    torch.save(model, conf.save_path)
                    logger.info(f"↓ New best val loss: {best:.6f} (model saved) -> {conf.save_path}")
                logger.info("-" * 100)

            writer.close()

        else:
            X_te, Y_te = read_test_data(file_path=conf.test_h5,
                                        use_sht=conf.use_sht,
                                        L=conf.L,
                                        sht_representation=conf.sht_representation)
            if conf.use_sht:
                M = n_sh_from_order(conf.L)
                in_ch = (2 * M) if conf.sht_representation == 'complex' else M
            else:
                in_ch = 1

            test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                    torch.tensor(Y_te, dtype=torch.long).view(-1))
            test_loader = DataLoader(test_ds, batch_size=conf.test_batch_size, shuffle=False)
            model = torch.load(conf.save_path, map_location=device).to(device)
            _ = test(model, test_loader, device=device, logger=logger)
            writer.close()

    else:
        base = Path('/data02/lpl/wav_dataset/uav/data')
        pairs = [
            ('6ft_train.h5',  '6ft_test.h5'),
            ('9ft_train.h5',  '9ft_test.h5'),
            ('12ft_train.h5', '12ft_test.h5'),
            ('15ft_train.h5', '15ft_test.h5'),
            ('ALL_train.h5',  'ALL_test.h5'),
        ]

        for tr_name, te_name in pairs:
            train_h5 = str(base / tr_name)
            test_h5  = str(base / te_name)

            split_tag = infer_split_tag_from_h5(train_h5)
            if (not Path(train_h5).exists()) or (not Path(test_h5).exists()):
                logger = setup_logger(log_file=f'log/{split_tag}_{timestamp_tag()}.log')
                logger.info(f"[SKIP] Missing files for split {split_tag}: train={train_h5}  test={test_h5}")
                continue

            conf.save_path = build_save_path('model_weight', split_tag)
            run_train_then_test_once(conf, train_h5, test_h5, device)
