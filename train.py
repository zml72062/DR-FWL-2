"""
train.py - Standardized NN training script for PyTorch.
"""
import random
import numpy
import torch
import sys
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def seed_everything(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def epoch(model, 
          loader, 
          pred_fn, 
          truth_fn, 
          loss_fn, 
          metric, 
          batch_len, 
          dataset_len, 
          device, 
          optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    result, loss = 0.0, 0.0
    for batch in tqdm(loader):
        if optimizer is not None:
            optimizer.zero_grad()
        batch = batch.to(device)
        pred = pred_fn(model, batch).squeeze()
        # print(pred)
        truth = truth_fn(batch)
        batch_loss = loss_fn(pred, truth)

        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss += batch_loss.item() * batch_len(batch)
            result += metric(pred, truth).item() * batch_len(batch)
    return loss/dataset_len, result/dataset_len


def run(epochs, 
        model, 
        train_loader, 
        valid_loader,
        test_loader,
        train_set,
        valid_set,
        test_set,
        pred_fn, 
        truth_fn, 
        loss_fn, 
        metric, 
        metric_str,
        batch_len, 
        device, 
        optimizer, 
        lr_scheduler=None, 
        choose_best='max',
        log_file=sys.stdout):
    """
    Parameters:

        epochs -- max epochs to train (int)
    
        model -- the model (callable)

        train_loader, valid_loader, test_loader -- the train/valid/test loader (iterable)

        train_set, valid_set, test_set -- the train/valid/test dataset

        pred_fn -- function that takes in `model` & `batch`, and gives an output tensor `pred`

        truth_fn -- function that takes in `batch`, and gives an output tensor `truth`

        loss_fn -- function that takes in `pred` & `truth`, and computes average loss tensor

        metric -- function that takes in `pred` & `truth`, and computes average metric tensor

        metric_str -- name of the metric

        batch_len -- function that takes in `batch`, and gives its length

        device -- the device (str)

        optimizer -- the optimizer

        lr_scheduler -- the learning rate scheduler (default None)

        choose_best -- whether to choose the epoch with best validation performance,
        if None then choose the last epoch (default 'max', means that larger metric
        is better)

        log_file -- file to write log
    """
    model.to(device)

    best_val_metric = 1e6 if choose_best == 'min' else 0
    best_test_metric = 0

    before_running = time.time()
    for idx in range(epochs):
        train_loss, train_metric = epoch(model, train_loader, pred_fn, truth_fn, 
                                         loss_fn, metric, batch_len, 
                                         len(train_set), device, optimizer)
        val_loss, val_metric = epoch(model, valid_loader, pred_fn, truth_fn,
                                     loss_fn, metric, batch_len, len(valid_set),
                                     device, None)
        test_loss, test_metric = epoch(model, test_loader, pred_fn, truth_fn,
                                       loss_fn, metric, batch_len, len(test_set),
                                       device, None)
        if choose_best == 'max' and val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        elif choose_best == 'min' and val_metric < best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        if lr_scheduler is not None:
            if lr_scheduler.__class__ != ReduceLROnPlateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(val_metric)

        if log_file is not None:
            print("Epoch %d: " % idx, file=log_file)
            print("Training Loss: %f    Training %s: %f" % (train_loss, metric_str, train_metric), file=log_file)
            print("Validation Loss: %f    Validation %s: %f" % (val_loss, metric_str, val_metric), file=log_file)
            print("Test Loss %f    Test %s: %f" % (test_loss, metric_str, test_metric), file=log_file)
    after_running = time.time()
    print(f"Running time for {epochs} epochs:", after_running - before_running)
    return best_test_metric

@torch.no_grad()
def inference(epochs, 
        model, 
        train_loader, 
        valid_loader,
        test_loader,
        train_set,
        valid_set,
        test_set,
        pred_fn, 
        truth_fn, 
        loss_fn, 
        metric, 
        metric_str,
        batch_len, 
        device, 
        choose_best='max'):
    """
    Parameters:

        epochs -- max epochs to train (int)
    
        model -- the model (callable)

        train_loader, valid_loader, test_loader -- the train/valid/test loader (iterable)

        train_set, valid_set, test_set -- the train/valid/test dataset

        pred_fn -- function that takes in `model` & `batch`, and gives an output tensor `pred`

        truth_fn -- function that takes in `batch`, and gives an output tensor `truth`

        loss_fn -- function that takes in `pred` & `truth`, and computes average loss tensor

        metric -- function that takes in `pred` & `truth`, and computes average metric tensor

        metric_str -- name of the metric

        batch_len -- function that takes in `batch`, and gives its length

        device -- the device (str)

        optimizer -- the optimizer

        lr_scheduler -- the learning rate scheduler (default None)

        choose_best -- whether to choose the epoch with best validation performance,
        if None then choose the last epoch (default 'max', means that larger metric
        is better)

        log_file -- file to write log
    """
    model.to(device)

    best_val_metric = 1e6 if choose_best == 'min' else 0
    best_test_metric = 0

    for idx in range(epochs):
        train_loss, train_metric = epoch(model, train_loader, pred_fn, truth_fn, 
                                         loss_fn, metric, batch_len, 
                                         len(train_set), device, None)
        val_loss, val_metric = epoch(model, valid_loader, pred_fn, truth_fn,
                                     loss_fn, metric, batch_len, len(valid_set),
                                     device, None)
        test_loss, test_metric = epoch(model, test_loader, pred_fn, truth_fn,
                                       loss_fn, metric, batch_len, len(test_set),
                                       device, None)
        if choose_best == 'max' and val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        elif choose_best == 'min' and val_metric < best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        print("Epoch %d: " % idx)
        print("Training Loss: %f    Training %s: %f" % (train_loss, metric_str, train_metric))
        print("Validation Loss: %f    Validation %s: %f" % (val_loss, metric_str, val_metric))
        print("Test Loss %f    Test %s: %f" % (test_loss, metric_str, test_metric))


    return best_test_metric