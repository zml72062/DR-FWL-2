# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation

from data_utils.preprocess import drfwl2_transform, drfwl3_transform
import numpy as np
import torch
import torch_geometric
from pygmmpp.data import DataLoader
from loguru import logger
import time
from data_utils.batch import collate
from BREC.BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
import argparse
from torch.nn import CosineEmbeddingLoss
from json import dumps


from BREC.core.config import cfg

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 10
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_count import DR2FWL2Kernel
from models.pool import GraphLevelPooling

class BRECModel2(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 norm_between_layers: str = "batch_norm",
                 residual: str = "none",
                 drop_prob: float = 0.0,
                 output_dim: int = 16):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob
        self.OUTPUT_DIM = output_dim

        self.node_transform = nn.Linear(1, self.hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling(hidden_channels)

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, self.OUTPUT_DIM))
        
        self.ker.add_aggr(1, 1, 1)
        if self.add_0:
            self.ker.add_aggr(0, 1, 1)
            self.ker.add_aggr(0, 2, 2)
        if self.add_112:
            self.ker.add_aggr(1, 1, 2)
        if self.add_212:
            self.ker.add_aggr(2, 2, 1)
        if self.add_222:
            self.ker.add_aggr(2, 2, 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_transform.reset_parameters()
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2]
        batch.x = batch.x.to(torch.float32)
        edge_attrs = [self.node_transform(batch.x),
                      self.node_transform(batch.x[batch.edge_index[0]]) +
                      self.node_transform(batch.x[batch.edge_index[1]]),
                      self.node_transform(batch.x[batch.edge_index2[0]]) +
                      self.node_transform(batch.x[batch.edge_index2[1]])
                      ]
        triangles = {
            (1, 1, 1): batch.triangle_1_1_1,
            (1, 1, 2): batch.triangle_1_1_2,
            (2, 2, 1): batch.triangle_2_2_1,
            (2, 2, 2): batch.triangle_2_2_2,
        }
        inverse_edges = [batch.inverse_edge_1, batch.inverse_edge_2]

        edge_attrs = self.ker(edge_attrs,
                              edge_indices,
                              triangles,
                              inverse_edges)


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, batch.batch0)
        x = self.post_mlp(x)
        # x = F.log_softmax(x, dim=1)
        return x

class BRECModel3(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 norm_between_layers: str = "batch_norm",
                 residual: str = "none",
                 drop_prob: float = 0.0,
                 output_dim: int = 16):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob
        self.OUTPUT_DIM = output_dim

        self.node_transform = nn.Linear(1, self.hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling(hidden_channels)

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, self.OUTPUT_DIM))
        
        self.ker.add_aggr(1, 1, 1)
        if self.add_0:
            self.ker.add_aggr(0, 1, 1)
            self.ker.add_aggr(0, 2, 2)
        if self.add_112:
            self.ker.add_aggr(1, 1, 2)
        if self.add_212:
            self.ker.add_aggr(2, 2, 1)
        if self.add_222:
            self.ker.add_aggr(2, 2, 2)
        self.ker.add_aggr(1, 2, 3)
        self.ker.add_aggr(3, 3, 1)
        self.ker.add_aggr(2, 2, 3)
        self.ker.add_aggr(3, 3, 2)
        self.ker.add_aggr(3, 3, 3)
        self.ker.add_aggr(0, 3, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_transform.reset_parameters()
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2, batch.edge_index3]
        batch.x = batch.x.to(torch.float32)
        edge_attrs = [self.node_transform(batch.x),
                      self.node_transform(batch.x[batch.edge_index[0]]) +
                      self.node_transform(batch.x[batch.edge_index[1]]),
                      self.node_transform(batch.x[batch.edge_index2[0]]) +
                      self.node_transform(batch.x[batch.edge_index2[1]]),
                      self.node_transform(batch.x[batch.edge_index3[0]]) +
                      self.node_transform(batch.x[batch.edge_index3[1]])
                      ]
        triangles = {
            (1, 1, 1): batch.triangle_1_1_1,
            (1, 1, 2): batch.triangle_1_1_2,
            (2, 2, 1): batch.triangle_2_2_1,
            (2, 2, 2): batch.triangle_2_2_2,
            (1, 2, 3): batch.triangle_1_2_3,
            (3, 3, 1): batch.triangle_3_3_1,
            (2, 2, 3): batch.triangle_2_2_3,
            (3, 3, 2): batch.triangle_3_3_2,
            (3, 3, 3): batch.triangle_3_3_3,
        }
        inverse_edges = [batch.inverse_edge_1, batch.inverse_edge_2, batch.inverse_edge_3]

        edge_attrs = self.ker(edge_attrs,
                              edge_indices,
                              triangles,
                              inverse_edges)


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, batch.batch0)
        x = self.post_mlp(x)
        # x = F.log_softmax(x, dim=1)
        return x


parser = argparse.ArgumentParser("arguments for training and testing")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=16)
parser.add_argument("--SEED", type=int, default=2022)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--D", type=int, default=2)
parser.add_argument("--DEVICE", type=str, default="1")

# parser.add_argument("--CONFIG", type=str, default="4_1_32")

args = parser.parse_args()

P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
D = args.D
DEVICE = args.DEVICE
torch_geometric.seed_everything(SEED)

torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 110),
    "Extension": (160, 260),
    "CFI": (260, 320),
}


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(cfg, dataset_name):
    time_start = time.process_time()

    # dataset = BRECDataset(transform=transform_eval)
    dataset = BRECDataset(name=dataset_name, pre_transform=eval(f"drfwl{D}_transform")())

    time_end = time.process_time()

    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(cfg):
    time_start = time.process_time()

    # Do something
    model = (eval(f"BRECModel{D}"))(cfg.model.hidden_size,
                    cfg.model.num_layers,
                    norm_type='none',
                    norm_between_layers='none',
                    residual='last',
                    output_dim=OUTPUT_DIM).to('cuda:'+DEVICE)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device):
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = DataLoader(dataset, collator=collate, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            # S = torch.nan_to_num(torch.cov(D))
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # inv_S = torch.inverse(S + S_epsilon)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()

    # Do something
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id - part_range[0]}")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM)
                * NUM_RELABEL
                * 2 : (id + SAMPLE_NUM + 1)
                * NUM_RELABEL
                * 2
            ]
            model.reset_parameters()
            model.train()
            for _ in range(EPOCH):
                traintest_loader = DataLoader(dataset_traintest, collator=collate, batch_size=BATCH_SIZE)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, True)
            T_square_reliability = T2_calculation(dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id - part_range[0])
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{path}/result_show.txt", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\thops\tlayers\tmini_layers\thidden\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tTHRESHOLD\tMARGIN\tLOSS_THRESHOLD\tEPOCH\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{cfg.model.num_layers}\t{D}\t{cfg.model.hidden_size}"
        f"\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{THRESHOLD}\t{MARGIN}\t{LOSS_THRESHOLD}\t{EPOCH}\t{SEED}"
    )


if __name__ == "__main__":
    file_name = "BREC/train/configs/BREC.yaml"
    cfg.merge_from_file(file_name)
    # cfg = update_cfg(cfg)

    # Command Line Arguments
    device = torch.device("cuda:"+DEVICE if torch.cuda.is_available() else "cpu")

    NAME = f"d={D}_layer={cfg.model.num_layers}_hidden={cfg.model.hidden_size}"
    DATASET_NAME = f"d={D}"
    OUT_PATH = "result_BREC"
    PATH = os.path.join(OUT_PATH, NAME)
    os.makedirs(PATH, exist_ok=True)

    LOG_NAME = os.path.join(PATH, "log.txt")
    logger.remove(handler_id=None)
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    logger.info(cfg)

    pre_calculation()
    dataset = get_dataset(cfg, DATASET_NAME)
    model = get_model(cfg)
    evaluation(dataset, model, OUT_PATH, device)
