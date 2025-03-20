import json

import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import structured_negative_sampling
import torch
from torch import optim, nn
from time import time

from models import (
    LightGCNPlus1,
    LightGCNPlus2,
    LightGCNPlus3
)

SEED = 111


def sampler(edge_index):

    num_users_nodes = edge_index[0, :].max()
    num_items_nodes = edge_index[1, :].max()
    samples_users = structured_negative_sampling(
        edge_index, 
        num_nodes=num_users_nodes+1, 
        contains_neg_self_loops=False
        )
    
    samples_items = structured_negative_sampling(
        edge_index, 
        num_nodes=num_items_nodes+1, 
        contains_neg_self_loops=False
        )
    
    return samples_users, samples_items


def sample_mini_batch(edge_index, batch_size):

    index = np.random.choice(range(edge_index.shape[1]), size=batch_size)

    num_nodes = edge_index[1, :].max() + 1

    edge_index = structured_negative_sampling(
        edge_index, 
        num_nodes=num_nodes, 
        contains_neg_self_loops=False
        )
    
    edge_index = torch.stack(edge_index, dim=0)
    
    user_index = edge_index[0, index]
    pos_item_index = edge_index[1, index]
    neg_item_index = edge_index[2, index]
    
    return user_index, pos_item_index, neg_item_index


def bpr_loss(
        emb_users: torch.Tensor, 
        emb_pos_items: torch.Tensor, 
        emb_neg_items: torch.Tensor,
        lmbda: float
    ):

    reg_loss = lmbda * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))
    pos_ratings = torch.mul(emb_users, emb_pos_items).sum(dim=-1)
    neg_ratings = torch.mul(emb_users, emb_neg_items).sum(dim=-1)
    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    return -bpr_loss + reg_loss


def get_user_items(edge_index):
    user_items = dict()
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_items:
            user_items[user] = []
        user_items[user].append(item)
    return user_items


def compute_recall_at_k(K, items_ground_truth, items_predicted):
    test_matrix = np.zeros((len(items_predicted), K))

    for i, items in enumerate(items_ground_truth):
        length = min(len(items), K)
        test_matrix[i, :length] = 1

    num_relevant_items = np.sum(test_matrix, axis=1)
    num_correct_predictions = np.sum(items_predicted * test_matrix, axis=1)

    num_relevant_items[num_relevant_items == 0] = 1  
    recall = num_correct_predictions / num_relevant_items

    return np.mean(recall)


def compute_ndcg_at_k(K, items_ground_truth, items_predicted):
    test_matrix = np.zeros((len(items_predicted), K))

    for i, items in enumerate(items_ground_truth):
        length = min(len(items), K)
        test_matrix[i, :length] = 1
    
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, K + 2)), axis=1)
    dcg = items_predicted * (1. / np.log2(np.arange(2, K + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    
    return np.mean(ndcg)


def get_metrics(model, K, edge_index, exclude_edge_indices):

    ratings = torch.matmul(model.emb_users.weight, model.emb_items.weight.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        ratings[exclude_users, exclude_items] = -1024

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(ratings, k=K)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    items_predicted = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        items_predicted.append(label)

    recall = compute_recall_at_k(K, test_user_pos_items_list, items_predicted)
    ndcg = compute_ndcg_at_k(K, test_user_pos_items_list, items_predicted)

    return recall, ndcg


def test(model, edge_index, exclude_edge_indices, lmbda, K):
    emb_users, emb_items = model.forward(edge_index)

    num_nodes = edge_index[1, :].max() + 1

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(
        edge_index, 
        num_nodes=num_nodes,
        contains_neg_self_loops=False
        )

    emb_users = emb_users[user_indices]

    emb_pos_items = emb_items[pos_item_indices]
    emb_neg_items = emb_items[neg_item_indices]

    loss = bpr_loss(emb_users, emb_pos_items, emb_neg_items, lmbda).item()

    recall, ndcg = get_metrics(model, K, edge_index, exclude_edge_indices)

    return loss, recall, ndcg


def initialize_result_container():
    return {
        'train_loss': [],
        'val_loss': [],
        'val_recall': [],
        'val_ndcg': [],
        'model_params': []
    }


def append_to_result(result, model_name, dataset_name, train_loss, val_loss, val_recall, val_ndcg):
    result['model_name'] = model_name
    result['dataset_name'] = dataset_name
    result['train_loss'].append(float(train_loss))
    result['val_loss'].append(float(val_loss))
    result['val_recall'].append(float(val_recall))
    result['val_ndcg'].append(float(val_ndcg))
    return result


def train_model(model, model_name, dataset_name, optimizer, train_edge_index, val_edge_index, n_epochs, n_batch, batch_size, lmbda, k):

    result = initialize_result_container()

    t_start = time() 
    params_hist = []
    for epoch in range(n_epochs):
        model.train()

        for _ in range(n_batch):
            optimizer.zero_grad()

            emb_users, emb_items = model.forward(train_edge_index)

            user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(train_edge_index, batch_size)
            
            emb_users = emb_users[user_indices]
            emb_pos_items = emb_items[pos_item_indices]
            emb_neg_items = emb_items[neg_item_indices]

            train_loss = bpr_loss(emb_users, emb_pos_items, emb_neg_items, lmbda)

            train_loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_recall, val_ndcg = test(model, val_edge_index, [train_edge_index], lmbda, k)
        print(f"Epoch {epoch} | Train loss: {train_loss.item():.5f} | Val loss: {val_loss:.5f} | Val recall@{k}: {val_recall:.5f} | Val ndcg@{k}: {val_ndcg:.5f}")
        append_to_result(result, model_name, dataset_name, train_loss, val_loss, val_recall, val_ndcg)
        append_model_params_to_results(model, result)
    t_end = time() 
    training_time = t_end - t_start
    print(f'Training time: {training_time:.2f}')
    result['training_time'] = training_time
    result['params_hist'] = params_hist
    return result


def append_model_params_to_results(model, results):
    if isinstance(model, LightGCNPlus1):
        results['model_params'].append((
            nn.functional.sigmoid(model.alpha_users).item(),
            nn.functional.sigmoid(model.alpha_items).item()
            ))
    elif isinstance(model, LightGCNPlus2):
        results['model_params'].append((
            nn.functional.sigmoid(model.users_coefs_vector).tolist(),
            nn.functional.sigmoid(model.items_coefs_vector).tolist()
            ))
    elif isinstance(model, LightGCNPlus3):
        results['model_params'].append((
            model.user_proj.weight.tolist(),
            model.item_proj.weight.tolist()
            ))
    else:
        pass
    return results


def training_routine(
        model, 
        model_name, 
        dataset_name, 
        train_edge_index, 
        val_edge_index, 
        n_epochs,
        n_batch,
        batch_size,
        lambd,
        K
    ):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    result = train_model(
        model,
        model_name, 
        dataset_name,
        optimizer,
        train_edge_index,
        val_edge_index,
        n_epochs,
        n_batch,
        batch_size,
        lambd,
        K
    )
    return result


def save_result(model, result, dataset_name, model_name):
    filename = f'{dataset_name}_{model_name}'
    torch.save(model.state_dict(), f'models/{filename}.bin')
    with open(f"results/{filename}.json", "w") as outfile: 
        json.dump(result, outfile)


class Plot:

    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4), )
        self.ax[0].set_title('Val recall@20')
        self.ax[1].set_title(' Val ndcg@20')

    def add(self, result, label):
        self.ax[0].plot(result['val_recall'], label=label)
        self.ax[1].plot(result['val_ndcg'], label=label)
        self.ax[0].legend()
        self.ax[1].legend()

    def show(self):
        plt.show()