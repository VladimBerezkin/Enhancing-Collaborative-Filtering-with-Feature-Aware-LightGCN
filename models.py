from copy import deepcopy

import torch
from torch import nn
from torch_geometric.nn import LGConv


class LightGCN(nn.Module):

    def __init__(self, emb_users, emb_items, num_layers=4):
        super().__init__()
        self.allow_unused=True
        self.num_users = emb_users.weight.shape[0]
        self.num_items = emb_items.weight.shape[0]
        self.num_layers = num_layers
        self.emb_users = deepcopy(emb_users)
        self.emb_items = deepcopy(emb_items)
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))

    def forward(self, edge_index):

        emb = torch.cat([self.emb_users.weight, self.emb_items.weight])
        embs = [emb]

        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        emb_final = 1/(self.num_layers+1) * torch.mean(torch.stack(embs, dim=1), dim=1)

        emb_users_final, emb_items_final = torch.split(emb_final, [self.num_users, self.num_items])

        return emb_users_final, emb_items_final
    

class LightGCNPlus0(LightGCN):

    def __init__(
        self, 
        emb_users, 
        emb_items, 
        users_features, 
        items_features, 
        user_proj,
        item_proj,
        num_layers=4
    ):
        super().__init__(emb_users, emb_items, num_layers)
        self.user_proj = user_proj
        self.item_proj = item_proj
        self.users_features_proj = torch.matmul(users_features, user_proj)
        self.items_features_proj = torch.matmul(items_features, item_proj)

    def forward(self, edge_index):
        emb_users_final, emb_items_final = super().forward(edge_index)
        emb_users_final = emb_users_final + self.users_features_proj
        emb_items_final = emb_items_final + self.items_features_proj
        return emb_users_final, emb_items_final
    

class LightGCNPlus1(LightGCN):

    def __init__(
        self, 
        emb_users, 
        emb_items, 
        users_features, 
        items_features, 
        user_proj,
        item_proj,
        num_layers=4
    ):
        super().__init__(emb_users, emb_items, num_layers)
        self.users_features_proj = torch.matmul(users_features, user_proj)
        self.items_features_proj = torch.matmul(items_features, item_proj)
        self.alpha_users = nn.Parameter(torch.tensor(0.))
        self.alpha_items = nn.Parameter(torch.tensor(0.))

    def forward(self, edge_index):
        emb_users_final, emb_items_final = super().forward(edge_index)
        alpha_users = nn.functional.sigmoid(self.alpha_users)
        alpha_items = nn.functional.sigmoid(self.alpha_items)
        emb_users_final = emb_users_final + alpha_users * self.users_features_proj
        emb_items_final = emb_items_final + alpha_items * self.items_features_proj
        return emb_users_final, emb_items_final
    

class LightGCNPlus2(LightGCN):

    def __init__(
        self, 
        emb_users, 
        emb_items, 
        users_features, 
        items_features, 
        user_proj,
        item_proj,
        num_layers=4
    ):
        super().__init__(emb_users, emb_items, num_layers)
        self.users_features_proj = torch.matmul(users_features, user_proj)
        self.items_features_proj = torch.matmul(items_features, item_proj)
        self.users_coefs_vector = nn.Parameter(torch.zeros(emb_users.weight.shape[1]))
        self.items_coefs_vector = nn.Parameter(torch.zeros(emb_items.weight.shape[1]))

    def forward(self, edge_index):
        emb_users_final, emb_items_final = super().forward(edge_index)
        users_coefs_vector = nn.functional.sigmoid(self.users_coefs_vector)
        items_coefs_vector = nn.functional.sigmoid(self.items_coefs_vector)
        emb_users_final = emb_users_final + users_coefs_vector * self.users_features_proj
        emb_items_final = emb_items_final + items_coefs_vector * self.items_features_proj
        return emb_users_final, emb_items_final
    

class LightGCNPlus3(LightGCN):

    def __init__(
        self, 
        emb_users, 
        emb_items, 
        users_features, 
        items_features, 
        user_proj,
        item_proj,
        num_layers=4
    ):
        super().__init__(emb_users, emb_items, num_layers)
        self.user_proj = torch.nn.Linear(users_features.shape[1], emb_users.weight.shape[1], bias=False)
        nn.init.zeros_(self.user_proj.weight)
        self.item_proj = torch.nn.Linear(items_features.shape[1], emb_items.weight.shape[1], bias=False)
        nn.init.zeros_(self.item_proj.weight)
        self.users_features = users_features
        self.items_features = items_features
        
    def forward(self, edge_index):
        emb_users_final, emb_items_final = super().forward(edge_index)
        users_features_emb = self.user_proj(self.users_features)
        items_features_emb = self.item_proj(self.items_features)
        emb_users_final = emb_users_final + users_features_emb
        emb_items_final = emb_items_final + items_features_emb
        return emb_users_final, emb_items_final
