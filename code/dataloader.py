"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import collections
import os
import logging
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

class KGDataset(Dataset):
    def __init__(self, kg_path, entity_num_per_item):
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')

        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.entity_num = entity_num_per_item

    @property
    def entity_count(self):
        # start from zero
        return self.kg_data['t'].max()+2

    @property
    def relation_count(self):
        return self.kg_data['r'].max()+2

    def get_kg_dict(self, item_num):
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) > self.entity_num):
                    i2es[item] = torch.IntTensor(tails).cuda()[:self.entity_num]
                    i2rs[item] = torch.IntTensor(relations).cuda()[:self.entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(self.entity_num-len(tails)))
                    relations.extend([self.relation_count]*(self.entity_num-len(relations)))
                    i2es[item] = torch.IntTensor(tails).cuda()
                    i2rs[item] = torch.IntTensor(relations).cuda()
            else:
                i2es[item] = torch.IntTensor([self.entity_count]*self.entity_num).cuda()
                i2rs[item] = torch.IntTensor([self.relation_count]*self.entity_num).cuda()
        return i2es, i2rs


    def generate_kg_data(self, kg_data): 
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail

class HisLoader(Dataset):
    @staticmethod
    def get_train_stats(train_file):
        """
        Calcola trainSize, n_users e m_items da un file .txt in cui ogni riga
        è: userID itemID1 itemID2 ... itemIDn
        
        Calculates trainSize, n_users, and m_items from a .txt file where each line
        is: userID itemID1 itemID2 ... itemIDn
        """
        train_size = 0
        n_users = 0
        max_item_id = -1

        with open(train_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue  # salta righe vuote
                user_id = int(parts[0])
                item_ids = list(map(int, parts[1:]))

                train_size += len(item_ids)
                n_users += 1
                if item_ids:
                    max_item_id = max(max_item_id, max(item_ids))

        m_items = max_item_id + 1 if max_item_id >= 0 else 0

        return train_size, n_users, m_items

    def __init__(self, config:dict):
        # train or test
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.maxhis = config['maxhis']  # 控制取最长历史 - Controls the maximum length of history
        self.mode = 'train' # 控制获取数据的种类 - Controls the type of data retrieval
        self.path = f'{config["path"]}/{config["dataset"]}'

        # Missing attributes
        tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.path, 'train.txt'))
        self.trainSize, self.n_user, self.m_item = self.get_train_stats(tmp_path)

        # Print with colors the stats
        logging.info(f'\033[36mtrainSize\033[0m: \033[35m{self.trainSize}\033[0m')
        logging.info(f'\033[36mn_user\033[0m: \033[35m{self.n_user}\033[0m')
        logging.info(f'\033[36mm_item\033[0m: \033[35m{self.m_item}\033[0m')

        self.testSize = 0

        logging.info(f'\033[36mloading [\033[35m' 
                     + os.path.abspath(os.path.join(os.path.dirname(__file__), self.path)) 
                     + '\033[36m]\033[0m')

        self.read_file('train')
        self.read_whole_line('test')
        self.read_whole_line('valid')

        # 因为是从0开始索引，因此需要+1
        self.m_item += 1
        self.n_user += 1
        
        self.Graph = None
        logging.info(f"{config['dataset']} Sparsity : {(self.trainSize+self.testSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        logging.info(f"{config['dataset']} is ready to go")

    def read_file(self, filetype):
        #filepath = self.path+f'/{filetype}.txt' old version
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), self.path, f'{filetype}.txt'))  #SALE UNA CARTELLA DI TROPPO NONOSTANTE self.path SIA GIUSTO
        if not os.path.exists(filepath): #TODO convert the if in a try catch
            logging.info(f"\033[91mfile {filepath} doesn't exist\033[0m")
            return

        UniqueUsers, Item, User = [], [], []
        UsersHis = []
        dataSize = 0

        with open(filepath) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if filetype == "train":
                        items, his = [int(l[1])], [] 
                        for i in l[2:]:
                            this_his = items if self.maxhis<=0 else items[-self.maxhis:]
                            this_his = this_his + [-1] * (self.maxhis-len(this_his))    # -1 表示没有记录
                            his.append(this_his)
                            items.append(int(i))
                        UsersHis.extend(his)
                        items = items[1:]
                    else:
                        items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    UniqueUsers.append(uid)
                    User.extend([uid] * len(items))
                    Item.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    dataSize += len(items)

        setattr(self, f'{filetype}UniqueUsers', np.array(UniqueUsers))
        setattr(self, f'{filetype}User', np.array(User))
        setattr(self, f'{filetype}Item', np.array(Item))
        setattr(self, f'{filetype}UsersHis', np.array(UsersHis))
        setattr(self, f'{filetype}Size', dataSize)
        
        logging.info(f"{dataSize} interactions for {filetype}")

    def read_whole_line(self, filetype):
        filepath = self.path+f'/{filetype}.txt'
        if not os.path.exists(filepath):
            return

        Item, User = [], []
        dataSize = 0

        with open(filepath) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if not l[1]:
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    User.append(uid)
                    Item.append(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    dataSize += 1
        
        setattr(self, f'{filetype}User', np.array(User))
        setattr(self, f'{filetype}Item', np.array(Item))
        setattr(self, f'{filetype}Size', dataSize)
        logging.info(f"{dataSize} interactions for {filetype}")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
        
    @property
    def allPos(self):
        return self._allPos
        
    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        '''计算图卷积中的连接图，包括A~等
        返回的数据是经过处理的列表，列表长度由self.fold决定，其中每一项都是一个稀疏矩阵，表示对应长度下该序号entity的连接矩阵
        '''
        logging.info("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                logging.info("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                logging.info("generating adjacency matrix")
                s = time()
                # adj_mat 的横纵坐标将用户与物品进行拼接，并将已知连接进行标记
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok() # 将矩阵转换为字典形式（键值对）
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv) # 对角线元素为每行求和
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                logging.info(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                logging.info("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                logging.info("don't split the matrix")
        return self.Graph

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
        
    # train loader and sampler part
    def __len__(self):
        return getattr(self, self.mode+'Size')

    def __getitem__(self, idx):
        user = getattr(self, self.mode+'User')[idx]
        pos = getattr(self, self.mode+'Item')[idx]
        if self.mode != 'train':
            return user, torch.tensor(self.UserItemNet[user].nonzero()[1][-self.maxhis:]), torch.tensor(pos)
        history = getattr(self, self.mode+'UsersHis')[idx]
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg, history