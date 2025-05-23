# coding=utf-8

import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import *
import numpy as np
import torch.nn as nn
from GAT import GAT
import os

class KGEncoder(nn.Module):
    def __init__(self,config,dataset, kg_dataset):
        super().__init__()
        self.kgcn = config['kgcn']
        self.dropout = config['dropout']
        self.keep_prob = 1 - self.dropout #Added
        self.A_split = config['A_split']

        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.dataset = dataset
        self.kg_dataset = kg_dataset
        self.gat = GAT(self.latent_dim, self.latent_dim,
                       dropout=0.4, alpha=0.2).train()
        self.__init_weight()
    
    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
       
        # Print with colors the stats
        logging.info(f'\033[36mtrainSize\033[0m: \033[35m{self.dataset.trainSize}\033[0m')
        logging.info(f'\033[36mn_user\033[0m: \033[35m{self.dataset.n_user}\033[0m')
        logging.info(f'\033[36mm_item\033[0m: \033[35m{self.dataset.m_item}\033[0m')

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # item and kg entity
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        # relation weights
        self.W_R = nn.Parameter(torch.Tensor(
            self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        logging.info('use NORMAL distribution UI')
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        logging.info('use NORMAL distribution ENTITY')
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)
        
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(
            self.num_items)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # logging.info(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        if (self.kgcn == "GAT"):
            return self.cal_item_embedding_gat(kg)
        elif self.kgcn == "RGAT":
            return self.cal_item_embedding_rgat(kg)
        elif (self.kgcn == "MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif (self.kgcn == "NO"):
            return self.embedding_item.weight

    def cal_item_embedding_gat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).cuda())  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat(item_embs, entity_embs, padding_mask)

    def cal_item_embedding_rgat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).cuda())  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(
            item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).cuda())  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * \
            padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / \
            padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs+entity_embs_mean


class KGLRR(nn.Module):
    def __init__(self, config, dataset, kg_dataset) -> None:
        super(KGLRR,self).__init__()
        self.encoder = KGEncoder(config, dataset, kg_dataset)
        self.latent_dim = config['latent_dim_rec']

        self.r_logic = config['r_logic']
        self.r_length = config['r_length']
        self.layers = config['layers']
        self.sim_scale = config['sim_scale']
        self.loss_sum = config['loss_sum']
        self.l2s_weight = config['l2_loss']
        
        self.dataset = dataset
        self.num_items = dataset.m_items
        self.kg_dataset = kg_dataset

        self._init_weights()
        self.bceloss = nn.BCEWithLogitsLoss()

    def _init_weights(self):
        self.true = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 1, size=[1, self.latent_dim]).astype(np.float32)).cuda(), requires_grad=False)

        self.and_layer = torch.nn.Linear(self.latent_dim * 2, self.latent_dim)
        for i in range(self.layers):
            setattr(self, 'and_layer_%d' % i, torch.nn.Linear(self.latent_dim * 2, self.latent_dim * 2))

        self.or_layer = torch.nn.Linear(self.latent_dim * 2, self.latent_dim)
        for i in range(self.layers):
            setattr(self, 'or_layer_%d' % i, torch.nn.Linear(self.latent_dim * 2, self.latent_dim * 2))

    def logic_or(self, vector1, vector2, train=False):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'or_layer_%d' % i)(vector))
        vector = self.or_layer(vector)
        return vector
    
    def logic_and(self, vector1, vector2, train=False):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'and_layer_%d' % i)(vector))
        vector = self.and_layer(vector)
        return vector

    def logic_regularizer(self, train:bool, check_list:list, constraint, constraint_valid):
        # 该函数计算逻辑表达与真实世界的差距

        # length
        r_length = constraint.norm(dim=2).sum()
        check_list.append(('r_length', r_length))
        
        # and
        r_and_true = 1 - self.similarity(self.logic_and(constraint, self.true, train=train), constraint)
        r_and_true = (r_and_true * constraint_valid).sum()
        check_list.append(('r_and_true', r_and_true))

        r_and_self = 1 - self.similarity(self.logic_and(constraint, constraint, train=train), constraint)
        r_and_self = (r_and_self * constraint_valid).sum()
        check_list.append(('r_and_self', r_and_self))
        
        # or
        r_or_true = 1 - self.similarity(self.logic_or(constraint, self.true, train=train), self.true)
        r_or_true = (r_or_true * constraint_valid).sum()
        check_list.append(('r_or_true', r_or_true))
        
        r_or_self = 1 - self.similarity(self.logic_or(constraint, constraint, train=train), constraint)
        r_or_self = (r_or_self * constraint_valid).sum()
        check_list.append(('r_or_self', r_or_self))

        r_loss = r_and_true + r_and_self \
                    + r_or_true + r_or_self
        
        if self.r_logic > 0:
            r_loss = r_loss * self.r_logic
        else:
            r_loss = torch.from_numpy(np.array(0.0, dtype=np.float32)).cuda()
            r_loss.requires_grad = True

        r_loss += r_length * self.r_length
        check_list.append(('r_loss', r_loss))
        return r_loss

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    # 删除掉了向量大小的归一化 - Removed vector size normalization
    def uniform_size(self, vector1, vector2, train=False):
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        elif vector2.size() != vector1.size():
            vector2 = vector2.expand_as(vector1)
        if train:
            r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
            r12 = r12.cuda().unsqueeze(-1)
            new_v1 = r12 * vector1 + (-r12 + 1) * vector2
            new_v2 = r12 * vector2 + (-r12 + 1) * vector1
            return new_v1, new_v2
        return vector1, vector2

    def just_predict(self, users, history, explain=True):
        bs = users.size(0)
        item_embed = self.encoder.computer()[1]   # item_num * V
        
        his_valid = history.ge(0).float()  # B * H

        maxlen = int(his_valid.sum(dim=1).max().item())
        
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V

        tmp_o = None
        for i in range(maxlen):
            tmp_o_valid = his_valid[:, i].unsqueeze(-1)
            if tmp_o is None:
                tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
            else:
                # 有valid标志才能运算or，否则就是历史长度没有这么长（才会不valid），此时直接保持原本的内容不变
                tmp_o = self.logic_or(tmp_o, elements[:, i, :]) * tmp_o_valid + \
                        tmp_o * (-tmp_o_valid + 1)  # B * V
        or_vector = tmp_o  # B * V
        left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1
        
        prediction = []
        for i in range(bs):
            sent_vector = left_valid[i] * self.logic_and(or_vector[i].unsqueeze(0
                                                            ).repeat(self.num_items, 1), item_embed) \
                            + (-left_valid[i] + 1) * item_embed  # item_size * V
            ithpred = self.similarity(sent_vector, self.true, sigmoid=True)  # item_size
            prediction.append(ithpred)

        prediction = torch.stack(prediction).cuda()

        # if explain:
        #     explaination = self.explain(users, history, prediction)
        #     return prediction, explaination
        return prediction

    # def explain(self, users, history, items):
    #     bs = users.size(0)
    #     users_embed, item_embed = self.encoder.computer()   # user_num/item_num * V
        
    #     his_valid = history.ge(0).float()  # B * H
    #     maxlen = int(his_valid.sum(dim=1).max().item())
    #     elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V

    #     similarity_rlt = []
    #     for i in range(bs):
    #         tmp_a_valid = his_valid[i, :].unsqueeze(-1)  # H
    #         tmp_a = self.logic_and(items[i].unsqueeze(0), elements[i]) * tmp_a_valid  # H * V
    #         similarity_rlt.append(self.similarity(tmp_a, self.true))
        
    #     return torch.stack(similarity_rlt).cuda()

    def predict_or_and(self, users, pos, neg, history):
        # 存储用于检查的内容：逻辑正则化 
        # 对嵌入计算L2正则化
        check_list = []
        bs = users.size(0)
        users_embed, item_embed = self.encoder.computer()

        # 历史数据中每个商品都为正标记，但是历史后段可能取-1表示没有这么长
        his_valid = history.ge(0).float()  # B * H
        maxlen = int(his_valid.sum(dim=1).max().item())
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V
        
        # 用于之后的验证，每个向量都应满足逻辑表达式中的相应约束，valid表示限制向量中对应元的有效性
        constraint = [elements.view([bs, -1, self.latent_dim])]  # B * H * V
        constraint_valid = [his_valid.view([bs, -1])]  # B * H
        
        tmp_o = None
        for i in range(maxlen):
            tmp_o_valid = his_valid[:, i].unsqueeze(-1)
            if tmp_o is None:
                tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
            else:
                # 有valid标志才能运算or，否则就是历史长度没有这么长（才会不valid），此时直接保持原本的内容不变
                tmp_o = self.logic_or(tmp_o, elements[:, i, :]) * tmp_o_valid + \
                        tmp_o * (-tmp_o_valid + 1)  # B * V
                constraint.append(tmp_o.view([bs, 1, self.latent_dim]))  # B * 1 * V
                constraint_valid.append(tmp_o_valid)  # B * 1
        or_vector = tmp_o  # B * V
        left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1

        right_vector_true = item_embed[pos]  # B * V
        right_vector_false = item_embed[neg]  # B * V
        
        constraint.append(right_vector_true.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(torch.ones((bs,1), device='cuda'))  # B * 1   # 表示所有将要判断的item都是有效的
        constraint.append(right_vector_false.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(torch.ones((bs,1), device='cuda'))  # B * 1

        sent_vector = self.logic_and(or_vector, right_vector_true) * left_valid \
                      + (-left_valid + 1) * right_vector_true  # B * V
        constraint.append(sent_vector.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1
        prediction_true = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])  # B
        check_list.append(('prediction_true', prediction_true))

        sent_vector = self.logic_and(or_vector, right_vector_false) * left_valid \
                      + (-left_valid + 1) * right_vector_false  # B * V
        constraint.append(sent_vector.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1
        prediction_false = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])  # B
        check_list.append(('prediction_false', prediction_false))

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)

        return prediction_true, prediction_false, check_list, constraint, constraint_valid
    
    def triple_loss(self, TItemScore, FItemScore):
        bce_loss = self.bceloss(TItemScore.sigmoid(), torch.ones_like(TItemScore)) + \
                    self.bceloss(FItemScore.sigmoid(), torch.zeros_like(FItemScore))
        # 输入正负例得分，使得分差距尽可能大
        if self.loss_sum == 1:
            loss = torch.sum(
            torch.nn.functional.softplus(-(TItemScore - FItemScore)))
        else:
            loss = torch.mean(
            torch.nn.functional.softplus(-(TItemScore - FItemScore)))
        return (loss+bce_loss)*0.5

    def l2_loss(self, users, pos, neg, history):
        users_embed, item_embed = self.encoder.computer()
        users_emb = users_embed[users]
        pos_emb = item_embed[pos]
        neg_emb = item_embed[neg]
        his_valid = history.ge(0).float()  # B * H
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V
        # L2 正则化损失
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2) +
                          elements.norm(2).pow(2))/float(len(users))
        if self.loss_sum == 0:
            reg_loss /= users.size(0)
        return reg_loss * self.l2s_weight

    def check(self, check_list):
        logging.info(os.linesep)
        for t in check_list:
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

    def forward(self, print_check:bool, return_pred:bool, *args, **kwards):
        prediction1, prediction0, check_list, constraint, constraint_valid = self.predict_or_and(*args, **kwards)
        rloss = self.logic_regularizer(False, check_list, constraint, constraint_valid)
        tloss = self.triple_loss(prediction1, prediction0)
        l2loss = self.l2_loss(*args, **kwards)

        if print_check:
            self.check(check_list)

        if return_pred:
            return prediction1, rloss+tloss+l2loss
        return rloss, tloss, l2loss
