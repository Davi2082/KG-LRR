# coding=utf-8

import logging
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import utils
import dataloader
from functools import partial
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2


def TransR_train(Recmodel, opt):
    Recmodel.train()
    kgloader = DataLoader(Recmodel.kg_dataset, batch_size=4096, drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].cuda()
        relations = data[1].cuda()
        pos_tails = data[2].cuda()
        neg_tails = data[3].cuda()
        kg_batch_loss = Recmodel.calc_kg_loss_transE(
            heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()

def BPR_train_original(dataset, recommend_model, optimizer, batch_size, epoch, w=None):
    dataset.mode = 'train'
    recommend_model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=12)

    total_batch = len(dataloader)
    aver_r, aver_t, aver_l2 = 0., 0., 0.
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_users = train_data[0].long().cuda()
        batch_pos = train_data[1].long().cuda()
        batch_neg = train_data[2].long().cuda()
        batch_history = train_data[3].long().cuda()
        rloss, tloss, l2loss = recommend_model(batch_i==len(dataloader)-1, 0, batch_users, batch_pos, batch_neg, batch_history)
        l_all = rloss + tloss + l2loss
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_r += rloss.cpu().item() / total_batch
        aver_t += tloss.cpu().item() / total_batch
        aver_l2 += l2loss.cpu().item() / total_batch
        if w:
            w.add_scalar(f'Loss', l_all, epoch *
                        len(dataloader) + batch_i)
    logging.info(f"AVERloss {aver_r+aver_t+aver_l2:.3f} \
        = r {aver_r:.3f} +t {aver_t:.3f} +l2 {aver_l2:.3f}")

def test_one_batch(topks, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, u_batch_size, topks, epoch, w=None, multicore=0):
    # eval mode with no dropout
    dataset.mode = 'test'
    dataloader = DataLoader(dataset, batch_size=u_batch_size,
                             num_workers=12,
                             collate_fn=utils.collate_test_data)
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    assert u_batch_size <= dataset.testSize / 10, \
                f"test_u_batch_size is too big for this dataset, try a small one {dataset.testSize // 10}"

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    pfunc = partial(test_one_batch, topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
               
    with torch.no_grad():
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        for (batch_users, batch_his, batch_items) in tqdm(dataloader, total=len(dataloader)):             
            batch_users = batch_users.cuda()
            batch_his = batch_his.cuda()
            batch_items = batch_items.cuda()
            rating = Recmodel.just_predict(batch_users, batch_his)
            #排除掉训练中已知的用户商品对
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(batch_his):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(batch_items)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(pfunc, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(pfunc(x))
        for result in pre_results:
            results['recall'] = results['recall'] + result['recall']
            results['precision'] = results['precision'] + result['precision']
            results['ndcg'] = results['ndcg'] + result['ndcg']
        results['recall'] /= float(len(dataset))
        results['precision'] /= float(len(dataset))
        results['ndcg'] /= float(len(dataset))
        # results['auc'] = np.mean(auc_record)
        if w:
            w.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        logging.info(results)
        return results["recall"] 
