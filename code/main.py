# coding=utf-8
import sys, os, time
import argparse
import dataloader
import logging
import numpy as np
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Model', add_help=False)
    parser.add_argument('--config', type=str, default="/home/wangshirui/recommend/KGLRR/config/yelp2018.yaml",
                             help='Path to the config file')
    # ------ Runner setting ------
    parser.add_argument('--path', type=str, default="/home/wangshirui/recommend/KGLRR",
                             help='Path to the root')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--check_epoch', type=int, default=1, help='Check every epochs.')
    parser.add_argument('--early_stop_cnt', type=int, default=10,
                        help='Max counting for early stop, -1 for no early stop.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight of l2_regularize in pytorch optimizer.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size during training.')
    parser.add_argument('--test_u_batch_size', type=int,default=1024,
                        help="the batch size of users for testing")
    parser.add_argument('--test_start_epoch', type=int, default=1, help='Starting to test.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability for each deep layer')
    parser.add_argument('--l2_bias', type=int, default=0,
                        help='Whether add l2 regularizer on bias.')
    parser.add_argument('--l2_loss', type=float, default=0.0,
                        help='Weight of l2_regularize in loss.')
    parser.add_argument('--ssl_reg', type=float, default=0.1,
                        help='Regularize parameters for ssl loss')
    parser.add_argument('--pretrain_kgc', type=bool, default=False,
                        help='Setting True means training only for kg')
    
    # ------ Base setting ------
    parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--test_verbose', type=int, default=1, help='the period to load test result')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--pretrain_file', type=str, default='', help='pretrained weight file to load')
    parser.add_argument('--model_save_path', type=str, default='', help='path to save trained model')
    parser.add_argument('--tensorboard', type=int, default=0, help="enable tensorboard")
    parser.add_argument('--board_path', type=str, default='', help="path to save tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--device', type=str, default="2", help='GPU index')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    
    # ------ Data setting ------
    parser.add_argument('--dataset', type=str, default="MIND", help='Dataset to use')
    parser.add_argument('--maxhis', type=int, default=10, help='Maxmum to load history')
    parser.add_argument('--topks', nargs='?',default="[20]", help="@k test list")
    parser.add_argument('--loss_sum', type=bool, default=False,
                        help='sum/mean the loss')

    # ------ Model setting ------
    parser.add_argument('--model', type=str, default="KGLRR",
                        help='Choose which model')
    parser.add_argument('--kgcn', type=str, default="RGAT",
                        help='Aggerate model for graph')
    parser.add_argument('--latent_dim_rec', type=int,default=64,
                        help="the embedding size")
    parser.add_argument('--lightGCN_n_layers', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--A_n_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--A_split', type=bool, default=False,
                        help='Whether to split adj matrix')

    parser.add_argument('--r_logic', type=float, default=0.1,
                        help='Weight of logic regularizer loss')
    parser.add_argument('--r_length', type=float, default=0.001,
                        help='Weight of vector length regularizer loss')
    parser.add_argument('--sim_scale', type=int, default=10,
                        help='Expand the raw similarity *sim_scale before sigmoid.')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of or/and/not hidden layers.')
    parser.add_argument('--explain', type=bool, default=True,
                        help='Whether output the explaination or not.')
                        
    parser.add_argument('--train_trans', type=bool, default=False,
                        help='Whether to train transe epoch')
    parser.add_argument('--entity_num_per_item', type=int, default=10,
                        help='Number of entity according to the item')
    parser.add_argument('--uicontrast', type=str, default="RANDOM",
                        help='Contrasting model for ui interaction, choosing from WEIGHTED / RANDOM / ITEM-BI / PGRACE /NO')
    parser.add_argument('--keep_prob', type=float, default=0.8, help='')
    parser.add_argument('--kgc_temp', type=float, default=0.2, help='')
    parser.add_argument('--kg_p_drop', type=float, default=0.5,
                        help='Probability for kg to drop')
    parser.add_argument('--ui_p_drop', type=float, default=0.001,
                        help='Probability for ui interaction to drop')
    parser.add_argument('--mix_ratio', type=float, default=0,
                        help='For WEIGHTED-MIX contrast mode, set the mixture ratio')
    parser.add_argument('--kgc_enable', type=bool, default=True, help='')
    parser.add_argument('--kgc_joint', type=bool, default=True, help='')
    parser.add_argument('--use_kgc_pretrain', type=bool, default=False, help='')
    
    return parser.parse_args()

def load_config(args):
    ori_opt = vars(args)
    opt = yaml.load(open(ori_opt['config']), Loader=yaml.FullLoader)
    ori_opt.update(opt) # 以config文件中的内容为准
    return ori_opt

def logging_init(config):   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=config['log_file'], level=config['verbose'], filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('===========config================')
    logging.info("KGCN:{}, TransR:{}, N:{}".format(config['kgcn'], config['train_trans'], config['entity_num_per_item']))
    logging.info("KGC: {} @ d_prob:{} @ joint:{} @ from_pretrain:{}".format(config['kgc_enable'], config['kg_p_drop'],\
                                                                     config['kgc_joint'], config['use_kgc_pretrain']))
    logging.info("UIC: {} @ d_prob:{} @ temp:{} @ reg:{}".format(config['uicontrast'], config['ui_p_drop'],\
                                                                 config['kgc_temp'], config['ssl_reg']))
    message = '\n'.join([f'{k:<20}: {v}' for k, v in config.items()])
    logging.info(message)
    logging.info('===========end===================')

def main():
    args = get_args()
    config = load_config(args)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']  # default '0'

    # 确定device之后再import torch相关
    import Procedure
    import torch
    from KGLRR import KGLRR

    # save setting
    file_name = [config['uicontrast'], config['dataset'], str(config['lightGCN_n_layers']), str(config['latent_dim_rec']), config['prefix']]
    file_name = '_'.join(file_name)
    config['log_file'] = f'{config["path"]}/log/{file_name}.log'
    config['result_file'] = f'{config["path"]}/result/{file_name}.npy'
    if config['model_save_path'] == '':
        config['model_save_path'] = f'{config["path"]}/result/{file_name}.pt'
    logging_init(config)
    logging.info(f"# used cuda device: {config['device']}")

    # random seed
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # tensorboard init
    if config['tensorboard']:
        w : SummaryWriter = SummaryWriter(
                                        os.path.join(config['board_path'], time.strftime("%m-%d") + "-" + config['dataset'])
                                        )
    else:
        w = None
        logging.info("not enable tensorflowboard")

    # data & model
    if config['dataset'] in ['movielens', 'last-fm', 'MIND', 'yelp2018', 'amazon-book']:
        dataset = dataloader.HisLoader(config)
        kg_dataset = dataloader.KGDataset(f'{config["path"]}/data/{config["dataset"]}/kg.txt', \
                                            config["entity_num_per_item"])
        # 还需要改成不同的设置对应不同模型名称
    Recmodel = KGLRR(config, dataset, kg_dataset).cuda()
    if config['pretrain']:
        try:
            Recmodel.load_state_dict(torch.load(config['pretrain_file'],map_location=torch.device('cpu')))
            logging.info(f"loaded model weights from {config['pretrain_file']}")
        except FileNotFoundError:
            logging.info(f"{config['pretrain_file']} not exists, start from beginning")

    # optim & schedu
    weight_p, bias_p = [], []
    for name, p in Recmodel.named_parameters():
        if not p.requires_grad:
            continue
        if 'bias' in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    if config['l2_bias'] == 1:
        optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': config['weight_decay']}]
    else:
        optimize_dict = [{'params': weight_p, 'weight_decay': config['weight_decay']},
                            {'params': bias_p, 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(optimize_dict, lr=config['lr'])
    if config['dataset'] == "MIND":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.2)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)
    
    try:
        # for early stop
        # recall@20
        least_loss = 1e5
        best_result = 0.
        stopping_step = 0

        for epoch in tqdm(range(config['epoch']), disable=True):
            # transR learning
            if epoch%3 == 0:
                if config['train_trans']:
                    logging.info("[Trans]")
                    trans_loss = Procedure.TransR_train(Recmodel, optimizer)
                    logging.info(f"trans Loss: {trans_loss:.3f}")

            
            # joint learning part
            if not config['pretrain_kgc']:
                Procedure.BPR_train_original(dataset, Recmodel, optimizer, epoch, w=w)
                
                if epoch<config['test_start_epoch']:
                    if epoch %5 == 0:
                        logging.info("[TEST]")
                        Procedure.Test(dataset, Recmodel, config['test_u_batch_size'], config['topks'], epoch, w, config['multicore'])
                else:
                    if epoch % config['test_verbose'] == 0:
                        logging.info("[TEST]")
                        recall = Procedure.Test(dataset, Recmodel, config['test_u_batch_size'], config['topks'], epoch, w, config['multicore'])
                        if recall[-1] > best_result:
                            stopping_step = 0
                            best_result = recall[-1]
                            logging.info("find a better model")
                            torch.save(Recmodel.state_dict(), config['model_save_path'])
                        else:
                            stopping_step += 1
                            if stopping_step >= config['early_stop_cnt']:
                                logging.info(f"early stop triggerd at epoch {epoch}")
                                break
            
            scheduler.step()
    finally:
        if config['tensorboard']:
            w.close()

if __name__ == "__main__":
    main()