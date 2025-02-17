# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import time
import json
from pprint import pprint

import torch
import numpy as np
from humanize import naturalsize

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop
from test.dataset import DataSet

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA


@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list, is_rerank, gemp, rgem, sgem, onemeval, depth, imgs_per_query):
    torch.backends.cudnn.benchmark = False
    model.eval()
    state_dict = model.state_dict()

    # initialize modules
    MDescAug_obj = MDescAug()
    RerankwMDA_obj = RerankwMDA()

    model.load_state_dict(state_dict)
    for dataset_db, dataset_q in dataset_list:
        text = f'>> {dataset_db}: Global Retrieval for scale {str(scale_list)} with CVNet-Global'
        print(text)
        
        print("extract database features")
        cache_path = os.path.join(data_dir, dataset_db, 'features_sg.pt')
        if os.path.exists(cache_path):
            print(f'load features from cache: {cache_path}')
            X = torch.load(cache_path, map_location='cpu', weights_only=False)
        else:
            X = extract_feature(model, data_dir, dataset_db, "db", [1.0], gemp, rgem, sgem, scale_list)
            torch.save(X, cache_path)
            
            cache_size = os.path.getsize(cache_path)
            cache_size = naturalsize(cache_size)
            print(f'{cache_size=}')

        if not dataset_q:
            return
        print("extract query features")
        Q = extract_feature(model, data_dir, dataset_q, "query", [1.0], gemp, rgem, sgem, scale_list)
        
        # cfg = config_gnd(dataset,data_dir)
        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()
        
        print("perform global feature reranking")
        sim = torch.matmul(X, Q.T) # 6322 70
        ranks = torch.argsort(-sim, axis=0) # 6322 70
        if is_rerank:
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy().T
        
        dataset_q = DataSet(data_dir, dataset_q, "query", scale_list)
        dataset_db = DataSet(data_dir, dataset_db, "db", scale_list)
        q_img_paths = dataset_q.get_data_paths()
        k = imgs_per_query
        
        # img_idxs = np.unique(ranks[:, :k]).flatten().tolist()
        img_idxs = ranks[:, :k]
        img_paths = dataset_db.get_im_paths(img_idxs)
        
        data = {q_img_path: paths for q_img_path, paths in zip(q_img_paths, img_paths)}
        pprint(data)
        out_file = os.path.join(data_dir, 'sim.json')
        with open(out_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)  # Indent for better readability

