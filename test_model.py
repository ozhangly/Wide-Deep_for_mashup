import os
import re
import json
import torch
import Wide_Deep_Plus
import utility.config

import numpy as np
import utility.metrics as metrics

from typing import Dict, List
from utility.transcoding import encode_api_context

args = utility.config.args
fold = re.findall('[0-9]', args.dataset)[0]       # 从训练数据集中读取折数
ks: List = [1, 3, 5, 10]                          # 推荐指标的top_n


def encode_test_data(data: Dict) -> Dict:
    api_list = data['api_list']

    batch_mashup_description_feature = []
    batch_api_description_feature = []

    encoded_api_context = encode_api_context(api_list=api_list)     # encoded_api_context: [used_api_num, 512]

    for i in range(utility.config.api_range):
        target_api = i + 1
        if target_api not in api_list:
            mashup_description_feature = np.loadtxt(data['description_feature'])
            api_description_feature = np.loadtxt(args.api_desc_path + str(target_api) + '.txt')

            batch_api_description_feature.append(api_description_feature)
            batch_mashup_description_feature.append(mashup_description_feature)

    input_data = {
        'encoded_api_context': torch.tensor(encoded_api_context, dtype=torch.float32).repeat(len(batch_mashup_description_feature), 1, 1),  # encoded_api_context: [batch_size, used_api_num,512]
        'mashup_description_feature': torch.tensor(batch_mashup_description_feature, dtype=torch.float32),      # mashup_description_feature: [batch_size, 512]
        'candidate_api_description_feature': torch.tensor(batch_api_description_feature, dtype=torch.float32)   # api_description_feature: [batch_size, 512]
    }

    return input_data


def get_top_n_api(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_api = []
    for i in range(top_n):
        top_n_api.append(p_list[i][1])
    return top_n_api


def get_performance(user_pos_test, r, auc) -> Dict:
    precision, recall, ndcg, \
    map, fone, mrr = [], [], [], [], [], []

    for k in ks:
        precision.append(metrics.precision_at_k(r, k))
        recall.append(metrics.recall_at_k(r, k, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, k, 1))
        map.append(metrics.average_precision(r, k))
        mrr.append(metrics.mrr_at_k(r, k))
        fone.append(metrics.F1(metrics.precision_at_k(r, k), metrics.recall_at_k(r, k, len(user_pos_test))))

    return {
        'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg),
        'map': np.array(map), 'mrr': np.array(mrr), 'fone': np.array(fone)
    }


def test_one(pos_test, user_rating) -> Dict:
    r = []
    for i in user_rating:
        if i in pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return get_performance(pos_test, r, auc)


def test_model(model_path: str) -> None:

    results = {
        'precision': np.zeros(len(ks)), 'ndcg': np.zeros(len(ks)), 'map': np.zeros(len(ks)),
        'recall': np.zeros(len(ks)), 'mrr': np.zeros(len(ks)), 'fone': np.zeros(len(ks))
    }

    result_file = args.output_path + 'result_' + fold + '.csv'
    recommend_file = args.output_path + 'testing_WD_' + fold + '.json'

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    # 推荐结果指标指针
    result_file_fp = open(file=result_file, mode='w')
    # 推荐结果指针
    write_recommend_fp = open(file=recommend_file, mode='w')
    # 读取测试文件指针
    test_fp = open(file=args.testing_data_path + args.test_dataset, mode='r')

    model = Wide_Deep_Plus.WideAndDeep()
    model.load_state_dict(torch.load(model_path))

    model = model.to(utility.config.device)
    result_list: List = []

    test_num: int = 0
    model.eval()
    with torch.no_grad():
        for lines in test_fp.readlines():
            test_obj = json.loads(lines.strip('\n'))
            test_num += 1
            inputs = encode_test_data(test_obj)

            outputs = model(inputs)
            outputs = outputs.view(-1).tolist()
            probability_list = []

            removed_api = test_obj['removed_api_list']

            api_list = test_obj['api_list']

            num = 0
            for i in range(utility.config.api_range):
                target_api = i + 1
                if target_api not in api_list:
                    probability_list.append((outputs[num], target_api))
                    num += 1

            top_n_api = get_top_n_api(probability_list, 10)
            write_data = {
                'mashup_id': test_obj['mashup_id'],
                'recommend_api': top_n_api,
                'removed_apis': test_obj['removed_api_list']
            }
            write_content = json.dumps(write_data) + '\n'
            write_recommend_fp.write(write_content)

            res = test_one(pos_test=removed_api, user_rating=top_n_api)
            results['precision'] += res['precision']
            results['map'] += res['map']
            results['mrr'] += res['mrr']
            results['fone'] += res['fone']
            results['ndcg'] += res['ndcg']
            results['recall'] += res['recall']

            set_true = set(removed_api) & set(top_n_api[:10])
            list_true = list(set_true)

            result_list.append(len(list_true) / 10.0)

            result = sum(result_list) / len(result_list)
            print(test_num)
            print(result)
            print('--------------------')

    test_fp.close()
    write_recommend_fp.close()

    result['precision'] /= test_num
    result['recall'] /= test_num
    result['ndcg'] /= test_num
    result['map'] /= test_num
    result['mrr'] /= test_num
    result['fone'] /= test_num

    result_content = '%s\n%s\n%s\n%s\n%s\n%s\n' % (
        ','.join(['%.5f' % r for r in result['precision']]),
        ','.join(['%.5f' % r for r in result['ndcg']]),
        ','.join(['%.5f' % r for r in result['recall']]),
        ','.join(['%.5f' % r for r in result['map']]),
        ','.join(['%.5f' % r for r in result['mrr']]),
        ','.join(['%.5f' % r for r in result['fone']])
    )

    result_file_fp.write(result_content)

    result_file_fp.close()


if __name__ == '__main__':
    path: str = 'model_wide_deep_plus'
    test_model('./' + path + '/model_' + fold + '.pth')
