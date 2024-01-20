import json
import torch
import metrics
import Wide_Deep
import utility.config
import numpy as np

from typing import Dict, List
from utility.transcoding import encode_api_list, encode_api, \
    cross_product_transformation

args = utility.config.args


def encode_test_data(data: Dict) -> Dict:
    api_list = data['api_list']

    batch_encoded_used_api = []
    batch_encoded_candidate_api = []
    batch_cross_product_used_candidate_api = []
    batch_mashup_description_feature = []
    batch_api_description_feature = []

    for i in range(utility.config.api_range):
        target_api = i + 1
        if target_api not in api_list:
            encoded_candidate_api = encode_api(target_api)
            encoded_used_api = encode_api_list(api_list)
            cross_product_used_candidate_api = cross_product_transformation(encoded_used_api, encoded_candidate_api)
            mashup_description_feature = np.loadtxt(data['description_feature'])
            api_description_feature = np.loadtxt(args.test_api_desc_path + str(target_api) + '.txt')

            batch_api_description_feature.append(api_description_feature)
            batch_mashup_description_feature.append(mashup_description_feature)
            batch_cross_product_used_candidate_api.append(cross_product_used_candidate_api)
            batch_encoded_candidate_api.append(encoded_candidate_api)
            batch_encoded_used_api.append(encoded_candidate_api)

    input_data = {
        'encoded_candidate_api': torch.tensor(batch_encoded_candidate_api, dtype=torch.float32),
        'encoded_used_api': torch.tensor(batch_encoded_used_api, dtype=torch.float32),
        'cross_product_used_candidate_api': torch.tensor(batch_cross_product_used_candidate_api, dtype=torch.float32),
        'mashup_description_feature': torch.tensor(batch_mashup_description_feature, dtype=torch.float32),
        'api_description_feature': torch.tensor(batch_api_description_feature, dtype=torch.float32)
    }

    return input_data


def get_top_n_api(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_api = []
    for i in range(top_n):
        top_n_api.append(p_list[i][1])
    return top_n_api


def get_performance(user_pos_test, r, auc, ks) -> Dict:
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
        'map': np.array(map), 'mrr': mrr, 'fone': np.array(fone)
    }


def test_one(pos_test, user_rating) -> Dict:
    r = []
    for i in user_rating:
        if i in pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return get_performance(pos_test, r, auc, [1, 3, 5, 10])


def test_model(model_path: str) -> None:

    result = {
        'precision': np.zeros(4), 'ndcg': np.zeros(4), 'map': np.zeros(4),
        'recall': np.zeros(4), 'mrr': np.zeros(4), 'fone': np.zeros(4)
    }
    # 写结果指标指针
    result_file = open(file=args.output_path + args.result, mode='w')
    # 推荐结果指针
    write_recommend_fp = open(file=args.output_path + args.recommend_res, mode='w')
    # 读取测试文件指针
    test_fp = open(file=args.testing_data_path + args.test_dataset, mode='r')

    model = Wide_Deep.WideAndDeep()
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

            removed_api = test_obj['removed_api']

            num = 0
            for i in range(utility.config.api_range):
                target_api = i + 1
                if target_api not in test_obj['api_list']:
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
            result['precision'] += res['precision']
            result['map'] += res['map']
            result['mrr'] += res['mrr']
            result['fone'] += res['fone']
            result['ndcg'] += res['ndcg']
            result['recall'] += res['recall']

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

    result_file.write(result_content)

    result_file.close()


if __name__ == '__main__':
    fold: str = '4'
    path: str = 'model_wide_deep'
    test_model('./' + path + '/model_' + fold + '.pth')
