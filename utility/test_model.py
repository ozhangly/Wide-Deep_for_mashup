import json
import torch
import Wide_Deep
import utility.config
import numpy as np

from typing import Dict, List
from utility.transcoding import encode_api_list, encode_api, \
    cross_product_transformation

args = utility.config.args


def encode_test_data(data: Dict) -> List:
    input_data: List = []
    api_list = data['api_list']
    for i in range(utility.config.api_range):
        target_api = i + 1
        if target_api not in api_list:
            encoded_candidate_api = encode_api(target_api)
            encoded_used_api = encode_api_list(api_list)
            input_data.append({
                'encoded_used_api': encoded_used_api,
                'encoded_candidate_api': encoded_candidate_api,
                'cross_product_used_candidate_api': cross_product_transformation(encoded_used_api, encoded_candidate_api),
                'mashup_description_feature': np.loadtxt(data['description_feature']),
                'api_description_feature': np.loadtxt(args.api_desc_path + str(target_api) + '.txt')
            })

    return input_data


def get_top_n_api(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_api = []
    for i in range(top_n):
        top_n_api.append(p_list[i][1])
    return top_n_api


def test_model(model_path: str) -> None:
    write_recommend_fp = open(file=args.output_path + args.recommend_res, mode='w')
    test_fp = open(file=args.testing_data_path + args.test_dataset, mode='r')
    model = Wide_Deep.WideDeep(utility.config.api_range)
    model.load_state_dict(torch.load(model_path))

    model = model(utility.config.device)
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

            set_true = set(removed_api) & set(top_n_api[:10])
            list_true = list(set_true)

            result_list.append(len(list_true) / 10.0)

            result = sum(result_list) / len(result_list)
            print(test_num)
            print(result)
            print('--------------------')

    test_fp.close()
    write_recommend_fp.close()


if __name__ == '__main__':
    fold: str = '4'
    path: str = 'model_wide_deep'
    test_model('./' + path + '/model_' + fold + '.pth')
