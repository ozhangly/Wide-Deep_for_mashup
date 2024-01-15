import numpy as np
import load_data

from typing import Dict, Union, List


args = load_data.args


def encode_api_list(api_list, api_range = load_data.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    for api in api_list:
        encoded_api[api - 1] = 1
    return encoded_api


def encode_api(api, api_range: int = load_data.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    encoded_api[api - 1] = 1
    return encoded_api


def cross_product_transformation(encoded_api_list, encoded_api) -> np.ndarray:
    cross_product_feature = np.array([x1*x2 for x1 in encoded_api_list for x2 in encoded_api])
    return cross_product_feature


# 对训练数据进行编码
def encode_data(data: Dict) -> Dict[str, Union[List, ]]:
    encoded_api_list = encode_api_list(api_list=data['api_list'])
    encoded_candidate_api = encode_api(api=data['target_api'])
    cross_product_feature = cross_product_transformation(encoded_api_list, encoded_candidate_api)

    mashup_description_feature = np.loadtxt(data['description_feature'])
    candidate_api_description_feature = np.loadtxt('.%s%d.txt' % (args.api_desc_path, data['target_api']))
    data = {
        'encoded_used_api': encoded_api_list,
        'encoded_candidate_api': encoded_candidate_api,
        'candidate_api': data['target_api'],
        'candidate_api_description_feature': candidate_api_description_feature,
        'mashup_description_feature': mashup_description_feature,
        'cross_product_used_candidate_api': cross_product_feature
    }

    return data

