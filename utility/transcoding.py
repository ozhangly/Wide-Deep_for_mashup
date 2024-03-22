import numpy as np
import utility.config as config

from numba import jit
from typing import Dict, Union, List


args = config.args

@jit
def encode_api_list(api_list, api_range=config.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    for api in api_list:
        encoded_api[api - 1] = 1
    return encoded_api

@jit
def encode_api(api, api_range: int = config.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    encoded_api[api - 1] = 1
    return encoded_api

@jit
def encode_api_context(api_list, api_range: int = config.api_range) -> np.ndarray:
    if len(api_list) == 0:
        return np.zeros(shape=(1, api_range))               # 如果api_list 为空, 初始化一个1 × api_range的二维0数组，目的是为了后面进行拼接
    return np.array([encode_api(api) for api in api_list])  # 为在data_list中的每一个api初始化一个one-hot向量

@jit
def cross_product_transformation(encoded_api_list, encoded_api) -> np.ndarray:
    cross_product_feature = np.array([x1*x2 for x1 in encoded_api_list for x2 in encoded_api])
    return cross_product_feature


# 对训练数据进行编码
def encode_data(data: Dict) -> Dict[str, Union[List, np.ndarray, int]]:

    encoded_api_context = encode_api_context(api_list=data['api_list'])

    mashup_description_feature = np.loadtxt(data['description_feature'])
    candidate_api_description_feature = np.loadtxt('%s%d.txt' % (args.api_desc_path, data['target_api']))
    data = {
        'encoded_api_context': encoded_api_context,
        'candidate_api_description_feature': candidate_api_description_feature,
        'mashup_description_feature': mashup_description_feature
    }

    return data

