import json
import random
import utility.config

from random import shuffle
from typing import List, Dict, Union


api_range = utility.config.api_range
# args = utility.config.args


def get_ma_api(data_file: str = '../origin dataset/inter.csv') -> List[Dict[str, Union[int, List[int]]]]:
    ma_api_list = []

    with open(file=data_file, mode='r') as fp:
        for ma_api in fp.readlines():
            ma_api = ma_api.strip('\n').split(',')
            ma_id = int(ma_api[0])
            api_list = [int(api) for api in ma_api[1:]]
            data = {'mashup_id': ma_id, 'api_list': api_list}
            ma_api_list.append(data)        # 所有的mashup都有文本描述

    return ma_api_list


def generate_shuffle_relation_file(relation_path: str) -> None:
    path = '../origin dataset/relation.json'
    ma_api_list = get_ma_api()
    shuffle(ma_api_list)
    with open(file=path, mode='w') as fp:
        for ma_api_data in ma_api_list:
            obj_s = json.dumps(obj=ma_api_data) + '\n'
            fp.write(obj_s)
#################################################################################


def read_relation_json(file: str = '../origin dataset/relation.json') -> List[Dict[str, Union[int, List[int]]]]:
    data = []
    with open(file=file, mode='r', encoding='utf8') as fp:
        for rela in fp.readlines():
            obj = json.loads(rela.strip('\n'))
            data.append(obj)

    return data


def generate_train_data(ma_api_list: List, file: str) -> None:
    with open(file=file, mode='w') as fp:
        for data in ma_api_list:
            mashup_id = data['mashup_id']
            api_list = data['api_list']
            for i in range(len(api_list)):
                # generate true sample
                # 第一次生成
                new_api_list = api_list.copy()
                new_api_list.pop(i)
                temp_data = {
                    'mashup_id': mashup_id,
                    'api_list': new_api_list,
                    'target_api': api_list[i],
                    'description_feature': '../origin dataset/ma_desc_feature/' + str(mashup_id) + '.txt',
                    'label': 1
                }
                obj_s = json.dumps(temp_data) + '\n'
                fp.write(obj_s)

                #######################################
                # generate false sample
                false_api_list = [x for x in range(1, api_range + 1) if x not in api_list]

                # 第二次随机生成
                false_api1 = random.choices(false_api_list, k=1)
                false_api1 = false_api1[0]
                temp_data2 = {
                    'mashup_id': mashup_id,
                    'api_list': new_api_list,
                    'target_api': false_api1,
                    'description_feature': '../origin dataset/ma_desc_feature/' + str(mashup_id) + '.txt',
                    'label': 0
                }
                obj_s = json.dumps(temp_data2) + '\n'
                fp.write(obj_s)

                #######################################
                # 第三次随机生成
                false_api1 = random.choices(false_api_list, k=1)
                false_api1 = false_api1[0]
                temp_data2 = {
                    'mashup_id': mashup_id,
                    'api_list': new_api_list,
                    'target_api': false_api1,
                    'description_feature': '../origin dataset/ma_desc_feature/' + str(mashup_id) + '.txt',
                    'label': 0
                }
                obj_s = json.dumps(temp_data2) + '\n'
                fp.write(obj_s)

                #######################################
                # 第四次随机生成
                false_api1 = random.choices(false_api_list, k=1)
                false_api1 = false_api1[0]
                temp_data2 = {
                    'mashup_id': mashup_id,
                    'api_list': new_api_list,
                    'target_api': false_api1,
                    'description_feature': '../origin dataset/ma_desc_feature/' + str(mashup_id) + '.txt',
                    'label': 0
                }
                obj_s = json.dumps(temp_data2) + '\n'
                fp.write(obj_s)


def main_training(json_relation_file: str = '../origin dataset/relation.json', n_fold: int = 5) -> None:
    ma_api_list = read_relation_json()
    length = len(ma_api_list)
    fold_num = int(length/n_fold)

    for i in range(n_fold):
        temp: List = []
        if i == 0:
            temp.extend(ma_api_list[fold_num * (i + 1):])
        else:
            temp.extend(ma_api_list[: fold_num * i])
            temp.extend(ma_api_list[(i+1)*fold_num: ])
        generate_train_data(temp, '../training data/train_' + str(i) + '_torch.json')


def generate_test_data(ma_api_list: List, file: str, remove_num: int = 1) -> None:
    with open(file=file, mode='w') as fp:
        for data in ma_api_list:
            mashup_id = data['mashup_id']
            api_list = data['api_list']

            removed_api_list = random.sample(api_list, remove_num)
            for r in removed_api_list:
                api_list.remove(r)

            temp = {
                'mashup_id': mashup_id,
                'api_list': api_list,
                'removed_api_list': removed_api_list,
                'description_feature': '../origin dataset/ma_desc_feature/' + str(mashup_id) + '.txt'
            }
            obj_s = json.dumps(obj=temp) + '\n'
            fp.write(obj_s)


def main_testing(json_relation_file: str = '../origin dataset/relation.json', remove_num: int = 1, n_fold: int = 5) -> None:
    ma_api_list = read_relation_json()
    length = len(ma_api_list)
    fold_num = int(length/n_fold)
    for i in range(n_fold):
        generate_test_data(ma_api_list[i*fold_num: (i+1)*fold_num], file='../testing data/testing_' + str(i) + '.json')


if __name__ == '__main__':
    # generate_shuffle_relation_file('../origin dataset/inter.csv')
    main_training()
    main_testing()