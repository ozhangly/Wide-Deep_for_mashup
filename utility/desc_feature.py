import os
import string
import torch
import pandas as pd
import numpy as np

from typing import List, Dict
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')


def read_data(path: str, mashup_or_api: str) -> Dict:
    name2id = dict()
    df = pd.read_csv(path, encoding='utf8')
    title_or_url = 'title' if mashup_or_api in ['mashup'] else 'url'
    for idx, data in df.iterrows():
        name2id[data[title_or_url]] = int(data['id'])
    return name2id


def remove_symbol(sentence: str) -> str:
    sentence = sentence.translate(str.maketrans({key: None for key in string.punctuation}))
    sentence = sentence.translate(str.maketrans({key: None for key in string.digits}))
    sentence = sentence.strip()
    return sentence


def get_sentences(description: str) -> List:
    # 先对description进行分句
    sentences = [sentence.strip().lower() for sentence in description.strip().split('.') if sentence != '']
    filter_sentence = []
    for sentence in sentences:
        # 对句子过滤一下
        sentence = remove_symbol(sentence)
        filter_sentence.append(sentence)

    return filter_sentence


def get_sentence_feature(text: str) -> torch.Tensor:
    with torch.no_grad():
        sentence_vector = model.encode(text, convert_to_tensor=True)
    return sentence_vector


def get_sentences_feature(sentences: List) -> np.ndarray:
    # 构建模型获得句子的特征
    feature_list: List = []
    for sentence in sentences:
        feature = get_sentence_feature(sentence)
        feature_list += [feature]
    feature_concat = torch.stack(feature_list, dim=0)
    feature_average = torch.mean(feature_concat, dim=0)

    array = feature_average.numpy()
    return array


if __name__ == '__main__':
    # 需要处理mashup和cloud api的文本描述
    ma_desc: str = '../origin dataset/ma_desc_feature/'
    api_desc: str = '../origin dataset/api_desc_feature/'
    # 先对mashup文本向量化
    ma_df = pd.read_csv('../cloud api dataset/mashups_detail.csv', encoding='utf8').loc[:, ['title', 'description']]
    api_df = pd.read_csv('../cloud api dataset/apis_detail.csv', encoding='ISO-8859-1').loc[:, ['url', 'description']]

    ma_name2id = read_data(path='../origin dataset/mashup detail.csv', mashup_or_api='mashup')
    api_name2id = read_data(path='../origin dataset/cloud api detail.csv', mashup_or_api='api')

    for idx, data in ma_df.iterrows():
        if pd.isna(data['description']):
            continue
        mashup_id = ma_name2id[data['title']]
        if os.path.exists(ma_desc + str(mashup_id) + '.txt'):
            continue
        sentences = get_sentences(data['description'])
        array = get_sentences_feature(sentences)
        np.savetxt(ma_desc + str(mashup_id) + '.txt', array)

    del ma_name2id, ma_df, ma_desc

    for idx, data in api_df.iterrows():
        if pd.isna(data['description']):        # 如果api描述为空，先不生成这个api的文本向量
            continue
        try:
            api_id = api_name2id[data['url'].strip('\n')]
        except KeyError:
            continue
        sentences = get_sentences(data['description'])
        array = get_sentences_feature(sentences)
        np.savetxt(api_desc + str(api_id) + '.txt', array)

    del api_name2id, api_df, api_desc
