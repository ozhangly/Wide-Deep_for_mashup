import argparse


def arg_parse():
    args = argparse.ArgumentParser()

    args.add_argument('--training_data_path', nargs='?', default='./training data/', type=str)

    args.add_argument('--testing_data_path', nargs='?', default='./testing data/', type=str)

    args.add_argument('--mashup_desc_path', nargs='?', default='./origin dataset/ma_desc_feature/', type=str)

    args.add_argument('--api_desc_path', nargs='?', default='./origin dataset/api_desc_feature/', type=str)

    # 还有一些模型的参数
    return args.parse_args()
