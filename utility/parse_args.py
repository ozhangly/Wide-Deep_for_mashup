import argparse


def arg_parse():
    args = argparse.ArgumentParser()

    args.add_argument('--training_data_path', nargs='?', default='./training data/', type=str)

    args.add_argument('--testing_data_path', nargs='?', default='./testing data/', type=str)

    args.add_argument('--mashup_desc_path', nargs='?', default='./origin dataset/ma_desc_feature/', type=str)

    args.add_argument('--api_desc_path', nargs='?', default='./origin dataset/api_desc_feature/', type=str)

    args.add_argument('--output_path', nargs='?', default='./output/', type=str)

    args.add_argument('--save_weight_path', nargs='?', default='./weight/', type=str)

    args.add_argument('--dataset', nargs='?', default='train_0_torch.json', type=str)

    args.add_argument('--test_dataset', nargs='?', default='testing_0.json', type=str)

    args.add_argument('--recommend_res', nargs='?', default='testing_WD_0.json', type=str)

    args.add_argument('--result', nargs='?', default='result.csv', type=str)

    # 还有一些模型和训练的参数
    args.add_argument('--continue_training', nargs='?', default=0, type=int)

    args.add_argument('--train_batch_size', nargs='?', default=256, type=int)

    args.add_argument('--test_batch_size', nargs='?', default=64, type=int)

    args.add_argument('--epoch', nargs='?', default=100, type=int)

    args.add_argument('--lr', nargs='?', default=0.01, type=float)

    args.add_argument('--weight_decay', nargs='?', default=0.0001, type=float)

    return args.parse_args()
