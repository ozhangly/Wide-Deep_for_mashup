import argparse


def arg_parse():
    args = argparse.ArgumentParser()

    # 训练集路径
    args.add_argument('--training_data_path', nargs='?', default='./training data/', type=str)

    # 测试集路径
    args.add_argument('--testing_data_path', nargs='?', default='./testing data/', type=str)

    # api文本描述的路径
    args.add_argument('--api_desc_path', nargs='?', default='./origin dataset/api_desc_feature/', type=str)

    # 输出结果路径
    args.add_argument('--output_path', nargs='?', default='./output/', type=str)

    # 训练数据集
    args.add_argument('--dataset', nargs='?', default='train_0_torch.json', type=str)

    # 测试数据集
    args.add_argument('--test_dataset', nargs='?', default='testing_0.json', type=str)

    # 还有一些模型和训练的参数
    args.add_argument('--d_k', nargs='?', default=64, type=int)

    args.add_argument('--d_v', nargs='?', default=64, type=int)

    args.add_argument('--d_q', nargs='?', default=64, type=int)

    args.add_argument('--continue_training', nargs='?', default=0, type=int)

    args.add_argument('--train_batch_size', nargs='?', default=256, type=int)

    args.add_argument('--test_batch_size', nargs='?', default=64, type=int)

    args.add_argument('--epoch', nargs='?', default=10, type=int)

    args.add_argument('--lr', nargs='?', default=0.01, type=float)

    args.add_argument('--weight_decay', nargs='?', default=0.0001, type=float)

    return args.parse_args()
