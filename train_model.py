# 接下来就开始写这个了
import torch
import Wide_Deep
import utility.dataset
import utility.config

from tqdm import tqdm
from typing import List
from tensorboardX import SummaryWriter


args = utility.dataset.args

writer = SummaryWriter(logdir='./logdir')

model = Wide_Deep.WideDeep(utility.config.api_range)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data_loader = utility.dataset.get_dataloader()

# 定义一个损失函数
criterion = torch.nn.BCELoss()
loss_list: List = []

# 两个部分的优化方式不同


if __name__ == '__main__':
    model.train()
    train_bar = tqdm(desc='train process...', leave=True, total=len(data_loader))
    # 什么时候开始训练呢?
    for i in range(args.epoch):
        for idx, train_data in enumerate(data_loader):
            train_input = train_data[0].to(device)      # train_input: 这个比较复杂: [batch_size×Dict[str, Union[List, int, np.ndarray]]
            label = train_data[1].to(device)            # label: [batch_size, 1]
            output = model(train_input)                 # output: [batch_size, 1]
            # 这里有两个损失值，该怎么设计？



            
            writer.add_scalar(tag='loss value', scalar_value=, global_step=)
            train_bar.update()
            if i % 10 == 0:
                torch.save(model.state_dict(), args.save_weight_path + args.model_type + '/' + args.lr + '/')

    train_bar.close()
