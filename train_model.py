# 接下来就开始写这个了
import os
import torch
import Wide_Deep
import utility.dataset
import utility.config

import torch.optim as optim

from tqdm import tqdm
from typing import Union
from tensorboardX import SummaryWriter


args = utility.dataset.args

writer = SummaryWriter(logdir='./logdir')

model = Wide_Deep.WideDeep(utility.config.api_range)
model = model.to(utility.config.device)

data_loader = utility.dataset.get_dataloader()

# 定义一个损失函数
criterion = torch.nn.BCELoss()

# 两个部分的优化方式不同
wide_optimizer = optim.Adagrad(Wide_Deep.Wide.parameters(), lr=args.lr, weight_decay=args.weight_decay)
deep_optimizer = optim.Adam(Wide_Deep.Deep.parameters(), lr=args.lr, weight_decay=args.weight_decay)

fold: str = '4'
path: str = 'model_wide_deep'


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def train() -> None:
    train_bar = tqdm(desc='train process...', leave=False, total=args.epoch)
    # 什么时候开始训练呢?
    model.train()
    for i in range(args.epoch):
        for idx, (train_input, label) in enumerate(data_loader):
            output = model(train_input)  # output: [batch_size, 1]
            # 这里有两个损失值，该怎么设计？
            labels = label.to(utility.config.device)

            output = output.view(-1).float()
            loss_wide = criterion(output, labels)

            # 优化wide部分
            wide_optimizer.zero_grad()
            deep_optimizer.zero_grad()
            loss_wide.backward()
            deep_optimizer.step()
            wide_optimizer.step()

            ###########################################
            # 优化deep部分, 这两个部分是分开优化的---------> 这里可能最后要用，这里先保存着，如果需要在把上面的改一下
            # output = model(train_input)
            # output = output.view(-1).float()
            # loss_deep = criterion(output, labels)
            #
            # deep_optimizer.zero_grad()
            # loss_deep.backward()
            # deep_optimizer.step()
            ###########################################

            global_step_idx = i * args.train_batch_size + idx + 1
            # writer.add_scalar(tag='deep loss curve', scalar_value=loss_deep.item(), global_step=global_step_idx)
            writer.add_scalar(tag='loss curve', scalar_value=loss_wide.item(), global_step=global_step_idx)

            # 记录一下模型的参数
            for name, parameter in model.named_parameters():
                writer.add_histogram(name, parameter.data, global_step=global_step_idx)

        # 然后每10个epoch记录模型
            if idx % 100 == 0:
                torch.save(wide_optimizer.state_dict(), './' + path + '/model_' + fold + '.pth')
                torch.save(deep_optimizer.state_dict(), './' + path + '/wide_optimizer_' + fold + '.pth')
                torch.save(model.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        torch.save(wide_optimizer.state_dict(), './' + path + '/model_' + fold + '.pth')
        torch.save(deep_optimizer.state_dict(), './' + path + '/wide_optimizer_' + fold + '.pth')
        torch.save(model.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')

        train_bar.update()
    train_bar.close()


if __name__ == '__main__':
    # 如果继续训练就加载模型
    if args.continue_training:
        model.load_state_dict(torch.load('./' + path + '/model_' + fold + '.pth'))
        wide_optimizer.load_state_dict(torch.load('./' + path + '/wide_optimizer_' + fold + '.pth'))
        deep_optimizer.load_state_dict(torch.load('./' + path + '/deep_optimizer_' + fold + '.pth'))

    train()
