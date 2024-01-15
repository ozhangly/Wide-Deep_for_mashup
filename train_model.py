# 接下来就开始写这个了
import torch
import Wide_Deep
import utility.dataset
import utility.config

import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter


args = utility.dataset.args

writer = SummaryWriter(logdir='./logdir')

model = Wide_Deep.WideDeep(utility.config.api_range)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data_loader = utility.dataset.get_dataloader()

# 定义一个损失函数
criterion = torch.nn.BCELoss()

# 两个部分的优化方式不同
wide_optimizer = optim.Adagrad(Wide_Deep.Wide.parameters(), lr=args.lr)
deep_optimizer = optim.Adam(Wide_Deep.Deep.parameters(), lr=args.lr)


if __name__ == '__main__':

    train_bar = tqdm(desc='train process...', leave=False, total=args.epoch)
    # 什么时候开始训练呢?
    for i in range(args.epoch):
        model.train()
        for idx, (train_input, label) in enumerate(data_loader):
            output = model(train_input)
                                                # output: [batch_size, 1]
            # 这里有两个损失值，该怎么设计？
            labels = label.to(device)

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
            writer.add_scalar(tag='wide loss curve', scalar_value=loss_wide.item(), global_step=global_step_idx)

            # 记录一下模型的参数
            for name, parameter in model.named_parameters():
                writer.add_histogram(name, parameter.data, global_step=global_step_idx)

        # 然后每10个epoch测试记录一下结果
        if (i+1) % 10 == 0:
            torch.save(model.state_dict(), args.save_weight_path + args.model_type + '/' + args.lr + '/')

        train_bar.update()
    #这里是最后训练结束的地方
    # 保存推荐结果，保存模型效果
    train_bar.close()
