import torch
import numpy as np
import torch.nn as nn
import utility.config as config


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.W_K = nn.Linear(config.api_range, config.args.n_heads * config.args.d_k)     # W_K: [api_range, head * d_k]
        self.W_V = nn.Linear(config.api_range, config.args.n_heads * config.args.d_v)     # W_V: [api_range, head * d_v]
        self.W_Q = nn.Linear(512, config.args.n_heads * config.args.d_q)                  # W_Q: [512, head * d_q]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_k, input_v, input_q):     # input_k.shape == input_v.shape: [batch_size, max_used_api_num, api_range]   input_q: [batch_size, 512]
        batch_size = input_k.shape[0]
        K = self.W_K(input_k).reshape(batch_size, -1, config.args.n_heads, config.args.d_k).transpose(1, 2)    # K: [batch_size, head, max_used_api_num, d_k]
        V = self.W_V(input_v).reshape(batch_size, -1, config.args.n_heads, config.args.d_v).transpose(1, 2)    # V: [batch_size, head, max_used_api_num, d_v]
        Q = self.W_Q(input_q).reshape(batch_size, -1, config.args.d_q)                                         # Q: [batch_size, head, d_q]

        # Q在做乘法之前要先unsqueeze一下, 因为torch要求除了最后的两个维度其他的维度要相同, 确保Q能和K进行相乘
        Q = torch.unsqueeze(Q, dim=2)                                     # Q的维度: [batch_size, head, 1, d_q]

        attn_mat = torch.matmul(Q, K.transpose(-1, -2))                   # attn_mat: [batch_size, head, 1, max_used_api_num]
        attn_score = self.softmax(attn_mat/np.sqrt(K.shape[-1]))          # attn_score=attn_mat
        context = torch.matmul(attn_score, V)                             # context: [batch_size, head, 1, d_v]
        context = torch.squeeze(context, dim=2)                           # context: [batch_size, head, d_v]
        context = context.reshape(batch_size, -1)                         # context: [batch_size, head * d_v]

        return context


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()

        self.attn_layer = AttentionModule()

        self.deep = nn.Sequential(
            nn.Linear(config.args.n_heads * config.args.d_v + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.sigmoid_act_fun = nn.Sigmoid()

    def forward(self, input):

        input_k, input_v, input_q = input['encoded_api_context'], input['encoded_api_context'], input['candidate_api_description_feature']
        input_k = input_k.cpu().float()                     # input_k: [batch_size, max_used_api_num, api_range]
        input_v = input_v.cpu().float()                     # input_v: [batch_size, max_used_api_num, api_range]
        input_q = input_q.cpu().float()                     # input_q: [batch_size, 512]

        # 注意力部分
        api_context = self.attn_layer(input_k, input_v, input_q)        # api_context: [batch_size, head * d_v]

        # deep 部分
        mashup_description_feature = input['mashup_description_feature']
        mashup_description_feature = mashup_description_feature.cpu().float()

        deep_inputs = [mashup_description_feature, api_context]
        deep_input = torch.cat(deep_inputs, dim=1)                      # deep_input: [batch_size, head*dv + 512]
        deep_output = self.deep(deep_input)                             # deep_output: [batch_size, 1]

        output = self.sigmoid_act_fun(deep_output)

        return output
