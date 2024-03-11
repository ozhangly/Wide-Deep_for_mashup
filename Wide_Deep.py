import torch
import numpy as np
import torch.nn as nn
import utility.config as config

from torch.nn.functional import softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):                                             # Q: [batch_size, d_q], K: [batch_size, d_k], V: [batch_size, d_v]  d_k == d_q
        attn_mat = torch.matmul(Q, K.transpose(-1, -2))                     # attn_mat: [batch_size, batch_size]
        attn_score = softmax(attn_mat/np.sqrt(K.shape[-1]), dim=-1)         # attn_score: [batch_size, batch_size]
        context = torch.matmul(attn_score, V)                               # context: [batch_size, d_v]

        return context


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        # 需要三个线性层和一个Softmax
        self.W_K = nn.Linear(config.api_range, config.args.d_k, bias=False)
        self.W_V = nn.Linear(config.api_range, config.args.d_v, bias=False)
        self.W_Q = nn.Linear(config.api_range, config.args.d_q, bias=False)

    def forward(self, input_K, input_V, input_Q):                      # inputs : [batch_size, api_range]
        Q = self.W_Q(input_Q)                                          # Q: [batch_size, d_q]
        K = self.W_K(input_K)                                          # K: [batch_size, d_k]
        V = self.W_V(input_V)                                          # V: [batch_size, d_v]
        context = ScaledDotProductAttention()(Q, K, V)                 # context: [batch_size, d_v]

        return context


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()
        self.wide = nn.Sequential(
            nn.Linear(config.api_range + config.api_range + config.api_range * config.api_range, 1),
            nn.Sigmoid()
        )

        self.attn = AttentionLayer()

        self.deep = nn.Sequential(
            nn.Linear(config.args.d_v + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.deep_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.wide_deep = nn.Sigmoid()

    def forward(self, input):
        wide_inputs = [input['encoded_used_api'], input['encoded_candidate_api'],
                       input['cross_product_used_candidate_api']]
        wide_input = torch.cat(wide_inputs, dim=1).float()
        wide_input = wide_input.to(config.device)
        wide_output = self.wide(wide_input)

        input_k, input_v, input_q = input['encoded_used_api'], input['encoded_used_api'], input['encoded_candidate_api']

        api_context = self.attn(input_k, input_v, input_q)                          # context: [batch_size, d_v]

        deep_inputs = [input['mashup_description'], api_context, input['candidate_api_description']]
        deep_input = torch.cat(deep_inputs, dim=1).float()
        deep_input = deep_input.to(config.device)

        deep_output = self.deep(deep_input)
        output = self.wide_deep(wide_output + deep_output + self.deep_bias)

        return output
