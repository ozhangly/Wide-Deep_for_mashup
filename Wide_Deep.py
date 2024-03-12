import torch
import numpy as np
import torch.nn as nn
import utility.config as config

from torch.nn.functional import softmax


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.W_K = nn.Linear(config.api_range, config.args.d_k)     # W_K: [api_range, d_k]
        self.W_V = nn.Linear(config.api_range, config.args.d_v)     # W_V: [api_range, d_v]
        self.W_Q = nn.Linear(config.api_range, config.args.d_q)     # W_Q: [api_range, d_q]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_k, input_v, input_q):   # input_k: [batch_size, api_range]  input_q=input_k  input_v: [batch_size, d_v]
        K = self.W_K(input_k)               # K: [batch_size, d_k]
        V = self.W_V(input_v)               # V: [batch_size, d_v]
        Q = self.W_Q(input_q)               # Q: [batch_size, d_q]    d_k == d_q

        attn_mat = torch.matmul(Q, K.transpose(-1, -2))             # attn_mat: [batch_size, batch_size]
        attn_score = self.softmax(attn_mat/np.sqrt(K.shape[-1]))    # attn_score=attn_mat
        context = torch.matmul(attn_score, V)                       # context: [batch_size, d_v]

        return context


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()

        self.wide = nn.Sequential(
            nn.Linear(config.api_range + config.api_range + config.api_range * config.api_range, 1),
            nn.Sigmoid()
        )

        self.attn_layer = AttentionModule()

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
        wide_input = torch.cat(wide_inputs, dim=1).float()                        # wide_input: [batch_size, api_range^2 + 2*api_range]
        wide_input = wide_input.to(config.device)
        wide_output = self.wide(wide_input)

        input_k, input_v, input_q = input['encoded_used_api'], input['encoded_used_api'], input['encoded_candidate_api']
        input_k = input_k.to(config.device).float()
        input_v = input_v.to(config.device).float()
        input_q = input_q.to(config.device).float()

        api_context = self.attn_layer(input_k, input_v, input_q)            # input_k=input_q: [batch_size, d_q], input_v: [batch_size, d_q]

        mashup_description_feature = input['mashup_description_feature']
        candidate_api_description_feature = input['candidate_api_description_feature']
        mashup_description_feature = mashup_description_feature.to(config.device).float()
        candidate_api_description_feature = candidate_api_description_feature.to(config.device).float()

        deep_inputs = [mashup_description_feature, api_context, candidate_api_description_feature]
        deep_input = torch.cat(deep_inputs, dim=1)                              # deep_input: [batch_size, dv + 1024]
        deep_output = self.deep(deep_input)

        output = self.wide_deep(wide_output + deep_output + self.deep_bias)

        return output
