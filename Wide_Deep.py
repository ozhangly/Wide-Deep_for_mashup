import torch
import torch.nn as nn


class Wide(nn.Module):
    def __init__(self, api_range: int):
        super(Wide, self).__init__()
        # 有一个线性层
        self.linear_layer = nn.Linear(api_range * (api_range + 2), 1)
        self.activate_f = nn.Sigmoid()

    def forward(self, input):                   # 输入的是什么？
        output = self.linear_layer(input)
        output = self.activate_f(output)
        return output


class Deep(nn.Module):
    def __init__(self, api_range: int):
        super(Deep, self).__init__()
        self.block = nn.Sequential(
            # 很多线性层
            nn.Linear(512 + api_range * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        output = self.block(input)
        return output


class WideDeep(nn.Module):
    def __init__(self, api_range: int):
        super(WideDeep, self).__init__()
        self.wide = Wide(api_range)
        self.deep = Deep(api_range)

        self.w_wide = torch.randn(size=(1, 1), requires_grad=True)
        self.w_deep = torch.randn(size=(1, 1), requires_grad=True)
        self.bias   = torch.randn(size=(1, 1), requires_grad=True)

        self.output_layer = nn.Sigmoid()

    def forward(self, input):
        wide_input = [input['encoded_used_api'],
                      input['encoded_candidate_api'],
                      input['cross_product_used_candidate_api']]
        deep_input = [input['encoded_used_api'],
                      input['encoded_candidate_api'],
                      input['mashup_description_feature'],
                      input['candidate_api_description_feature']]

        deep_input = torch.cat(deep_input, dim=1).cuda()
        wide_input = torch.cat(wide_input, dim=1).cuda()
        wide_output = self.wide(wide_input)
        deep_output = self.deep(deep_input)

        output = wide_output @ self.w_wide + deep_output @ self.w_deep + self.bias

        output = self.output_layer(output)

        return output
