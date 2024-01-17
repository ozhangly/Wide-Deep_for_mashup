import torch
import torch.nn as nn
import utility.config


class Wide(nn.Module):
    def __init__(self, api_range: int):
        super(Wide, self).__init__()
        # 有一个线性层
        self.linear_layer = nn.Linear(in_features=api_range * (api_range + 2), out_features=1, bias=True)
        self.output_layer = nn.Linear(in_features=2, out_features=1, bias=True)
        self.activate_f = nn.Sigmoid()

    def forward(self, input, deep_output):
        output = self.linear_layer(input)
        output = self.activate_f(output)

        final_input = torch.cat((output, deep_output), dim=1)           # final_input: [batch_size, 2]
        output = self.output_layer(final_input)
        return output


class Deep(nn.Module):
    def __init__(self, api_range: int):
        super(Deep, self).__init__()
        self.block = nn.Sequential(
            # 很多线性层
            nn.Linear(1024 + api_range * 2, 512),
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
        self.output_activate_f = nn.Sigmoid()

    def forward(self, input):                       # batch_size长的字典列表
        wide_input = [input['encoded_used_api'],            # 这个不用管，pytorch自动整合了
                      input['encoded_candidate_api'],
                      input['cross_product_used_candidate_api']]
        deep_input = [input['encoded_used_api'],
                      input['encoded_candidate_api'],
                      input['mashup_description_feature'],
                      input['candidate_api_description_feature']]

        deep_input = torch.cat(deep_input, dim=1).float().to(utility.config.device)
        wide_input = torch.cat(wide_input, dim=1).float().to(utility.config.device)

        deep_output = self.deep(deep_input)                 # deep_output: [batch_size, 1]
        wide_output = self.wide(wide_input, deep_output)

        output = self.output_activate_f(wide_output)

        return output
