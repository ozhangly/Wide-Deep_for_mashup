import torch
import torch.nn as nn
import utility.config as config


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()
        self.wide = nn.Sequential(
            nn.Linear(config.api_range + config.api_range + config.api_range * config.api_range, 1),
            nn.Sigmoid()
        )
        # self.deep_tpl_embedding = nn.Embedding(config.tpl_range + 1, 512)
        self.deep = nn.Sequential(
            nn.Linear(config.api_range + config.api_range + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.ReLU()
        )
        self.deep_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.wide_deep = nn.Sigmoid()

    def forward(self, input):
        wide_inputs = [input['encoded_used_api'], input['encoded_candidate_api'],
                       input['cross_product_used_candidate_api']]
        wide_input = torch.cat(wide_inputs, dim=1).float()
        wide_input = wide_input.to(config.device)
        # wide_input = torch.tensor(wide_input, dtype=torch.float32)
        wide_output = self.wide(wide_input)

        deep_inputs = [input['encoded_used_api'], input['encoded_candidate_api'], input['mashup_description_feature'], input['candidate_api_description_feature']]
        deep_input = torch.cat(deep_inputs, dim=1).float()
        deep_input = deep_input.to(config.device)

        deep_output = self.deep(deep_input)
        output = self.wide_deep(wide_output + deep_output + self.deep_bias)

        return output
