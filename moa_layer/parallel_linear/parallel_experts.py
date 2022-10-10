import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


class ParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size, weight, bias=None):
        output_list = []
        expert_size_list = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        for i in range(weight.size(0)):
            if bias is not None:
                o_i = torch.mm(input_list[i], weight[i]) + bias[i]
            else:
                o_i = torch.mm(input_list[i], weight[i])
            output_list.append(o_i)
        output = torch.cat(output_list, dim=0)
        variables = (input, expert_size, weight, bias)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        num_linears = weight.size(0)

        expert_size_list = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_list = []
        for i in range(num_linears):
            d_input_list.append(torch.einsum('bi,ji->bj', grad_list[i], weight[i]))
        d_input = torch.cat(d_input_list, dim=0)

        d_weight_list = []
        for i in range(num_linears):
            d_weight_list.append(torch.einsum('bi,bj->ij', input_list[i], grad_list[i]))
        d_weight = torch.stack(d_weight_list, dim=0)

        if bias is not None:
            d_bias_list = []
            for i in range(num_linears):
                d_bias_list.append(grad_list[i].sum(0))
            d_bias = torch.stack(d_bias_list, dim=0)
        else:
            d_bias = None
        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.b = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(self.w, -a, a)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
        return results
