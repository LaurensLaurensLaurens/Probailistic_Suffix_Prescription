"""
Use of same experimental setup as for our Probabilistic Suffix Prediction.

Reimplementation for comparison: 
- Paper: Weytjens, Hans, and Jochen De Weerdt. "Learning uncertainty with artificial neural networks for predictive process monitoring." Applied Soft Computing 125 (2022): 109134.
- Github (code) from: https://github.com/hansweytjens/uncertainty-remaining_time/blob/main/LSTM.ipynb

Delete terms:
- ConcreteDropout: We assume fixed non-learnable dropout rate.
- Bayes: We assume to be always bayesian.
- stop_dropout: We assume to no stop dropout during inference.

"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple

class StochasticLSTMCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 weight_reg: float,
                 p_fix: float,
                 device: str = 'cpu'):
        '''
        ARGUMENTS:
        input_size: number of features (after embedding layer)
        hidden_size: number of nodes in LSTM layers
        weight_reg: weight regularization parameter
        p_fix: dropout parameter used in case of not self.concrete
        device: device to run the model on ('cpu' or 'cuda')
        '''
        super(StochasticLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device(device)
        self.p_logit = torch.full([1], p_fix).to(self.device)
        self.wr = weight_reg

        self.Wi = nn.Linear(self.input_size, self.hidden_size).to(self.device)
        self.Wf = nn.Linear(self.input_size, self.hidden_size).to(self.device)
        self.Wo = nn.Linear(self.input_size, self.hidden_size).to(self.device)
        self.Wg = nn.Linear(self.input_size, self.hidden_size).to(self.device)

        self.Ui = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)

        self.init_weights()

    def init_weights(self):
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt().to(self.device)

        self.Wi.weight.data.uniform_(-k, k)
        self.Wi.bias.data.uniform_(-k, k)

        self.Wf.weight.data.uniform_(-k, k)
        self.Wf.bias.data.uniform_(-k, k)

        self.Wo.weight.data.uniform_(-k, k)
        self.Wo.bias.data.uniform_(-k, k)

        self.Wg.weight.data.uniform_(-k, k)
        self.Wg.bias.data.uniform_(-k, k)

        self.Ui.weight.data.uniform_(-k, k)
        self.Ui.bias.data.uniform_(-k, k)

        self.Uf.weight.data.uniform_(-k, k)
        self.Uf.bias.data.uniform_(-k, k)

        self.Uo.weight.data.uniform_(-k, k)
        self.Uo.bias.data.uniform_(-k, k)

        self.Ug.weight.data.uniform_(-k, k)
        self.Ug.bias.data.uniform_(-k, k)

    def _sample_mask(self, batch_size):
        '''
        ARGUMENTS:
        batch_size: batch size

        OUTPUTS:
        zx: dropout masks for inputs. Tensor (GATES x batch_size x input size (after embedding))
        zh: dropout masks for hidden states. Tensor (GATES x batch_size x number hidden states)
        '''
        p = torch.sigmoid(self.p_logit)
        GATES = 4
        eps = torch.tensor(1e-7).to(self.device)
        t = 1e-1

        ux = torch.rand(GATES, batch_size, self.input_size).to(self.device)
        uh = torch.rand(GATES, batch_size, self.hidden_size).to(self.device)

        if self.input_size == 1:
            zx = (1 - torch.sigmoid((torch.log(eps) - torch.log(1 + eps) + torch.log(ux + eps) - torch.log(1 - ux + eps)) / t))
        else:
            zx = (1 - torch.sigmoid((torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(ux + eps) - torch.log(1 - ux + eps)) / t)) / (1 - p)
            
        zh = (1 - torch.sigmoid((torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(uh + eps) - torch.log(1 - uh + eps)) / t)) / (1 - p)

        return zx, zh

    def regularizer(self):
        '''
        OUTPUTS:
        self.wr * weight_sum: weight regularization in reformulated ELBO
        self.wr * bias_sum: bias regularization in reformulated ELBO
        '''
        p = torch.sigmoid(self.p_logit)
        weight_sum = torch.tensor([torch.sum(params ** 2) for name, params in self.named_parameters() if name.endswith("weight")]).sum() / (1. - p)
        bias_sum = torch.tensor([torch.sum(params ** 2) for name, params in self.named_parameters() if name.endswith("bias")]).sum()
        
        return self.wr * weight_sum, self.wr * bias_sum

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        '''
        ARGUMENTS:
        input: Tensor (sequence length x batch size x input size(after embedding))
       
        OUTPUTS:
        hn: tensor of hidden states h_t. Dimension (sequence_length x batch_size x hidden size)
        h_t: hidden states at time t. Dimension (batch size x hidden size (number of nodes in LSTM layer))
        c_t: cell states. Dimension (batch size x hidden size (number of nodes in LSTM layer))
        '''
        device = input.device
        seq_len = input.shape[0]
        batch_size = input.shape[1]

        if hx is None:
            h_t = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=device)
        else:
            h_t = hx[0]
            c_t = hx[1]
            
        hn = torch.empty(seq_len, batch_size, self.hidden_size, dtype=input.dtype, device=device)
        zx, zh = self._sample_mask(batch_size=batch_size)
        
        zx = zx.to(device)
        zh = zh.to(device)
        
        for t in range(seq_len):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t

        return hn, (h_t, c_t)