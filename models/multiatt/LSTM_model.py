import torch.nn as nn
import torch
from torch.autograd import Variable
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MYLSTM(nn.Module):

    def __init__(self, input_size, knowledge_size, hidden_size):
        super(MYLSTM, self).__init__()
        self.input_size = input_size
        self.knowledge_size = knowledge_size
        self.hidden_size = hidden_size
       
        self.fw_cell = nn.LSTMCell(input_size,hidden_size)
        self.bw_cell = nn.LSTMCell(input_size,hidden_size)

        self.weight_W_a = Parameter(torch.Tensor(512,512))
        self.weight_U_a = Parameter(torch.Tensor(512,512))
        self.bias_b_a = Parameter(torch.Tensor(512))
        self.weight_w_a = Parameter(torch.Tensor(512,1))

        self.weight_U_lambda = Parameter(torch.Tensor(512,512))
        self.weight_C_lambda = Parameter(torch.Tensor(512,512))
        self.bias_b_lambda = Parameter(torch.Tensor(512))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        for name,weight in self.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal(weight)
            if name.startswith("bias"):
                nn.init.constant(weight,0)

    def forward(self, x, y, mask):
        """
        Parameters
        ----------
        inputs x: ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)

        inputs y: ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, knowledge_dim)

        mask : ``torch.LongTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_dim).
        """
        fw_h, fw_c = Variable(torch.zeros(x.size(0), self.hidden_size)).cuda(),\
                       Variable(torch.zeros( x.size(0), self.hidden_size)).cuda()
        bw_h, bw_c = Variable(torch.zeros(x.size(0), self.hidden_size)).cuda(),\
                       Variable(torch.zeros(x.size(0), self.hidden_size)).cuda()

        outputs = []
        batch_size,steps,_ = x.shape

        source = y @ self.weight_U_a # we precompute it to reduce computation cost, because it does not depend on k
        for k in range(steps):
            curx = x[:, k, :]
            curx_bw = x[:, x.size(1)-k-1, :]

            hidden = torch.cat((fw_h,bw_h),dim = -1) #[b,hidden_dim*2]
            hidden_tmp = (hidden @ self.weight_W_a).unsqueeze(1).repeat(1,steps,1) #[b,steps,hidden_dim*2]
            matrix = ((hidden_tmp + source  + self.bias_b_a) @ self.weight_w_a).squeeze(-1) #[b,steps]

            attention_weights = F.softmax(matrix,dim=1) #[b,steps]
            s_k = (y * (attention_weights.unsqueeze(-1))).sum(dim=1) #[b,hidden_dim*2]
            lambda_k = torch.sigmoid(hidden @ self.weight_U_lambda + s_k @ self.weight_C_lambda + self.bias_b_lambda)
            
            hidden = (1-lambda_k) * hidden + lambda_k * s_k #[b,hidden_dim*2]
            fw_h = hidden[:,0:self.hidden_size]
            bw_h = hidden[:,self.hidden_size:]
            
            fw_h, fw_c = self.fw_cell(curx, (fw_h, fw_c))
            bw_h, bw_c = self.bw_cell(curx_bw, (bw_h, bw_c))

            outputs += [hidden] 
        outputs = torch.stack(outputs,1) #[b,steps,hidden_dim*2]
        return outputs

"""
LSTM = MYLSTM(768,512,512)
inputs = Variable(torch.Tensor(64, 40, 768))
outputs = LSTM(inputs)
print(outputs.size())
#print(outputs)
"""
