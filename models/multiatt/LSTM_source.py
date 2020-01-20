import torch.nn as nn
import torch
from torch.autograd import Variable

class source_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(source_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fw_cell = nn.LSTMCell(input_size,hidden_size)
        self.bw_cell = nn.LSTMCell(input_size,hidden_size)
        
        for p in self.parameters():
            p.requires_grad=True


    def forward(self, x, mask):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
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
        steps = x.size(1) 
        for i in range(steps):
            curx = x[:, i, :]
            curx_bw = x[:,steps-i-1,:]

            fw_h, fw_c = self.fw_cell(curx, (fw_h, fw_c))
            bw_h, bw_c = self.bw_cell(curx_bw, (bw_h, bw_c))
           
            hidden = torch.cat((fw_h,bw_h),dim = -1) #(batch_size,2*hidden_dim)
            outputs += [hidden]

        outputs = torch.stack(outputs,1) #(batch_size,time_steps,2*hidden_dim)
        return outputs

"""
LSTM = MYLSTM(768,512,512)
inputs = Variable(torch.Tensor(64, 40, 768))
outputs = LSTM(inputs)
print(outputs.size())
#print(outputs)
"""
