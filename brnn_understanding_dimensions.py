"""
Demonstrating how a bidirectional RNN cell works
"""

import numpy as np
from torch import nn
from torch import from_numpy, zeros


class RNN_debug(nn.Module):
    def __init__(self, input_size, seq_len, output_size, hidden_dim, n_layers):
        super(RNN_debug, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):

        
        # get RNN outputs, hidden is  unused
        r_out, hidden = self.rnn(x, hidden)
        
        
        forward_out = r_out[:, :, :self.hidden_dim ]  
        backwards_out = r_out[:, :, self.hidden_dim:]
      
        
        
        print("output of brnn cell: ")
        print(r_out)
        print("")
        
        print("output of forward brnn cell: ")
        print(forward_out)
        print("")
        
        print("output of backward brnn cell: ")
        print(backwards_out)
        print("")
        print("forward and backward sepearte are not used, just for information")
        print("the stacked output vector is passed forward")
        print("")
        
        
        print(" last hidden state of rnn cell: ")
        print(hidden)
        print("")
        

        
        
        
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out[:, self.seq_len-1]
        
        print("selecting only last element of the outupt sequence, as this goes to the fc layer: ")
        print(r_out)
        print("")


        # get final output
        r_out = self.fc(r_out)
        
        print("fully connected layer: ")
        print(r_out)
        print("")
        
        
        output = self.sigmoid(r_out)
        
        print("sigmoid layer: ")
        print(output)
        print("")
        
        
        
        
        return output
        
    
    def initHidden(self):
        return zeros(1, self.seq_len, self.hidden_dim)


X = np.array([[[0,1], [1,0], [1,0]],
              [[1,1], [1,0], [1,1]],
              [[0,1], [0,0], [1,0]],
              [[0,0], [1,0], [1,0]]])
y = np.array([1,1,0,0])


#X = np.array([[[0,1], [1,0], [1,0], [1,1]]])
#y = np.array([1])



N, seq_len, dummy_dim = X.shape

X_T = from_numpy(X).float()
y_T = from_numpy(y).float()


input_size=dummy_dim
hidden_dim=5
n_layers=1
output_size=1

model = RNN_debug(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)


hidden = zeros(2, seq_len, hidden_dim)


print("N: ", N)
print("seq_len: ", seq_len)
print("dummy_dim: ", dummy_dim)
print("hidden_dim: ", hidden_dim)
print("")

print("input to network: ")
print(X_T)
print("")

y = model.forward(X_T, hidden)


