import torch
import torch.nn as nn

rnn = nn.LSTMCell(10, 20)
input = torch.randn(6, 3, 10)
print(input)
hx = torch.randn(3, 20)
print(hx)
cx = torch.randn(3, 20)
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
    print(output)