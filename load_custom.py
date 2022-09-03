import torch
import numpy as np
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

'''checkpoint = torch.load('latest_log_dir/best.pth.tar')
epoch = checkpoint['epoch']
state_dict = checkpoint['state_dict']
arch = checkpoint['arch']

for param_tensor in state_dict:
    print(param_tensor, "\t", state_dict[param_tensor].size())'''


model = torch.load('model.pt', map_location='cpu')


model_children = list(model.module.children())
#print(model_children)

module1 = model_children[:4]
#module2 = model_children[4:-1]

#x1 = torch.tensor(np.random.randint(-128, 127, size=(3, 1, 28, 28), dtype=np.int64)).type(torch.FloatTensor)
x2 = torch.tensor(np.random.randint(-128, 127, size=(1, 20, 7, 7), dtype=np.int64)).type(torch.FloatTensor)

m1 = nn.Sequential(*module1)
#m2 = nn.Sequential(*module2)
m2= nn.Sequential(*[nn.Flatten(start_dim=1), model_children[-1]])
print(m2)
#print("\n")
print(m2(x2))