import torch

sample = torch.normal(0, 1., (1, 128))
samples = sample.repeat(100, 1) 
interp = torch.linspace(-1, 1, 10)
# X, Y = torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10))
print(interp.repeat(10, 1).reshape(-1, 1))
print(interp.repeat(1, 10).reshape(-1, 1))
samples[:, :2] = torch.cat((interp.repeat(10, 1).reshape(-1, 1), interp.repeat(1, 10).reshape(-1, 1)), 1).cuda()