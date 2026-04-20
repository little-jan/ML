import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)
# output is tensor(5.) tensor(6.) tensor(1.5000) tensor(9.)

x = torch.arange(3)
print(x)
# output is tensor([0, 1, 2])

print(x.shape)
# output is torch.Size([3])
# we can access the length of the vector via the .shape as well as len(x)
# .shape indicates the tensor's length across each axis


# TENSOR ARITHMETIC:
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
print(A,'\n', A + B)
# output is:
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])
#  tensor([[ 0.,  2.,  4.],
#         [ 6.,  8., 10.]])
print(A*B)
# output is:
# tensor([[ 0.,  1.,  4.],
#         [ 9., 16., 25.]])

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('this is X: \n', X)
print('this is a + X: \n',a + X, '\n this is the last one: \n', (a * X).shape)
# output is:
# this is X:
#  tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
# this is a + X:
#  tensor([[[ 2,  3,  4,  5],
#          [ 6,  7,  8,  9],
#          [10, 11, 12, 13]],
#
#         [[14, 15, 16, 17],
#          [18, 19, 20, 21],
#          [22, 23, 24, 25]]])
#  this is the last one:
#  torch.Size([2, 3, 4])


# DOT PRODUCT
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
torch.sum(x * y)  # this is the dot product

# MATRIX MULTIPLICATION
B = torch.ones(3, 4)
torch.mm(A, B), A@B  # both of these are ways to do matrix multiplication

u = torch.tensor([3.0, -4.0])
torch.norm(u)  # calculates the l2 norm (abs of vector)
torch.abs(u).sum()  # calculates the l1 norm (less sensitive to outliers than l2 norm)