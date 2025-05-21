import torch

class ExpCosFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        exp_x = torch.exp(x)
        cos_y = torch.cos(y)
        ctx.save_for_backward(exp_x, y) #юзаем save_for_backward
        return exp_x + cos_y

    @staticmethod
    def backward(ctx, grad_output):
        exp_x, y = ctx.saved_tensors
        grad_x = grad_output * exp_x
        grad_y = grad_output * (-torch.sin(y))
        return grad_x, grad_y

def test_exp_cos_function():
    torch.manual_seed(0)
    x = torch.randn(5, requires_grad=True)
    y = torch.randn(5, requires_grad=True)

    x1 = x.clone().detach().requires_grad_(True) # кастомная ф
    y1 = y.clone().detach().requires_grad_(True)
    out_custom = ExpCosFunction.apply(x1, y1)
    out_custom.sum().backward()
    grad_x_custom = x1.grad.clone()
    grad_y_custom = y1.grad.clone()

    x2 = x.clone().detach().requires_grad_(True) # обычный вариант
    y2 = y.clone().detach().requires_grad_(True)
    out_native = torch.exp(x2) + torch.cos(y2)
    out_native.sum().backward()
    grad_x_native = x2.grad
    grad_y_native = y2.grad

    print("Forward is the same:", torch.allclose(out_custom, out_native))
    print("Градиент по x is the same:", torch.allclose(grad_x_custom, grad_x_native))
    print("Градиент по y is the same:", torch.allclose(grad_y_custom, grad_y_native))

test_exp_cos_function()
