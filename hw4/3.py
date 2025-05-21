import torch
import torch.optim as optim
import math

class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not (0.0 <= lr):
            raise ValueError("Invalid lr: {}".format(lr))
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError("Invalid beta0: {}".format(betas[0]))
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError("Invalid beta1: {}".format(betas[1]))
        if not (0.0 <= weight_decay):
            raise ValueError("Invalid wei_dec value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)

                grad = p.grad
                momentum = state['momentum']

                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                update = momentum * beta1 + grad * (1 - beta1)
                update = update.sign()
                new_momentum = momentum * beta2 + grad * (1 - beta2)

                if weight_decay != 0:
                    update = update + p * weight_decay
                p.add_(update, alpha=-lr)
                state['momentum'] = new_momentum

        return loss

def manual_lion_step(weight, gradient, momentum, lr, beta1, beta2, weight_decay):
    update_val = (1 - beta1) * gradient + beta1 * momentum
    update_signed = torch.sign(update_val)

    new_momentum = (1 - beta2) * gradient + beta2 * momentum
    update_with_wd = update_signed + weight * weight_decay
    new_weight = weight - update_with_wd * lr

    return new_weight, new_momentum

def test_lion_manual():
    print("Тест сравнения с ручным расчетом")
    print("\n")
    torch.manual_seed(42)

    lr = 1e-4
    betas = (0.9, 0.99)
    weight_decay = 0.01

    param = torch.nn.Parameter(torch.randn(5))
    param.grad = torch.randn(5)

    initial_param_value = param.data.clone()
    initial_grad_value = param.grad.data.clone()
    initial_momentum = torch.randn(5)

    optimizer = Lion([param], lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer.state[param]['momentum'] = initial_momentum.clone()

    expected_param_value, expected_momentum_value = manual_lion_step(
        initial_param_value, initial_grad_value, initial_momentum, lr, betas[0], betas[1], weight_decay)

    print("Ожидаемое знач параметра:", expected_param_value)
    print("Ожидаемое знач момента:", expected_momentum_value)
    optimizer.step()
    print("\n")

    actual_param_value = param.data
    actual_momentum_value = optimizer.state[param]['momentum']
    print("Фактич знач параметра:", actual_param_value)
    print("Фактич знач момента:", actual_momentum_value)

    param_match = torch.allclose(actual_param_value, expected_param_value, atol=1e-6)
    momentum_match = torch.allclose(actual_momentum_value, expected_momentum_value, atol=1e-6)

    print("\n")
    print("Значение параметра совпадает с расчетом:", param_match)
    print("Значение момента совпадает с расчетом:", momentum_match)
    assert param_match, "Ошибочка: Знач параметра после шага Lion не совпад с расчетом("
    assert momentum_match, "Ошибочка: Знач момента после шага Lion не совпад с расчетом("

    print("Тест пройден)")

test_lion_manual()

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_simple_model():
    print("Обучение простой модели с Lion")
    torch.manual_seed(42)

    input_dim = 10
    output_dim = 1
    learning_rate = 1e-4
    betas = (0.9, 0.99)
    weight_decay = 0.01
    num_epochs = 1000
    batch_size = 32

    model = SimpleModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = Lion(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    X = torch.randn(batch_size * num_epochs, input_dim)
    true_weights = torch.arange(1, input_dim + 1).float()
    true_bias = torch.tensor([5.0])
    Y = X @ true_weights.unsqueeze(-1) + true_bias + torch.randn(X.shape[0], 1) * 0.5

    print(f"Нач веса: {model.linear.weight.data[:2]}...")
    print(f"Нач сдвиг: {model.linear.bias.data}")

    for epoch in range(num_epochs):
        start_idx = epoch * batch_size
        end_idx = start_idx + batch_size
        inputs = X[start_idx:end_idx]
        targets = Y[start_idx:end_idx]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Эпоха [{epoch+1}/{num_epochs}], Лосс: {loss.item():.4f}')

    print(f"Конечные веса: {model.linear.weight.data[:2]}...")
    print(f"Конечный сдвиг: {model.linear.bias.data}")

train_simple_model()
