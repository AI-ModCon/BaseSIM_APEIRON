def normalize_grad(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
