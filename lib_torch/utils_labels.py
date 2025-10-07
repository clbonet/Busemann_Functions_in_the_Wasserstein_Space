import torch


def pad_and_group_by_class(x, labels):
    """
    Args:
        x: Tensor of shape (n, d)
        labels: LongTensor of shape (n,), values in 0..C-1

    Returns:
        out: Tensor of shape (C, nc, d), with padded zeros
        weights_by_class: Tensor of shape (C, nc), with padded zeros or uniform weights
        weights_by_samples: Tensor of shape (C, nc), with padded zeroes or uniform weights
    """
    n, d = x.shape
    device = x.device
    dtype = x.dtype

    # Compute number of classes (C) and max samples per class (nc)
    C = int(labels.max().item()) + 1
    class_counts = torch.bincount(labels.type(torch.int), minlength=C)
    nc = class_counts.max().item()

    # Initialize output tensors
    out = torch.zeros(C, nc, d, device=device, dtype=dtype)
    weights_by_class = torch.zeros(C, nc, device=device, dtype=dtype)  # weights matrix, uniform by class
    weights_by_samples = torch.zeros(C, nc, device=device, dtype=dtype)  # weights matrix, uniform on the full set of samples

    for c in range(C):
        class_mask = (labels == c)
        class_x = x[class_mask]  # (num_samples_c, d)
        num_samples = class_x.size(0)

        out[c, :num_samples] = class_x
        weights_by_class[c, :num_samples] = 1.0 / class_counts[c]
        weights_by_samples[c, :num_samples] = 1.0 / len(labels)

    return out, weights_by_class, weights_by_samples
