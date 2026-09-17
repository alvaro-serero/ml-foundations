import numpy as np
import torch
import torch.nn.functional as F

from deep_learning.simple_convolutional_2d_layer import simple_conv2d
from pytorch.layer_norm_from_scratch import layer_norm


def test_simple_conv2d_matches_known_output():
    input_matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])

    output = simple_conv2d(input_matrix, kernel, padding=0, stride=1)

    expected = np.full((2, 2), -4.0)
    np.testing.assert_allclose(output, expected)


def test_layer_norm_matches_pytorch_reference():
    x = torch.tensor(
        [[1.0, 2.0, 4.0], [3.0, 6.0, 9.0]],
        dtype=torch.float32,
    )
    gamma = torch.tensor([1.0, 0.5, 2.0])
    beta = torch.tensor([0.0, 1.0, -1.0])

    output = layer_norm(x, gamma, beta)
    expected = F.layer_norm(x, normalized_shape=(3,), weight=gamma, bias=beta)

    torch.testing.assert_close(output, expected)
