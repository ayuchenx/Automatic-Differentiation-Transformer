from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        A, B = input_values
        dims = list(range(-len(node.normalized_shape), 0))
        
        AB = A @ B
        mean = torch.mean(AB, dim = dims, keepdim = True)
        variance = torch.var(AB, dim = dims, keepdim = True, unbiased = False)
        res = (AB - mean) / torch.sqrt(variance + node.eps)
        return res
        

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        A, B = node.inputs
        AB = matmul(A, B)
        dims = tuple(range(-len(node.normalized_shape), 0)) 
        mean_val = mean(AB, dim = dims, keepdim = True)
        variance = power(sub(AB, mean_val), 2)
        variance = mean(variance, dim = dims, keepdim = True)
        
        g_mean = mean(output_grad, dim = dims, keepdim = True)
        another_mean = mean(mul(output_grad, sub(AB, mean_val)), dim = dims, keepdim = True)
        
        res = sub(output_grad, g_mean)
        t1 = div(sub(AB, mean_val), add_by_const(variance, node.eps))
        
        res = sub(res, mul(t1, another_mean))
        res = div(res, sqrt(add_by_const(variance, node.eps)))
        
        dA = matmul(res, transpose(B, -1, -2))
        dB = matmul(transpose(A, -1, -2), res)
        
        return [dA, dB]
        


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        A, B = input_values
        AB = A @ B
        max_val, _ = torch.max(AB, dim = node.dim, keepdim = True)
        numerator = torch.exp(AB- max_val)
        denominator = torch.sum(numerator, dim = node.dim, keepdim = True)
        return numerator / denominator



    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        A, B = node.inputs
        AB = matmul(A, B)
        soft = softmax(AB, node.dim)
        summed = sum_op(soft * output_grad, dim = node.dim, keepdim = True)
        soft_grad = soft * (output_grad - summed)
        
        dA = matmul(soft_grad, transpose(B, -1, -2))
        dB = matmul(transpose(A, -1, -2), soft_grad)
        
        return [dA, dB]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()