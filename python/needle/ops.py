"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        # TODO: test PowerScalar gradient function
        return out_grad*self.scalar*power_scalar(node.inputs[0],self.scalar-1)

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a/b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad/rhs, out_grad*(-lhs/(rhs**2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a/self.scalar

    def gradient(self, out_grad, node):
        return out_grad/self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        tmpAxes=[i for i in range(len(array_api.shape(a))) ]
        if(self.axes is not None):
            tmpAxes[self.axes[0]],tmpAxes[self.axes[1]]=tmpAxes[self.axes[1]],tmpAxes[self.axes[0]]
            return array_api.transpose(a,axes=tmpAxes)
        tmpAxes[-1],tmpAxes[-2]=tmpAxes[-2],tmpAxes[-1]
        return array_api.transpose(a,axes=tmpAxes)

    def gradient(self, out_grad, node):
        return transpose(out_grad,axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a,self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
 #https://github.com/bettersemut/dlsys_hw2/blob/8b16e4ecac6cf5d5efb2c4840f9107cdfe64e00b/python/needle/ops.py#L222
        # (2,3) => (4,2,3) not ok for (2, 3, 4)
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        # 只能在最前面加维度，后面只能做1->x的提升
        # 分两步，一步是新增维度做sum，去除axis
        # 第二步是保留dim的sum
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        if len(shape_in) != len(shape_out):
            out_grad = summation(out_grad, tuple(i for i in range(len(shape_out) - len(shape_in))))
        axes = []
        for i, dim in enumerate(shape_in):
            if dim == 1:
                axes.append(i)
        # print("axes:\t", axes)
        return summation(out_grad, tuple(axes)).reshape(shape_in)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a,axis=self.axes)

    def gradient(self, out_grad, node):
        #https://github.com/hnyoumfk/dlsyshw/blob/057e7a9e9c9497d8a2f576ef734b5fd4347cdf5f/hw1/python/needle/ops.py#L207
        a = node.inputs[0]
        new_shape = list(a.shape)
        # print("self.axes :",self.axes)
        if self.axes is None:
            self.axes = list(range(len(a.shape)))
        if type(self.axes) is int:
            new_shape[self.axes]=1
        else:
            for i in self.axes:
                new_shape[i] = 1
        g =  broadcast_to(reshape(out_grad, tuple(new_shape)), a.shape)
        return g

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        # https://github.com/bettersemut/dlsys_hw2/blob/8b16e4ecac6cf5d5efb2c4840f9107cdfe64e00b/python/needle/ops.py#L273
        lhs,rhs=node.inputs
        lhs_grad=out_grad@rhs.transpose()
        rhs_grad=lhs.transpose()@out_grad
        if len(lhs_grad.shape) != len(lhs.shape):
            lhs_grad = summation(lhs_grad, axes=tuple(i for i in range(len(lhs_grad.shape) - len(lhs.shape))))
        if len(rhs_grad.shape) != len(rhs.shape):
            rhs_grad = summation(rhs_grad, axes=tuple(i for i in range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad,rhs_grad

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad/node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a) 

    def gradient(self, out_grad, node):
        return out_grad*exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad*Tensor(node.inputs[0].cached_data > 0)

def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
# https://github.com/bettersemut/dlsys_hw2/blob/8b16e4ecac6cf5d5efb2c4840f9107cdfe64e00b/python/needle/ops.py#L351
        maxz=array_api.max(Z,axis=self.axes)
        maxz_keepDims=array_api.max(Z,axis=self.axes, keepdims=True)
        computeInput=Tensor(Z)
        minu=computeInput+Tensor(-maxz_keepDims)
        expMinus=exp(minu)
        summaExpMinus=summation(expMinus,axes=self.axes)
        logSummaExpMinus=log(summaExpMinus)
        result=logSummaExpMinus+maxz
        self.cache_input=computeInput
        self.cache_result=result
        return result.numpy()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        self.cache_result.backward(out_grad=out_grad)
        return self.cache_input.grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
