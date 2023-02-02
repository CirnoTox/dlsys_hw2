"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # BEGIN YOUR SOLUTION
        mom = self.momentum
        for w in self.params:
            if w.grad==None:
                continue
            if self.u.get(w) == None:
                self.u[w] = (1-mom)*(w.grad.data + self.weight_decay*w.data)
            else:
                self.u[w] = mom*self.u[w] + \
                    (1-mom)*(w.grad.data + self.weight_decay*w.data)
            w.data = w.data + (-self.lr) * self.u[w]
        # END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # BEGIN YOUR SOLUTION
        beta1 = self.beta1
        beta2 = self.beta2
        weight_decay = self.weight_decay
        self.t += 1
        for i, w in enumerate(self.params):
            if w.grad==None:
                continue
            loss_data = w.grad.data + weight_decay*w
            if self.m.get(i) is None:
                self.m[i] = ndl.init.zeros(*w.shape)
            if self.v.get(i) is None:
                self.v[i] = ndl.init.zeros(*w.shape)
            self.m[i] = beta1*self.m[i]+(1-beta1)*loss_data
            self.v[i] = beta2*self.v[i]+(1-beta2)*(loss_data**2)
            mt = self.m[i]/(1-beta1**self.t)
            vt = self.v[i]/(1-beta2**self.t)
            w.data = w.data+(-self.lr*mt/(vt**0.5+self.eps))
        # END YOUR SOLUTION
