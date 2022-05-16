"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
from abc import ABC, abstractmethod
from typing import Sequence


class Transform(ABC):
    @abstractmethod
    def forward(self, x):
        """
        :param x: An object to be transform
        :return: Transformed x and additional context, will be used as a input for 'backward'
        """

    # noinspection PyMethodMayBeStatic
    def backward(self, f_res):
        """
        Inverse or a similar operation designed to act on outputs of the stage
        Defaults to no-op
        :param f_res: A transformed value returned from the `forward` function
                    parts of it may have passed through other transforms
        :return: A an approximation of the original transformed object
        """
        return f_res

    def roundtrip(self, x):
        return self.backward(self.forward(x))


class FunctionalTransform(Transform):
    def __init__(self, forward, backward):
        self._forward = forward
        self._backward = backward

    def forward(self, x):
        return self._forward(x)

    def backward(self, f_res):
        return self._backward(f_res)


def pipeline(fns):
    def _pipeline_fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _pipeline_fn


class OnionTransform(FunctionalTransform):
    def __init__(self,
                 transforms: Sequence[Transform]):  # TODO in python 3.9 use collections.abc.Sequence[Transformer]
        """
        :param transforms: Transformer to be run in sequence during transform
                             while their inv function is run in reverse order during inv
        """

        def _forward(x):
            context = []
            for t in transforms:
                x, new_context = t.forward(x)
                context.append(new_context)
            return x, context

        def _backward(f_res):
            x, context = f_res

            for t, c in reversed(list(zip(transforms, context))):
                x = t.backward((x, c))
            return x

        super().__init__(_forward, _backward)


class NoopTransform(Transform):
    def forward(self, x):
        return x


# singleton
NoopTransform.s = NoopTransform()
