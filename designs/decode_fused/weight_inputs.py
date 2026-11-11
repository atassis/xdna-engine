# SPDX-License-Identifier: Apache-2.0
"""Record physical weight requirements at the graph's existing transformation sites."""
from math import prod


class WeightInputs:
    def __init__(self, enabled):
        self.enabled = enabled
        self.buffers = {}

    def rows(self, output, tensor, shape, *, first=0, count=None, scale=None):
        if not self.enabled:
            return
        op = dict(op='rows', tensor=tensor, shape=list(shape), first=first,
                  count=shape[0] if count is None else count)
        if scale is not None:
            op['scale'] = float(scale)
        self.buffers[output] = [op]

    def concat(self, output, inputs):
        if self.enabled and all(n in self.buffers for n in inputs):
            self.buffers[output] = [op.copy() for n in inputs for op in self.buffers[n]]

    def pad(self, output, byte_count):
        if self.enabled and output in self.buffers and byte_count:
            self.buffers[output].append(dict(op='zero', bytes=byte_count))

    def finish(self, lengths, embedding, shape, config):
        # An unrecorded transform is unsupported, never an invitation to infer from a buffer name.
        if not self.enabled or not lengths or not set(lengths) <= self.buffers.keys():
            return None
        buffers = {n: self.buffers[n] for n in lengths}
        for name, ops in buffers.items():
            size = sum(op['bytes'] if op['op'] == 'zero' else
                       op['count'] * prod(op['shape'][1:]) * 2 for op in ops)
            if size != lengths[name]:
                raise ValueError(f'{name}: weight-input byte extent {size} != {lengths[name]}')
        return dict(version=1, config=config, buffers=buffers,
                    embedding=dict(tensor=embedding, shape=list(shape)))
