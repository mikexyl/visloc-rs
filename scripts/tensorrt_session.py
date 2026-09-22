"""NumPy binding to the same native TensorRT bridge used by visloc's Rust runtime.

No second inference implementation, PyTorch, or CPU fallback. A session and all
its CUDA resources belong to the thread that opens it. ctypes releases the GIL
while the native bridge runs, so the replay can keep feeding the VIO process.
"""
import ctypes as ct
from pathlib import Path
import threading

import numpy as np


class Info(ct.Structure):
    _fields_ = [('name', ct.c_char_p), ('dtype', ct.c_int32), ('input', ct.c_int32),
                ('rank', ct.c_int32), ('dims', ct.c_int64 * 8)]


class Input(ct.Structure):
    _fields_ = [('name', ct.c_char_p), ('data', ct.POINTER(ct.c_uint8)),
                ('bytes', ct.c_size_t), ('dtype', ct.c_int32), ('rank', ct.c_int32),
                ('dims', ct.c_int64 * 8)]


DTYPES = {0: np.dtype('float32'), 1: np.dtype('float16'), 2: np.dtype('int8'),
          3: np.dtype('int32'), 4: np.dtype('bool'), 5: np.dtype('uint8'), 8: np.dtype('int64')}


def metadata(info):
    if not info.name or not 0 <= info.rank <= 8 or info.dtype not in DTYPES:
        raise RuntimeError('Unsupported native TensorRT tensor metadata')
    return dict(name=info.name.decode(), dtype=info.dtype, input=bool(info.input),
                shape=tuple(info.dims[:info.rank]))


class Session:
    def __init__(self, engine, library, device=0):
        self.owner = threading.get_ident()
        self.handle = None
        self.lib = ct.CDLL(str(Path(library).resolve()))
        signatures = {
            'vt_error': (ct.c_char_p, []),
            'vt_open': (ct.c_void_p, [ct.POINTER(ct.c_uint8), ct.c_size_t, ct.c_int32]),
            'vt_close': (None, [ct.c_void_p]),
            'vt_count': (ct.c_int32, [ct.c_void_p]),
            'vt_info': (ct.c_int32, [ct.c_void_p, ct.c_int32, ct.POINTER(Info)]),
            'vt_run': (ct.c_int32, [ct.c_void_p, ct.POINTER(Input), ct.c_size_t, ct.c_int32]),
            'vt_output': (ct.c_int32, [ct.c_void_p, ct.c_int32, ct.POINTER(Info),
                                     ct.POINTER(ct.POINTER(ct.c_uint8)), ct.POINTER(ct.c_size_t)]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, argtypes
        plan = Path(engine).read_bytes()
        if not plan or device < 0:
            raise ValueError('Nonempty engine and nonnegative device required')
        data = (ct.c_uint8 * len(plan)).from_buffer_copy(plan)
        self.handle = self.lib.vt_open(data, len(plan), device)
        if not self.handle:
            raise RuntimeError(self.lib.vt_error().decode())
        try:
            self.tensors = []
            for i in range(self.lib.vt_count(self.handle)):
                info = Info()
                self.check(self.lib.vt_info(self.handle, i, ct.byref(info)))
                self.tensors.append(metadata(info))
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.vt_error().decode())

    def check_owner(self):
        if threading.get_ident() != self.owner or not self.handle:
            raise RuntimeError('TensorRT session must be used by its live owner thread')

    def run(self, inputs):
        self.check_owner()
        expected = {t['name']: t for t in self.tensors if t['input']}
        if set(inputs) != set(expected):
            raise ValueError(f'Engine inputs must be exactly {list(expected)}')
        arrays, names, native = [], [], []
        for name, tensor in expected.items():
            array = np.ascontiguousarray(inputs[name])
            if array.dtype != DTYPES[tensor['dtype']] or array.ndim != len(tensor['shape']):
                raise ValueError(f'Wrong dtype or rank for {name}')
            if any(want >= 0 and got != want for got, want in zip(array.shape, tensor['shape'])):
                raise ValueError(f'Wrong shape for {name}: {array.shape}, expected {tensor["shape"]}')
            arrays.append(array)
            names.append(name.encode())
            native.append(Input(names[-1], array.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                                array.nbytes, tensor['dtype'], array.ndim,
                                (ct.c_int64 * 8)(*array.shape)))
        self.check(self.lib.vt_run(self.handle, (Input * len(native))(*native), len(native), 0))
        outputs = {}
        for i, tensor in enumerate(self.tensors):
            if tensor['input']:
                continue
            info, data, size = Info(), ct.POINTER(ct.c_uint8)(), ct.c_size_t()
            self.check(self.lib.vt_output(self.handle, i, ct.byref(info), ct.byref(data), ct.byref(size)))
            meta = metadata(info)
            if any(d < 0 for d in meta['shape']):
                raise RuntimeError('Unresolved TensorRT output dimensions')
            dtype = DTYPES[meta['dtype']]
            if int(np.prod(meta['shape'], dtype=np.int64)) * dtype.itemsize != size.value:
                raise RuntimeError('TensorRT output byte count differs from its shape')
            # Own the bytes before another call can reuse the native buffers.
            outputs[meta['name']] = np.frombuffer(ct.string_at(data, size.value), dtype=dtype).copy().reshape(meta['shape'])
        return outputs

    def close(self):
        if self.handle:
            self.check_owner()
            self.lib.vt_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
