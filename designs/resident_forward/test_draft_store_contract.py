import os
from functools import wraps
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if TYPE_CHECKING:
    import draft_store
    import rhead


def draft_model(test):
    @wraps(test)
    def run():
        subprocess.run([sys.executable, __file__, test.__name__], check=True,
                       env=os.environ | {"RF_MODEL": "gemma3-270m-rail"})
    return run


@draft_model
def test_padded_norm_parameters_preserve_logical_gemma3_rmsnorm():
    model = draft_store.MODEL
    assert model.norm_width == draft_store.D0
    assert model.norm_eps == np.float32(1e-6)
    assert np.float32(model.physical_norm_eps) == np.float32(model.norm_eps * model.norm_width / model.D)

    logical = np.array([[0.125, -0.25, 0.5, -1.0]], np.float32)
    logical = np.tile(logical, (1, draft_store.D0 // logical.shape[1]))
    padded = np.pad(logical, ((0, 0), (0, model.D - model.norm_width)))
    logical_den = np.sum(logical * logical, axis=1) / model.norm_width + model.norm_eps
    padded_den = np.sum(padded * padded, axis=1) / model.D + np.float32(model.physical_norm_eps)
    np.testing.assert_allclose(padded_den * model.D / model.norm_width, logical_den, rtol=1e-6, atol=0)


@draft_model
def test_head_stream_has_rhead_gains_layout_and_padding():
    ndb = 64
    q = (np.arange(ndb * 32 * 8 * rhead.NSUB * rhead.SUBK, dtype=np.int16) % 16 - 8).astype(np.int8)
    q = q.reshape(ndb * 32, 8 * rhead.NSUB * rhead.SUBK)
    scales = np.arange(q.shape[0] * (q.shape[1] // 32), dtype=np.float32).reshape(q.shape[0], -1)
    final_norm = np.linspace(-0.5, 0.5, q.shape[1], dtype=np.float32)

    stream = draft_store._head_stream_layout(rhead.pack_block(q, scales), final_norm)

    ndb = q.shape[0] // rhead.NROW // 32
    nwp = -(-(2 + ndb * rhead.NSUB) // (2 * rhead.GW)) * (2 * rhead.GW)
    expected = np.zeros((rhead.NCOL, nwp, rhead.NROW, rhead.ELEM), np.uint8)
    for col in range(rhead.NCOL):
        gain = draft_store.rmlp_ref.gain_elements(final_norm, np.float32(1.0), col)
        expected[col, :2] = gain.reshape(2, rhead.NROW, rhead.ELEM)
    rows_per_core = q.shape[0] // rhead.NROW
    for row in range(rhead.NROW):
        part = slice(row * rows_per_core, (row + 1) * rows_per_core)
        packed = rhead.pack_block(q[part], scales[part])
        for col in range(rhead.NCOL):
            expected[col, 2:2 + ndb * rhead.NSUB, row] = packed[:, col].reshape(-1, rhead.ELEM)
    expected = expected.reshape(rhead.NCOL, nwp // rhead.GW, rhead.GW, rhead.NROW, rhead.ELEM).transpose(0, 1, 3, 2, 4)

    np.testing.assert_array_equal(stream, expected)
    unpacked = stream.transpose(0, 1, 3, 2, 4).reshape(rhead.NCOL, nwp, rhead.NROW, rhead.ELEM)
    assert np.all(unpacked[:, 2 + ndb * rhead.NSUB:] == 0)


@draft_model
def test_head_producers_reject_an_aligned_wrong_vocabulary():
    q = np.zeros((128, draft_store.MODEL.D), np.int8)
    scales = np.zeros((128, draft_store.MODEL.D // 32), np.float32)
    gain = np.ones(draft_store.MODEL.D, np.float32)
    for call in (lambda: draft_store.head_stream(q, scales, gain),
                 lambda: draft_store.head_stream_from_blocks(rhead.pack_block(q, scales), gain)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('fixed-vocabulary consumer accepted a shorter head')


@draft_model
def test_head_loader_checks_layout_and_consumer_shape():
    import json
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
        root = Path(directory)
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'head_layout': 'raw_blocks'}))
        try:
            draft_store.load_head_stream(root)
        except ValueError:
            pass
        else:
            raise AssertionError('raw layout accepted')
        manifest.write_text(json.dumps({'head_layout': rhead.HEAD_STREAM_LAYOUT}))
        shape = (rhead.NCOL, rhead.NWP // rhead.GW, rhead.NROW, rhead.GW, rhead.ELEM)
        mapped = np.lib.format.open_memmap(root / 'w_head.npy', mode='w+', dtype=np.uint8, shape=shape)
        del mapped
        loaded = draft_store.load_head_stream(root)
        assert loaded.shape == shape and loaded.dtype == np.uint8
        del loaded
        np.save(root / 'w_head.npy', np.zeros((1, rhead.NCOL, rhead.NSUB, rhead.ELEM), np.uint8))
        try:
            draft_store.load_head_stream(root)
        except ValueError:
            pass
        else:
            raise AssertionError('raw block shape accepted as a runtime stream')


if __name__ == '__main__':
    import draft_store
    import rhead
    globals()[sys.argv[1]].__wrapped__()
