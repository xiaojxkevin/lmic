import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import jax
jax.config.update("jax_traceback_filtering", "off")
jax.default_backend()

from collections.abc import Generator
import functools
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import constants
import data_loaders
import utils
from compressors.language_model import decompress, compress


_DATASET = flags.DEFINE_enum(
    'dataset',
    'enwik9',
    data_loaders.GET_DATA_GENERATOR_FN_DICT.keys(),
    'Dataset to use.',
)


def encode_and_decode(
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    mask_fn: Callable[[bytes], tuple[bytes, int]] | None = None,
) -> tuple[float, float]:

  data_generator = get_data_generator_fn()

  for it, data in enumerate(data_generator): # type(data) is 'bytes'
    if it == 1024:
      data, missed_bits = mask_fn(data)
      print("<><><>The original data:\n", data)

      compressed_data = compress(data, use_slow_lossless_compression=True)
      print("<><><>Compressed data:\n", compressed_data)

      decoded_data = decompress(data=compressed_data)
      print("<><><>Decoded data:\n", decoded_data)

      break
    continue
  

def main(_) -> None:

  get_data_generator_fn = functools.partial(
    data_loaders.GET_DATA_GENERATOR_FN_DICT[_DATASET.value],
    num_chunks=constants.NUM_CHUNKS,
  )

  mask_fn = utils.zero_most_significant_bit_if_not_ascii_decodable

  encode_and_decode(
    get_data_generator_fn=get_data_generator_fn,
    mask_fn=mask_fn,
  )
    

if __name__ == '__main__':
  app.run(main)