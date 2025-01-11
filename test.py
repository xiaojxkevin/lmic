import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
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


def encode_and_decode(
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    mask_fn: Callable[[bytes], tuple[bytes, int]] | None = None,
) -> tuple[float, float]:

  data_generator = get_data_generator_fn()

  for it, data in enumerate(data_generator): # type(data) is 'bytes'
    if it == 1024:
      data, missed_bits = mask_fn(data)
      print(f"There are {missed_bits} missed bits")
      print("<><><>The original data:\n", data)

      compressed_data, num_padded_bits = compress(data, return_num_padded_bits=True, use_slow_lossless_compression=True)
      print(f"There are {num_padded_bits} padded bits")
      print("<><><>Compressed data:\n", compressed_data)

      # compressed_data = bytes(b'\x01\x95\xda\xd9JS`N&\x80\xc3^~7\xf7t\x82\xde\x98\xce\x97\xbc\xab\xfaK9\xb4o(n\x19\xba\x1a\xad\xb1dN\x13\x87\xf6\x9f#\x06\x8f\xcbI\x83\x9bi\x882>K\x06ekt\x19\x86\x0c\xd3Q\x9a\xe4\xd4\x82S\x02\x9dp\xd4z\xa9\xf7\x06\xfc!"\x99\xb5J\xc1C\x15?\x8e\x8bWo\x1d\xcc#\xe8r\xbc"\'Q\xed\x1b\xfb\x1a\x89\xa6i\xf6%\x83\x94\xe7\xa5E\x15\xb7\x92\xe4\x18\xa5M\xe0\xaa\xecV\x9f\xe1\xddSr<J\xa9$k\x13U^\x0c\xc4s\xe2rP\x93\x03L\xdda\xa3\x86\xde?\x19\x0f_\xee:\x95\xed\'\x99\xfcbu\xcd\x87\x85Y\xad*\xf7\xcd\x01&\xc6\xff\xbd\xfb\xdf\xf9E\xdd&Y\xf6\x95\x07\x11\xf8*Udx>\xf4%|L\x15\x108!@\x88\xb1\xf2drt\x9c\x84\xcc\x99\xdbv')
      # num_padded_bits = 7

      decoded_data = decompress(data=compressed_data, num_padded_bits=num_padded_bits)
      print("<><><>Decoded data:\n", decoded_data)

      break
    continue
  

def main(_) -> None:

  encode_and_decode(
    get_data_generator_fn=data_loaders.get_enwik9_iterator,
    mask_fn=utils.zero_most_significant_bit_if_not_ascii_decodable,
  )
    

if __name__ == '__main__':
  app.run(main)