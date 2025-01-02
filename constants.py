# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines project-wide constants."""

NUM_CHUNKS = 488281
CHUNK_H, CHUNK_W = 16, 32 # 512
CHUNK_SIZE_BYTES = CHUNK_H * CHUNK_W
CHUNK_SHAPE_2D = (CHUNK_H, CHUNK_W)
ALPHABET_SIZE = 256

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32
