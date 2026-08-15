#!/usr/bin/env python3
"""Build TensorRT engine from ONNX (replaces trtexec, which is not shipped with the pip tensorrt wheel).

Usage: python3 build_vocos_trt_engine.py <onnx_path> <engine_path>
"""
import sys

import tensorrt as trt

ONNX_PATH = sys.argv[1]
ENGINE_PATH = sys.argv[2]

MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 1, 8
MIN_LEN, OPT_LEN, MAX_LEN = 1, 1000, 3000

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as f:
    assert parser.parse(f.read()), "ONNX parse failed"

profile = builder.create_optimization_profile()
profile.set_shape("mel", (MIN_BATCH, 100, MIN_LEN), (OPT_BATCH, 100, OPT_LEN), (MAX_BATCH, 100, MAX_LEN))
config = builder.create_builder_config()
config.add_optimization_profile(profile)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

engine = builder.build_serialized_network(network, config)
assert engine is not None, "Engine build failed"
with open(ENGINE_PATH, "wb") as f:
    f.write(engine)
print(f"Engine saved to {ENGINE_PATH}")
