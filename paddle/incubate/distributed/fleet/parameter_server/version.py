

# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

from paddle.incubate.distributed.fleet.base import Mode

BUILD_MODE=Mode.TRANSPILER

def is_transpiler():
    return Mode.TRANSPILER == BUILD_MODE

