import tensorrt as trt
import uff
from tensorrt import UffParser

G_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(G_LOGGER, '')

model_file = './mask_rcnn_nucleus_0080.uff'

output_nodes = ['mrcnn_detection', "mrcnn_mask/Sigmoid"]

trt_output_nodes = output_nodes

INPUT_NODE = "input_image"
INPUT_SIZE = [3, 1024, 1024]

with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    parser.register_input(INPUT_NODE, INPUT_SIZE)
    parser.register_output(output_nodes[0])
    parser.register_output(output_nodes[1])
    parser.parse(model_file, network)

    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 28 # 256MiB

    engine = builder.build_cuda_engine(network)
    for binding in engine:
        print(engine.get_binding_shape(binding))
    with open("nucleus.engine", "wb") as f:
       f.write(engine.serialize())