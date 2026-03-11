import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

# Import the CADT architecture defined previously
# Assuming the script is run from the project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cadt_transformer import CADT

def parse_args():
    """
    Parses command-line arguments for the MindIR export pipeline.
    """
    parser = argparse.ArgumentParser(description="CADT MindIR Export for Ascend 310B")
    parser.add_argument('--ckpt_path', type=str, default=None, 
                        help='Path to the trained CADT checkpoint (.ckpt) file.')
    parser.add_argument('--output_name', type=str, default='cadt_edge_model', 
                        help='Prefix for the exported MindIR file name.')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Inference batch size. Default is 1 for real-time edge streaming.')
    parser.add_argument('--seq_len', type=int, default=16, 
                        help='Temporal sequence length defined during alignment.')
    return parser.parse_args()

def export_to_mindir():
    """
    Core function to freeze the PyNative/Graph model and export it to the 
    MindIR intermediate representation. This format is hardware-agnostic and 
    highly optimized for Huawei Ascend NPU compilation.
    """
    args = parse_args()
    
    print("[INFO] Initiating CADT Architecture Export Protocol...")
    
    # 1. Initialize the Target Hardware Context
    # Enforce GRAPH_MODE for static shape compilation and optimization
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU") # Use CPU for export if NPU is unavailable locally
    
    # 2. Instantiate the Network Architecture
    # Ensure hyperparameters match the training configuration
    network = CADT(d_model=256, n_heads=8, num_layers=3, num_classes=2)
    network.set_train(False) # Strict evaluation mode (disables Dropout/BatchNorm updates)
    
    # 3. Load Pre-trained Weights (Optional during structural validation)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"[INFO] Loading trained parameters from {args.ckpt_path}")
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(network, param_dict)
    else:
        print("[WARNING] No valid checkpoint provided. Exporting structural graph with random initialization.")

    # 4. Define Dummy Tensors for Static Shape Tracing
    # The export function requires input examples to trace the computational graph.
    # Shapes must precisely match the output of MultiModalStreamAligner.
    dummy_vis = Tensor(np.ones((args.batch_size, args.seq_len, 512)), ms.float32)
    dummy_sonar = Tensor(np.ones((args.batch_size, args.seq_len, 128)), ms.float32)
    dummy_physio = Tensor(np.ones((args.batch_size, args.seq_len, 64)), ms.float32)
    
    print(f"[INFO] Tracing computational graph with dummy inputs:")
    print(f"       Visual: {dummy_vis.shape}, Sonar: {dummy_sonar.shape}, Physio: {dummy_physio.shape}")

    # 5. Execute MindIR Export
    output_path = os.path.join(os.path.dirname(__file__), args.output_name)
    
    try:
        export(
            network, 
            dummy_vis, 
            dummy_sonar, 
            dummy_physio, 
            file_name=output_path, 
            file_format='MINDIR'
        )
        print(f"[SUCCESS] Model successfully exported to {output_path}.mindir")
        print("[INFO] The exported MindIR artifact is ready for Ascend ACL edge deployment.")
    except Exception as e:
        print(f"[ERROR] Export failed during graph tracing: {e}")

if __name__ == '__main__':
    export_to_mindir()