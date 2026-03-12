import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

# Import internal modules
from models.cadt_transformer import CADT
from data.dataset import create_cadt_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="CADT Model Training Pipeline")
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--device', type=str, default='Ascend', choices=['CPU', 'GPU', 'Ascend'], 
                        help='Target computation device.')
    return parser.parse_args()

def train_cadt():
    """
    Main training execution loop for the Cross-modal Aquatic Distress Transformer.
    """
    args = parse_args()
    
    print("="*50)
    print("[INFO] Initiating CADT Training Sequence")
    print(f"[INFO] Target Device: {args.device}")
    print(f"[INFO] Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("="*50)

    # 1. Hardware Context Initialization
    # GRAPH_MODE is highly recommended for training efficiency
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)
    
    # 2. Initialize Data Pipeline
    print("[INFO] Constructing Data Pipeline...")
    train_dataset = create_cadt_dataloader(batch_size=args.batch_size, num_samples=2000, is_training=True)
    steps_per_epoch = train_dataset.get_dataset_size()
    
    # 3. Instantiate Network Architecture
    print("[INFO] Initializing CADT Architecture...")
    network = CADT(d_model=256, n_heads=8, num_layers=3, num_classes=2)
    
    # 4. Define Loss Function and Optimizer
    # Using SoftmaxCrossEntropyWithLogits for multi-class classification (Normal vs Distress)
    loss_fn = nn.CrossEntropyLoss()
    
    # AdamWeightDecay mitigates overfitting, crucial for transformer architectures
    optimizer = nn.AdamWeightDecay(network.trainable_params(), learning_rate=args.lr, weight_decay=1e-4)
    
    # 5. Compile the Training Model
    # Integrating Network, Loss, and Optimizer into a unified MindSpore Model wrapper
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy': nn.Accuracy()})
    
    # 6. Configure Callbacks (Logging and Checkpointing)
    output_dir = os.path.join(os.path.dirname(__file__), "results", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    config_ck = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch * 5, # Save every 5 epochs
        keep_checkpoint_max=3                      # Keep top 3 recent checkpoints
    )
    ckpoint_cb = ModelCheckpoint(prefix="cadt_model", directory=output_dir, config=config_ck)
    
    callbacks = [
        LossMonitor(per_print_times=steps_per_epoch), # Print loss at the end of each epoch
        TimeMonitor(data_size=steps_per_epoch),       # Profile step times
        ckpoint_cb                                    # Auto-save weights
    ]
    
    # 7. Execute Training Loop
    print("[INFO] Commencing Training Loop...")
    model.train(
        epoch=args.epochs, 
        train_dataset=train_dataset, 
        callbacks=callbacks, 
        dataset_sink_mode=False # Set to True for Ascend NPU to maximize throughput
    )
    
    print("\n[SUCCESS] Training Sequence Completed.")
    print(f"[INFO] Model artifacts saved to: {output_dir}")

if __name__ == '__main__':
    train_cadt()