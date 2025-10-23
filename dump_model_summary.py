"""
Script to load pretrained BLIP model and dump a comprehensive summary of operations.
Includes model architecture, parameter counts, shapes, and CPU profiling.
Outputs to CSV files for easy analysis.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import sys
import os
import csv
from collections import defaultdict
from unittest.mock import MagicMock, Mock

# Mock transformers before importing BLIP to avoid tokenizer compilation issues
print("Setting up mocks for transformers dependencies...")
mock_transformers = MagicMock()
mock_transformers.BertTokenizer = MagicMock()
mock_transformers.BertModel = MagicMock()
mock_transformers.BertConfig = MagicMock()
mock_transformers.BertLMHeadModel = MagicMock()
mock_transformers.activations = MagicMock()
mock_transformers.activations.ACT2FN = {}
mock_transformers.modeling_utils = MagicMock()
mock_transformers.file_utils = MagicMock()
mock_transformers.modeling_outputs = MagicMock()

# Add all the necessary attributes to file_utils
mock_transformers.file_utils.ModelOutput = object
mock_transformers.file_utils.add_start_docstrings = lambda *args, **kwargs: lambda x: x
mock_transformers.file_utils.add_start_docstrings_to_model_forward = lambda *args, **kwargs: lambda x: x
mock_transformers.file_utils.replace_return_docstrings = lambda *args, **kwargs: lambda x: x

# Add modeling_outputs classes
mock_transformers.modeling_outputs.BaseModelOutput = object
mock_transformers.modeling_outputs.BaseModelOutputWithPooling = object
mock_transformers.modeling_outputs.CausalLMOutputWithCrossAttentions = object

sys.modules['transformers'] = mock_transformers
sys.modules['transformers.activations'] = mock_transformers.activations
sys.modules['transformers.modeling_utils'] = mock_transformers.modeling_utils
sys.modules['transformers.file_utils'] = mock_transformers.file_utils
sys.modules['transformers.modeling_outputs'] = mock_transformers.modeling_outputs

# Add the current directory to path to import BLIP modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import BLIP modules - import directly from vit to avoid med.py
from models.vit import VisionTransformer, interpolate_pos_embed


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    """
    Create a Vision Transformer model.
    Simplified version from models/blip.py to avoid importing transformers dependencies.
    """
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def capture_layer_shapes(model, dummy_input):
    """
    Capture input/output shapes for each layer using forward hooks.
    
    Args:
        model: PyTorch model
        dummy_input: Sample input tensor
        
    Returns:
        Dictionary mapping layer names to (input_shape, output_shape)
    """
    shapes = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            try:
                input_shape = str(list(input[0].shape)) if isinstance(input, tuple) and len(input) > 0 else "N/A"
                if isinstance(output, torch.Tensor):
                    output_shape = str(list(output.shape))
                elif isinstance(output, tuple) and len(output) > 0:
                    output_shape = str(list(output[0].shape))
                else:
                    output_shape = "N/A"
                shapes[name] = (input_shape, output_shape)
            except:
                shapes[name] = ("N/A", "N/A")
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if name:  # Skip root module
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run forward pass
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"Note: Forward pass partially completed: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return shapes


def dump_layers_csv(model, shapes, output_file='model_layers.csv'):
    """
    Dump all layers to CSV with their properties.
    
    Args:
        model: PyTorch model
        shapes: Dictionary of layer shapes from capture_layer_shapes
        output_file: Path to save the CSV
    """
    print("\n" + "=" * 80)
    print("GENERATING LAYERS CSV")
    print("=" * 80)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'layer_type', 'parameters', 'input_shape', 'output_shape', 'trainable'])
        
        total_layers = 0
        for name, module in model.named_modules():
            if name == '':  # Skip root module
                continue
            
            # Count parameters in this specific module (not including submodules)
            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            
            module_type = type(module).__name__
            input_shape, output_shape = shapes.get(name, ("N/A", "N/A"))
            
            writer.writerow([name, module_type, module_params, input_shape, output_shape, trainable])
            total_layers += 1
        
        print(f"✓ Wrote {total_layers} layers to {output_file}")
    
    return output_file


def dump_operations_csv(prof, output_file='model_operations.csv'):
    """
    Dump profiling operations to CSV.
    
    Args:
        prof: PyTorch profiler object
        output_file: Path to save the CSV
    """
    print("\n" + "=" * 80)
    print("GENERATING OPERATIONS CSV")
    print("=" * 80)
    
    events = prof.key_averages()
    total_time = sum(evt.cpu_time_total for evt in events)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['operation', 'count', 'cpu_time_total_us', 'cpu_time_avg_us', 
                        'self_cpu_time_total_us', 'cpu_memory_bytes', 'percentage'])
        
        # Sort by total CPU time
        sorted_events = sorted(events, key=lambda x: x.cpu_time_total, reverse=True)
        
        for evt in sorted_events:
            if evt.cpu_time_total > 0:  # Only include operations that took time
                percentage = (evt.cpu_time_total / total_time * 100) if total_time > 0 else 0
                writer.writerow([
                    evt.key,
                    evt.count,
                    evt.cpu_time_total,
                    evt.cpu_time_total / evt.count if evt.count > 0 else 0,
                    evt.self_cpu_time_total,
                    evt.cpu_memory_usage,
                    f"{percentage:.2f}"
                ])
        
        print(f"✓ Wrote operations to {output_file}")
    
    return output_file


def dump_summary_csv(model, prof, forward_time_ms, output_file='model_summary.csv'):
    """
    Dump overall model summary to CSV.
    
    Args:
        model: PyTorch model
        prof: PyTorch profiler object
        forward_time_ms: Forward pass time in milliseconds
        output_file: Path to save the CSV
    """
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY CSV")
    print("=" * 80)
    
    total_params, trainable_params = count_parameters(model)
    total_layers = sum(1 for _ in model.named_modules()) - 1  # Exclude root
    
    events = prof.key_averages()
    total_operations = sum(evt.count for evt in events)
    peak_memory_bytes = max((evt.cpu_memory_usage for evt in events), default=0)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_parameters', total_params])
        writer.writerow(['trainable_parameters', trainable_params])
        writer.writerow(['non_trainable_parameters', total_params - trainable_params])
        writer.writerow(['total_layers', total_layers])
        writer.writerow(['total_operations', total_operations])
        writer.writerow(['forward_pass_time_ms', f"{forward_time_ms:.2f}"])
        writer.writerow(['peak_memory_mb', f"{peak_memory_mb:.2f}"])
        
        print(f"✓ Wrote summary to {output_file}")
    
    # Also print to console
    print("\nMODEL SUMMARY:")
    print("-" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Layers: {total_layers}")
    print(f"Total Operations: {total_operations}")
    print(f"Forward Pass Time: {forward_time_ms:.2f} ms")
    print(f"Peak Memory: {peak_memory_mb:.2f} MB")
    
    return output_file


def profile_model_cpu(model, dummy_input):
    """
    Profile the model's CPU performance during a forward pass.
    
    Args:
        model: PyTorch model
        dummy_input: Sample input tensor
        
    Returns:
        Tuple of (profiler, forward_time_ms)
    """
    print("\n" + "=" * 80)
    print("CPU PROFILING")
    print("=" * 80)
    
    model.eval()
    
    print(f"\nProfiling forward pass with input shape: {list(dummy_input.shape)}")
    print("This may take a moment...\n")
    
    # Warm up
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception as e:
            print(f"Warmup note: {e}")
    
    # Measure forward pass time
    import time
    start_time = time.time()
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception as e:
            print(f"Forward pass note: {e}")
    forward_time_ms = (time.time() - start_time) * 1000
    
    # Profile with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                try:
                    _ = model(dummy_input)
                except Exception as e:
                    print(f"Profiling note: {e}")
    
    # Print top operations
    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    return prof, forward_time_ms


def main():
    """Main function to load model and generate summaries."""
    
    # Configuration
    checkpoint_path = r"C:\Users\Administrator\Downloads\model_base_capfilt_large.pth"
    image_size = 384
    vit_type = 'base'
    
    # Extract model name from checkpoint path
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    print("=" * 80)
    print("BLIP MODEL ANALYSIS SCRIPT")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Model Name: {model_name}")
    print(f"Image Size: {image_size}")
    print(f"ViT Type: {vit_type}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print("\n[1/5] Creating model structure...")
    try:
        # Create vision transformer (this is the main component we can analyze)
        visual_encoder, vision_width = create_vit(vit_type, image_size, use_grad_checkpointing=False, ckpt_layer=0)
        print(f"✓ Created ViT model with vision_width={vision_width}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2/5] Loading checkpoint weights...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter state dict for visual encoder only
        visual_encoder_state = {k.replace('visual_encoder.', ''): v 
                               for k, v in state_dict.items() 
                               if k.startswith('visual_encoder.')}
        
        # Interpolate position embeddings if size mismatch
        if 'pos_embed' in visual_encoder_state:
            visual_encoder_state['pos_embed'] = interpolate_pos_embed(
                visual_encoder_state['pos_embed'], visual_encoder)
        
        # Load weights
        msg = visual_encoder.load_state_dict(visual_encoder_state, strict=False)
        print(f"✓ Loaded checkpoint ({len(visual_encoder_state)} keys)")
        if msg.missing_keys:
            print(f"  Missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/5] Capturing layer shapes with forward pass...")
    try:
        dummy_input = torch.randn(1, 3, image_size, image_size)
        shapes = capture_layer_shapes(visual_encoder, dummy_input)
        print(f"✓ Captured shapes for {len(shapes)} layers")
    except Exception as e:
        print(f"✗ Error capturing shapes: {e}")
        shapes = {}
        import traceback
        traceback.print_exc()
    
    print("\n[4/5] Profiling CPU performance...")
    try:
        prof, forward_time_ms = profile_model_cpu(visual_encoder, dummy_input)
        print(f"✓ Profiling complete")
    except Exception as e:
        print(f"✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[5/5] Generating CSV files...")
    try:
        layers_file = dump_layers_csv(visual_encoder, shapes, output_file=f'{model_name}_layers.csv')
        ops_file = dump_operations_csv(prof, output_file=f'{model_name}_operations.csv')
        summary_file = dump_summary_csv(visual_encoder, prof, forward_time_ms, output_file=f'{model_name}_summary.csv')
    except Exception as e:
        print(f"✗ Error generating CSV files: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  1. {layers_file} - All layers with types, parameters, and shapes")
    print(f"  2. {ops_file} - All operations with CPU profiling data")
    print(f"  3. {summary_file} - Overall model summary statistics")
    print("\n")


if __name__ == "__main__":
    main()
