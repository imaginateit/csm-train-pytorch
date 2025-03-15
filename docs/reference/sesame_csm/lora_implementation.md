# Technical Implementation of LoRA in CSM

This document provides technical details about the Low-Rank Adaptation (LoRA) implementation in the CSM framework.

## Architecture Overview

The LoRA implementation in CSM follows the architecture described in the [LoRA paper](https://arxiv.org/abs/2106.09685), with specific adaptations for speech models and MLX compatibility.

### Key Components

1. **LoRALinear**: Core implementation of the low-rank adaptation for linear layers
2. **LoRATransformerLayer**: Adapter for transformer layers that handles attention and MLP components
3. **LoRATransformer**: Top-level adapter for transformer models, managing multiple layers
4. **CSMLoRATrainer**: Training logic for fine-tuning with LoRA

### Design Principles

- **Parameter Efficiency**: Only LoRA parameters are trained, with the base model frozen
- **Flexibility**: Configurable rank, target modules, and layers
- **MLX Compatibility**: Designed to work with MLX's lazy evaluation and array operations
- **Minimal Memory Footprint**: Optimized for Apple Silicon's unified memory architecture

## LoRA Layer Implementation

The core LoRA implementation adds low-rank adapters to existing weights:

```
h = W₀x + ∆Wx
  = W₀x + BAx
```

Where:
- W₀ is the pre-trained weight matrix
- B is a learned matrix of size d × r
- A is a learned matrix of size r × k
- r is the LoRA rank (r ≪ min(d, k))

### Code Structure

```python
class LoRALinear:
    def __init__(self, base_weight, base_bias=None, r=8, alpha=16.0):
        self.base_weight = base_weight  # Original frozen weight
        self.base_bias = base_bias      # Original frozen bias
        
        # Low-rank matrices
        self.lora_A = mx.random.normal((r, base_weight.shape[1]))
        self.lora_B = mx.zeros((base_weight.shape[0], r))
        
        # Scaling factor
        self.scaling = alpha / r
    
    def __call__(self, x):
        # Original (frozen) path
        base_output = mx.matmul(x, self.base_weight.T)
        
        # LoRA path
        lora_output = mx.matmul(x, self.lora_A.T)
        lora_output = mx.matmul(lora_output, self.lora_B.T)
        lora_output = lora_output * self.scaling
        
        # Combine outputs
        return base_output + lora_output
```

## Integration with MLX

The LoRA implementation is designed to work with MLX's unique characteristics:

1. **Lazy Evaluation**: MLX uses lazy evaluation to build computation graphs before execution
2. **Array Operations**: MLX provides numpy-like array operations optimized for Apple Silicon
3. **In-place Updates**: The implementation avoids in-place updates that could interfere with MLX's graph construction

### MLX-Specific Considerations

```python
# Parameter collection for optimization
def get_lora_params(self):
    params = {}
    for layer_idx, layer in self.lora_layers:
        if isinstance(layer, LoRATransformerLayer):
            params.update(layer.parameters())
    return params

# Gradient computation
def train_step(self, batch):
    # Get only LoRA parameters
    params = self.model.get_lora_params()
    
    # Define loss function
    def loss_fn(model_params):
        self.model.update(model_params)
        loss, _ = compute_loss_mlx(
            self.model,
            batch["input_tokens"],
            batch["input_masks"],
            batch["target_audio_tokens"]
        )
        return loss
    
    # Get loss and gradients
    loss, grads = nn.value_and_grad(loss_fn)(params)
    
    # Update model with optimizer
    self.optimizer.update(self.model, grads)
    
    # Ensure computation completes (MLX is lazy)
    mx.eval(loss)
    return loss
```

## Target Module Selection

The implementation allows for selective application of LoRA to different modules:

### Available Target Modules

| Module | Description | Recommended |
|--------|-------------|-------------|
| `q_proj` | Query projection in attention | Yes |
| `k_proj` | Key projection in attention | Optional |
| `v_proj` | Value projection in attention | Yes |
| `o_proj` | Output projection in attention | Optional |
| `gate_proj` | Gate projection in MLP | Optional |
| `up_proj` | Up projection in MLP | Optional |
| `down_proj` | Down projection in MLP | Optional |

### Efficiency Analysis

The table below shows parameter counts and efficiency measurements for different configurations (16-layer model):

| Configuration | Target Modules | LoRA Rank | Parameter % | Memory Usage | Training Speed |
|---------------|---------------|-----------|-------------|--------------|----------------|
| Minimal | q_proj, v_proj | 4 | 0.12% | Low | Fastest |
| Balanced | q_proj, v_proj | 16 | 0.47% | Medium | Fast |
| Attention-focused | q_proj, k_proj, v_proj, o_proj | 8 | 0.47% | Medium | Medium |
| Full | All modules | 8 | 0.82% | High | Slowest |

## Weight Merging

The implementation supports merging LoRA weights with base weights for deployment:

```python
def merge_lora_weights(self):
    # Create a copy of the model
    merged_model = deepcopy(self.base_model)
    
    # Merge weights for each LoRA layer
    for layer_idx, layer in self.lora_layers:
        if isinstance(layer, LoRATransformerLayer):
            for module_name, lora_adapter in layer.lora_adapters.items():
                merged_weight = lora_adapter.merge_with_base()
                
                # Update weights in the merged model
                if module_name == "q_proj":
                    merged_model.layers[layer_idx].q_proj_weight = merged_weight
                elif module_name == "v_proj":
                    merged_model.layers[layer_idx].v_proj_weight = merged_weight
                # ... other modules
    
    return merged_model
```

## Performance Optimizations

The implementation includes several optimizations for better performance:

1. **Layer-selective Application**: Apply LoRA only to specific layers (e.g., upper layers) for better efficiency
2. **Memory Optimizations**: Careful management of tensor operations to minimize memory footprint
3. **Gradient Clipping**: Optional gradient clipping to improve training stability

## Future Improvements

Planned enhancements for the LoRA implementation:

1. **QLoRA Support**: Add quantization-aware LoRA training for even better memory efficiency
2. **Conditional LoRA**: Support for conditional computation based on input properties
3. **Multi-device Training**: Distributed training across multiple Apple Silicon devices
4. **Hyperparameter Autotuning**: Automated LoRA rank and learning rate selection