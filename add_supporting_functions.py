#!/usr/bin/env python3

import json

# Load notebook
with open('/home/engine/project/Unsloth_Puzzles.ipynb', 'r') as f:
    nb = json.load(f)

# Add supporting functions cell (cell 36)
supporting_functions = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Supporting functions for MemoryEfficientLinear',
        'import torch.nn.functional as F',
        '',
        'def chunked_cross_entropy_forward(X, linear, labels, start_idx, end_idx):',
        '    """Cross entropy forward function for chunked processing."""',
        '    logits = linear(X)  # [batch_size, chunk_size]',
        '    ',
        '    if labels is not None:',
        '        # Mask labels that are not in this chunk',
        '        mask = (labels >= start_idx) & (labels < end_idx)',
        '        if mask.any():',
        '            chunk_labels = labels[mask] - start_idx  # Adjust to chunk-local indices',
        '            chunk_logits = logits[mask]',
        '            loss = F.cross_entropy(chunk_logits, chunk_labels, reduction=\'mean\')',
        '            # Scale by fraction of labels in this chunk',
        '            return loss * mask.float().mean().item()',
        '        else:',
        '            return torch.tensor(0.0, device=X.device, dtype=X.dtype)',
        '    else:',
        '        # For backward pass - return dummy loss',
        '        return torch.tensor(0.0, device=X.device, dtype=X.dtype)',
        '',
        'def chunked_kl_div_forward(X, linear, labels, start_idx, end_idx):',
        '    """KL Divergence forward function for chunked processing."""',
        '    logits = linear(X)  # [batch_size, chunk_size]',
        '    log_probs = F.log_softmax(logits, dim=-1)',
        '    ',
        '    if labels is not None:',
        '        # For simplicity, assume labels are target distributions',
        '        mask = (labels >= start_idx) & (labels < end_idx)',
        '        if mask.any():',
        '            chunk_labels = labels[mask] - start_idx',
        '            chunk_log_probs = log_probs[mask]',
        '            target_probs = F.one_hot(chunk_labels, num_classes=end_idx-start_idx).float()',
        '            loss = F.kl_div(chunk_log_probs, target_probs, reduction=\'batchmean\')',
        '            return loss * mask.float().mean().item()',
        '        else:',
        '            return torch.tensor(0.0, device=X.device, dtype=X.dtype)',
        '    else:',
        '        return torch.tensor(0.0, device=X.device, dtype=X.dtype)',
        '',
        'def memory_efficient_forward(X, linear, labels, forward_fn, chunk_size=4096):',
        '    """Wrapper for MemoryEfficientLinear forward."""',
        '    return MemoryEfficientLinear.apply(X, linear, labels, forward_fn, chunk_size)',
        '',
        'def vanilla_forward(X, linear, labels, forward_fn):',
        '    """Vanilla forward for comparison."""',
        '    return forward_fn(X, linear, labels, 0, linear.weight.shape[0])'
    ]
}

# Insert at position 36 (after cell 35)
nb['cells'].insert(36, supporting_functions)

# Save the notebook
with open('/home/engine/project/Unsloth_Puzzles.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Added supporting functions cell at position 36")
print(f"Total cells in notebook: {len(nb['cells'])}")