from core.residuals import get_inner


def make_ablation_hook(r_hat_cpu):
    """
    Forward hook that projects r_hat out of the residual stream output (h <- h - (h @ r) * r). 
    r_hat_cpu must be pre-normalized; it is moved to the activation's device/dtype on the fly.
    
    Input:
    - r_hat_cpu: the reference vector

    Output: hook function
    """
    # Define the hook function
    def hook(module, inputs, output):
        # Check if the output is a tuple
        is_tuple = isinstance(output, tuple)

        # Get the hidden state
        h = output[0] if is_tuple else output

        # Check if the hidden state is a 3D tensor
        if h.dim() != 3:
            return output
        r = r_hat_cpu.to(h.device).to(h.dtype)
        proj = (h @ r).unsqueeze(-1)
        h_new = h - proj * r
        if is_tuple:
            return (h_new,) + output[1:]
        return h_new

    return hook


def install_ablation(model, r_hat_cpu, layers):
    """
    Install a forward hook that projects a (pre-normalized) direction out of the residual stream at each listed layer:
    h <- h - (h . r) r
    
    Input:
    - model: the model to install the hook on
    - r_hat_cpu: the reference vector
    - layers: list of layer names

    Output: list of handles
    """
    # Get the inner model
    inner = get_inner(model)
    handles = []

    # Install the hook for each layer
    for L in layers:
        handles.append(inner.layers[L].register_forward_hook(make_ablation_hook(r_hat_cpu)))
    return handles


def remove_handles(handles):
    """
    Remove a list of forward hook handles.

    Input:
    - handles: list of forward hook handles

    Output: None
    """

    # Remove the handles
    for h in handles:
        h.remove()
