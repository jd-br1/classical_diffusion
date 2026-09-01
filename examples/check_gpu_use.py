import jax

# Check the active backend
print(jax.default_backend())  # Output: 'gpu' or 'cpu'

# List all available devices
print(jax.devices())  # Output: [cuda(id=0)] if GPU is active
