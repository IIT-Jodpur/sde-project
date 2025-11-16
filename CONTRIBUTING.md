# Contributing to GPU Time Slicing Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a new branch for your feature/fix
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone repository
git clone <your-fork-url>
cd SDE

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

We follow PEP 8 style guidelines with a few modifications:

- Line length: 100 characters max
- Use black for formatting: `black .`
- Use flake8 for linting: `flake8 .`
- Use type hints where appropriate

## Adding New Workloads

To add a new ML workload:

1. Create a new Python file in appropriate directory:
   - `workloads/training/` for training workloads
   - `workloads/inference/` for inference workloads
   - `workloads/interactive/` for interactive workloads

2. Include the following components:
   ```python
   import argparse
   import json
   import os
   from datetime import datetime
   
   def your_workload_function(args):
       # Workload implementation
       pass
   
   if __name__ == '__main__':
       parser = argparse.ArgumentParser(description='Your Workload')
       # Add arguments
       args = parser.parse_args()
       your_workload_function(args)
   ```

3. Save results in JSON format to `results/` directory

4. Update documentation

## Adding GPU Sharing Modes

To add a new GPU sharing configuration:

1. Create setup script in `gpu-configs/`
2. Follow naming convention: `setup_<mode>.sh` and `disable_<mode>.sh`
3. Include comprehensive error checking
4. Update `gpu-configs/README.md` with usage instructions

## Testing

Run tests before submitting PR:

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Linting
flake8 .
black --check .
mypy .
```

## Pull Request Process

1. **Update documentation** for any changes to functionality
2. **Add tests** for new features
3. **Follow commit message conventions**:
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation changes
   - refactor: Code refactoring
   - test: Adding tests
   - chore: Maintenance tasks

4. **Ensure all tests pass**
5. **Request review** from maintainers

## Example Contributions

### Adding a New Benchmark Metric

```python
# In benchmarking/benchmark_suite.py

def calculate_aggregate_metrics(self):
    # ... existing code ...
    
    # Add new metric
    self.results['aggregate_metrics']['your_new_metric'] = compute_your_metric()
```

### Adding Monitoring Metric

```python
# In monitoring/gpu_monitor.py

def get_gpu_metrics(self):
    metrics = {
        # ... existing metrics ...
        'your_new_metric': get_your_metric()
    }
    return metrics
```

## Documentation

- Update README.md for major changes
- Update QUICKSTART.md for workflow changes
- Add docstrings to all functions
- Include usage examples

## Questions?

- Open an issue for bugs
- Start a discussion for feature requests
- Check existing issues/PRs before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

Thank you for contributing! 🎉

