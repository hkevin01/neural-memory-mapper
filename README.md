# Neural Memory Mapper

A Brain-Computer Interface (BCI) system that visualizes memory formation patterns in real-time, helping users optimize learning and recall through neurofeedback.

## Features

- Real-time EEG signal processing and frequency analysis
- Memory task engine with various cognitive exercises
- Brain activity visualization and memory formation patterns
- Interactive neurofeedback system
- Performance tracking and analytics

## Requirements

- Python 3.8+
- OpenBCI, Emotiv headset, or mock data generator
- Dependencies listed in `requirements.txt`

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/hkevin01/neural-memory-mapper.git
cd neural-memory-mapper
```

1. Set up the development environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

1. Run the setup script:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## Project Structure

```text
neural-memory-mapper/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── data_acquisition.py
│   │   └── signal_processor.py
│   ├── tasks/             # Memory tasks
│   │   └── memory_tasks.py
│   ├── visualization/     # Visualization components
│   │   └── brain_visualizer.py
│   └── utils/             # Utility functions
│       └── config.py
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/               # Configuration files
├── data/                 # Data storage
│   ├── raw/             # Raw EEG data
│   └── processed/       # Processed data
├── assets/              # Static assets
├── scripts/             # Utility scripts
└── .github/             # GitHub specific files
```

## Memory Tasks

### Word List Task

- Configurable list size and presentation time
- Performance metrics for recall accuracy
- Difficulty levels: easy, medium, hard

### Pattern Memory Task

- Grid-based spatial memory exercise
- Sequence memory with increasing complexity
- Real-time performance tracking

## Brain Activity Visualization

- Real-time heatmap of brain activity
- Frequency band power visualization
- Memory formation strength indicators
- State transition visualization

## Development

### Testing

Run the test suite:

```bash
pytest
```

Generate coverage report:

```bash
pytest --cov=src --cov-report=html
```

### Code Style

This project follows:

- Black for Python code formatting
- Pylint for Python code analysis
- Type hints for better code understanding

## Configuration

The system can be configured through `config/default_config.yml`:

- EEG device settings
- Signal processing parameters
- Visualization preferences
- Memory task configurations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See [CONTRIBUTING.md](/.github/CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Kevin H (@hkevin01)

## Acknowledgments

- OpenBCI for BCI hardware support
- MNE-Python for signal processing capabilities
- PyViz for visualization components
