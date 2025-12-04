"""
Usage:
    python vizualize.py --config examples/mnist/mnist.toml
"""

import sys

from src.visualization.metrics import get_visualization_config
from src.visualization.metrics import dashboard

from src.config.configuration import build_config, Config


def main(argv=None) -> int:
    # Load configuration from TOML file
    cfg: Config = build_config(argv)
    baseline, csv_path, output_path = get_visualization_config(cfg)

    dashboard(baseline, csv_path, output_path, cfg.data.name, cfg.continuous_learning.max_iter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
