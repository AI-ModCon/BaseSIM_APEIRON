"""
This should be the driver for the SIM loop.

"""

from src.config.configuration import build_config, Config


def main(argv=None) -> int:
    cfg: Config = build_config(argv)

    print(cfg)

    """
    Todo: structure the driver.   
    
    """

    return 0
