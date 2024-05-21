# Utilities.
#
# Created by Yi on 21/05/2024.
#

from datetime import datetime
from pathlib import Path

from loguru import logger


def get_timestamp(fmt: str = '%Y-%m-%d %H-%M-%S'):
    return datetime.now().strftime(fmt)


DEFAULT_LOG_DIR = Path('log') / f'{get_timestamp()}'
DEFAULT_LOG_DIR.parent.mkdir(parents=True, exist_ok=True)


def init_logger(name, out_dir=None, level='INFO'):
    logger.remove()  # remove the initial handler

    if out_dir is None:
        out_dir = DEFAULT_LOG_DIR

    out_name = out_dir / name / f'{get_timestamp()}'
    logger.add(out_name.with_suffix(".log"), format="{time} {level} {message}", level=level)
    return logger
