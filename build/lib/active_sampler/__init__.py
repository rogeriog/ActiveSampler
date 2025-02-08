# active_sampler/active_sampler/__init__.py
import logging

# Set up logging to only output the message
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Only display the log message without prefixes
    handlers=[
        logging.StreamHandler()  # Ensure logs print to the console
    ]
)
logger = logging.getLogger(__name__)

from .core import active_sampling, generate_sampling_grid
from .utils import load_and_preprocess_data, get_unique_kfold_splits

__all__ = ["active_sampling", "load_and_preprocess_data"]