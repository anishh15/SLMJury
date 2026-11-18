"""SLMJury: Can Small Language Models Judge as Well as Large Ones?

16 small judges (0.6B-14B) and six large ones (70B-122B), configured in slmjury/configs/models.yaml, on 8
closed-ended and 2 open-ended datasets, with quick and reasoned verdicts, personas, juries and debate.
"""

__version__ = "0.2.0"
__author__ = "Anish Laddha, Nitesh Pradhan, Gaurav Srivastava"
__license__ = "Apache-2.0"

# Configs
from slmjury.configs import load_models_config

# Parsers
from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.judgement import parse_judgement, parse_judgement_detailed
from slmjury.parsers.normalizer import answers_are_equivalent, normalize_numeric
from slmjury.parsers.score import parse_score, parse_multi_score

# Data loaders
from slmjury.data import load_dataset

# Strategies
from slmjury.strategies.ensemble import majority_vote, get_ensemble_judges
from slmjury.strategies.persona import get_personas, PERSONAS

__all__ = [
    # Configs
    "load_models_config",
    # Parsers
    "extract_answer",
    "get_dataset_type",
    "parse_judgement",
    "parse_judgement_detailed",
    "answers_are_equivalent",
    "normalize_numeric",
    "parse_score",
    "parse_multi_score",
    # Data
    "load_dataset",
    # Strategies
    "majority_vote",
    "get_ensemble_judges",
    "get_personas",
    "PERSONAS",
]
