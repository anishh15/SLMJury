"""SLMJury: Can Small Language Models Judge Reasoning as Well as Large Ones?

A framework for evaluating Small Language Models (0.6B-14B) as judges
of reasoning quality, with support for majority voting ensembles,
multi-agent debate, and persona effects.

Key Features:
- 17 SLM judges across Qwen, Llama, and Phi-4 model families
- 5 datasets (GSM8K, GSM-Plus, MATH, ARC-Easy, ARC-Challenge)
- Majority voting ensembles from top-5 judges
- Multi-agent debate with RCR prompting
- 6 persona system prompts for bias analysis
"""

__version__ = "0.1.0"
__author__ = "Anish Laddha, Nitesh Pradhan, Gaurav Srivastava"
__license__ = "Apache-2.0"

# Configs
from slmjury.configs import load_models_config

# Parsers
from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.judgement import parse_judgement, parse_judgement_detailed
from slmjury.parsers.normalizer import answers_are_equivalent, normalize_numeric

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
    # Data
    "load_dataset",
    # Strategies
    "majority_vote",
    "get_ensemble_judges",
    "get_personas",
    "PERSONAS",
]
