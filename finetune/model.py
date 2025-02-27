from taoverse.model.competition.data import ModelConstraints
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_tokenizer(
    model_constraints: ModelConstraints, cache_dir: str = None
) -> PreTrainedTokenizer:
    """Returns the fixed tokenizer for the given model constraints."""
    return AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_constraints.model_name,
    cache_dir=cache_dir,
    token=os.getenv("HF_ACCESS_TOKEN")  # Ensure the token is used properly
)


