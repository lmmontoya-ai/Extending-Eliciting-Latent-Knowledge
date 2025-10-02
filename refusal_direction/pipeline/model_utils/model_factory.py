from refusal_direction.pipeline.model_utils.model_base import ModelBase


def construct_model_base(model_path: str) -> ModelBase:
    model_path_lower = model_path.lower()

    if "qwen" in model_path_lower:
        from refusal_direction.pipeline.model_utils.qwen_model import QwenModel

        return QwenModel(model_path)

    if "llama-3" in model_path_lower:
        from refusal_direction.pipeline.model_utils.llama3_model import Llama3Model

        return Llama3Model(model_path)

    if "llama" in model_path_lower:
        from refusal_direction.pipeline.model_utils.llama2_model import Llama2Model

        return Llama2Model(model_path)

    if "gemma-2" in model_path_lower or "gemma2" in model_path_lower:
        from refusal_direction.pipeline.model_utils.gemma2_model import Gemma2Model

        return Gemma2Model(model_path)

    if "gemma" in model_path_lower:
        from refusal_direction.pipeline.model_utils.gemma_model import GemmaModel

        return GemmaModel(model_path)

    if "yi" in model_path_lower:
        from refusal_direction.pipeline.model_utils.yi_model import YiModel

        return YiModel(model_path)

    raise ValueError(f"Unknown model family: {model_path}")
