import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM, PeftConfig
except ImportError:  # peft is optional unless loading adapters
    AutoPeftModelForCausalLM = None
    PeftConfig = None


def get_device_config():
    if torch.cuda.is_available():
        return {"device": "cuda", "device_map": "auto", "dtype": torch.float16}
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return {"device": "mps", "device_map": None, "dtype": torch.float16}
    return {"device": "cpu", "device_map": None, "dtype": torch.float32}


def _ensure_padding(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


@st.cache_resource(show_spinner="Loading model...")
def load_chat_model(
    model_id: str, device_map: str | None, dtype: torch.dtype, target_device: str
):
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    else:
        load_kwargs["low_cpu_mem_usage"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except OSError as original_error:
        if PeftConfig is None or AutoPeftModelForCausalLM is None:
            raise RuntimeError(
                "Model weights were not found. If this is a PEFT adapter, install peft and retry."
            ) from original_error
        try:
            peft_config = PeftConfig.from_pretrained(model_id)
        except Exception as peft_error:
            raise original_error from peft_error
        base_model_id = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    _ensure_padding(tokenizer)
    if device_map is None:
        model.to(torch.device(target_device))
    model.eval()
    return tokenizer, model


def generate_response(
    tokenizer, model, messages, max_new_tokens: int, temperature: float
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    target_device = getattr(model, "device", None)
    if target_device is None:
        target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
    prompt_length = input_ids.shape[-1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        generated_ids = model.generate(input_ids, **generation_kwargs)

    response_ids = generated_ids[0, prompt_length:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response_text


def clear_conversation():
    st.session_state.messages = []


st.set_page_config(page_title="Eliciting Latent Knowledge", page_icon="ELK")
st.title("Eliciting Latent Knowledge")
st.write("Chat with a Hugging Face model while preserving the conversation context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Model settings")
    model_id = st.text_input(
        "Hugging Face model",
        value="bcywinski/gemma-2-9b-it-taboo-ship",
        help="Any chat-capable AutoModelForCausalLM checkpoint.",
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 256, 16)
    if st.button("Clear conversation"):
        clear_conversation()

device_cfg = get_device_config()

if not model_id:
    st.info("Enter a model id in the sidebar to begin.")
    st.stop()

try:
    tokenizer, model = load_chat_model(
        model_id, device_cfg["device_map"], device_cfg["dtype"], device_cfg["device"]
    )
except Exception as exc:
    st.error(f"Could not load model `{model_id}`: {exc}")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Send a message")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Generating response..."):
        try:
            response = generate_response(
                tokenizer,
                model,
                st.session_state.messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
