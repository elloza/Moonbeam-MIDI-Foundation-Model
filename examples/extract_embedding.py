import argparse
import torch
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

from llama_recipes.transformers_minimal.src.transformers.models.llama.configuration_llama import (
    LlamaConfig,
)
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)

def load_model(checkpoint_path: str, config_path: str) -> LlamaForCausalLM:
    """Load the pretrained model weights."""
    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt["model_state_dict"]
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def get_tokenizer(config: LlamaConfig) -> MusicTokenizer:
    return MusicTokenizer(
        timeshift_vocab_size=config.onset_vocab_size,
        dur_vocab_size=config.dur_vocab_size,
        octave_vocab_size=config.octave_vocab_size,
        pitch_class_vocab_size=config.pitch_class_vocab_size,
        instrument_vocab_size=config.instrument_vocab_size,
        velocity_vocab_size=config.velocity_vocab_size,
        sos_token=config.sos_token,
        eos_token=config.eos_token,
        pad_token=config.pad_token,
    )


def midi_to_input_ids(tokenizer: MusicTokenizer, midi_path: str) -> torch.Tensor:
    compound = tokenizer.midi_to_compound(midi_path)
    encoded = tokenizer.encode_series(compound, if_add_sos=True, if_add_eos=True)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def extract_embedding(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(model.device), output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Extract Moonbeam embedding from MIDI")
    parser.add_argument("midi", help="Path to MIDI file")
    parser.add_argument("checkpoint", help="Path to moonbeam_*.pt checkpoint")
    parser.add_argument(
        "--config",
        default="src/llama_recipes/configs/model_config.json",
        help="Path to model configuration JSON",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    config = model.config
    tokenizer = get_tokenizer(config)
    input_ids = midi_to_input_ids(tokenizer, args.midi)

    embedding = extract_embedding(model, input_ids)
    print("Embedding shape:", embedding.shape)
    print(embedding)


if __name__ == "__main__":
    main()
