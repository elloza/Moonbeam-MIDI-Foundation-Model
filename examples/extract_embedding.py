import argparse
import torch
from pathlib import Path
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.configuration_llama import (
    LlamaConfig,
)
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)

# --------------------------------------------------------------------------------------
# utilidades nuevas --------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def count_tokens(tokenizer: MusicTokenizer, midi_path: str,
                 add_sos: bool = True, add_eos: bool = True) -> int:
    compound = tokenizer.midi_to_compound(midi_path)
    encoded  = tokenizer.encode_series(compound, if_add_sos=add_sos, if_add_eos=add_eos)
    return len(encoded)

def split_into_windows(encoded: list[int], max_len: int = 1021) -> list[list[int]]:
    """
    Divide la lista de tokens en segmentos de longitud <= max_len.
    max_len debería dejar hueco para los tokens especiales (<sos>, <eos>, <cls>...).
    """
    return [encoded[i : i + max_len] for i in range(0, len(encoded), max_len)]

# --------------------------------------------------------------------------------------
# código original con pequeños cambios -------------------------------------------------
# --------------------------------------------------------------------------------------
def load_model(checkpoint_path: str, config_path: str) -> LlamaForCausalLM:
    config = LlamaConfig.from_pretrained(config_path)
    model  = LlamaForCausalLM(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
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


def midi_to_input_ids(tokenizer: MusicTokenizer, midi_path: str,
                      add_sos=True, add_eos=True) -> torch.Tensor:
    compound = tokenizer.midi_to_compound(midi_path)
    encoded  = tokenizer.encode_series(compound, if_add_sos=add_sos, if_add_eos=add_eos)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0), encoded  # devuelvo ambos


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
    parser.add_argument(
        "--max-len",
        type=int,
        default=1021,   # 1024 - (sos/eos/cls) si los añades a mano
        help="Optional sliding-window size (tokens) for long MIDIs",
    )
    args = parser.parse_args()

    if not Path(args.midi).is_file():
        raise FileNotFoundError(f"MIDI not found: {args.midi}")

    model = load_model(args.checkpoint, args.config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    config    = model.config
    tokenizer = get_tokenizer(config)

    # --- contar tokens ---------------------------------------------------------------
    n_tokens = count_tokens(tokenizer, args.midi)
    print(f"► La pieza contiene {n_tokens} tokens (incl. <sos>/<eos>).")

    # --- codificar -------------------------------------------------------------------
    input_tensor, encoded = midi_to_input_ids(tokenizer, args.midi)

    # --- ¿excede el límite? ----------------------------------------------------------
    if n_tokens > 1024 and args.max_len:
        print(f"⚠️  {n_tokens} supera el límite 1024: aplico sliding-window de {args.max_len}.")
        windows = split_into_windows(encoded, max_len=args.max_len)
        embs = []
        for w in windows:
            ids = torch.tensor(w, dtype=torch.long).unsqueeze(0)
            embs.append(extract_embedding(model, ids))
        embedding = torch.stack(embs).mean(0)          # agrega como prefieras
    else:
        embedding = extract_embedding(model, input_tensor)

    print("Embedding shape:", embedding.shape)
    print(embedding)


if __name__ == "__main__":
    main()
