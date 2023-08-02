from pathlib import Path

import torch

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import safetensors.torch as safe_torch


def _load_state_dict(path_to_checkpoint: str | Path) -> torch.Tensor:
    path_to_checkpoint = str(path_to_checkpoint)
    if (
        path_to_checkpoint.endswith(".ckpt")
        or path_to_checkpoint.endswith(".pt")
        or path_to_checkpoint.endswith(".bin")
    ):
        state_dict = torch.load(path_to_checkpoint, map_location="cpu")
    elif path_to_checkpoint.endswith(".safetensors"):
        state_dict = safe_torch.load_file(path_to_checkpoint, device="cpu")
    else:
        raise ValueError(
            f"Unknown checkpoint format: {path_to_checkpoint}. Supported formats are .ckpt, .pt, .bin, .safetensors"
        )
    return state_dict


def _format_token(token: str) -> str:
    return f"<{token}>"  # TI と同じ名前で学習できるように別の名前にする


# ref: https://github.com/huggingface/diffusers/blob/6c49d542a352b556e412d8763592050d3a0dec77/src/diffusers/loaders.py#L641
# TI の名前と埋め込みだけもつクラス
class TextualInversionModel:
    token: str
    embedding: torch.Tensor

    def __init__(self, token: str, embedding: torch.Tensor):
        self.token = token
        self.embedding = embedding

    def from_pretrained(pretrained_model_name_or_path: str):
        model_path = Path(pretrained_model_name_or_path)
        if model_path.exists():
            # ローカルにある
            state_dict = _load_state_dict(model_path)

            if "string_to_param" in state_dict:
                embedding = state_dict["string_to_param"]["*"]  # AUTOMATIC1111 style
            elif "emb_params" in state_dict:
                embedding = state_dict["emb_params"]  # kohya-ss style
            elif len(state_dict) == 1:
                _, embedding = next(iter(state_dict.items()))  # diffusers style
            else:
                raise ValueError(
                    f"Unknown state dict format for textual inversion model: {pretrained_model_name_or_path}"
                )

            token = _format_token(model_path.stem)  # TIと同じ名前で学習できるようにこっちをずらす

            return TextualInversionModel(token, embedding)
        else:
            # diffusers 形式のモデル
            raise NotImplementedError(
                "diffusers format textual inversiom model is not supported."
            )

    def apply_textual_inversion(
        self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel
    ) -> tuple[CLIPTokenizer, CLIPTextModel]:
        token_ids_and_embeddings = []

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = tokenizer.get_vocab()
        if self.token in vocab:
            raise ValueError(
                f"Token {self.token} already in tokenizer vocabulary. Please choose a different token name or remove {self.token} and embedding from the tokenizer and text encoder."
            )
        elif f"{self.token}_1" in vocab:
            multi_vector_tokens = [self.token]
            i = 1
            while f"{self.token}_{i}" in tokenizer.added_tokens_encoder:
                multi_vector_tokens.append(f"{self.token}_{i}")
                i += 1

            raise ValueError(
                f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
            )

        is_multi_vector = len(self.embedding.shape) > 1 and self.embedding.shape[0] > 1

        if is_multi_vector:
            tokens = [self.token] + [
                f"{self.token}_{i}" for i in range(1, self.embedding.shape[0])
            ]
            embeddings = [e for e in self.embedding]  # noqa: C416
        else:
            tokens = [self.token]
            embeddings = (
                [self.embedding[0]]
                if len(self.embedding.shape) > 1
                else [self.embedding]
            )

        # add tokens and get ids
        tokenizer.add_tokens(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_and_embeddings += zip(token_ids, embeddings)

        # resize token embeddings and set all new embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding

        return tokenizer, text_encoder


def get_all_embeddings_in_folder(path_to_embeddings: str | Path) -> list[Path]:
    path_to_embeddings = Path(path_to_embeddings)

    if path_to_embeddings.exists():
        # 再帰的に全部取得 (拡張子が .bin, .pt, .ckpt, .safetensors のもの)
        embeddings = []
        for path in path_to_embeddings.glob("**/*"):
            if path.suffix in [".bin", ".pt", ".ckpt", ".safetensors"]:
                embeddings.append(path)
        return embeddings
    else:
        raise ValueError(f"{path_to_embeddings} is not exists.")


def filter_embeddings_by_prompt(embeddings: list[Path], prompt: str):
    return [
        embedding for embedding in embeddings if _format_token(embedding.stem) in prompt
    ]
