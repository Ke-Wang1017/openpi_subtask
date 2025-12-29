import logging
import os
import string

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "/workspace/fast"):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        
        # ⭐ 可选：加载 FAST tokenizer
        self._fast_tokenizer = None
        self._fast_skip_tokens = 128
        if fast_tokenizer_path is not None:
                self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, local_files_only=True, trust_remote_code=True)
                logging.info(f"Loaded FAST tokenizer from {fast_tokenizer_path}")


    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def tokenize_high_low_prompt_infer(self, high_prompt: str, low_prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_high_text = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        cleaned_low_text = low_prompt.lower().strip().replace("_", " ").replace("\n", " ")

        # This is the Pi05 format, where the state is part of the discrete language input.
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))

        # Remove the last punctuation character if present and add custom symbol
        if cleaned_high_text and cleaned_high_text[-1] in string.punctuation:
            cleaned_high_text = cleaned_high_text[:-1]
        cleaned_high_text += '.'  # Add your custom symbol here
        sub_prompt_1 = f"Task: {cleaned_high_text}, State: {state_str}; Subtask: "
        tokens_1 = self._tokenizer.encode(sub_prompt_1, add_bos=True)
        ar_mask = [1] * len(tokens_1)
        loss_mask = [False] * len(tokens_1)

        # Remove the last punctuation character if present and add custom symbol
        # if cleaned_low_text and cleaned_low_text[-1] in string.punctuation:
        #     cleaned_low_text = cleaned_low_text[:-1]
        # # cleaned_low_text += '.'  # Add your custom symbol here
        # sub_prompt_2 = f"{cleaned_low_text}" 
        # # sub_prompt_2 = f"{cleaned_low_text}"# Warning: State is not included here which is different from original pi05
        # tokens_2 = self._tokenizer.encode(sub_prompt_2, add_eos=True)
        # ar_mask += [1] * len(tokens_2)
        # loss_mask += [True] * len(tokens_2)


        # tokens = tokens_1 + tokens_2
        tokens = tokens_1 
        

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def tokenize_high_level_prompt(self, high_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_high_text = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        if cleaned_high_text and cleaned_high_text[-1] in string.punctuation:
            cleaned_high_text = cleaned_high_text[:-1]
        cleaned_high_text += '.'  # Add your custom symbol here
        sub_prompt_1 = f"Task: {cleaned_high_text} Subtask: "
        tokens_1 = self._tokenizer.encode(sub_prompt_1, add_bos=True)
        if len(tokens_1) < self._max_len:
            padding = [False] * (self._max_len - len(tokens_1))
            tokens = tokens_1 + padding
            mask = [True] * len(tokens_1) + padding
        else:
            if len(tokens_1) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens_1)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens_1 = tokens_1[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens), np.asarray(mask)

    def tokenize_high_low_prompt(
        self, 
        high_prompt: str, 
        low_prompt: str, 
        state: np.ndarray | None = None,
        actions: np.ndarray | None = None  # ⭐ 新增：可选的 actions 用于 FAST tokenization
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize high-low prompt with optional FAST action tokens.
        
        Returns:
            tuple: (tokens, mask, ar_mask, loss_mask, subtask_region_mask, action_region_mask)
            - tokens: Token IDs
            - mask: Valid token mask
            - ar_mask: Auto-regressive mask
            - loss_mask: Overall loss mask (subtask_region_mask OR action_region_mask)
            - subtask_region_mask: Mask for subtask tokens (for separate loss calculation)
            - action_region_mask: Mask for action tokens (for separate loss calculation)
        
        If actions is None or FAST tokenizer not loaded, only generate subtask tokens (for flow matching mode).
        If actions is provided and FAST tokenizer loaded, append FAST action tokens.
        """
        cleaned_high_text = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        cleaned_low_text = low_prompt.lower().strip().replace("_", " ").replace("\n", " ")

        # This is the Pi05 format, where the state is part of the discrete language input.
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))

        # 1. High prompt (不计算 loss)
        if cleaned_high_text and cleaned_high_text[-1] in string.punctuation:
            cleaned_high_text = cleaned_high_text[:-1]
        cleaned_high_text += '.'
        sub_prompt_1 = f"Task: {cleaned_high_text}; State: {state_str}; Subtask: "
        tokens_1 = self._tokenizer.encode(sub_prompt_1, add_bos=True)
        ar_mask = [1] * len(tokens_1)
        loss_mask = [False] * len(tokens_1)
        subtask_region_mask = [False] * len(tokens_1)  # ⭐ 标记 subtask 区域
        action_region_mask = [False] * len(tokens_1)   # ⭐ 标记 action token 区域

        # 2. Low prompt (subtask) - 根据是否需要 FAST tokens 决定结尾
        if cleaned_low_text and cleaned_low_text[-1] in string.punctuation:
            cleaned_low_text = cleaned_low_text[:-1]
        cleaned_low_text += '.'
        
        if actions is None or self._fast_tokenizer is None:
            # Flow matching mode: 以 ";\nAction: " 结尾
            sub_prompt_2 = f"{cleaned_low_text};\nAction: "
            tokens_2 = self._tokenizer.encode(sub_prompt_2, add_eos=True)
        else:
            # FAST token mode: 先 subtask，后面会加 action tokens
            sub_prompt_2 = f"{cleaned_low_text};"
            tokens_2 = self._tokenizer.encode(sub_prompt_2)
        
        ar_mask += [1] * len(tokens_2)
        loss_mask += [True] * len(tokens_2)
        subtask_region_mask += [True] * len(tokens_2)  # ⭐ Subtask tokens
        action_region_mask += [False] * len(tokens_2)
        
        tokens = tokens_1 + tokens_2

        # 3. ⭐ 可选：添加 FAST action tokens
        if actions is not None and self._fast_tokenizer is not None:
            # Tokenize actions with FAST
            action_tokens_fast = self._fast_tokenizer(actions[None])[0]
            action_tokens_pg = self._act_tokens_to_paligemma_tokens(action_tokens_fast)
            
            # Add "Action: " prefix and "|" suffix
            action_seq = (
                self._tokenizer.encode("\nAction: ") +
                action_tokens_pg.tolist() +
                self._tokenizer.encode("|", add_eos=True)
            )
            
            tokens += action_seq
            ar_mask += [1] * len(action_seq)
            loss_mask += [True] * len(action_seq)
            subtask_region_mask += [False] * len(action_seq)
            action_region_mask += [True] * len(action_seq)  # ⭐ Action tokens

        # 4. Padding
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
            subtask_region_mask = subtask_region_mask + padding
            action_region_mask = action_region_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
            subtask_region_mask = subtask_region_mask[: self._max_len]
            action_region_mask = action_region_mask[: self._max_len]

        return (
            np.asarray(tokens), 
            np.asarray(mask), 
            np.asarray(ar_mask), 
            np.asarray(loss_mask),
            np.asarray(subtask_region_mask),  # ⭐ 新增
            np.asarray(action_region_mask)     # ⭐ 新增
        )
    
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def detokenize(self, tokens: np.ndarray) -> str:
        """Decode tokens back to text, removing padding tokens."""
        # Remove padding tokens (tokens with value 0)
        non_padding_tokens = tokens[tokens != 0]
        return self._tokenizer.decode(non_padding_tokens.tolist())


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("/root/.cache/openpi/big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def tokenize_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        text_tokens = self._paligemma_tokenizer.encode(cleaned_text, add_bos=True)
        token_mask = [True] * len(text_tokens)
        ar_mask = [0] * len(text_tokens)
        loss_mask = [False] * len(text_tokens)

        tokens_len = len(text_tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            text_tokens = text_tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            assert False

        return np.asarray(text_tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
