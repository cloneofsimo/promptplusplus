import torch


def pp_extend(text, main_token, preserve_prefix, extend_amount):
    token_identifier = main_token[1:-1]
    assert token_identifier in text, f"{token_identifier} not in {text}"

    rets = []

    for idx in range(16):
        total_token = main_token if preserve_prefix else ""

        for jdx in range(extend_amount):
            total_token = total_token + f"<{token_identifier}-{idx}-{jdx}>"

        _text = text.replace(f"<{token_identifier}>", total_token)
        rets.append(_text)

    return rets


class PPPAttenProc:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None):

        is_dict_format = True
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, dict):
                this_idx = encoder_hidden_states["this_idx"]

                _ehs = encoder_hidden_states[f"CONTEXT_TENSOR_{this_idx}"]
                encoder_hidden_states["this_idx"] += 1
                encoder_hidden_states["this_idx"] %= 16
            else:
                _ehs = encoder_hidden_states
        else:
            _ehs = None

        batch_size, sequence_length, _ = (
            hidden_states.shape if _ehs is None else _ehs.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if _ehs is None:
            _ehs = hidden_states
        elif attn.cross_attention_norm:
            _ehs = attn.norm_cross(_ehs)

        key = attn.to_k(_ehs)
        value = attn.to_v(_ehs)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class PPPPromptManager:
    def __init__(
        self, tokenizer, text_encoder, main_token, preserve_prefix, extend_amount
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.main_token = main_token
        self.preserve_prefix = preserve_prefix
        self.extend_amount = extend_amount

    def expand_prompt(self, text: str):

        pp_extended = pp_extend(
            text, self.main_token, self.preserve_prefix, self.extend_amount
        )

        return pp_extended

    def embed_prompt(self, text: str):
        texts = self.expand_prompt(text)
        ids = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        encoder_hidden_states = self.text_encoder(ids.to(self.text_encoder.device))[0]
        _hs = {"this_idx": 0}
        for idx in range(16):
            _hs[f"CONTEXT_TENSOR_{idx}"] = encoder_hidden_states[idx : idx + 1, :, :]

        return _hs
