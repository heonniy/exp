from experiments.data.prepare_lmsys import build_fixed_example


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        history,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        tokens = []
        for message in history:
            tokens.extend(ord(character) for character in message["content"])
        tokens.extend([900, 901])
        return tokens


def test_fixed_example_is_real_token_truncation() -> None:
    conversation = [
        {"role": "user", "content": "old context"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "final user request"},
        {"role": "assistant", "content": "target answer text"},
    ]
    example, rejection = build_fixed_example(
        DummyTokenizer(), "id", conversation, input_tokens=12, output_tokens=6
    )
    assert rejection is None
    assert example["input_length"] == 12
    assert example["output_length"] == 6
    assert example["input_ids"][-2:] == [900, 901]
    assert example["forced_output_ids"] == [ord(c) for c in "target"]


def test_short_target_is_rejected() -> None:
    conversation = [
        {"role": "user", "content": "long enough prompt"},
        {"role": "assistant", "content": "no"},
    ]
    example, rejection = build_fixed_example(
        DummyTokenizer(), "id", conversation, input_tokens=4, output_tokens=3
    )
    assert example is None
    assert rejection == "short_output"
