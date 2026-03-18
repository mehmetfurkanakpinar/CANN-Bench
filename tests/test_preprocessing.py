"""Unit tests for src/preprocessing/engine.py"""

import pytest
import pandas as pd
from src.preprocessing.engine import clean_text, shannon_entropy, preprocess_dataframe


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_url(self):
        assert "http" not in clean_text("Visit https://example.com for info")

    def test_removes_punctuation(self):
        result = clean_text("Hello, world!")
        assert "," not in result and "!" not in result

    def test_collapses_whitespace(self):
        assert clean_text("too   many   spaces") == "too many spaces"

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            clean_text(12345)

    def test_empty_string(self):
        assert clean_text("") == ""


class TestShannonEntropy:
    def test_empty_text_returns_zero(self):
        assert shannon_entropy("") == 0.0

    def test_single_token_returns_zero(self):
        assert shannon_entropy("hello") == 0.0

    def test_uniform_distribution_is_higher(self):
        uniform = shannon_entropy("a b c d e f g h")
        skewed = shannon_entropy("a a a a a a a b")
        assert uniform > skewed

    def test_repeated_tokens_low_entropy(self):
        result = shannon_entropy("the the the the the")
        assert result == 0.0

    def test_returns_float(self):
        assert isinstance(shannon_entropy("hello world"), float)


class TestPreprocessDataframe:
    def _make_df(self):
        return pd.DataFrame({
            "text": ["Hello world!", "The quick brown fox", "test test test"],
            "label": [0, 1, 0],
        })

    def test_output_has_expected_columns(self):
        df = preprocess_dataframe(self._make_df())
        assert set(df.columns) == {"text_clean", "entropy", "label"}

    def test_output_length_matches_input(self):
        df = self._make_df()
        result = preprocess_dataframe(df)
        assert len(result) == len(df)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            preprocess_dataframe(pd.DataFrame())

    def test_missing_text_column_raises(self):
        df = pd.DataFrame({"body": ["some text"], "label": [1]})
        with pytest.raises(ValueError, match="text"):
            preprocess_dataframe(df, text_col="text")

    def test_no_label_col(self):
        df = pd.DataFrame({"text": ["hello world", "foo bar"]})
        result = preprocess_dataframe(df, label_col=None)
        assert "label" not in result.columns
        assert "text_clean" in result.columns
