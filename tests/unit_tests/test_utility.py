"""Tests for the utility functions."""
import pytest
from utility import (
    combine_text_from_list,
    create_tmp_folder,
    generate_prompt,
    get_token,
    load_vec_db_conn,
)


def test_combine_text_from_list() -> None:
    """Test the combine_text_from_list function."""
    assert combine_text_from_list(["Hello", "World"]) == "Hello\nWorld"
    with pytest.raises(TypeError):
        combine_text_from_list(["Hello", 123])


def test_generate_prompt() -> None:
    """Test the generate_prompt function."""
    assert generate_prompt("qa.j2", "This is a test text.", "What is the meaning of life?", "en")
    with pytest.raises(FileNotFoundError):
        generate_prompt("non_existent.j2", "This is a test text.", "What is the meaning of life?", "en")


def test_create_tmp_folder() -> None:
    """Test the create_tmp_folder function."""
    assert "/tmp_" in create_tmp_folder()


def test_get_token() -> None:
    """Test the get_token function."""
    assert get_token("token", None) == "token"
    assert get_token(None, "aleph_alpha_key") == "aleph_alpha_key"
    with pytest.raises(ValueError):
        get_token(None, None)


def test_load_vec_db_conn() -> None:
    """Test the load_vec_db_conn function."""
    assert load_vec_db_conn()
