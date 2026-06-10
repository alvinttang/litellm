from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base


def test_get_api_base_gemini_stream_endpoint():
    """
    When stream=True is passed via the optional_params dict, the returned
    gemini api_base must use the :streamGenerateContent endpoint.
    """
    api_base = get_api_base(
        model="gemini/gemini-1.5-pro", optional_params={"stream": True}
    )
    assert api_base is not None
    assert api_base.endswith(":streamGenerateContent")


def test_get_api_base_gemini_non_stream_endpoint():
    api_base = get_api_base(
        model="gemini/gemini-1.5-pro", optional_params={"stream": False}
    )
    assert api_base is not None
    assert api_base.endswith(":generateContent")
