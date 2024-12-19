from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.swissaiplatform import SwissAIPlatform


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SwissAIPlatform.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
