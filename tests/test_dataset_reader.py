import pytest
from typing import List

from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.common.util import get_spacy_model

from my_project.dataset_reader import ClassificationTsvReader


class TestTextClassificationJsonReader:
    def test_read_from_file_ag_news_corpus_and_truncates_properly(self):
        reader = ClassificationTsvReader()
        data_path = "data/dev.tsv"
        instances = reader.read(data_path)

        assert len(instances) == 2313

        fields = instances[0].fields
        expected_tokens = ["Find", "me", "information", "about", "the"]
        assert [t.text for t in fields["initial_request"].tokens][:5] == expected_tokens
        assert fields["label"].label == 2
