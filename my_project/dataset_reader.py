from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

import pandas as pd


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, line: dict) -> Instance:

        topic_id = line.get("topic_id", "")
        facet_id = line.get("facet_id", "")
        initial_request = self.tokenizer.tokenize(
            line.get("initial_request", ""))
        topic_desc = self.tokenizer.tokenize(line.get("topic_desc", ""))
        clarification_need = int(line.get("clarification_need", None))
        facet_desc = self.tokenizer.tokenize(line.get("facet_desc", ""))
        question = self.tokenizer.tokenize(str(line.get("question", "")))
        answer = self.tokenizer.tokenize(str(line.get("answer", "")))

        if self.max_tokens:
            initial_request = initial_request[:self.max_tokens]
            topic_desc = topic_desc[:self.max_tokens]
            facet_desc = facet_desc[:self.max_tokens]
            question = question[:self.max_tokens]
            answer = answer[:self.max_tokens]

        initial_request = TextField(initial_request, self.token_indexers)
        topic_desc = TextField(topic_desc, self.token_indexers)
        facet_desc = TextField(facet_desc, self.token_indexers)
        question = TextField(question, self.token_indexers)
        answer = TextField(answer, self.token_indexers)

        fields = {'topic_id': topic_id, 'facet_id': facet_id,
                  'initial_request': initial_request,
                  'topic_desc': topic_desc,
                  'facet_desc': facet_desc,
                  'question': question,
                  'answer': answer, }
        if clarification_need:
            fields['label'] = LabelField(clarification_need,"clarification_need",skip_indexing=True)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            df = pd.read_csv(f, "\t")
            for index, row in df.iterrows():
                yield self.text_to_instance(row.to_dict())


"""
topic_id	initial_request	topic_desc	clarification_need	facet_id	facet_desc	question_id	question	answer
101	Find me information about the Ritz Carlton Lake Las Vegas.	Find information about the Ritz Carlton resort at Lake Las Vegas.	2	F0010	Find information about the Ritz Carlton resort at Lake Las Vegas.	Q00697	are you looking for a specific web site	yes for the ritz carlton resort at lake las vegas

"""
