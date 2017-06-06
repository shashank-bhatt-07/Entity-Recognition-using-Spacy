from __future__ import unicode_literals, print_function

from data import getData
from words_to_feature import sent2features,sent2labels

import ujson as json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger
from spacy.vocab import Vocab



nlp = spacy.load('en')

def train_ner(nlp, train_data, entity_types):
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]
    ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            ner.update(doc, gold)
    ner.model.end_training()
    return ner


def main(model_dir=None):
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
        if not model_dir.exists():
            model_dir.mkdir()
        assert model_dir.is_dir()

    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)

    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    train_data =getData()
    ner = train_ner(nlp, train_data, ['ACC','EMAIL'])

    doc = nlp.make_doc('update email of account pepsico to ab@gmail.com')
    #nlp.tagger(doc)
    ner(doc)
    for word in doc:
        print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)

    if model_dir is not None:
        with (model_dir / 'config.json').open('wb') as file_:
            json.dump(ner.cfg, file_)
        ner.model.dump(str(model_dir / 'model'))
        if not (model_dir / 'vocab').exists():
            (model_dir / 'vocab').mkdir()
        ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
        with (model_dir / 'vocab' / 'strings.json').open('w',encoding='utf8') as file_:
            ner.vocab.strings.dump(file_)
        
def predict(query):
    # Load NER
    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)
    vocab_dir = pathlib.Path('ner/vocab')
    with (vocab_dir / 'strings.json').open('r',encoding='utf8') as file_:
        nlp.vocab.strings.load(file_)
    nlp.vocab.load_lexemes(vocab_dir / 'lexemes.bin')


    ner = EntityRecognizer.load(pathlib.Path("ner"), nlp.vocab, require=False)
    doc = nlp.make_doc(query)
    #nlp.tagger(doc)
    ner(doc)
    for word in doc:
        print(word.text, word.orth, word.lower, word.ent_type_)

    for word in doc:
        if word.ent_type_:
            print ('word -> {} and entity-> {}'.format(word.text,word.ent_type_))
        #print(word.text, word.orth, word.lower, word.ent_type_)

if __name__ == '__main__':
    main('ner')
    predict('update email of account hal to ab@gmail.com')