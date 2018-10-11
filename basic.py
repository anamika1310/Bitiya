#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 21:11:58 2018

@author: anamika
"""

import spacy

nlp=spacy.load("en")

doc = nlp(u'U.K. is wonderful horse.') ##tokenizing
for t in doc:
    print(t.text,t.lemma_,t.pos_,t.shape_,t.tag_,t.is_alpha,t.is_stop,t.dep_)
    
""" Name entity that is recognizing various types of named entities in a document like person,location and so on"""
for ent in doc.ents: ## doc.ents recognizes the entiries present in doc it will be able to recognize only some pretrained entities like INDIA ,USA
    print(ent.text,ent.start_char, ent.end_char,ent.label_)    

""" Word vector and similarity"""

tokens=nlp(u'dog cat banana')
for tok1 in tokens:
    print(len(tok1.vector))
    for tok2 in tokens:
        print(tok1.text,tok2.text,tok1.similarity(tok2))
        
""" Vocab,lexemes and hashes """

doc = nlp(u'I love coffee')
print(doc.vocab.strings['coffee']) 
for word in doc:
    lexeme=doc.vocab[word.text]
    print(lexeme.text,lexeme.orth,lexeme.shape_,lexeme.prefix_,lexeme.is_title,lexeme.lang_,lexeme.is_digit)      

from spacy.tokens import Doc
from spacy.vocab import Vocab
from pathlib import Path
emp=Doc(Vocab())
#print(emp.vocab.strings[3197928453018144401])     will raise an error

print(emp.vocab.strings.add('coffee'))

new_doc = Doc(doc.vocab)  # create new doc with first doc's vocab
print(new_doc.vocab.strings[3197928453018144401])  # 'coffee' 

""" Training own model and adding new entites"""
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
from pathlib import Path

train_data = [
    ('Who is Chaka Khan?', [(7, 17, 'PERSON')]),
    ('I like London and Berlin.', [(7, 13, 'LOC'), (18, 24, 'LOC')])
]
nlp=spacy.load('en',entity=False ,parser=False)
ner=EntityRecognizer(nlp.vocab,entity_types=['PERSON','LOC'])

for _ in range(100):
    random.shuffle(train_data)
    for raw_text, entity_offset in train_data:
        doc=nlp.make_doc(raw_text)
        #print(entity_offset,raw_text)
        gold=GoldParse(doc,entities=entity_offset)
        nlp.tagger(doc)
        nlp.update([doc],[gold])
        
for text, _ in train_data:
    doc = nlp('i am anamika chaka')
    print(doc.ents)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
if(not Path(u'Spacy/model').exists()):
    nlp.to_disk(u'Spacy/model')    

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# training data
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)
        