import gensim
import transformers
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
# downloading models to current directory
from nlpaug.util.file.download import DownloadUtil
# options to choose from is BERT, ROBERT,Distill-BERT, XL-NET
# initialising the augmentor with "Glove"
caug = naw.ContextualWordEmbsAug(
        # option to choose from is "word2vec", "glove" or "fasttext"
        model_path='distilbert-base-uncased',

        # options available are insert or substitute
        action='substitute')
# augmented text

text = """involves using many prompt-completion examples as the labeled 
            training dataset to continue training the model by updating its weights.
            This is different from"""
augmented_text = caug.augment(text,n=30)
for sentence in augmented_text:
    print(sentence)