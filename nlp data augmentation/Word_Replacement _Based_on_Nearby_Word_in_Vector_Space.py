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

# downloading word2vec embeddings
# DownloadUtil.download_word2vec(dest_dir='.')
# downloading fasttext embeddings
# DownloadUtil.download_fasttext(dest_dir='.',model_name='crawl-300d-2M')
# downloading glove embeddings
DownloadUtil.download_glove(dest_dir='.',model_name='glove.6B')

text = """involves using many prompt-completion examples as the labeled 
            training dataset to continue training the model by updating its weights.
            This is different from"""

# initialising the augmentor with "Glove"
aug = naw.WordEmbsAug(
        # option to choose from is "word2vec", "glove" or "fasttext"
        model_type='glove',
        # chossing the specific model from the list in path downloaded earlier
        model_path = 'glove.6B.300d.txt',
        # options available are insert or substitute
        action='substitute')
# augmented text

print("Original : ")
print(text)
augmented_text = aug.augment(text,n=30)
for sentence in augmented_text:
    print(sentence)