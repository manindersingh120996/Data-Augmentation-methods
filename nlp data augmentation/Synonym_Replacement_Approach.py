---

Enhancing NLP Models with Data Augmentation Techniques

Photo by Aleks Dorohovich on Unsplash

In the world of Natural Language Processing (NLP), data is the lifeblood that powers machine learning models. The quality and quantity of data often determine the success of these models. However, gathering and labelling large datasets can be a challenging and resource-intensive task. This is where data augmentation techniques come to the rescue. Data augmentation involves creating variations of existing data to expand the dataset, leading to more robust and accurate models. 
In this article, I will explain a few techniques that I have used over the course of my Projects to Augment Textual data or balance the classes.
Following is the list of methods which I will be explaining:
Synonym Replacement Approach
Sentence Embedding with SMOTE Approach
Word Replacement Based on Nearby Word in Vector Space(using nlpaug library)
Word Replacement Based on Contextual Word Embeddings

---

Synonym Replacement Approach
The Synonym Replacement Approach is a simple yet effective technique for data augmentation in Natural Language Processing (NLP). It involves replacing words in a sentence with their synonyms to create variations of the original text. By doing so, we generate new instances of data but sometimes synonyms could vary in meaning (like a synonym of a sentence is also conviction, which is very far away from its actual meaning in a sentence) so sentence form could be a little different compared to the original but it helps in introducing Diversification in a dataset.
Synonym from Power ThesaurusSynonym Replacement Process:
Word Identification: In this approach, we start by identifying individual words in the input text that can be replaced with synonyms. This can be done by tokenizing the text into words using techniques like word tokenization.
Synonym Retrieval: Once we have identified the words to be replaced, we use lexical resources like WordNet to retrieve synonyms for each word. WordNet is a large lexical database that groups words into sets of synonyms called synsets.
Synonym Selection: From the retrieved synonyms, we select a synonym to replace the original word. 
Replacing Words: After selecting a synonym, we replace the original word in the text with the chosen synonym. This results in an augmented version of the input text, where specific words have been altered while maintaining the sentence's overall structure and meaning.

It's important to note that while some synonyms may be perfect replacements, others might introduce subtle changes in meaning. The choice of synonyms depends on the desired level of variation.
Benefits of Synonym Replacement:
Diversity: Synonym replacement introduces diversity into the dataset by generating alternative versions of sentences. This diversity can help the model generalize better to different variations of the same context.
Data Expansion: With each word replaced by its synonym, we generate new instances of data that are similar in context to the original text. This expanded dataset can lead to improved model performance.
Context Preservation: Synonym replacement retains the context of the original sentence. It ensures that the sentence's overall meaning remains intact, even though specific words have changed.

Implementation with NLTK and WordNet:
The Natural Language Toolkit (NLTK) library in Python provides tools for text processing, including access to WordNet.
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import random
nltk.download('punkt')
nltk.download('wordnet')

def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]

def get_synonyms(word, pos_tag):
    synonyms = []
    for syn in wordnet.synsets(word):#, pos=wordnet._wordnet_postag_map.get(pos_tag)):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def replace_with_synonym(word, synonyms):
    if synonyms:
#         return synonyms[0]  # Replace with the first synonym
        return random.choice(synonyms)
    return word  # If no synonyms found, retain the original word

def augment_with_synonyms(text):
    sentences = sent_tokenize(text)
#     print(sentences)
    augmented_sentences = []
    for sentence in sentences:
        tokenized = word_tokenize(sentence)
#         print(tokenized)
        augmented_tokens = []
        for token in tokenized:
            pos_tag = get_pos_tag(token)
#             print(token,pos_tag)
            synonyms = get_synonyms(token, pos_tag)
#             print(token,pos_tag,synonyms)
            augmented_tokens.append(replace_with_synonym(token, synonyms))
        augmented_sentence = ' '.join(augmented_tokens)
        augmented_sentences.append(augmented_sentence)
    augmented_text = ' '.join(augmented_sentences)
    return augmented_text
# Example text for augmentation
original_text = "Create easily interpretable topics with Large Language\\
Models - With the advent of Llama 2, running strong LLMs locally\\
has become more and more a reality. "

# Augment the text with synonyms

print("Original Text:", original_text)
count_of_generated_examples = 5
for gen in range(count_of_generated_examples):
    augmented_text = augment_with_synonyms(original_text)
    print(f"Augmented Text {gen+1}:", augmented_text,'\n')