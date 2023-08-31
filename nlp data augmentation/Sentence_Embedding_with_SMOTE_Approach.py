import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
data = pd.read_csv('data_path.csv',)
data = data[['output_class','Input_text_description']]
## as smote works on KNN it atleast requires 6 data points to work with
data =data.groupby('output_class').filter(lambda x : len(x)>6) 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(result_df['output_class'].values)
vectorfile='glove.6B.200d.txt'
glove_embeddings = {}
with open(vectorfile,'r',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.array(values[1:], dtype='float32')
        glove_embeddings[word] = embedding
def preprocess_to_vectors(df):
    X = df['Input_text_description'].values
    Y = df['output_class'].values
    Y = le.transform(Y)
    #document level embeddings
    document_embeddings = np.zeros(shape=(len(X),200))
    vocab = set()
    index = 0
    for document in X:
        # print(document)
        # print('-'*12)
        embeddings = []
        for word in document.split():
            embedding = glove_embeddings.get(word)
            if embedding is not None:
                embeddings.append(embedding)
                vocab.add(word)
        if embeddings:
            document_embeddings[index] = (np.mean(embeddings, axis=0)) # document embedding
        else:
            document_embeddings[index] = np.array(0)
        index += 1
            # document_embeddings.append(np.array(embeddings)) # word embedding
    return document_embeddings,Y

document_embeddings_train,Y = preprocess_to_vectors(data)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(document_embeddings,np.array(Y))