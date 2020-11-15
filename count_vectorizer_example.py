

from sklearn.feature_extraction.text import CountVectorizer

text = [
    'This is the first phrase is',
    'This is the second one',
    'And this is the last',
]
test_text = ['My name is Robert']

count_vect.fit(text)  # learn the vocabulary
print(count_vect.vocabulary_)

X = count_vect.transform(text)  # sparse matrix

X = X.toarray()  # numpy matrix
print(X)

print(count_vect.get_feature_names())

