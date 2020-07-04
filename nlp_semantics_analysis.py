import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
fil = []
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words = set(stop_words)
ps = PorterStemmer()
for i in range(1000):
	review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # replace all the punctution with space
	review = review.lower()
	review = review.split()
	review = [ps.stem(i) for i in review if not i in stop_words]
	review = ' '.join(review)
	fil.append(review)
print(fil)

cv = CountVectorizer(max_features=1000)  # here while creating the object of the countvectorizer class we need max_feature after knowing the number of features columns
x = cv.fit_transform(fil).toarray()
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)
classifier = LinearSVC()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))
#####################################

review = re.sub('[^a-zA-Z]', ' ',input())  # replace all the punctution with space
review = review.lower()
review = review.split()
review = [ps.stem(i) for i in review if not i in stop_words]
review = ' '.join(review)
r=[]
r.append(review)
w=cv.transform(r).toarray()
print(classifier.predict(w))


# 82.5%
'''penalty: Any = 'l2',
             loss: Any = 'squared_hinge',
             *,
             dual: Any = True,
             tol: Any = 1e-4,
             C: Any = 1.0,
             multi_class: Any = 'ovr',
             fit_intercept: Any = True,
             intercept_scaling: Any = 1,
             class_weight: Any = None,
             verbose: Any = 0,
             random_state: Any = None,
             max_iter: Any = 1000)'''