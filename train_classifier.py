import sys, json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
texts,labels=[],[]
with open(sys.argv[1],'r',encoding='utf-8') as f:
    for line in f:
        if line.strip():
            o=json.loads(line); texts.append(o['text']); labels.append(o['label'])
Xtr,Xte,ytr,yte=train_test_split(texts,labels,test_size=0.2,random_state=42,stratify=labels)
vec=TfidfVectorizer(ngram_range=(1,2)); Xtrv=vec.fit_transform(Xtr); Xtev=vec.transform(Xte)
clf=LogisticRegression(max_iter=1000).fit(Xtrv,ytr)
print(classification_report(yte, clf.predict(Xtev)))
joblib.dump(vec,'email_vectorizer.joblib'); joblib.dump(clf,'email_intent_model.joblib')
print('Saved vectorizer & model')
