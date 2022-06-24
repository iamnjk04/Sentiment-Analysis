from django.shortcuts import render
from .forms import textForm
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from .preprocess import preprocess
loaded_model = pickle.load(open('model/79_57_logreg_final.pkl', 'rb'))
tf           = pickle.load(open('model/tf_notebook.pkl', 'rb'))

# Create your views here.

def my_model(request):
    data = None
    pred = None
    if request.method == "POST":
        formdata = textForm(request.POST)
        if formdata.is_valid():
            text = formdata.cleaned_data['text']
            data = text
            df = pd.DataFrame(list([text,]), columns=['text'])
            df['text'] = preprocess(df)
            df['text'] = df['text'].apply(lambda x: ' '.join(x))
            X = tf.transform(df['text'])
            pr = loaded_model.predict(X)
            if (len(df['text'][0]) == 0):
                pred = "Invalid text"
            elif pr.astype(int) == 0:
                pred = 'Negative'
            elif pr.astype(int) == 4:
                pred = "Positive"
            else: 
                pred = "Undeterminable"
    else:
        formdata = textForm()
    
    return render(request, 'model/index.html' , {'form': formdata, 'data': data , 'pred' : pred})
    
