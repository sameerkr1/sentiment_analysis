from flask import Flask, render_template,request
import numpy as np
import pickle
import fastai
from fastai.tabular import *
import os
import tensorflow 
import torch.tensor
from fastai.text import*
import keras 
import tensorflow 
from tensorflow import keras
import h5py
from product import products

app = Flask(__name__)

cwd=os.getcwd()
path=cwd+'\\files'
# print(path)
model=load_learner(path,'bobmar.pkl')


@app.route('/')
def home():
    return render_template('main_page.html', products=products)

@app.route('/product/<pid>',methods=['GET','POST'])
def predict(pid):

    review = 0

    if request.method == "POST":
        data=request.form['inputA']         

        review=model.predict(data)
        review=review[2][1]
        review=review.item()
        review*=5
        review=round(review,2)
            
        length = len(products[pid]['review'])
        products[pid]['avg_review'] = ( (products[pid]['avg_review']*length ) + review ) / (length+1)

        # rate the review
        if review >= 3:
            products[pid]['review'].append({
                'description': data,
                'sentiment': 'positive',
                'rating': review,
            })
        else:
            products[pid]['review'].append({
                'description': data,
                'sentiment': 'negative',
                'rating': review,
            })
    
    return render_template('products.html', pid=pid, product=products[pid], review=review, int=int)  

@app.route("/rating")
def rating():
    return render_template('rating.html')

if __name__ == '__main__':
    app.run(port=2999,debug=True)