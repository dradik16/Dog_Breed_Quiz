from io import TextIOWrapper
import os
import numpy as np
import random
import pandas as pd
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications import  mobilenet_v2

df=pd.read_csv('dog_database_shorter.csv',index_col=0)

app = Flask(__name__)

@app.route('/')
def landing_page():
    '''
    this page takes in user info via a html form

    '''
    return render_template('landing_page.html')


@app.route('/quiz')
def quiz():
    
    imagelist=os.listdir('./static/webimages_dogs/')
    image=random.sample(imagelist,k=1)[0]
    random_image=f'static/webimages_dogs/{image}'
    return render_template('quiz.html',random_image=random_image)

@app.route('/answer')
def answer():
    '''
    this page give the user info about the movie
    '''
    print(request.args)
    random_image=request.args['random_image']
    img = tf.keras.preprocessing.image.load_img(random_image, target_size=(224, 224))
    img = np.array(img)
    img = img.reshape(1,224,224,3)
    img =mobilenet_v2.preprocess_input(img)
    model = mobilenet_v2.MobileNetV2()
    y_pred=model.predict(img)
    mobilenet_v2.decode_predictions(y_pred, top=1)
    nas=np.array(mobilenet_v2.decode_predictions(y_pred, top=1))
    animal_name=nas[0,0,1]

    dog_name = pd.Series(animal_name).str.lower()
   
    if dog_name.isin(df.breed).item() == False:
        return render_template('answer_error.html')
    else:
        doggie=dog_name[0].replace('_', ' ')
        suggestion=df[df['breed']==dog_name[0]]
        dog_size=suggestion['size'].item() 
        dog_intelligence=suggestion['intelligence'].item() 
        dog_adaptability=suggestion['adaptability'].item() 
        dog_friendliness=suggestion['friendliness'].item() 
        infa=suggestion['url'].item()
        return render_template('answer.html', random_image=random_image,doggie=doggie,dog_size=dog_size,dog_intelligence=dog_intelligence,dog_adaptability=dog_adaptability,dog_friendliness=dog_friendliness,infa=infa)

if __name__ == "__main__":
    app.run(debug=True, port=5000)