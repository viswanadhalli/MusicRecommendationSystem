from flask import Flask, render_template,request
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import spot as sp
app=Flask(__name__)
@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path="./images/"+imagefile.filename
    i=imagefile.filename
    imagefile.save(image_path)
    model =tf.keras.models.load_model('model100.h5')
    img = image.load_img('images\\'+str(i), grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x)
    final_Emotion,k=emotion_analysis(custom[0])
    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);
    final_list = sp.songs_by_emotion(final_Emotion)
    flist=str(final_list)
    flist=flist.split(",")
    
    return render_template('index.html',acc=k,femotion=final_Emotion,songs=flist)
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    res=(max(emotions))
    j=0
    for i in emotions:
        if(i==res):
            break
        else:
            j=j+1
    Emotion=str(objects[j])
    print('Emotion Detected : ' + Emotion)
    print('Accuracy : '+ str(res*100))
    k=str(res*100)
    plt.show()
    return (Emotion,k)

def songs_by_emotion(emotion):
    results = sp.search(q=emotion,type='playlist', limit=playlist_limit)
    gs = []
    for el in results['playlists']['items']:
        temp = {}
        temp['playlist_name'] = el['name']
        temp['playlist_href'] = el['href']
        temp['playlist_id'] = el['id']
        temp['playlist_spotify_link'] = el['external_urls']['spotify']
        gs.append(temp)
    fnl_playlist_songs = gs
    for i in range(0,len(gs)):
        res = sp.playlist(playlist_id = gs[i]['playlist_id'])
        srn = res['tracks']['items'][0:song_limit_per_playlist]
        tlist = []
        for el in srn:
            tlist.append(el['track']['name'])
        fnl_playlist_songs[i]['playlist_songs'] = tlist
    return fnl_playlist_songs
def print_songs(fnl_playlist_songs):
    for el in fnl_playlist_songs:
        print('playlist_name : ' + str(el['playlist_name']))
        print('playlist_href : ' + str(el['playlist_href']))
        print('playlist_spotify_link : ' + str(el['playlist_spotify_link']))
        print('playlist_songs : ' )
        for i in range(0,len(el['playlist_songs'])):
            print(str(i+1) + ') ' + el['playlist_songs'][i])
        print('-----------------------------------------------')  
def facecrop(image): 
    image = image[23:]
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img_b64decode = base64.b64decode(image)
    img_array = np.fromstring(img_b64decode,np.uint8) 
    img=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB) 
    try: 
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)
        faces = cascade.detectMultiScale(miniframe)
        print(faces)
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            sub_face = img[y:y+h, x:x+w]
            cv2.imwrite('capture.jpg', sub_face)
    except Exception as e:
        print (e)


if __name__=="__main__":
    app.run(debug=True)



