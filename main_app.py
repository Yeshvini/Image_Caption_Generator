from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from flask import Flask,render_template,url_for,request
'''
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
'''
def extract_features(filename, model):
        try:
            print("Image loadingg.....")
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

app =  Flask(__name__)

APP_ROOT =  os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        #pickle_path =os.getcwd()
        #image_path = pickle_path+ filename

        image_file = open('pk_image','wb')
        pk_image=pickle.dump(destination,image_file)
        image_file.close()

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("upload.html", image_name=filename)

'''
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
'''
@app.route('/predict',methods=['POST'])
def predict():
    print("predict--",os.getcwd())
    
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    filename = os.getcwd() + '\pk_image'
    print("pk file",filename)
    infile =open(filename,'rb')
    img_path = pickle.load(infile)
    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)

    description = generate_desc(model, tokenizer, photo, max_length)
    #print("\n\n")
    print(description)
    description=description.replace("start"," ").replace("end", " ")
   # plt.imshow(img)

    return render_template('upload.html',prediction=description)


if __name__ == "__main__":
    app.run(host="localhost",port=5001,debug=True)



