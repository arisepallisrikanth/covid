from flask import Flask,render_template, request
from tensorflow.keras.models import Sequential, Model, load_model
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale, StandardScaler






app=Flask(__name__)
@app.route('/',methods=['GET'])
def welcome():
    return render_template('index2.html')
@app.route('/pred',methods=['POST'])
def predict():
    audiofile=request.files['audio']
    aud_path="./audios/"+audiofile.filename
    audiofile.save(aud_path)
    pr=load_model("Srikanth_corona.h5")
    df=pd.read_csv(r"obtained_values.csv")
    df.drop(["filename"], axis=1, inplace=True)
    map_dict = {"covid":1, "not_covid":0}
    df['label'] = df['label'].map(map_dict)
    shuffle_train_df = df.reindex(np.random.permutation(df.index))
    y = shuffle_train_df['label'].to_numpy()
    X = (shuffle_train_df.iloc[:, :-1]).to_numpy()
    from imblearn.over_sampling import SMOTE
    smote=SMOTE(sampling_strategy='minority')
    x_sm,y_sm=smote.fit_resample(X,y)
    stsc = StandardScaler().fit(x_sm)




    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 41):
        header += f' mfcc{i}'
    header = header.split()
    
    def feature_extract(file_name):
        y,sr = librosa.load(file_name, mono=True, duration=5)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        return to_append
    file=aud_path
    gita=feature_extract(file).split()
    gita
    df=pd.DataFrame([gita],columns=header)
    df
    X1 = df.to_numpy()
    X1.reshape(1,-1)
    X1=stsc.transform(X1)
    r=np.argmax(pr.predict(X1),axis=1)[0]
    if r==1:
        result="corona positive"
    else:
        result="corona negative"
        

    return render_template('index2.html',prediction=result) 
if __name__=='__main__':
    app.run(port=1000,debug=True)
