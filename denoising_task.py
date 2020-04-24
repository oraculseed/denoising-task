import sys
sys.path.append('lib')
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Input,LeakyReLU,Conv2D, MaxPooling2D, GlobalAveragePooling2D  

from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.models import Model,Sequential 
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
import argparse
from sklearn.preprocessing import LabelEncoder


    
def train_classification(args):
    
    root_dataset = args.dataset
    k_train = args.k_train #999
    batch_size = args.batch_size #768
    epochs = args.epochs #100
    
    y_train = []
    y_val = []
    X_train = []
    X_val = []
    
    
    for top, dirs, files in tqdm(os.walk(root_dataset + 'train/noisy')):
        for nm in files[:k_train]:       
            file_name = os.path.join(top, nm)
            data_n = np.load(file_name)
            data_c = np.load(file_name.replace('train/noisy','train\\clean'))  
    
            if data_n.shape[1] == 80 and data_n.shape[0] > 200 \
            and data_c.shape[1] == 80 and data_c.shape[0] > 200 \
            and data_n.shape[0] == data_c.shape[0] : #filter
                
                X_train.append(data_n[:200,:])              
                y_train.append(1)   
                X_train.append(data_c[:200,:])
                y_train.append(0)                  
                
    for top, dirs, files in tqdm(os.walk(root_dataset + 'val/noisy')):
        for nm in files[:k_train]:       
            file_name = os.path.join(top, nm)
            data_n = np.load(file_name)
            data_c = np.load(file_name.replace('val/noisy','val\\clean'))  
    
            if data_n.shape[1] == 80 and data_n.shape[0] > 200 \
            and data_c.shape[1] == 80 and data_c.shape[0] > 200 \
            and data_n.shape[0] == data_c.shape[0] : #filter
                
                X_train.append(data_n[:200,:])              
                y_train.append(1)   
                X_train.append(data_c[:200,:])
                y_train.append(0)                              
    
    
    LEncoder = LabelEncoder()
    y_train = to_categorical(LEncoder.fit_transform(y_train)) 

    
    X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), y_train, \
                                        test_size=0.14, random_state=87)
        
        
    K_test = np.int(np.ceil(len(X_train) / 100))
        
    #train/test
    X_test = X_train[:K_test] 
    y_test = y_train[:K_test] 
    X_train = X_train[K_test:] 
    y_train = y_train[K_test:] 
        
    num_rows = 200
    num_columns = 80
    num_channels = 1


    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

    
    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, \
                            num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    plot_model(model, to_file='model_classification.png', show_shapes=True, \
               show_layer_names=True)    
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath='model/train_classification_chk.hdf5', 
                                   verbose=1, save_best_only=True)
    print("Fit")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, \
              validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

    #save model
    model_json = model.to_json()
    with open("model/model_classification.json", 'w') as f:
        f.write(model_json)
    model.save_weights("model/model_classification.h5")
    print("Done")

def train(args):
    
    root_dataset = args.dataset
    
    k_train = args.k_train #999
    augmentation = args.augmentation #True
    batch_size = args.batch_size #768
    epochs = args.epochs #100
    
    y_train = np.empty((0,80), dtype=float)
    y_val = np.empty((0,80), dtype=float)
    X_train = np.empty((0,80), dtype=float)
    X_val = np.empty((0,80), dtype=float)
    
    
    
    for top, dirs, files in tqdm(os.walk(root_dataset + 'train/noisy'), \
                                 desc='train',total=799):
        for nm in files[:k_train]:       
            file_name = os.path.join(top, nm)
            data_n = np.load(file_name)
            data_c = np.load(file_name.replace('train/noisy','train\\clean'))  
    
            if data_n.shape[1] == 80 and data_n.shape[0] > 20 \
            and data_c.shape[1] == 80 and data_c.shape[0] > 20 \
            and data_n.shape[0] == data_c.shape[0]: #filter
            
                X_train = np.append(X_train,data_n,axis=0)    
                if augmentation:
                    X_train = np.append(X_train,np.flip(data_n,1),axis=0) #aug
               
                y_train = np.append(y_train, data_c,axis=0)  
                
                if augmentation:
                    y_train = np.append(y_train,np.flip(data_c,1),axis=0) #aug
                
    for top, dirs, files in tqdm(os.walk(root_dataset + 'val/noisy'), \
                                 desc='val', total=150):
        for nm in files[:k_train]:       
            file_name = os.path.join(top, nm)       
            data_n = np.load(file_name)
            data_c = np.load(file_name.replace('val/noisy','val\\clean'))  
            
            if data_n.shape[1] == 80 and data_n.shape[0] > 20 \
            and data_c.shape[1] == 80 and data_c.shape[0] > 20 \
            and data_n.shape[0] == data_c.shape[0]: #filter
            
                #X_val = np.append(X_val,data_n,axis=0)
                #X_val = np.append(X_val,np.flip(data_n,1),axis=0) #aug
                
                #y_val = np.append(y_val, data_c,axis=0)    
                #y_val = np.append(y_val,np.flip(data_c,1),axis=0) #aug
    
                #for shuffle
                X_train = np.append(X_train,data_n,axis=0) 
                if augmentation:
                    X_train = np.append(X_train,np.flip(data_n,1),axis=0) #aug
               
                y_train = np.append(y_train, data_c,axis=0)
                
                if augmentation:            
                    y_train = np.append(y_train,np.flip(data_c,1),axis=0) #aug
    
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
                                        test_size=0.14, random_state=87)
    
    
    K_test = np.int(np.ceil(X_train.shape[0] / 200))
    
    #train/test
    X_test = X_train[:K_test] 
    y_test = y_train[:K_test] 
    X_train = X_train[K_test:] 
    y_train = y_train[K_test:] 
    
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    num_hidden = [2049,500,180]
    
    ILayer1 = Input(shape=(input_dim,), name="ILayer")
    ILayer2 = BatchNormalization(axis=1, momentum=0.6)(ILayer1)
    
    #H1
    HLayer1_1 = Dense(num_hidden[0], activation='relu', \
                      name="HLayer1", kernel_initializer=he_normal(seed=20))(ILayer2)
    HLayer1_2 = BatchNormalization(axis=1, momentum=0.55)(HLayer1_1)
    HLayer1_3 = Dropout(0.1)(HLayer1_2)
    
    #H2
    HLayer2_1 = Dense(num_hidden[1], activation='relu', \
                      name="HLayer2", kernel_initializer=he_normal(seed=60))(HLayer1_3)
    HLayer2_2 = BatchNormalization(axis=1, momentum=0.55)(HLayer2_1)
    
    #H3
    HLayer3_1 = Dense(num_hidden[2], activation='relu', \
                      name="HLayer3", kernel_initializer=he_normal(seed=120))(HLayer2_2)
        
    HLayer3_2 = BatchNormalization(axis=1, momentum=0.55)(HLayer3_1)
    HLayer3_2 = Dropout(0.1)(HLayer3_2)
    #H2_R
    HLayer2__1 = Dense(num_hidden[1], activation='relu', \
                       name="HLayer2_R", kernel_initializer=he_normal(seed=60))(HLayer3_2)
    HLayer2__2 = BatchNormalization(axis=1, momentum=0.55)(HLayer2__1)
    
    #H1_R
    HLayer1__1 = Dense(num_hidden[0], activation='relu', \
                       name="HLayer1_R", kernel_initializer=he_normal(seed=20))(HLayer2__2)
    HLayer1__2 = BatchNormalization(axis=1, momentum=0.55)(HLayer1__1)
    HLayer1__3 = Dropout(0.1)(HLayer1__2)
    
    OLayer = Dense(output_dim,  \
                        name="OLayer",kernel_initializer=he_normal(seed=62))(HLayer1__3)
    
    model = Model(inputs=[ILayer1], outputs=[OLayer])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, \
                          decay=0.0001, amsgrad=False)
    #compile
    model.compile(loss='mse', optimizer=opt)
    
    plot_model(model, to_file='model.png', show_shapes=True, \
               show_layer_names=True)
    model.summary()
    
    tensorboard = TensorBoard(log_dir="logs", histogram_freq=0, \
                              write_graph=True, write_images=True)
    print("Fit")
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, \
                     verbose=1, validation_data=([X_val], [y_val]),
                     callbacks=[tensorboard])
    
    plt.figure(figsize=(10, 8))
    plt.plot(hist.history['loss'], label='Loss')
    plt.plot(hist.history['val_loss'], label='Val_Loss')
    plt.legend(loc='best')
    plt.title('Train/Val Loss')
    plt.savefig('train_val_Loss.png')
    #plt.show()
    
    results = model.evaluate(X_test, y_test, batch_size=len(y_test))
    print('Test loss:%3f' % results)
    
    #save model
    model_json = model.to_json()
    with open("model/model.json", 'w') as f:
        f.write(model_json)
    model.save_weights("model/model.h5")
    print("Done")

def predict(args):
    output_dir = args.output   
    input_dir = args.input

    csv_file = output_dir + "results.csv"


    # load model
    print("Load model")
    with open('model/model.json','r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model/model.h5")    

    print("Load model classification")
    with open('model/model_classification.json','r') as f:
        loaded_model_classification_json = f.read()
    model_classification = model_from_json(loaded_model_classification_json)
    model_classification.load_weights("model/model_classification.h5")  

    num_rows = 200
    num_columns = 80
    num_channels = 1

    
    print("Predict")    
    with open(csv_file, 'a') as f:
        
        f.write("file_name;result;denoised_file" + "\r")
        
        for top, dirs, files in os.walk(input_dir):
            for nm in tqdm(files):       
                file_name = os.path.join(top, nm)
                
                data = np.load(file_name)      
                
                
                data_clf = data[:200,:]
                data_clf = data_clf.reshape(-1, num_rows, num_columns, num_channels)
                
                data_predict_classification = model_classification.predict(data_clf)

                data_predict = model.predict(data) 
                
                if data_predict_classification[0][0] > 0.501:
                    label = "clean"
                    output_file = ""
                else:
                    label = "noisy"                    
                    np.save(output_dir + "denoised/" + nm,data_predict)
                    output_file = output_dir + "denoised/" + nm
                     
                f.write(file_name + ";" + label + ";" + output_file + "\r")
                
                
    print("Done")
    #file_name - имя текущего файла для обработки
    #result - результат обработки (noisy или clean)
    #denoised_file - путь к файлу после удаления шума, если файл распознан как



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset', type=str, required=True)
    parser_train.add_argument('--k_train', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--epochs', type=int, required=True)
    parser_train.add_argument('--augmentation', type=int, required=True)
    

    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--input', type=str, required=True)
    parser_predict.add_argument('--output', type=str, required=True)


    parser_train_classification = subparsers.add_parser('train_classification')
    parser_train_classification.add_argument('--dataset', type=str, required=True)
    parser_train_classification.add_argument('--k_train', type=int, required=True)
    parser_train_classification.add_argument('--batch_size', type=int, required=True)
    parser_train_classification.add_argument('--epochs', type=int, required=True)
    
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'train_classification':
        train_classification(args)        
    elif args.mode == 'predict':
        predict(args)
    else:
        raise Exception("train/predict? Error args.")