from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import Adam
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def create_model_1():

    model = Sequential([
        Dense(13, activation='tanh', kernel_initializer='glorot_normal'),
        BatchNormalization(),
        Dense(13, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(1, activation='relu', kernel_initializer='glorot_normal') 
    ])

    model.compile(optimizer='adam', loss='mse')

    return(model)

def create_model_2():

    model = Sequential([
        Dense(13, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(26, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(26, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(13, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(13, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(1, activation='tanh', kernel_initializer='glorot_normal') 
    ])

    model.compile(optimizer='adadelta', loss='mae')

    return(model)


def Train_model():

    iri_model = create_model_1()
    rutting_model = create_model_2()

    data_dir = input('Enter dataset location: ')
    train_data = pd.read_excel(data_dir, header=None, skiprows=2, usecols='A:R')
    X = train_data.iloc[:,0:10]
    X_iri = np.array(X.join(train_data.iloc[:,10:13]))
    X_rutting = np.array(X.join(train_data.iloc[:,14:17]))
    y_iri = np.array(train_data[13])
    y_rutting = np.array(train_data[17])

    #training IRI Model
    print('\n\nTraining IRI Model')
    X_train, X_test, y_train, y_test = train_test_split(X_iri, y_iri,
                                            test_size=0.25, random_state=42)
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    
    iri_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=800, verbose=2)
    

    score = iri_model.evaluate(X_test, y_test, verbose=2)
    RMSE = np.sqrt(score)
    Err = (RMSE/np.average(y_test))*100
    print("RMSE : {}".format(RMSE))
    print("Error% : {}".format(Err))

    losses = pd.DataFrame(iri_model.history.history)
    plt.plot(losses)
    plt.title('IRI Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['test', 'train'], loc='upper right')
    plt.show()

    #training Ruting Model
    print('\n\nTraining RUTTING Model')
    X_train, X_test, y_train, y_test = train_test_split(X_rutting, y_rutting,
                                            test_size=0.25, random_state=42)
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    
    rutting_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=1500, verbose=2)

    MAE = rutting_model.evaluate(X_test, y_test, verbose=2)
    Err = (MAE/np.average(y_test))*100
    print("MAE : {}".format(MAE))
    print("Error% : {}".format(Err))

    losses = pd.DataFrame(rutting_model.history.history)
    plt.plot(losses)
    plt.title('Rutting Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['test', 'train'], loc='upper right')
    plt.show()

    iri_model.save('IRI_model_data.h5')
    rutting_model.save('RUTTING_model_data.h5')

    print('\n\nModels Saved\n')

    return


def Predict_data():

    data_dir = input('Enter dataset location: ')
    pred_data = pd.read_excel(data_dir, header=None, skiprows=2, usecols='A:R')
    X = np.array(pred_data.iloc[:,0:10])
    X_iri = np.array(X.join(train_data.iloc[:,10:13]))
    X_rutting = np.array(X.join(train_data.iloc[:,13:16]))
    X_iri = MinMaxScaler().fit_transform(X_iri)
    X_rutting = MinMaxScaler().fit_transform(X_rutting)

    iri_model = load_model('IRI_model_data.h5')
    rutting_model = load_model('RUTTING_model_data.h5')

    iri_pred = iri_model.predict(X_iri)
    rutting_pred = rutting_model.predict(X_rutting)
    
    print('''The expected IRI value is: {}\n
The expected Rutting value is: {}'''.format(iri_pred, rutting_pred))

    return
    

def main():

    while True:

        print('''Choose the operation you want to carry out:
              \n1. Train model\n2. Predict data using inputs\n3. Exit''')
        opt = int(input('Enter the correspoinsing option number: '))
    
        if opt == 1:
            Train_model()
        elif opt == 2:
            Predict_data()
        elif opt == 3:
            break
        else:
            opt = input('Invalid input. Please enter again: ')

    return    
    
if __name__ == "__main__":
    main()
    
