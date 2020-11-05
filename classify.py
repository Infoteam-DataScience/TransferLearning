import sys
import numpy

from os                          import path
from glob                        import glob
from sklearn.datasets            import load_files       
from keras.utils                 import np_utils
from keras.layers                import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers                import Dropout, Flatten, Dense
from keras.models                import Sequential
from keras.callbacks             import ModelCheckpoint  
from keras.preprocessing         import image  
from keras.applications.resnet50 import ResNet50, preprocess_input


def load(filename):
    '''
    Loading the image from filename to tensor.
    '''

    # checks if the file-name exists
    if not path.exists(filename):     

        # display error message
        print('Could not find file!')   

        return None

    # load image and also resizes it
    picture = image.load_img(filename, target_size=(224, 224))    

    # 4D tensor dim (1, 224, 224, 3)    
    return numpy.expand_dims(image.img_to_array(picture), axis=0)

def data():
    '''
    Loading the dog breed data set (splittet)
    '''

    def dataset(filename):

        # checks if file exists
        if not path.exists(filename):             

            return None, None          

        # load the dog data set
        files = load_files(filename)
        
        # define target vectors
        y = np_utils.to_categorical(numpy.array(files['target']), 133)

        return y

    # training and validation data-sets
    yT = dataset('./dataset/train')
    yV = dataset('./dataset/valid')

    # receives all pre-trained features 
    trained = 'transfer/network.npz'
    feature = numpy.load(trained)

    # training and validation data-sets
    xT = feature['train']
    xV = feature['valid']

    return xT, yT, xV, yV

def name():
    '''
    Returns the names of all dog breed as list
    '''
    pattern = 'dataset/train/*/'

    return [item[20:-1] for item in sorted(glob(pattern))]

def init():
    '''
    Initialize step of the pre-trained network
    '''

    # training and validation
    xT, yT, xV, yV = data()

    # checking if data exists
    if xT is None or yT is None:

        # show error message
        print('Missing data-sets for training!')  

        return None 

    # define filenames for loading
    network = './transfer/network.npz'
    weights = './transfer/weights.hdf5'

    model = Sequential()

    # dense classification layer
    model.add(GlobalAveragePooling2D(input_shape=xT.shape[1:]))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.250)) # regularization

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.125)) # regularization

    model.add(Dense(133, activation='softmax'))
    
    # checks pre-trained weights
    if path.exists(weights):

        print('Apply Transfer Learning...')

    else:

        print('Start Transfer Learning...')

        # optimization parameters 
        model.compile(optimizer='rmsprop', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # model checkpoints
        checkpoints = ModelCheckpoint(filepath=weights, verbose=1, 
                                      save_best_only=True)

        model.fit(xT, yT, verbose=1, validation_data=(xV, yV),
                  epochs=10, batch_size=20, callbacks=[checkpoints])

    # loading the model weights
    model.load_weights(weights)

    return model

def eval(model, image):
    '''
    Predicting dog breed on handed in image
    '''

    # receiving all available dog breed' names
    dogs = name()

    def pretrained(tensor):
        '''
        Extract features prom pre-trained network
        '''    
        network = ResNet50(weights='imagenet', include_top=False)

        return network.predict(preprocess_input(tensor))

    # receive features from pretrained network
    features = pretrained(image)

    # evaluate prediction and convert to breed
    prediction = dogs[numpy.argmax(model.predict(features))]

    print('Dog breed: ' + str(prediction))


def main(args):

    image = load(args[0])

    if image is None:   
        return 

    model = init()

    if model is None:
        return 

    # predict the dog breed
    eval(model, image)

if __name__ == "__main__":
    main(sys.argv[1:])