from keras.models import model_from_json
import pickle

def loadFromJSON(filename):
    """
    Create model from loaded JSON and h5

    Parameters
    ----------
    filename : str
        Name of JSON and h5 file that has the model

    Returns
    -------
    loaded_model : keras Sequential
        Trained Keras Sequential model loaded from the JSON
    scaler: sklearn MinMaxScaler
        Fitted scaler loaded from sav

    """
    json_file = open(filename +".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename +".h5")
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return loaded_model, scaler
  

def predictByFeatures(features, loaded_model, scaler):
    features = scaler.transform(features)
    prediction = (loaded_model.predict(features) > 0.5).astype(int)
    return prediction;
