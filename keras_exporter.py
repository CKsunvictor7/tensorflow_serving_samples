from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras import regularizers
from keras.models import Sequential, Model
import os

# very important to do this as a first thing
K.set_learning_phase(0)  # all new operations will be in test mode from now on

model_path = os.path.join(os.path.sep, 'home', 'chuenkai.shie', 'code', 'diet','Inceptv3_Mix_6.hdf5')
network_path = os.path.join(os.path.sep, 'home', 'chuenkai.shie', 'code', 'diet','Inceptv3_Mix_6.json')
nb_classes = 2

def json_gen():
    from keras.applications.xception import Xception
    base_model = Xception(include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.2)(x)
    # add a fully-connected layer
    x = Dense(512, activation='relu', )(x)
    # and a logistic layer
    predictions = Dense(nb_classes, kernel_regularizer=regularizers.l2(0.1), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
    # json_string = model.to_json()
    # open('t.json', 'w').write(json_string)


# serialize the model and get its weights, for quick re-building
model = model_from_json(open(network_path).read())
model.summary()
model.load_weights(model_path)

if (model.uses_learning_phase):
    raise ValueError('Model using learning phase.')


print('start exporting')
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def




# export_path can't already exit
export_path = os.path.join(os.path.sep, 'home', 'chuenkai.shie', 'TFS_models', '1')
builder = saved_model_builder.SavedModelBuilder(export_path)

# original
prediction_signature = predict_signature_def(inputs={'images': model.input},
                                  outputs={'scores': model.output},)

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={
                                             'predict': prediction_signature,
                                         },
                                         )
    builder.save()

print('Done exporting!')



