from keras.models import Sequential,model_from_json,Model
from keras.layers import Conv2D,LSTM,GRU,TimeDistributed,Dense,Flatten,Input,Lambda,multiply
from keras.optimizers import RMSprop,Adam
from keras import backend as K
import tensorflow as tf
from .loss import mean_huber_loss


class Network:
    def __init__(self,stateCnt,actionCnt,recurrent,mode,learning_rate,name):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.learning_rate = learning_rate
        #self.batch_size = batch_size
        
        self.recurrent = recurrent
        self.mode = mode
        self.name = name

    def build2(self):
        with tf.variable_scope(self.name+'-'+self.mode):
            inpt = Input(shape = self.stateCnt, name = "input")
            if self.mode == "linear":                   
                flatten_hidden = Flatten(name = "flatten")(input_data)
                output = Dense(num_actions, name = "output")(flatten_hidden)
            else:
                if self.recurrent:
                    # shape should be (timesteps,height,width,color)
                    conv1 = TimeDistributed(Conv2D(32, (8, 8), strides = 4, activation = "relu", name = "conv1"))(inpt)
                    conv2 = TimeDistributed(Conv2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2"))(conv1)
                    conv3 = TimeDistributed(Conv2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3"))(conv2)
                    flatten_hidden = TimeDistributed(Flatten())(conv3)
                    hidden_input = TimeDistributed(Dense(512, activation = 'relu', name = 'flat_to_512')) (flatten_hidden)
                    context = LSTM(512, return_sequences=False, stateful=False) (hidden_input)

                if self.mode == "dqn":
                    h4 = Dense(512, activation='relu', name = "fc")(context)
                    output = Dense(num_actions, name = "output")(h4)
                elif self.mode == "duel":
                    value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(context)
                    value = Dense(1, name = "value")(value_hidden)

                    action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(context)
                    action = Dense(self.actionCnt, name = "action")(action_hidden)

                    action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keep_dims = True), name = 'action_mean')(action) 
                    output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'output')([action, value, action_mean])
            model = Model(inputs = inpt, outputs = output)
            #model.compile(loss=mean_huber_loss,optimizer=Adam(lr=self.learning_rate))
            return model
            