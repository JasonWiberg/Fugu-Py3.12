# isort: skip_file
# fmt: off
import argparse
import numpy as np

from fugu import backends
from fugu.backends import snn_Backend
from fugu.bricks.keras_dense_bricks import keras_dense_2d_4dinput
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput as convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.keras_pooling_bricks import keras_pooling_2d_4dinput as pooling_2d
from fugu.scaffold import Scaffold


try:
    from tensorflow.keras.layers import Layer, Dense, Lambda, TimeDistributed, GRU, MaxPooling2D, Reshape, Conv2D, Flatten, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import Callback
    import tensorflow.keras.backend as K

except ModuleNotFoundError:
    print('tensorflow module not found!')
    import pytest
    pytest.skip(reason="Tensorflow package missing.", allow_module_level=True)
except ImportError:
    raise SystemExit('\n *** Tensorflow package is not installed. *** \n')

# Here is the necessary code from Whetstone package, working in python 3.12:
#---------------------------------------------------------------------------
class Spiking(Layer):
    """Abstract base layer for all spiking activation Layers.

    This layer should not be instantiated, but rather inherited.

    # Arguments
        sharpness: Float, abstract 'sharpness' of the activation.
            Setting sharpness to 0.0 leaves the activation function unmodified.
            Setting sharpness to 1.0 sets the activation function to a threshold gate.
    """
    sharpen_start_limit = 0.0
    sharpen_end_limit = 1.0

    def __init__(self, sharpness=0.0, **kwargs):
        super(Spiking, self).__init__(**kwargs)
        self.supports_masking = True
        self.sharpness = K.variable(K.cast_to_floatx(sharpness))

    def build(self, input_shape):
        super(Spiking, self).build(input_shape)

    def sharpen(self, amount=0.01):
        """Sharpens the activation function by the specified amount.

        # Arguments
            amount: Float, the amount to sharpen.
        """
        K.set_value(self.sharpness, min(max(K.get_value(self.sharpness)+amount, Spiking.sharpen_start_limit), Spiking.sharpen_end_limit))

    def get_config(self):
        """ Provides configuration info so model can be saved and loaded.

        # Returns
            A dictionary of the layer's configuration.
        """
        config = {'sharpness':K.get_value(self.sharpness)}
        base_config = super(Spiking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Spiking_BRelu(Spiking):
    """ A Bounded Rectified Linear Unit layer that can be sharpened to a threshold gate.

        The sharpness value of the layer is inverted to determine the width of the
        linear-region (i.e. non-binary region), which determines the slope
        of the line in the linear-region such that the line intersects y = 0 and y = 1
        at the current step-function borders. The line will always pass through the
        point (0.5, 0.5).
    """
    def __init__(self, **kwargs):
        super(Spiking_BRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Spiking_BRelu, self).build(input_shape)

    def call(self, inputs):
        step_function = K.cast(K.greater_equal(inputs, 0.5), K.floatx())
        width = 1.0 - self.sharpness # width of 'non-binary' region
        _lambda = 0.001
        pbrelu = K.clip((1.0/(width + _lambda))*(inputs - 0.5) + 0.5, 0.0, 1.0)
        return K.switch(K.equal(self.sharpness, 1.0), step_function, pbrelu)

    def get_config(self):
        base_config = super(Spiking_BRelu, self).get_config()
        return dict(list(base_config.items()))

class Spiking_Sigmoid(Spiking):
    """ A Sigmoid layer that can be sharpened to a threshold gate.

        The sharpness value of the layer is inverted to determine the width of the
        linear-region (i.e. non-binary region). The roots of the third derivative of
        the sigmoid are used to map the width to a 'k' value which is used to scale
        the 'x' value in the sigmoid function, which places the knees of sigmoid
        approximately at the current step-function borders.
    """
    def __init__(self, **kwargs):
        super(Spiking_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Spiking_Sigmoid, self).build(input_shape)

    def call(self, inputs):
        step_function = K.cast(K.greater_equal(inputs, 0.0), K.floatx())
        width = 1.0 - self.sharpness # width of 'non-binary' region.
        _lambda = 0.001
        k = (4.0*math.log(2.0 + 3.0**0.5))/(width + _lambda)
        psigmoid = 1.0/(1.0 + K.exp(K.clip(-(inputs)*k, -40, 40)))
        return K.switch(K.equal(self.sharpness, 1.0), step_function, psigmoid)

    def get_config(self):
        base_config = super(Spiking_Sigmoid, self).get_config()
        return dict(list(base_config.items()))


class SpikingGRU(Spiking):
    """ A custom GRU layer that uses sharpenable activations (Spiking_Sigmoid and Spiking_Tanh). """

    def __init__(self, units, **kwargs):
        self.units = units
        super(SpikingGRU, self).__init__(**kwargs)
        self.spiking_sigmoid = Spiking_Sigmoid(sharpness=self.sharpness)
        self.spiking_tanh = Spiking_BRelu(sharpness=self.sharpness)

    def build(self, input_shape):
        # GRU's weight initialization
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units * 3),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units * 3,),
            initializer='zeros',
            name='bias'
        )
        super(SpikingGRU, self).build(input_shape)

    def call(self, inputs):
        batch_size = tensorflow.shape(inputs)[0]
        time_steps = tensorflow.shape(inputs)[1]

        h_t = tensorflow.zeros((batch_size, self.units))  # initial hidden state

        for t in range(inputs.shape[1]):  # loop over time dimension
            x_t = inputs[:, t, :]

            # Compute update gate
            z_t = self.spiking_sigmoid(K.dot(x_t, self.kernel[:, :self.units]) +
                                      K.dot(h_t, self.recurrent_kernel[:, :self.units]) +
                                      self.bias[:self.units])

            # Compute reset gate
            r_t = self.spiking_sigmoid(K.dot(x_t, self.kernel[:, self.units:self.units*2]) +
                                      K.dot(h_t, self.recurrent_kernel[:, self.units:self.units*2]) +
                                      self.bias[self.units:self.units*2])

            # Compute candidate hidden state
            h_t_candidate = self.spiking_tanh(K.dot(x_t, self.kernel[:, self.units*2:]) +
                                              K.dot(r_t * h_t, self.recurrent_kernel[:, self.units*2:]) +
                                              self.bias[self.units*2:])

            h_t = (1 - z_t) * h_t + z_t * h_t_candidate
        #print(h_t)
        return h_t

    def get_config(self):
        base_config = super(SpikingGRU, self).get_config()
        return dict(list(base_config.items()))



class Softmax_Decode(Layer):
    """ A layer which uses a key to decode a sparse representation into a softmax.

    Makes it easier to train spiking classifiers by allowing the use of
    softmax and catagorical-crossentropy loss. Allows for encodings that are
    n-hot where 'n' is the number of outputs assigned to each class. Allows
    encodings to overlap, where a given output neuron can contribute
    to the probability of more than one class.

    # Arguments
        key: A numpy array (num_classes, input_dims) with an input_dim-sized
            {0,1}-vector representative for each class.
        size: A tuple (num_classes, input_dim).  If ``key`` is not specified, then
            size must be specified.  In which case, a key will automatically be generated.
    """
    def __init__(self, key=None, size=None, **kwargs):
        super(Softmax_Decode, self).__init__(**kwargs)
        self.key = _key_check(key, size)
        if type(self.key) is dict and 'value' in self.key.keys():
            self.key = np.array(self.key['value'], dtype=np.float32)
        elif type(self.key) is list:
            self.key = np.array(self.key, dtype=np.float32)
        #self._rescaled_key = K.variable(np.transpose(2*self.key-1))
        ###self.key_shape = self.key.shape[0]
        self._rescaled_key = K.variable(2*np.transpose(self.key)-1)

    def build(self, input_shape):
        super(Softmax_Decode, self).build(input_shape)

    def call(self, inputs):
        #return K.softmax(K.dot(2*(1-inputs),self._rescaled_key))
        return K.softmax(K.dot(2*inputs-1, self._rescaled_key))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],10)
        #return (input_shape[0],self.key_shape)

    def get_config(self):
        base_config = super(Softmax_Decode, self).get_config()
        return dict(list(base_config.items()) + [('key', self.key)])


def _key_check(key, size):
    if(key is None):
        if(size is not None):
            return key_generator(size[0], size[1])
        else:
            raise ValueError("You must specifiy a key or a size tuple.")
    else:
        return key

def key_generator(num_classes, width, sparsity = 0.1, overlapping=True):
    """ Generates a key to encode and decode a one-hot vector into a sparse {0,1}-vector.

    # Arguments
        num_classes: Integer, number of classes represented by the one-hot vector.
        width: Integer, dimensionality of the expansion
        sparsity: Float, approximate ratio of 1's to 0's in the encoded vectors.
        overlapping: Boolean, if ``False``, the encoded vectors are assured to
            be linearly independent.

    # Returns
        An ndarray of size (num_classes, width)
    """
    key = np.zeros((num_classes, width))
    validIdx = list(range(0,width))
    entries_per_class = width//num_classes
    for i in range(0, num_classes):
        row_idx = np.random.choice(validIdx,entries_per_class, replace=False)
        key[i, row_idx] = 1
        if(not overlapping):
            for idx in row_idx:
                validIdx.remove(idx)
    return key

spikingLayersDict = {
    'Spiking_BRelu': Spiking_BRelu,
    'Spiking_Sigmoid': Spiking_Sigmoid,
    'SpikingGRU': SpikingGRU,
    'Softmax_Decode': Softmax_Decode,
    'Spiking': Spiking,
    'Dense': Dense,
    'Lambda': Lambda,
    'TimeDistributed': TimeDistributed,
    'Reshape': Reshape,
    'Conv2D': Conv2D,
    'Flatten': Flatten,
    'Dropout': Dropout,
    'BatchNormalization': BatchNormalization,
    'MaxPooling2D': MaxPooling2D
}

def load_model(filepath):
    """Loads a keras model that can contain custom Whetstone layers.

    Loads and returns the Keras/Whetstone model from a .h5 file at ``filepath``, handling custom layer
    deserialization for you.

    # Arguments
        filepath: Path to Keras/Whetstone model which should be a .h5 file produced by model.save(filepath).

    # Returns
        A keras Model.
    """
    with CustomObjectScope(spikingLayersDict):
        return keras.models.load_model(filepath)

def get_spiking_layer_indices(model):
    """Returns indices of layers that can be sharpened.

    # Arguments
        model: Keras model with one or more Spiking layers.
    """
    return [i for i in range(0, len(model.layers)) if isinstance(model.layers[i], Spiking)]


def set_layer_sharpness(model, values):
    """Sets the sharpness values of all spiking layers.

    # Arguments
        model: Keras model with one or more Spiking layers.
        values: A list of sharpness values (between 0.0 and 1.0 inclusive) for each
            spiking layer in the same order as their indices.
    """
    assert type(values) is list and all([type(i) is float and i >= 0.0 and i <= 1.0 for i in values])
    for i, v in enumerate(values):
        layer = model.layers[get_spiking_layer_indices(model=model)[i]]
        K.set_value(layer.sharpness, K.cast_to_floatx(v))

def set_model_sharpness(model, value, bottom_up):
    """Sets the sharpness of the whole model.

       If ``bottom_up`` is ``True`` sharpens in bottom-up order, otherwise sharpens uniformly.

       # Arguments
            model: Keras model with one or more Spiking layers.
            value: Float, between 0.0 and 1.0 inclusive that specifies the sharpness of the model.
            bottom_up: Boolean, if ``True`` then sharpens in bottom-up order, else uniform.
    """
    assert type(value) is float and value >= 0.0 and value <= 1.0
    num_spiking_layers = len(get_spiking_layer_indices(model=model))
    if bottom_up:
        if value == 1.0: # this makes sure rounding errors don't prevent full sharpening at 1.0
            values = [1.0 for _ in range(num_spiking_layers)]
            set_layer_sharpness(model=model, values=values)
        else:
            portion_per_layer = 1.0 / num_spiking_layers
            num_fully_sharpened = int(value / portion_per_layer)
            scaled_remainder = (value % portion_per_layer) / portion_per_layer
            values = [1.0 for _ in range(num_fully_sharpened)] # for the layers already done sharpening.
            values.append(scaled_remainder) # for the layer that's currently undergoing sharpening.
            values.extend([0.0 for _ in range(num_spiking_layers - num_fully_sharpened - 1)]) # for the layers that have not yet begun to sharpen.
            set_layer_sharpness(model=model, values=values)
    else: # uniform
        values = [value for _ in range(num_spiking_layers)]
        set_layer_sharpness(model=model, values=values)
    return values



class Sharpener(Callback):
    """Absract base class used for different sharpening callbacks.

    # Arguments
        bottom_up: Boolean, if ``True``, sharpens one layer at a time,
            sequentially, starting with the first. If ``False``, sharpens all layers uniformly.
        verbose: Boolean, if ``True``, prints status updates during training.
    """
    def __init__(self, bottom_up=True, verbose=False):
        super(Callback, self).__init__()
        assert type(bottom_up) is bool
        assert type(verbose) is bool
        self.bottom_up = bottom_up
        self.verbose = verbose
        self.current_epoch = 0

    def get_config(self):
        config = {'bottom_up':self.bottom_up, 'verbose':self.verbose}
        return config

    def on_train_begin(self, logs=None):
        self.sharpness = [0.5 for _ in range(self._num_spiking_layers())]
        #self.sharpness = [1.0, 1.0, 0.5]
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        if all([i == 1.0 for i in self.sharpness]):
            self.model.stop_training = True


    def _spiking_layer_indices(self):
        """Returns indices of layers that can be sharpened. """
        return get_spiking_layer_indices(model=self.model)

    def _num_spiking_layers(self):
        """Returns number of layers in self.model that can be sharpened. """
        return len(self._spiking_layer_indices())

    def set_layer_sharpness(self, values):
        """Sets the sharpness values of all spiking layers.

        # Arguments
            values: A list of sharpness values (between 0.0 and 1.0 inclusive) for each
                spiking layer in the same order as their indices.
        """
        set_layer_sharpness(model=self.model, values=values)
        self.sharpness = values

    def set_model_sharpness(self, value):
        """Sets the sharpness of the whole model either in a bottom_up or uniform fashion depending on the
           value of the bottom_up instance variable.

        # Arguments
            value: Float, value between 0.0 and 1.0 inclusive that specifies the sharpness of the model.
        """
        values = set_model_sharpness(model=self.model, value=value, bottom_up=self.bottom_up)
        self.sharpness = values

class AdaptiveSharpener(Sharpener):
    """Sharpens a model automatically, using training loss to control the process.

    # Arguments
        min_init_epochs: Integer, minimum number of epochs to train before sharpening begins.
        rate: Float, amount to sharpen a layer per epoch.
        cz_rate: Float, rate of sharpening in Critical Zone, which is when layer sharpness >= ``critical``.
        critical: Float, critical sharpness after which to apply cz_rate.
        first_layer_relative_rate: Float, percentage of normal sharpening rate to use in first layer.
        patience: Integer, how many epochs to wait for significant improvement.
        sig_increase: Float, percent increase in loss considered significant.
        sig_decrease: Float, percent decrease in loss considered significant.
    """
    def __init__(self, min_init_epochs=10,
                 rate=0.25,
                 cz_rate=0.126,
                 critical=0.75,
                 first_layer_relative_rate=1.0,
                 patience=1,
                 sig_increase=0.15,
                 sig_decrease=0.15,
                 **kwargs):
        super(AdaptiveSharpener, self).__init__(**kwargs)
        assert type(min_init_epochs) is int and min_init_epochs >= 1
        assert type(rate) is float and rate > 0.0 and rate <= 1.0
        assert type(cz_rate) is float and cz_rate > 0.0 and cz_rate <= 1.0
        assert type(critical) is float and critical >= 0.0 and critical <= 1.0
        assert type(first_layer_relative_rate) is float and first_layer_relative_rate > 0.0
        assert type(patience) is int and patience >= 0
        assert type(sig_increase) is float and sig_increase > 0.0
        assert type(sig_decrease) is float and sig_decrease > 0.0
        self.min_init_epochs = min_init_epochs
        self.rate = rate
        self.cz_rate = cz_rate
        self.critical = critical
        self.first_layer_relative_rate = first_layer_relative_rate
        self.patience = patience
        self.sig_increase = sig_increase
        self.sig_decrease = sig_decrease
        try:
            self.batches_per_epoch = batches_per_epoch
        except:
            pass

    def get_config(self):
        config = {'min_init_epochs':self.min_init_epochs,
                  'rate':self.rate,
                  'cz_rate':self.cz_rate,
                  'critical':self.critical,
                  'first_layer_relative_rate':self.first_layer_relative_rate,
                  'patience':self.patience,
                  'sig_increase':self.sig_increase,
                  'sig_decrease':self.sig_decrease,
                 }
        try:
            config['batches_per_epoch'] = self.batches_per_epoch
        except:
            pass
        base_config = super(AdaptiveSharpener, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def on_train_begin(self, logs=None):
        super(AdaptiveSharpener, self).on_train_begin(logs)
        self.sharpening = False # state variable.
        self.reference_loss = 1000000.0 # loss after last significant change.
        self.epochs_no_improvement = 0  # number of epochs since the loss improved significantly.
        self.batch = 0
        self.batches_per_epoch = None
        self.wait = False

    def _perform_sharpening(self, logs=None):
        unfinished_layers = [idx for idx, s in enumerate(self.sharpness) if s < 1.0]
        if len(unfinished_layers) > 0:
            if self.bottom_up:
                if not self.wait:
                    sharpen_idx = min(unfinished_layers)
                    sharpen_amount = self.rate
                    if self.sharpness[sharpen_idx] >= self.critical:
                        sharpen_amount = self.cz_rate
                    if sharpen_idx == 0: # if first spiking layer
                        sharpen_amount *= self.first_layer_relative_rate
                    sharpen_amount *= (1.0/float(self.batches_per_epoch))
                    self.sharpness[sharpen_idx] = min(1.0, self.sharpness[sharpen_idx] + sharpen_amount)
                    if 1.0 - self.sharpness[sharpen_idx] < 0.001:
                        self.sharpness[sharpen_idx] = 1.0
                    if self.sharpness[sharpen_idx] == 1.0:
                        self.wait = True
            else: # uniform sharpen
                sharpen_amount = self.rate
                if self.sharpness[0] >= self.critical:
                    sharpen_amount = self.cz_rate
                sharpen_amount *= (1.0/float(self.batches_per_epoch))
                new_uniform_sharpness = min(1.0, self.sharpness[0] + sharpen_amount)
                if 1.0 - new_uniform_sharpness < 0.000001:
                    new_uniform_sharpness = 1.0
                self.sharpness = [new_uniform_sharpness for _ in range(len(self.sharpness))]
            self.set_layer_sharpness(values=self.sharpness)
        else:
            self.sharpening = False

    def on_epoch_end(self, epoch, logs=None):
        super(AdaptiveSharpener, self).on_epoch_end(epoch, logs)
        self.wait = False # reset overshoot protection flag
        improved, degraded = False, False
        percent_change = (logs['loss'] - self.reference_loss) / self.reference_loss
        if percent_change >= self.sig_increase:
            degraded = True
        elif percent_change <= -self.sig_decrease:
            improved = True
        if self.current_epoch >= self.min_init_epochs - 1:
            if improved:
                self.reference_loss = logs['loss']
                self.epochs_no_improvement = 0
            else: # degraded or remained unchanged
                self.epochs_no_improvement += 1
            if self.sharpening:
                if degraded:
                    self.reference_loss = logs['loss']
                    self.epochs_no_improvement = 0
                    self.sharpening = False
            else: # not sharpening
                if self.epochs_no_improvement > self.patience:
                    self.reference_loss = logs['loss']
                    self.epochs_no_improvement = 0
                    self.sharpening = True
        else: # not time to consider sharpening yet.
            self.reference_loss = logs['loss']
        if epoch == 0:
            self.batches_per_epoch = self.batch + 1
        if self.verbose:
            print('\nloss =', logs['loss'])
            print('current_reference_loss =', self.reference_loss)
            print('percent_change =', percent_change)
            print('improved =', improved, 'degraded =', degraded)
            print('epochs_not_improved =', self.epochs_no_improvement)
            print('sharpening =', self.sharpening)
            print('sharpness =', [round(i, 4) for i in self.sharpness])

    def on_batch_end(self, batch, logs=None):
        if self.sharpening:
            self._perform_sharpening(logs)
        self.batch = batch


class WhetstoneLogger(Callback):
    """Keras callback that handles logging (not a type of beer).

       Automatically creates a separate subfolder for each epoch.

    # Arguments
        logdir: Directory in which to log results.
        sharpener: Reference to callback of type ``Sharpener``.
            If passed, metadata from the sharpener will be recorded.
        test_set: Test set tuple in form (x_test, y_test).
            If passed, test set accuracy will be evaluated on current and
            fully-sharpened versions of the net at the end of each epoch.
        log_weights: Boolean, if ``True``, logs weights of the entire net at the end of
            each epoch.
    """
    def __init__(self, logdir,
                 sharpener=None,
                 test_set=None,
                 log_weights=False):
        super(Callback, self).__init__()
        assert os.path.exists(logdir) and os.path.isdir(logdir)
        assert sharpener is None or isinstance(sharpener, Sharpener)
        assert test_set is None or (type(test_set) is tuple and len(test_set) == 2)
        assert type(log_weights) is bool
        self.logdir = logdir
        self.sharpener = sharpener
        self.test_set = test_set
        self.log_weights = log_weights

    def on_train_begin(self, logs=None):
        # Create metadata files that store sharpener params and copy of exemplar set.
        with open(os.path.join(self.logdir, 'sharpener_params.pkl'), 'wb') as f:
            pickle.dump(self.sharpener.get_config(), f, protocol=1)
        environ_info = {'time':time.time()}
        try:
            environ_info['whetstone_version'] = pkg_resources.get_distribution('whetstone').version
            environ_info['keras_version'] = keras.__version__
            environ_info['numpy_version'] = np.__version__
            environ_info['python_version'] = sys.version
            environ_info['backend'] = str(K._backend)
            if environ_info['backend'] == 'tensorflow':
                environ_info['tensorflow_version'] = K.tf.__version__
        except:
            pass
        with open(os.path.join(self.logdir, 'environ.pkl'), 'wb') as f:
            pickle.dump(environ_info, f, protocol=1)

    def on_epoch_end(self, epoch, logs=None):
        # Create directory to store logs for the current epoch
        epoch_path = os.path.join(self.logdir, 'epoch_'+str(epoch))
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        # Store general logs in a human-readable form.
        logs_ = {'train_loss':logs['loss'], 'train_accuracy':logs['accuracy']}
        if self.sharpener is not None:
            logs_['sharpness'] = self.sharpener.sharpness
        if self.test_set is not None:
            (x_test, y_test) = self.test_set
            logs_['test_loss'], logs_['test_accuracy'] = self.model.evaluate(x_test, y_test, verbose=0)[0:2]
            if self.sharpener is not None:
                self.sharpener.set_layer_sharpness(values=[1.0 for _ in logs_['sharpness']])
                logs_['test_loss_spiking'], logs_['test_accuracy_spiking'] = self.model.evaluate(x_test, y_test, verbose=0)[0:2]
                self.sharpener.set_layer_sharpness(values=logs_['sharpness']) # restore
        log_path = os.path.join(epoch_path, 'log.json')
        with open(log_path, 'w') as f:
            json.dump(logs_, f, indent=4)
        if self.log_weights:
            self.model.save(os.path.join(epoch_path, 'model_epoch_'+str(epoch)+'.h5'))

#--------------------------------------------------------------------------------------


# slightly modified conversion function
def whetstone_2_fugu(keras_model, basep=4, bits=4, scaffold=None, backend=None):
    '''

    '''
    if scaffold is None:
        scaffold = Scaffold()

    if backend is None:
        backend = snn_Backend()

    # model = copy_remove_batchnorm(keras_model)
    model = keras_model
    layerID = 0
    batch_size = 1
    for idx, layer in enumerate(model.layers):
        # print(f"Layer: {layer.name}", type(layer))
        if type(layer) is Conv2D:
            # TODO: Add capability to handle "data_format='channels_first'". Current implementation assumes data_format='channels_last'.
            # TODO: Add capability to handle removal of batchnormalization layer
            # need pvector shape, filters, thresholds, basep, bits, and mode

            # Check if BatchNormalization is the next layer. If so, merge BatchNormalization layer
            # with Convolution2D layer to get the new weights and biases
            next_layer = model.layers[idx + 1] if idx < len(model.layers) - 1 else None
            if type(next_layer) == BatchNormalization:
                kernel, biases = merge_layers(layer,next_layer)
            else:
                kernel = layer.get_weights()[0]
                biases = layer.get_weights()[1]

            if layer.data_format == 'channels_last':
                input_shape_tensor = layer.input.shape
                batch_size = input_shape_tensor[0] or 1  # Default to 1 if None
            elif layer.data_format == 'channels_first':
                #TODO : Handle this scenario later
                pass

            input_shape = tuple([batch_size if dim is None else dim for dim in layer.input.shape])
            output_shape = tuple([batch_size if dim is None else dim for dim in layer.output.shape])
            mode = layer.padding
            strides = layer.strides
            print(f"Conv2D:: LayerID: {layerID+1}")
            scaffold.add_brick(keras_convolution_2d_4dinput(input_shape,np.flip(kernel,(0,1)),0.5,basep,bits,name=f"convolution_layer{layerID}_",mode=mode,strides=strides,biases=biases),[(layerID, 0)],output=True)
            layerID += 1

        if type(layer) is Spiking_BRelu:
            pass

        if type(layer) is MaxPooling2D:
            # need pool size, strides, thresholds, and method
            pool_size = layer.pool_size
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            padding = layer.padding
            strides = layer.strides

            # TODO: update pooling brick to accept 2D tuples for pool size and strides. For now, the brick assumes the pool size/strides is constant in both directions
            scaffold.add_brick(keras_pooling_2d_4dinput(pool_size,strides,name=f"pool_layer_{layerID}",padding=padding,method="max"),[(layerID,0)],output=True)
            layerID += 1

        if type(layer) in [Dense, Softmax_Decode]:
            # need output shape, weights, thresholds
            input_shape = tuple([batch_size if value == None else value for value in layer.input.shape])
            #output_shape = tuple([batch_size if value == None else value for value in layer.output_shape])

            if isinstance(layer, Softmax_Decode):
                units = layer._rescaled_key.shape[1]
            else:
                units = layer.units

            output_shape = (batch_size, units)

            weights = layer.weights[0].numpy()
            try:
                biases = layer.weights[1].numpy()
            except IndexError:
                biases = 0.0
            units = layer.units
            scaffold.add_brick(keras_dense_2d_4dinput(units=units,weights=weights,thresholds=0.5,name=f"dense_layer_{layerID}",input_shape=input_shape,biases=biases),[(layerID,0)],output=True)
            layerID += 1

    return scaffold, backend


# Auxillary/Helper functions
def normalization(batch, batch_normalization_layer):
    gamma, beta, mean, variance = batch_normalization_layer.get_weights()
    epsilon = batch_normalization_layer.epsilon
    return apply_normalization(batch,gamma,beta,mean,variance,epsilon)

def apply_normalization(batch,gamma, beta, moving_mean, moving_var, epsilon):
    return gamma*(batch - moving_mean) / np.sqrt(moving_var+epsilon) + beta

def merge_layers(convolution2d_layer, batch_normalization_layer):
    '''
        Assumes the current layer is Convolution2D layer and the next layer
        is the BatchNormalization layer.
    '''
    gamma, beta, mean, variance = batch_normalization_layer.get_weights()
    epsilon = batch_normalization_layer.epsilon

    # TODO: Add check on weight as bias may not be present.
    weights = convolution2d_layer.get_weights()[0]
    biases = convolution2d_layer.get_weights()[1]

    stdev = np.sqrt(variance + epsilon)
    new_weights = weights * gamma / stdev
    new_biases = (gamma / stdev) * (biases - mean) + beta
    return new_weights, new_biases

def get_merged_layers(current_layer, batch_normalization_layer):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Whetstone keras model filename.")
    parser.add_argument("neural_timesteps", type=int, help="Neural timesteps for the Fugu backend.")
    parser.add_argument("--bits", default=4, type=int, help="Number of bits to use in Fugu.")
    parser.add_argument("--basep", default=4, type=int, help="Base number to use in Fugu.")
    parser.add_argument("--fugu_backend", default="snn", type=str, help="Backend to use in Fugu.")
    args = parser.parse_args()

    whetstone_model = load_model(args.filename)
    scaffold = whetstone_2_fugu(whetstone_model, basep=args.basep, bits=args.bits)

    if args.fugu_backend.lower() == "snn":
        backend = backends.snn_Backend()
    backend_args = {}
    backend.compile(scaffold, backend_args)
    result = backend.run(args.neural_timesteps)
