import numpy as np
from btb.hyper_parameter import HyperParameter
import pandas as pd
import os
from .dm_pipeline import DmPipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from tempfile import NamedTemporaryFile
from keras.models import save_model, load_model, Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import Adadelta, Adam


class LSTM_Text(DmPipeline):
    """Pipeline abstract class based on that of DeepMining.
        Will be subclassed for each type of pipeline
        (eg simple image, image with keras, etc)
    From http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/

    Class variables:
        HYPERPARAMETER_RANGES: The default hyperparam_ranges.
            List of HyperParameter objects

    Attributes:
        hyperparam_ranges: List of HyperParameter objects, can be fine tuned from
            class defauls based on cpu/ram/dataset.
        d3m_primitives: list of strings of d3m primitives used in pipeline
        recommended_scoring: string represengint suggested scoring metric
        hyperparams: current hyperparameters
        d3mds: object of D3M dataset for pipeline
        cpu: cpu information of system that pipeline is run on. Used to fine
            tune hyperparam ranges
        ram: ram of system that pipeline is run on. Used to fine tune hyperparam
            ranges.

    """

    HYPERPARAMETER_RANGES = [
        ('num_top_words', HyperParameter('int', [1000, 40000])),
        ('embedding_size', HyperParameter('int', [100, 500])),
        ('conv_kernel_dim', HyperParameter('int', [3, 10])),
        ('pool_size', HyperParameter('int', [2, 10])),
        ('dropout_percent', HyperParameter('float', [0.1, 0.75])),

    ]

    D3M_PRIMITIVES = [
        "tempfile.NamedTemporaryFile",
        "keras.models.save_model",
        "keras.models.load_model",
        "keras.preprocessing.text.Tokenizer",
        "keras.models.Sequential",
        "keras.layers.Embedding",
        "keras.layers.Conv1D",
        "keras.layers.MaxPooling1D",
        "keras.layers.Dropout",
        "keras.layers.Flatten",
        "keras.layers.LSTM",
        "keras.layers.Dense",
        "keras.preprocessing.sequence.pad_sequences",
    ]

    def __init__(self, cpus=None, ram=None, batch_size=10, epochs=10):
        super(LSTM_Text, self).__init__(cpus, ram)
        self.batch_size = batch_size
        self.epochs = epochs
        # Initialized in _build_model.
        self.model = None
        self.tokenizer = None
        self.pad_length = 1500 #set to length of longest sequence in fit
        # Target encoder only exists for multiclass classification
        self.target_encoder = None

    @classmethod
    def get_d3m_primitives(cls):
        return list(set(cls.D3M_PRIMITIVES[:]))


    # Methods to make this network pickleable.
    def __getstate__(self):
        with NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        print(self.__dict__)
        pickle_dict = self.__dict__.copy()
        pickle_dict['model_str'] = model_str
        pickle_dict.pop('model', None)
        return pickle_dict

    def __setstate__(self, state):
        with NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            pickle_model = load_model(fd.name)
            state.pop('model_str', None)
        self.__dict__ = state
        self.model = pickle_model
        print(self.__dict__)

    def _load_task_data(self, d3mds):
        task_type = d3mds.problem.get_taskType()
        task_sub_type = d3mds.problem.get_taskSubType()
        train_target_name = d3mds.problem.get_targets()[0]['colName']
        return task_type, task_sub_type, train_target_name

    def _load_train_data(self, d3mds):
        text_dir = d3mds.dataset.get_text_path()
        text_names = d3mds.get_train_data()['raw_text_file'].values.T
        labels = d3mds.get_train_targets()[:, 0]
        return text_dir, text_names, labels

    def _train_data_generator(self, text_dir, text_names, labels, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch_start_idx = 0
        for text_batch in self._text_data_generator(text_dir, text_names, batch_size):
            batch_end_idx = batch_start_idx + len(text_batch)
            label_batch = labels[batch_start_idx: batch_end_idx]
            yield text_batch, label_batch
            batch_start_idx = batch_end_idx
            # Reset if we have reached the end of our dataset and must loop to the beginning.
            if batch_end_idx >= len(labels):
                batch_start_idx = 0

    def _text_data_generator(self, text_dir, text_filenames, batch_size=None, terminate = False):
        if batch_size is None:
            batch_size = self.batch_size
        while True:
            text_batch = []
            for text_filename in text_filenames:
                with open(os.path.join(text_dir, text_filename), 'r') as text_file:
                    text=text_file.read()
                    text_batch.append(self.tokenizer.texts_to_sequences([text])[0])
                if len(text_batch) >= batch_size:
                    padded = pad_sequences(text_batch, maxlen=self.pad_length)
                    yield padded
                    text_batch = []
            if len(text_batch) > 0:
                padded = pad_sequences(text_batch, maxlen=self.pad_length)
                yield padded

    def _load_all_text_data(self, text_dir, text_filenames):
        texts = []
        max_words = 0
        for text_filename in text_filenames:
            with open(os.path.join(text_dir, text_filename), 'r') as text_file:
                text=text_file.read()
                translator = str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                num_words = len(text.translate(translator).split(' '))
                texts.append(text)
                if num_words > max_words:
                    max_words = num_words
        return texts, max_words

    def _build_model(self, d3mds, num_top_words = 20000, embedding_size = 128,
        conv_kernel_dim=5, pool_size=4, dropout_percent= 0.2):
        #make embedding_size always even for integer half
        if embedding_size%2 != 0:
            embedding_size += 1
        task_type, task_sub_type, _ = self._load_task_data(d3mds)
        # Make model
        tokenizer = Tokenizer(num_words=num_top_words)

        model = Sequential()
        model.add(Embedding(num_top_words, embedding_size, input_length=self.pad_length))
        model.add(Dropout(dropout_percent))
        model.add(Conv1D(embedding_size//2, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(embedding_size))
        labels = d3mds.get_train_targets()[:, 0]

        # select model to use
        if task_type == "classification":
            if task_sub_type == "binary":
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            elif task_sub_type == "multiClass":
                num_classes =  len(np.unique(labels).tolist())
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(labels)
                self.D3M_PRIMITIVES.append('sklearn.preprocessing.LabelEncoder')
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(
                    loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy']
                )

        elif task_type == "regression":
            model.add(Dense(1, activation='linear'))
            model.compile(
                loss=keras.losses.mean_squared_error,
                optimizer=keras.optimizers.Adam())
        else:
            raise UnsupportedTaskTypeException(task_type)
        return tokenizer, model

    def cv_score(self, d3mds, cv_scoring=(None, None), cv=3):
        if cv_scoring is None:
            cv_scoring_name, cv_scoring_func = None, None
        else:
            cv_scoring_name, cv_scoring_func = cv_scoring
        task_type, task_sub_type, _ = self._load_task_data(d3mds)

        # Set up recommended CV scoring.
        # Assign the splitter while we're at it.
        if task_type == 'classification':
            if task_sub_type == 'binary':
                recommended_scoring = ('roc_auc', roc_auc_score)
                self.D3M_PRIMITIVES.append('sklearn.metrics.roc_auc_score')
            elif task_sub_type == 'multiClass':
                recommended_scoring =('f1_micro', lambda y_t, y_p: f1_score(y_t, y_p, average='micro'))
                self.D3M_PRIMITIVES.append('sklearn.metrics.f1_micro')
            else:
                recommended_scoring = ('f1', f1_score)
                self.D3M_PRIMITIVES.append('sklearn.metrics.f1_score')
            splitter = StratifiedKFold(n_splits=cv)
            self.D3M_PRIMITIVES.append('sklearn.model_selection.StratifiedKFold')

        elif task_type == 'regression':
            recommended_scoring = ('r2', r2_score)
            self.D3M_PRIMITIVES.append('sklearn.metrics.r2_score')
            splitter = KFold(n_splits=cv)
            self.D3M_PRIMITIVES.append('sklearn.model_selection.KFold')
        else:
            raise UnsupportedTaskTypeException(task_type)

        if cv_scoring_name is None or cv_scoring_func is None:
            cv_scoring_name, cv_scoring_func = recommended_scoring

        print("Doing Cross Validation")
        print(("Scoring: %s" % cv_scoring_name))
        text_dir, text_names, labels = self._load_train_data(d3mds)
        cv_scores = []
        for train_idx, val_idx in splitter.split(text_names, labels):
            train_names = text_names[train_idx]
            train_labels = labels[train_idx]
            val_names = text_names[val_idx]
            val_labels = labels[val_idx]
            texts, max_words = self._load_all_text_data(text_dir, train_names)
            self.pad_length = max_words
            self.tokenizer, self.model = self._build_model(d3mds, **self.hyperparams)
            self.tokenizer.fit_on_texts(texts)
            self.model.fit_generator(self._train_data_generator(text_dir, train_names, train_labels), epochs=self.epochs, steps_per_epoch=len(train_names)//self.batch_size)
            steps_needed = len(val_labels)//self.batch_size
            if len(val_labels)//self.batch_size ==  len(val_labels)/self.batch_size:
                steps_needed = len(val_labels)//self.batch_size
            else:
                steps_needed = len(val_labels)//self.batch_size + 1
            predictions = self.model.predict_generator(
                self._text_data_generator(text_dir, val_names, terminate = True), steps = steps_needed
            )
            if task_sub_type == 'multiClass':
                predictions = self.target_encoder.inverse_transform(predictions)
            elif task_sub_type  == 'binary':
                predictions = np.round(predictions)

            min_len = min(len(predictions), len(val_labels))
            predictions = predictions[:min_len]
            val_labels = val_labels[:min_len]

            score = cv_scoring_func(val_labels, predictions)
            cv_scores.append(score)

        cv_score = (np.mean(cv_scores), np.std(cv_scores))
        return cv_score

    def predict(self, d3mds):
        fm = d3mds.get_data_all(dropTargets=True)
        text_dir = d3mds.dataset.get_text_path()
        text_names = fm['raw_text_file']
        if len(text_names)//self.batch_size ==  len(text_names)/self.batch_size:
            steps_needed = len(text_names)//self.batch_size
        else:
            steps_needed = len(text_names)//self.batch_size + 1
        _, task_sub_type, train_target_name = self._load_task_data(d3mds)

        out_predict = self.model.predict_generator(self._text_data_generator(text_dir, text_names), steps = steps_needed)

        if task_sub_type == 'multiClass':
            out_predict = self.target_encoder.inverse_transform(out_predict)
        elif task_sub_type  == 'binary':
            out_predict = np.round(out_predict)
        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[train_target_name] = out_predict
        return out_df

    def fit(self, d3mds):
        print("Training model")
        text_dir, text_names, labels = self._load_train_data(d3mds)
        texts, max_words = self._load_all_text_data(text_dir, text_names)
        self.pad_length = max_words
        self.tokenizer, self.model = self._build_model(d3mds, **self.hyperparams)
        self.tokenizer.fit_on_texts(texts)
        self.model.fit_generator(generator=self._train_data_generator(text_dir, text_names, labels), epochs=self.epochs, steps_per_epoch=len(texts)//self.batch_size)


class UnsupportedTaskTypeException(Exception):
    def __init__(self, task_type):
        super(UnsupportedTaskTypeException, self).__init__("Unsupported task type %s" % task_type)
