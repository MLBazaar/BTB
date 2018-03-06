import os
import tempfile

import pandas as pd
import numpy as np

from btb.hyper_parameter import HyperParameter
from .dm_pipeline import DmPipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score

import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class SimpleCNN(DmPipeline):
    """Pipeline class for image datasets.

    Implemented as a 1-layer CNN.

    Class variables:
        HYPERPARAMETER_RANGES: The default hyperparam_ranges. List of HyperParameter objects
        D3M_PRIMITIVES: A list of primitives used in this pipeline.
        IMAGE_WIDTH: The width of the input images. Provided images are rescaled to this width.
        IMAGE_HEIGHT: The height of the input images. Provided images are rescaled to this height.

    Attributes:
        batch_size: default batch size to use when training and predicting.
        epochs: default number of epochs to use when training.
        model: the simple CNN model used for making predictions.
        target_encoder: encodes targets for classification.

    """

    HYPERPARAMETER_RANGES = [
        ('conv_kernel_dim', HyperParameter('int', [3, 10])),
        ('pool_size', HyperParameter('int', [2, 10])),
        ('dropout_percent', HyperParameter('float', [0.0, 0.75]))
    ]

    D3M_PRIMITIVES = ["keras.models.Sequential",
                      "keras.layers.Conv2D",
                      "keras.layers.MaxPooling2D",
                      "keras.layers.Dropout",
                      "keras.layers.Flatten",
                      "keras.layers.Dense"]

    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self, cpus=None, ram=None, batch_size=32, epochs=1):
        super(SimpleCNN, self).__init__(cpus, ram)
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialized in _build_model.
        self.model = None
        # Target encoder only exists for classification
        self.target_encoder = None

    # Methods to make this network pickleable.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        print(self.__dict__)
        pickle_dict = self.__dict__.copy()
        pickle_dict['model_str'] = model_str
        pickle_dict.pop('model', None)
        return pickle_dict

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            pickle_model = keras.models.load_model(fd.name)
            state.pop('model_str', None)
        self.__dict__ = state
        self.model = pickle_model
        print(self.__dict__)

    @classmethod
    def get_d3m_primitives(cls):
        return list(set(cls.D3M_PRIMITIVES[:]))

    @staticmethod
    def _load_task_data(d3mds):
        task_type = d3mds.problem.get_taskType()
        task_sub_type = d3mds.problem.get_taskSubType()
        train_target_name = d3mds.problem.get_targets()[0]['colName']
        return task_type, task_sub_type, train_target_name

    @staticmethod
    def _load_train_data(d3mds):
        img_dir = d3mds.dataset.get_image_path()
        img_names = d3mds.get_train_data()['image_file']
        labels = d3mds.get_train_targets()[:, 0]
        return img_dir, img_names, labels

    def _train_data_generator(self, img_dir, img_names, labels, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch_start_idx = 0
        for img_batch in self._img_data_generator(img_dir, img_names, batch_size):
            batch_end_idx = batch_start_idx + len(img_batch)
            label_batch = np.array(labels[batch_start_idx:batch_end_idx])
            yield img_batch, label_batch
            batch_start_idx = batch_end_idx
            # Reset if we have reached the end of our dataset and must loop to the beginning.
            if batch_end_idx >= len(labels):
                batch_start_idx = 0

    def _img_data_generator(self, img_dir, img_filenames, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        while True:
            img_batch = []
            for img_filename in img_filenames:
                img = load_img(os.path.join(img_dir, img_filename))
                img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
                img = img_to_array(img)
                img_batch.append(img)
                # Yield if we have a full batch
                if len(img_batch) >= batch_size:
                    yield np.array(img_batch)
                    img_batch = []
            # Yield the remaining data in the last batch if there is data.
            if len(img_batch) > 0:
                yield np.array(img_batch)

    def _build_model(self, d3mds, conv_kernel_dim=4, pool_size=4, dropout_percent=0.50):
        task_type, _, _ = self._load_task_data(d3mds)

        # Make model
        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=conv_kernel_dim,
                activation='relu',
                input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 3)))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_percent))
        model.add(Flatten())

        # select model to use
        if task_type == "classification":
            self.target_encoder = LabelEncoder()
            self.train_y = self.target_encoder.fit_transform(self.train_y)
            self.D3M_PRIMITIVES.append('sklearn.preprocessing.LabelEncoder')

            labels = self.train_y.unique().tolist()
            num_classes = len(labels)
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
        elif task_type == "regression":
            print("Setting up regressor")
            model.add(Dense(1, activation='linear'))

            model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())
        else:
            raise Exception("Unsupported task type %s" % task_type)

        return model

    def cv_score(self, d3mds, cv_scoring=(None, None), cv=3):
        cv_scoring_name, cv_scoring_func = cv_scoring
        task_type, task_sub_type, _ = self._load_task_data(d3mds)

        # Set up recommended CV scoring.
        # Assign the splitter while we're at it.
        if task_type == 'classification':
            recommended_scoring = ('f1', f1_score)
            if task_sub_type == 'binary':
                recommended_scoring = ('f1', f1_score)
            elif task_sub_type == 'multiClass':
                recommended_scoring = ('f1_micro',
                                       lambda y_t, y_p: f1_score(y_t, y_p, average='micro'))
            splitter = StratifiedKFold(n_splits=cv)

        elif task_type == 'regression':
            recommended_scoring = ('r2', r2_score)
            splitter = KFold(n_splits=cv)
        else:
            raise Exception("Unsupported task type %s" % task_type)

        if cv_scoring_name is None or cv_scoring_func is None:
            cv_scoring_name, cv_scoring_func = recommended_scoring

        print("Doing Cross Validation")
        print(("Scoring: %s" % cv_scoring_name))
        self.model = self._build_model(d3mds, **self.hyperparams)
        img_dir, img_names, labels = self._load_train_data(d3mds)

        cv_scores = []
        for train_idx, val_idx in splitter.split(img_names, labels):
            train_names = img_names[train_idx]
            train_labels = labels[train_idx]
            val_names = img_names[val_idx]
            val_labels = labels[val_idx]
            self.model.fit_generator(
                self._train_data_generator(img_dir, train_names, train_labels),
                epochs=self.epochs,
                steps_per_epoch=len(train_names) / self.batch_size,
                verbose=0)
            predictions = self.model.predict_generator(
                self._img_data_generator(img_dir, val_names),
                steps=len(val_names) / self.batch_size)
            cv_scores.append(cv_scoring_func(val_labels, predictions))
        cv_score = (np.mean(cv_scores), np.std(cv_scores))

        return cv_score

    def fit(self, d3mds):
        print("Training model")
        self.model = self._build_model(d3mds, **self.hyperparams)
        img_dir, img_names, labels = self._load_train_data(d3mds)
        self.model.fit_generator(
            generator=self._train_data_generator(img_dir, img_names, labels),
            epochs=self.epochs,
            steps_per_epoch=len(img_names) / self.batch_size)

    def predict(self, d3mds):
        fm = d3mds.get_data_all(dropTargets=True)
        img_dir = d3mds.dataset.get_image_path()
        img_names = fm['image_file']
        _, task_sub_type, train_target_name = self._load_task_data(d3mds)

        out_predict = self.model.predict_generator(
            self._img_data_generator(img_dir, img_names), steps=len(img_names) / self.batch_size)

        if task_sub_type in ['multiClass', 'binary']:
            out_predict = self.target_encoder.inverse_transform(out_predict)

        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[train_target_name] = out_predict
        return out_df
