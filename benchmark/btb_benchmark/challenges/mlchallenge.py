# -*- coding: utf-8 -*-

"""Package where the MLChallenge class is defined."""

import logging
import os
from copy import deepcopy
from urllib.parse import urljoin

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from btb_benchmark.challenges.challenge import Challenge

BASE_DATASET_URL = 'https://atm-data.s3.amazonaws.com/'
BUCKET_NAME = 'atm-data'
LOGGER = logging.getLogger(__name__)

# Available datasets sorted by execution time, slowest first
DATASETS_BY_TIME = [
    'BNG(cmc)_1.csv',
    'eye_movements_1.csv',
    'wall-robot-navigation_1.csv',
    'analcatdata_germangss_2.csv',
    'scene_1.csv',
    'arcene_1.csv',
    'mv_1.csv',
    'bank32nh_1.csv',
    'BNG(cmc,nominal,55296)_1.csv',
    'AP_Endometrium_Prostate_1.csv',
    'house_16H_1.csv',
    'waveform-5000_1.csv',
    'autoUniv-au4-2500_1.csv',
    'BNG(breast-w)_1.csv',
    'wall-robot-navigation_2.csv',
    'madelon_1.csv',
    'Click_prediction_small_1.csv',
    'wall-robot-navigation_3.csv',
    'nursery_1.csv',
    'kdd_JapaneseVowels_1.csv',
    'house_8L_1.csv',
    'MagicTelescope_1.csv',
    'vehicle_1.csv',
    'musk_1.csv',
    'BNG(vote)_1.csv',
    'PopularKids_1.csv',
    'abalone_2.csv',
    'rmftsa_sleepdata_2.csv',
    'yeast_ml8_1.csv',
    'leukemia_1.csv',
    'electricity_1.csv',
    'CreditCardSubset_1.csv',
    'cmc_1.csv',
    'car_1.csv',
    'autoUniv-au7-700_1.csv',
    'grub-damage_1.csv',
    'analcatdata_authorship_1.csv',
    'skin-segmentation_1.csv',
    'splice_1.csv',
    'cpu_small_1.csv',
    'tumors_C_1.csv',
    'cpu_act_1.csv',
    'ringnorm_1.csv',
    'bank-marketing_2.csv',
    'mfeat-factors_1.csv',
    'pol_1.csv',
    '2dplanes_1.csv',
    'Amazon_employee_access_1.csv',
    'cal_housing_1.csv',
    'houses_1.csv',
    'BNG(tic-tac-toe)_1.csv',
    'mozilla4_1.csv',
    'cardiotocography_1.csv',
    'meta_ensembles.arff_1.csv',
    'balance-scale_1.csv',
    'fried_1.csv',
    'tae_1.csv',
    'spambase_1.csv',
    'ozone-level-8hr_1.csv',
    'PhishingWebsites_1.csv',
    'elevators_1.csv',
    'bank8FM_1.csv',
    'mammography_1.csv',
    'hayes-roth_2.csv',
    'sylva_prior_1.csv',
    'meta_batchincremental.arff_1.csv',
    'waveform-5000_2.csv',
    'teachingAssistant_1.csv',
    'ailerons_1.csv',
    'robot-failures-lp1_1.csv',
    'desharnais_1.csv',
    'PieChart4_1.csv',
    'eeg-eye-state_1.csv',
    'Engine1_1.csv',
    'puma32H_1.csv',
    'lymph_1.csv',
    'wind_1.csv',
    'mfeat-fourier_1.csv',
    'steel-plates-fault_1.csv',
    'mfeat-karhunen_1.csv',
    'meta_instanceincremental.arff_1.csv',
    'page-blocks_1.csv',
    'twonorm_1.csv',
    'mc1_1.csv',
    'wine_1.csv',
    'hill-valley_1.csv',
    'robot-failures-lp4_1.csv',
    'kc1_1.csv',
    'segment_1.csv',
    'spectrometer_1.csv',
    'seismic-bumps_1.csv',
    'pc4_1.csv',
    'pc3_1.csv',
    'ada_agnostic_1.csv',
    'splice_2.csv',
    'bank-marketing_1.csv',
    'seeds_1.csv',
    'delta_ailerons_1.csv',
    'white-clover_1.csv',
    'boston_1.csv',
    'letter_1.csv',
    'prnn_viruses_1.csv',
    'oil_spill_1.csv',
    'credit-g_1.csv',
    'diabetes_1.csv',
    'boston_corrected_1.csv',
    'PieChart3_1.csv',
    'optdigits_1.csv',
    'PizzaCutter3_1.csv',
    'qsar-biodeg_1.csv',
    'abalone_1.csv',
    'cmc_2.csv',
    'iris_1.csv',
    'vinnie_1.csv',
    'pc1_1.csv',
    'balloon_1.csv',
    'tic-tac-toe_1.csv',
    'fri_c4_1000_100_1.csv',
    'robot-failures-lp3_1.csv',
    'autoUniv-au1-1000_1.csv',
    'lsvt_1.csv',
    'hill-valley_2.csv',
    'banana_1.csv',
    'vertebra-column_2.csv',
    'quake_1.csv',
    'rmftsa_sleepdata_1.csv',
    'kr-vs-kp_1.csv',
    'mfeat-zernike_1.csv',
    'wisconsin_1.csv',
    'CastMetal1_1.csv',
    'car_2.csv',
    'pendigits_1.csv',
    'PizzaCutter1_1.csv',
    'kc2_1.csv',
    'delta_elevators_1.csv',
    'mw1_1.csv',
    'analcatdata_asbestos_1.csv',
    'mu284_1.csv',
    'plasma_retinol_1.csv',
    'quake_2.csv',
    'socmob_1.csv',
    'PieChart1_1.csv',
    'housing_1.csv',
    'pasture_1.csv',
    'sensory_1.csv',
    'wine_2.csv',
    'chscase_geyser1_1.csv',
    'monks-problems-2_1.csv',
    'mfeat-pixel_1.csv',
    'analcatdata_dmft_1.csv',
    'autoPrice_1.csv',
    'SPECTF_1.csv',
    'breast-tissue_1.csv',
    'phoneme_1.csv',
    'vehicle_2.csv',
    'visualizing_soil_1.csv',
    'rabe_266_1.csv',
    'blood-transfusion-service-center_1.csv',
    'thoracic-surgery_1.csv',
    'ionosphere_1.csv',
    'pm10_1.csv',
    'sa-heart_1.csv',
    'analcatdata_authorship_2.csv',
    'jEdit_4.2_4.3_1.csv',
    'Australian_1.csv',
    'ecoli_1.csv',
    'ilpd_1.csv',
    'climate-model-simulation-crashes_1.csv',
    'pc1_req_1.csv',
    'sonar_1.csv',
    'mc2_1.csv',
    'auto_price_1.csv',
    'arsenic-female-bladder_1.csv',
    'kdd_synthetic_control_1.csv',
    'analcatdata_supreme_1.csv',
    'prnn_fglass_1.csv',
    'kin8nm_1.csv',
    'no2_1.csv',
    'vowel_1.csv',
    'pc2_1.csv',
    'fri_c4_1000_50_1.csv',
    'stock_1.csv',
    'tae_2.csv',
    'transplant_1.csv',
    'chscase_funds_1.csv',
    'SPECT_1.csv',
    'visualizing_galaxy_1.csv',
    'analcatdata_vineyard_1.csv',
    'newton_hema_1.csv',
    'puma8NH_1.csv',
    'fri_c4_500_100_1.csv',
    'wilt_1.csv',
    'analcatdata_germangss_1.csv',
    'dresses-sales_1.csv',
    'haberman_1.csv',
    'fri_c1_1000_50_1.csv',
    'fri_c2_1000_50_1.csv',
    'white-clover_2.csv',
    'ar4_1.csv',
    'chscase_vine2_1.csv',
    'balance-scale_2.csv',
    'CostaMadre1_1.csv',
    'diggle_table_a2_1.csv',
    'arsenic-male-bladder_1.csv',
    'sleuth_case2002_1.csv',
    'grub-damage_2.csv',
    'glass_1.csv',
    'rmftsa_ctoarrivals_1.csv',
    'jEdit_4.0_4.2_1.csv',
    'analcatdata_apnea1_1.csv',
    'prnn_cushings_1.csv',
    'pwLinear_1.csv',
    'analcatdata_apnea2_1.csv',
    'analcatdata_boxing2_1.csv',
    'visualizing_livestock_1.csv',
    'PieChart2_1.csv',
    'monks-problems-3_1.csv',
    'wholesale-customers_1.csv',
    'analcatdata_wildcat_1.csv',
    'fri_c3_1000_50_1.csv',
    'flags_1.csv',
    'analcatdata_cyyoung9302_1.csv',
    'fri_c3_1000_5_1.csv',
    'monks-problems-1_1.csv',
    'bodyfat_1.csv',
    'fri_c1_1000_25_1.csv',
    'fri_c3_1000_25_1.csv',
    'veteran_1.csv',
    'baskball_1.csv',
    'backache_1.csv',
    'kc3_1.csv',
    'kc1-binary_1.csv',
    'heart-statlog_1.csv',
    'fri_c0_1000_50_1.csv',
    'visualizing_ethanol_1.csv',
    'parkinsons_1.csv',
    'lowbwt_1.csv',
    'analcatdata_michiganacc_1.csv',
    'sleuth_ex1605_1.csv',
    'fruitfly_1.csv',
    'machine_cpu_1.csv',
    'tecator_1.csv',
    'triazines_1.csv',
    'pyrim_1.csv',
    'fri_c2_1000_10_1.csv',
    'lymph_2.csv',
    'fri_c3_500_50_1.csv',
    'fri_c0_1000_25_1.csv',
    'fri_c2_500_50_1.csv',
    'cloud_1.csv',
    'fri_c2_500_25_1.csv',
    'pollution_1_train.csv',
    'analcatdata_seropositive_1.csv',
    'pasture_2.csv',
    'analcatdata_lawsuit_1.csv',
    'dbworld-bodies_1.csv',
    'fri_c4_250_100_1.csv',
    'sleuth_ex2016_1.csv',
    'vineyard_1.csv',
    'analcatdata_apnea3_1.csv',
    'mfeat-morphological_1.csv',
    'analcatdata_boxing1_1.csv',
    'MegaWatt1_1.csv',
    'cm1_req_1.csv',
    'iris_2.csv',
    'fri_c2_1000_25_1.csv',
    'servo_1.csv',
    'fri_c3_1000_10_1.csv',
    'space_ga_1.csv',
    'nursery_2.csv',
    'fri_c4_1000_25_1.csv',
    'pollution_1.csv',
    'fri_c0_100_10_1.csv',
    'cpu_1.csv',
    'strikes_1.csv',
    'rabe_148_1.csv',
    'fri_c1_500_50_1.csv',
    'rmftsa_ladata_1.csv',
    'dbworld-subjects-stemmed_1.csv',
    'blogger_1.csv',
    'molecular-biology_promoters_1.csv',
    'kidney_1.csv',
    'fri_c1_500_25_1.csv',
    'fri_c2_250_50_1.csv',
    'diggle_table_a1_1.csv',
    'fri_c1_1000_10_1.csv',
    'sleuth_ex1221_1.csv',
    'analcatdata_vehicle_1.csv',
    'fl2000_1.csv',
    'kc1-top5_1.csv',
    'qualitative-bankruptcy_1.csv',
    'hutsof99_logis_1.csv',
    'sleuth_case1202_1.csv',
    'ar5_1.csv',
    'datatrieve_1.csv',
    'chscase_census4_1.csv',
    'analcatdata_cyyoung8092_1.csv',
    'acute-inflammations_1.csv',
    'fri_c3_250_25_1.csv',
    'chscase_census5_1.csv',
    'analcatdata_election2000_1.csv',
    'fri_c4_500_50_1.csv',
    'fri_c0_500_10_1.csv',
    'fri_c0_1000_10_1.csv',
    'hutsof99_child_witness_1.csv',
    'chscase_vine1_1.csv',
    'mbagrade_1.csv',
    'fri_c4_100_25_1.csv',
    'fri_c1_250_25_1.csv',
    'fri_c2_250_25_1.csv',
    'acute-inflammations_2.csv',
    'fri_c3_500_5_1.csv',
    'sleuth_case1102_1.csv',
    'fri_c2_100_25_1.csv',
    'analcatdata_gviolence_1.csv',
    'banknote-authentication_1.csv',
    'humandevel_1.csv',
    'fri_c0_500_50_1.csv',
    'chscase_health_1.csv',
    'pollution_1_test.csv',
    'fri_c0_250_50_1.csv',
    'fertility_1.csv',
    'chscase_census2_1.csv',
    'pollen_1.csv',
    'ar1_1.csv',
    'fri_c1_250_50_1.csv',
    'dbworld-subjects_1.csv',
    'arsenic-female-lung_1.csv',
    'fri_c0_100_25_1.csv',
    'schlvote_1.csv',
    'ar3_1.csv',
    'disclosure_x_bias_1.csv',
    'vertebra-column_1.csv',
    'fri_c4_500_25_1.csv',
    'molecular-biology_promoters_2.csv',
    'hayes-roth_1.csv',
    'fri_c3_100_50_1.csv',
    'disclosure_x_noise_1.csv',
    'wind_correlations_1.csv',
    'fri_c3_100_10_1.csv',
    'visualizing_environmental_1.csv',
    'fri_c2_100_5_1.csv',
    'fri_c4_250_10_1.csv',
    'fri_c4_100_100_1.csv',
    'fri_c0_250_10_1.csv',
    'wdbc_1.csv',
    'fri_c1_100_50_1.csv',
    'fri_c0_100_5_1.csv',
    'confidence_1.csv',
    'analcatdata_bankruptcy_1.csv',
    'fri_c3_100_5_1.csv',
    'fri_c0_250_25_1.csv',
    'elusage_1.csv',
    'fri_c4_100_50_1.csv',
    'sleuth_case1201_1.csv',
    'fri_c1_100_25_1.csv',
    'chatfield_4_1.csv',
    'analcatdata_chlamydia_1.csv',
    'fri_c2_500_5_1.csv',
    'chscase_census6_1.csv',
    'fri_c4_250_25_1.csv',
    'aids_1.csv',
    'bolts_1.csv',
    'fri_c0_500_25_1.csv',
    'fri_c4_1000_10_1.csv',
    'fri_c2_100_50_1.csv',
    'rabe_97_1.csv',
    'fri_c3_250_50_1.csv',
    'analcatdata_olympic2000_1.csv',
    'planning-relax_1.csv',
    'collins_1.csv',
    'rabe_265_1.csv',
    'sleuth_ex2015_1.csv',
    'fri_c1_250_10_1.csv',
    'fri_c0_1000_5_1.csv',
    'lupus_1.csv',
    'visualizing_hamster_1.csv',
    'sleuth_ex1714_1.csv',
    'chscase_adopt_1.csv',
    'fri_c1_500_5_1.csv',
    'fri_c2_1000_5_1.csv',
    'fri_c3_500_25_1.csv',
    'fri_c4_250_50_1.csv',
    'rabe_131_1.csv',
    'fri_c3_250_10_1.csv',
    'zoo_1.csv',
    'ar6_1.csv',
    'witmer_census_1980_1.csv',
    'fri_c0_100_50_1.csv',
    'analcatdata_challenger_1.csv',
    'fri_c4_500_10_1.csv',
    'fri_c1_100_5_1.csv',
    'fri_c2_500_10_1.csv',
    'fri_c3_100_25_1.csv',
    'fri_c2_250_10_1.csv',
    'fri_c3_500_10_1.csv',
    'rabe_166_1.csv',
    'fri_c3_250_5_1.csv',
    'fri_c0_500_5_1.csv',
    'analcatdata_creditscore_1.csv',
    'chscase_census3_1.csv',
    'fri_c1_1000_5_1.csv',
    'fri_c2_100_10_1.csv',
    'fri_c1_500_10_1.csv',
    'prnn_synth_1.csv',
    'fri_c0_250_5_1.csv',
    'badges2_1.csv',
    'fri_c2_250_5_1.csv',
    'visualizing_slope_1.csv',
    'disclosure_x_tampered_1.csv',
    'fri_c1_100_10_1.csv',
    'analcatdata_neavote_1.csv',
    'disclosure_z_1.csv',
    'diabetes_numeric_1.csv',
    'analcatdata_challenger_2.csv',
    'fri_c1_250_5_1.csv',
    'fri_c4_100_10_1.csv',
    'rabe_176_1.csv',
    'arsenic-male-lung_1.csv',
    'sleuth_ex2016_2.csv',
    'sleuth_ex2015_2.csv',
    'analcatdata_japansolvent_1.csv'
]


class MLChallenge(Challenge):
    """Machine Learning Challenge class.

    The MLChallenge class rerpresents a single ``machine learning challenge`` that can be used for
    benchmark.

    Args:
        model (class):
            Class of a machine learning estimator.

        dataset (str):
            Name or path to a dataset. If it's a name it will try to read it from
            https://btb-data.s3.amazonaws.com/

        target_column (str):
            Name of the target column in the dataset.

        encode (bool):
            Either or not to encode the dataset using ``sklearn.preprocessing.OneHotEncoder``.

        model_defaults (dict):
            Dictionary with default keyword args for the model instantiation.

        make_binary (bool):
            Either or not to make the target column binary.

        tunable_hyperparameters (dict):
            Dictionary representing the tunable hyperparameters for the challenge.

        metric (callable):
            Metric function. If ``None``, then the estimator's metric function
            will be used in case there is otherwise the default that ``cross_val_score`` function
            offers will be used.
    """

    _data = None

    @classmethod
    def get_dataset_url(cls, name):
        if not name.endswith('.csv'):
            name = name + '.csv'

        return urljoin(BASE_DATASET_URL, name)

    @classmethod
    def get_available_dataset_names(cls):
        return DATASETS_BY_TIME.copy()

    @classmethod
    def get_all_challenges(cls, challenges=None):
        """Return a list containing the instance of the datasets available."""
        datasets = challenges or cls.get_available_dataset_names()
        loaded_challenges = []
        for dataset in datasets:
            try:
                loaded_challenges.append(cls(dataset))
                LOGGER.info('Dataset %s loaded', dataset)
            except Exception as ex:
                LOGGER.warn('Dataset: %s could not be loaded. Error: %s', dataset, ex)

        LOGGER.info('%s / %s datasets loaded.', len(loaded_challenges), len(datasets))

        return loaded_challenges

    def load_data(self):
        """Load ``X`` and ``y`` over which to perform fit and evaluate."""
        if os.path.isdir(self.dataset):
            X = pd.read_csv(self.dataset)

        else:
            url = self.get_dataset_url(self.dataset)
            X = pd.read_csv(url)

        y = X.pop(self.target_column)

        if self.make_binary:
            y = y.iloc[0] == y

        if self.encode:
            ohe = OneHotEncoder(categories='auto')
            X = ohe.fit_transform(X)

        return X, y

    @property
    def data(self):
        if self._data is None:
            self._data = self.load_data()

        return self._data

    def __init__(self, dataset, model=None, target_column=None,
                 encode=None, tunable_hyperparameters=None, metric=None,
                 model_defaults=None, make_binary=None, stratified=None,
                 cv_splits=5, cv_random_state=42, cv_shuffle=True, metric_args={}):

        self.model = model or self.MODEL
        self.dataset = dataset or self.DATASET
        self.target_column = target_column or self.TARGET_COLUMN
        self.model_defaults = model_defaults or self.MODEL_DEFAULTS
        self.make_binary = make_binary or self.MAKE_BINARY
        self.tunable_hyperparameters = tunable_hyperparameters or self.TUNABLE_HYPERPARAMETERS

        if metric:
            self.metric = metric
            self.metric_args = metric_args

        else:
            # Allow to either write a metric method or assign a METRIC function
            self.metric = getattr(self, 'metric', self.__class__.METRIC)
            self.metric_args = getattr(self, 'metric_args', self.__class__.METRIC_ARGS)

        self.stratified = self.STRATIFIED if stratified is None else stratified
        # self.X, self.y = self.load_data()

        self.encode = self.ENCODE if encode is None else encode
        self.scorer = make_scorer(self.metric, **self.metric_args)

        if self.stratified:
            self.cv = StratifiedKFold(
                shuffle=cv_shuffle,
                n_splits=cv_splits,
                random_state=cv_random_state
            )
        else:
            self.cv = KFold(
                shuffle=cv_shuffle,
                n_splits=cv_splits,
                random_state=cv_random_state
            )

    def get_tunable_hyperparameters(self):
        return deepcopy(self.tunable_hyperparameters)

    def evaluate(self, **hyperparams):
        """Apply cross validation to hyperparameter combination.

        Args:
            hyperparams (dict):
                A combination of ``self.tunable_hyperparams``.

        Returns:
            score (float):
                Returns the ``mean`` cross validated score.
        """
        hyperparams.update((self.model_defaults or {}))
        model = self.model(**hyperparams)
        X, y = self.data

        return cross_val_score(model, X, y, cv=self.cv, scoring=self.scorer).mean()

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.dataset)
