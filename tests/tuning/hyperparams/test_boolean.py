# -*- coding: utf-8 -*-

from unittest import TestCase

from btb.tuning.hyperparams.boolean import BooleanHyperParam


class TestBooleanHyperParam(TestCase):
    """Unit test for the class ``BooleanHyperParam``."""

    def setUp(self):
        self.instance = BooleanHyperParam()

    def test_trasnform_list(self):
        """Test that the method ``transform`` performs a normalization over a ``list`` of boolean
        values and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [True, False, True]
        values_2 = [False, False, True]

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        assert all(isinstance(res[0], int) for res in result_1.tolist())
        assert all(isinstance(res[0], int) for res in result_2.tolist())

    def test_transform_scalar(self):
        """Test that the method ``transform`` performs a normalization over a sinlge boolean
        value and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = False
        values_2 = True

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        assert all(isinstance(res[0], int) for res in result_1.tolist())
        assert all(isinstance(res[0], int) for res in result_2.tolist())

    def test_inverse_transform_list(self):
        """Test that the method ``inverse_transform`` performs a denormalization over the search
        space values from a list to the original hyperparameter space.
        """

        # setup
        values_1 = [0, 1, 0]
        values_2 = [1, 0, 1]

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        assert all(isinstance(res[0], bool) for res in result_1.tolist())
        assert all(isinstance(res[0], bool) for res in result_2.tolist())

    def test_inverse_transform_scalar(self):
        """Test that the method ``inverse_transform`` performs a denormalization over the search
        space values from a scalar to the original hyperparameter space.
        """
        # setup
        values_1 = 0
        values_2 = 1

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        assert all(isinstance(res[0], bool) for res in result_1.tolist())
        assert all(isinstance(res[0], bool) for res in result_2.tolist())

    def test_sample(self):
        """Test that the method ``sample`` returns values from the search space and not the
        original hyperparameter space.
        """

        # run
        result_1 = self.instance.sample(3)
        result_2 = self.instance.sample(1)
        result_3 = self.instance.sample(2)

        # assert
        assert len(result_1) == 3
        assert len(result_2) == 1
        assert len(result_3) == 2

        assert all(isinstance(res[0], int) for res in result_1.tolist())
        assert all(isinstance(res[0], int) for res in result_2.tolist())
        assert all(isinstance(res[0], int) for res in result_3.tolist())
