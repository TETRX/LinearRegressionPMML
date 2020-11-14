# from src.model_calculations.validators.polynomial_regression_validator import PolynomialRegressionValidator
# from unittest.mock import Mock
# import unittest

# class TestPolynomialRegressionValidator(unittest.TestCase):
#     def test_basic(self):
#         training_dataset=Mock()
#         validation_dataset=Mock()
#         trainer=Mock()
#         trainer.error=Mock(return_value=1)
#         trainer.LAMBDAS_NUM=2
#         training_dataset.x=2
#         rv=PolynomialRegressionValidator(training_dataset,validation_dataset,degrees=[0,1],lambda_val=[0,1])
#         rv.get_best_hyper(trainer)
#         self.assertTrue(True)