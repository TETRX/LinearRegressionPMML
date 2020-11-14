from ..data_processing.data_reader import DataReader
from ..model_calculations.validators.polynomial_regression_validator import PolynomialRegressionValidator
from ..model_calculations.trainers.lasso_regression_trainer import LassoRegressionTrainer
from ..data_processing.data_divider import DataDivider
from ..data_processing.data_normalizer import DataNormalizer
data_reader=DataReader("data/halas.data")
dataset=data_reader.read()
data_divider=DataDivider()
data_normalizer=DataNormalizer()
data_normalizer.normalize(dataset)
training_dataset, validation_dataset, test_dataset= data_divider.divide([0.8,0.1,0.1],dataset)
# print(training_dataset.data_lines(),validation_dataset.data_lines(),test_dataset.data_lines())
regression_validator=PolynomialRegressionValidator(training_dataset,validation_dataset)
trainer=LassoRegressionTrainer()

regression_validator.get_best_hyper(trainer)