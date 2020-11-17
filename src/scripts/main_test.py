from ..data_processing.dataset import Dataset
from ..data_processing.data_reader import DataReader
from ..model_calculations.validators.polynomial_regression_validator import PolynomialRegressionValidator
from ..model_calculations.validators.gaussian_regression_validator import GaussianRegressionValidator
from ..model_calculations.trainers.elastic_net_regression_trainer import ElasticNetRegressionTrainer
from ..model_calculations.trainers.lasso_regression_trainer import LassoRegressionTrainer
from ..model_calculations.trainers.ridge_regression_trainer import RidgeRegressionTrainer
from ..data_processing.data_divider import DataDivider
from ..data_processing.data_normalizer import DataNormalizer
from ..data_processing.data_standardizer import DataStandardizer
from ..result_processing.result_getter import ResultGetter
from ..result_processing.csv_result_saver import CSVResultSaver

data_reader=DataReader("data/halas.data")
dataset=data_reader.read()
data_divider=DataDivider()
data_normalizer=DataNormalizer()
# data_normalizer=DataStandardizer()
dataset=data_normalizer.normalize(dataset)

training_dataset, validation_dataset, test_dataset= data_divider.divide([0.8,0.1,0.1],dataset)
# print(training_dataset.data_lines(),validation_dataset.data_lines(),test_dataset.data_lines())
# regression_validator=PolynomialRegressionValidator(training_dataset,validation_dataset,degrees=[1,2,3],lambda_val=[0])
regression_validator=GaussianRegressionValidator(training_dataset,validation_dataset,lambda_val=[0])
trainer=RidgeRegressionTrainer()
# trainer=LassoRegressionTrainer()
training_and_test=Dataset(training_dataset.data_lines()+test_dataset.data_lines())

hyper=regression_validator.get_best_hyper(trainer)

training_and_test=training_and_test.apply_functions(hyper[1])
result_getter=ResultGetter()
result=result_getter.get_results(10,trainer,data_divider,training_and_test,[8,1],hyper[0])


result_saver=CSVResultSaver('results/gauss_normal.csv')
result_saver.save(result)