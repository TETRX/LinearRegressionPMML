class DataNormalizer():
    def normalize(self, dataset):
        mean=dataset.columnwise_mean()
        std=dataset.columnwise_std()
        def norm(mean,std):
            return lambda num: (num-mean)/std
        functions=[norm(mean[i],std[i]) for i in range(dataset.x)]
        return dataset.apply_functions(functions)