class DataStandardizer():
    def normalize(self, dataset): #not standardize to be able to substitute DataNormalizer with this class (could be done more elegantly with common ancestor, but this also works)
        data_lines=dataset.data_lines()
        mins=[a for a in data_lines[0]]
        maxes=[a for a in data_lines[0]]
        for line in data_lines:
            for i in range(len(line)):
                if mins[i]>line[i]:
                    mins[i]=line[i]
                if maxes[i]<line[i]:
                    maxes[i]=line[i]
        def standardizer_func(min, max):
            def standardize(x):
                return (x-min)/(max-min)
            return standardize
        standardizer_funcs=[standardizer_func(mins[i],maxes[i]) for i in range(len(maxes))]
        standardized_dataset=dataset.apply_functions(standardizer_funcs)

        #technically already standardized, but it's gonna be easier if mean=0
        mean=standardized_dataset.columnwise_mean()
        def norm(mean):
            return lambda num: (num-mean)
        functions=[norm(mean[i]) for i in range(dataset.y)]
        return standardized_dataset.apply_functions(functions)