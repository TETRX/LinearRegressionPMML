class ResultGetter:
    NUM_OF_RESULTS=100
    def get_results(self,iterations,trainer,data_divider,dataset,proportions, lambdas):
        results={}
        for j in range(ResultGetter.NUM_OF_RESULTS):
            j+=1
            results[j/ResultGetter.NUM_OF_RESULTS]=0
        for i in range(iterations):
            training_dataset, testing_dataset=data_divider.divide(proportions,dataset)
            for j in range(ResultGetter.NUM_OF_RESULTS):
                j+=1
                fraction=data_divider.get_first([j,ResultGetter.NUM_OF_RESULTS-j],training_dataset)
                theta=trainer.train(fraction,lambdas)
                results[j/ResultGetter.NUM_OF_RESULTS]+=trainer.cost(testing_dataset,theta)
        for j in range(ResultGetter.NUM_OF_RESULTS):
            j+=1
            results[j/ResultGetter.NUM_OF_RESULTS]/=iterations
        return results