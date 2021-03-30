from try_gan.pipeline import Pipeline


class SplitPipeline(Pipeline):
    def __init__(self, train_pipeline, test_pipeline):
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline

    def make_train(self):
        return self.train_pipeline.make_train()

    def make_test(self):
        return self.test_pipeline.make_test()
