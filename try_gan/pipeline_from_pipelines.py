from try_gan.pipeline import Pipeline


class PipelinePipeline(Pipeline):
    def __init__(self, pipelines):
        self.i = 0
        self.pipelines = pipelines

    @property
    def current(self):
        return self.pipelines[self.i % len(self.pipelines)]

    def move_to_next(self):
        self.i += 1

    def make_train(self):
        return self.current.make_train()

    def make_test(self):
        return self.current.make_test()

    def make_datasets(self):
        self.move_to_next()
        return Pipeline.make_datasets(self)
