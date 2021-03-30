from try_gan.pipeline import Pipeline


class PipelinePipeline(Pipeline):
    def __init__(self, pipelines):
        self.i = 0
        self.pipelines = pipelines

    @property
    def current(self):
        return self.pipelines[i % len(self.pipelines)]

    def make_train(self):
        self.i += 1
        return self.current.make_train()

    def make_test(self):
        return self.current.make_test()
