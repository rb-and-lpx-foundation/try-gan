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

    def make_train(self, move_next=True):
        if move_next:
            self.move_to_next()
        return self.current.make_train()

    def make_test(self, move_next=True):
        if move_next:
            self.move_to_next()
        return self.current.make_test()

    def make_datasets(self):
        self.move_to_next()
        return self.make_train(move_next=False), self.make_test(move_next=False)

    def set_batch_size(self, batch_size):
        Pipeline.set_batch_size(self, batch_size)
        for p in self.pipelines:
            p.set_batch_size(batch_size)
