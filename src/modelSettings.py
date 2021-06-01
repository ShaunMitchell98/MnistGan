
class ModelSettings:

    def __init__(self, epochs, batchSize, bufferSize, noiseDim, numExamplesToGenerate):
        self.epochs = epochs
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.noiseDim =  noiseDim
        self.numExamplesToGenerate = numExamplesToGenerate
