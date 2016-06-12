import random
import subprocess
import sys


if len(sys.argv) != 3:
    print "usage: application <jar> <config>"
    sys.exit(1)


learningRateLow = 0.001
batchSizeLow = 100
epochsLow = 1
layerSizeLow = 100
windowSizeLow = 3

learningRateHigh = 0.1
batchSizeHigh = 1000
epochsHigh = 5
layerSizeHigh = 1000
windowSizeHigh = 10


# one loop
for i in range(1, 100):
    learningRate = round(random.uniform(learningRateLow, learningRateHigh), 5)
    batchSize = random.randint(batchSizeLow, batchSizeHigh)
    epochs = random.randint(epochsLow, epochsHigh)
    layerSize = random.randint(layerSizeLow, layerSizeHigh)
    windowSize = random.randint(windowSizeLow, windowSizeHigh)
    C = 1000
    outputFileName = "400cmop_classify_whole_extraction"+ "_C_" + str(C)+ "_learnRate_" + str(learningRate)+ "_batch_" + str(batchSize)+ "_epochs_" + str(epochs)+ "_layer_" + str(layerSize)+ "_window_" + str(windowSize)
    outputFile = open(outputFileName,'w')
    listPassed = ["java", "-cp", sys.argv[1], "TextClassification", sys.argv[2], str(C), str(learningRate), str(batchSize), str(epochs), str(layerSize), str(windowSize)]
    subprocess.call(listPassed, stdout=outputFile)
    outputFile.flush()
    outputFile.close()
    print i

