package neuralnet.learn;

import neuralnet.*;
import neuralnet.data.NeuralDataSet;

import java.util.ArrayList;

public class Backpropogation extends DeltaRule {

    private double MomentumRate = 0.7;
    public ArrayList<ArrayList<Double>> deltaNeuron;
    public ArrayList<ArrayList<ArrayList<Double>>> lastDeltaWeights;

    public Backpropogation(NeuralNet neuralNet) {
        super(neuralNet);
        initDeltaNeuron();
        initLastDeltaWeights();
    }

    public Backpropogation(NeuralNet neuralNet, NeuralDataSet trainDataSet, LearningMode learningMode) {
        super(neuralNet, trainDataSet, learningMode);
        initDeltaNeuron();
        initLastDeltaWeights();
    }

    public Backpropogation(NeuralNet neuralNet, NeuralDataSet trainDataSet) {
        super(neuralNet, trainDataSet);
        initDeltaNeuron();
        initLastDeltaWeights();
    }

    private void initDeltaNeuron() {
        deltaNeuron = new ArrayList<>();
        int numberOfHiddenLayers = neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer;
            deltaNeuron.add(new ArrayList<Double>());
            if (l == numberOfHiddenLayers) {
                numberOfNeuronsInLayer = neuralNet.getOutputLayer()
                        .getNumberOfNeuronsInLayer();
            } else {
                numberOfNeuronsInLayer = neuralNet.getHiddenLayer(l)
                        .getNumberOfNeuronsInLayer();
            }
            for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                deltaNeuron.get(l).add(null);
            }
        }
    }

    private void initLastDeltaWeights() {
        this.lastDeltaWeights = new ArrayList<>();
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer, numberOfInputsInNeuron;
            this.lastDeltaWeights.add(new ArrayList<ArrayList<Double>>());
            if (l < numberOfHiddenLayers) {
                numberOfNeuronsInLayer = this.neuralNet.getHiddenLayer(l)
                        .getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = this.neuralNet.getHiddenLayer(l)
                            .getNeuron(j).getNumberOfInputs();
                    this.lastDeltaWeights.get(l).add(new ArrayList<Double>());
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        this.lastDeltaWeights.get(l).get(j).add(0.0);
                    }
                }
            } else {
                numberOfNeuronsInLayer = this.neuralNet.getOutputLayer()
                        .getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = this.neuralNet.getOutputLayer()
                            .getNeuron(j).getNumberOfInputs();
                    this.lastDeltaWeights.get(l).add(new ArrayList<Double>());
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        this.lastDeltaWeights.get(l).get(j).add(0.0);
                    }
                }
            }
        }

    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron) {
        double deltaWeight = calcDeltaWeight(layer, input, neuron);
        return newWeights.get(layer).get(neuron).get(input) + deltaWeight;
    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron, double error) {
        return calcNewWeight(layer, input, neuron);
    }

    public double calcDeltaWeight(int layer, int input, int neuron) {
        double deltaWeight = 1.0;
        // Для обоих видов обучения (online and batch)
        deltaWeight *= LearningRate;
        NeuralLayer currLayer;
        Neuron currNeuron;
        double deltaNeuron;
        if (layer == neuralNet.getNumberOfHiddenLayers()) {
            currLayer = neuralNet.getOutputLayer();
            currNeuron = currLayer.getNeuron(neuron);
            deltaNeuron = error.get(currentRecord).get(neuron) * currNeuron.derivative(currLayer.getInputs());
        } else {
            currLayer = neuralNet.getHiddenLayer(layer);
            currNeuron = currLayer.getNeuron(neuron);
            double sumDeltaNextLayer = 0;
            NeuralLayer nextLayer = currLayer.getNextLayer();
            for (int k = 0; k < nextLayer.getNumberOfNeuronsInLayer(); k++)
                sumDeltaNextLayer += nextLayer.getWeight(neuron, k) * this.deltaNeuron.get(layer + 1).get(k);
            deltaNeuron = sumDeltaNextLayer * currNeuron.derivative(currLayer.getInputs());
        }
        this.deltaNeuron.get(layer).set(neuron, deltaNeuron);
        deltaWeight *= deltaNeuron;
        if (input < currNeuron.getNumberOfInputs())
            deltaWeight *= currNeuron.getInput(input);
        return deltaWeight;
    }

    // Алгоритм обучения (включает в себя помимо дельта-правила, обратное распространение ошибки)
    @Override
    public void train() throws NeuralException {
        neuralNet.setNeuralNetMode(NeuralNet.NeuralNetMode.TRAINING);
        epoch = 0;
        int k = 0;
        currentRecord = 0;
        forward();
        forward(k);
        if (printTraining)
            print();
        while (epoch < MaxEpochs && overallGeneralError > MinOverallError) {
            backward();
            switch (learningMode) {
                case BATCH:
                    if (k == trainingDataSet.getNumberOfRecords() - 1)
                        applyNewWeights();
                    break;
                case ONLINE:
                    applyNewWeights();
            }
            currentRecord = ++k;
            if (k >= trainingDataSet.getNumberOfRecords()) {
                k = 0;
                currentRecord = 0;
                epoch++;
            }
            forward(k);
            if (printTraining && (learningMode == LearningMode.ONLINE || (k == 0)))
                print();
        }
        neuralNet.setNeuralNetMode(NeuralNet.NeuralNetMode.RUN);
    }

    @Override
    public void forward(int i) throws NeuralException {
        neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
        neuralNet.calculate();
        trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
        generalError.set(i, generalError(trainingDataSet.getArrayTargetOutputRecord(i)
                        , trainingDataSet.getArrayNeuralOutputRecord(i)));
        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
            overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j)
                            , trainingDataSet.getIthNeuralOutputArrayList(j)));
            error.get(i).set(j, simpleError(trainingDataSet.getIthTargetOutputArrayList(j).get(i)
                            , trainingDataSet.getIthNeuralOutputArrayList(j).get(i)));
        }
        overallGeneralError = overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData()
                , trainingDataSet.getArrayNeuralOutputData());
    }

    public void backward() {
        int numberOfLayers = neuralNet.getNumberOfHiddenLayers();
        for (int l = numberOfLayers; l >= 0; l--) {
            int numberOfNeuronsInLayer = deltaNeuron.get(l).size();
            for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                for (int i = 0; i < newWeights.get(l).get(j).size(); i++) {
                    double currNewWeight = this.newWeights.get(l).get(j).get(i);
                    if (currNewWeight == 0.0 && epoch == 0.0)
                        if (l == numberOfLayers)
                            currNewWeight = neuralNet.getOutputLayer().getWeight(i, j);
                        else
                            currNewWeight = neuralNet.getHiddenLayer(l).getWeight(i, j);
                    double deltaWeight = calcDeltaWeight(l, i, j);
                    newWeights.get(l).get(j).set(i, currNewWeight + deltaWeight);
                }
            }
        }
    }

    @Override
    public void forward() {
        for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++) {
            neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            generalError.set(i, generalError(trainingDataSet.getArrayTargetOutputRecord(i)
                            , trainingDataSet.getArrayNeuralOutputRecord(i)));
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                error.get(i).set(j, simpleError(trainingDataSet.getArrayListTargetOutputRecord(i).get(j)
                                , trainingDataSet.getArrayListNeuralOutputRecord(i).get(j)));
        }
        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
            overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j)
                            , trainingDataSet.getIthNeuralOutputArrayList(j)));
        overallGeneralError = overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData()
                , trainingDataSet.getArrayNeuralOutputData());
    }

    @Override
    public void test(int i) throws NeuralException {
        neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
        neuralNet.calculate();
        testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
        testingGeneralError.set(i, generalError(testingDataSet.getArrayTargetOutputRecord(i)
                        , testingDataSet.getArrayNeuralOutputRecord(i)));
        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
            testingOverallError.set(j, overallError(testingDataSet.getIthTargetOutputArrayList(j)
                            , testingDataSet.getIthNeuralOutputArrayList(j)));
            testingError.get(i).set(j, simpleError(testingDataSet.getIthTargetOutputArrayList(j).get(i)
                            , testingDataSet.getIthNeuralOutputArrayList(j).get(i)));
        }
        testingOverallGeneralError = overallGeneralErrorArrayList(testingDataSet.getArrayTargetOutputData()
                , testingDataSet.getArrayNeuralOutputData());
    }

    @Override
    public void test() throws NeuralException {
        for (int i = 0; i < testingDataSet.getNumberOfRecords(); i++) {
            neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            testingGeneralError.set(i, generalError(testingDataSet.getArrayTargetOutputRecord(i)
                            , testingDataSet.getArrayNeuralOutputRecord(i)));
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                testingError.get(i).set(j, simpleError(testingDataSet.getArrayListTargetOutputRecord(i).get(j)
                                , testingDataSet.getArrayListNeuralOutputRecord(i).get(j)));
        }
        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
            testingOverallError.set(j, overallError(testingDataSet.getIthTargetOutputArrayList(j)
                            , testingDataSet.getIthNeuralOutputArrayList(j)));
        testingOverallGeneralError = overallGeneralErrorArrayList(testingDataSet.getArrayTargetOutputData()
                , testingDataSet.getArrayNeuralOutputData());
    }

    @Override
    public void applyNewWeights() throws NeuralException {
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer, numberOfInputsInNeuron;
            if (l < numberOfHiddenLayers) {
                HiddenLayer hl = (HiddenLayer) this.neuralNet.getHiddenLayer(l);
                numberOfNeuronsInLayer = hl.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = hl.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double lastDeltaWeight = lastDeltaWeights.get(l).get(j).get(i);
                        double momentum = MomentumRate * lastDeltaWeight;
                        double newWeight = this.newWeights.get(l).get(j).get(i) - momentum;
                        this.newWeights.get(l).get(j).set(i, newWeight);
                        Neuron n = hl.getNeuron(j);
                        double deltaWeight = (newWeight - n.getWeight(i));
                        lastDeltaWeights.get(l).get(j).set(i, (double) deltaWeight);
                        hl.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            } else {
                OutputLayer ol = this.neuralNet.getOutputLayer();
                numberOfNeuronsInLayer = ol.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = ol.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double lastDeltaWeight = lastDeltaWeights.get(l).get(j).get(i);
                        double momentum = MomentumRate * lastDeltaWeight;
                        Neuron n = ol.getNeuron(j);
                        double newWeight = this.newWeights.get(l).get(j).get(i) + momentum;
                        this.newWeights.get(l).get(j).set(i, newWeight);
                        double deltaWeight = (newWeight - n.getWeight(i));
                        lastDeltaWeights.get(l).get(j).set(i, deltaWeight);
                        ol.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
        }
    }

    public void setMomentumRate(double momentumRate) {
        MomentumRate = momentumRate;
    }

    public double getMomentumRate() {
        return MomentumRate;
    }
}
