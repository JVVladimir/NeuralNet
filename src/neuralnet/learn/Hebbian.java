package neuralnet.learn;

import neuralnet.*;
import neuralnet.data.NeuralDataSet;

import java.util.ArrayList;

public class Hebbian extends LearningAlgorithm {

    private int currentRecord = 0;

    private ArrayList<ArrayList<ArrayList<Double>>> newWeights;
    private ArrayList<Double> currentOutputMean;
    private ArrayList<Double> lastOutputMean;


    public Hebbian(NeuralNet neuralNet) throws NeuralException {
        this.learningParadigm = LearningParadigm.UNSUPERVISED;
        this.neuralNet = neuralNet;
        this.newWeights = new ArrayList<>();
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer, numberOfInputsInNeuron;
            this.newWeights.add(new ArrayList<>());
            if (l < numberOfHiddenLayers) {
                numberOfNeuronsInLayer = this.neuralNet.getHiddenLayer(l).getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = this.neuralNet.getHiddenLayer(l).getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<>());
                    for (int i = 0; i <= numberOfInputsInNeuron; i++)
                        this.newWeights.get(l).get(j).add(0.0);
                }
            } else {
                numberOfNeuronsInLayer = this.neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = this.neuralNet.getOutputLayer().getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<>());
                    for (int i = 0; i <= numberOfInputsInNeuron; i++)
                        this.newWeights.get(l).get(j).add(0.0);
                }
            }
        }
    }

    public Hebbian(NeuralNet neuralNet, NeuralDataSet trainDataSet) throws NeuralException {
        this(neuralNet);
        this.trainingDataSet = trainDataSet;
        // Начальное получение средних значений входов
        forward();
    }

    public Hebbian(NeuralNet neuralNet, NeuralDataSet trainDataSet, LearningMode _learningMode) throws NeuralException {
        this(neuralNet, trainDataSet);
        this.learningMode = _learningMode;
    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron) throws NeuralException {
        if (layer > 0)
            throw new NeuralException("Hebbian can be used only with single layer neural network yet");
        else {
            double deltaWeight = LearningRate;
            Neuron currNeuron = neuralNet.getOutputLayer().getNeuron(neuron);
            switch (learningMode) {
                case BATCH:
                    ArrayList<Double> ithInput;
                    if (input < currNeuron.getNumberOfInputs()) {
                        ithInput = trainingDataSet.getIthInputArrayList(input);
                        double multResultIthInput = 0.0;
                        for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++)
                            multResultIthInput += trainingDataSet.getArrayListNeuralOutputRecord(i).get(neuron) * ithInput.get(i);
                        deltaWeight *= multResultIthInput;
                    } else
                        deltaWeight = 0;
                    break;
                case ONLINE:
                    deltaWeight *= currNeuron.getOutput();
                    if (input < currNeuron.getNumberOfInputs())
                        deltaWeight *= neuralNet.getInput(input);
                    break;
            }
            return currNeuron.getWeight(input) + deltaWeight;
        }
    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException {
        throw new NeuralException("Hebbian learning can be used only with the neuron's inputs and outputs, no error is used");
    }

    @Override
    public void train() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Hebbian learning can be used only with single layer neural network");
        else {
            switch (learningMode) {
                case BATCH:
                    epoch = 0;
                    forward();
                    if (printTraining)
                        print();
                    setLastOutputMean();
                    while (!stopCriteria()) {
                        epoch++;
                        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                            for (int i = 0; i <= neuralNet.getNumberOfInputs(); i++)
                                newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        applyNewWeights();
                        setLastOutputMean();
                        forward();
                        if (printTraining)
                            print();
                    }
                    break;
                case ONLINE:
                    epoch = 0;
                    int k = 0;
                    currentRecord = 0;
                    if (currentOutputMean.get(0) == null)
                        forward();
                    forward(k);
                    if (printTraining)
                        print();
                    setLastOutputMean();
                    while (!stopCriteria()) {
                        for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                            for (int i = 0; i <= neuralNet.getNumberOfInputs(); i++)
                                newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        applyNewWeights();
                        currentRecord = ++k;
                        if (k >= trainingDataSet.getNumberOfRecords()) {
                            k = 0;
                            setLastOutputMean();
                            currentOutputMean = trainingDataSet.getMeanNeuralOutput();
                            currentRecord = 0;
                            epoch++;
                        }
                        forward(k);
                        if (printTraining)
                            print();
                    }
                    break;
            }
        }
    }

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
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        hl.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            } else {
                OutputLayer ol = this.neuralNet.getOutputLayer();
                numberOfNeuronsInLayer = ol.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = ol.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        ol.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
        }
    }

    @Override
    public void forward(int i) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Hebbian learning can be used only with single layer neural network");
        else {
            neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
        }
    }

    @Override
    public void forward() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Hebbian learning can be used only with single layer neural network");
        else {
            for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++) {
                neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
                neuralNet.calculate();
                trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            }
            currentOutputMean = trainingDataSet.getMeanNeuralOutput();
        }
    }

    @Override
    public void test(int i) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Hebbian learning can be used only with single layer neural network");
        else {
            neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
        }
    }

    @Override
    public void test() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Hebbian learning can be used only with single layer neural network");
        else {
            for (int i = 0; i < testingDataSet.getNumberOfRecords(); i++) {
                neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
                neuralNet.calculate();
                testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            }
        }
    }

    @Override
    public void print() {
        if (learningMode == LearningMode.ONLINE)
            System.out.println("Epoch = " + epoch + "; Record=" + currentRecord);
        else
            System.out.println("Epoch = " + epoch);
    }

    @Override
    public void printTestError() throws NeuralException {
        throw new NeuralException("Hebbian learning can be used only with the neuron's inputs and outputs, no error is used");
    }

    public boolean stopCriteria() {
        boolean stop = true;
        for (int i = 0; i < currentOutputMean.size(); i++)
            if (currentOutputMean.get(i) <= lastOutputMean.get(i))
                stop = false;
        return stop || epoch >= MaxEpochs;
    }

    private void setLastOutputMean() {
        lastOutputMean = new ArrayList<>();
        for (double d : currentOutputMean)
            lastOutputMean.add(d);
    }

}
