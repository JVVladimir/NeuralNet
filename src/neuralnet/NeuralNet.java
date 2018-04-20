package neuralnet;

import neuralnet.math.IActivationFunction;
import neuralnet.math.RandomNumberGenerator;

import java.util.ArrayList;

public class NeuralNet {

    private InputLayer inputLayer;
    private ArrayList<HiddenLayer> hiddenLayer;
    private OutputLayer outputLayer;
    private int numberOfHiddenLayers;
    private int numberOfInputs;
    private int numberOfOutputs;
    private ArrayList<Double> input;
    private ArrayList<Double> output;
    private long seed = 0;
    private boolean isBiasActive = true;

    public enum NeuralNetMode { BUILD, TRAINING, RUN };
    private NeuralNetMode neuralNetMode = NeuralNetMode.BUILD;


    public NeuralNet(int numberOfInputs, int numberOfOutputs, int[] numberOfHiddenNeurons,
                     IActivationFunction[] hiddenAcFnc, IActivationFunction outputAcFnc) throws NeuralException {
        if (outputAcFnc == null)
            throw new NeuralException("Не задана функция активации для выхода!");
        if (numberOfHiddenNeurons == null || hiddenAcFnc == null)
            throw new NullPointerException("Ошибка в создании скрытых слоёв!");
        if (numberOfOutputs < 1 || numberOfInputs < 1)
            throw new NeuralException("Недопустимое кол-во входов или выходов!");
        if (numberOfHiddenNeurons.length != hiddenAcFnc.length)
            throw new NeuralException("Ошибка в создании скрытых слоёв!");
        RandomNumberGenerator.setSeed(seed);
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;
        this.numberOfHiddenLayers = numberOfHiddenNeurons.length;
        input = new ArrayList<>(numberOfInputs);
        hiddenLayer = new ArrayList<>(numberOfHiddenLayers);
        // Входной слой
        inputLayer = new InputLayer(this.numberOfInputs);
        // Скрытые слои
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            if (i == 0) {
                hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i],
                        hiddenAcFnc[i], inputLayer.getNumberOfInputsInLayer()));
                hiddenLayer.get(i).setPreviousLayer(inputLayer);
                inputLayer.setNextLayer(hiddenLayer.get(i));
            } else {
                hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i],
                        hiddenAcFnc[i], hiddenLayer.get(i - 1).getNumberOfNeuronsInLayer()));
                hiddenLayer.get(i - 1).setNextLayer(hiddenLayer.get(i));
                hiddenLayer.get(i).setPreviousLayer(hiddenLayer.get(i - 1));
            }
        }
        // Выходной слой
        outputLayer = new OutputLayer(this.numberOfOutputs, outputAcFnc,
                hiddenLayer.get(numberOfHiddenLayers - 1).getNumberOfNeuronsInLayer());
        hiddenLayer.get(numberOfHiddenLayers - 1).setNextLayer(outputLayer);
        outputLayer.setPreviousLayer(hiddenLayer.get(numberOfHiddenLayers - 1));
    }

    public NeuralNet(int numberOfInputs, int numberOfOutputs, IActivationFunction outputAcFnc) throws NeuralException {
        if (outputAcFnc == null)
            throw new NeuralException("Не задана функция активации для выхода!");
        if (numberOfOutputs < 1 || numberOfInputs < 1)
            throw new NeuralException("Недопустимое кол-во входов или выходов!");
        RandomNumberGenerator.setSeed(seed);
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;
        this.numberOfHiddenLayers = 0;
        input = new ArrayList<>(numberOfInputs);
        // Входной слой
        inputLayer = new InputLayer(this.numberOfInputs);
        // Выходной слой
        outputLayer = new OutputLayer(numberOfOutputs, outputAcFnc, this.numberOfInputs);
        inputLayer.setNextLayer(outputLayer);
        outputLayer.setPreviousLayer(inputLayer);
    }

    public void setData(double[] input) {
        int len = input.length;
        for (int i = 0; i < len; i++)
            this.input.add(input[i]);
    }

    public void calculate() {
        inputLayer.setInputs(input);
        inputLayer.calculate();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            HiddenLayer hiddenLayer = this.hiddenLayer.get(i);
            hiddenLayer.setInputs(hiddenLayer.getPreviousLayer().getOutputs());
            hiddenLayer.calculate();
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calculate();
        this.output = outputLayer.getOutputs();
    }

    public ArrayList<Double> getOutput() {
        return output;
    }

    public void printNet() {
        System.out.println("Structure: ");
        System.out.println("\tInputs: " + numberOfInputs);
        System.out.println("\tHiddenLayers: " + numberOfHiddenLayers);
        for (int i = 0; i < numberOfHiddenLayers; i++)
            System.out.println("\t\tHiddenLayer (neurons): " + hiddenLayer.get(i).numberOfNeuronsInLayer + "    " + hiddenLayer.get(i).activeFnc);
        System.out.println("\tOutputs: " + numberOfOutputs + "   " + outputLayer.activeFnc);
    }

    public void setSeed(long s) {
        seed = s;
    }

    public int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public NeuralLayer getHiddenLayer(int i) {
        return hiddenLayer.get(i);
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public double getInput(int input) {
        return this.input.get(input);
    }

    public ArrayList<Double> getInputs() {
        return input;
    }

    public void setInputs(ArrayList<Double> inputs) {
        this.input = inputs;
    }

    public double[] getOutputs() {
        double[] output = new double[this.output.size()];
        for (int i = 0; i < output.length; i++)
            output[i] = this.output.get(i);
        return output;
    }

    // Деактивирование весов для метода Хебба

    public void deactivateBias() {
        if (numberOfHiddenLayers > 0) {
            for (HiddenLayer hl : hiddenLayer) {
                for (Neuron n : hl.getListOfNeurons()) {
                    n.deactivateBias();
                }
            }
        }
        for (Neuron n : outputLayer.getListOfNeurons()) {
            n.deactivateBias();
        }
    }

    public void activateBias() {
        for (HiddenLayer hl : hiddenLayer) {
            for (Neuron n : hl.getListOfNeurons()) {
                n.activateBias();
            }
        }
        for (Neuron n : outputLayer.getListOfNeurons()) {
            n.activateBias();
        }
    }

    public boolean isBiasActive() {
        return isBiasActive;
    }

    public void setNeuralNetMode(NeuralNetMode neuralNetMode) {
        this.neuralNetMode = neuralNetMode;
    }

    public NeuralNetMode getNeuralNetMode() {
        return neuralNetMode;
    }

}
