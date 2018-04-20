package neuralnet;

import neuralnet.math.IActivationFunction;

import java.util.ArrayList;

public abstract class NeuralLayer {

    protected int numberOfNeuronsInLayer;
    protected ArrayList<Neuron> neurons = null;
    protected IActivationFunction activeFnc = null;
    protected NeuralLayer prevLayer = null;
    protected NeuralLayer nextLayer = null;
    protected ArrayList<Double> input = null;
    protected ArrayList<Double> output = null;
    protected int numberOfInputs;

    protected void init() {
        for (int i = 0; i < numberOfNeuronsInLayer; i++) {
            neurons.add(new Neuron(numberOfInputs, activeFnc));
            neurons.get(i).initWeights();
        }
    }

    protected void calculate() {
        for (int i = 0; i < numberOfNeuronsInLayer; i++) {
            neurons.get(i).setInputs(this.input);
            neurons.get(i).calculate();
            try {
                output.set(i, neurons.get(i).getOutput());
            } catch (IndexOutOfBoundsException ex) {
                output.add(neurons.get(i).getOutput());
            }
        }
    }

    public void setInputs(ArrayList<Double> input) {
        this.input = input;
    }

    public ArrayList<Double> getInputs() {
        return input;
    }

    public int getNumberOfInputsInLayer() {
        return numberOfInputs;
    }

    public void setNextLayer(NeuralLayer layer) {
        nextLayer = layer;
    }

    public NeuralLayer getNextLayer() {return nextLayer;}

    public int getNumberOfNeuronsInLayer() {
        return numberOfNeuronsInLayer;
    }

    public void setPreviousLayer(NeuralLayer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public NeuralLayer getPreviousLayer() {
        return prevLayer;
    }

    public ArrayList<Double> getOutputs() {
        return output;
    }

    public Neuron getNeuron(int j) {
        return neurons.get(j);
    }

    public double getWeight(int i, int i1) {
        return neurons.get(i1).weight.get(i);
    }

    public Neuron[] getListOfNeurons() {
        Neuron[] neurons = new Neuron[numberOfNeuronsInLayer];
        for(int i = 0; i < numberOfNeuronsInLayer; i++)
            neurons[i] = this.neurons.get(i);
        return neurons;
    }
}
