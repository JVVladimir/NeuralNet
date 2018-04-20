package neuralnet;

import neuralnet.math.IActivationFunction;
import neuralnet.math.RandomNumberGenerator;

import java.util.ArrayList;

public class Neuron {

    protected ArrayList<Double> weight;
    protected ArrayList<Double> input;
    private double output;
    private double outputBeforeActivation;
    private int numberOfInputs;
    protected double bias = 1.0;
    private IActivationFunction activationFunction;


    public Neuron(int numberOfInputs, IActivationFunction iaf) {
        this.numberOfInputs = numberOfInputs;
        activationFunction = iaf;
        weight = new ArrayList<>(numberOfInputs + 1);
        input = new ArrayList<>(numberOfInputs);
    }

    public void initWeights() {
        for (int i = 0; i <= numberOfInputs; i++) {
            double newWeight = RandomNumberGenerator.GenerateNext();
            this.weight.add(newWeight);
        }
    }

    public void calculate() {
        outputBeforeActivation = 0;
        if (numberOfInputs > 0)
            if (input != null && weight != null)
                for (int i = 0; i <= numberOfInputs; i++) {
                    outputBeforeActivation += ((i == numberOfInputs ? bias : input.get(i))
                            * weight.get(i));
                }
        output = activationFunction.calculate(outputBeforeActivation);
    }

    public void setActivationFunction(IActivationFunction activeFnc) {
        activationFunction = activeFnc;
    }

    public void setInputs(ArrayList<Double> input) {
        this.input = input;
    }

    public void updateWeight(int i, double value) throws NeuralException {
        if (i > -1 && i <= numberOfInputs)
            weight.set(i, value);
        else
            throw new NeuralException("Попытка изменения несуществующего веса!");
    }

    public ArrayList<Double> derivativeBatch(ArrayList<ArrayList<Double>> input) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            result.add(0.0);
            double outputBeforeActivation = 0.0;
            for (int j = 0; j < numberOfInputs; j++)
                outputBeforeActivation += (j == numberOfInputs ? bias : input.get(i).get(j)) * weight.get(j);
            result.set(i, activationFunction.derivative(outputBeforeActivation));
        }
        return result;
    }

    public double derivative(ArrayList<Double> input) {
        double outputBeforeActivation = 0.0;
        if (numberOfInputs > 0)
            if (weight != null)
                for (int i = 0; i <= numberOfInputs; i++)
                    outputBeforeActivation += (i == numberOfInputs ? bias : input.get(i)) * weight.get(i);
        return activationFunction.derivative(outputBeforeActivation);
    }

    public ArrayList<Double> calcBatch(ArrayList<ArrayList<Double>> input) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            result.add(0.0);
            double outputBeforeActivation = 0.0;
            for (int j = 0; j < numberOfInputs; j++)
                outputBeforeActivation += (j == numberOfInputs ? bias : input.get(i).get(j)) * weight.get(j);
            result.set(i, activationFunction.calculate(outputBeforeActivation));
        }
        return result;
    }

    public double getOutput() {
        return output;
    }

    public ArrayList<Double> getInputs() {
        return input;
    }

    public double getInput(int i) {
        return input.get(i);
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public double getWeight(int input) {
        return weight.get(input);
    }

    public void deactivateBias() {
        this.bias = 0;
    }

    public void activateBias() {
        this.bias = 1;
    }

    public double getOutputBeforeActivation() {
        return outputBeforeActivation;
    }
}
