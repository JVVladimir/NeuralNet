package neuralnet;

import neuralnet.math.IActivationFunction;

import java.util.ArrayList;

public class OutputLayer extends NeuralLayer {

    public OutputLayer(int numberOfNeurons, IActivationFunction iaf, int numberOfInputs) {
        this.numberOfNeuronsInLayer = numberOfNeurons;
        this.activeFnc = iaf;
        this.numberOfInputs = numberOfInputs;
        neurons = new ArrayList<>(numberOfNeurons);
        output = new ArrayList<>(numberOfInputs);
        init();
    }

}
