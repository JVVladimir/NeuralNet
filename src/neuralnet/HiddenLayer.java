package neuralnet;

import neuralnet.math.IActivationFunction;

import java.util.ArrayList;

public class HiddenLayer extends NeuralLayer {

    public HiddenLayer(int numberOfNeurons, IActivationFunction iaf, int numberOfInputs) {
        this.numberOfNeuronsInLayer = numberOfNeurons;
        this.activeFnc = iaf;
        this.numberOfInputs = numberOfInputs;
        neurons = new ArrayList<>(numberOfNeurons);
        output = new ArrayList<>(numberOfInputs);
        init();
    }

}
