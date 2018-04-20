package neuralnet;

import java.util.ArrayList;

public class InputLayer extends NeuralLayer {

    public InputLayer(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
        this.numberOfNeuronsInLayer = numberOfInputs;
        output = new ArrayList<>(numberOfNeuronsInLayer);
    }

    protected void init() {}

    protected void calculate() {
        for (int i = 0; i < numberOfNeuronsInLayer; i++)
            try {
                output.set(i, input.get(i));
            } catch (IndexOutOfBoundsException ex) {
                output.add(input.get(i));
            }
    }

}
