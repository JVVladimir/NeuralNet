package neuralnet.math;

public interface IActivationFunction {

    double calculate(double x);

    double derivative(double outputBeforeActivation);

    enum ActivationFunctions {
        STEP, LINEAR, SIGMOID, HYPERTAN
    }
}
