package neuralnet.math;

public class Sign implements IActivationFunction {
    @Override
    public double calculate(double x) {
        return x >= 0 ? 1 : 0;
    }

    // Не знаю пока
    @Override
    public double derivative(double outputBeforeActivation) {
        return 0;
    }
}
