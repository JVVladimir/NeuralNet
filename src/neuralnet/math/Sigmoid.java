package neuralnet.math;

public class Sigmoid implements IActivationFunction {

    private double a = 1.0;

    public Sigmoid(double a) {
        this.a = a;
    }

    public Sigmoid() {}

    @Override
    public double calculate(double x) {
        return 1.0 / (1 + Math.exp(-a * x));
    }

    @Override
    public double derivative(double x){
        return calculate(x)*(1-calculate(x));
    }

    @Override
    public String toString() {
        return "Sigmoid";
    }
}
