package neuralnet.math;

public class Step implements IActivationFunction {

    private double cond = 0;

    public Step() {
    }

    public Step(double condition) {
        cond = condition;
    }

    @Override
    public double calculate(double x) {
        return x < cond ? 0 : 1;
    }

    @Override
    public double derivative(double x) {
        if (x == 0) return Double.MAX_VALUE;
        else return 0.0;
    }

    public void setCondition(double x) {
        cond = x;
    }

    @Override
    public String toString() {
        return "Step";
    }
}
