package neuralnet.math;

public class Linear implements IActivationFunction {

    private double a = 1.0;

    public Linear(double a) {
        this.a = a;
    }

    public Linear() { }

    @Override
    public double calculate(double x) {
        return a * x;
    }

    @Override
    public double derivative(double x){
        return a;
    }

    @Override
    public String toString() {
        return "Linear";
    }
}
