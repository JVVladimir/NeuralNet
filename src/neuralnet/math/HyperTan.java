package neuralnet.math;

public class HyperTan implements IActivationFunction {

    private double a = 1.0;

    public HyperTan() {
    }

    public HyperTan(double a) {
        this.a = a;
    }

    @Override
    public double calculate(double x) {
        return (1 - Math.exp(-a * x)) / (1 + Math.exp(-a * x));
    }


    @Override
    public double derivative(double x){
        return (1.0)-Math.pow(calculate(x),2.0);
    }

    @Override
    public String toString() {
        return "HyperTan";
    }
}
