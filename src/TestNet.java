import neuralnet.Graphics.MainGraph;
import neuralnet.NeuralException;
import neuralnet.NeuralNet;
import neuralnet.data.NeuralDataSet;
import neuralnet.learn.*;
import neuralnet.math.*;

import java.util.ArrayList;
import java.util.Arrays;

public class TestNet {


    public static void main(String[] args) throws NeuralException {
        int n = 1000;
        double[][] train = new double[n][2];
        double[][] test = new double[n][2];
        ArrayList<Double> xlist = new ArrayList<>();
        double x = -10;
        for (int i = 0; i < n; i++, x += 0.1) {
            train[i][0] = x;
            train[i][1] = F(x);
            test[i][0] = x;
            test[i][1] = F(x);
            xlist.add(x);
        }
        int numbersOfInput = 1;
        int numdersOfOutput = 1;
        int[] numbersOfHiddenNeurons = {15};
        IActivationFunction[] hiddenActFcn = {new Sigmoid()};
        IActivationFunction outputActFcn = new Linear(1.0);
        NeuralNet net = new NeuralNet(numbersOfInput, numdersOfOutput, numbersOfHiddenNeurons, hiddenActFcn, new Linear());
        // net.printNet();
        NeuralDataSet dataTrain = new NeuralDataSet(train, new int[]{0}, new int[]{1});
        NeuralDataSet dataTest = new NeuralDataSet(test, new int[]{0}, new int[]{1});
        Backpropogation prop = new Backpropogation(net, dataTrain, LearningAlgorithm.LearningMode.ONLINE);
        prop.setMomentumRate(0.7);
        prop.setLearningRate(0.0001);
        prop.setMinOverallError(0.00001);
        prop.setPrintTraining(true);
        prop.setMaxEpochs(250);
        prop.train();
        prop.printInfo();
        //prop.setTestingDataSet(dataTest);
        //prop.test();
        //dataTest.printTargetOutput();
        //dataTest.printNeuralOutput();
        //printData(dataTrain);
        MainGraph m = new MainGraph();
        m.setTestData(dataTrain.getIthTargetOutputArrayList(0), dataTrain.getIthNeuralOutputArrayList(0), xlist);
        m.show();
    }

    private static void printData(NeuralDataSet dataSet) {
        ArrayList<Double> dnet = dataSet.getIthNeuralOutputArrayList(0);
        ArrayList<Double> dt = dataSet.getIthTargetOutputArrayList(0);
        for(int i = 0; i < dnet.size(); i++) {
            System.out.println(dnet.get(i) + "    " + dt.get(i));
        }
    }

    public static double F(double x) {
        return x*Math.sin(0.5*x)*Math.cos(x/0.5);
    }

    /*public static void main(String[] args) throws NeuralException {
        int numbersOfInput = 2;
        int numdersOfOutput = 1;
        int[] numbersOfHiddenNeurons = {3,5,2};
        IActivationFunction[] hiddenActFcn = {new Sigmoid(), new Linear(), new HyperTan()};
        IActivationFunction outputActFcn = new Linear(1.0);
        System.out.println("*****Создание нейронной сети*****");
        NeuralNet net = new NeuralNet(numbersOfInput, numdersOfOutput, numbersOfHiddenNeurons, hiddenActFcn, outputActFcn);
        net.printNet();
        double[] neuralInput = {1.5, 0.5};
        net.setData(neuralInput);
        net.calculate();
        System.out.println("Input: " + Arrays.toString(neuralInput));
        System.out.println("Output: " + net.getOutput());
    }*/
    /*public static void main(String[] args) throws NeuralException {
        RandomNumberGenerator.s = 0;
        int numberOfInputs = 2;
        int numberOfOutputs = 1;
        Sigmoid htAcFnc = new Sigmoid(1);
        HyperTan h = new HyperTan(0.85);
        NeuralNet nn = new NeuralNet(numberOfInputs, numberOfOutputs, htAcFnc);
        nn.deactivateBias();
        nn.printNet();
        double[][] _neuralDataSet = {
                {1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {1, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
                ,{1, 1}
                , {1, 0}
                , {0, 1}
                , {0, 0}
        };
        NeuralDataSet neuralDataSet = new NeuralDataSet(_neuralDataSet, numberOfOutputs);
        System.out.println("Dataset created");
        neuralDataSet.printInput();
        Hebbian deltaRule = new Hebbian(nn, neuralDataSet, LearningAlgorithm.LearningMode.ONLINE);
        deltaRule.setPrintTraining(true);
        deltaRule.setLearningRate(0.2);
        deltaRule.setMaxEpochs(100);
        try {
            System.out.println("Beginning training");
            deltaRule.train();
            System.out.println("End of training");
            System.out.println("Epochs of training: " + deltaRule.getEpoch());
            System.out.println("Neural Output after training:");
            deltaRule.forward();
            neuralDataSet.printNeuralOutput();
            double[][] _testDataSet = {
                    {1, 1}
                    , {1, 0}
                    , {0, 1}
                    , {0, 0}
            };
            NeuralDataSet testDataSet = new NeuralDataSet(_testDataSet, numberOfOutputs);
            deltaRule.setTestingDataSet(testDataSet);
            deltaRule.test();
            testDataSet.printNeuralOutput();
        } catch (NeuralException ne) { }
    }*/

    static double[][] dataTrain = new double[20][2];
    static double[][] dataTest = new double[10][2];

    public static void f() {
        double k = -1;
        for (int i = 0; i < 20; i++) {
            dataTrain[i][0] = k;
            dataTrain[i][1] = fncTest(k);
            k += 0.1;
        }
    }

    public static void f2() {
        double k = -1;
        for (int i = 0; i < 10; i++) {
            dataTest[i][0] = k;
            dataTest[i][1] = fncTest(k);
            k += 0.2;
        }
    }

  /*  public static void main(String[] args) throws NeuralException {
        f();
        f2();
        RandomNumberGenerator.s = 0;
        int numberOfInputs = 1;
        int numberOfOutputs = 1;
        Linear l = new Linear();
        Sigmoid s = new Sigmoid(0.85);
        HyperTan h = new HyperTan(0.85);
        NeuralNet nn = new NeuralNet(numberOfInputs, numberOfOutputs, l);
        nn.printNet();
        int[] inputColumns = {0};
        int[] outputColumns = {1};
        NeuralDataSet neuralDataSet = new NeuralDataSet(dataTrain, inputColumns, outputColumns);
        System.out.println("Dataset created");
        neuralDataSet.printInput();
        neuralDataSet.printTargetOutput();
        DeltaRule deltaRule = new DeltaRule(nn, neuralDataSet, LearningAlgorithm.LearningMode.ONLINE);
        deltaRule.setPrintTraining(true);
        deltaRule.setLearningRate(0.3);
        deltaRule.setMaxEpochs(4);
        deltaRule.setGeneralErrorMeasurement(DeltaRule.ErrorMeasurement.SimpleError);
        deltaRule.setOverallErrorMeasurement(DeltaRule.ErrorMeasurement.MSE);
        deltaRule.setMinOverallError(0.0001);
        try {
            System.out.println("Beginning training");
            deltaRule.train();
            System.out.println("End of training");
            deltaRule.printInfo();
            /*MainGraph mainGraph = new MainGraph();
            mainGraph.setDataError(deltaRule.generalError, deltaRule.getEpoch());
            mainGraph.show();
            // System.out.println("Target Outputs:");
            // neuralDataSet.printTargetOutput();
            NeuralDataSet testDataSet = new NeuralDataSet(dataTest, inputColumns, outputColumns);
            deltaRule.setTestingDataSet(testDataSet);
            deltaRule.test();
            testDataSet.printNeuralOutput();
            testDataSet.printTargetOutput();
            deltaRule.printTestError();
            MainGraph mainGraph2 = new MainGraph();
            mainGraph2.setTestData(testDataSet.getIthTargetOutputArrayList(0),
                    testDataSet.getIthNeuralOutputArrayList(0),testDataSet.getIthInputArrayList(0));
            mainGraph2.show();
        } catch (NeuralException ne) {
        }
    }*/

    public static double fncTest(double x) {
        return 5.4 * x + 2;
    }

}