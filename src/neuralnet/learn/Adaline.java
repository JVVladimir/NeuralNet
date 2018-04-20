package neuralnet.learn;


import neuralnet.*;
import neuralnet.data.NeuralDataSet;

import java.util.ArrayList;

public class Adaline extends LearningAlgorithm {

    // Список ошибок
    public ArrayList<ArrayList<Double>> error;
    // Ошибка обучения на одном наборе данных
    public ArrayList<Double> generalError;
    // Ошибка обучения на всех данных
    public ArrayList<Double> overallError;
    // Ошибка на всех данных
    public double overallGeneralError;

    // Ошибки на тестовых данных
    public ArrayList<ArrayList<Double>> testingError;
    public ArrayList<Double> testingGeneralError;
    public ArrayList<Double> testingOverallError;
    public double testingOverallGeneralError;


    public double degreeGeneralError = 2.0;
    public double degreeOverallError = 1.0;

    // Виды ошибок
    public enum ErrorMeasurement {
        SimpleError, SquareError, NDegreeError, MSE
    }

    // Разность квадратов
    public ErrorMeasurement generalErrorMeasurement = ErrorMeasurement.SquareError;
    // Среднеквадратичная ошибка
    public ErrorMeasurement overallErrorMeasurement = ErrorMeasurement.MSE;

    // Текущая запись
    private int currentRecord = 0;

    // Тензор новых весовых коэффициентов
    private ArrayList<ArrayList<ArrayList<Double>>> newWeights;

    // Создаётся тензор (трёхмерный) для хранения весов нейронов, нейронов, скрытых слоёв + выходной
    public Adaline(NeuralNet neuralNet) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network yet");
        this.learningParadigm = LearningParadigm.SUPERVISED;
        this.neuralNet = neuralNet;
        this.newWeights = new ArrayList<>();
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        int numberOfNeuronsInLayer, numberOfInputsInNeuron;
        this.newWeights.add(new ArrayList<>());
        numberOfNeuronsInLayer = this.neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
        for (int j = 0; j < numberOfNeuronsInLayer; j++) {
            numberOfInputsInNeuron = this.neuralNet.getOutputLayer().getNeuron(j).getNumberOfInputs();
            this.newWeights.get(0).add(new ArrayList<>());
            for (int i = 0; i <= numberOfInputsInNeuron; i++)
                this.newWeights.get(0).get(j).add(0.0);
        }
    }

    public Adaline(NeuralNet neuralNet, NeuralDataSet trainDataSet) throws NeuralException {
        this(neuralNet);
        this.trainingDataSet = trainDataSet;
        this.generalError = new ArrayList<>();
        this.error = new ArrayList<>();
        this.overallError = new ArrayList<>();
        for (int i = 0; i < trainDataSet.getNumberOfRecords(); i++) {
            this.generalError.add(null); // общая ошибка на одной строке данных
            this.error.add(new ArrayList<>());
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                if (i == 0)
                    this.overallError.add(null); // общая ошибка на каждый выход сети за время обучения
                this.error.get(i).add(null); // текущая ошибка для каждого входа для каждой строки данных
            }
        }
    }

    public Adaline(NeuralNet neuralNet, NeuralDataSet trainDataSet, LearningMode learningMode) throws NeuralException {
        this(neuralNet, trainDataSet);
        this.learningMode = learningMode;
    }

    // Установка тестовых данных для сети
    @Override
    public void setTestingDataSet(NeuralDataSet testingDataSet) {
        this.testingDataSet = testingDataSet;
        this.testingGeneralError = new ArrayList<>();
        this.testingError = new ArrayList<>();
        this.testingOverallError = new ArrayList<>();
        for (int i = 0; i < testingDataSet.getNumberOfRecords(); i++) {
            this.testingGeneralError.add(null);
            this.testingError.add(new ArrayList<>());
            for (int j = 0; j < this.neuralNet.getNumberOfOutputs(); j++) {
                if (i == 0)
                    this.testingOverallError.add(null);
                this.testingError.get(i).add(null);
            }
        }
    }

    // Установка способа замера общей ошибки
    public void setGeneralErrorMeasurement(ErrorMeasurement errorMeasurement) {
        switch (errorMeasurement) {
            case SimpleError:
                this.degreeGeneralError = 1;
                break;
            case SquareError:
            case MSE:
                this.degreeGeneralError = 2;
        }
        this.generalErrorMeasurement = errorMeasurement;
    }

    public void setOverallErrorMeasurement(ErrorMeasurement errorMeasurement) {
        switch (errorMeasurement) {
            case SimpleError:
                this.degreeOverallError = 1;
                break;
            case SquareError:
            case MSE:
                this.degreeOverallError = 2;
        }
        this.overallErrorMeasurement = errorMeasurement;
    }

    // Функция обучения сети
    @Override
    public void train() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            if (learningMode == LearningMode.BATCH) {
                epoch = 0;
                forward();
                if (printTraining)
                    print();
                while (epoch < MaxEpochs && overallGeneralError > MinOverallError) {
                    epoch++;
                    for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                        for (int i = 0; i <= neuralNet.getNumberOfInputs(); i++)
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                    applyNewWeights();
                    forward();
                    if (printTraining)
                        print();
                }
            } else {
                epoch = 0;
                int k = 0;
                currentRecord = 0;
                forward(k);
                if (printTraining)
                    print();
                while (epoch < MaxEpochs && overallGeneralError > MinOverallError) {
                    for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                        for (int i = 0; i <= neuralNet.getNumberOfInputs(); i++)
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                    applyNewWeights();
                    currentRecord = ++k;
                    if (k >= trainingDataSet.getNumberOfRecords()) {
                        k = 0;
                        currentRecord = 0;
                        epoch++;
                    }
                    forward(k);
                    if (printTraining)
                        print();
                }
            }
        }
    }

    // Обновление весов в сети согласно изменениям в тензоре
    public void applyNewWeights() throws NeuralException {
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer, numberOfInputsInNeuron;
            if (l < numberOfHiddenLayers) {
                HiddenLayer hl = (HiddenLayer) this.neuralNet.getHiddenLayer(l);
                numberOfNeuronsInLayer = hl.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = hl.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        hl.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            } else {
                OutputLayer ol = this.neuralNet.getOutputLayer();
                numberOfNeuronsInLayer = ol.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = ol.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        ol.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
        }
    }

    // Применяем данные для обучения, рассчитываем все виды ошибок и выход сети на трен. данные
    @Override
    public void forward() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++) {
                // Передаём первую строку входов
                neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
                neuralNet.calculate();
                // Записываем выход сети
                trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
                generalError.set(i, generalError(trainingDataSet.getArrayListTargetOutputRecord(i)
                        , trainingDataSet.getArrayListNeuralOutputRecord(i)));
                for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                    error.get(i).set(j, simpleError(trainingDataSet.getArrayListTargetOutputRecord(i).get(j)
                            , trainingDataSet.getArrayListNeuralOutputRecord(i).get(j)));
            }
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j)
                        , trainingDataSet.getIthNeuralOutputArrayList(j)));
            overallGeneralError = overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData()
                    , trainingDataSet.getArrayNeuralOutputData());
        }
    }

    // Применяем одну строку данных для обучения, рассчитываем все виды ошибок и выход сети на трен. данные
    @Override
    public void forward(int i) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            neuralNet.setInputs(trainingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            generalError.set(i, generalError(trainingDataSet.getArrayListTargetOutputRecord(i),
                    trainingDataSet.getArrayListNeuralOutputRecord(i)));
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j)
                        , trainingDataSet.getIthNeuralOutputArrayList(j)));
                error.get(i).set(j, simpleError(trainingDataSet.getIthTargetOutputArrayList(j).get(i)
                        , trainingDataSet.getIthNeuralOutputArrayList(j).get(i)));
            }
            overallGeneralError = overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData()
                    , trainingDataSet.getArrayNeuralOutputData());
        }
    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron) throws NeuralException {
        if (layer > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            double deltaWeight = LearningRate;
            Neuron currNeuron = neuralNet.getOutputLayer().getNeuron(neuron);
            switch (learningMode) {
                case BATCH:
                    ArrayList<Double> ithInput;
                    if (input < currNeuron.getNumberOfInputs())
                        ithInput = trainingDataSet.getIthInputArrayList(input);
                    else {
                        ithInput = new ArrayList<>();
                        for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++)
                            ithInput.add(1.0);
                    }
                    double multDerivResultIthInput = 0.0;
                    for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++)
                        multDerivResultIthInput += ((error.get(currentRecord).get(neuron) + currNeuron.getOutput())
                                -currNeuron.getOutputBeforeActivation()) * ithInput.get(i);
                    deltaWeight *= multDerivResultIthInput;
                    break;
                case ONLINE:
                    deltaWeight *= (error.get(currentRecord).get(neuron) + currNeuron.getOutput())
                            -currNeuron.getOutputBeforeActivation();
                    if (input < currNeuron.getNumberOfInputs())
                        deltaWeight *= neuralNet.getInput(input);
                    break;
            }
            return currNeuron.getWeight(input) + deltaWeight;
        }
    }

    @Override
    public double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException {
        if (layer > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            Neuron currNeuron = neuralNet.getOutputLayer().getNeuron(neuron);
            double deltaWeight = LearningRate * ((error + currNeuron.getOutput())
                    -currNeuron.getOutputBeforeActivation());
            switch (learningMode) {
                case BATCH:
                    ArrayList<Double> ithInput;
                    if (input < currNeuron.getNumberOfInputs())
                        ithInput = trainingDataSet.getIthInputArrayList(input);
                    else {
                        ithInput = new ArrayList<>();
                        for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++)
                            ithInput.add(1.0);
                    }
                    double multDerivResultIthInput = 0.0;
                    for (int i = 0; i < trainingDataSet.getNumberOfRecords(); i++)
                        multDerivResultIthInput += ithInput.get(i);
                    deltaWeight *= multDerivResultIthInput;
                    break;
                case ONLINE:
                    if (input < currNeuron.getNumberOfInputs())
                        deltaWeight *= neuralNet.getInput(input);
                    break;
            }
            return currNeuron.getWeight(input) + deltaWeight;
        }
    }

    // Тестирование на полном наборе данных
    @Override
    public void test() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            for (int i = 0; i < testingDataSet.getNumberOfRecords(); i++) {
                neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
                neuralNet.calculate();
                testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
                testingGeneralError.set(i, generalError(testingDataSet.getArrayListTargetOutputRecord(i)
                        , testingDataSet.getArrayListNeuralOutputRecord(i)));
                for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                    testingError.get(i).set(j, simpleError(testingDataSet.getArrayListTargetOutputRecord(i).get(j)
                            , testingDataSet.getArrayListNeuralOutputRecord(i).get(j)));
            }
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++)
                testingOverallError.set(j, overallError(testingDataSet.getIthTargetOutputArrayList(j)
                        , testingDataSet.getIthNeuralOutputArrayList(j)));
            testingOverallGeneralError = overallGeneralErrorArrayList(testingDataSet.getArrayTargetOutputData()
                    , testingDataSet.getArrayNeuralOutputData());
        }
    }

    // Тестирование на одном наборе данных
    @Override
    public void test(int i) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0)
            throw new NeuralException("Adaline can be used only with single layer neural network");
        else {
            neuralNet.setInputs(testingDataSet.getArrayListInputRecord(i));
            neuralNet.calculate();
            testingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            testingGeneralError.set(i, generalError(testingDataSet.getArrayListTargetOutputRecord(i)
                    , testingDataSet.getArrayListNeuralOutputRecord(i)));
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                testingOverallError.set(j, overallError(testingDataSet.getIthTargetOutputArrayList(j)
                        , testingDataSet.getIthNeuralOutputArrayList(j)));
                testingError.get(i).set(j, simpleError(testingDataSet.getIthTargetOutputArrayList(j).get(i)
                        , testingDataSet.getIthNeuralOutputArrayList(j).get(i)));
            }
            testingOverallGeneralError = overallGeneralErrorArrayList(testingDataSet.getArrayTargetOutputData()
                    , testingDataSet.getArrayNeuralOutputData());
        }
    }

    // Получение общей ошибки за всё время обучения (массив)
    public double overallGeneralErrorArray(double[][] YT, double[][] Y) {
        int N = YT.length;
        int Ny = YT[0].length;
        double result = 0;
        for (int i = 0; i < N; i++) {
            double resultY = 0;
            for (int j = 0; j < Ny; j++)
                resultY += Math.pow(YT[i][j] - Y[i][j], degreeGeneralError);
            if (generalErrorMeasurement == ErrorMeasurement.MSE)
                result += Math.pow((1.0 / Ny) * resultY, degreeOverallError);
            else
                result += Math.pow((1.0 / degreeGeneralError) * resultY, degreeOverallError);
        }
        return (1.0 / N) * result;
    }

    // Получение общей ошибки за всё время обучения (список)
    public double overallGeneralErrorArrayList(ArrayList<ArrayList<Double>> YT, ArrayList<ArrayList<Double>> Y) {
        int N = YT.size();
        int Ny = YT.get(0).size();
        double result = 0;
        for (int i = 0; i < N; i++) {
            double resultY = 0;
            for (int j = 0; j < Ny; j++)
                resultY += Math.pow(YT.get(i).get(j) - Y.get(i).get(j), degreeGeneralError);
            if (generalErrorMeasurement == ErrorMeasurement.MSE)
                result += Math.pow((1.0 / Ny) * resultY, degreeOverallError);
            else
                result += Math.pow((1.0 / degreeGeneralError) * resultY, degreeOverallError);
        }
        if (overallErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeOverallError);
        return result;
    }

    // Получение ошибки за одну эпоху
    public double generalError(ArrayList<Double> YT, ArrayList<Double> Y) {
        int N = YT.size();
        double result = 0.0;
        for (int i = 0; i < N; i++)
            result += Math.pow(YT.get(i) - Y.get(i), degreeGeneralError);
        if (generalErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeGeneralError);
        return result;
    }

    // Получение общей ошибки за эпоху
    public double overallError(ArrayList<Double> YT, ArrayList<Double> Y) {
        int N = YT.size();
        double result = 0;
        for (int i = 0; i < N; i++)
            result += Math.pow(YT.get(i) - Y.get(i), degreeOverallError);
        if (overallErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeOverallError);
        return result;
    }

    // Получение ошибки за одну эпоху
    public double generalError(double[] YT, double[] Y) {
        int N = YT.length;
        double result = 0.0;
        for (int i = 0; i < N; i++)
            result += Math.pow(YT[i] - Y[i], degreeGeneralError);
        if (generalErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeGeneralError);
        return result;
    }

    // Получение общей ошибки за эпоху !!!!!!!!!! деление на ноль
    public double overallError(double[] YT, double[] Y) {
        int N = YT.length;
        double result = 0;
        for (int i = 0; i < N; i++)
            result += Math.pow(YT[i] - Y[i], degreeOverallError);
        if (overallErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeOverallError); //////////
        return result;
    }

    // Ошибка, как разность
    public double simpleError(double YT, double Y) {
        return YT - Y;
    }

    // Квадратичная ошибка
    public double squareError(double YT, double Y) {
        return (1.0 / 2.0) * Math.pow(YT - Y, 2.0);
    }

    // Вывод информации о процессе обучения
    @Override
    public void print() {
        if (learningMode == LearningMode.ONLINE)
            System.out.println("Epoch = " + epoch + "; Record = " + currentRecord + "; Overall Error = "
                    + overallGeneralError);
        else
            System.out.println("Epoch = " + epoch + "; Overall Error =" + overallGeneralError);
    }

    // Вывод результатов обучения
    public void printInfo() {
        System.out.println("Overall Error: " + getOverallGeneralError());
        System.out.println("Min Overall Error: " + getMinOverallError());
        System.out.println("Epochs of training: " + getEpoch());
    }

    @Override
    public void printTestError() {
        System.out.println("Overall error on tests: "+testingOverallGeneralError);
    }

    // Получение общей ошибки на всех данных обучения
    public double getOverallGeneralError() {
        return overallGeneralError;
    }

    // Получение общей ошибки обучения за эпоху обучения
    public double getOverallError(int output) {
        return overallError.get(output);
    }

    // Получение общей ошибки на всех данных тестирования
    public double getTestingOverallGeneralError() {
        return testingOverallGeneralError;
    }

    // Получение общей ошибки на данных за эпоху тестирования
    public double getTestingOverallError(int output) {
        return testingOverallError.get(output);
    }

}
