package neuralnet.data;


import java.util.ArrayList;

public class NeuralDataSet {

    // Входные и выходные данные
    protected NeuralInputData inputData;
    protected NeuralOutputData outputData;

    // Число входов и выходов
    protected int numberOfInputs;
    protected int numberOfOutputs;
    // Число записей с примерами от учителя
    protected int numberOfRecords;

    // Матрица с данными (входы, выходы) и массив с номерами столбцов, где конкретно входы, а где выходы
    public NeuralDataSet(ArrayList<ArrayList<Double>> data, int[] inputColumns, int[] outputColumns) {
        numberOfInputs = inputColumns.length;
        numberOfOutputs = outputColumns.length;
        numberOfRecords = data.size();
        ArrayList<ArrayList<Double>> inputData = new ArrayList<>();
        ArrayList<ArrayList<Double>> outputData = new ArrayList<>();
        // Вынимаем вектора входов и выходов
        for (int i = 0; i < numberOfInputs; i++)
            inputData.add(data.get(inputColumns[i]));
        for (int i = 0; i < numberOfOutputs; i++)
            outputData.add(data.get(outputColumns[i]));
        this.inputData = new NeuralInputData(inputData);
        this.outputData = new NeuralOutputData(outputData);
    }

    public NeuralDataSet(double[][] data, int[] inputColumns, int[] outputColumns) {
        numberOfInputs = inputColumns.length;
        numberOfOutputs = outputColumns.length;
        numberOfRecords = data.length;
        double[][] inputData = new double[numberOfRecords][numberOfInputs];
        double[][] outputData = new double[numberOfRecords][numberOfOutputs];
        for (int i = 0; i < numberOfInputs; i++)
            for (int j = 0; j < numberOfRecords; j++)
                inputData[j][i] = data[j][inputColumns[i]];
        for (int i = 0; i < numberOfOutputs; i++)
            for (int j = 0; j < numberOfRecords; j++)
                outputData[j][i] = data[j][outputColumns[i]];
        this.inputData = new NeuralInputData(inputData);
        this.outputData = new NeuralOutputData(outputData);
    }

    // Конструктор для обучения без учителя
    public NeuralDataSet(double[][] data, int numberOfOutputColumns) {
        numberOfInputs = data[0].length;
        numberOfOutputs = numberOfOutputColumns;
        numberOfRecords = data.length;
        double[][] inputData = data;
        this.inputData = new NeuralInputData(inputData);
        outputData = new NeuralOutputData(numberOfRecords, numberOfOutputs);
    }

    // Возвращает входные данные, которые сам же передал
    public ArrayList<ArrayList<Double>> getArrayInputData() {
        return inputData.data;
    }

    // Возвращает выходные данные, которые сам же передал
    public ArrayList<ArrayList<Double>> getArrayTargetOutputData() {
        return outputData.getTargetDataArrayList();
    }

    // Возвращает выход нейронной сети от входных данных
    public ArrayList<ArrayList<Double>> getArrayNeuralOutputData() {
        return outputData.getNeuralDataArrayList();
    }

    // Возвращает отдельную строку с примерами входов учителя (список)
    public ArrayList<Double> getArrayListInputRecord(int i) {
        return inputData.getRecordArrayList(i);
    }

    // Возвращает отдельную строку с примерами входов учителя (массив)
    public double[] getInputArrayRecord(int i) {
        return inputData.getRecordArray(i);
    }

    // Возвращает отдельную строку с примерами выходов учителя (список)
    public ArrayList<Double> getArrayListTargetOutputRecord(int i) {
        return outputData.getTargetRecordArrayList(i);
    }

    // Возвращает отдельную строку с примерами выходов учителя (массив)
    public double[] getArrayTargetOutputRecord(int i) {
        return outputData.getTargetRecordArray(i);
    }

    // Возвращает выход сети (список)
    public ArrayList<Double> getArrayListNeuralOutputRecord(int i) {
        return outputData.getRecordArrayList(i);
    }

    // Возвращает выход сети (массив)
    public double[] getArrayNeuralOutputRecord(int i) {
        return outputData.getRecordArray(i);
    }

    // Установить выходные данные сети (список)
    public void setNeuralOutput(int i, ArrayList<Double> neuralData) {
        this.outputData.setNeuralData(i, neuralData);
    }

    // Установить выходные данные сети (массив)
    public void setNeuralOutput(int i, double[] neuralData) {
        this.outputData.setNeuralData(i, neuralData);
    }

    // Возвращает примеры входов по одной переменной (список)
    public ArrayList<Double> getIthInputArrayList(int i) {
        return this.inputData.getColumnDataArrayList(i);
    }

    // Возвращает примеры входов по одной переменной (массив)
    public double[] getIthInputArray(int i) {
        return this.inputData.getColumn(i);
    }

    // Возвращает примеры выходов по одной переменной (список)
    public ArrayList<Double> getIthTargetOutputArrayList(int i) {
        return this.outputData.getTargetColumnArrayList(i);
    }

    // Возвращает примеры выходов по одной переменной (массив)
    public double[] getIthTargetOutputArray(int i) {
        return this.outputData.getTargetColumnArray(i);
    }

    // Возвращает выход сети по одной переменной (список)
    public ArrayList<Double> getIthNeuralOutputArrayList(int i) {
        return this.outputData.getNeuralColumnArrayList(i);
    }

    // Возвращает выход сети по одной переменной (массив)
    public double[] getIthNeuralOutputArray(int i) {
        return this.outputData.getNeuralColumnArray(i);
    }

    // Печатает входные данные
    public void printInput() {
        this.inputData.print();
    }

    // Печает нужные выходные данные
    public void printTargetOutput() {
        this.outputData.printTarget();
    }

    // Печатает выход сети
    public void printNeuralOutput() {
        this.outputData.printNeural();
    }

    // Возващает среднее значение входных данных
    public ArrayList<Double> getMeanInput() {
        return this.inputData.getMeanInputData();
    }

    // Возвращает среднее значение выходов сети
    public ArrayList<Double> getMeanNeuralOutput() {
        return this.outputData.getMeanNeuralData();
    }

    public int getNumberOfRecords(){
        return numberOfRecords;
    }
}
