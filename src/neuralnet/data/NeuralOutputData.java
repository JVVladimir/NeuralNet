package neuralnet.data;

import java.util.ArrayList;

public class NeuralOutputData {

    // Число выходов и записей от учителя
    protected int numberOfOutputs = 0;
    protected int numberOfRecords = 0;

    // Список выходов от учителя
    protected ArrayList<ArrayList<Double>> targetData;
    // Список выходов от сети
    protected ArrayList<ArrayList<Double>> neuralData;

    // Пока не понятно к чему это!!!
    public NeuralOutputData(int numberOfOutputs) {
        this.numberOfOutputs = numberOfOutputs;
    }

    public NeuralOutputData(ArrayList<ArrayList<Double>> data) {
        this.numberOfRecords = data.size();
        this.numberOfOutputs = data.get(0).size();
        this.targetData = data;
        this.neuralData = new ArrayList<>();
        // Заполняем список выходов сети нулями
        for (int i = 0; i < numberOfRecords; i++) {
            this.neuralData.add(new ArrayList());
            for (int j = 0; j < numberOfOutputs; j++)
                this.neuralData.get(i).add(0.0);
        }
    }

    public NeuralOutputData(double[][] data) {
        this.numberOfRecords = data.length;
        this.numberOfOutputs = data[0].length;
        this.targetData = new ArrayList<>();
        this.neuralData = new ArrayList<>();
        for (int i = 0; i < numberOfRecords; i++) {
            this.targetData.add(new ArrayList<>());
            this.neuralData.add(new ArrayList<>());
            for (int j = 0; j < numberOfOutputs; j++) {
                this.targetData.get(i).add(data[i][j]);
                this.neuralData.get(i).add(0.0);
            }
        }
    }

    // Конструктор для обучения без учителя (второй аргумент нужен только для заполнения выхода сети)
    public NeuralOutputData(int numberOfRecords, int numberOfOutputs) {
        this.numberOfRecords = numberOfRecords;
        this.numberOfOutputs = numberOfOutputs;
        this.targetData = null;
        this.neuralData = new ArrayList<>();
        for (int i = 0; i < this.numberOfRecords; i++) {
            this.neuralData.add(new ArrayList<>());
            for (int j = 0; j < this.numberOfOutputs; j++)
                this.neuralData.get(i).add(0.0);
        }
    }

    // Возвращает выходы учителя (список)
    public ArrayList<ArrayList<Double>> getTargetDataArrayList() {
        return this.targetData;
    }

    // Возвращает выходы учителя (массив)
    public double[][] getTargetDataArray() {
        double[][] result = new double[numberOfRecords][numberOfOutputs];
        for (int i = 0; i < numberOfRecords; i++)
            for (int j = 0; j < numberOfOutputs; j++)
                result[i][j] = this.targetData.get(i).get(j);
        return result;
    }

    // Возвращает выходы сети (список)
    public ArrayList<ArrayList<Double>> getNeuralDataArrayList() {
        return this.neuralData;
    }

    // Возвращает выходы сети (массив)
    public double[][] getNeuralDataArray() {
        double[][] result = new double[numberOfRecords][numberOfOutputs];
        for (int i = 0; i < numberOfRecords; i++)
            for (int j = 0; j < numberOfOutputs; j++)
                result[i][j] = this.neuralData.get(i).get(j);
        return result;
    }

    // Устанавливаем выходные данные учителя для сети (матрица)
    public void setNeuralData(double[][] data) {
        this.neuralData = new ArrayList<>();
        this.numberOfOutputs = data[0].length;
        for (int i = 0; i < numberOfRecords; i++) {
            this.neuralData.add(new ArrayList<>());
            for (int j = 0; j < numberOfOutputs; j++)
                this.neuralData.get(i).add(data[i][j]);
        }
    }

    // Устанавливаем выходные данные учителя для сети (список)
    public void setNeuralData(ArrayList<ArrayList<Double>> data) {
        this.neuralData = data;
    }

    // Устанавливаем выходные данные учителя для строки сети (список)
    public void setNeuralData(int i, ArrayList<Double> data) {
        this.neuralData.set(i, data);
    }

    // Устанавливаем выходные данные учителя для сети (массив)
    public void setNeuralData(int i, double[] data) {
        for (int j = 0; j < numberOfOutputs; j++)
            this.neuralData.get(i).set(j, data[j]);
    }

    // Возвращает конкретную выходную запись учителя (список)
    public ArrayList<Double> getTargetRecordArrayList(int i) {
        return this.targetData.get(i);
    }

    // Возвращает конкретную выходную запись учителя (массив)
    public double[] getTargetRecordArray(int i) {
        double[] result = new double[numberOfOutputs];
        for (int j = 0; j < numberOfOutputs; j++)
            result[j] = this.targetData.get(i).get(j);
        return result;
    }

    // Возвращает конкретную выходную строку сети (список)
    public ArrayList<Double> getRecordArrayList(int i) {
        return this.neuralData.get(i);
    }

    // Возвращает конкретную выходную строку сети (массив)
    public double[] getRecordArray(int i) {
        double[] result = new double[numberOfOutputs];
        for (int j = 0; j < numberOfOutputs; j++)
            result[j] = this.neuralData.get(i).get(j);
        return result;
    }

    // Возвращает конкретный выходной столбец учителя (список)
    public ArrayList<Double> getTargetColumnArrayList(int i) {
        ArrayList<Double> result = new ArrayList<>();
        for (int j = 0; j < numberOfRecords; j++)
            result.add(targetData.get(j).get(i));
        return result;
    }

    // Возвращает конкретный выходной столбец учителя (массив)
    public double[] getTargetColumnArray(int i) {
        double[] result = new double[numberOfRecords];
        for (int j = 0; j < numberOfRecords; j++)
            result[j] = targetData.get(j).get(i);
        return result;
    }

    // Возвращает конкретный выходной столбец сети (список)
    public ArrayList<Double> getNeuralColumnArrayList(int i) {
        ArrayList<Double> result = new ArrayList<>();
        for (int j = 0; j < numberOfRecords; j++)
            if (neuralData.get(j).get(i) == null)
                result.add(0.0);
            else
                result.add(neuralData.get(j).get(i));
        return result;
    }

    // Возвращает конкретный выходной столбец сети (массив)
    public double[] getNeuralColumnArray(int i) {
        double[] result = new double[numberOfRecords];
        for (int j = 0; j < numberOfRecords; j++)
            result[j] = neuralData.get(j).get(i);
        return result;
    }

    // Печатает выходы учителя
    public void printTarget() {
        System.out.println("Targets:");
        for (int k = 0; k < numberOfRecords; k++) {
            System.out.print("\tTarget Output[" + k + "]={ ");
            for (int i = 0; i < numberOfOutputs; i++)
                if (i != numberOfOutputs - 1)
                    System.out.print(this.targetData.get(k).get(i) + "\t");
                else
                    System.out.print(this.targetData.get(k).get(i) + "}\n");
        }
    }

    // Печатает выходы сети
    public void printNeural() {
        System.out.println("Neural:");
        for (int k = 0; k < numberOfRecords; k++) {
            System.out.print("\tOutput[" + k + "]={ ");
            for (int i = 0; i < numberOfOutputs; i++)
                if (i != numberOfOutputs - 1)
                    System.out.print(this.neuralData.get(k).get(i) + "\t");
                else
                    System.out.print(this.neuralData.get(k).get(i) + "}\n");
        }
    }

    // Рассчёт среднего значения выходов сети по столбцам
    public ArrayList<Double> getMeanNeuralData() {
        ArrayList<Double> result = new ArrayList<>();
        for (int j = 0; j < numberOfOutputs; j++) {
            double r = 0;
            for (int k = 0; k < numberOfRecords; k++)
                r += neuralData.get(k).get(j);
            result.add(r / ((double) numberOfRecords));
        }
        return result;
    }

    // Рассчёт среднего значения выходов учителя по столбцам
    public ArrayList<Double> getMeanTargetData() {
        ArrayList<Double> result = new ArrayList<>();
        for (int j = 0; j < numberOfOutputs; j++) {
            double r = 0;
            for (int k = 0; k < numberOfRecords; k++)
                r += targetData.get(k).get(j);
            result.add(r / ((double) numberOfRecords));
        }
        return result;
    }

    public int getNumberOfRecords(){
        return numberOfRecords;
    }
}
