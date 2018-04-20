package neuralnet.data;


import java.util.ArrayList;

public class NeuralInputData {

    // Число входов и записей
    protected int numberOfInputs = 0;
    protected int numberOfRecords = 0;

    // Список всех входов построчно без выходов!!!
    protected ArrayList<ArrayList<Double>> data;

    public NeuralInputData(ArrayList<ArrayList<Double>> data) {
        this.numberOfRecords = data.size();
        this.numberOfInputs = data.get(0).size();
        this.data = data;
    }

    public NeuralInputData(double[][] data) {
        this.numberOfRecords = data.length;
        this.data = new ArrayList<>();
        this.numberOfInputs = data[0].length;
        for (int i = 0; i < numberOfRecords; i++) {
            this.data.add(new ArrayList<>());
            for (int j = 0; j < numberOfInputs; j++)
                this.data.get(i).add(data[i][j]);
        }
    }

    // Возвращает строчку входных данных (список)
    public ArrayList<Double> getRecordArrayList(int i) {
        return this.data.get(i);
    }

    // Возвращает строчку входных данных (массив)
    public double[] getRecordArray(int i) {
        double[] result = new double[numberOfInputs];
        for (int j = 0; j < numberOfInputs; j++)
            result[j] = this.data.get(i).get(j);
        return result;
    }

    // Возвращает всю матрицу входных данных (список)
    public ArrayList<ArrayList<Double>> getDataArrayList() {
        return this.data;
    }

    // Возвращает всю матрицу входных данных (матрица)
    public double[][] getDataArray() {
        double[][] result = new double[numberOfRecords][numberOfInputs];
        for (int i = 0; i < numberOfRecords; i++)
            for (int j = 0; j < numberOfInputs; j++)
                result[i][j] = this.data.get(i).get(j);
        return result;
    }

    // Возвращает столбец входов по одной переменной (список)
    public ArrayList<Double> getColumnDataArrayList(int i) {
        ArrayList<Double> result = new ArrayList<>();
        for (int j = 0; j < numberOfRecords; j++)
            result.add(data.get(j).get(i));
        return result;
    }

    // Возвращает столбец входов по одной переменной (массив)
    public double[] getColumn(int i) {
        double[] result = new double[numberOfRecords];
        for (int j = 0; j < numberOfRecords; j++)
            result[j] = data.get(j).get(i);
        return result;
    }

    // Печатает входы
    public void print() {
        System.out.println("Inputs:");
        for (int k = 0; k < numberOfRecords; k++) {
            System.out.print("\tInput[" + k + "]={ ");
            for (int i = 0; i < numberOfInputs; i++)
                if (i != numberOfInputs - 1)
                    System.out.print(this.data.get(k).get(i) + "\t");
                else
                    System.out.print(this.data.get(k).get(i) + " }\n");
        }
    }

    // Возвращает список средних значений по каждому столбцу входной матрицы
    public ArrayList<Double> getMeanInputData() {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < numberOfInputs; i++) {
            double r = 0;
            for (int k = 0; k < numberOfRecords; k++)
                r += data.get(k).get(i);
            result.add(r / ((double) numberOfRecords));
        }
        return result;
    }

    public int getNumberOfRecords(){
        return numberOfRecords;
    }

}
