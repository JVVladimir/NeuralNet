package neuralnet.learn;

import neuralnet.NeuralException;
import neuralnet.NeuralNet;
import neuralnet.data.NeuralDataSet;

public abstract class LearningAlgorithm {

    // Нейронная сеть для обучения
    protected NeuralNet neuralNet;

    // Виды обучения
    public enum LearningMode {
        ONLINE, BATCH
    }

    ;

    // Парадигмы обучения
    protected enum LearningParadigm {
        SUPERVISED, UNSUPERVISED
    }

    ;
    protected LearningMode learningMode;
    protected LearningParadigm learningParadigm;
    // Максимальное число эпох обучения
    protected int MaxEpochs = 100;
    protected int epoch = 0;
    // Минимальная общая ошибка
    protected double MinOverallError = 0.001;
    // Скорость обучения сети
    protected double LearningRate = 0.1;
    // Данные для тренировки сети
    protected NeuralDataSet trainingDataSet;
    // Данные для тестирования сети
    protected NeuralDataSet testingDataSet;
    // Данные для проверки сети
    protected NeuralDataSet validatingDataSet;
    // Флаг для печатания процесса обучения сети
    protected boolean printTraining = false;

    // Функция тренировки
    public abstract void train() throws NeuralException;

    public abstract void forward() throws NeuralException;

    public abstract void forward(int i) throws NeuralException;

    // Рассчёт новых весов
    public abstract double calcNewWeight(int layer, int input, int neuron) throws NeuralException;

    // Рассчёт новых весов с ошибкой
    public abstract double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException;

    // Тестирование сети
    public abstract void test() throws NeuralException;

    // Тестировние сети на определённых данных
    public abstract void test(int i) throws NeuralException;

    // Печатает информацию о процессе обучения
    public abstract void print();

    public abstract void printTestError() throws NeuralException;

    // Установка максимального числа эпох
    public void setMaxEpochs(int maxEpochs) {
        this.MaxEpochs = maxEpochs;
    }

    // Возвращает максимальное число эпох
    public int getMaxEpochs() {
        return this.MaxEpochs;
    }

    // Получает текущую эпоху обучения
    public int getEpoch() {
        return epoch;
    }

    // Установка минимальной общей ошибки обучения
    public void setMinOverallError(double minOverallError) {
        this.MinOverallError = minOverallError;
    }

    // Возвращает минимальную общую ошибку обучения
    public double getMinOverallError() {
        return this.MinOverallError;
    }

    // Установка скорости обучения
    public void setLearningRate(double learningRate) {
        this.LearningRate = learningRate;
    }

    // Возвращает скорость обучения
    public double getLearningDate() {
        return this.LearningRate;
    }

    // Установка вида обучения
    public void setLearningMode(LearningMode learningMode) {
        this.learningMode = learningMode;
    }

    // Возвращает вид обучения
    public LearningMode getLearningMode() {
        return this.learningMode;
    }

    // Установка тестовых данных для сети
    public void setTestingDataSet(NeuralDataSet testingDataSet) {
        this.testingDataSet = testingDataSet;
    }

    // Установка тестовых данных для сети
    public void setValidatingDataSet(NeuralDataSet validatingDataSet) {
        this.validatingDataSet = validatingDataSet;
    }

    // Установка вывода информации о процессе обучения
    public void setPrintTraining(boolean flag) {
        printTraining = flag;
    }

    public void setTrainingDataSet(NeuralDataSet trainingDataSet) { this.trainingDataSet = trainingDataSet; }
}
