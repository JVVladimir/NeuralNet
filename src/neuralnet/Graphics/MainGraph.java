package neuralnet.Graphics;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Group;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.paint.ImagePattern;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class MainGraph extends Application {

    private static ArrayList<Double> Ytarget;
    private static ArrayList<Double> Ynet;
    private static ArrayList<Double> X;
    private static int epochs;
    private static ArrayList<Double> errors;
    private static boolean isTest = false;

    public void setTestData(ArrayList<Double> Yt, ArrayList<Double> Ynet2, ArrayList<Double> X2) {
        Ytarget = Yt;
        Ynet = Ynet2;
        X = X2;
        isTest = true;
    }

    public void setTestData(double[] Yt, double[] Ynet2, double[] X2) {
        ArrayList<Double> yt = new ArrayList<>();
        ArrayList<Double> ynet2 = new ArrayList<>();
        ArrayList<Double> x2 = new ArrayList<>();
        for(int i = 0; i < Yt.length; i++) {
            yt.add(Yt[i]);
            ynet2.add(Ynet2[i]);
            x2.add(X2[i]);
        }
        Ytarget = yt;
        Ynet = ynet2;
        X = x2;
        isTest = true;
    }

    public void setDataError(ArrayList<Double> error, int epoch) {
        epochs = epoch;
        errors = error;
    }

    public void show() {
        launch();
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Оценка результатов");
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        final LineChart<Number, Number> lineChart =
                new LineChart<>(xAxis, yAxis);
        lineChart.setAnimated(false);
        lineChart.setCreateSymbols(false);
        xAxis.setTickUnit(1);
        yAxis.setTickUnit(0.5);
        if(isTest) {
            XYChart.Series series = new XYChart.Series();
            XYChart.Series series2 = new XYChart.Series();
            xAxis.setLabel("Входные значения");
            series.setName("Нейронная сеть");
            series2.setName("Учитель");
            for (int i = 0; i < X.size(); i++) {
                series.getData().add(new XYChart.Data(X.get(i), Ynet.get(i)));
                series2.getData().add(new XYChart.Data(X.get(i), Ytarget.get(i)));
            }
            lineChart.getData().addAll(series, series2);
        }else {
            XYChart.Series series = new XYChart.Series();
            xAxis.setLabel("Номер эпохи");
            series.setName("Ошибка");
            for (int i = 0; i < epochs; i++)
                series.getData().add(new XYChart.Data(i, errors.get(i)));
            lineChart.getData().addAll(series);
        }
        Scene scene = new Scene(lineChart, 800, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

}
