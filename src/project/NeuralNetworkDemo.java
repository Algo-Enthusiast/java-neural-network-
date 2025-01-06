package project;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class NeuralNetworkDemo {
    public static double[][] readDataFromFile(String filename) {
        List<double[]> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("{") && line.endsWith("},")) {
                    line = line.substring(1, line.length() - 2).trim();
                    String[] values = line.split(",");
                    double[] data = new double[values.length];
                    for (int i = 0; i < values.length; i++) {
                        data[i] = Double.parseDouble(values[i].trim());
                    }
                    dataList.add(data);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataList.toArray(new double[0][]);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        NeuralNetwork neuralNetwork = null;

        System.out.println("Do you want to load an existing model? (yes/no)");
        String loadModel = scanner.nextLine();

        if (loadModel.equalsIgnoreCase("yes")) {
            System.out.print("Enter the model file name: ");
            String modelFileName = scanner.nextLine();
            neuralNetwork = NeuralNetwork.loadModel(modelFileName);
            if (neuralNetwork == null) {
                System.out.println("Failed to load model. Exiting.");
                return;
            }
        } else {
            // Define the network architecture
            int numLayers = 4;
            int[] neuronsPerLayer = {784, 128, 64, 10};
            double learningRate = 0.01;
            double momentum = 0.9;
            boolean useLeakyReLU = true;

            // Create a new neural network
            neuralNetwork = new NeuralNetwork(numLayers, neuronsPerLayer, learningRate, momentum, useLeakyReLU, 0.3);
        }

        // Read training data from file
        double[][] inputs = readDataFromFile("datasets/digit_recognition/trainingData.txt");
        double[][] targets = readDataFromFile("datasets/digit_recognition/trainingLabels.txt");

        // Read testing data from file
        double[][] tests = readDataFromFile("datasets/digit_recognition/testingData.txt");

        // Train the neural network
        System.out.print("Enter the number of epochs: ");
        int epochs = scanner.nextInt();
        int batchSize = 16;
        neuralNetwork.train(inputs, targets, batchSize, epochs);

        // Save the model after training
        System.out.print("Enter the file name to save the model: ");
        scanner.nextLine(); // Consume newline left-over
        String saveModelFileName = scanner.nextLine();
        neuralNetwork.saveModel(saveModelFileName);

        // Example prediction
        for (int i = 0; i < tests.length; i++) {
            System.out.println("Sample " + i + ": " + getMaxIndexValue(neuralNetwork.forward(tests[i], false)));
        }

        scanner.close();
    }

    public static int getMaxIndexValue(double[] array) {
        int maxIndex = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
