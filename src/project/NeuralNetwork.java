package project;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

class NeuralNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 8809951551275617589L;

    private int numLayers;
    private int[] neuronsPerLayer;
    private double learningRate;
    private double momentum;
    private boolean useLeakyReLU;
    private double dropoutRate;
    private double[][][] weights;
    private double[][] biases;
    private double[][][] weightUpdates; // For momentum
    private double[][] activations; // To store activations for each layer
    private double[][] dropouts; // To store dropout masks for each layer
    private Random rand;

    public NeuralNetwork(int numLayers, int[] neuronsPerLayer, double learningRate, double momentum, boolean useLeakyReLU, double dropoutRate) {
        this.numLayers = numLayers;
        this.neuronsPerLayer = neuronsPerLayer;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.useLeakyReLU = useLeakyReLU;
        this.dropoutRate = dropoutRate;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];
        this.weightUpdates = new double[numLayers][][];
        this.activations = new double[numLayers][];
        this.dropouts = new double[numLayers][];
        this.rand = new Random(42); // Set a fixed random seed for consistency

        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 1; i < numLayers; i++) {
            weights[i] = new double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
            biases[i] = new double[neuronsPerLayer[i]];
            weightUpdates[i] = new double[neuronsPerLayer[i]][neuronsPerLayer[i - 1]];
            dropouts[i] = new double[neuronsPerLayer[i]];

            for (int j = 0; j < neuronsPerLayer[i]; j++) {
                biases[i][j] = rand.nextGaussian() * Math.sqrt(2.0 / neuronsPerLayer[i - 1]);
                for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
                    weights[i][j][k] = rand.nextGaussian() * Math.sqrt(2.0 / neuronsPerLayer[i - 1]);
                }
            }
        }
    }

    public double[] forward(double[] input, boolean isTraining) {
        activations[0] = input; // Set input as activations for layer 0

        for (int i = 1; i < numLayers; i++) {
            double[] newActivations = new double[neuronsPerLayer[i]];
            for (int j = 0; j < neuronsPerLayer[i]; j++) {
                newActivations[j] = biases[i][j];
                for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
                    newActivations[j] += weights[i][j][k] * activations[i - 1][k];
                }
                newActivations[j] = useLeakyReLU ? (newActivations[j] > 0 ? newActivations[j] : 0.01 * newActivations[j]) : Math.max(0, newActivations[j]);
            }
            if (isTraining) {
                applyDropout(newActivations, i);
            }
            activations[i] = newActivations;
        }
        
        if(isTraining == false) {
            System.out.print("\n\n");
            System.out.println(Arrays.toString(activations[this.numLayers - 1]));
        }

        return activations[numLayers - 1];
    }

    private void applyDropout(double[] activations, int layerIndex) {
        for (int i = 0; i < activations.length; i++) {
            if (rand.nextDouble() < dropoutRate) {
                activations[i] = 0;
                dropouts[layerIndex][i] = 0;
            } else {
                dropouts[layerIndex][i] = 1 / (1 - dropoutRate);
            }
        }
    }

    public void train(double[][] inputs, double[][] targets, int batchSize, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            double totalLoss = 0;
            int correctPredictions = 0;

            int lastShownProgress = 0;

            for (int i = 0; i < inputs.length; i += batchSize) {
                int end = Math.min(i + batchSize, inputs.length);
                for (int j = i; j < end; j++) {
                    double[] output = trainSingle(inputs[j], targets[j]);
                    totalLoss += computeLoss(output, targets[j]);
                    correctPredictions += computeAccuracy(output, targets[j]);
                }
                // Display progress
                int progress = (i + batchSize) * 100 / inputs.length;

                if (progress % 20 == 0 && lastShownProgress != progress) {
                    lastShownProgress = progress;
                    //System.out.printf("Epoch %d Progress: %d%%%n", epoch + 1, progress);
                }
            }

            double averageLoss = totalLoss / inputs.length;
            double accuracy = (double) correctPredictions / inputs.length * 100;
            long endTime = System.currentTimeMillis();
            long duration = (endTime - startTime) / 1000;

            System.out.printf("Epoch %d: Loss = %.6f, Accuracy = %.2f%%, Time = %d s%n", epoch + 1, averageLoss, accuracy, duration);
        }
    }

    private double[] trainSingle(double[] input, double[] target) {
        double[] output = forward(input, true);
        double[][] deltas = new double[numLayers][];
        for (int i = numLayers - 1; i >= 1; i--) {
            deltas[i] = new double[neuronsPerLayer[i]];
            for (int j = 0; j < neuronsPerLayer[i]; j++) {
                if (i == numLayers - 1) {
                    deltas[i][j] = (output[j] - target[j]) * (useLeakyReLU ? (output[j] > 0 ? 1 : 0.01) : (output[j] > 0 ? 1 : 0));
                } else {
                    deltas[i][j] = 0;
                    for (int k = 0; k < neuronsPerLayer[i + 1]; k++) {
                        deltas[i][j] += deltas[i + 1][k] * weights[i + 1][k][j];
                    }
                    deltas[i][j] *= (useLeakyReLU ? (activations[i][j] > 0 ? 1 : 0.01) : (activations[i][j] > 0 ? 1 : 0));
                }
                deltas[i][j] *= dropouts[i][j]; // Apply dropout scaling
            }
        }

        updateWeights(deltas);
        return output;
    }

    private void updateWeights(double[][] deltas) {
        for (int i = 1; i < numLayers; i++) {
            double[] prevActivations = activations[i - 1];

            for (int j = 0; j < neuronsPerLayer[i]; j++) {
                biases[i][j] -= learningRate * deltas[i][j];
                for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
                    weightUpdates[i][j][k] = momentum * weightUpdates[i][j][k] + (1 - momentum) * deltas[i][j] * prevActivations[k];
                    weights[i][j][k] -= learningRate * weightUpdates[i][j][k];
                }
            }
        }
    }

    private double computeLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss += Math.pow(output[i] - target[i], 2);
        }
        return loss / output.length;
    }

    private int computeAccuracy(double[] output, double[] target) {
        int maxIndexOutput = 0;
        int maxIndexTarget = 0;
        for (int i = 0; i < output.length; i++) {
            if (output[i] > output[maxIndexOutput]) {
                maxIndexOutput = i;
            }
            if (target[i] > target[maxIndexTarget]) {
                maxIndexTarget = i;
            }
        }
        return maxIndexOutput == maxIndexTarget ? 1 : 0;
    }

    // Method to save the model to a file
    public void saveModel(String fileName) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("trainedModels/" + fileName))) {
            out.writeObject(this);
            System.out.println("Model saved successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load the model from a file
    public static NeuralNetwork loadModel(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream("trainedModels/" + fileName))) {
            NeuralNetwork model = (NeuralNetwork) in.readObject();
            System.out.println("Model loaded successfully.");
            return model;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}
