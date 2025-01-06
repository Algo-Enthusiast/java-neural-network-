import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);
        Scanner fileScanner;
        FileWriter writer;
        File saveFile;

        // Variables to store network configuration
        int numberOfInputNeurons = 0;
        int numberOfHiddenLayers = 0;
        int numberOfNeuronsPerHiddenLayer = 0;
        int numberOfOutputNeurons = 0;
        int numberOfLayers = 0;

        // Prompt user to create a new neural network
        System.out.println("Do you want to create a new neural network? (y/n)");

        if(scanner.next().equals("y")) {
            // Get network parameters from user
            System.out.println("Hello! This will guide you through the creation of a basic neural network.");
            System.out.println("How many input neurons do you want?");
            System.out.print("> ");
            numberOfInputNeurons = scanner.nextInt();
            System.out.println("How many hidden layers do you want?");
            System.out.print("> ");
            numberOfHiddenLayers = scanner.nextInt();
            System.out.println("How many neurons per hidden layer do you want?");
            System.out.print("> ");
            numberOfNeuronsPerHiddenLayer = scanner.nextInt();
            System.out.println("How many output neurons do you want?");
            System.out.print("> ");
            numberOfOutputNeurons = scanner.nextInt();
            System.out.println("What do you want to call the save file?");
            System.out.print("> ");
            String fileName = scanner.next();

            // Create save file for network configuration
            saveFile = new File(fileName + ".txt");
            if (saveFile.createNewFile()) {
                try {
                    // Write network parameters to save file
                    writer = new FileWriter(saveFile);
                    writer.write(String.valueOf(numberOfInputNeurons) + "\n");
                    writer.write(String.valueOf(numberOfHiddenLayers) + "\n");
                    writer.write(String.valueOf(numberOfNeuronsPerHiddenLayer) + "\n");
                    writer.write(String.valueOf(numberOfOutputNeurons) + "\n\n");
                    writer.write("add your training data below this line, do not delete or modify anything higher than this line unless you know what you are doing, check the documentation for help.");
                    writer.close();
                    System.out.println("Data saved, go to the generated file to add training data.");

                } catch (IOException error) {
                    System.out.println("An error occurred: " + error.getMessage() + " Try deleting the file.");
                }
            }
        } else {
            // Load existing network configuration from file
            System.out.println("What is the name of the save file? (case sensitive)");
            System.out.print("> ");
            saveFile = new File(scanner.next() + ".txt");
            fileScanner = new Scanner(saveFile);
            numberOfInputNeurons = Integer.parseInt(fileScanner.nextLine());
            numberOfHiddenLayers = Integer.parseInt(fileScanner.nextLine());
            numberOfNeuronsPerHiddenLayer = Integer.parseInt(fileScanner.nextLine());
            numberOfOutputNeurons = Integer.parseInt(fileScanner.nextLine());
            fileScanner.nextLine();
            System.out.println("Data loaded! \n\n\n");
        }

        // Calculate total number of layers
        numberOfLayers = 2 + numberOfHiddenLayers;

        System.out.println(" 1. Train the network \n 2. Test the network \n 3. Exit");
        System.out.print("> ");

        // Determine the maximum number of neurons in any layer
        ArrayList<Integer> numberOfNeuronsPerLayer = new ArrayList<Integer>();
        numberOfNeuronsPerLayer.add(numberOfInputNeurons);
        numberOfNeuronsPerLayer.add(numberOfNeuronsPerHiddenLayer);
        numberOfNeuronsPerLayer.add(numberOfOutputNeurons);
        int maxAmountOfNeuronsInLayers = Collections.max(numberOfNeuronsPerLayer);

        // Initialize neurons, weights, and biases arrays
        double[][] neurons = new double[numberOfLayers][maxAmountOfNeuronsInLayers];
        double[][][][] weights = generateWeights(numberOfLayers, maxAmountOfNeuronsInLayers);
        double[][] biases = generateBiases(numberOfLayers, maxAmountOfNeuronsInLayers);

        // Set initial input values
        neurons[0][0] = 0.42;
        neurons[0][1] = 1.32;

        // Calculate activation for the network
        double[][] result = calculateActivation(neurons, weights, biases, numberOfInputNeurons, numberOfOutputNeurons, numberOfNeuronsPerHiddenLayer);

        // Apply softmax function to the output layer
        double total = 0;
        for(int index = 0; index != 8; index++) {
            total += softmax(result[numberOfLayers - 1][index], result[numberOfLayers - 1]);
        }

        // Define the intended output result
        double[] intendedResult = new double[numberOfOutputNeurons];

        for (int i = 0; i < intendedResult.length; i++) {
            intendedResult[i] = 0.0;
        }

        // Calculate costs based on the results and the intended results
        double[] resultLayer = result[result.length - 1];
        double[] costs = calculateCost(resultLayer, intendedResult);
        System.out.println(intendedResult.length + " result");
        System.out.println(Arrays.toString(costs));
        System.out.println("sum of costs: " + Arrays.stream(costs).sum());
    }

    // Calculate activation for each neuron in the network
    public static double[][] calculateActivation(double[][] neurons, double[][][][] weights, double[][] biases, int numberOfInputNeurons, int numberOfOutputNeurons, int numberOfNeuronsInHiddenLayers) {
        double[][] result = neurons;
        double activation = 0;

        int layerIndex;
        int firstNeuronIndex;
        int secondNeuronIndex;

        // Calculate activations for each layer
        for(layerIndex = 1; layerIndex != neurons.length - 1; layerIndex++) {
            for(firstNeuronIndex = 0; firstNeuronIndex != numberOfNeuronsInHiddenLayers; firstNeuronIndex++) {
                for(secondNeuronIndex = 0; secondNeuronIndex != numberOfNeuronsInHiddenLayers; secondNeuronIndex++) {
                    activation += weights[layerIndex - 1][secondNeuronIndex][layerIndex][firstNeuronIndex] * neurons[layerIndex][firstNeuronIndex];
                }
                activation += biases[layerIndex][firstNeuronIndex];
                result[layerIndex][firstNeuronIndex] = sigmoid(activation);
                activation = 0;
            }
        }

        // Calculate activations for the output layer
        for(firstNeuronIndex = 0; firstNeuronIndex != numberOfOutputNeurons; firstNeuronIndex++) {
            for(secondNeuronIndex = 0; secondNeuronIndex != numberOfNeuronsInHiddenLayers; secondNeuronIndex++) {
                activation += weights[neurons.length - 2][secondNeuronIndex][neurons.length - 1][firstNeuronIndex] * neurons[neurons.length - 1][firstNeuronIndex];
            }
            activation += biases[layerIndex][firstNeuronIndex];
            result[layerIndex][firstNeuronIndex] = sigmoid(activation);
            activation = 0;
        }

        return result;
    }

    // Generate weights for the network
    public static double[][][][] generateWeights(int numberOfLayers, int maxNumberOfNeurons) {
        Random random = new Random();
        double[][][][] result = new double[numberOfLayers][maxNumberOfNeurons][numberOfLayers][maxNumberOfNeurons];

        // Initialize weights randomly
        for(int layerIndex=0; layerIndex+1 != numberOfLayers; layerIndex++) {
            for(int firstNeuronIndex = 0; firstNeuronIndex != maxNumberOfNeurons; firstNeuronIndex++) {
                for(int secondNeuronIndex = 0;secondNeuronIndex != maxNumberOfNeurons; secondNeuronIndex++) {
                    result[layerIndex][firstNeuronIndex][layerIndex + 1][secondNeuronIndex] = random.nextDouble(-3, 3);
                }
            }
        }

        return result;
    }

    // Generate biases for the network
    public static double[][] generateBiases(int numberOfLayers, int maxNumberOfNeuronsInLayers) {
        Random random = new Random();
        double[][] biases = new double[numberOfLayers][maxNumberOfNeuronsInLayers];

        // Initialize biases randomly
        for(int layerIndex = 0; layerIndex != numberOfLayers; layerIndex++) {
            for(int Index = 0; Index != maxNumberOfNeuronsInLayers; Index++)
            {
                biases[layerIndex][Index] = random.nextDouble(-10, 10);
            }
        }
        return biases;
    }

    // Calculate the cost of the network output
    public static double[] calculateCost(double[] givenResult, double[] intendedResult) {
        double[] costs = new double[givenResult.length];

        // Calculate costs
        int minLength = Math.min(costs.length, Math.min(givenResult.length, intendedResult.length));
        for (int index = 0; index < minLength; index++) {
            costs[index] += Math.pow(givenResult[index] + intendedResult[index], 2);
        }

        System.out.println(costs.length + " cost");
        return costs;
    }

    // Activation function: Sigmoid
    public static double sigmoid(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    // Activation function: Softmax
    public static double softmax(double input, double[] neuronValues) {
        double total = Arrays.stream(neuronValues).map(Math::exp).sum();
        return Math.exp(input) / total;
    }
}

class Training {

}