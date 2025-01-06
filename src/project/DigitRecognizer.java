package project;

import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import javax.swing.*;

public class DigitRecognizer extends JPanel {
    private static final int IMAGE_SIZE = 28; // Size of the drawing canvas
    private static final int SCALE = 10; // Scale up the canvas size for easier drawing
    private static final int CANVAS_SIZE = IMAGE_SIZE * SCALE; // Actual canvas size

    private BufferedImage image;
    private Graphics2D g2d;
    private NeuralNetwork neuralNetwork;
    private JLabel predictionLabel;

    public DigitRecognizer(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        this.image = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        this.g2d = image.createGraphics();
        this.g2d.setColor(Color.WHITE);
        this.g2d.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);

        setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
        setBackground(Color.WHITE);
        setOpaque(true);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                draw(e.getX(), e.getY());
            }
        });

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                draw(e.getX(), e.getY());
            }
        });

        JButton predictButton = new JButton("Prédire");
        predictButton.addActionListener(e -> predictDigit());

        JButton clearButton = new JButton("Vider");
        clearButton.addActionListener(e -> clearCanvas());

        predictionLabel = new JLabel("Dessine un nombre et appuis sur 'predire'");

        JFrame frame = new JFrame("Prédicteur de chiffre");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(this, BorderLayout.CENTER);

        JPanel bottomPanel = new JPanel();
        bottomPanel.setLayout(new FlowLayout());
        bottomPanel.add(predictButton);
        bottomPanel.add(clearButton);

        frame.add(bottomPanel, BorderLayout.SOUTH);
        frame.add(predictionLabel, BorderLayout.NORTH);
        frame.pack();
        frame.setVisible(true);
    }

    private void draw(int x, int y) {
        g2d.setColor(Color.BLACK);
        g2d.fillRect(x / SCALE, y / SCALE, 1, 1);
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(image, 0, 0, CANVAS_SIZE, CANVAS_SIZE, null);
    }

    private void clearCanvas() {
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
        repaint();
        predictionLabel.setText("Dessine un nombre et appuis sur 'prédire'");
    }

    private void predictDigit() {
        double[] input = new double[IMAGE_SIZE * IMAGE_SIZE];
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                input[i * IMAGE_SIZE + j] = (255 - image.getRGB(j, i) & 0xFF) / 255.0;
            }
        }
        double[] output = neuralNetwork.forward(input, false);
        int predictedDigit = getMaxIndexValue(output);
        predictionLabel.setText("Chiffre prédit: " + predictedDigit);
    }

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

            // Launch the drawing canvas
            new DigitRecognizer(neuralNetwork);
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
