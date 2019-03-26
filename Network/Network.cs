using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;
using System.Diagnostics;

namespace NeuralNet
{
    class Network
    {
        private List<Layer> Layers = new List<Layer>();

        public Network(List<int> sizes, Activator activator, IWeightBiasInitializer defaultInitializer = null)
        {
            Layer layer = null;
            foreach (var size in Enumerable.Reverse(sizes))
            {
                layer = new Layer(size, layer, activator, defaultInitializer);
                Layers.Add(layer);
            }
            Layers.Reverse();
        }

        public void Print()
        {
            foreach (var l in Layers)
            {
                l.Print();
            }
        }

        public Vec FeedForward(Vec input)
        {
            Layers[0].Activations = input;
            foreach (Layer l in Layers.Skip(1))
            {
                l.Activate();
            }
            return Layers.Last().Activations;
        }

        public void SGD(List<Example> trainingData, int epochs, int batchSize, double learningRate, List<Example> testData)
        {
            for (int e = 0; e < epochs; e++)
            {
                var w = Stopwatch.StartNew();
                trainingData.Shuffle();
                int batchStart = 0;
                while (batchStart < trainingData.Count)
                {
                    var batch = trainingData.GetRange(batchStart, batchSize);
                    // use backPropagation to calculate changes to weights and biases for the miniBatch and update
                    UpdateMiniBatch(batch, learningRate);

                    // Clean up batch
                    batchStart += batchSize;
                }

                System.Console.WriteLine($"Epoch {e + 1} of {epochs} completed in {w.Elapsed.TotalSeconds}s");
                if (testData.Count > 0)
                {
                    EvaluateTestData(testData);
                }
            }
        }

        private void EvaluateTestData(List<Example> testData)
        {
            var successCount = 0;
            foreach (var example in testData)
            {
                var result = FeedForward(example.Image);
                var digit = ConvertActivationToDigit(result);
                if (example.Label[digit] == 1)
                {
                    successCount++;
                }
            }
            var percent = ((double)successCount / testData.Count).ToString("0.00%");
            System.Console.WriteLine($"\tTest Results: {successCount}/{testData.Count} examples correct. ({percent})");
        }

        private int ConvertActivationToDigit(Vec outputActivation)
        {
            return outputActivation.MaximumIndex();
        }

        private void UpdateMiniBatch(List<Example> batch, double learningRate)
        {            
            var nablaBiases = Layers.Skip(1).Select(l => Vec.Build.Dense(l.Biases.Count, 0)).ToList();
            var nablaWeights = Layers.Skip(1).Select(l => Matrix<Double>.Build.Dense(l.Weights.RowCount, l.Weights.ColumnCount, 0)).ToList();
            foreach (var example in batch)
            {
                Backpropagate(example);
                for (int i = 1; i < Layers.Count; i++)
                {
                    nablaBiases[i - 1] += Layers[i].DCostDBiases;
                    nablaWeights[i - 1] += Layers[i].DCostDWeights;
                }
                
            }
            for (int i = 1; i < Layers.Count; i++)
            {
                var biasChange = (learningRate / batch.Count) * nablaBiases[i - 1];
                Layers[i].Biases -= biasChange;
                var weightChange = (learningRate / batch.Count) * nablaWeights[i - 1];
                Layers[i].Weights -= weightChange;
            }
        }

        private void Backpropagate(Example example)
        {
            // Feed forward the input data -> output activations
            FeedForward(example.Image);

            // Caclulate errors in output layer:
            var last = Layers.Last();
            last.Errors = (last.Activations - example.Label).PointwiseMultiply(last.Activator.ActivatePrime(last.ZValues));

            // Back Propagate errors
            foreach (var l in Enumerable.Reverse(Layers.Skip(1)).Skip(1))  //Reverse order excluding first and last layers
            {
                l.Backpropagate();
            }
        }
    }
}