using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

namespace neuralNet
{
    class Network
    {
        public int NumLayers { get; }
        public List<int> Sizes { get; }
        public List<Vec> Biases { get; }
        public List<Matrix<double>> Weights { get; }
        private Func<Vec, Vec> Activation;
        private Func<Vec, Vec> ActivationPrime;

        public Network(List<int> sizes, Activation activation)
        {
            Activation = activation.GetActivation();
            ActivationPrime = activation.GetActivationPrime();

            NumLayers = sizes.Count;
            Sizes = sizes;
            Biases = sizes.Skip(1).Select(layerSize => Vec.Build.Random(layerSize)).ToList();
            Weights = sizes.Zip(sizes.Skip(1), (columns, rows) => Matrix<double>.Build.Random(rows, columns)).ToList();
        }

        public void Print()
        {
            Console.WriteLine(NumLayers);
            Console.WriteLine(String.Join(", ", Sizes));
            Console.WriteLine(String.Join("\n", Biases));
            Console.WriteLine(String.Join("\n", Weights));
        }

        public FeedForwardResult FeedForward(Vec input)
        {
            var activations = new List<Vec>();
            activations.Add(input.Clone());

            var layers = Biases.Zip(Weights, (b, w) => (biases: b, weights: w)).ToList();
            var zValues = new List<Vec>();
            foreach (var l in layers)
            {
                var z = l.weights.Multiply(activations.Last()) + l.biases;
                zValues.Add(z);
                activations.Add(Activation(z));
            }
            return (new FeedForwardResult()
            {
                Activations = activations,
                ZValues = zValues
            });
        }

        public void SGD(List<Example> trainingData, int epochs, int batchSize, double learningRate, List<Example> testData)
        {
            for (int e = 0; e < epochs; e++)
            {
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

                System.Console.WriteLine($"Epoch {e + 1} of {epochs} completed. ");
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
                var forwardResult = FeedForward(example.Image);
                var digit = ConvertActivationToDigit(forwardResult.Activations.Last());
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
            var nablaBiases = Biases.Select(b => Vec.Build.Dense(b.Count, 0)).ToList();
            var nablaWeights = Weights.Select(w => Matrix<Double>.Build.Dense(w.RowCount, w.ColumnCount, 0)).ToList();
            foreach (var example in batch)
            {
                var (deltaBiases, deltaWeights) = BackPropagate(example);
                for (int i = 0; i < NumLayers - 1; i++)
                {
                    nablaBiases[i] += deltaBiases[i];
                    nablaWeights[i] += deltaWeights[i];
                }
            }
            for (int i = 0; i < NumLayers - 1; i++)
            {
                Biases[i] -= (learningRate / batch.Count) * nablaBiases[i];
                Weights[i] -= (learningRate / batch.Count) * nablaWeights[i];
            }
        }

        private (List<Vec>, List<Matrix<Double>>) BackPropagate(Example example)
        {
            // Feed forward the input data -> output activations
            var x = example.Image;
            var y = example.Label;
            var forwardResults = FeedForward(x);
            var activations = forwardResults.Activations;
            var zValues = forwardResults.ZValues;


            var errors = Biases.Select(b => Vec.Build.Dense(b.Count, 0)).ToList();

            // Caclulate errors in output layer:
            errors[errors.Count - 1] = (activations.Last() - y).PointwiseMultiply(ActivationPrime(zValues[zValues.Count - 1]));

            // Back Propagate errors
            for (int layer = errors.Count - 2; layer >= 0; layer--)
            {
                errors[layer] = Weights[layer + 1]
                    .TransposeThisAndMultiply(errors[layer + 1])
                    .PointwiseMultiply(ActivationPrime(zValues[layer]));
            }

            var delBiases = errors;
            var delWeights = errors.Zip(activations, (e, a) =>
                e.ToColumnMatrix().Multiply(a.ToRowMatrix())
            ).ToList();

            return (delBiases, delWeights);
        }
    }

    public class FeedForwardResult
    {
        public List<Vec> Activations { get; set; }
        public List<Vec> ZValues { get; set; }
    }

    public class Layer
    {
        public Matrix Weights { get; private set; } = null;
        public Vec Biases { get; private set; } = null;
        public int NeuronCount { get; private set; }

        public Layer NextLayer { get; set; }
        public Layer PreviousLayer { get; set; }

        public Layer(int size, Layer next = null)
        {
            NeuronCount = size;

            if (next != null)
            {
                NextLayer = next;
                next.PreviousLayer = this;
                Weights = Matrix.Build.Random(next.NeuronCount, this.NeuronCount);
                Biases = Vec.Build.Random(size);
            }

        }
    }
}
