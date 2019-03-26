using System;
using System.Collections.Generic;
using NeuralNet;
using Xunit;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

namespace NetworkTests
{
    public class LayerTest
    {
        [Fact]
        public void HasLayerLinks()
        {
            Layer last = new Layer(2);
            Layer middle = new Layer(2, last);
            Layer first = new Layer(2, middle);

            Assert.Same(middle.NextLayer, last);
            Assert.Same(middle.PreviousLayer, first);
            Assert.Null(last.NextLayer);
            Assert.Null(first.PreviousLayer);
        }

        [Fact]
        public void HasNeuronCount()
        {
            int size = 10;
            var l = new Layer(size, null);
            Assert.Equal(l.NeuronCount, size);
        }

        [Fact]
        public void InitializesWeights()
        {
            var last = new Layer(10);
            var first = new Layer(2, last);

            Assert.Null(first.Weights);
            Assert.NotNull(last.Weights);
            Assert.Equal(last.Weights.ColumnCount, last.PreviousLayer.NeuronCount);
            Assert.Equal(last.Weights.RowCount, last.NeuronCount);
        }

        [Fact]
        public void InitializesBiases()
        {
            var last = new Layer(10);
            var first = new Layer(2, last);

            Assert.Null(first.Biases);
            Assert.NotNull(last.Biases);
            Assert.Equal(last.Biases.Count, last.NeuronCount);
        }

        [Fact]
        public void CalculatesZWithoutActivator()
        {
            var last = new Layer(3);
            var first = new Layer(2, last);

            last.Weights = Matrix.Build.DenseOfArray(new double[3, 2] {
                {1, 2},
                {1.3, 0.1},
                {10, -5},
            });

            last.Biases = Vec.Build.DenseOfArray(new double[] { 100, 200, 300 });

            var input = Vec.Build.DenseOfArray(new double[] { 1, 10 });
            var expectedResult = Vec.Build.DenseOfArray(new double[] { 121, 202.3, 260 });

            var result = last.Activate(input);
            Assert.Equal(result, expectedResult);
        }

        [Fact]
        public void CalculatesActivationsWithActivator()
        {
            var activator = new Sigmoid();
            var last = new Layer(3, null, activator);
            var first = new Layer(2, last);

            last.Weights = Matrix.Build.DenseOfArray(new double[3, 2] {
                {1, 2},
                {1.3, 0.1},
                {10, -5},
            });

            last.Biases = Vec.Build.DenseOfArray(new double[] { 100, 200, 300 });

            var input = Vec.Build.DenseOfArray(new double[] { 1, 10 });
            var expectedZ = Vec.Build.DenseOfArray(new double[] { 121, 202.3, 260 });
            var expectedActivation = activator.Activate(expectedZ);

            var result = last.Activate(input);
            Assert.Equal(result, expectedActivation);
        }

        [Fact]
        public void CalculatesActivationsFromPreviousLayer()
        {
            var last = new Layer(3);
            var middle = new Layer(3, last);
            var first = new Layer(2, middle);

            middle.Weights = Matrix.Build.DenseOfArray(new double[3, 2] {
                {1, 2},
                {1.3, 0.1},
                {10, -5},
            });
            middle.Biases = Vec.Build.DenseOfArray(new double[] { 100, 200, 300 });
            last.Weights = Matrix.Build.DenseIdentity(3);
            last.Biases = Vec.Build.Dense(3, 10);

            first.Activations = Vec.Build.DenseOfArray(new double[] { 1, 10 });
            middle.Activate();
            var result = last.Activate();

            var expected = Vec.Build.DenseOfArray(new double[] { 131, 212.3, 270 });
            Assert.Equal(expected, result);
        }

        [Fact]
        public void BackpropagatesErrors()
        {
            var last = new Layer(3);
            var middle = new Layer(3, last, new Sigmoid());
            var first = new Layer(3, middle);

            last.Errors = Vec.Build.DenseOfArray(new double[] { -0.2, 0.1, 0.5 });
            last.Weights = Matrix.Build.DenseOfArray(new double[3, 3] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
            });
            last.Biases = Vec.Build.Dense(3, 10);

            middle.Weights = Matrix.Build.DenseIdentity(3, 3);
            middle.Biases = Vec.Build.Dense(3, 0);

            first.Activations = Vec.Build.DenseOfArray(new double[] { 1, 2, 3 });
            middle.Activate();
            var sigmaPrimeOfZs = new Sigmoid().ActivatePrime(Vec.Build.DenseOfArray(new double[] { 1, 2, 3 }));
            var expectedErrors = Vec.Build.DenseOfArray(new double[] { 3.7, 4.1, 4.5 }).PointwiseMultiply(sigmaPrimeOfZs);

            var errors = middle.Backpropagate();
            Assert.Equal(expectedErrors, errors);
        }

        [Fact]
        public void CalculatesDCostDBiases()
        {
            var last = new Layer(3);
            var first = new Layer(2, last);

            first.Activations = Vec.Build.DenseOfArray(new double[] { 0.1, 0.2 });
            last.Errors = Vec.Build.DenseOfArray(new double[] { 0.3, 0.4, 0.5 });
            Assert.Equal(last.Errors, last.DCostDBiases);
        }

        [Fact]
        public void CalculatesDCostDWeights()
        {
            var last = new Layer(3);
            var first = new Layer(2, last);

            first.Activations = Vec.Build.DenseOfArray(new double[] { 0.1, 0.2 });
            last.Errors = Vec.Build.DenseOfArray(new double[] { 0.3, 0.4, 0.5 });
            var expected = Matrix.Build.DenseOfArray(new double[3, 2] {
                {0.03, 0.06},
                {0.04, 0.08},
                {0.05, 0.10},
            });

            var err = 0.0001;
            var actual = last.DCostDWeights;
            for (int r = 0; r < expected.RowCount; r++)
            {
                for (int c = 0; c < expected.ColumnCount; c++)
                {
                    Assert.InRange(actual[r,c], expected[r,c] - err, expected[r,c] + err);
                }
            }
        }
    }
}