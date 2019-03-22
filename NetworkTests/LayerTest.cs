using System;
using System.Collections.Generic;
using neuralNet;
using Xunit;
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;

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

            Assert.Null(last.Weights);
            Assert.NotNull(first.Weights);
            Assert.Equal(first.Weights.ColumnCount, first.NeuronCount);
            Assert.Equal(first.Weights.RowCount, first.NextLayer.NeuronCount);
        }

        [Fact]
        public void InitializesBiases()
        {
            var last = new Layer(10);
            var first = new Layer(2, last);

            Assert.Null(last.Biases);
            Assert.NotNull(first.Biases);
            Assert.Equal(first.Biases.Count, first.NeuronCount);
        }
    }
}
