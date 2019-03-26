
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;
using System;

namespace NeuralNet
{
    public class ScaledInitializer : GaussianInitializer
    {
        public override Matrix GetWeights(Layer layer)
        {
            var divisor = Math.Sqrt(layer.NeuronCount);
            return base.GetWeights(layer).Divide(divisor);
        }

        public override Vec GetBiases(Layer layer)
        {
            var divisor = Math.Sqrt(layer.NeuronCount);
            return base.GetBiases(layer).Divide(divisor);
        }
    }
}