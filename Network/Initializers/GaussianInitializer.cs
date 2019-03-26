using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

namespace NeuralNet
{
    public class GaussianInitializer : IWeightBiasInitializer
    {
        public virtual Matrix GetWeights(Layer layer)
        {
            return Matrix.Build.Random(layer.NeuronCount, layer.PreviousLayer.NeuronCount);
        }

        public virtual Vec GetBiases(Layer layer)
        {
            return Vec.Build.Random(layer.NeuronCount);
        }
    }
}