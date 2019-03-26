
using Vec = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

namespace NeuralNet
{
    public interface IWeightBiasInitializer
    {
        Matrix GetWeights(Layer layer);
        Vec GetBiases(Layer layer);
    }
}