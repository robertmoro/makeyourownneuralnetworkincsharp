using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkUsingMathLibrary
{
    public static class ExtensionMethods
    {
        public static Vector<float> ToVector(this IEnumerable<float> data)
        {
            return Vector<float>.Build.DenseOfEnumerable(data);
        }

        public static Matrix<float> ActivationFunction(this Matrix<float> matrix)
        {
            return matrix.Map(v => Sigmoid(v));
        }

        public static Vector<float> ActivationFunction(this Vector<float> vector)
        {
            return vector.Map(v => Sigmoid(v));
        }

        private static float Sigmoid(float value) => 1.0f / (1.0f + (float)Math.Exp(-value));

        public static List<float> CreateTargetOutput(int digit)
        {
            var result = Enumerable.Repeat(0.01f, 10).ToList();
            result[digit] = 0.99f;
            return result;
        }

        public static List<float> Transform(byte[] data)
        {
            return data.Select(v => (v / 255f) * 0.99f + 0.01f).ToList();
        }

        /// <summary>
        /// Return value read by the neural network as digit.
        /// </summary>
        /// <remarks>
        /// The index (0-9) corresponds directly the digits.
        /// The index with the highest value is the most likely result (digit).
        /// </remarks>
        /// <param name="results"></param>
        /// <returns></returns>
        public static int Result(this float[] results)
        {
            int i = 0;
            var dic = results.ToDictionary(r => i++, r => r);
            return dic.OrderByDescending(r => r.Value).First().Key;
        }
    }
}
