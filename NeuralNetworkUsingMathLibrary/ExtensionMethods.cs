using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworkUsingMathLibrary
{
    public static class ExtensionMethods
    {
        public static Vector<float> ToVector(this IEnumerable<float> data)
        {
            return Vector<float>.Build.DenseOfEnumerable(data);
        }

        public static Matrix<float> ToMatrix(this IEnumerable<float> data)
        {
            IReadOnlyList<float> list = data.ToList();
            var matrix = Matrix<float>.Build.Dense(list.Count, 1);
            for (int rowIndex = 0; rowIndex < matrix.RowCount; rowIndex++)
            {
                matrix[rowIndex, 0] = list[rowIndex];
            }
            return matrix;
        }

        public static Matrix<float> ActivationFunction(this Matrix<float> matrix)
        {
            float Sigmoid(float value) => 1.0f / (1.0f + (float)Math.Exp(-value));

            Matrix<float> result = Matrix<float>.Build.Dense(matrix.RowCount, matrix.ColumnCount);

            for (int heightIndex = 0; heightIndex < matrix.RowCount; heightIndex++)
            {
                for (int widthIndex = 0; widthIndex < matrix.ColumnCount; widthIndex++)
                {
                    result[heightIndex, widthIndex] = Sigmoid(matrix[heightIndex, widthIndex]);
                }
            }

            return result;
        }

        public static Vector<float> ActivationFunction(this Vector<float> vector)
        {
            float Sigmoid(float value) => 1.0f / (1.0f + (float)Math.Exp(-value));

            Vector<float> result = Vector<float>.Build.Dense(vector.Count);

            for (int heightIndex = 0; heightIndex < vector.Count; heightIndex++)
            {
                result[heightIndex] = Sigmoid(vector[heightIndex]);
            }

            return result;
        }

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
