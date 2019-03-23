﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworkUsingMathLibrary
{
    public static class ExtensionMethods
    {
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

        public static Matrix<float> Apply(this Matrix<float> m1, Matrix<float> m2, Func<float, float, float> func)
        {
            Debug.Assert(m1.RowCount == m2.RowCount);
            Debug.Assert(m1.ColumnCount == m2.ColumnCount);

            Matrix<float> result = Matrix<float>.Build.Dense(m1.RowCount, m1.ColumnCount);

            for (int rowIndex = 0; rowIndex < m1.RowCount; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < m1.ColumnCount; columnIndex++)
                {
                    result[rowIndex, columnIndex] = func(m1[rowIndex, columnIndex], m2[rowIndex, columnIndex]);
                }
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

        public static int ReadBigEndianInt32(this byte[] buffer, int offset)
        {
            var int32Value = new byte[4];
            Array.Copy(buffer, offset, int32Value, 0, 4);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(int32Value);
            }
            return BitConverter.ToInt32(int32Value, 0);
        }
    }
}