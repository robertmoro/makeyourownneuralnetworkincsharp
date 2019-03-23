using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworkDomain
{
    public static class ExtensionMethods
    {
        public static Matrix Multiply(this Matrix m1, Matrix m2)
        {
            Matrix result = new Matrix(m1.RowCount, m2.ColumnCount);
            for (int i = 0; i < result.RowCount; i++)
            {
                for (int j = 0; j < result.ColumnCount; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < m1.ColumnCount; k++)
                    {
                        result[i, j] += m1[i, k] * m2[k, j];
                    }
                }
            }
            return result;
        }

        public static Matrix Transpose(this Matrix matrix)
        {
            Matrix result = new Matrix(matrix.ColumnCount, matrix.RowCount);

            for (int columnIndex = 0; columnIndex < matrix.ColumnCount; columnIndex++)
            {
                for (int rowIndex = 0; rowIndex < matrix.RowCount; rowIndex++)
                {
                    result[columnIndex, rowIndex] = matrix[rowIndex, columnIndex];
                }
            }

            return result;
        }

        public static Matrix ToMatrix(this IEnumerable<float> data)
        {
            IReadOnlyList<float> list = data.ToList();
            Matrix matrix = new Matrix(list.Count, 1);
            for (int rowIndex = 0; rowIndex < matrix.RowCount; rowIndex++)
            {
                matrix[rowIndex, 0] = list[rowIndex];
            }
            return matrix;
        }

        public static Matrix ActivationFunction(this Matrix matrix)
        {
            float Sigmoid(float value) => 1.0f / (1.0f + (float)Math.Exp(-value));

            Matrix result = new Matrix(matrix.RowCount, matrix.ColumnCount);

            for (int heightIndex = 0; heightIndex < matrix.RowCount; heightIndex++)
            {
                for (int widthIndex = 0; widthIndex < matrix.ColumnCount; widthIndex++)
                {
                    result[heightIndex, widthIndex] = Sigmoid(matrix[heightIndex, widthIndex]);
                }
            }

            return result;
        }

        public static Matrix Apply(this Matrix m1, Matrix m2, Func<float, float, float> func)
        {
            Debug.Assert(m1.RowCount == m2.RowCount);
            Debug.Assert(m1.ColumnCount == m2.ColumnCount);

            Matrix result = new Matrix(m1.RowCount, m1.ColumnCount);

            //Parallel.For(0, m1.RowCount, rowIndex =>
            for (int rowIndex = 0; rowIndex < m1.RowCount; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < m1.ColumnCount; columnIndex++)
                {
                    result[rowIndex, columnIndex] = func(m1[rowIndex, columnIndex], m2[rowIndex, columnIndex]);
                }
            }
            return result;
        }

        public static Matrix Apply(this float f, Matrix m, Func<float, float, float> func)
        {
            Matrix result = new Matrix(m.RowCount, m.ColumnCount);

            for (int rowIndex = 0; rowIndex < m.RowCount; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < m.ColumnCount; columnIndex++)
                {
                    result[rowIndex, columnIndex] = func(f, m[rowIndex, columnIndex]);
                }
            }
            return result;
        }

        public static void Init(this Matrix matrix, Func<float> func)
        {
            for (int rowIndex = 0; rowIndex < matrix.RowCount; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < matrix.ColumnCount; columnIndex++)
                {
                    matrix[rowIndex, columnIndex] = func();
                }
            }
        }

        public static int GetNumericValue(this char digit) => (int)char.GetNumericValue(digit);

        public static double NextGaussian(this Random r, double mu = 0, double sigma = 1)
        {
        Kak:
            var u1 = r.NextDouble();
            var u2 = r.NextDouble();

            var rand_std_normal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                  Math.Sin(2.0 * Math.PI * u2);

            var rand_normal = mu + sigma * rand_std_normal;

            if (Math.Abs(rand_normal) >= 0.5) goto Kak;

            return rand_normal;
        }

        public static List<float> CreateTargetOutput(int digit)
        {
            var result = Enumerable.Repeat(0.01f, 10).ToList();
            result[digit] = 0.99f;
            return result;
        }

        public static List<float> Transform(IEnumerable<int> data)
        {
            return data.Select(v => (v / 255f) * 0.99f + 0.01f).ToList();
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
        public static int Result(this List<float> results)
        {
            int i = 0;
            var dic = results.ToDictionary(r => i++, r => r);
            return dic.OrderByDescending(r => r.Value).First().Key;
        }

        public static List<int> ParseData(string data)
        {
            return data.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(v => int.Parse(v)).ToList();
        }

        public static int ReadBigEndianInt32(this byte[] buffer, int offset)
        {
            var int32Value = new byte[4];
            Array.Copy(buffer, offset, int32Value, 0, 4);
            if(BitConverter.IsLittleEndian)
            {
                Array.Reverse(int32Value);
            }
            return BitConverter.ToInt32(int32Value, 0);
        }
    }
}
