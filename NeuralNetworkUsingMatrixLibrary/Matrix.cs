using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkDomain
{
    public class Matrix
    {
        private readonly float[,] _matrix;

        public Matrix(int rowCount, int columnCount) => _matrix = new float[rowCount, columnCount];

        public int RowCount => _matrix.GetLength(0);
        public int ColumnCount => _matrix.GetLength(1);

        public float this[int row, int column]
        {
            get => _matrix[row, column];
            set => _matrix[row, column] = value;
        }

        public static Matrix operator +(Matrix m1, Matrix m2) => m1.Apply(m2, (x, y) => x + y);

        public static Matrix operator -(Matrix m1, Matrix m2) => m1.Apply(m2, (x, y) => x - y);
        public static Matrix operator *(Matrix m1, Matrix m2) => m1.Apply(m2, (x, y) => x * y);
        public static Matrix operator *(float f, Matrix m) => f.Apply(m, (x, y) => x * y);

        public List<float> Flatten() => _matrix.Cast<float>().ToList();
    }
}
