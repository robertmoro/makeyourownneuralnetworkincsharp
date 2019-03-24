using System;
using System.Collections.Generic;
using System.Linq;

namespace MnistDatabase
{
    public static class MnistExtensionMethods
    {
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

        public static List<float> Transform(byte[] data)
        {
            return data.Select(v => (v / 255f) * 0.99f + 0.01f).ToList();
        }
    }
}
