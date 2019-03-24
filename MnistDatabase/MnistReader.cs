using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;

namespace MnistDatabase
{
    public interface IMnistReader
    {
        IList<byte> LoadLabels(string fileName);

        Task<IEnumerable<byte[]>> LoadImages(string fileName, Action<long> setProgressBarMax, Action incrementCounter);
    }

    public class MnistReader : IMnistReader
    {
        public IList<byte> LoadLabels(string fileName)
        {
            if (!File.Exists(fileName))
            {


                return Enumerable.Empty<byte>().ToList();
            }

            FileInfo fileToDecompress = new FileInfo(fileName);

            using (FileStream originalFileStream = fileToDecompress.OpenRead())
            using (GZipStream decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
            using (var resultStream = new MemoryStream())
            {
                decompressionStream.CopyTo(resultStream);
                return resultStream.ToArray().Skip(8).ToList();
            }
        }

        public async Task<IEnumerable<byte[]>> LoadImages(string fileName, Action<long> setProgressMax, Action incrementProgress)
        {
            FileInfo fileToDecompress = new FileInfo(fileName);
            List<byte[]> images = new List<byte[]>();

            using (FileStream originalFileStream = fileToDecompress.OpenRead())
            using (GZipStream decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
            {
                byte[] buffer = new byte[28 * 28];

                // Read file header
                await ReadBytes(decompressionStream, buffer, 16);
                // Read number of images from header
                setProgressMax(buffer.ReadBigEndianInt32(4));

                // Read all the images
                while (await ReadBytes(decompressionStream, buffer, buffer.Length))
                {
                    images.Add(buffer);
                    incrementProgress();
                }
            }

            return images;
        }

        private async Task<bool> ReadBytes(Stream decompressionStream, byte[] buffer, int count)
        {
            int offset = 0;
            int bytesRead = 0;
            while ((bytesRead = await decompressionStream.ReadAsync(buffer, offset, count - offset)) > 0)
            {
                if (bytesRead + offset < count)
                {
                    offset += bytesRead;
                }
                else
                {
                    return true;
                }
            }
            return false;
        }
    }
}
