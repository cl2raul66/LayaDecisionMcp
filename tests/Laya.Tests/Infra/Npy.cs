using System.IO.Compression;

namespace Laya.Tests.Infra;

/// <summary>Lector mínimo de arrays .npy (v1/v2) y .npz (zip) — solo los dtype de los fixtures de Laya.</summary>
public sealed record NpyArray(int[] Shape, string DType, Array Data)
{
    public long[] AsInt64() => (long[])Data;
    public float[] AsFloat32() => (float[])Data;
    public bool[] AsBool() => (bool[])Data;
}

public static class Npy
{
    public static NpyArray Read(string path)
    {
        var b = File.ReadAllBytes(path);
        if (b.Length < 10 || b[0] != 0x93 || b[1] != 'N' || b[2] != 'U' || b[3] != 'M' || b[4] != 'P' || b[5] != 'Y')
            throw new InvalidDataException($"{path} no es un archivo .npy válido.");

        var ver = b[6];
        var headerLen = ver == 1 ? (int)BitConverter.ToUInt16(b, 8) : (int)BitConverter.ToUInt32(b, 8);
        var desc = System.Text.Encoding.ASCII.GetString(b, 10, headerLen);

        var shapeM = System.Text.RegularExpressions.Regex.Match(desc, "'shape'\\s*:\\s*\\(([^)]*)\\)");
        var shape = shapeM.Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();
        if (shape.Length == 0) shape = [1];

        var dtype = System.Text.RegularExpressions.Regex.Match(desc, "'descr'\\s*:\\s*'([^']+)'").Groups[1].Value;
        var off = 10 + headerLen;
        var count = shape.Aggregate(1, (a, s) => a * s);

        Array data = dtype switch
        {
            "<i8" => ReadPrims<long>(b, off, count, 8),
            "<i4" => ReadPrims<int>(b, off, count, 4),
            "<f4" => ReadPrims<float>(b, off, count, 4),
            "|b1" => ReadPrims<bool>(b, off, count, 1),
            _ => throw new NotSupportedException($"dtype '{dtype}' no soportado en {path}"),
        };
        return new NpyArray(shape, dtype, data);
    }

    private static T[] ReadPrims<T>(byte[] b, int off, int count, int size) where T : unmanaged
    {
        var span = b.AsSpan(off, count * size);
        return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(span).ToArray();
    }

    /// <summary>Lee las entradas de un .npz (todas las .npy).</summary>
    public static IReadOnlyDictionary<string, NpyArray> ReadNpz(string npzPath)
    {
        var result = new Dictionary<string, NpyArray>(StringComparer.Ordinal);
        using var zip = ZipFile.OpenRead(npzPath);
        foreach (var entry in zip.Entries)
        {
            using var ms = new MemoryStream();
            entry.Open().CopyTo(ms);
            var tmp = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            File.WriteAllBytes(tmp, ms.ToArray());
            try
            {
                var name = Path.GetFileNameWithoutExtension(entry.Name);
                result[Path.GetFileName(entry.Name)] = Read(tmp);
            }
            finally
            {
                File.Delete(tmp);
            }
        }
        return result;
    }

    /// <summary>Rellena las alineaciones de memoria de bool (xunit/env tolera) y convierte long[] a bool[] por valor.</summary>
    public static bool[] ToBoolArray(IEnumerable<long> ids) => ids.Select(x => x != 0).ToArray();
}