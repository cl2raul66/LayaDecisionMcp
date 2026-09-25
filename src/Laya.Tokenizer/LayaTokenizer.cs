using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Laya.Tokenizer;

/// <summary>
/// Tokenizador ByteLevel BPE (estilo GPT-2) puro en C#, compatible con el tokenizer.json de
/// laya-typed-decisions (vocab 50280, merges 50009) y con su tokenizer HF de referencia.
/// AOT-safe: JsonDocument + [GeneratedRegex], sin reflexión en el hot path.
/// </summary>
public sealed partial class LayaTokenizer
{
    // Regex GPT-2 oficial (tabla de pre-tokenización byte-level con use_regex=true).
    [GeneratedRegex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")]
    private static partial Regex Gpt2PreTokenizeRegex();

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _rev;            // id -> token (vocab + added_tokens)
    private readonly Dictionary<string, int> _ranks;          // "a b" -> índice de merge (0 = más fuerte)
    private readonly char[] _byteToSymbol;
    private readonly Dictionary<char, byte> _symbolToByte;
    private readonly Dictionary<string, long> _specialIds;    // contenido special -> id

    /// <summary>IDs reservados por el tokenizer de Laya (verificados contra tokenizer.json real).</summary>
    public const int UnkId = 50280;
    public const int ClsId = 50281;
    public const int SepId = 50282;
    public const int PadId = 50283;
    public const int MaskId = 50284;
    public const int MaxContext = 1024; // laya_config.max_len

    public long MaskIdLong => MaskId;
    public long ClsIdLong => ClsId;
    public long SepIdLong => SepId;
    public long PadIdLong => PadId;

    public int VocabSize => _vocab.Count;

    public LayaTokenizer(string tokenizerJsonPath)
    {
        var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJsonPath));
        try
        {
            var root = doc.RootElement;
            var model = root.GetProperty("model");
            if (!string.Equals(model.GetProperty("type").GetString(), "BPE", StringComparison.Ordinal))
                throw new NotSupportedException($"Se esperaba un modelo BPE, encontrado '{model.GetProperty("type").GetString()}'.");

            var vocab = model.GetProperty("vocab");
            _vocab = new Dictionary<string, int>(vocab.EnumerateObject().Count(), StringComparer.Ordinal);
            _rev = [];
            foreach (var prop in vocab.EnumerateObject())
            {
                var id = prop.Value.GetInt32();
                _vocab[prop.Name] = id;
                if (!_rev.ContainsKey(id)) _rev[id] = prop.Name;
            }

            var merges = model.GetProperty("merges");
            _ranks = new Dictionary<string, int>(merges.GetArrayLength(), StringComparer.Ordinal);
            for (var i = 0; i < merges.GetArrayLength(); i++)
            {
                var pair = merges[i];
                _ranks[$"{pair[0].GetString()} {pair[1].GetString()}"] = i;
            }

            _specialIds = new Dictionary<string, long>(StringComparer.Ordinal);
            foreach (var e in root.GetProperty("added_tokens").EnumerateArray())
            {
                var id = e.GetProperty("id").GetInt64();
                var content = e.GetProperty("content").GetString()!;
                var special = e.GetProperty("special").GetBoolean();
                if (!_rev.ContainsKey((int)id)) _rev[(int)id] = content;
                if (special && !_specialIds.ContainsKey(content)) _specialIds[content] = id;
            }
        }
        finally
        {
            doc.Dispose();
        }

        (_byteToSymbol, _symbolToByte) = BuildByteToSymbolTables();
    }

    /// <summary>Resuelve un token especial por su contenido (p. ej. "[MASK]").</summary>
    public long GetSpecialId(string content)
        => _specialIds.GetValueOrDefault(content, UnkId);

    /// <summary>
    /// Convierte texto a IDs usando exactamente el algoritmo del tokenizer HF (byte-level BPE):
    /// NFC -> pre-tokenización GPT-2 -> UTF-8 -> símbolos byte -> merges BPE -> ids.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text)) return [];

        var ids = new List<int>();
        var normalized = text.Normalize(NormalizationForm.FormC);

        foreach (Match m in Gpt2PreTokenizeRegex().Matches(normalized))
        {
            var piece = m.Value;
            var utf8 = Encoding.UTF8.GetBytes(piece);
            var word = new List<string>(utf8.Length);
            var wordIds = new List<int>(utf8.Length);
            for (var i = 0; i < utf8.Length; i++)
            {
                var symbol = _byteToSymbol[utf8[i]].ToString();
                word.Add(symbol);
                wordIds.Add(_vocab.GetValueOrDefault(symbol, UnkId));
            }
            ApplyMerges(word, wordIds);
            ids.AddRange(wordIds);
        }
        return [.. ids];
    }

    private void ApplyMerges(List<string> word, List<int> wordIds)
    {
        // BPE estilo GPT-2: repetir fusionando siempre el par adyacente con el rango mínimo.
        while (word.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIdx = -1;
            for (var i = 0; i < word.Count - 1; i++)
            {
                var key = word[i] + " " + word[i + 1];
                if (_ranks.TryGetValue(key, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx = i;
                    if (bestRank == 0) break;
                }
            }
            if (bestIdx < 0) break;

            var merged = word[bestIdx] + word[bestIdx + 1];
            word[bestIdx] = merged;
            wordIds[bestIdx] = _vocab.GetValueOrDefault(merged, UnkId);
            word.RemoveAt(bestIdx + 1);
            wordIds.RemoveAt(bestIdx + 1);
        }
    }

    /// <summary>Inverso (decoder byte-level) para depuración y tests.</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();
        foreach (var id in ids)
        {
            if (!_rev.TryGetValue(id, out var tok)) continue;
            foreach (var ch in tok)
            {
                if (_symbolToByte.TryGetValue(ch, out var b)) bytes.Add(b);
            }
        }
        return Encoding.UTF8.GetString([.. bytes]);
    }

    /// <summary>Tablas byte-&gt;símbolo y símbolo-&gt;byte (bytes_to_unicode de GPT-2).</summary>
    private static (char[], Dictionary<char, byte>) BuildByteToSymbolTables()
    {
        var bs = new List<int>();
        for (var b = '!'; b <= '~'; b++) bs.Add(b);
        for (var b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (var b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        var cs = new List<int>(bs);
        var n = 0;
        for (var b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }
        var toSymbol = new char[256];
        var toByte = new Dictionary<char, byte>(256);
        for (var i = 0; i < 256; i++)
        {
            toSymbol[bs[i]] = (char)cs[i];
            toByte[(char)cs[i]] = (byte)bs[i];
        }
        return (toSymbol, toByte);
    }
}
