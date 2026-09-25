using Laya.Tokenizer;

namespace Laya.Inference;

public enum LayaQType { Choice = 0, Score = 1, Noul = 2 }

/// <summary>Petición de decisión. <c>Options</c> son las opciones YA renderizadas en el formato de Laya:
/// choice/noul => "etiqueta: descripción"; score => "level N: etiqueta".</summary>
public sealed record LayaDecisionRequest(
    LayaQType Type,
    string Question,
    string Instructions,
    IReadOnlyList<string> Options,
    string? State);

/// <summary>Tensores de entrada 1:1 con la firma ONNX exportada (reference.npz).</summary>
public sealed record LayaInputTensor(
    long[] InputIds,
    long[] AttentionMask,
    long[] MarkerPos,
    bool[] MarkerMask,
    long[] Qtype);

/// <summary>
/// Construye la secuencia Laya: [CLS] {qtype} question: {instr} [SEP] [MASK] option0 ... [SEP] {state} [SEP].
/// Los marcadores [MASK] (id 50284) se sitúan como prefijo de cada opción; marker_pos contiene la posición
/// absoluta de cada marcador y marker_mask indica cuáles están presentes (máx 3).
/// </summary>
public static class LayaSequence
{
    public const int MaxOptions = 3;

    public static LayaInputTensor Build(LayaTokenizer tok, LayaDecisionRequest req, int padTo = 0)
    {
        var typeWord = req.Type switch
        {
            LayaQType.Score => "score",
            LayaQType.Noul => "noul",
            _ => "choice",
        };

        if (req.Options.Count == 0) throw new ArgumentException("Se necesita al menos 1 opción.");
        if (req.Options.Count > MaxOptions) throw new ArgumentException($"Máximo {MaxOptions} opciones (head compartido de 256).");

        var ids = new List<long>(128) { LayaTokenizer.ClsId };
        ids.AddRange(ToLong(tok.Encode($"{typeWord} question: {req.Instructions}")));
        ids.Add(LayaTokenizer.SepId);

        var markerPos = new long[MaxOptions];
        var markerMask = new bool[MaxOptions];
        for (var i = 0; i < MaxOptions; i++)
        {
            if (i < req.Options.Count)
            {
                markerPos[i] = ids.Count;
                markerMask[i] = true;
                ids.Add(LayaTokenizer.MaskId);
                // El prompt de referencia antepone un espacio a cada opción ("[MASK] billing: ...")
                ids.AddRange(ToLong(tok.Encode(" " + req.Options[i])));
            }
            else
            {
                markerPos[i] = 0;
                markerMask[i] = false;
            }
        }

        ids.Add(LayaTokenizer.SepId);
        if (!string.IsNullOrEmpty(req.State))
            ids.AddRange(ToLong(tok.Encode(req.State!)));
        ids.Add(LayaTokenizer.SepId);

        if (ids.Count > LayaTokenizer.MaxContext)
            throw new InvalidOperationException($"Secuencia de {ids.Count} tokens supera el contexto máx de {LayaTokenizer.MaxContext}.");

        var seqLen = padTo > ids.Count ? padTo : ids.Count;
        var inputIds = new long[seqLen];
        var attn = new long[seqLen];
        for (var i = 0; i < ids.Count; i++)
        {
            inputIds[i] = ids[i];
            attn[i] = 1;
        }
        for (var i = ids.Count; i < seqLen; i++)
        {
            inputIds[i] = LayaTokenizer.PadId;
            attn[i] = 0;
        }

        return new LayaInputTensor(inputIds, attn, markerPos, markerMask, [(long)req.Type]);
    }

    private static IEnumerable<long> ToLong(IEnumerable<int> xs)
    {
        foreach (var x in xs) yield return x;
    }
}