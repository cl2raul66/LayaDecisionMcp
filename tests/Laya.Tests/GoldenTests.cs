using Laya.Inference;
using Laya.Tests.Infra;
using Laya.Tokenizer;

namespace Laya.Tests;

/// <summary>
/// Tests golden contra los fixtures del exportador (ti3x-m/laya-typed-decisions-onnx):
/// reference.npz (input_ids/markers/qtype/logits/act_probs) y reference.json (respuestas).
/// Requiere artefactos descargados; los tests dependientes del modelo se omiten si falta model.onnx.
/// </summary>
public sealed class GoldenTests
{
    private const int SeqLen = 72; // padding del fixture de referencia

    private static readonly string State =
        """{"ticket": {"subject": "Refund request for a duplicate charge", "body": "I was charged twice. Please refund one payment.", "customer_language": "English"}}""";

    private static readonly LayaDecisionRequest[] Requests =
    [
        new(LayaQType.Choice, "department", "Route this ticket to one department.",
            ["billing: Payments and refunds", "support: Product help", "sales: New purchases"], State),
        new(LayaQType.Score, "urgency", "Score urgency from low to high.",
            ["level 0: low", "level 1: medium", "level 2: high"], State),
        new(LayaQType.Noul, "churn_risk", "Is this customer at risk of leaving?",
            ["false: no, the statement does not hold", "true: yes, the statement holds"], State),
    ];

    private static LayaTokenizer? _tokenizer;
    private static LayaTokenizer Tokenizer => _tokenizer ??= new LayaTokenizer(ArtifactPaths.TokenizerJson);

    private static LayaModel? _model;
    private static LayaModel Model() => _model ??= new LayaModel(ArtifactPaths.ModelOnnx, ArtifactPaths.TokenizerJson, ArtifactPaths.LayaConfigJson);

    private static bool HasModel =>
        File.Exists(ArtifactPaths.ModelOnnx) && File.Exists(ArtifactPaths.ModelOnnxData);

    [Fact]
    public void Tokenizer_Reproduces_Reference_InputIds()
    {
        var npz = Npy.ReadNpz(ArtifactPaths.ReferenceNpz);
        var refInputIds = npz["input_ids.npy"].AsInt64();
        var refAttn = npz["attention_mask.npy"].AsInt64();
        var refMarkerPos = npz["marker_pos.npy"].AsInt64();
        var refMarkerMask = npz["marker_mask.npy"].AsBool();
        var refQtype = npz["qtype.npy"].AsInt64();

        var tok = Tokenizer;
        for (var r = 0; r < 3; r++)
        {
            var t = LayaSequence.Build(tok, Requests[r], padTo: SeqLen);

            var expIds = refInputIds[(r * SeqLen)..((r + 1) * SeqLen)];
            var expAttn = refAttn[(r * SeqLen)..((r + 1) * SeqLen)];
            if (!t.InputIds.SequenceEqual(expIds))
            {
                var firstMismatch = Enumerable.Range(0, SeqLen).First(i => t.InputIds[i] != expIds[i]);
                var expectedText = tok.Decode(expIds.Take(SeqLen).Select(x => (int)x));
                var actualText = tok.Decode(t.InputIds.Select(x => (int)x));
                Assert.Fail($"""
                    row {r}: input_ids distintos en posición {firstMismatch}
                      esperado[{firstMismatch}]: {expIds[firstMismatch]}  obtenido: {t.InputIds[firstMismatch]}
                      texto esperado: {expectedText}
                      texto obtenido: {actualText}
                    """);
            }
            Assert.True(t.AttentionMask.SequenceEqual(expAttn), $"row {r}: attention_mask distinto");

            var expPos = refMarkerPos[(r * 3)..(r * 3 + 3)];
            var expMask = refMarkerMask[(r * 3)..(r * 3 + 3)];
            Assert.True(t.MarkerPos.SequenceEqual(expPos), $"row {r}: marker_pos distinto");
            Assert.True(t.MarkerMask.SequenceEqual(expMask), $"row {r}: marker_mask distinto");
            Assert.Equal(refQtype[r], t.Qtype[0]);
        }
    }

    [Fact]
    public void Config_Temperatures_Match_LayaConfig()
    {
        var cfg = LayaConfig.Load(ArtifactPaths.LayaConfigJson);
        Assert.Equal(256, cfg.HeadMaxLen);
        Assert.Equal(1024, cfg.MaxLen);

        // Buckets verificados en el experimento de F0 (ti3x)
        Assert.Equal("choice:2", cfg.GetTemperature("choice", 2).Bucket);
        Assert.Equal("choice:3-5", cfg.GetTemperature("choice", 4).Bucket);
        Assert.Equal("choice:6-10", cfg.GetTemperature("choice", 7).Bucket);
        Assert.Equal("choice:11+", cfg.GetTemperature("choice", 15).Bucket);
        Assert.Equal("score:3-5", cfg.GetTemperature("score", 4).Bucket);
        Assert.Equal("noul:2", cfg.GetTemperature("noul", 2).Bucket);
        Assert.Equal(1.9834, cfg.GetTemperature("noul", 2).Temp, 3);
        Assert.Equal(1.0, cfg.GetTemperature("score", 6).Temp);       // combo no calibrado -> 1.0
        Assert.Equal(1.0, cfg.GetTemperature("choice", 1).Temp);
    }

    [Fact]
    public void Inference_Matches_Reference_Logits_And_ActProbs()
    {
        if (!HasModel) return; // se omite hasta descargar model.onnx

        var npz = Npy.ReadNpz(ArtifactPaths.ReferenceNpz);
        var model = Model();
        var allIds = npz["input_ids.npy"].AsInt64();
        var allAttn = npz["attention_mask.npy"].AsInt64();
        var allPos = npz["marker_pos.npy"].AsInt64();
        var allMask = npz["marker_mask.npy"].AsBool();
        var allQtype = npz["qtype.npy"].AsInt64();
        var refLogits = npz["logits.npy"].AsFloat32();      // (3,3)
        var refAct = npz["act_probs.npy"].AsFloat32();      // (3,2)

        // El modelo ONNX exportado tiene batch fijo = 1; la referencia apila 3 ejecuciones batch-1.
        var batchLogits = new List<float>();
        var batchAct = new List<float>();
        for (var r = 0; r < 3; r++)
        {
            var t = new LayaInputTensor(
                allIds[(r * SeqLen)..((r + 1) * SeqLen)],
                allAttn[(r * SeqLen)..((r + 1) * SeqLen)],
                allPos[(r * 3)..(r * 3 + 3)],
                allMask[(r * 3)..(r * 3 + 3)],
                [allQtype[r]]);
            var (logits, actProbs) = model.ExecuteRaw(t);
            batchLogits.AddRange(logits);
            batchAct.AddRange(actProbs);
        }

        for (var i = 0; i < refLogits.Length; i++)
            Assert.True(Math.Abs(batchLogits[i] - refLogits[i]) < 1e-4,
                $"logits[{i}]: {batchLogits[i]:F6} != {refLogits[i]:F6}");
        for (var i = 0; i < refAct.Length; i++)
            Assert.True(Math.Abs(batchAct[i] - refAct[i]) < 1e-5,
                $"act_probs[{i}]: {batchAct[i]:F6} != {refAct[i]:F6}");
    }

    [Theory]
    [InlineData("fp32", 1e-3, 1e-3)]
    [InlineData("int8", 5e-2, 5e-2)]
    public void Yehor_Logits_Match_Reference(string model, double logitsAtol, double actAtol)
    {
        var onnxPath = model == "int8" ? ArtifactPaths.YehorInt8 : ArtifactPaths.YehorFp32;
        if (!File.Exists(onnxPath)) return;

        // yehor exporta act_logits (sin normalizar); ExecuteRaw aplica softmax -> comparable con act_probs.
        using var m = new LayaModel(onnxPath, ArtifactPaths.TokenizerJson, ArtifactPaths.LayaConfigJson);
        var npz = Npy.ReadNpz(ArtifactPaths.ReferenceNpz);
        var allIds = npz["input_ids.npy"].AsInt64();
        var allAttn = npz["attention_mask.npy"].AsInt64();
        var allPos = npz["marker_pos.npy"].AsInt64();
        var allMask = npz["marker_mask.npy"].AsBool();
        var allQtype = npz["qtype.npy"].AsInt64();
        var refLogits = npz["logits.npy"].AsFloat32();
        var refAct = npz["act_probs.npy"].AsFloat32();

        var batchLogits = new List<float>();
        var batchAct = new List<float>();
        for (var r = 0; r < 3; r++)
        {
            var t = new LayaInputTensor(
                allIds[(r * SeqLen)..((r + 1) * SeqLen)],
                allAttn[(r * SeqLen)..((r + 1) * SeqLen)],
                allPos[(r * 3)..(r * 3 + 3)],
                allMask[(r * 3)..(r * 3 + 3)],
                [allQtype[r]]);
            var (logits, actProbs) = m.ExecuteRaw(t);
            batchLogits.AddRange(logits);
            batchAct.AddRange(actProbs);
        }

        for (var i = 0; i < refLogits.Length; i++)
            Assert.True(Math.Abs(batchLogits[i] - refLogits[i]) < logitsAtol,
                $"{model} logits[{i}]: {batchLogits[i]:F6} != {refLogits[i]:F6}");
        for (var i = 0; i < refAct.Length; i++)
            Assert.True(Math.Abs(batchAct[i] - refAct[i]) < actAtol,
                $"{model} act_probs[{i}]: {batchAct[i]:F6} != {refAct[i]:F6}");
    }

    [Theory]
    [InlineData("ti3x", 1.5e-4, 1e-3, 1e-4)]
    [InlineData("fp32", 1.5e-4, 1e-3, 1e-4)]
    [InlineData("int8", 2e-2, 5e-2, 5e-2)]
    public void EndToEnd_Decode_Matches_Reference_Json(string model, double probsAtol, double confAtol, double scoreAtol)
    {
        var onnxPath = model switch
        {
            "fp32" => ArtifactPaths.YehorFp32,
            "int8" => ArtifactPaths.YehorInt8,
            _ => ArtifactPaths.ModelOnnx,
        };
        if (!File.Exists(onnxPath)) return;
        if (model == "ti3x" && !HasModel) return;

        using var m = new LayaModel(onnxPath, ArtifactPaths.TokenizerJson, ArtifactPaths.LayaConfigJson);

        // department (choice, 3 opciones, temp choice:3-5=1.7601518630981445)
        var dep = m.Decide(Requests[0]);
        Assert.Equal("choice:3-5", dep.TemperatureBucket);
        Assert.Equal(0, dep.ChoiceIndex);
        Assert.Equal("billing: Payments and refunds", dep.Choice);
        AssertProbs(dep.Probabilities, [0.7876, 0.127, 0.0854], probsAtol);
        AssertNear(dep.Confidence, 0.399, confAtol, "dep.confidence");
        AssertNear(dep.ActProbability, 1.0, confAtol, "dep.act");

        // urgency (score, 3 niveles, temp score:3-5=1.2514300346374512)
        var urg = m.Decide(Requests[1]);
        Assert.Equal("score:3-5", urg.TemperatureBucket);
        AssertProbs(urg.Probabilities, [0.0963, 0.3478, 0.5559], probsAtol);
        AssertNear(urg.Score!.Value, 1.4596, scoreAtol, "urg.score");
        AssertNear(urg.Confidence, 0.1634, confAtol, "urg.confidence");

        // churn_risk (noul, 2 opciones, temp noul:2=1.983399510383606)
        var churn = m.Decide(Requests[2]);
        Assert.Equal("noul:2", churn.TemperatureBucket);
        AssertProbs(churn.Probabilities, [0.726, 0.274], probsAtol);
        AssertNear(churn.Noul!.Value, 0.274, scoreAtol, "churn.noul");
        AssertNear(churn.ActProbability, 1.0, confAtol, "churn.act");
    }

    private static void AssertNear(double actual, double expected, double atol, string name)
    {
        Assert.True(Math.Abs(actual - expected) <= atol, $"{name}: {actual:F6} != {expected:F6} (±{atol})");
    }

    private static void AssertProbs(double[] actual, double[] expected, double atol = 1.5e-4)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= atol,
                $"prob[{i}]: {actual[i]:F6} != {expected[i]:F6} (±{atol})");
    }

    [Fact]
    public void Tokenizer_Known_Ids_And_RoundTrip()
    {
        var tok = Tokenizer;
        Assert.Equal(50280, tok.VocabSize);
        Assert.Equal(50284L, tok.GetSpecialId("[MASK]"));
        Assert.Equal(50281L, tok.GetSpecialId("[CLS]"));
        Assert.Equal(50282L, tok.GetSpecialId("[SEP]"));
        Assert.Equal(50283L, tok.GetSpecialId("[PAD]"));
        Assert.Equal(50280L, tok.GetSpecialId("[UNK]"));

        // Round-trip byte-level: encode -> decode debe reproducir el texto normalizado.
        var text = "question: Route this ticket to one department.";
        var decoded = tok.Decode(tok.Encode(text));
        Assert.Equal(text.Normalize(System.Text.NormalizationForm.FormC), decoded);

        // El template [CLS] {qtype} question: ... [SEP] es responsabilidad del builder de secuencia.
        var head = tok.Encode("choice question: Route this ticket.");
        Assert.True(head.Length > 0);
    }
}