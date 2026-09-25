using Laya.Tokenizer;
using Microsoft.ML.OnnxRuntime;

namespace Laya.Inference;

/// <summary>Resultado tipado de una decisión Laya, con transparencia de calibración.</summary>
public sealed record LayaDecision(
    LayaQType Type,
    string Question,
    string? Choice,          // choice: opción ganadora (texto renderado)
    int? ChoiceIndex,        // choice: índice de la ganadora
    double? Score,           // score: valor esperado (Σ p_i·i)
    double? Noul,            // noul: P(opción verdadera) — la última opción ("true")
    double[] Probabilities,  // softmax por opción (post-temperatura)
    double Confidence,       // 1 − H/ln(N) — confianza normalizada (actuar solo si supera umbral)
    double ActProbability,   // cabeza act/escalate del modelo (ya normalizada por el propio modelo)
    double Temperature,
    string TemperatureBucket,
    int SeqLen);

/// <summary>
/// Sesión ONNX del modelo laya-typed-decisions: ejecuta los 5 inputs exportados y decodifica
/// salidas tipadas (choice/score/noul) usando el laya_config.json del checkpoint.
/// AOT-safe: sin reflexión en el hot path.
/// </summary>
public sealed class LayaModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly LayaTokenizer _tok;
    private readonly LayaConfig _config;
    private readonly string _inputIds, _attn, _markerPos, _markerMask, _qtype;
    private readonly string _logits, _actProbs;
    private readonly bool _actIsLogits; // yehor exporta act SIN normalizar (act_logits) -> softmax aquí
    private readonly int _maxBatch, _seqIn, _seqOut;

    public LayaModel(string onnxPath, string tokenizerJsonPath, string layaConfigPath)
    {
        _tok = new LayaTokenizer(tokenizerJsonPath);
        _config = LayaConfig.Load(layaConfigPath);

        var opts = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        opts.AppendExecutionProvider_CPU();
        _session = new InferenceSession(onnxPath, opts);

        // Mapeo de nombres de entrada/salida (los del exportador; tolera renombrados).
        _inputIds = PickInput("input_ids");
        _attn = PickInput("attention_mask");
        _markerPos = PickInput("marker_pos");
        _markerMask = PickInput("marker_mask");
        _qtype = PickInput("qtype");
        _logits = PickOutput("logits", "output0");
        _actProbs = PickOutput("act_probs", "act_probabilities", "act_logits", "output1");
        _actIsLogits = string.Equals(_actProbs, "act_logits", StringComparison.Ordinal);

        _seqIn = (int)(_session.InputMetadata[_inputIds].Dimensions.Length > 1 ? _session.InputMetadata[_inputIds].Dimensions[1] : -1);
        _seqOut = (int)(_session.OutputMetadata[_logits].Dimensions.Length > 1 ? _session.OutputMetadata[_logits].Dimensions[1] : -1);
        _maxBatch = (int)(_session.InputMetadata[_inputIds].Dimensions.Length > 0 ? _session.InputMetadata[_inputIds].Dimensions[0] : -1);
    }

    public LayaTokenizer Tokenizer => _tok;
    public LayaConfig Config => _config;

    public IReadOnlyDictionary<string, NodeMetadata> InputMetadata => _session.InputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata => _session.OutputMetadata;

    /// <summary>
    /// Ejecución cruda del modelo (misma firma que el exportador; batch como primer eje).
    /// Garantiza act SIEMPRE normalizada: si el exportador entrega act_logits, aplica softmax estable.
    /// </summary>
    public (float[] Logits, float[] ActProbs) ExecuteRaw(LayaInputTensor t, int batch = 1)
    {
        float[] logits;
        float[] actProbs;
        using var runOptions = new RunOptions();
        var ortInputs = CreateInputs(t, batch);
        try
        {
            var results = _session.Run(runOptions, ortInputs, [_logits, _actProbs]);
            try
            {
                var logitsTensor = results.ElementAt(0);
                var actTensor = results.ElementAt(1);
                logits = logitsTensor.GetTensorDataAsSpan<float>().ToArray();
                actProbs = actTensor.GetTensorDataAsSpan<float>().ToArray();
                if (_actIsLogits) actProbs = Softmax2(actProbs);
            }
            finally
            {
                foreach (var v in results) v.Dispose();
                if (results is IDisposable d) d.Dispose();
            }
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
        return (logits, actProbs);
    }

    /// <summary>Decide una única pregunta tipada.</summary>
    public LayaDecision Decide(LayaDecisionRequest req)
    {
        var tensor = LayaSequence.Build(_tok, req);
        var (temp, bucket) = _config.GetTemperature(TypeWord(req.Type), req.Options.Count);
        var (logits, actProbs) = ExecuteRaw(tensor);
        return Decode(req, tensor, logits, actProbs, temp, bucket);
    }

    private Dictionary<string, OrtValue> CreateInputs(LayaInputTensor t, int batch)
    {
        var seq = t.InputIds.Length / batch;
        var markers = t.MarkerPos.Length / batch;
        var dict = new Dictionary<string, OrtValue>(5);
        try
        {
            dict[_inputIds] = OrtValue.CreateTensorValueFromMemory(t.InputIds, [batch, seq]);
            dict[_attn] = OrtValue.CreateTensorValueFromMemory(t.AttentionMask, [batch, seq]);
            dict[_markerPos] = OrtValue.CreateTensorValueFromMemory(t.MarkerPos, [batch, markers]);
            dict[_markerMask] = OrtValue.CreateTensorValueFromMemory(t.MarkerMask, [batch, markers]);
            dict[_qtype] = OrtValue.CreateTensorValueFromMemory(t.Qtype, [batch]);
            return dict;
        }
        catch
        {
            foreach (var v in dict.Values) v.Dispose();
            throw;
        }
    }

    private static LayaDecision Decode(LayaDecisionRequest req, LayaInputTensor tensor, float[] logits, float[] actProbs, double temp, string bucket)
    {
        var n = req.Options.Count;

        // Softmax estable sobre los logits válidos, escalados por temperatura.
        var raw = new double[n];
        var max = double.NegativeInfinity;
        for (var i = 0; i < n; i++) raw[i] = logits[i] / temp;
        for (var i = 0; i < n; i++) if (raw[i] > max) max = raw[i];
        var probs = new double[n];
        var sum = 0.0;
        for (var i = 0; i < n; i++) { probs[i] = Math.Exp(raw[i] - max); sum += probs[i]; }
        for (var i = 0; i < n; i++) probs[i] /= sum;

        // Confidence = 1 − H/ln(N) (verificado contra reference.json).
        var h = 0.0;
        foreach (var p in probs) h -= p > 0 ? p * Math.Log(p) : 0.0;
        var confidence = 1.0 - h / Math.Log(n);

        string? choice = null;
        int? choiceIndex = null;
        double? score = null;
        double? noul = null;

        switch (req.Type)
        {
            case LayaQType.Choice:
                var best = 0;
                for (var i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
                choiceIndex = best;
                choice = req.Options[best];
                break;
            case LayaQType.Score:
                var exp = 0.0;
                for (var i = 0; i < n; i++) exp += probs[i] * i;
                score = Math.Round(exp, 4);
                break;
            case LayaQType.Noul:
                noul = Math.Round(probs[^1], 4); // opción final ("true")
                break;
        }

        return new LayaDecision(
            req.Type, req.Question, choice, choiceIndex, score, noul,
            probs, confidence, actProbs.Length > 0 ? actProbs[0] : double.NaN,
            temp, bucket, tensor.InputIds.Length);
    }

    private static float[] Softmax2(float[] v)
    {
        var m = Math.Max(v[0], v[1]);
        var e0 = MathF.Exp(v[0] - m);
        var e1 = MathF.Exp(v[1] - m);
        var s = e0 + e1;
        return [e0 / s, e1 / s];
    }

    private static string TypeWord(LayaQType t) => t switch
    {
        LayaQType.Score => "score",
        LayaQType.Noul => "noul",
        _ => "choice",
    };

    private string PickInput(params string[] candidates)
    {
        foreach (var c in candidates)
            if (_session.InputMetadata.ContainsKey(c)) return c;
        throw new InvalidOperationException($"Modelo sin entrada esperada {string.Join("/", candidates)}. Entradas: {string.Join(", ", _session.InputMetadata.Keys)}");
    }

    private string PickOutput(params string[] candidates)
    {
        foreach (var c in candidates)
            if (_session.OutputMetadata.ContainsKey(c)) return c;
        throw new InvalidOperationException($"Modelo sin salida esperada {string.Join("/", candidates)}. Salidas: {string.Join(", ", _session.OutputMetadata.Keys)}");
    }

    public void Dispose() => _session.Dispose();
}