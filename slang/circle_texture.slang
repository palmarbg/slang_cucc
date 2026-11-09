import slangpy;

// Activation function
[Differentiable]
float leaky_relu(float x)
{
    return x > 0.0 ? x : 0.1 * x;
}

[Differentiable]
float tanh_act(float x)
{
    return (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
}

[Differentiable]
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
public float forward(
    no_diff int index,

    IDiffTensor<float, 2> uvs,

    IDiffTensor<float, 2> weights1,
    IDiffTensor<float, 1> bias1,

    IDiffTensor<float, 2> weights2,
    IDiffTensor<float, 1> bias2,

    IDiffTensor<float, 2> weights3,
    IDiffTensor<float, 1> bias3,
) {
    // Read the input uv coordinates for this thread
    float2 uv = float2(uvs[index, 0], uvs[index, 1]);

    return evaluateNetwork(uv,
                    weights1,
                    bias1,
                    weights2,
                    bias2,
                    weights3,
                    bias3);
}

[Differentiable]
public float evaluateNetwork(
    float2 uv,

    IDiffTensor<float, 2> weights1,
    IDiffTensor<float, 1> bias1,

    IDiffTensor<float, 2> weights2,
    IDiffTensor<float, 1> bias2,

    IDiffTensor<float, 2> weights3,
    IDiffTensor<float, 1> bias3
)
{
    // Layer 1: 2 -> 32
    float h1[32];
    
    for (int i = 0; i < 32; ++i)
    {
        float value = bias1[i];

        for (int j = 0; j < 2; j++)
        {
            value += weights1[i, j] * uv[j];
        }
        
        h1[i] = leaky_relu(value);
    }

    // Layer 2: 32 -> 32
    float h2[32];

    for (int i = 0; i < 32; ++i)
    {
        float value = bias2[i];

        for (int j = 0; j < 32; j++)
        {
            value += weights2[i, j] * h1[j];
        }

        h2[i] = leaky_relu(value);
    }

    // Layer 3: 32 -> 1
    float h3[1];

    for (int i = 0; i < 1; ++i)
    {
        float value = bias3[i];

        for (int j = 0; j < 32; j++)
        {
            value += weights3[i, j] * h2[j];
        }

        h3[i] = sigmoid(value);
    }

    return h3[0];
}

/************************************** FRAGMENT SHADER **************************************/

public struct GLSLTensor<int D> : IDiffTensor<float, D>
{
    public Tensor<float, D> data;
    public uint[D] _shape;
    public property shape : uint[D] { get { return _shape; } }

    public static GLSLTensor<D> fromTensor(Tensor<float, D> data) {
        GLSLTensor<D> instance;
        instance.data = data;
        instance._shape = data._shape;
        return instance;
    }

    float _read_primal_each<each I : __BuiltinIntegerType>(expand each I idx) {
        return data._read_each(idx);
    }

    void _write_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    void _accumulate_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    [Differentiable]
    [TreatAsDifferentiable]
    // [BackwardDerivative(_load_bwd_array)]
    public float load<I : __BuiltinIntegerType>(I idx[D]) {
        return data.load(idx);
    }

    // void _load_bwd_array<I : __BuiltinIntegerType>(I idx[D], float.Differential grad) {
    //     return;
    // }

    [Differentiable]
    [TreatAsDifferentiable]
    // [BackwardDerivative(_load_bwd_vector)]
    public float load<I : __BuiltinIntegerType>(vector<I, D> idx) {
        return data.load(idx);
    }

    // void _load_bwd_vector<I : __BuiltinIntegerType>(vector<I, D> idx, float.Differential grad)
    // {
    //     return;
    // }

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->float
    {
        [Differentiable] get { return load(indices); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->float
    {
        [Differentiable] get { return load(indices); }
    }
}

Tensor<float, 2> weights1;
Tensor<float, 1> bias1;

Tensor<float, 2> weights2;
Tensor<float, 1> bias2;

Tensor<float, 2> weights3;
Tensor<float, 1> bias3;

// OpenGL Fragment Shader Entry Point
[shader("fragment")]
void fragmentMain(
    in float2 textureCoords: TEXCOORD0,
    out float4 outputColor: SV_Target
) {
    // evaluate texture
    float prediction = evaluateNetwork(textureCoords,
                                       GLSLTensor<2>.fromTensor(weights1),
                                       GLSLTensor<1>.fromTensor(bias1),
                                       GLSLTensor<2>.fromTensor(weights2),
                                       GLSLTensor<1>.fromTensor(bias2),
                                       GLSLTensor<2>.fromTensor(weights3),
                                       GLSLTensor<1>.fromTensor(bias3)
                                    );
    outputColor = float4(prediction, prediction, prediction, 1.0);
}