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
public float[3] forward(
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
public float[3] evaluateNetwork(
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

    // Layer 3: 32 -> 3
    float h3[3];

    for (int i = 0; i < 3; ++i)
    {
        float value = bias3[i];

        for (int j = 0; j < 32; j++)
        {
            value += weights3[i, j] * h2[j];
        }

        h3[i] = sigmoid(value);
    }

    return h3;
}

/************************************** FRAGMENT SHADER **************************************/

interface IStaticInt
{
    static int get();
}

struct IntConst<let N : int> : IStaticInt
{
    static int get() { return N; }
}

public struct GLSLVectorWrapper<let D: int> : IDiffTensor<float, 1>
{
    public float data[D];
    public uint[1] _shape;
    public uint[1] _stride = { 1 };
    public property shape : uint[1] { [ForceInline] get { return _shape; } }

    [ForceInline]
    __init(const float data[D], uint shape[1]) {
        this.data = data;
        this._shape = shape;
    }

    // float _read_primal_each<I : __BuiltinIntegerType>() {
    //     return 0;
    // }

    // float _read_primal_each<each I : __BuiltinIntegerType>(I idx) where nonempty(I) {
    //     return (*data)[(__first(idx)).toInt()];
    // }

    [ForceInline]
    float _read_primal_each<each I : __BuiltinIntegerType>(I idx) {
        int lin_idx = _idx_each(expand each idx, _stride, 0);
        return data[lin_idx];
    }

    [ForceInline]
    void _write_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    [ForceInline]
    void _accumulate_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    [Differentiable]
    [TreatAsDifferentiable]
    [ForceInline]
    public float load<I : __BuiltinIntegerType>(I idx[1]) {
        return data[idx[0].toInt()];
    }

    [Differentiable]
    [TreatAsDifferentiable]
    [ForceInline]
    public float load<I : __BuiltinIntegerType>(vector<I, 1> idx) {
        return data[idx[0].toInt()];
    }

    public __subscript<I : __BuiltinIntegerType>(I indices[1])->float
    {
        [Differentiable] [ForceInline] get { return load(indices); }
    }

    public __subscript<I : __BuiltinIntegerType>(vector<I, 1> indices)->float
    {
        [Differentiable] [ForceInline] get { return load(indices); }
    }
}

public struct GLSLMatrixWrapper<let D:int> : IDiffTensor<float, 2>
{
    public float data[D];
    public uint[2] _shape;
    public uint[2] _stride;
    public property shape : uint[2] { [ForceInline] get { return _shape; } }

    [ForceInline]
    __init(const float data[D], uint shape[2]) {
        this.data = data;
        this._shape = shape;
        this._stride = { shape[1], 1 };
    }

    [ForceInline]
    float _read_primal_each<each I : __BuiltinIntegerType>(I idx) {
        int lin_idx = _idx_each(expand each idx, _stride, 0);
        return data[lin_idx];
    }

    [ForceInline]
    void _write_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    [ForceInline]
    void _accumulate_grad_each<each I : __BuiltinIntegerType>(float.Differential grad, expand each I idx) {
        return;
    }

    [Differentiable]
    [TreatAsDifferentiable]
    [ForceInline]
    public float load<I : __BuiltinIntegerType>(I idx[2]) {
        return data[idx[0].toInt() * _stride[0] + idx[1].toInt() * _stride[1]];
    }

    [Differentiable]
    [TreatAsDifferentiable]
    [ForceInline]
    public float load<I : __BuiltinIntegerType>(vector<I, 2> idx) {
        return data[idx[0].toInt() * _stride[0] + idx[1].toInt() * _stride[1]];
    }

    public __subscript<I : __BuiltinIntegerType>(I indices[2])->float
    {
        [Differentiable] [ForceInline] get { return load(indices); }
    }

    public __subscript<I : __BuiltinIntegerType>(vector<I, 2> indices)->float
    {
        [Differentiable] [ForceInline] get { return load(indices); }
    }
}

#ifndef USE_DEFINITON_FILE
#define WEIGHTS1_DATA_V { 0}
#define BIAS1_DATA_V { 0}
#define WEIGHTS2_DATA_V { 0}
#define BIAS2_DATA_V { 0}
#define WEIGHTS3_DATA_V { 0}
#define BIAS3_DATA_V { 0}
#else
#include "definitions.slangh"
#endif


// Layer 1: 2 -> 32
static const float WEIGHTS1_DATA[32*2] = WEIGHTS1_DATA_V;
static const float BIAS1_DATA[32] = BIAS1_DATA_V;

// Layer 2: 32 -> 32
static const float WEIGHTS2_DATA[32*32] = WEIGHTS2_DATA_V;
static const float BIAS2_DATA[32] = BIAS2_DATA_V;

// Layer 3: 32 -> 1
static const float WEIGHTS3_DATA[3*32] = WEIGHTS3_DATA_V;
static const float BIAS3_DATA[3] = BIAS3_DATA_V;

// OpenGL Fragment Shader Entry Point
[shader("fragment")]
void fragmentMain(
    in float2 textureCoords: TEXCOORD0,
    out float4 outputColor: SV_Target
) {
    // evaluate texture
    float[3] prediction = evaluateNetwork(textureCoords,
                                       GLSLMatrixWrapper<32*2>(WEIGHTS1_DATA, { 32, 2 }),
                                       GLSLVectorWrapper<32>(BIAS1_DATA, { 32 }),
                                       GLSLMatrixWrapper<32*32>(WEIGHTS2_DATA, { 32, 32 }),
                                       GLSLVectorWrapper<32>(BIAS2_DATA, { 32 }),
                                       GLSLMatrixWrapper<3*32>(WEIGHTS3_DATA, { 3, 32}),
                                       GLSLVectorWrapper<3>(BIAS3_DATA, { 3 }),
                                    );
    outputColor = float4(prediction[0], prediction[1], prediction[2], 1.0);
}