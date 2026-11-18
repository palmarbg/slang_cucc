import slangpy;

#define LAYER_SIZE_1 2
#define LAYER_SIZE_2 32
#define LAYER_SIZE_3 32
#define LAYER_SIZE_4 32
#define LAYER_SIZE_5 32
#define LAYER_SIZE_6 3

// Activation function
[Differentiable]
float leaky_relu(float x)
{
    return x > 0.0 ? x : 0.01 * x;
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

    no_diff float2 uv,

    IDiffTensor<float, 2> weights1,
    IDiffTensor<float, 1> bias1,

    IDiffTensor<float, 2> weights2,
    IDiffTensor<float, 1> bias2,

    IDiffTensor<float, 2> weights3,
    IDiffTensor<float, 1> bias3,

    IDiffTensor<float, 2> weights4,
    IDiffTensor<float, 1> bias4,

    IDiffTensor<float, 2> weights5,
    IDiffTensor<float, 1> bias5
) {
    // Read the input uv coordinates for this thread
    // float2 uv = float2(uvs[index, 0], uvs[index, 1]);

    return evaluateNetwork(uv,
                    weights1,
                    bias1,
                    weights2,
                    bias2,
                    weights3,
                    bias3,
                    weights4,
                    bias4,
                    weights5,
                    bias5
                );
}

[Differentiable]
public float[3] evaluateNetwork(
    no_diff float2 uv,

    IDiffTensor<float, 2> weights1,
    IDiffTensor<float, 1> bias1,

    IDiffTensor<float, 2> weights2,
    IDiffTensor<float, 1> bias2,

    IDiffTensor<float, 2> weights3,
    IDiffTensor<float, 1> bias3,

    IDiffTensor<float, 2> weights4,
    IDiffTensor<float, 1> bias4,

    IDiffTensor<float, 2> weights5,
    IDiffTensor<float, 1> bias5
)
{
    // Layer 1: 2 -> 32
    float h1[LAYER_SIZE_2];
    
    for (int i = 0; i < LAYER_SIZE_2; ++i)
    {
        float value = bias1[i];

        for (int j = 0; j < LAYER_SIZE_1; j++)
        {
            value += weights1[i, j] * uv[j];
        }
        
        h1[i] = leaky_relu(value);
    }

    // Layer 2: 32 -> 32
    float h2[LAYER_SIZE_3];

    for (int i = 0; i < LAYER_SIZE_3; ++i)
    {
        float value = bias2[i];

        for (int j = 0; j < LAYER_SIZE_2; j++)
        {
            value += weights2[i, j] * h1[j];
        }

        h2[i] = leaky_relu(value);
    }

    // Layer 3: 32 -> 32
    float h3[LAYER_SIZE_4];

    for (int i = 0; i < LAYER_SIZE_4; ++i)
    {
        float value = bias3[i];

        for (int j = 0; j < LAYER_SIZE_3; j++)
        {
            value += weights3[i, j] * h2[j];
        }

        h3[i] = leaky_relu(value);
    }

    // Layer 4: 32 -> 32
    float h4[LAYER_SIZE_5];

    for (int i = 0; i < LAYER_SIZE_5; ++i)
    {
        float value = bias4[i];

        for (int j = 0; j < LAYER_SIZE_4; j++)
        {
            value += weights4[i, j] * h3[j];
        }

        h4[i] = leaky_relu(value);
    }

    // Layer 5: 32 -> 3
    float h5[LAYER_SIZE_6];

    for (int i = 0; i < LAYER_SIZE_6; ++i)
    {
        float value = bias5[i];

        for (int j = 0; j < LAYER_SIZE_5; j++)
        {
            value += weights5[i, j] * h4[j];
        }

        h5[i] = sigmoid(value);
    }

    return h5;
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
    __init(const float data[D], const uint shape[1]) {
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
    __init(const float data[D], const uint shape[2]) {
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
#define WEIGHTS1_DATA_V { 0 }
#define BIAS1_DATA_V { 0 }
#define WEIGHTS2_DATA_V { 0 }
#define BIAS2_DATA_V { 0 }
#define WEIGHTS3_DATA_V { 0 }
#define BIAS3_DATA_V { 0 }
#define WEIGHTS4_DATA_V { 0 }
#define BIAS4_DATA_V { 0 }
#define WEIGHTS5_DATA_V { 0 }
#define BIAS5_DATA_V { 0 }
#else
#include "definitions.slangh"
#endif


// Layer 1: 2 -> 32
static const float WEIGHTS1_DATA[LAYER_SIZE_2*LAYER_SIZE_1] = WEIGHTS1_DATA_V;
static const float BIAS1_DATA[LAYER_SIZE_2] = BIAS1_DATA_V;

// Layer 2: 32 -> 32
static const float WEIGHTS2_DATA[LAYER_SIZE_3*LAYER_SIZE_2] = WEIGHTS2_DATA_V;
static const float BIAS2_DATA[LAYER_SIZE_3] = BIAS2_DATA_V;

// Layer 3: 32 -> 32
static const float WEIGHTS3_DATA[LAYER_SIZE_4*LAYER_SIZE_3] = WEIGHTS3_DATA_V;
static const float BIAS3_DATA[LAYER_SIZE_4] = BIAS3_DATA_V;

// Layer 4: 32 -> 32
static const float WEIGHTS4_DATA[LAYER_SIZE_5*LAYER_SIZE_4] = WEIGHTS4_DATA_V;
static const float BIAS4_DATA[LAYER_SIZE_5] = BIAS4_DATA_V;

// Layer 5: 32 -> 3
static const float WEIGHTS5_DATA[LAYER_SIZE_6*LAYER_SIZE_5] = WEIGHTS5_DATA_V;
static const float BIAS5_DATA[LAYER_SIZE_6] = BIAS5_DATA_V;

// OpenGL Fragment Shader Entry Point
[shader("fragment")]
void fragmentMain(
    in float2 textureCoords: TEXCOORD0,
    out float4 outputColor: SV_Target
) {
    // evaluate texture
    float[3] prediction = evaluateNetwork(textureCoords,
                                          GLSLMatrixWrapper<LAYER_SIZE_2 * LAYER_SIZE_1>(WEIGHTS1_DATA, { LAYER_SIZE_2, LAYER_SIZE_1 }),
                                          GLSLVectorWrapper<LAYER_SIZE_2>(BIAS1_DATA, { LAYER_SIZE_2 }),

                                          GLSLMatrixWrapper<LAYER_SIZE_3 * LAYER_SIZE_2>(WEIGHTS2_DATA, { LAYER_SIZE_3, LAYER_SIZE_2 }),
                                          GLSLVectorWrapper<LAYER_SIZE_3>(BIAS2_DATA, { LAYER_SIZE_3 }),

                                          GLSLMatrixWrapper<LAYER_SIZE_4 * LAYER_SIZE_3>(WEIGHTS3_DATA, { LAYER_SIZE_4, LAYER_SIZE_3 }),
                                          GLSLVectorWrapper<LAYER_SIZE_4>(BIAS3_DATA, { LAYER_SIZE_4 }),

                                          GLSLMatrixWrapper<LAYER_SIZE_5 * LAYER_SIZE_4>(WEIGHTS4_DATA, { LAYER_SIZE_5, LAYER_SIZE_4 }),
                                          GLSLVectorWrapper<LAYER_SIZE_5>(BIAS4_DATA, { LAYER_SIZE_5 }),

                                          GLSLMatrixWrapper<LAYER_SIZE_6 * LAYER_SIZE_5>(WEIGHTS5_DATA, { LAYER_SIZE_6, LAYER_SIZE_5 }),
                                          GLSLVectorWrapper<LAYER_SIZE_6>(BIAS5_DATA, { LAYER_SIZE_6 }),
                                    );

    outputColor = float4(prediction[0], prediction[1], prediction[2], 1.0);
}