import slangpy;

#define MAX_LAYER_SIZE 32
#define MAX_LAYER_COUNT 20

#define LAYER_ACT_LEAKY_RELU -1
#define LAYER_ACT_TANH -2
#define LAYER_ACT_SIGMOID -3


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

[ForceInline]
no_diff int get_weight_index(no_diff int weight_ptr, no_diff int w, no_diff int i, no_diff int j) {
    return weight_ptr + i * w + j;
}

[ForceInline]
no_diff int get_bias_index(no_diff int weight_ptr, no_diff int w, no_diff int h, no_diff int i) {
    return weight_ptr + w * h + i;
}

[Differentiable]
public float forward(
    no_diff int index,
    IDiffTensor<float, 2> uvs,
    IDiffTensor<float, 1> weights,
    no_diff ITensor<int, 1> layers
)
{
    // Read the input uv coordinates for this thread
    float2 uv = float2(uvs[index, 0], uvs[index, 1]);

    // Layer 1: 2 -> 32
    float h1[32] = {0};

    for (int i = 0; i < 32; ++i)
    {
        if(layers[i%2] == 0){
        float value = weights[get_bias_index(0, 2, 32, i)];

        for (int j = 0; j < 2; j++)
        {
            value += weights[get_weight_index(0, 2, i, j)] * uv[j];
        }

        h1[i] = leaky_relu(value);
        }
    }

    // Layer 2: 32 -> 32
    float h2[32];

    for (int i = 0; i < 32; ++i)
    {
        float value = weights[get_bias_index(2 * 32 + 32, 32, 32, i)];

        for (int j = 0; j < 32; j++)
        {
            value += weights[get_weight_index(2*32+32, 32, i, j)] * h1[j];
        }

        h2[i] = leaky_relu(value);
    }

    // Layer 3: 32 -> 1
    float h3[1];

    for (int i = 0; i < 1; ++i)
    {
        float value = weights[get_bias_index(2 * 32 + 32 + 32 * 32 + 32, 32, 1, i)];

        for (int j = 0; j < 32; j++)
        {
            value += weights[get_weight_index(2*32+32 + 32*32+32, 32, i, j)] * h2[j];
        }

        h3[i] = sigmoid(value);
    }

    return h3[0];
}
