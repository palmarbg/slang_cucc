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
    // linear layers' input and output
    float h[MAX_LAYER_COUNT][MAX_LAYER_SIZE] = { 0 };

    // current vector size
    no_diff int height_current = layers[0];

    // Read the input uv coordinates for this thread
    for (int i = 0; i < MAX_LAYER_SIZE; ++i) {
        if (i < height_current)
            h[0][i] = uvs[index, i];
    }

    no_diff int weight_ptr = 0;
    // iterate layers

    for (int l = 1; l < MAX_LAYER_COUNT; ++l)
    {
        if (l < layers.shape[0])
        {
            const int layer_type = layers[l];

            // if linear layer
            if (layer_type > 0)
            {
                for (int o = 0; o < MAX_LAYER_SIZE; ++o)
                {
                    if (o < layer_type)
                    {
                        float value = weights[0];

                        // for (int i = 0; i < MAX_LAYER_SIZE; i++)
                        // {
                        //     if (i < height_current)
                        //         value += weights[get_weight_index(weight_ptr, layer_type, o, i)] * h[l-1][i];
                        // }

                        h[l][o] = value;
                    }
                }
                weight_ptr += layer_type * height_current + layer_type;
                height_current = layer_type;
            } else if (layer_type == LAYER_ACT_LEAKY_RELU)
            {
                for (int i = 0; i < MAX_LAYER_SIZE; i++)
                {
                    h[l][i] = (i < height_current) ? leaky_relu(h[l - 1][i]) : 0;
                }
            } else if (layer_type == LAYER_ACT_TANH)
            {
                for (int i = 0; i < MAX_LAYER_SIZE; i++)
                {
                    h[l][i] = (i < height_current) ? tanh_act(h[l - 1][i]) : 0;
                }
            } else if (layer_type == LAYER_ACT_SIGMOID)
            {
                for (int i = 0; i < MAX_LAYER_SIZE; i++)
                {
                    h[l][i] = (i < height_current) ? sigmoid(h[l - 1][i]) : 0;
                }
            }
        }
    }

    return h[layers.shape[0] - 1][0];
}

/*
[Differentiable]
public float forward(
    no_diff int index,
    no_diff ITensor<float, 2> uvs,
    IDiffTensor<float, 1> weights,
    no_diff ITensor<int, 1> layers
)
{
    // linear layers' input and output
    float h_in[MAX_LAYER_SIZE] = { 0 };
    float h_out[MAX_LAYER_SIZE] = { 0 };

    // current vector size
    int height_current = layers[0];

    // Read the input uv coordinates for this thread
    [ForceUnroll]
    for (int i = 0; i < MAX_LAYER_SIZE; ++i) {
        if (i < layers[0])
            h_in[i] = uvs[index, i];
    }

    int weight_ptr = 0;
    // iterate layers
    [ForceUnroll]
    for (int l = 1; l < MAX_LAYER_COUNT && l < layers.shape[0]; ++l)
    {
        const int layer_type = layers[l];

        // if linear layer
        if (layer_type > 0)
        {
            [ForceUnroll]
            for (int o = 0; o < MAX_LAYER_SIZE; ++o)
            {
                if (o < layer_type) {
                    h_out[o] = weights[get_bias_index(weight_ptr, layer_type, height_current, o)];

                    [ForceUnroll]
                    for (int i = 0; i < MAX_LAYER_SIZE; i++)
                    {
                        if (i < height_current)
                            h_out[o] += weights[get_weight_index(weight_ptr, layer_type, o, i)] * h_in[i];
                    }
                }
            }
            [ForceUnroll]
            for (int i = 0; i < MAX_LAYER_SIZE; i++)
            {
                if (i < layer_type)
                    h_in[i] = h_out[i];
            }
            weight_ptr += layer_type * height_current + layer_type;
            height_current = layer_type;
        } else if (layer_type == LAYER_ACT_LEAKY_RELU)
        {
            [ForceUnroll]
            for (int i = 0; i < MAX_LAYER_SIZE; i++)
            {
                if(i < height_current)
                    h_in[i] = leaky_relu(h_in[i]);
            }
        } else if (layer_type == LAYER_ACT_TANH)
        {
            [ForceUnroll]
            for (int i = 0; i < MAX_LAYER_SIZE; i++)
            {
                if(i < height_current)
                    h_in[i] = tanh_act(h_in[i]);
            }
        } else if (layer_type == LAYER_ACT_SIGMOID)
        {
            [ForceUnroll]
            for (int i = 0; i < MAX_LAYER_SIZE; i++)
            {
                if(i < height_current)
                    h_in[i] = sigmoid(h_in[i]);
            }
        }
    }

    return h_in[0];
}*/

/*
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
no_diff uint get_weight_index(int weight_ptr, int w, int i, int j) {
    return weight_ptr + i * w + j;
}

[ForceInline]
no_diff uint get_bias_index(int weight_ptr, int w, int h, int i) {
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
    // linear layers' input and output
    float h[MAX_LAYER_COUNT][MAX_LAYER_SIZE] = { 0 };

    // current vector size
    int height_current = layers[0];

    // Read the input uv coordinates for this thread
    for (int i = 0; i < MAX_LAYER_SIZE; ++i) {
        if(i < height_current)
            h[0][i] = uvs[index, i];
    }

    int weight_ptr = 0;
    // iterate layers

    for (int l = 1; l < MAX_LAYER_COUNT; ++l)
    {
        if (l == layers.shape[0])
            break;

        const int layer_type = layers[l];

        // if linear layer
        if (layer_type > 0)
        {
            for (int o = 0; o < MAX_LAYER_SIZE ; ++o)
{
    if (o < layer_type)
    {
        float value = weights[0];

        // for (int i = 0; i < MAX_LAYER_SIZE; i++)
        // {
        //     if (i < height_current)
        //         value += weights[get_weight_index(weight_ptr, layer_type, o, i)] * h[l-1][i];
        // }

        h[l][o] = value;
    }
}
weight_ptr += layer_type * height_current + layer_type;
height_current = layer_type;
} else if (layer_type == LAYER_ACT_LEAKY_RELU)
{
    for (int i = 0; i < MAX_LAYER_SIZE; i++)
    {
        h[l][i] = (i < height_current) ? leaky_relu(h[l - 1][i]) : 0;
    }
} else if (layer_type == LAYER_ACT_TANH)
{
    for (int i = 0; i < MAX_LAYER_SIZE; i++)
    {
        h[l][i] = (i < height_current) ? tanh_act(h[l - 1][i]) : 0;
    }
} else if (layer_type == LAYER_ACT_SIGMOID)
{
    for (int i = 0; i < MAX_LAYER_SIZE; i++)
    {
        h[l][i] = (i < height_current) ? sigmoid(h[l - 1][i]) : 0;
    }
}
}

return h[layers.shape[0] - 1][0];
}

*/

/*
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
uint get_weight_index(int weight_ptr, int w, int i, int j) {
    return weight_ptr + i * w + j;
}

[ForceInline]
uint get_bias_index(int weight_ptr, int w, int h, int i) {
    return weight_ptr + w * h + i;
}

[Differentiable]
public float forward(
    no_diff int index,
    no_diff ITensor<float, 2> uvs,
    IDiffTensor<float, 1> weights,
    no_diff ITensor<int, 1> layers
)
{
    // linear layers' input and output
    float h_in[MAX_LAYER_SIZE] = { 0 };
    float h_out[MAX_LAYER_SIZE] = { 0 };

    // current vector size
    int height_current = layers[0];

    // Read the input uv coordinates for this thread
    [ForceUnroll]
    for (int i = 0; i < layers[0]; ++i) {
        h_in[i] = uvs[index, i];
    }

    int weight_ptr = 0;
    // iterate layers
    [MaxIters(MAX_LAYER_COUNT)]
    for (int l = 1; l < layers.shape[0]; ++l)
    {
        const int layer_type = layers[l];

        // if linear layer
        if (layer_type > 0)
        {
            [ForceUnroll]
            for (int o = 0; o < layer_type  ; ++o)
{
    h_out[o] = weights[get_bias_index(weight_ptr, layer_type, height_current, o)];

    [ForceUnroll]
    for (int i = 0; i < height_current; i++)
    {
        h_out[o] += weights[get_weight_index(weight_ptr, layer_type , o, i)] * h_in[i];
    }
}
[ForceUnroll]
for (int i = 0; i < layer_type; i++)
{
    h_in[i] = h_out[i];
}
weight_ptr += layer_type * height_current + layer_type;
height_current = layer_type;
} else if (layer_type == LAYER_ACT_LEAKY_RELU)
{
    [ForceUnroll]
    for (int i = 0; i < height_current; i++)
    {
        h_in[i] = leaky_relu(h_in[i]);
    }
} else if (layer_type == LAYER_ACT_TANH)
{
    [ForceUnroll]
    for (int i = 0; i < height_current; i++)
    {
        h_in[i] = tanh_act(h_in[i]);
    }
} else if (layer_type == LAYER_ACT_SIGMOID)
{
    [ForceUnroll]
    for (int i = 0; i < height_current; i++)
    {
        h_in[i] = sigmoid(h_in[i]);
    }
}
}

return h_in[0];
}
*/