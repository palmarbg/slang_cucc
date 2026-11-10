#version 450
layout(row_major) uniform;
layout(row_major) buffer;

#line 286 0
struct Tensor_0
{
    uint  _shape_0[2];
    uint  _strides_0[2];
    uint _offset_0;
};


#line 286
struct Tensor_1
{
    uint  _shape_0[1];
    uint  _strides_0[1];
    uint _offset_0;
};


#line 286
struct GlobalParams_0
{
    Tensor_0 weights1_0;
    Tensor_1 bias1_0;
    Tensor_0 weights2_0;
    Tensor_1 bias2_0;
    Tensor_0 weights3_0;
    Tensor_1 bias3_0;
};


#line 1084 1
layout(binding = 0)
layout(std140) uniform block_GlobalParams_0
{
    Tensor_0 weights1_0;
    Tensor_1 bias1_0;
    Tensor_0 weights2_0;
    Tensor_1 bias2_0;
    Tensor_0 weights3_0;
    Tensor_1 bias3_0;
}globalParams_0;

#line 293 0
layout(std430, binding = 1) readonly buffer StructuredBuffer_float_t_0 {
    float _data[];
} globalParams_weights1_data_0;

#line 293
layout(std430, binding = 2) readonly buffer StructuredBuffer_float_t_1 {
    float _data[];
} globalParams_bias1_data_0;

#line 293
layout(std430, binding = 3) readonly buffer StructuredBuffer_float_t_2 {
    float _data[];
} globalParams_weights2_data_0;

#line 293
layout(std430, binding = 4) readonly buffer StructuredBuffer_float_t_3 {
    float _data[];
} globalParams_bias2_data_0;

#line 293
layout(std430, binding = 5) readonly buffer StructuredBuffer_float_t_4 {
    float _data[];
} globalParams_weights3_data_0;

#line 293
layout(std430, binding = 6) readonly buffer StructuredBuffer_float_t_5 {
    float _data[];
} globalParams_bias3_data_0;

#line 34 2
void _idx_accumulate_helper_0(int value_0, inout uint dest_0, inout uint dim_0, uint  stride_0[1])
{
    uint _S1 = uint(value_0);

#line 36
    uint _S2 = dim_0;

#line 36
    dim_0 = dim_0 + 1U;

#line 36
    dest_0 = dest_0 + _S1 * stride_0[_S2];
    return;
}


#line 34
void _idx_accumulate_helper_1(int value_1, inout uint dest_1, inout uint dim_1, uint  stride_1[2])
{
    uint _S3 = uint(value_1);

#line 36
    uint _S4 = dim_1;

#line 36
    dim_1 = dim_1 + 1U;

#line 36
    dest_1 = dest_1 + _S3 * stride_1[_S4];
    return;
}


#line 38
int _idx_each_0(int _S5, int _S6, uint  stride_2[2], uint offset_0)
{

    uint res_0 = 0U;
    uint dim_2 = 0U;
    _idx_accumulate_helper_1(_S5, res_0, dim_2, stride_2);

#line 43
    _idx_accumulate_helper_1(_S6, res_0, dim_2, stride_2);
    return int(res_0 + offset_0);
}


#line 38
int _idx_each_1(int _S7, uint  stride_3[1], uint offset_1)
{

    uint res_1 = 0U;
    uint dim_3 = 0U;
    _idx_accumulate_helper_0(_S7, res_1, dim_3, stride_3);
    return int(res_1 + offset_1);
}


#line 5 3
float leaky_relu_0(float x_0)
{

#line 5
    float _S8;

    if(x_0 > 0.0)
    {

#line 7
        _S8 = x_0;

#line 7
    }
    else
    {

#line 7
        _S8 = 0.10000000149011612 * x_0;

#line 7
    }

#line 7
    return _S8;
}


#line 17
float sigmoid_0(float x_1)
{
    return 1.0 / (1.0 + exp(- x_1));
}


#line 754 1
layout(location = 0)
in vec2 textureCoords_0;


#line 182 3
layout(location = 0)
out vec4 outputColor_0;


#line 113
struct GLSLTensor_0
{
    Tensor_0 data_0;
    uint  _shape_1[2];
};


#line 113
GLSLTensor_0 GLSLTensor_fromTensor_0()
{

#line 113
    GLSLTensor_0 instance_0;

#line 113
    instance_0.data_0 = globalParams_0.weights1_0;

#line 122
    instance_0._shape_1 = globalParams_0.weights1_0._shape_0;

#line 122
    return instance_0;
}


#line 113
struct GLSLTensor_1
{
    Tensor_1 data_0;
    uint  _shape_1[1];
};


#line 113
GLSLTensor_1 GLSLTensor_fromTensor_1()
{

#line 113
    GLSLTensor_1 instance_1;

#line 113
    instance_1.data_0 = globalParams_0.bias1_0;

#line 122
    instance_1._shape_1 = globalParams_0.bias1_0._shape_0;

#line 122
    return instance_1;
}


#line 122
GLSLTensor_0 GLSLTensor_fromTensor_2()
{

#line 122
    GLSLTensor_0 instance_2;

#line 122
    instance_2.data_0 = globalParams_0.weights2_0;

#line 122
    instance_2._shape_1 = globalParams_0.weights2_0._shape_0;

#line 122
    return instance_2;
}


#line 122
GLSLTensor_1 GLSLTensor_fromTensor_3()
{

#line 122
    GLSLTensor_1 instance_3;

#line 122
    instance_3.data_0 = globalParams_0.bias2_0;

#line 122
    instance_3._shape_1 = globalParams_0.bias2_0._shape_0;

#line 122
    return instance_3;
}


#line 122
GLSLTensor_0 GLSLTensor_fromTensor_4()
{

#line 122
    GLSLTensor_0 instance_4;

#line 122
    instance_4.data_0 = globalParams_0.weights3_0;

#line 122
    instance_4._shape_1 = globalParams_0.weights3_0._shape_0;

#line 122
    return instance_4;
}


#line 122
GLSLTensor_1 GLSLTensor_fromTensor_5()
{

#line 122
    GLSLTensor_1 instance_5;

#line 122
    instance_5.data_0 = globalParams_0.bias3_0;

#line 122
    instance_5._shape_1 = globalParams_0.bias3_0._shape_0;

#line 122
    return instance_5;
}


#line 1363 4
float Tensor_read_buffer_0(Tensor_1 _S9, int _S10)
{

#line 293 0
    return globalParams_bias1_data_0._data[uint(_S10)];
}


#line 293
float GLSLTensor_read_primal_each_0(GLSLTensor_1 _S11, int _S12)
{

#line 127 3
    return Tensor_read_buffer_0(_S11.data_0, _idx_each_1(_S12, _S11.data_0._strides_0, _S11.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_0(GLSLTensor_1 _S13, int _S14)
{

#line 446 5
    return GLSLTensor_read_primal_each_0(_S13, _S14);
}


#line 446
float Tensor_read_buffer_1(Tensor_0 _S15, int _S16)
{

#line 293 0
    return globalParams_weights1_data_0._data[uint(_S16)];
}


#line 293
float GLSLTensor_read_primal_each_1(GLSLTensor_0 _S17, int _S18, int _S19)
{

#line 127 3
    return Tensor_read_buffer_1(_S17.data_0, _idx_each_0(_S18, _S19, _S17.data_0._strides_0, _S17.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_1(GLSLTensor_0 _S20, int _S21, int _S22)
{

#line 466 5
    return GLSLTensor_read_primal_each_1(_S20, _S21, _S22);
}


#line 466
float Tensor_read_buffer_2(Tensor_1 _S23, int _S24)
{

#line 293 0
    return globalParams_bias2_data_0._data[uint(_S24)];
}


#line 293
float GLSLTensor_read_primal_each_2(GLSLTensor_1 _S25, int _S26)
{

#line 127 3
    return Tensor_read_buffer_2(_S25.data_0, _idx_each_1(_S26, _S25.data_0._strides_0, _S25.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_2(GLSLTensor_1 _S27, int _S28)
{

#line 446 5
    return GLSLTensor_read_primal_each_2(_S27, _S28);
}


#line 446
float Tensor_read_buffer_3(Tensor_0 _S29, int _S30)
{

#line 293 0
    return globalParams_weights2_data_0._data[uint(_S30)];
}


#line 293
float GLSLTensor_read_primal_each_3(GLSLTensor_0 _S31, int _S32, int _S33)
{

#line 127 3
    return Tensor_read_buffer_3(_S31.data_0, _idx_each_0(_S32, _S33, _S31.data_0._strides_0, _S31.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_3(GLSLTensor_0 _S34, int _S35, int _S36)
{

#line 466 5
    return GLSLTensor_read_primal_each_3(_S34, _S35, _S36);
}


#line 1363 4
float Tensor_read_buffer_4(Tensor_1 _S37, int _S38)
{

#line 293 0
    return globalParams_bias3_data_0._data[uint(_S38)];
}


#line 293
float GLSLTensor_read_primal_each_4(GLSLTensor_1 _S39, int _S40)
{

#line 127 3
    return Tensor_read_buffer_4(_S39.data_0, _idx_each_1(_S40, _S39.data_0._strides_0, _S39.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_4(GLSLTensor_1 _S41, int _S42)
{

#line 446 5
    return GLSLTensor_read_primal_each_4(_S41, _S42);
}


#line 446
float Tensor_read_buffer_5(Tensor_0 _S43, int _S44)
{

#line 293 0
    return globalParams_weights3_data_0._data[uint(_S44)];
}


#line 293
float GLSLTensor_read_primal_each_5(GLSLTensor_0 _S45, int _S46, int _S47)
{

#line 127 3
    return Tensor_read_buffer_5(_S45.data_0, _idx_each_0(_S46, _S47, _S45.data_0._strides_0, _S45.data_0._offset_0));
}


#line 127
float operatorx5Bx5D_get_5(GLSLTensor_0 _S48, int _S49, int _S50)
{

#line 466 5
    return GLSLTensor_read_primal_each_5(_S48, _S49, _S50);
}


#line 466
float evaluateNetwork_0(vec2 _S51, GLSLTensor_0 _S52, GLSLTensor_1 _S53, GLSLTensor_0 _S54, GLSLTensor_1 _S55, GLSLTensor_0 _S56, GLSLTensor_1 _S57)
{

#line 466
    int j_0;

#line 466
    float value_2;

#line 64 3
    float  h1_0[32];

#line 64
    int i_0 = 0;

    for(;;)
    {

#line 66
        if(i_0 < 32)
        {
        }
        else
        {

#line 66
            break;
        }

#line 66
        float _S58 = operatorx5Bx5D_get_0(_S53, i_0);

#line 66
        j_0 = 0;

#line 66
        value_2 = _S58;



        for(;;)
        {

#line 70
            if(j_0 < 2)
            {
            }
            else
            {

#line 70
                break;
            }
            float value_3 = value_2 + operatorx5Bx5D_get_1(_S52, i_0, j_0) * _S51[j_0];

#line 70
            j_0 = j_0 + 1;

#line 70
            value_2 = value_3;

#line 70
        }

#line 75
        h1_0[i_0] = leaky_relu_0(value_2);

#line 66
        i_0 = i_0 + 1;

#line 66
    }

#line 79
    float  h2_0[32];

#line 79
    i_0 = 0;

    for(;;)
    {

#line 81
        if(i_0 < 32)
        {
        }
        else
        {

#line 81
            break;
        }

#line 81
        float _S59 = operatorx5Bx5D_get_2(_S55, i_0);

#line 81
        j_0 = 0;

#line 81
        value_2 = _S59;



        for(;;)
        {

#line 85
            if(j_0 < 32)
            {
            }
            else
            {

#line 85
                break;
            }
            float value_4 = value_2 + operatorx5Bx5D_get_3(_S54, i_0, j_0) * h1_0[j_0];

#line 85
            j_0 = j_0 + 1;

#line 85
            value_2 = value_4;

#line 85
        }

#line 90
        h2_0[i_0] = leaky_relu_0(value_2);

#line 81
        i_0 = i_0 + 1;

#line 81
    }

#line 94
    float  h3_0[1];

#line 94
    i_0 = 0;

    for(;;)
    {

#line 96
        if(i_0 < 1)
        {
        }
        else
        {

#line 96
            break;
        }

#line 96
        float _S60 = operatorx5Bx5D_get_4(_S57, i_0);

#line 96
        j_0 = 0;

#line 96
        value_2 = _S60;



        for(;;)
        {

#line 100
            if(j_0 < 32)
            {
            }
            else
            {

#line 100
                break;
            }
            float value_5 = value_2 + operatorx5Bx5D_get_5(_S56, i_0, j_0) * h2_0[j_0];

#line 100
            j_0 = j_0 + 1;

#line 100
            value_2 = value_5;

#line 100
        }

#line 105
        h3_0[i_0] = sigmoid_0(value_2);

#line 96
        i_0 = i_0 + 1;

#line 96
    }

#line 108
    return h3_0[0];
}


#line 182
void main()
{

#line 182
    float _S61 = evaluateNetwork_0(textureCoords_0, GLSLTensor_fromTensor_0(), GLSLTensor_fromTensor_1(), GLSLTensor_fromTensor_2(), GLSLTensor_fromTensor_3(), GLSLTensor_fromTensor_4(), GLSLTensor_fromTensor_5());

#line 182
    outputColor_0 = vec4(_S61, _S61, _S61, 1.0);

#line 196
    return;
}

