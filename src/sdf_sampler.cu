#include <neural-graphics-primitives/sdf_sampler.h>

#include <neural-graphics-primitives/random_val.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>

#include <GL/glew.h>

NGP_NAMESPACE_BEGIN

void compute_distances_normals(
    cudaStream_t stream,
    tcnn::Network<float, precision_t>& network,
    SamplesSoa& samples,
    size_t start_index,
    size_t n_elements
) {
    // Implementation based on DifferentiableObject::input_gradient in tiny-cuda-nn/object.h
    constexpr float backprop_scale = 128.0f;
    n_elements = tcnn::next_multiple<size_t>(n_elements, tcnn::batch_size_granularity);

    tcnn::GPUMatrix<float> positions_matrix(
        (float*) (samples.points.data() + start_index), 
        3, n_elements
    );
    tcnn::GPUMatrix<float> normals_matrix(
        (float*) (samples.normals.data() + start_index),
        3, n_elements
    );
    // Row major layout to make casting simpler
    tcnn::GPUMatrix<float, tcnn::RM> distances_matrix(
        samples.distances.data() + start_index,
        1, n_elements
    );
    tcnn::GPUMatrix<precision_t> d_doutput = {
        network.padded_output_width(), (uint32_t) n_elements //, stream
    };
    // Same layout as distances_matrix
    tcnn::GPUMatrix<precision_t, tcnn::RM> output = {
        network.padded_output_width(), (uint32_t) n_elements //, stream
    };
    // Set "loss gradient" at network outputs to 1 at the chosen dimension and 0 elsewhere.
    // The chosen dimension in this case is 0.
    tcnn::one_hot_batched(
        stream,
        d_doutput.n_elements(),
        network.padded_output_width(),
        0,
        d_doutput.data(),
        backprop_scale
    );

    auto ctx = network.forward(
        stream, positions_matrix, &output,
        true /* inference matrices */, true /* prep forward buffers for input gradients */
    );
    network.backward(
        stream, *ctx,
        positions_matrix, output,
        d_doutput, &normals_matrix,
        true /* inference matrices */, false /* no param gradients */
    );
    
    tcnn::mult(stream, normals_matrix.n_elements(), normals_matrix.data(), 1.0f / backprop_scale);
    
    // See CutlassMLP::inference in tiny-cuda-nn/src/cutlass_mlp.cu
    tcnn::cast_from<<<tcnn::n_blocks_linear(n_elements), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, output.data(), distances_matrix.data()
    );
}

__global__ void move_along_normals_kernel(
    const size_t n_elements,
    Eigen::Vector3f* __restrict__ points,
    const float* __restrict__ distances,
    const Eigen::Vector3f* __restrict__ normals
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < n_elements) points[i] -= normals[i].normalized() * distances[i];
}

void move_along_normals(
    cudaStream_t stream,
    SamplesSoa& samples,
    size_t start_index,
    size_t n_elements
) {
    tcnn::linear_kernel(move_along_normals_kernel, 0, stream,
        n_elements,
        samples.points.data() + start_index,
        samples.distances.data() + start_index,
        samples.normals.data() + start_index
    );
}

void project_samples(
    cudaStream_t stream,
    tcnn::Network<float, precision_t>& network,
    SamplesSoa& samples,
    size_t start_index,
    size_t n_elements,
    int n_iter
) {
    #pragma unroll
    for(int i = 0; i < n_iter; ++i) {
        compute_distances_normals(stream, network, samples, start_index, n_elements);
        move_along_normals(stream, samples, start_index, n_elements);
    }
    compute_distances_normals(stream, network, samples, start_index, n_elements);
}

__global__ void gather_indices_kernel(
    size_t n_elements,
    const uint32_t* __restrict__ indices,
    const Eigen::Vector3f* __restrict__ in,
    Eigen::Vector3f* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n_elements) out[i] = in[indices[i]];
}

__global__ void filter_samples_kernel(
    const size_t n_elements,
    const BoundingBox bb,
    const float threshold,
    uint32_t* __restrict__ counter,
    const float* __restrict__ distances,
    const Eigen::Vector3f* __restrict__ in,
    Eigen::Vector3f* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements) return;

    Eigen::Vector3f point = in[i];
    if (bb.contains(point) && abs(distances[i]) < threshold) {
        uint32_t out_idx = atomicAdd(counter, 1);
        out[out_idx] = point;
    }
}

uint32_t SDFSampler::filter_samples(
    cudaStream_t stream,
    size_t start_index,
    size_t n_elements,
    float threshold
) {
    CUDA_CHECK_THROW(cudaMemsetAsync(m_counter.data(), 0, sizeof(uint32_t), stream));
    
    tcnn::linear_kernel(filter_samples_kernel, 0, stream,
        n_elements,
        m_aabb,
        threshold,
        m_counter.data(),
        m_samples.distances.data() + start_index,
        m_samples.points.data() + start_index,
        m_buffer.data()
    );

    uint32_t cpu_counter;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&cpu_counter, m_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    if (cpu_counter != n_elements) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            m_samples.points.data() + start_index,
            m_buffer.data(),
            cpu_counter * sizeof(Eigen::Vector3f),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
    
    return cpu_counter;
}

const char* vertexSource = R"foo(
#version 330 core
layout (location = 0) in vec3 pos;

uniform mat4 view;
uniform vec2 f;
uniform ivec2 res;

void main() {
    vec4 p = view * vec4(pos, 1.0);
	p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
	p.w = p.z;
	gl_Position = p;
}
)foo";

const char* fragmentSource = R"foo(
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(0.9, 0.2, 0.1, 1.0);
}
)foo";

void SamplesSoa::render(
    size_t n_elements,
    const Eigen::Vector2i& resolution,
    const Eigen::Vector2f& focal_length,
    const Eigen::Matrix<float, 3, 4>& camera_matrix
) const {
    GLuint VAO, VBO;
    std::unique_ptr<Eigen::Vector3f[]> cpu_points = std::make_unique<Eigen::Vector3f[]>(n_elements);
    CUDA_CHECK_THROW(cudaMemcpy(
        cpu_points.get(), points.data(), n_elements * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost
    ));

    // Setup Vertex Array
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    
    // Pass postition data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, n_elements * sizeof(Eigen::Vector3f), cpu_points.get(), GL_STATIC_DRAW);

    // Enable attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Eigen::Vector3f), reinterpret_cast<void*>(0));

    // Create shader
    GLuint vs, fs;
    GLuint prog;
    GLint success;
    char infoLog[512];

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vs, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    };

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fs, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    };

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Do the rendering
    Eigen::Matrix4f view2world = Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();

    glUseProgram(prog);

	glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, (GLfloat*) &world2view);
    glUniform2f(glGetUniformLocation(prog, "f"), focal_length.x(), focal_length.y());
	glUniform2i(glGetUniformLocation(prog, "res"), resolution.x(), resolution.y());

    glDrawArrays(GL_POINTS, 0, n_elements);

    glUseProgram(0);
    glBindVertexArray(0);

    // Clean up
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(prog);
}

NGP_NAMESPACE_END