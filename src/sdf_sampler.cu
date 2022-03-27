#include <neural-graphics-primitives/sdf_sampler.h>

#include <neural-graphics-primitives/random_val.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>

#include <GL/glew.h>

NGP_NAMESPACE_BEGIN

// Refinement steps when initializing
constexpr int INIT_REFINEMENT_STEPS = 7;
// Refinement steps when sampling
constexpr int SAMPLING_REFINEMENT_STEPS = 7;
constexpr float REFINEMENT_THRESHOLD = 0.03;
constexpr float ACCEPTANCE_THRESHOLD = 0.009;
constexpr float SIGMA_PERTURB = 0.04;

template<typename RNG, size_t N_TO_GENERATE>
__global__ void random_uints_kernel(
    const size_t n_elements,
    RNG rng,
    uint32_t max_val,
    uint32_t* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i*N_TO_GENERATE);
    
    #pragma unroll
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;
		if (idx >= n_elements) return;

		out[idx] = rng.next_uint() % max_val;
	}
}

template<typename RNG>
void random_uints(
    cudaStream_t stream,
    const size_t n_elements,
    RNG rng,
    uint32_t max_val,
    tcnn::GPUMemory<uint32_t>& out
) {
    constexpr size_t N_TO_GENERATE = 4;

    out.enlarge(n_elements);

    size_t n_threads = tcnn::div_round_up(n_elements, N_TO_GENERATE);
    random_uints_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, max_val, out.data()
    );
}

template<typename RNG, size_t N_TO_GENERATE>
__global__ void sample_bounding_box_uniform_kernel(
    const size_t n_elements,
    RNG rng,
    const BoundingBox bb,
    Eigen::Vector3f* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i * N_TO_GENERATE);
	
	#pragma unroll
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;		
        const size_t offset = idx / 3;

        if (offset >= n_elements) return;
        
        const size_t xyz = idx % 3;
        const float val = rng.next_float(), l = bb.diag()[xyz], o = bb.min[xyz];
        out[offset][xyz] = o + val * l;
    }
}

template<typename RNG>
void sample_bounding_box_uniform(
    cudaStream_t stream,
    size_t n_elements,
    RNG& rng,
    const BoundingBox& bb,
    tcnn::GPUMemory<Eigen::Vector3f>& out
) {
    constexpr size_t N_TO_GENERATE = 4;

    out.enlarge(n_elements);

    size_t n_threads = tcnn::div_round_up(n_elements, N_TO_GENERATE);
    sample_bounding_box_uniform_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, bb, out.data()
    );
}

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

template<typename RNG, size_t N_TO_GENERATE>
__global__ void perturb_samples_logistic_kernel(
    const size_t n_elements,
    RNG rng,
    const float scale,
    Eigen::Vector3f* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i * N_TO_GENERATE);
	
	#pragma unroll
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;		
        const size_t offset = idx / 3;

        if (offset >= n_elements) return;
        
        const size_t xyz = idx % 3;
        float val = rng.next_float();
        val = tcnn::logit(val) * scale * 0.551328895f;
        out[offset][xyz] += val;
    }
}

template<typename RNG>
void perturb_samples_logistic(
    cudaStream_t stream,
    RNG& rng,
    SamplesSoa& samples,
    size_t start_index,
    size_t n_elements,
    float scale
) {
    constexpr size_t N_TO_GENERATE = 4;
    size_t n_threads = tcnn::div_round_up(n_elements, N_TO_GENERATE);
    
    perturb_samples_logistic_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, scale, samples.points.data() + start_index
    );
}

template<typename RNG>
__global__ void perturb_samples_disk_kernel(
    size_t n_elements,
    RNG rng,
    float scale,
    const Eigen::Vector3f* __restrict__ normals,
    Eigen::Vector3f* __restrict__ out
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;

	rng.advance(i * 2);

    Eigen::Vector2f uv = scale * random_uniform_disc(rng);
    Eigen::Vector3f n, t1, t2;
    n = normals[i];
    coordinateSystem(n, &t1, &t2);
    
    out[i] += uv.x() * t1 + uv.y() * t2;
}

template<typename RNG>
void perturb_samples_disk(
    cudaStream_t stream,
    RNG& rng,
    SamplesSoa& samples,
    size_t start_index,
    size_t n_elements,
    float scale
) {
    tcnn::linear_kernel(perturb_samples_disk_kernel<RNG>, 0, stream,
        n_elements,
        rng,
        scale,
        samples.normals.data() + start_index,
        samples.points.data() + start_index
    );
}

void SDFSampler::initialize() {
    m_curr_size = 0;
    extend_samples();
}

SamplesSoa& SDFSampler::sample() {
    perturb_samples_disk(m_stream, m_rng, m_samples, 0, m_size, SIGMA_PERTURB);
    project_samples(m_stream, *m_network, m_samples, 0, m_size, SAMPLING_REFINEMENT_STEPS);
    
    m_curr_size = filter_samples(0, m_size, ACCEPTANCE_THRESHOLD);
    if (m_curr_size < m_size) {
        // std::cout << m_curr_size << std::endl;
        extend_samples();
    }
      
    return m_samples;
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

void SDFSampler::extend_samples() {
    if (m_curr_size == 0) {
        sample_bounding_box_uniform(
            m_stream,
            m_samples_per_extension,
            m_rng,
            m_bounding_box,
            m_samples.points
        );
    } else {
        random_uints(m_stream, m_samples_per_extension, m_rng, (uint32_t) m_curr_size, m_indices);
        
        Eigen::Vector3f* ptr = m_samples.points.data();
        tcnn::linear_kernel(gather_indices_kernel, 0, m_stream,
            m_samples_per_extension, m_indices.data(), ptr, ptr + m_curr_size
        );

    }

    while (true) {
        project_samples(m_stream, *m_network, m_samples, m_curr_size, m_samples_per_extension, INIT_REFINEMENT_STEPS);
        
        uint32_t n_accepted = filter_samples(m_curr_size, m_samples_per_extension, ACCEPTANCE_THRESHOLD);
        uint32_t prev_size = m_curr_size;
        m_curr_size = min(m_curr_size + n_accepted, m_size);

        if (m_curr_size == m_size) break;

        // Sample with replacement points from prev_size to m_curr_size
        random_uints(m_stream, m_samples_per_extension, m_rng, (uint32_t) n_accepted, m_indices);
        
        Eigen::Vector3f* ptr = m_samples.points.data();
        tcnn::linear_kernel(gather_indices_kernel, 0, m_stream,
            m_samples_per_extension, m_indices.data(), ptr + prev_size, ptr + m_curr_size
        );
        
        // random_uints(m_stream, m_samples_per_extension, m_rng, (uint32_t) m_curr_size, m_indices);
        
        // Eigen::Vector3f* ptr = m_samples.points.data();
        // tcnn::linear_kernel(gather_indices_kernel, 0, m_stream,
        //     m_samples_per_extension, m_indices.data(), ptr, ptr + m_curr_size
        // );
        
        perturb_samples_logistic(m_stream, m_rng, m_samples, m_curr_size, m_samples_per_extension, SIGMA_PERTURB);
    }
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

uint32_t SDFSampler::filter_samples(size_t start_index, size_t n_elements, float threshold) {
    CUDA_CHECK_THROW(cudaMemsetAsync(m_counter.data(), 0, sizeof(uint32_t), m_stream));
    
    tcnn::linear_kernel(filter_samples_kernel, 0, m_stream,
        n_elements,
        m_bounding_box,
        threshold,
        m_counter.data(),
        m_samples.distances.data() + start_index,
        m_samples.points.data() + start_index,
        m_buffer.data()
    );

    uint32_t cpu_counter;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&cpu_counter, m_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));

    if (cpu_counter != n_elements) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            m_samples.points.data() + start_index,
            m_buffer.data(),
            cpu_counter * sizeof(Eigen::Vector3f),
            cudaMemcpyDeviceToDevice,
            m_stream
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

void SDFSampler::render(
    const Eigen::Vector2i& resolution,
    const Eigen::Vector2f& focal_length,
    const Eigen::Matrix<float, 3, 4>& camera_matrix
) {
    GLuint VAO, VBO;
    std::unique_ptr<Eigen::Vector3f[]> points = std::make_unique<Eigen::Vector3f[]>(m_size);
    CUDA_CHECK_THROW(cudaMemcpy(
        points.get(), m_samples.points.data(), m_size * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost
    ));

    // Setup Vertex Array
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    
    // Pass postition data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, m_size * sizeof(Eigen::Vector3f), points.get(), GL_STATIC_DRAW);

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

    glDrawArrays(GL_POINTS, 0, m_size);

    glUseProgram(0);
    glBindVertexArray(0);

    // Clean up
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(prog);
}

NGP_NAMESPACE_END