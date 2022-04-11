#include <neural-graphics-primitives/brush.h>

#include <neural-graphics-primitives/random_val.cuh>

const char* vertexSource = R"foo(
#version 330 core
layout (location = 0) in vec2 pos;

uniform mat4 view;
uniform vec3 inter;
uniform vec3 t1, t2;
uniform vec2 f;
uniform ivec2 res;
uniform float radius;

out vec2 plane_pos;

void main() {
    plane_pos = pos * (1.1 * radius);
    vec3 t = plane_pos.x * t1 + plane_pos.y * t2 + inter;
    vec4 p = view * vec4(t, 1.0);
	p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
	p.w = p.z;
	p.z = p.z - 0.1;
    gl_Position = p;
}
)foo";

const char* fragmentSource = R"foo(
#version 330 core
out vec4 FragColor;

in vec2 plane_pos;

uniform float radius;

void main() {
    float thickness = 0.05 * radius;

    float l = length(plane_pos);
    float d = abs(l - radius);
    
    float edge = smoothstep(thickness, 0., d);
    float disk = smoothstep(radius, 0., l);
    float center = smoothstep(radius * 0.05, 0., l);

    vec4 edge_col = vec4(0.05, 0.1, 0.8, 1.0) * edge;
    vec4 disk_col = vec4(0.7, 0.7, 0.7, 0.5) * disk;
    vec4 center_col = vec4(0., 0., 0., 1.0) * center;
    vec4 col = mix(edge_col + disk_col, center_col, center_col.w);
    
    FragColor = col;
}
)foo";

NGP_NAMESPACE_BEGIN

void Brush::initialize() {
    // A square made by two triangles
    Eigen::Vector2f verts[] = {{1., 1.}, {1., -1.}, {-1., -1.},
                               {-1., -1.}, {-1., 1.}, {1., 1.}};
    
    // Setup Vertex Array
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    
    // Pass postition data
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Eigen::Vector2f) * sizeof(verts), verts, GL_STATIC_DRAW);

    // Enable attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Eigen::Vector2f), reinterpret_cast<void*>(0));
    
    glBindVertexArray(0);

    // Create shader
    GLuint vs, fs;
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

    m_prog = glCreateProgram();
    glAttachShader(m_prog, vs);
    glAttachShader(m_prog, fs);
    glLinkProgram(m_prog);
    
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void Brush::get_interaction_point(
    const RaysSdfSoa& rays,
    const Eigen::Vector2i& screen_res,
    const Eigen::Vector2i& buffer_res,
    const Eigen::Vector2f& mouse
) {
    // The screen and the renderbuffer for the model might have different resolutions
    // so scale cursor coordinates accordingly
    Eigen::Vector2f uv = mouse.cwiseQuotient(screen_res.cast<float>());
    Eigen::Vector2i xy = uv.cwiseProduct(buffer_res.cast<float>()).cast<int>();
    int x = max(min(xy.x(), buffer_res.x()-1), 0);
    int y = max(min(xy.y(), buffer_res.y()-1), 0);
    size_t idx = x + buffer_res.x() * y;
    CUDA_CHECK_THROW(cudaMemcpy(&m_position, rays.pos.data() + idx, sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(&m_normal, rays.normal.data() + idx, sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    m_normal.normalize();
    coordinateSystem(m_normal, &m_tangent1, &m_tangent2);
}

void Brush::render(
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix
) {
    if (!has_interaction()) return;

    Eigen::Matrix4f view2world = Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();

    glBindVertexArray(m_VAO);
    glUseProgram(m_prog);

	glUniformMatrix4fv(glGetUniformLocation(m_prog, "view"), 1, GL_FALSE, (GLfloat*) &world2view);
    glUniform3f(glGetUniformLocation(m_prog, "inter"), m_position.x(), m_position.y(), m_position.z());
	glUniform3f(glGetUniformLocation(m_prog, "t1"), m_tangent1.x(), m_tangent1.y(), m_tangent1.z());
    glUniform3f(glGetUniformLocation(m_prog, "t2"), m_tangent2.x(), m_tangent2.y(), m_tangent2.z());
    glUniform2f(glGetUniformLocation(m_prog, "f"), focal_length.x(), focal_length.y());
	glUniform2i(glGetUniformLocation(m_prog, "res"), resolution.x(), resolution.y());
    glUniform1f(glGetUniformLocation(m_prog, "radius"), m_radius);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
    glBindVertexArray(0);
}

const SamplesSoa& Brush::generate_training_samples(cudaStream_t stream) {
    m_samples.enlarge(m_size);
    m_samples_buffer.enlarge(m_size);

    sample_bounding_box_uniform(stream, m_uniform_samples, m_rng, m_aabb, m_samples_buffer.points);
    compute_distances_normals(stream, *m_network, m_samples_buffer, 0, m_uniform_samples);

    const SamplesSoa& reg_samples = sample_model(stream);
    
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        m_samples_buffer.points.data() + m_uniform_samples,
        reg_samples.points.data(),
        m_model_samples * sizeof(Eigen::Vector3f),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        m_samples_buffer.distances.data() + m_uniform_samples,
        reg_samples.distances.data(),
        m_model_samples * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    
    uint32_t n_accepted = filter_points_inside_interaction_area(
        stream,
        m_uniform_samples + m_model_samples,
        m_samples_buffer,
        m_samples
    );

    size_t n_interaction_samples = m_size - n_accepted;
    sample_interaction(stream, n_accepted, n_interaction_samples);
    apply_brush_to_samples(stream, n_accepted, n_interaction_samples);

    return m_samples;
}

__global__ void filter_points_inside_interaction_area_kernel(
    const size_t n_elements,
    const Eigen::Vector3f center,
    const float squared_radius,
    uint32_t* counter,
    const Eigen::Vector3f* __restrict__ in_points,
    const float* __restrict__ in_distances,
    Eigen::Vector3f* __restrict__ out_points,
    float* out_distances
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements) return;

    Eigen::Vector3f point = in_points[i];
    if ((point - center).squaredNorm() > squared_radius) {
        uint32_t out_idx = atomicAdd(counter, 1);
        out_points[out_idx] = point;
        out_distances[out_idx] = in_distances[i];
    }
}

uint32_t Brush::filter_points_inside_interaction_area(
    cudaStream_t stream,
    size_t n_elements,
    const SamplesSoa& in,
    SamplesSoa& out
) {
    CUDA_CHECK_THROW(cudaMemsetAsync(m_counter.data(), 0, sizeof(uint32_t), stream));
    
    tcnn::linear_kernel(filter_points_inside_interaction_area_kernel, 0, stream,
        n_elements,
        m_position,
        m_radius * m_radius,
        m_counter.data(),
        in.points.data(),
        in.distances.data(),
        out.points.data(),
        out.distances.data()
    );

    uint32_t cpu_counter;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&cpu_counter, m_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    return cpu_counter;
}

template<typename RNG>
__global__ void sample_uniform_disk_kernel(
    const size_t n_elements,
    RNG rng,
    const Eigen::Vector3f center,
    const float radius,
    const Eigen::Vector3f t1,
    const Eigen::Vector3f t2,
    Eigen::Vector3f* __restrict__ points,
    float* __restrict__ radii
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i >= n_elements) return;
    
    rng.advance(i * 2);

    Eigen::Vector2f uv = random_uniform_disc(rng);
    radii[i] = uv.norm();
    uv *= radius;
    points[i] = center + uv[0] * t1 + uv[1] * t2;
}

void Brush::sample_interaction(cudaStream_t stream, size_t start_index, size_t n_samples) {
    tcnn::linear_kernel(sample_uniform_disk_kernel<tcnn::default_rng_t>, 0, stream,
        n_samples,
        m_rng,
        m_position,
        m_radius,
        m_tangent1,
        m_tangent2,
        m_samples.points.data() + start_index,
        m_samples_buffer.distances.data()
    );

    m_rng.advance(n_samples * 2);

    project_samples(stream, *m_network, m_samples, start_index, n_samples, 3);
}

__host__ __device__ float brush_shape(float x) {
    x = 1 - x;
    float x_sq = x * x;

    return 3 * x_sq - 2 * x_sq * x;
}

__global__ void apply_brush_to_samples_kernel(
    const size_t n_elements,
    const float intensity,
    const float radius,
    const Eigen::Vector3f dir,
    const float* __restrict__ x,
    Eigen::Vector3f* __restrict__ points,
    float* __restrict__ distances
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i >= n_elements) return;

    float val = intensity * brush_shape(x[i]);
    points[i] += val * dir;
    distances[i] -= val;
}

void Brush::apply_brush_to_samples(cudaStream_t stream, size_t start_index, size_t n_samples) {
    tcnn::linear_kernel(apply_brush_to_samples_kernel, 0, stream,
        n_samples,
        m_intensity,
        m_radius,
        m_normal,
        m_samples_buffer.distances.data(),
        m_samples.points.data() + start_index,
        m_samples.distances.data() + start_index
    );
}

NGP_NAMESPACE_END
