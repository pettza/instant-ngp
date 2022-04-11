#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/sdf.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/sdf_sampler.h>

#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <GL/glew.h>


NGP_NAMESPACE_BEGIN

class Brush
{
public:
    Brush(float radius = 0.04, float intensity = 0.02):
        m_radius(radius),
        m_intensity(intensity),
        m_counter(1),
        m_position(0., 0., 0.)
    { }

    ~Brush() {
        glDeleteBuffers(1, &m_VBO);
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteProgram(m_prog);
    }
    
    void initialize();

    bool has_interaction() { return !m_position.isZero(); }

    void set_radius(float radius) { m_radius = radius; }
    void set_intensity(float intensity) { m_intensity = intensity; }
    void set_size(size_t size) {
        m_size = size;
        size_t base = size / 8;
        m_model_samples = base * 4;
        m_uniform_samples = base * 2;
    }
    void initialize_sampler(std::shared_ptr<SDFSampler> sampler, cudaStream_t stream = nullptr) {
        m_sampler = sampler;
        m_sampler->set_size(m_model_samples);
        m_sampler->set_bounding_box(m_aabb);
        m_sampler->set_network(m_network);
        m_sampler->initialize(stream, m_rng);
    }
    void set_bounding_box(const BoundingBox& bb) { m_aabb = bb; }
    void set_network(std::shared_ptr<tcnn::Network<float, precision_t>> network) {
        m_network = network;
    }

    size_t get_size() { return m_size; }
    
    void get_interaction_point(
        const RaysSdfSoa& rays,
        const Eigen::Vector2i& screen_res,
        const Eigen::Vector2i& buffer_res,
        const Eigen::Vector2f& mouse
    );
    void render(
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix
    );
    const SamplesSoa& generate_training_samples(cudaStream_t stream);
    const SamplesSoa& sample_model(cudaStream_t stream) { return m_sampler->sample(stream, m_rng); }

private:
    uint32_t filter_points_inside_interaction_area(
        cudaStream_t stream,
        size_t n_elements,
        const SamplesSoa& in,
        SamplesSoa& out
    );
    void sample_interaction(cudaStream_t stream, size_t start_index, size_t n_samples);
    void apply_brush_to_samples(cudaStream_t stream, size_t start_index, size_t n_samples);

private:
    Eigen::Vector3f m_position;
    Eigen::Vector3f m_normal, m_tangent1, m_tangent2;
    float m_radius;
    float m_intensity;

    size_t m_size, m_model_samples, m_uniform_samples;
    
    SamplesSoa m_samples, m_samples_buffer;
    tcnn::GPUMemory<uint32_t> m_counter;
    
    GLuint m_VAO, m_VBO;
    GLuint m_prog;
    
    tcnn::default_rng_t m_rng;
    
    BoundingBox m_aabb;
    std::shared_ptr<tcnn::Network<float, precision_t>> m_network;
    std::shared_ptr<SDFSampler> m_sampler;
};

NGP_NAMESPACE_END
