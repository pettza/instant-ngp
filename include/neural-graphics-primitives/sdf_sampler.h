#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/bounding_box.cuh>

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>


NGP_NAMESPACE_BEGIN

struct SamplesSoa {
    void enlarge(size_t size) {
        points.enlarge(size);
        normals.enlarge(size);
        distances.enlarge(size);
    }

    void resize(size_t size) {
        points.resize(size);
        normals.resize(size);
        distances.resize(size);
    }

    tcnn::GPUMemory<Eigen::Vector3f> points;
    tcnn::GPUMemory<Eigen::Vector3f> normals;
    tcnn::GPUMemory<float> distances;
};


void compute_distances_normals(cudaStream_t stream, tcnn::Network<float, precision_t>& network, SamplesSoa& samples, size_t start_index, size_t n_elements);
void move_along_normals(cudaStream_t stream, SamplesSoa& samples, size_t start_index, size_t n_elements);
void project_samples(cudaStream_t stream, tcnn::Network<float, precision_t>& network, SamplesSoa& samples, size_t start_index, size_t n_elements, int n_iter);
template<typename RNG>
void perturb_samples_logistic(cudaStream_t stream, RNG& rng, SamplesSoa& samples, size_t start_index, size_t n_elements, float scale);
template<typename RNG>
void perturb_samples_disk(cudaStream_t stream, RNG& rng, SamplesSoa& samples, size_t start_index, size_t n_elements, float scale);


class SDFSampler
{
public:
    SDFSampler(): m_curr_size(0), m_size(0), m_counter(1) { CUDA_CHECK_THROW(cudaStreamCreate(&m_stream)); }

    SDFSampler(
        size_t size,
        const BoundingBox& bb,
        std::shared_ptr<tcnn::Network<float, precision_t>> network
    ): m_curr_size(0), m_size(size), m_samples_per_extension(size / 10), m_bounding_box(bb), m_network(network), m_counter(1) {
        CUDA_CHECK_THROW(cudaStreamCreate(&m_stream));
        m_samples.enlarge(m_size + m_samples_per_extension);
        m_buffer.enlarge(m_size + m_samples_per_extension);
        m_indices.enlarge(m_samples_per_extension);
    }

    void set_size(size_t size) {
        m_curr_size = 0;
        m_size = size;
        m_samples_per_extension = m_size / 10;
        m_samples.resize(m_size + m_samples_per_extension);
        m_buffer.resize(m_size + m_samples_per_extension);
        m_indices.resize(m_samples_per_extension);   
    }

    void set_bounding_box(const BoundingBox& bb) { m_bounding_box = bb; }

    void set_network(std::shared_ptr<tcnn::Network<float, precision_t>> network) {
        m_network = network;
    }
    
    void initialize();
    SamplesSoa& sample();
    void render(
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix
    );

private:
    void extend_samples();
    uint32_t filter_samples(size_t start_index, size_t n_elements, float threshold);

    size_t m_size, m_curr_size;
    size_t m_samples_per_extension;
    tcnn::default_rng_t m_rng;
    cudaStream_t m_stream;
    BoundingBox m_bounding_box;
    std::shared_ptr<tcnn::Network<float, precision_t>> m_network;
    SamplesSoa m_samples;
    tcnn::GPUMemory<Eigen::Vector3f> m_buffer;
    tcnn::GPUMemory<uint32_t> m_indices;
    tcnn::GPUMemory<uint32_t> m_counter;
};

NGP_NAMESPACE_END
