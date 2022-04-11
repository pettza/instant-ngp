#pragma once

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

    void render(
        size_t n_elements,
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix
    ) const;

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
    SDFSampler(): m_curr_size(0), m_size(0), m_counter(1) { }

    SDFSampler(
        size_t size,
        const BoundingBox& bb,
        std::shared_ptr<tcnn::Network<float, precision_t>> network
    ): m_curr_size(0), m_size(size), m_samples_per_extension(size / 10), m_aabb(bb), m_network(network), m_counter(1) {
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
    size_t get_size() { return m_size; }
    void set_bounding_box(const BoundingBox& bb) { m_aabb = bb; }

    void set_network(std::shared_ptr<tcnn::Network<float, precision_t>> network) {
        m_network = network;
    }
    
    template<typename RNG>
    void initialize(cudaStream_t stream, RNG& rng);
    template<typename RNG>
    const SamplesSoa& sample(cudaStream_t stream, RNG& rng);

private:
    template<typename RNG>
    void extend_samples(cudaStream_t stream, RNG& rng);
    uint32_t filter_samples(cudaStream_t stream, size_t start_index, size_t n_elements, float threshold);

private:
    size_t m_size, m_curr_size;
    size_t m_samples_per_extension;
    BoundingBox m_aabb;
    std::shared_ptr<tcnn::Network<float, precision_t>> m_network;
    SamplesSoa m_samples;
    tcnn::GPUMemory<Eigen::Vector3f> m_buffer;
    tcnn::GPUMemory<uint32_t> m_indices;
    tcnn::GPUMemory<uint32_t> m_counter;
};

// Refinement steps when initializing
constexpr int INIT_REFINEMENT_STEPS = 7;
// Refinement steps when sampling
constexpr int SAMPLING_REFINEMENT_STEPS = 7;
constexpr float REFINEMENT_THRESHOLD = 0.03;
constexpr float ACCEPTANCE_THRESHOLD = 0.009;
constexpr float SIGMA_PERTURB = 0.04;

// Definitions of template functions might need to be visible
// by translation units that include this file
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

    size_t n_threads = tcnn::div_round_up(n_elements, N_TO_GENERATE);
    random_uints_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, max_val, out.data()
    );

    rng.advance(n_elements);
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

    size_t n_threads = tcnn::div_round_up(n_elements * 3, N_TO_GENERATE);
    sample_bounding_box_uniform_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, bb, out.data()
    );

    rng.advance(n_elements * 3);
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
    
    size_t n_threads = tcnn::div_round_up(n_elements * 3, N_TO_GENERATE);
    perturb_samples_logistic_kernel<RNG, N_TO_GENERATE><<<tcnn::n_blocks_linear(n_threads), tcnn::n_threads_linear, 0, stream>>>(
        n_elements, rng, scale, samples.points.data() + start_index
    );

    rng.advance(n_elements * 3);
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

    if (i >= n_elements) return;

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

template<typename RNG>
void SDFSampler::initialize(cudaStream_t stream, RNG& rng) {
    m_curr_size = 0;
    extend_samples(stream, rng);
}

template<typename RNG>
const SamplesSoa& SDFSampler::sample(cudaStream_t stream, RNG& rng) {
    perturb_samples_disk(stream, rng, m_samples, 0, m_size, SIGMA_PERTURB);
    project_samples(stream, *m_network, m_samples, 0, m_size, SAMPLING_REFINEMENT_STEPS);
    
    m_curr_size = filter_samples(stream, 0, m_size, ACCEPTANCE_THRESHOLD);
    if (m_curr_size < m_size) {
        // std::cout << m_curr_size << std::endl;
        extend_samples(stream, rng);
    }

    return m_samples;
}

__global__ void gather_indices_kernel(
    size_t n_elements,
    const uint32_t* __restrict__ indices,
    const Eigen::Vector3f* __restrict__ in,
    Eigen::Vector3f* __restrict__ out
);

template<typename RNG>
void SDFSampler::extend_samples(cudaStream_t stream, RNG& rng) {
    if (m_curr_size == 0) {
        sample_bounding_box_uniform(
            stream,
            m_samples_per_extension,
            rng,
            m_aabb,
            m_samples.points
        );
    } else {
        random_uints(stream, m_samples_per_extension, rng, (uint32_t) m_curr_size, m_indices);
        
        Eigen::Vector3f* ptr = m_samples.points.data();
        tcnn::linear_kernel(gather_indices_kernel, 0, stream,
            m_samples_per_extension, m_indices.data(), ptr, ptr + m_curr_size
        );

    }

    while (true) {
        project_samples(stream, *m_network, m_samples, m_curr_size, m_samples_per_extension, INIT_REFINEMENT_STEPS);
        
        uint32_t n_accepted = filter_samples(stream, m_curr_size, m_samples_per_extension, ACCEPTANCE_THRESHOLD);
        uint32_t prev_size = m_curr_size;
        m_curr_size = min(m_curr_size + n_accepted, m_size);

        if (m_curr_size == m_size) break;

        // Sample with replacement points from prev_size to m_curr_size
        random_uints(stream, m_samples_per_extension, rng, (uint32_t) n_accepted, m_indices);
        
        Eigen::Vector3f* ptr = m_samples.points.data();
        tcnn::linear_kernel(gather_indices_kernel, 0, stream,
            m_samples_per_extension, m_indices.data(), ptr + prev_size, ptr + m_curr_size
        );
        
        // random_uints(stream, m_samples_per_extension, rng, (uint32_t) m_curr_size, m_indices);
        
        // Eigen::Vector3f* ptr = m_samples.points.data();
        // tcnn::linear_kernel(gather_indices_kernel, 0, stream,
        //     m_samples_per_extension, m_indices.data(), ptr, ptr + m_curr_size
        // );
        
        perturb_samples_logistic(stream, rng, m_samples, m_curr_size, m_samples_per_extension, SIGMA_PERTURB);
    }
}

NGP_NAMESPACE_END
