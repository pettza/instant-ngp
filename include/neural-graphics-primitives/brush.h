#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/sdf.h>

#include <GL/glew.h>


NGP_NAMESPACE_BEGIN

class Brush
{
public:
    Brush(): m_radius(0.04) { }

    ~Brush() {
        glDeleteBuffers(1, &m_VBO);
        glDeleteVertexArrays(1, &m_VAO);
        glDeleteProgram(m_prog);
    }

    void initialize();
    void getInteractionPoint(
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

private:
    Eigen::Vector3f m_position;
    Eigen::Vector3f m_normal;
    GLuint m_VAO, m_VBO;
    GLuint m_prog;

public:
    float m_radius;
};

NGP_NAMESPACE_END
