#include <neural-graphics-primitives/brush.h>

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

void Brush::getInteractionPoint(
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
}

void Brush::render(
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix
) {
    if (m_position.isZero()) return;

    Eigen::Matrix4f view2world = Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();

    // Tangent vectors    
    Eigen::Vector3f t1, t2;
    coordinateSystem(m_normal, &t1, &t2);

    glBindVertexArray(m_VAO);
    glUseProgram(m_prog);

	glUniformMatrix4fv(glGetUniformLocation(m_prog, "view"), 1, GL_FALSE, (GLfloat*) &world2view);
    glUniform3f(glGetUniformLocation(m_prog, "inter"), m_position.x(), m_position.y(), m_position.z());
	glUniform3f(glGetUniformLocation(m_prog, "t1"), t1.x(), t1.y(), t1.z());
    glUniform3f(glGetUniformLocation(m_prog, "t2"), t2.x(), t2.y(), t2.z());
    glUniform2f(glGetUniformLocation(m_prog, "f"), focal_length.x(), focal_length.y());
	glUniform2i(glGetUniformLocation(m_prog, "res"), resolution.x(), resolution.y());
    glUniform1f(glGetUniformLocation(m_prog, "radius"), m_radius);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
    glBindVertexArray(0);
}

NGP_NAMESPACE_END
