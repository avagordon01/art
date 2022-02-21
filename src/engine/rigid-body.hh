struct rigid_body {
    glm::vec3 position {};
    glm::vec3 velocity {};
    glm::vec3 acceleration = glm::vec3();

    glm::quat angular_position {};
    glm::quat angular_velocity {};
    glm::quat angular_acceleration = glm::quat();

    void tick() {
        float dt = 1.0f / 60.0f;
        velocity += acceleration * dt;
        position += velocity * dt;
        angular_velocity += 0.5f * angular_velocity * angular_acceleration * dt;
        angular_position += 0.5f * angular_position * angular_velocity * dt;

        angular_position = glm::normalize(angular_position);
    }
};
