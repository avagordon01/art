#pragma once

glm::quat quaternionRand() {
    glm::vec3 axis = glm::sphericalRand(1.0);
    float angle = glm::linearRand(0.0, 360.0);
    glm::quat q = glm::angleAxis(angle, axis);
    return q;
};
