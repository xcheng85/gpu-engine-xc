#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <misc.h>

#include <cameraBase.h>
#include <SDL_mouse.h>
#include <SDL_events.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>



class ArcballCamera : public CameraBase
{
public:
    explicit ArcballCamera(
        const glm::vec3 &eye,
        const glm::vec3 &target,
        const glm::vec3 &up,
        float nearPlaneD,
        float farPlaneD,
        float verticalFov,
        float aspect)
    {
        _eye = eye;
        _nearPlaneD = nearPlaneD;
        _farPlaneD = farPlaneD;
        _verticalFov = verticalFov;
        _aspect = aspect;
        {
            const glm::vec3 dir = target - eye;
            glm::vec3 z_axis = glm::normalize(dir);
            glm::vec3 x_axis = glm::normalize(glm::cross(z_axis, glm::normalize(up)));
            glm::vec3 y_axis = glm::normalize(glm::cross(x_axis, z_axis));
            x_axis = glm::normalize(glm::cross(z_axis, y_axis));

            _targetTranslationMatrix = glm::inverse(glm::translate(target));
            _lookatTranslationMatrix = glm::translate(glm::vec3(0.f, 0.f, -glm::length(dir)));
            // transpose due to opengl colum-major
            _rotation = glm::normalize(glm::quat_cast(glm::transpose(glm::mat3(x_axis, y_axis, -z_axis))));

            // log(Level::Info, "rotation : ", rotation.x);
            // log(Level::Info, "rotation : ", rotation.y);
            // log(Level::Info, "rotation : ", rotation.z);
            // log(Level::Info, "rotation : ", rotation.w);

            update_camera();
        }

        // {
        //     const glm::vec3 dir1 = target - eye;
        //     const vec3f dir{std::array<float, 3>({dir1.x, dir1.y, dir1.z})};
        //     const auto z = normalize(dir);

        //     const vec3f up1{std::array<float, 3>({up.x, up.y, up.z})};
        //     auto x = normalize(crossProduct(z, normalize(up1)));

        //     const auto y = normalize(crossProduct(x, normalize(z)));
        //     x = normalize(crossProduct(z, y));

        //     _center_translation = MatrixTranslation4x4(-center[COMPONENT::X],
        //                                                -center[COMPONENT::Y],
        //                                                -center[COMPONENT::Z]);
        //     _translation = MatrixTranslation4x4(0.f, 0.f, static_cast<float>(-dir.vectorLength()));

        //     // column-major
        //     std::array<float, 9> cols{
        //         x[COMPONENT::X],
        //         x[COMPONENT::Y],
        //         x[COMPONENT::Z],
        //         y[COMPONENT::X],
        //         y[COMPONENT::Y],
        //         y[COMPONENT::Z],
        //         -z[COMPONENT::X],
        //         -z[COMPONENT::Y],
        //         -z[COMPONENT::Z],
        //     };
        //     mat3x3f rot(cols);
        //     rot.transpose();
        //     _rotation = QuaternionFromRotationMatrix(rot);
        //     _rotation = normalize(_rotation);
        //     log(Level::Info, "_rotation mat: ", _rotation);

        //     update_camera1();
        // }
    }

    glm::vec3 viewPos() const override
    {
        return glm::vec3{_invCamera * glm::vec4{0, 0, 0, 1}};
    }

    glm::mat4 viewTransformLH() const override
    {
        return _camera;
    }

    void handleKeyboardEvent(CameraActionType actionType, float dt) override
    {
    }

    void handleMouseCursorEvent(
        int button,
        int state,
        const glm::ivec2 &currMouseInScreenSpace,
        const glm::ivec2&screenDimension) override
    {
        const auto currMouseCoordsInNdc = screenSpace2Ndc(currMouseInScreenSpace, screenDimension);
        if (_prevMouseCoordsInNdc != glm::vec2(-2.f, -2.f))
        {
            if (state & SDL_BUTTON_LMASK)
            {
                rotate(
                    glm::vec2(_prevMouseCoordsInNdc[0], _prevMouseCoordsInNdc[1]),
                    glm::vec2(currMouseCoordsInNdc[0], currMouseCoordsInNdc[1]));
            }
            else if (state & SDL_BUTTON_RMASK)
            {
                auto proj = glm::perspective(glm::radians(_verticalFov), (float)screenDimension[0] / (float)screenDimension[1], _nearPlaneD, _farPlaneD);
                auto projInverse = glm::inverse(proj);
                auto delta = currMouseCoordsInNdc - _prevMouseCoordsInNdc;
                glm::vec4 dxy4 = projInverse * glm::vec4(delta[0], delta[1], 0, 1);
                pan(glm::vec2(dxy4.x, dxy4.y));
            }
        }
        // log(Level::Info, "_prevMouseCoordsInNdc: ", _prevMouseCoordsInNdc);
        _prevMouseCoordsInNdc = currMouseCoordsInNdc;
    }

    void handleMouseClickEvent(int button, int state, int x, int y) override
    {
        // log(Level::Info, "handleMouseClickEvent: ",
        //     "button: ", button,
        //     "x: ", x,
        //     "y: ", y);
    }

    void handleMouseWheelEvent(float v) override
    {
        zoom(v);
    }

    // http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-extracting-the-planes/

    virtual Fustrum fustrumPlanes() const
    {
        // A couple more unit vectors are required,
        // namely the up vector and the right vector.
        //  The former is obtained by normalizing the vector (ux,uy,uz) (the components of this vector are the last parameters of the gluLookAt function);
        // the latter is obtained with the cross product between the up vector and the d vector.

        // rotation is built based upon x(right), y(up), z(forward)
        // fetch the right from the rotation quat
        const auto viewMatrix = glm::mat4_cast(_rotation);
        // the transpose in line: 40
        const auto right = glm::vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
        auto forward = glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);
        const auto up = glm::normalize(glm::cross(right, forward));

        forward = glm::normalize(glm::cross(right, up));

        const auto verticalFovTan = glm::tan(glm::radians(_verticalFov) * 0.5);

        const float halfHeightForNearPlane = verticalFovTan * _nearPlaneD;
        const float halfHeightForFarPlane = verticalFovTan * _farPlaneD;

        // (up * Hfar/2)
        //- (right * Wfar/2)
        // w = h * _aspect
        const auto halfUpVectorForNearPlane = halfHeightForNearPlane * up;
        const auto halfRightVectorForNearPlane = _aspect * halfUpVectorForNearPlane;

        const auto halfUpVectorForFarPlane = halfHeightForFarPlane * up;
        const auto halfRightVectorForFarPlane = _aspect * halfUpVectorForFarPlane;
        // _eye is obsolete
        const auto eyeMappingToNearPlane = viewPos() + forward * _nearPlaneD;
        const auto eyeMappingToFarPlane = viewPos() + forward * _farPlaneD;

        // eight corners of fustrum
        // Upper: +
        // Right: +
        // Bottom: -
        // Left: -
        const auto upperRightInNearPlane = eyeMappingToNearPlane + halfRightVectorForNearPlane + halfUpVectorForNearPlane;
        const auto bottomRightInNearPlane = eyeMappingToNearPlane + halfRightVectorForNearPlane - halfUpVectorForNearPlane;
        const auto bottomLeftInNearPlane = eyeMappingToNearPlane - halfRightVectorForNearPlane - halfUpVectorForNearPlane;
        const auto upperLeftInNearPlane = eyeMappingToNearPlane - halfRightVectorForNearPlane + halfUpVectorForNearPlane;

        const auto upperRightInFarPlane = eyeMappingToFarPlane + halfRightVectorForFarPlane + halfUpVectorForFarPlane;
        const auto bottomRightInFarPlane = eyeMappingToFarPlane + halfRightVectorForFarPlane - halfUpVectorForFarPlane;
        const auto bottomLeftInFarPlane = eyeMappingToFarPlane - halfRightVectorForFarPlane - halfUpVectorForFarPlane;
        const auto upperLeftInFarPlane = eyeMappingToFarPlane - halfRightVectorForFarPlane + halfUpVectorForFarPlane;

        // create 6 planes
        // plane: normal vector + a point in the plane
        Fustrum fustrum;
        // left plane(all left)
        fustrum.planes[0] = createPlaneFromPoints(bottomLeftInFarPlane, bottomLeftInNearPlane, upperLeftInFarPlane);
        // bottom plane
        fustrum.planes[1] = createPlaneFromPoints(bottomLeftInFarPlane, bottomRightInFarPlane, bottomLeftInNearPlane);
        // right plane
        fustrum.planes[2] = createPlaneFromPoints(bottomRightInFarPlane, bottomRightInNearPlane, upperRightInFarPlane);
        // upper plane
        fustrum.planes[3] = createPlaneFromPoints(upperLeftInFarPlane, upperRightInNearPlane, upperRightInFarPlane);
        // near plane
        fustrum.planes[4] = createPlaneFromPoints(bottomRightInNearPlane, bottomLeftInNearPlane, upperRightInNearPlane);
        // far plane
        fustrum.planes[5] = createPlaneFromPoints(bottomRightInFarPlane, bottomLeftInFarPlane, upperRightInFarPlane);

        return fustrum;
    }

private:
    glm::quat screen_to_arcball(const glm::vec2 &p)
    {
        const float dist = glm::dot(p, p);
        // If we're on/in the sphere return the point on it
        if (dist <= 1.f)
        {
            return glm::quat(0.0, p.x, p.y, std::sqrt(1.f - dist));
        }
        else
        {
            // otherwise we project the point onto the sphere
            const glm::vec2 proj = glm::normalize(p);
            return glm::quat(0.0, proj.x, proj.y, 0.f);
        }
    }

    void pan(glm::vec2 mouse_delta)
    {
        const float zoom_amount = std::abs(_lookatTranslationMatrix[3][2]);
        glm::vec4 motion(mouse_delta.x * zoom_amount, mouse_delta.y * zoom_amount, 0.f, 0.f);
        // Find the panning amount in the world space
        motion = _invCamera * motion;

        _targetTranslationMatrix = glm::translate(glm::vec3(motion)) * _targetTranslationMatrix;
        update_camera();
    }

    void rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse)
    {
        // Clamp mouse positions to stay in NDC
        cur_mouse = glm::clamp(cur_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});
        prev_mouse = glm::clamp(prev_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});

        const glm::quat mouse_cur_ball = screen_to_arcball(cur_mouse);
        const glm::quat mouse_prev_ball = screen_to_arcball(prev_mouse);

        _rotation = mouse_cur_ball * mouse_prev_ball * _rotation;
        update_camera();
    }

    void zoom(const float zoom_amount)
    {
        const glm::vec3 motion(0.f, 0.f, zoom_amount);

        _lookatTranslationMatrix = glm::translate(motion) * _lookatTranslationMatrix;
        update_camera();
    }

    void update_camera()
    {
        // combination of pan, rotation and zoom
        _camera = _lookatTranslationMatrix * glm::mat4_cast(_rotation) * _targetTranslationMatrix;
        _invCamera = glm::inverse(_camera);
        // log(Level::Info, "camera mat: ", camera);
        // log(Level::Info, "inv_camera mat: ", inv_camera);
    }

    // void update_camera1()
    // {
    //     auto rotationMatrix = RotationMatrixFromQuaternion(_rotation);
    //     _camera = MatrixMultiply4x4(_translation, MatrixMultiply4x4(rotationMatrix, _center_translation));
    //     _inv_camera = Inverse(_camera);
    //     log(Level::Info, "_camera mat: ", _camera);
    //     log(Level::Info, "_inv_camera mat: ", _inv_camera);
    // }

    // _targetTranslationMatrix: is updated through pan op
    glm::mat4 _targetTranslationMatrix;
    // _lookatTranslationMatrix: is updated through zoom op
    glm::mat4 _lookatTranslationMatrix;
    glm::quat _rotation;
    glm::mat4 _camera, _invCamera;

    // mat4x4f _center_translation, _translation;
    // quatf _rotation{0, 0, 0, 1};
    // mat4x4f _camera, _inv_camera;
};