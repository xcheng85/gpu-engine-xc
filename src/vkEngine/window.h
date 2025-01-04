#pragma once
#include <iostream>
#include <format>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <SDL_video.h>
#include <cameraBase.h>

extern bool gRunning;
extern double gDt;
extern uint64_t gLastFrame;

class SDL_Window;

struct WindowConfig
{
    uint32_t width{800};
    uint32_t height{600};
    std::string title{"default"};
};

// template <typename T>
// requires requires(T t) {
//     t.title;
//     t.width;
//     t.height;
// }

class Window
{
public:
    Window() = delete;
    Window(void *configuration, CameraBase &camera) : _camera{camera}
    {
        const auto &config = *(static_cast<WindowConfig *>(configuration));
        if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
        {
            std::cerr << std::format("SDL Init error: {}\n", SDL_GetError());
            return;
        }

        SDL_Vulkan_LoadLibrary(nullptr);

        SDL_DisplayMode current;
        SDL_GetCurrentDisplayMode(0, &current);

        

        SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
        _window = SDL_CreateWindow(config.title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, config.width, config.height, window_flags);
        int w, h;
        SDL_Vulkan_GetDrawableSize(_window, &w, &h);
        SDL_SetWindowFullscreen(_window, 0);
        _winDimension[0] = w;
        _winDimension[1] = h;
    }
    void shutdown()
    {
        SDL_DestroyWindow(_window);
        SDL_Quit();
    }
    void pollEvents()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                gRunning = false;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    gRunning = false;
                    break;
                    // case SDLK_w:
                    //     _camera.handleKeyboardEvent(Camera::CameraActionType::FORWARD, gDt);
                    //     break;
                    // case SDLK_s:
                    //     _camera.handleKeyboardEvent(Camera::CameraActionType::BACKWARD, gDt);
                    //     break;
                    // case SDLK_a:
                    //     _camera.handleKeyboardEvent(Camera::CameraActionType::LEFT, gDt);
                    //     break;
                    // case SDLK_d:
                    //     _camera.handleKeyboardEvent(Camera::CameraActionType::RIGHT, gDt);
                    //     break;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                // switch (event.button.button)
                // {
                // case SDL_BUTTON_LEFT:
                //     SDL_ShowSimpleMessageBox(0, "Mouse", "Left button was pressed!", _window);
                //     break;
                // case SDL_BUTTON_RIGHT:
                //     SDL_ShowSimpleMessageBox(0, "Mouse", "Right button was pressed!", _window);
                //     break;
                // default:
                //     SDL_ShowSimpleMessageBox(0, "Mouse", "Some other button was pressed!", _window);
                //     break;
                // }
                _camera.handleMouseClickEvent(
                    event.button.button,
                    event.button.state,
                    event.button.x,
                    event.button.y);
                break;
            case SDL_MOUSEMOTION:
                _camera.handleMouseCursorEvent(
                    event.button.button,
                    event.motion.state,
                    glm::ivec2(event.motion.x,event.motion.y),
                    _winDimension);
                break;
            case SDL_MOUSEWHEEL:
                _camera.handleMouseWheelEvent(event.wheel.y * 0.1);
                break;
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    _winDimension[0] = event.window.data1;
                    _winDimension[1] = event.window.data2;
                }
                break;
            }
        }
    }
    inline SDL_Window *nativeHandle() const
    {
        return this->_window;
    };

private:
    SDL_Window *_window;
    CameraBase &_camera;
    glm::ivec2 _winDimension{0};

    // bool _firstMouseCursor{true};
    // int _lastMouseCursorX;
    // int _lastMouseCursorY;
};