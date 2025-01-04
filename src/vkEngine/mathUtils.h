#pragma once

#include <type_traits>

namespace vkEngine::math
{
    // limit T to be numerical type
    // template <typename T>
    // concept FloatOrInt = std::is_same_v<T, int> || std::is_same_v<T, float>;

    // template <FloatOrInt T>
    // struct MyStruct
    // {
    //     T myFloatOrInt;
    // };

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    struct cuComplex
    {
        T r;
        T i;

        cuComplex(T r, T i)
            : r{r}, i{i}
        {
        }

        cuComplex() = delete;
        ~cuComplex() = default;

        cuComplex operator*(const cuComplex &other)
        {
            return cuComplex(r * other.r - i * other.i, i * other.r + r * other.i);
        }

        cuComplex operator+(const cuComplex &other)
        {
            return cuComplex(r + other.r, i + other.i);
        }

        T magnitude2() {
            return r * r + i * i;
        }
    };

    using cuComplexf = cuComplex<float>;
};
