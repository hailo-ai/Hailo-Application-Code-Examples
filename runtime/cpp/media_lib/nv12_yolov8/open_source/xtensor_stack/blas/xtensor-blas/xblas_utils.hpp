/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_UTILS_HPP
#define XBLAS_UTILS_HPP

#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "xtensor-blas/xblas_config.hpp"
#include "xflens/cxxblas/typedefs.h"
#include "xtensor/xutils.hpp"

#ifndef DEFAULT_LEADING_STRIDE_BEHAVIOR
#define DEFAULT_LEADING_STRIDE_BEHAVIOR XTENSOR_THROW(std::runtime_error, "No valid layout chosen.");
#endif

#ifndef DEFAULT_STORAGE_ORDER_BEHAVIOR
#define DEFAULT_STORAGE_ORDER_BEHAVIOR XTENSOR_THROW(std::runtime_error, "Cannot handle layout_type of e.");
#endif

namespace xt
{
    template <layout_type L = layout_type::row_major, class T>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<has_data_interface<std::decay_t<T>>::value && std::decay_t<T>::static_layout == L, T&&>
    {
        return std::forward<T>(t);
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_data_interface<std::decay_t<T>>::value || I::static_layout != L)
                            && detail::is_array<typename I::shape_type>::value,
                            xtensor<typename I::value_type,
                                    std::tuple_size<typename I::shape_type>::value,
                                    layout_remove_any(L)>>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto view_eval(T&& t)
        -> std::enable_if_t<(!has_data_interface<std::decay_t<T>>::value || I::static_layout != L) &&
                            !detail::is_array<typename I::shape_type>::value,
                            xarray<typename I::value_type, layout_remove_any(L)>>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout == L, T>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout != L && detail::is_array<typename I::shape_type>::value,
                            xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value, L>>
    {
        return t;
    }

    template <layout_type L = layout_type::row_major, class T, class I = std::decay_t<T>>
    inline auto copy_to_layout(T&& t)
        -> std::enable_if_t<std::decay_t<T>::static_layout != L && !detail::is_array<typename I::shape_type>::value,
                            xarray<typename I::value_type, L>>
    {
        return t;
    }

    template <class E>
    inline cxxblas::StorageOrder get_blas_storage_order(const E& e)
    {
        if (e.layout() == layout_type::row_major)
        {
            return cxxblas::StorageOrder::RowMajor;
        }
        else if (e.layout() == layout_type::column_major)
        {
            return cxxblas::StorageOrder::ColMajor;
        }
        DEFAULT_STORAGE_ORDER_BEHAVIOR;
    }

    /**
     * Get leading stride
     */

    namespace detail
    {
        template <class T, class U>
        inline blas_index_t get_leading_stride_impl(const T& str, const U& sh)
        {
            return str == T(0) ? static_cast<blas_index_t>(sh) : static_cast<blas_index_t>(str);
        }
    }

    template <class A, std::enable_if_t<A::static_layout == layout_type::row_major>* = nullptr>
    inline blas_index_t get_leading_stride(const A& a)
    {
        return detail::get_leading_stride_impl(a.strides().front(), a.shape().back());
    }

    template <class A, std::enable_if_t<A::static_layout == layout_type::column_major>* = nullptr>
    inline blas_index_t get_leading_stride(const A& a)
    {
        return detail::get_leading_stride_impl(a.strides().back(), a.shape().front());
    }

    template <class A, std::enable_if_t<A::static_layout != layout_type::row_major && A::static_layout != layout_type::column_major>* = nullptr>
    inline blas_index_t get_leading_stride(const A& a)
    {
        if (a.layout() == layout_type::row_major)
        {
            return detail::get_leading_stride_impl(a.strides().front(), a.shape().back());
        }
        else if (a.layout() == layout_type::column_major)
        {
            return detail::get_leading_stride_impl(a.strides().back(), a.shape().front());
        }
        DEFAULT_LEADING_STRIDE_BEHAVIOR;
    }

    /********************
     * Get front stride *
     ********************/

    template <class E>
    inline blas_index_t stride_front(const E& e)
    {
        if (E::static_layout == layout_type::column_major)
        {
            return blas_index_t(1);
        }
        else
        {
            return static_cast<blas_index_t>(e.strides().front() == 0 ? 1 : e.strides().front());
        }
    }

    /*******************
     * Get back stride *
     *******************/

    template <class E>
    inline blas_index_t stride_back(const E& e)
    {
        if (E::static_layout == layout_type::row_major)
        {
            return blas_index_t(1);
        }
        else
        {
            return static_cast<blas_index_t>(e.strides().back() == 0 ? 1 : e.strides().back());
        }
    }

    /*******************************
     * is_xfunction implementation *
     *******************************/

    namespace detail
    {
        template<class xF, class xR, class fE, class... xE>
        std::true_type  is_xfunction_impl(const xfunction<xF, xR, fE, xE...>&);

        std::false_type is_xfunction_impl(...);
    }

    template<class T>
    constexpr bool is_xfunction(T&& t) {
        return decltype(detail::is_xfunction_impl(t))::value;
    }

    /***********************************
     * assert_nd_square implementation *
     ***********************************/

    template <class T>
#if !defined(_MSC_VER) || _MSC_VER >= 1910
    constexpr
#endif
    void assert_nd_square(const xexpression<T>& t)
    {
        auto& dt = t.derived_cast();
        if (dt.shape()[dt.dimension() - 1] != dt.shape()[dt.dimension() - 2])
        {
            XTENSOR_THROW(std::runtime_error, "Last 2 dimensions of the array must be square.");
        }
    }
}
#endif
