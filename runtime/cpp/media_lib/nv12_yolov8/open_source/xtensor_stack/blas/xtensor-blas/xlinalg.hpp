/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XLINALG_HPP
#define XLINALG_HPP

#include <algorithm>
#include <limits>
#include <sstream>
#include <chrono>

#include "xtl/xcomplex.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlapack.hpp"
#include "xtensor-blas/xblas_utils.hpp"

namespace xt
{
namespace linalg
{

    /// Selects special norm orders
    enum class normorder {
        frob,    ///< Frobenius norm
        nuc,     ///< Nuclear norm
        inf,     ///< Positive infinity norm
        neg_inf  ///< Negative infinity norm
    };

    /**
     * Calculate norm of vector, or matrix
     *
     * @param vec input vector
     * @param ord order of norm. This can be any integer for a vector
     *            or [-2,-1,1,2] for a matrix.
     * @return scalar result
     *
     * @tparam type of xexpression
     */
    template <class E>
    auto norm(const xexpression<E>& vec, int ord)
    {
        using value_type = typename E::value_type;
        using underlying_value_type = xtl::complex_value_type_t<value_type>;

        const auto& v = vec.derived_cast();

        underlying_value_type result = 0;
        if (v.dimension() == 1)
        {
            if (ord == 1)
            {
                if (xtl::is_complex<value_type>::value)
                {
                    for (std::size_t i = 0; i < v.size(); ++i)
                    {
                        result += std::abs(v(i));
                    }
                }
                else
                {
                    blas::asum(v, result);
                }
            }
            else if (ord == 2)
            {
                blas::nrm2(v, result);
            }
            else if (ord == 0)
            {
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    result += (v(i) != underlying_value_type(0));
                }
            }
            else
            {
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    result += std::abs(std::pow(v(i), ord));
                }
                result = std::pow(result, 1. / static_cast<double>(ord));
            }
            return result;
        }
        else if (v.dimension() == 2)
        {
            if (ord == 1 || ord == -1)
            {
                xtensor<underlying_value_type, 1> s = sum(abs(v), {0});
                if (ord == 1)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
            if (ord == 2 || ord == -2)
            {
                auto M = copy_to_layout<layout_type::column_major>(v);
                auto gesdd_res = lapack::gesdd(M, 'N');
                auto& s = std::get<2>(gesdd_res);
                if (ord == 2)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
        }
        std::stringstream ss;
        ss << "Norm " << ord << " not implemented!" << std::endl;
        XTENSOR_THROW(std::runtime_error, ss.str());
    }

    /**
     * Calculate matrix or vector norm using \ref normorder.
     *
     * @param vec The matrix or vector to take the norm of
     * @param ord normorder (frob, nuc, inf, neg_inf)
     *
     * @return norm value
     */
    template <class E>
    auto norm(const xexpression<E>& vec, normorder ord)
    {
        using value_type = xtl::complex_value_type_t<typename E::value_type>;

        const auto& v = vec.derived_cast();
        if (v.dimension() == 2)
        {
            if (ord == normorder::frob)
            {
                return static_cast<value_type>(std::sqrt(sum(pow(abs(v), 2))()));
            }
            if (ord == normorder::nuc)
            {
                auto M = copy_to_layout<layout_type::column_major>(v);
                auto gesdd_res = lapack::gesdd(M, 'N');
                auto& s = std::get<2>(gesdd_res);
                return std::accumulate(s.begin(), s.end(), value_type(0));
            }
            if (ord == normorder::inf || ord == normorder::neg_inf)
            {
                xtensor<value_type, 1> s = xt::sum(abs(v), {1});
                if (ord == normorder::inf)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
        }
        else if (v.dimension() == 1)
        {
            if (ord == normorder::inf || ord == normorder::neg_inf)
            {
                auto s = abs(v);
                if (ord == normorder::inf)
                {
                    return *std::max_element(s.begin(), s.end());
                }
                else
                {
                    return *std::min_element(s.begin(), s.end());
                }
            }
        }
        std::stringstream ss;
        ss << "Norm not implemented!" << std::endl;
        XTENSOR_THROW(std::runtime_error, ss.str());
    }

    /**
     * Calculate default norm (2-norm for vector, Frobenius norm for matrix)
     *
     * @param vec Input vector or matrix
     * @return norm
     */
    template <class E>
    typename E::value_type norm(const xexpression<E>& vec)
    {
        const auto& v = vec.derived_cast();
        if (v.dimension() == 1)
        {
            return norm(vec, 2);
        }
        else
        {
            return norm(vec, normorder::frob);
        }
    }

    /**
     * Solve a linear matrix equation, or system of linear scalar equations.
     * Computes the “exact” solution, x, of the well-determined, i.e., full rank,
     * linear matrix equation ax = b.
     *
     * @param a Coefficient matrix
     * @param b Ordinate or “dependent variable” values.
     * @return Solution to the system a x = b. Returned shape is identical to b.
     */
    template <class E1, class E2>
    auto solve(const xexpression<E1>& A, const xexpression<E2>& b)
    {
        assert_nd_square(A);
        auto dA = copy_to_layout<layout_type::column_major>(A.derived_cast());
        auto db = copy_to_layout<layout_type::column_major>(b.derived_cast());

        int info = lapack::gesv(dA, db);

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "The solution could not be computed");
        }

        return db;
    }

    /**
     * Compute the (multiplicative) inverse of a matrix.
     *
     * @param A xexpression to be inverted
     * @return (Multiplicative) inverse of the matrix a.
     */
    template <class E1>
    auto inv(const xexpression<E1>& A)
    {
        assert_nd_square(A);
        auto dA = copy_to_layout<layout_type::column_major>(A.derived_cast());

        uvector<blas_index_t> piv(std::min(dA.shape()[0], dA.shape()[1]));

        // DEV note: numpy uses gesv here, instead of getrf and getri. Might
        //           be interesting to investigate if there is a perf or accuracy
        //           difference.
        int info = lapack::getrf(dA, piv);
        if (info > 0)
        {
            XTENSOR_THROW(std::runtime_error, "Singular matrix not invertible (getrf).");
        }

        info = lapack::getri(dA, piv);
        if (info > 0)
        {
            XTENSOR_THROW(std::runtime_error, "Singular matrix not invertible (getri).");
        }
        return dA;
    }

    /**
     * Calculate the condition number of matrix M
     */
    template <class E>
    auto cond(const xexpression<E>& M, int ord)
    {
        return norm(M, ord) * norm(inv(M), ord);
    }

    template <class E>
    auto cond(const xexpression<E>& M, normorder ord)
    {
        return norm(M, ord) * norm(inv(M), ord);
    }

    /**
     * Compute the eigenvalues and right eigenvectors of a square array.
     *
     * @param Matrix for which the eigenvalues and right eigenvectors will be computed
     * @return (eigenvalues, eigenvectors) tuple. The first element corresponds to the eigenvalues,
     *         each repeated according to its multiplicity. The eigenvalues are not necessarily
     *         ordered. The second (1) element are the normalized (unit “length”) eigenvectors,
     *         such that the column v[:, i] corresponds to the eigenvalue w[i].
     */
    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eig(const xexpression<E>& A)
    {
        using underlying_type = typename E::value_type;
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> wr(vN), wi(vN);

        std::array<std::size_t, 2> shp = {N, N};
        xtensor<value_type, 2, layout_type::column_major> VL(shp), VR(shp);

        // jobvl N: left eigenvectors of A are not computed
        // jobvr V: then right eigenvectors of A are computed
        int info = lapack::geev(M, 'N', 'V', wr, wi, VL, VR);

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue calculation did not converge.");
        }

        auto eig_vecs = xtensor<std::complex<underlying_type>, 2>::from_shape({ N, N });
        auto eig_vals = xtensor<std::complex<underlying_type>, 1>::from_shape({ N });

        xt::real(eig_vals) = wr;
        xt::imag(eig_vals) = wi;

        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
            {
                if (wi(j) != 0)
                {
                    eig_vecs(i, j)     = std::complex<underlying_type>(VR(i, j),  VR(i, j + 1));
                    eig_vecs(i, j + 1) = std::complex<underlying_type>(VR(i, j), -VR(i, j + 1));
                    ++j;
                }
                else
                {
                    eig_vecs(i, j) = std::complex<underlying_type>(VR(i, j), 0);
                }
            }
        }

        return std::make_tuple(std::move(eig_vals), std::move(eig_vecs));
    }

    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eig(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        std::array<std::size_t, 2> shp = {N, N};
        xtensor<value_type, 2, layout_type::column_major> VL(shp), VR(shp);

        // jobvl N: left eigenvectors of A are not computed
        // jobvr V: then right eigenvectors of A are computed
        int info = lapack::geev(M, 'N', 'V', w, VL, VR);

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue calculation did not converge.");
        }

        return std::make_tuple(std::move(w), std::move(VR));
    }

    /**
     * Compute the eigenvalues and eigenvectors of a square Hermitian or real symmetric xexpression.
     *
     * @param A Matrix for which the eigenvalues and right eigenvectors are computed
     * @return xtensor containing the eigenvalues.
     */
    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigh(const xexpression<E>& A, char UPLO = 'L')
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::syevd(M, 'V', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return std::make_tuple(std::move(w), std::move(M));
    }

    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigh(const xexpression<E>& A, char UPLO = 'L')
    {
        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<underlying_value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::heevd(M, 'V', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return std::make_tuple(std::move(w), std::move(M));
    }

    /**
     * Compute the generalized eigenvalues and eigenvectors of a square Hermitian or real symmetric
     * xexpression.
     *
     * @param A,B Matrices for which the generalized eigenvalues and right eigenvectors are computed
     * @return xtensor containing the eigenvalues.
     */
    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigh(const xexpression<E>& A, const xexpression<E>& B, const char UPLO = 'L')
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M1 = copy_to_layout<layout_type::column_major>(A.derived_cast());
        auto M2 = copy_to_layout<layout_type::column_major>(B.derived_cast());

        std::size_t N = M1.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::sygvd(M1, M2, 1, 'V', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return std::make_tuple(std::move(w), std::move(M1));
    }

    #if 0
    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigh(const xexpression<E>& A, char UPLO = 'L')
    {
        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<underlying_value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::heevd(M, 'V', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return std::make_tuple(std::move(w), std::move(M));
    }
    #endif

    /**
     * Compute the eigenvalues of a square xexpression.
     *
     * @param A Matrix for which the eigenvalues are computed
     * @return xtensor containing the eigenvalues.
     */
    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvals(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> wr(vN);
        xtensor<value_type, 1, layout_type::column_major> wi(vN);

        // TODO check if we can remove allocation and pass nullptr as VL / VR
        std::array<std::size_t, 2> shp = {N, N};
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        auto info = lapack::geev(M, 'N', 'N', wr, wi, VL, VR);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Failed to compute eigenvalue " +
                std::to_string(std::abs(info)) + ".");
        }

        xtensor<std::complex<value_type>, 1> eig_vals;
        eig_vals.resize({N});

        xt::real(eig_vals) = wr;
        xt::imag(eig_vals) = wi;

        return eig_vals;
    }

    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvals(const xexpression<E>& A)
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        // TODO check if we can remove allocation and pass nullptr as VL / VR
        std::array<std::size_t, 2> shp = {N, N};
        xtensor<value_type, 2, layout_type::column_major> VL(shp);
        xtensor<value_type, 2, layout_type::column_major> VR(shp);

        auto info = lapack::geev(M, 'N', 'N', w, VL, VR);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Failed to compute eigenvalue " +
                std::to_string(std::abs(info)) + ".");
        }

        using value_type = typename E::value_type;

        return w;
    }

    /**
     * Compute the eigenvalues of a Hermitian or real symmetric matrix xexpression.
     *
     * @param Matrix for which the eigenvalues are computed
     * @return xtensor containing the eigenvalues.
     */
    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvalsh(const xexpression<E>& A, char UPLO = 'L')
    {
        using value_type = typename E::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::syevd(M, 'N', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return w;
    }

    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto eigvalsh(const xexpression<E>& A, char UPLO = 'L')
    {
        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        std::size_t N = M.shape()[0];
        std::array<std::size_t, 1> vN = {N};
        xtensor<underlying_value_type, 1, layout_type::column_major> w(vN);

        int info = lapack::heevd(M, 'N', UPLO, w);
        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Eigenvalue computation did not converge.");
        }

        return w;
    }

    namespace detail
    {
        template <class A>
        struct offset_iter_without_axis
        {

            using shape_type = typename A::shape_type;
            using size_type = typename A::size_type;
            using index_type = xindex_type_t<shape_type>;

            offset_iter_without_axis(const A& a, std::size_t axis)
                : m_a(a), m_axis(axis)
            {
                resize_container(m_idx, a.dimension());
                std::fill(m_idx.begin(), m_idx.end(), 0);
                m_offset = 0;
            }

            #define SC(X) static_cast<std::size_t>(X)
            inline bool next()
            {
                size_type dim = static_cast<size_type>(m_a.dimension());
                for (size_type j = dim; j != 0; --j)
                {
                    size_type i = j - 1;
                    if (i == m_axis)
                    {
                        // skip
                    }
                    else if (m_idx[SC(i)] == m_a.shape()[SC(i)] - 1)
                    {
                        m_offset -= static_cast<std::ptrdiff_t>(m_idx[i]) * m_a.strides()[i];
                        m_idx[i] = size_type(0);
                        if (i == 0 || m_axis == 0 && i == 1)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        ++m_idx[i];
                        m_offset += m_a.strides()[i];
                        return true;
                    }
                }
                return false;
            }
            #undef SC
            inline std::ptrdiff_t offset() const
            {
                return m_offset;
            }

        private:
            const A& m_a;
            index_type m_idx;
            size_type m_axis;
            std::ptrdiff_t m_offset;
        };
    }

    /**
     * Non-broadcasting dot function.
     * In the case of two 1D vectors, computes the vector dot
     * product. In the case of complex vectors, computes the dot
     * product without conjugating the first argument.
     * If \em t or \em o is a 2D matrix, computes the matrix-times-vector
     * product. If both \em t and \em o ar 2D matrices, computes
     * the matrix-product.
     *
     * @param t input array
     * @param o input array
     *
     * @return resulting array
     */
    template <class T, class O>
    auto dot(const xexpression<T>& xt, const xexpression<O>& xo)
    {
        using value_type = std::common_type_t<typename T::value_type, typename O::value_type>;

        using return_type = std::conditional_t<(T::static_layout == O::static_layout) &&
                                               (T::static_layout != layout_type::dynamic && T::static_layout != layout_type::any),
                                               xarray<value_type, T::static_layout>,
                                               xarray<value_type, XTENSOR_DEFAULT_LAYOUT>>;
        return_type result;

        auto&& t = view_eval<T::static_layout>(xt.derived_cast());
        auto&& o = view_eval<O::static_layout>(xo.derived_cast());

        // is one of each a scalar? just multiply
        if (t.dimension() == 0 || o.dimension() == 0)
        {
            return return_type(t * o);
        }
        if (t.dimension() == 1 && o.dimension() == 1)
        {
            result.resize(std::vector<std::size_t>{1});
            if (t.shape()[0] != o.shape()[0])
            {
                XTENSOR_THROW(std::runtime_error, "Dot: shape mismatch.");
            }

            if (xtl::is_complex<typename T::value_type>::value)
            {
                blas::dotu(t, o, result(0));
            }
            else
            {
                blas::dot(t, o, result(0));
            }
            return result;
        }
        else
        {
            if (t.dimension() == 2 && o.dimension() == 1)
            {
                XTENSOR_ASSERT(t.layout() == layout_type::row_major || t.layout() == layout_type::column_major);
                XTENSOR_ASSERT(std::min(t.strides()[0], t.strides()[1]) <= 1);

                if (t.shape()[1] != o.shape()[0])
                {
                    XTENSOR_THROW(std::runtime_error, "Dot: shape mismatch.");
                }

                result.resize({static_cast<std::size_t>(t.shape()[0])});

                blas_index_t shape_x, shape_y;
                cxxblas::Transpose trans;
                if (result.layout() != t.layout())
                {
                    shape_x = static_cast<blas_index_t>(t.shape()[1]);
                    shape_y = static_cast<blas_index_t>(t.shape()[0]);
                    trans = cxxblas::Transpose::Trans;
                }
                else
                {
                    shape_x = static_cast<blas_index_t>(t.shape()[0]);
                    shape_y = static_cast<blas_index_t>(t.shape()[1]);
                    trans = cxxblas::Transpose::NoTrans;
                }

                cxxblas::gemv<blas_index_t>(
                    get_blas_storage_order(result),
                    trans,
                    shape_x,
                    shape_y,
                    value_type(1.0),
                    t.data() + t.data_offset(),
                    get_leading_stride(t),
                    o.data() + o.data_offset(),
                    get_leading_stride(o),
                    value_type(0.0),
                    result.data(),
                    get_leading_stride(result)
                );
            }
            else if (t.dimension() == 1 && o.dimension() == 2)
            {
                XTENSOR_ASSERT(o.layout() == layout_type::row_major || o.layout() == layout_type::column_major);
                XTENSOR_ASSERT(std::min(o.strides()[0], o.strides()[1]) <= 1);

                if (t.shape()[0] != o.shape()[0])
                {
                    XTENSOR_THROW(std::runtime_error, "Dot: shape mismatch.");
                }

                result.resize({static_cast<std::size_t>(o.shape()[1])});

                blas_index_t shape_x, shape_y;
                cxxblas::Transpose trans;
                if (result.layout() != o.layout())
                {
                    shape_x = static_cast<blas_index_t>(o.shape()[1]);
                    shape_y = static_cast<blas_index_t>(o.shape()[0]);
                    trans = cxxblas::Transpose::NoTrans;
                }
                else
                {
                    shape_x = static_cast<blas_index_t>(o.shape()[0]);
                    shape_y = static_cast<blas_index_t>(o.shape()[1]);
                    trans = cxxblas::Transpose::Trans;
                }

                cxxblas::gemv<blas_index_t>(
                    get_blas_storage_order(result),
                    trans,
                    shape_x,
                    shape_y,
                    value_type(1.0),
                    o.data() + o.data_offset(),
                    get_leading_stride(o),
                    t.data() + t.data_offset(),
                    get_leading_stride(t),
                    value_type(0.0),
                    result.data(),
                    get_leading_stride(result)
                );
            }
            else if (t.dimension() == 2 && o.dimension() == 2)
            {
                XTENSOR_ASSERT(o.layout() == layout_type::row_major || o.layout() == layout_type::column_major);
                XTENSOR_ASSERT(std::min(o.strides()[0], o.strides()[1]) <= 1);
                XTENSOR_ASSERT(t.layout() == layout_type::row_major || t.layout() == layout_type::column_major);
                XTENSOR_ASSERT(std::min(t.strides()[0], t.strides()[1]) <= 1);

                if (t.shape()[1] != o.shape()[0])
                {
                    XTENSOR_THROW(std::runtime_error, "Dot: shape mismatch.");
                }

                cxxblas::Transpose transpose_A = cxxblas::Transpose::NoTrans,
                                   transpose_B = cxxblas::Transpose::NoTrans;

                if (result.layout() != t.layout())
                {
                    transpose_A = cxxblas::Transpose::Trans;
                }
                if (result.layout() != o.layout())
                {
                    transpose_B = cxxblas::Transpose::Trans;
                }

                // This adds a fast path for A * A' by calling SYRK and only computing
                // the upper triangle
                if (std::is_same<typename T::value_type, typename O::value_type>::value &&
                    (static_cast<const void*>(t.data() + t.data_offset()) == static_cast<const void*>(o.data() + o.data_offset())) &&
                    ((transpose_A == cxxblas::Transpose::Trans && transpose_B == cxxblas::Transpose::NoTrans) ||
                     (transpose_A == cxxblas::Transpose::NoTrans && transpose_B == cxxblas::Transpose::Trans)))
                {
                    // TODO add check to compare strides & shape

                    result.resize({static_cast<std::size_t>(t.shape()[0]), static_cast<std::size_t>(t.shape()[0])});

                    cxxblas::syrk<blas_index_t>(
                        get_blas_storage_order(result),
                        cxxblas::StorageUpLo::Upper,
                        transpose_A,
                        static_cast<blas_index_t>(t.shape()[0]),
                        static_cast<blas_index_t>(t.shape()[1]),
                        value_type(1.0),
                        t.data() + t.data_offset(),
                        get_leading_stride(t),
                        value_type(0.0),
                        result.data(),
                        get_leading_stride(result)
                    );

                    for (std::size_t i = 0; i < t.shape()[0]; ++i)
                    {
                        for (std::size_t j = i + 1; j < t.shape()[0]; ++j)
                        {
                            result(j, i) = result(i, j);
                        }
                    }
                    return result;
                }

                result.resize({static_cast<std::size_t>(t.shape()[0]), static_cast<std::size_t>(o.shape()[1])});

                cxxblas::gemm<blas_index_t>(
                    get_blas_storage_order(result),
                    transpose_A,
                    transpose_B,
                    static_cast<blas_index_t>(t.shape()[0]),
                    static_cast<blas_index_t>(o.shape()[1]),
                    static_cast<blas_index_t>(o.shape()[0]),
                    value_type(1.0),
                    t.data() + t.data_offset(),
                    get_leading_stride(t),
                    o.data() + o.data_offset(),
                    get_leading_stride(o),
                    value_type(0.0),
                    result.data(),
                    get_leading_stride(result)
                );
            }
            else
            {
                // TODO more testing for different layouts!
                std::size_t l = t.shape().back();
                std::size_t match_dim = 0;

                if (o.dimension() > 1)
                {
                    match_dim = o.dimension() - 2;
                }
                if (o.shape()[match_dim] != l)
                {
                    XTENSOR_THROW(std::runtime_error, "Dot: shape mismatch.");
                }

                blas_index_t a_dim = static_cast<blas_index_t>(t.dimension());
                blas_index_t b_dim = static_cast<blas_index_t>(o.dimension());

                blas_index_t nd = a_dim + b_dim - 2;

                std::size_t j = 0;
                std::vector<std::size_t> dimensions(static_cast<std::size_t>(nd));

                for (blas_index_t i = 0; i < a_dim - 1; ++i)
                {
                    dimensions[j++] = t.shape()[static_cast<std::size_t>(i)];
                }
                for (blas_index_t i = 0; i < b_dim - 2; ++i)
                {
                    dimensions[j++] = o.shape()[static_cast<std::size_t>(i)];
                }
                if (b_dim > 1)
                {
                    dimensions[j++] = o.shape().back();
                }

                result.resize(dimensions);

                blas_index_t a_stride = static_cast<blas_index_t>(t.strides().back());
                blas_index_t b_stride = static_cast<blas_index_t>(o.strides()[match_dim]);

                auto a_iter = detail::offset_iter_without_axis<std::decay_t<decltype(t)>>(t, t.dimension() - 1);
                auto b_iter = detail::offset_iter_without_axis<std::decay_t<decltype(o)>>(o, match_dim);

                value_type temp;
                auto result_it = result.begin();

                do
                {
                    do
                    {
                        cxxblas::dot<blas_index_t>(
                            static_cast<blas_index_t>(l),
                            t.data() + a_iter.offset(),
                            a_stride,
                            o.data() + b_iter.offset(),
                            b_stride,
                            temp
                        );
                        *(result_it++) = temp;

                    } while (b_iter.next());

                } while (a_iter.next());

            }
            return result;
        }
    }

    /**
     * Computes the dot product for two vectors.
     *
     * Behaves different from \ref dot in the case of complex
     * vectors. If vectors are complex, vdot conjugates the first
     * argument \em t.
     * Note: Unlike NumPy, xtensor-blas currently doesn't flatten
     * the input arguments.
     *
     * @param t input vector (1D)
     * @param o input vector (1D)
     *
     * @return resulting array
     */
    template <class T, class O>
    auto vdot(const xexpression<T>& a, const xexpression<O>& b) {

        using common_type = std::common_type_t<typename T::value_type, typename O::value_type>;

        XTENSOR_ASSERT(a.derived_cast().dimension() == 1);
        XTENSOR_ASSERT(b.derived_cast().dimension() == 1);

        common_type result = 0;
        blas::dot(a, b, result);

        return result;
    }

    /**
     * Compute the outer product of two vectors.
     *
     * @param t input vector (1D)
     * @param o input vector (1D)
     *
     * @return resulting array
     */
    template <class T, class O>
    auto outer(const xexpression<T>& a, const xexpression<O>& b) {
        using common_type = std::common_type_t<typename T::value_type, typename O::value_type>;
        using return_type = xtensor<common_type, 2>;

        XTENSOR_ASSERT(a.derived_cast().dimension() == 1);
        XTENSOR_ASSERT(b.derived_cast().dimension() == 1);

        typename return_type::shape_type s = {a.derived_cast().shape()[0], b.derived_cast().shape()[0]};
        return_type result(s, common_type(0));

        blas::ger(a, b, result);

        return result;
    }

    /**
     * Compute the determinant by utilizing LU factorization
     *
     * @param A matrix for which determinant is to be computed
     * @returns determinant of the \em A
     */
    template <class T>
    auto det(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        assert_nd_square(A);

        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();
        uvector<blas_index_t> piv(std::min(LU.shape()[0], LU.shape()[1]));

        lapack::getrf(LU, piv);

        value_type result(1);
        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            if (piv[i] != int(i + 1))
            {
                result *= value_type(-1);
            }
        }

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            result *= LU(i, i);
        }
        return result;
    }

    /**
     * Compute the sign and (natural) logarithm of the determinant of an xexpression.
     *
     * If an array has a very small or very large determinant, then a call to det may
     * overflow or underflow. This routine is more robust against such issues, because
     * it computes the logarithm of the determinant rather than the determinant itself.
     *
     * @param A matrix for which determinant is to be computed
     * @returns tuple containing (sign, determinant)
     */
    template <class T, std::enable_if_t<xtl::is_complex<typename T::value_type>::value, int> = 0>
    auto slogdet(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        assert_nd_square(A);

        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();
        uvector<blas_index_t> piv(std::min(LU.shape()[0], LU.shape()[1]));

        int info = lapack::getrf(LU, piv);

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "LU factorization did not compute.");
        }

        value_type result(0);
        int sign = 0;
        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            sign += (piv[i] != int(i + 1));
        }

        value_type v_sign = (sign % 2) ? -1 : 1;

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            auto abs_elem = std::abs(LU(i, i));
            v_sign *= (LU(i, i) / abs_elem);
            result += std::log(abs_elem);
        }
        return std::make_tuple(std::move(v_sign), std::move(result));
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <class T, std::enable_if_t<!xtl::is_complex<typename T::value_type>::value, int> = 0>
    auto slogdet(const xexpression<T>& A)
    {
        using value_type = typename T::value_type;
        assert_nd_square(A);

        xtensor<value_type, 2, layout_type::column_major> LU = A.derived_cast();
        uvector<blas_index_t> piv(std::min(LU.shape()[0], LU.shape()[1]));

        int info = lapack::getrf(LU, piv);

        if (info != 0)
        {
            return std::make_tuple(value_type(0),
                                   -std::numeric_limits<value_type>::infinity());
        }

        value_type result(0);
        int sign = 0;

        for (std::size_t i = 0; i < piv.size(); ++i)
        {
            sign += (piv[i] != int(i + 1));
        }

        for (std::size_t i = 0; i < LU.shape()[0]; ++i)
        {
            value_type abs_el = LU(i, i);
            if (abs_el < 0)
            {
                sign += 1;
                abs_el = -abs_el;
            }
            result += std::log(abs_el);
        }

        value_type v_sign = (sign % 2) ? -1 : 1;
        return std::make_tuple(std::move(v_sign), std::move(result));
    }
    /// @endcond

    namespace detail
    {
        template <class E, class T>
        inline auto call_gqr(E& A, T& tau, blas_index_t n)
            -> std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>
        {
            int info = lapack::orgqr(A, tau, n);
            if (info > 0)
            {
                XTENSOR_THROW(std::runtime_error, "Could not find Q (orgqr).");
            }
        }

        template <class E, class T>
        inline auto call_gqr(E& A, T& tau, blas_index_t n)
            -> std::enable_if_t<xtl::is_complex<typename E::value_type>::value>
        {
            int info = lapack::ungqr(A, tau, n);
            if (info > 0)
            {
                XTENSOR_THROW(std::runtime_error, "Could not find Q (ungqr).");
            }
        }
    }

    namespace xblas_detail
    {
        template <class T>
        inline void triu_inplace(T& R)
        {
            for (std::size_t i = 0; i < R.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < i && j < R.shape()[1]; j++)
                {
                    R(i, j) = 0;
                }
            }
        }
    }

    /// Select the mode for the qr decomposition ``K = min(M, K)``
    enum class qrmode {
        reduced,  ///< return Q, R with dimensions (M, K), (K, N) (default)
        complete, ///< return Q, R with dimensions (M, M), (M, N)
        r,        ///< return empty Q and R with dimensions (0, 0), (K, N)
        raw       ///< return H, Tau with dimensions (N, M), (K, 1)
    };

    /**
     * Compute the QR decomposition of \em A.
     * @param t The matrix to calculate Q and R for
     * @return std::tuple with Q and R
     */
    template <class T>
    auto qr(const xexpression<T>& A, qrmode mode = qrmode::reduced)
    {
        using value_type = typename T::value_type;
        using xtype = xarray<value_type, layout_type::column_major>;

        xtype R = A.derived_cast();

        std::size_t M = R.shape()[0];
        std::size_t N = R.shape()[1];
        std::size_t K = std::min(M, N);

        auto tau = xarray<value_type, layout_type::column_major>::from_shape({K});
        int info = lapack::geqrf(R, tau);

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "QR decomposition failed.");
        }

        // explicitly set shape/size == 0!
        auto Q = xtype::from_shape({0});

        if (mode == qrmode::r)
        {
            R = xt::view(R, range(0, K), all());
            xblas_detail::triu_inplace(R);
            return std::make_tuple(std::move(Q), std::move(R));
        }

        if (mode == qrmode::raw)
        {
            R = transpose(R);
            return std::make_tuple(std::move(R), std::move(tau));
        }

        blas_index_t mc;

        if (mode == qrmode::complete && M > N)
        {
            mc = static_cast<blas_index_t>(M);
            Q.resize({M, M});
        }
        else
        {
            mc = static_cast<blas_index_t>(K);
            Q.resize({M, N});
        }

        xt::view(Q, all(), range(0, N)) = R;
        detail::call_gqr(Q, tau, mc);

        Q = xt::view(Q, all(), range(0, mc));
        R = xt::view(R, range(0, mc), all());

        xblas_detail::triu_inplace(R);

        return std::make_tuple(std::move(Q), std::move(R));
    }

    /**
     * Compute the Cholesky decomposition of \em A.
     * @return the decomposed matrix
     */
    template <class T>
    auto cholesky(const xexpression<T>& A)
    {
        assert_nd_square(A);
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        int info = lapack::potr(M, 'L');

        if (info > 0)
        {
            XTENSOR_THROW(std::runtime_error, "Cholesky decomposition failed.");
        }

        // delete upper triangle
        XTENSOR_ASSERT(M.shape()[0] > 1 && M.shape()[1] > 1);

        for (std::size_t i = 0; i < M.shape()[0]; ++i)
        {
            for (std::size_t j = i + 1; j < M.shape()[1]; ++j)
            {
                M(i, j) = 0;
            }
        }

        return M;
    }

    /**
    * Solves a system of linear equations M*X = B with a symmetric
    * where M = A*A**T if uplo is L.
    * Factorization of M can be computed with cholesky.
    * @return solution X
    */
    template <class T, class D>
    auto solve_cholesky(const xexpression<T>& A, const xexpression<D>& b)
    {
      assert_nd_square(A);
      auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());
      auto p = copy_to_layout<layout_type::column_major>(b.derived_cast());

      int info = lapack::potrs(M, p, 'L');

      if (info > 0)
      {
        XTENSOR_THROW(std::runtime_error, "Cholesky decomposition failed.");
      }

      return p;
    }

    /**
    * Solves Ax = b, where A is a lower triangular matrix
    * @return solution x
    */
    template <class T, class D>
    auto solve_triangular(const xexpression<T>& A, const xexpression<D>& b)
    {
      assert_nd_square(A);
      auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());
      auto p = copy_to_layout<layout_type::column_major>(b.derived_cast());

      int info = lapack::trtrs(M, p, 'L', 'N');

      if (info > 0)
      {
        XTENSOR_THROW(std::runtime_error, "Cholesky decomposition failed.");
      }

      return p;
    }

    /**
     * Compute the SVD decomposition of \em A.
     * @return tuple containing S, V, and D
     */
    template <class T>
    auto svd(const xexpression<T>& A, bool full_matrices = true, bool compute_uv = true)
    {
        auto M = copy_to_layout<layout_type::column_major>(A.derived_cast());

        char job_type = 'A';
        if (!compute_uv)
        {
            job_type = 'N';
        }
        else if (!full_matrices && compute_uv)
        {
            job_type = 'S';
        }

        auto result = lapack::gesdd(M, job_type);

        if (std::get<0>(result) > 0)
        {
            XTENSOR_THROW(std::runtime_error, "SVD decomposition failed.");
        }

        return std::make_tuple(std::move(std::get<1>(result)), std::move(std::get<2>(result)), std::move(std::get<3>(result)));
    }

    namespace detail
    {
        template <bool>
        struct conj_eval_impl
        {
            template <class E>
            static const E& run(const E& e)
            {
                return e;
            }
        };

        template <>
        struct conj_eval_impl<true>
        {
            template <class E>
            static auto run(const E& e)
            {
                return xt::conj(e);
            }
        };

        template <class E>
        inline decltype(auto) conj_eval(const E& e)
        {
            return conj_eval_impl<xtl::is_complex<typename E::value_type>::value>::run(e);
        }
    }

    /**
     * Calculate Moore-Rose pseudo inverse using LAPACK SVD.
     */
    template <class T>
    auto pinv(const xexpression<T>& A, double rcond = 1e-15)
    {
        using value_type = typename T::value_type;
        const auto& dA = A.derived_cast();

        xtensor<value_type, 2, layout_type::column_major> M = detail::conj_eval(dA);

        auto gesdd_res = svd(M, false, true);

        auto u = std::move(std::get<0>(gesdd_res));
        auto s = std::move(std::get<1>(gesdd_res));
        auto vt = std::move(std::get<2>(gesdd_res));

        using real_value_type = typename decltype(s)::value_type;
        real_value_type cutoff = static_cast<real_value_type>(rcond) * (*std::max_element(s.begin(), s.end()));

        for (std::size_t i = 0; i < s.size(); ++i)
        {
            if (s(i) > cutoff)
            {
                s(i) = real_value_type(1.) / s(i);
            }
            else
            {
                s(i) = 0;
            }
        }
        auto ut = xt::transpose(u);
        auto vww = xt::view(s, xt::all(), xt::newaxis());
        auto m = xt::eval(vww * ut);
        auto vtt = xt::transpose(vt);
        auto result = dot(vtt, m);
        return result;
    }

    /**
     * Calculate matrix power A**n
     *
     * @param mat  The matrix
     * @param n    The exponent
     *
     * @return resulting array
     */
    template <class E>
    auto matrix_power(const xexpression<E>& A, long n)
    {
        using input_type = std::decay_t<E>;
        using value_type = typename input_type::value_type;
        using xtype = xtensor<value_type, 2>;

        // copy input matrix
        xtype mat = A.derived_cast();

        XTENSOR_ASSERT(mat.dimension() == 2);
        XTENSOR_ASSERT(mat.shape()[0] == mat.shape()[1]);

        xtype res(mat.shape());
        if (n == 0)
        {
            res = eye(mat.shape()[0]);
            return res;
        }
        else if (n < 0)
        {
            mat = inv(mat);
            n = -n;
        }


        xtype temp(mat.shape());
        res = mat;
        if (n <= 3)
        {
            for (int i = 0; i < n - 1; ++i)
            {
                blas::gemm(res, mat, temp);
                res = temp;
            }
            return res;
        }

        // if n > 3, do a binary decomposition (copied from NumPy)
        long bits, var = n, i = 0;
        for(bits = 0; var != 0; ++bits)
        {
            var >>= 1;
        }
        while (~n & (1 << i))
        {
            blas::gemm(mat, mat, temp);
            temp = res;
            ++i;
        }
        ++i;
        res = mat;
        for (; i < bits; ++i)
        {
            blas::gemm(mat, mat, temp);
            mat = temp;
            if (n & (1 << i))
            {
                blas::gemm(res, mat, temp);
                res = temp;
            }
        }
        return res;
    }


    /**
     * Compute the trace of a xexpression.
     */
    template <class T>
    auto trace(const xexpression<T>& M, int offset = 0, int axis1 = 0, int axis2 = 1)
    {
        const auto& dM = M.derived_cast();
        auto d = xt::diagonal(dM, offset, std::size_t(axis1), std::size_t(axis2));

        std::size_t dim = d.dimension();
        if (dim == 1)
        {
            return xt::xarray<double>(xt::sum(d)());
        }
        else
        {
            return xt::xarray<double>(xt::sum(d, {dim - 1}));
        }
    }

    /**
     * Calculate the Kronecker product between two 2D xexpressions.
     */
    template <class T, class E>
    auto kron(const xexpression<T>& a, const xexpression<E>& b)
    {
        using value_type = std::common_type_t<typename T::value_type, typename E::value_type>;

        const auto& da = a.derived_cast();
        const auto& db = b.derived_cast();

        XTENSOR_ASSERT(da.dimension() == 2);
        XTENSOR_ASSERT(db.dimension() == 2);

        std::array<std::size_t, 2> shp = {da.shape()[0] * db.shape()[0], da.shape()[1] * db.shape()[1]};
        xtensor<value_type, 2> res(shp);

        for (std::size_t i = 0; i < da.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < da.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < db.shape()[0]; ++k)
                {
                    for (std::size_t h = 0; h < db.shape()[1]; ++h)
                    {
                        res(i * db.shape()[0] + k, j * db.shape()[1] + h) = da(i, j) * db(k, h);
                    }
                }
            }
        }

        return res;
    }

    /**
     * Calculate the matrix rank of \ref m.
     * If tol == -1, the tolerance is automatically computed.
     *
     * @param m matrix for which rank is calculated
     * @param tol tolerance for finding rank
     */
    template <class T>
    int matrix_rank(const xexpression<T>& m, double tol = -1.0)
    {
        using value_type = typename T::value_type;
        xtensor<value_type, 2, layout_type::column_major> M = m.derived_cast();

        auto svd_res = svd(m, true, false);
        auto s = std::get<1>(svd_res);
        auto max_el = std::max_element(s.begin(), s.end());

        if (tol == -1.0)
        {
            tol = (*max_el) * static_cast<double>(std::max(M.shape()[0], M.shape()[1])) * std::numeric_limits<value_type>::epsilon();
        }

        int sm = 0;
        for (const auto& el : s)
        {
            if (el > tol)
            {
                ++sm;
            }
        }
        return sm;
    }

    /**
     * Calculate the least-squares solution to a linear matrix equation.
     *
     * @param A coefficient matrix
     * @param b Ordinate, or dependent variable values. If b is two-dimensional,
     *          the least-squares solution is calculated for each of the K columns of b.
     * @param rcond Cut-off ratio for small singular values of \em A.
     *              For the purposes of rank determination, singular values are treated
     *              as zero if they are smaller than rcond times the largest singular value of a.
     *
     * @return tuple containing (x, residuals, rank, s) where:
     *         \em x is the least squares solution. Note that the solution is always returned as
     *               a 2D matrix where the columns are the solutions (even for a 1D \em b).
     *         \em s Sums of residuals; squared Euclidean 2-norm for each column in b - a*x.
     *               If the rank of \em A is < N or M <= N, this is an empty xtensor.
     *         \em rank the rank of \em A
     *         \em s singular values of \em A
     */
    template <class T, class E>
    auto lstsq(const xexpression<T>& A, const xexpression<E>& b, double rcond = -1.0)
    {
        using value_type = typename T::value_type;
        using underlying_value_type = xtl::complex_value_type_t<typename T::value_type>;

        xtensor<value_type, 2, layout_type::column_major> dA = A.derived_cast();

        std::size_t M = dA.shape()[0];
        std::size_t N = dA.shape()[1];

        auto& b_ref = b.derived_cast();

        if (dA.dimension() != 2)
        {
            XTENSOR_THROW(std::runtime_error, "Expected 2D expression for A");
        }

        if (!(b_ref.dimension() <= 2))
        {
            XTENSOR_THROW(std::runtime_error, "Expected 1- or 2D expression for A.");
        }

        if (b_ref.shape()[0] != M)
        {
            XTENSOR_THROW(std::runtime_error, "Shape of 'b' for lstsq does not match.");
        }

        // find number of rhs
        std::size_t nrhs = (b_ref.dimension() == 1) ? 1 : b_ref.shape()[1];

        // as the dgelsd docs say, on entry it's M-by-nrhs, then result N-by-nrhs
        // that is why we need to allocate *MORE* space than just b here for M > N
        auto db = xarray<value_type, layout_type::column_major>::from_shape({ std::max(M, N), nrhs });

        bool is_1d = false;
        if (b_ref.dimension() == 1)
        {
            is_1d = true;
            xt::view(db, range(0, M), xt::all()) = xt::view(b_ref, xt::all(), xt::newaxis());
        }
        else
        {
            xt::view(db, range(0, M), xt::all()) = b_ref;
        }

        auto s = xtensor<underlying_value_type, 1, layout_type::column_major>::from_shape({ std::min(M, N) });

        blas_index_t rank;

        lapack::gelsd(dA, db, s, rank, rcond);
        auto residuals = xtensor<underlying_value_type, 1>::from_shape({0});

        if (std::size_t(rank) == N && M > N)
        {
            residuals.resize({ db.shape()[1] });
            for (std::size_t i = 0; i < db.shape()[1]; ++i)
            {
                underlying_value_type temp = 0;
                for (std::size_t j = N; j < db.shape()[0]; ++j)
                {
                    temp += std::pow(std::abs(db(j, i)), 2);
                }
                residuals(i) = temp;
            }
        }

        auto vdb = view(db, range(std::size_t(0), N), xt::all());
        db = vdb;
        if (is_1d)
        {
            db = xt::squeeze(db);
        }

        return std::make_tuple(std::move(db), std::move(residuals), std::move(rank), std::move(s));
    }

    /**
     * @brief Non-broadcasting cross product between two vectors.
     *
     * Calculate cross product between two 1D vectors with 2- or 3 entries.
     * If only two entries are available, the third entry is assumed to be 0.
     *
     * @param a input vector
     * @param b input vector
     * @return resulting array
     */
    template <class E1, class E2>
    auto cross(const xexpression<E1>& a, const xexpression<E2>& b)
    {
        using return_type = xtensor<typename E1::value_type, 1>;
        auto res = return_type::from_shape({ 3 });
        const E1& da = a.derived_cast();
        const E2& db = b.derived_cast();

        if (da.size() == 3 && db.size() == 3)
        {
            res(0) = da(1) * db(2) - da(2) * db(1);
            res(1) = da(2) * db(0) - da(0) * db(2);
            res(2) = da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 2 && db.size() == 3)
        {
            res(0) =   da(1) * db(2);
            res(1) = -(da(0) * db(2));
            res(2) =   da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 3 && db.size() == 2)
        {
            res(0) = -(da(2) * db(1));
            res(1) =   da(2) * db(0);
            res(2) =   da(0) * db(1) - da(1) * db(0);
        }
        else if (da.size() == 2 && db.size() == 2)
        {
            res(0) = 0;
            res(1) = 0;
            res(2) = da(0) * db(1) - da(1) * db(0);
        }
        else
        {
            XTENSOR_THROW(std::runtime_error, "a or b did not have appropriate size 2 or 3.");
        }
        return res;
    }

    /**
     * @brief Compute tensor dot product along specified axes for arrays
     *
     * Compute the sum of products along the last \em naxes axes of a and first
     * \em naxes axes of b.
     *
     * @param xa input array
     * @param xb input array
     * @param naxes the number of axes to sum over
     * @return resulting array
     */
    template <class T, class O>
    auto tensordot(const xexpression<T>& xa, const xexpression<O>& xb, std::size_t naxes = 2)
    {
        using value_type = std::common_type_t<typename T::value_type, typename O::value_type>;
        using result_type = std::conditional_t<T::static_layout == O::static_layout &&
                                               (T::static_layout != layout_type::dynamic && T::static_layout != layout_type::any),
                                               xarray<value_type, T::static_layout>,
                                               xarray<value_type, XTENSOR_DEFAULT_LAYOUT>>;

        result_type result;
        auto&& a = view_eval<T::static_layout>(xa.derived_cast());
        auto&& b = view_eval<O::static_layout>(xb.derived_cast());
        if (naxes == 0)
        {
            // special case tensor outer product product
            xt::dynamic_shape<std::size_t> result_shape(a.dimension() + b.dimension());
            std::size_t j = 0;
            for (std::size_t i = 0; i < a.dimension(); ++i)
            {
                result_shape[j++] = a.shape()[i];
            }

            for (std::size_t i = 0; i < b.dimension(); ++i)
            {
                result_shape[j++] = b.shape()[i];
            }
            // flatten a/b
            auto vec_a = xt::ravel<result_type::static_layout>(a);
            auto vec_b = xt::ravel<result_type::static_layout>(b);
            // take the outer product of the two vectors
            result = outer(vec_a, vec_b);
            // reshape the result
            result.reshape(result_shape);
        }
        else
        {
            // Sum of products over last n axes of A and the first n axis of b
            XTENSOR_ASSERT(a.dimension() >= naxes);
            XTENSOR_ASSERT(b.dimension() >= naxes);

            auto as_it = a.shape().begin() + (a.dimension() - naxes);
            auto bs_it = b.shape().begin();
            std::size_t sum_len = 1;
            for (std::size_t i = 0; i < naxes; ++i)
            {
                auto a_val = *as_it;
                auto b_val = *bs_it;
                // check for axes size match
                if (a_val != b_val)
                {
                    XTENSOR_THROW(std::runtime_error, "Shape mismatch for sum");
                }
                else
                {
                    sum_len *= a_val;
                }
                ++as_it;
                ++bs_it;
            }
            xt::dynamic_shape<std::size_t> result_shape;
            std::size_t keep_a_len = 1;
            for (auto it = a.shape().begin(); it != a.shape().begin() + (a.dimension() - naxes); ++it)
            {
                std::size_t len = *it;
                keep_a_len *= len;
                result_shape.push_back(len);
            }
            std::size_t keep_b_len = 1;
            for (auto it = b.shape().begin() + naxes; it != b.shape().end(); ++it)
            {
                std::size_t len = *it;
                keep_b_len *= len;
                result_shape.push_back(len);
            }
            xarray<value_type, T::static_layout> a_mat = a;
            a_mat.reshape({keep_a_len, sum_len});
            xarray<value_type, O::static_layout> b_mat = b;
            b_mat.reshape({sum_len, keep_b_len});
            result = dot(a_mat, b_mat);
            if(result_shape.empty())
            {
                result.reshape({1});
            }
            else
            {
                result.reshape(result_shape);
            }

        }
        return result;
    }

    /**
     * @brief Compute tensor dot product along specified axes for arrays
     *
     * Compute the sum of products along the axes \em ax_a for a and \em ax_b for b
     *
     * @param xa input array
     * @param xb input array
     * @param ax_a axes to sum over for \em a
     * @param ax_b axes to sum over for \em b
     * @return resulting array
     */
    template <class T, class O>
    auto tensordot(const xexpression<T>& xa, const xexpression<O>& xb, const std::vector<std::size_t>& ax_a,
                   const std::vector<std::size_t>& ax_b)
    {
        auto&& a = view_eval<T::static_layout>(xa.derived_cast());
        auto&& b = view_eval<O::static_layout>(xb.derived_cast());
        XTENSOR_ASSERT(ax_a.size() == ax_b.size());
        XTENSOR_ASSERT(ax_a.size() < a.dimension());
        XTENSOR_ASSERT(ax_b.size() < b.dimension());
        std::size_t n_ax = ax_a.size();
        for (std::size_t i = 0; i < n_ax; ++i)
        {
            XTENSOR_ASSERT(ax_a[i] < a.dimension());
            XTENSOR_ASSERT(ax_b[i] < b.dimension());
        }

        // Move the axes to sum over to the end of a
        xt::dynamic_shape<std::size_t> newaxes_a;
        xt::dynamic_shape<std::size_t> result_shape;
        for (std::size_t i = 0; i < a.dimension(); ++i)
        {
            auto a_ax_it = std::find(ax_a.begin(), ax_a.end(), i);
            // first pass if i is not in ax_a, add to newaxes_a
            if (a_ax_it == ax_a.end())
            {
                newaxes_a.push_back(i);
            }
        }
        for (auto& a_ax_it : ax_a)
        {
            newaxes_a.push_back(a_ax_it);
        }

        // Move the axes to sum over to the start of b
        xt::dynamic_shape<std::size_t> newaxes_b;
        for(auto& b_ax_it : ax_b)
        {
            newaxes_b.push_back(b_ax_it);
        }
        for (std::size_t i = 0; i < b.dimension(); ++i)
        {
            auto b_ax_it = std::find(ax_b.begin(), ax_b.end(), i);
            // second pass if i is not in ax_b add to newaxes_b
            if (b_ax_it == ax_b.end())
            {
                newaxes_b.push_back(i);
            }
        }
        auto a_t = xt::transpose(a, newaxes_a);
        auto b_t = xt::transpose(b, newaxes_b);

        // the integer arg form of tensordot will handle the reshape of output for us
        return tensordot(a_t, b_t, n_ax);
    }
}
}
#endif
